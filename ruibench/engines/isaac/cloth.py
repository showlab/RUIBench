"""Cloth on Isaac: a PhysX particle cloth, taken hold of by attachment.

What a cloth is, the rule a hand takes it by and its place in the state are
the engines' common ground (``engines/cloth.py``).  Here is what Isaac does
with them (worked out with ``research/cloth_probe``, 10/5):

* **the solver.**  Particle cloth exists only on the graphics card, so a
  scene with a cloth in it is simulated there whole (``DEVICE``); every other
  scene stays on the CPU, where one robot is several times quicker.  Hence
  ``engines.cloth.scene_problems``: on the card a body moved by writing its
  pose is not moved for the solver.
* **the cloth.**  A triangle mesh of the cloth's points, each a particle
  ``ClothSpec.radius`` thick, springs along the triangles' edges; it collides
  with the table, the hands and itself.  Positions are read back through the
  tensor view (the USD mesh's own points never change).
* **the grasp.**  Friction between two fingers does not hold it.  When the
  rule says a hand has taken a patch (``engines.cloth.pinched``), those
  particles are attached to the hand until it opens -- not to the hand's own
  link, which pins them to the world, but to a small body without shape or
  weight that a joint fixes at the hand's tool centre point (a *hold*).

Everything Isaac is imported inside functions (``app.py`` says why).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from ...core.errors import ConfigError
from ...core.geometry import quat_conj, rotate
from ...core.types import ObjectState, Pose
from ..base import GRIP_CLOSING, BodySpec, SceneSpec
from ..cloth import cloth_of, cloth_state, cloths, flat_points, pinched, scene_problems

ENV = "/World/envs/env_0"
#: where a scene with a cloth is simulated
DEVICE = "cuda:0"
#: the particle solver's passes a step, and how far beyond a particle's own
#: radius it starts to answer a contact
SOLVER_ITERATIONS = 16
CONTACT_MARGIN = 1.5
#: a hold: its mass and inertia (it has no weight -- gravity is off for it --
#: and no shape; it has to be a body for the cloth to be attached to)
HOLD_MASS, HOLD_INERTIA = 0.2, 1.0e-4
#: the fingers have closed: this long since the command, and their opening
#: changed by less than this over this long (as ``IsaacEngine.gripper_settled``)
CLOSE_MIN_SECONDS, STALL_WINDOW, STALL_TOLERANCE = 0.3, 0.1, 0.0005
#: only what is this near the tool centre point is looked at for the patch
REACH = 0.06
#: the weave's repeats a metre of cloth (the linen photograph is 12.5 cm wide)
WEAVE = 8.0


def cloth_path(name: str) -> str:
    return f"{ENV}/cloth_{name}"


def hold_path(key: str) -> str:
    return f"{ENV}/{key}_hold"


def check(scene: SceneSpec) -> None:
    problems = scene_problems(scene)
    if problems:
        raise ConfigError(f"{scene.name}: " + "; ".join(problems))


# -- the stage ----------------------------------------------------------------------
def spawn(scene: SceneSpec) -> None:
    """Every cloth of the scene and a hold on every commanded hand, on the
    stage -- before the simulation starts."""
    import isaacsim.core.utils.stage as stage_utils
    from pxr import PhysxSchema

    stage = stage_utils.get_current_stage()
    PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/physicsScene")) \
        .CreateEnableExternalForcesEveryIterationAttr().Set(True)
    for body in cloths(scene):
        _spawn_cloth(stage, body)
    _spawn_holds(stage, scene)


def _spawn_cloth(stage, body: BodySpec) -> None:
    from omni.physx.scripts import particleUtils, physicsUtils
    from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade, Vt
    try:
        import Semantics
    except ImportError:                       # Isaac Sim < 4.2
        from pxr import Semantics

    spec = cloth_of(body)
    radius = spec.radius
    path = cloth_path(body.name)
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr().Set(Vt.Vec3fArray([Gf.Vec3f(float(x), float(y), 0.0) for x, y in spec.nodes]))
    mesh.CreateFaceVertexCountsAttr().Set([3] * len(spec.triangles))
    mesh.CreateFaceVertexIndicesAttr().Set([int(i) for t in spec.triangles for i in t])
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    mesh.CreateDoubleSidedAttr().Set(True)
    # what it is printed or woven with lies on the cloth, and goes where the
    # cloth goes: texture coordinates 0 to 1 across the box round it (the
    # picture's x the cloth's own, its top the cloth's left edge)
    x0, x1, y0, y1 = spec.bounds()
    UsdGeom.PrimvarsAPI(mesh).CreatePrimvar(
        "st", Sdf.ValueTypeNames.TexCoord2fArray, UsdGeom.Tokens.vertex).Set(
        Vt.Vec2fArray([Gf.Vec2f(float((x - x0) / (x1 - x0)), float((y - y0) / (y1 - y0)))
                       for x, y in spec.nodes]))
    physicsUtils.setup_transform_as_scale_orient_translate(mesh)
    physicsUtils.set_or_add_translate_op(mesh, Gf.Vec3f(*[float(v) for v in body.pose.pos]))
    physicsUtils.set_or_add_orient_op(mesh, Gf.Quatf(float(body.pose.quat[0]),
                                                     Gf.Vec3f(*[float(v) for v in body.pose.quat[1:]])))
    system = path + "_particles"
    particleUtils.add_physx_particle_system(
        stage=stage, particle_system_path=Sdf.Path(system), simulation_owner=Sdf.Path("/physicsScene"),
        contact_offset=CONTACT_MARGIN * radius, rest_offset=radius,
        particle_contact_offset=CONTACT_MARGIN * radius, solid_rest_offset=radius,
        fluid_rest_offset=0.0, solver_position_iterations=SOLVER_ITERATIONS)
    material = path + "_cloth"
    particleUtils.add_pbd_particle_material(stage, Sdf.Path(material), friction=spec.friction)
    physicsUtils.add_physics_material_to_prim(stage, stage.GetPrimAtPath(system), Sdf.Path(material))
    particleUtils.add_physx_particle_cloth(
        stage=stage, path=Sdf.Path(path), dynamic_mesh_path=None, particle_system_path=Sdf.Path(system),
        spring_stretch_stiffness=spec.stretch_stiffness, spring_bend_stiffness=spec.bend_stiffness,
        spring_shear_stiffness=spec.shear_stiffness, spring_damping=spec.spring_damping,
        self_collision=True, self_collision_filter=True)
    UsdPhysics.MassAPI.Apply(mesh.GetPrim()).CreateMassAttr().Set(float(body.mass))
    # as every body is labelled: by its name
    sem = Semantics.SemanticsAPI.Apply(mesh.GetPrim(), "class_body")
    sem.CreateSemanticTypeAttr().Set("class")
    sem.CreateSemanticDataAttr().Set(body.name)
    look = _look(stage, path + "_look", body.color, (x1 - x0, y1 - y0))
    UsdShade.MaterialBindingAPI.Apply(mesh.GetPrim()).Bind(UsdShade.Material(stage.GetPrimAtPath(look)))


def _look(stage, path: str, color, extent) -> str:
    """The cloth's own material: the finish it names -- a dyed weave
    (``fabric:blue``) or a print that covers it (``paper:towel:blue``) --
    laid by the cloth's own coordinates and not, as on a rigid body,
    projected from the world: the weave would stand still while the cloth
    moved under it.  ``extent``: how long and wide the cloth is (m)."""
    from pxr import Gf, Sdf, UsdShade

    from .finishes import material

    made = material(color or "fabric:white", path)
    shader = UsdShade.Shader(stage.GetPrimAtPath(f"{made}/Shader"))
    if shader.GetInput("project_uvw"):
        # a weave, repeated: as many times as the cloth is long and wide
        shader.GetInput("project_uvw").Set(False)
        shader.CreateInput("texture_scale", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(float(WEAVE * extent[0]), float(WEAVE * extent[1])))
    return made


def _hands(scene: SceneSpec) -> List[Tuple[str, str, str, object]]:
    """(arm id, its articulation's key, the link the tool is on, the robot's
    model) for every arm a task commands."""
    from ..robots import model_of
    from .robots import arm_key, isaac_robot

    out = []
    for robot in scene.robots:
        model = model_of(robot)
        for arm in robot.arms:
            out.append((f"{robot.name}/{arm}", arm_key(robot, arm),
                        isaac_robot(model.name).grasp_body, model))
    return out


def _spawn_holds(stage, scene: SceneSpec) -> None:
    from pxr import Gf, PhysxSchema, Sdf, UsdGeom, UsdPhysics

    from ..robots import arm_start_pose

    for arm_id, key, link, model in _hands(scene):
        robot = next(r for r in scene.robots if r.name == arm_id.split("/")[0])
        start = arm_start_pose(scene, robot, arm_id.split("/")[1])
        path = hold_path(key)
        cube = UsdGeom.Cube.Define(stage, path)
        cube.CreateSizeAttr(0.01)
        xf = UsdGeom.Xformable(cube)
        xf.AddTranslateOp().Set(Gf.Vec3d(*[float(v) for v in start.pos]))
        xf.AddOrientOp().Set(Gf.Quatf(float(start.quat[0]), Gf.Vec3f(*[float(v) for v in start.quat[1:]])))
        UsdGeom.Imageable(cube).MakeInvisible()
        UsdPhysics.RigidBodyAPI.Apply(cube.GetPrim())
        PhysxSchema.PhysxRigidBodyAPI.Apply(cube.GetPrim()).CreateDisableGravityAttr(True)
        mass = UsdPhysics.MassAPI.Apply(cube.GetPrim())
        mass.CreateMassAttr(HOLD_MASS)
        mass.CreateDiagonalInertiaAttr(Gf.Vec3f(HOLD_INERTIA, HOLD_INERTIA, HOLD_INERTIA))
        joint = UsdPhysics.FixedJoint.Define(stage, path + "_joint")
        joint.CreateBody0Rel().SetTargets([Sdf.Path(f"{ENV}/{key}/{link}")])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(path)])
        tcp = model.tcp_from_body
        joint.CreateLocalPos0Attr(Gf.Vec3f(*[float(v) for v in tcp.pos]))
        joint.CreateLocalRot0Attr(Gf.Quatf(float(tcp.quat[0]), Gf.Vec3f(*[float(v) for v in tcp.quat[1:]])))
        joint.CreateLocalPos1Attr(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr(Gf.Quatf(1.0, Gf.Vec3f(0.0, 0.0, 0.0)))
        # a joint of its own, not a link in the arm's tree (``spawners``: included,
        # PhysX merges the two and neither is found by name)
        joint.CreateExcludeFromArticulationAttr(True)


# -- while it runs ------------------------------------------------------------------
class Cloths:
    """The cloths of a loaded scene: where they are, who holds them."""

    def __init__(self, engine, scene: SceneSpec) -> None:
        from isaacsim.core.simulation_manager import SimulationManager

        self.engine = engine
        self.bodies: Dict[str, BodySpec] = {b.name: b for b in cloths(scene)}
        sim_view = SimulationManager.get_physics_sim_view()
        self._views = {name: sim_view.create_particle_cloth_view(cloth_path(name))
                       for name in self.bodies}
        self._holds = {arm_id: sim_view.create_rigid_body_view(hold_path(key))
                       for arm_id, key, _, _ in _hands(scene)}
        self._hold_of = {arm_id: hold_path(key) for arm_id, key, _, _ in _hands(scene)}
        #: arm -> (the cloth it holds, the attachment's prim)
        self.held: Dict[str, Tuple[str, str]] = {}
        #: hands that closed on no cloth: they take none until they open again
        self._closed_empty: set = set()
        self._gaps: Dict[str, deque] = {}
        #: cloth -> its last two samples, (time, every particle): the newest last
        self._samples: Dict[str, deque] = {name: deque(maxlen=2) for name in self.bodies}

    # -- reading ---------------------------------------------------------------------
    def _read(self, name: str):
        """Every particle of a cloth now, as an (n, 3) array on the host."""
        import numpy as np

        raw = self._views[name].get_positions()
        n = len(cloth_of(self.bodies[name]).nodes)
        return raw.detach().cpu().numpy().reshape(-1, 3)[:n].astype(np.float64)

    def sample(self) -> None:
        """Note where every cloth is now (once a control step: two samples
        apart give its speed)."""
        now = round(self.engine.sim_time, 6)
        for name, samples in self._samples.items():
            if samples and samples[-1][0] == now:
                continue
            samples.append((now, self._read(name)))

    def particles(self, name: str):
        self.sample()
        return self._samples[name][-1][1]

    def state(self, name: str, visible_from=None) -> ObjectState:
        import numpy as np

        body = self.bodies[name]
        marks = list(cloth_of(body).marks)
        now = self.particles(name)[marks]
        velocity = (0.0, 0.0, 0.0)
        samples = self._samples[name]
        if len(samples) == 2 and samples[1][0] - samples[0][0] > 1e-6:
            moved = (now - samples[0][1][marks]) / (samples[1][0] - samples[0][0])
            velocity = tuple(float(v) for v in moved[int(np.argmax(np.linalg.norm(moved, axis=1)))])
        holder = next((arm for arm, (held, _) in self.held.items() if held == name), None)
        return cloth_state(body, [tuple(float(v) for v in p) for p in now], holder, velocity,
                           visible_from=visible_from)

    def holding(self, arm: str) -> Optional[str]:
        held = self.held.get(arm)
        return held[0] if held else None

    # -- a new episode ---------------------------------------------------------------
    def write_initial(self, scene: SceneSpec) -> None:
        """Each cloth flat at this episode's pose, at rest, in no hand."""
        import torch

        self.bodies = {b.name: b for b in cloths(scene)}
        self.let_go_all()
        device = self.engine.device
        ids = torch.tensor([0], device=device, dtype=torch.int32)
        for name, body in self.bodies.items():
            view = self._views[name]
            flat = torch.tensor([[v for p in flat_points(body) for v in p]], device=device,
                                dtype=torch.float32)
            width = int(view.get_positions().shape[-1])
            if flat.shape[-1] < width:                    # the view's buffer may be longer
                flat = torch.nn.functional.pad(flat, (0, width - flat.shape[-1]))
            view.set_positions(flat, ids)
            view.set_velocities(torch.zeros_like(flat), ids)
            self._samples[name].clear()
        self._closed_empty = set()
        self._gaps = {}
        # the arms are written back to where they start: so are their holds
        from ..robots import arm_start_pose

        robots = {r.name: r for r in scene.robots}
        self.place_holds({arm: arm_start_pose(scene, robots[arm.split("/")[0]], arm.split("/")[1])
                          for arm in self._holds})

    def place_holds(self, poses: Optional[Dict[str, Pose]] = None) -> None:
        """Each hold at its hand's tool centre point -- where that is now, or
        at ``poses`` --, at rest: an arm set back to its start would
        otherwise drag it there by its joint."""
        import torch

        device = self.engine.device
        ids = torch.tensor([0], device=device, dtype=torch.int32)
        for arm, view in self._holds.items():
            tcp = poses[arm] if poses is not None else self.engine.eef_pose(arm)
            w, x, y, z = tcp.quat
            view.set_transforms(torch.tensor([[*tcp.pos, x, y, z, w]], device=device,
                                             dtype=torch.float32), ids)
            view.set_velocities(torch.zeros((1, 6), device=device, dtype=torch.float32), ids)

    def let_go_all(self) -> None:
        for arm in list(self.held):
            self._let_go(arm)

    # -- the grasp, once a control step ----------------------------------------------
    def step(self) -> None:
        engine = self.engine
        for arm in self._holds:
            closing = engine._grippers.get(arm, 0.0) >= GRIP_CLOSING
            if not closing:
                if arm in self.held:
                    self._let_go(arm)
                self._closed_empty.discard(arm)
                self._gaps.pop(arm, None)
                continue
            if arm in self.held or arm in self._closed_empty or not self._stalled(arm):
                continue
            taken = None if engine._holding_rigid(arm) else self._take(arm)
            if taken is None:
                self._closed_empty.add(arm)
        self.sample()

    def _stalled(self, arm: str) -> bool:
        """Have this hand's fingers finished closing?"""
        engine = self.engine
        now = engine.sim_time
        gap = engine._finger_gap(arm)
        history = self._gaps.setdefault(arm, deque(maxlen=32))
        history.append((now, gap))
        if now - engine._gripper_changed_at.get(arm, -1e9) < CLOSE_MIN_SECONDS or gap is None:
            return False
        past = [g for (t, g) in history if now - t >= STALL_WINDOW]
        return bool(past) and abs(gap - past[-1]) < STALL_TOLERANCE

    def _take(self, arm: str) -> Optional[str]:
        """The rule: the patch under the fingertips, of the first cloth that
        has one there, attached to this hand's hold."""
        import numpy as np
        from pxr import PhysxSchema, Sdf, Vt

        import omni.usd

        tcp = self.engine.eef_pose(arm)
        for name, body in self.bodies.items():
            points = self._read(name)
            near = np.where(np.linalg.norm(points - np.asarray(tcp.pos), axis=1) <= REACH)[0]
            patch = [int(near[k]) for k in pinched([tuple(points[i]) for i in near], tcp)]
            if not patch:
                continue
            spec = cloth_of(body)
            back = quat_conj(tcp.quat)
            in_hold = [rotate(back, tuple(float(a - b) for a, b in zip(points[i], tcp.pos)))
                       for i in patch]
            path = f"{cloth_path(name)}/hold_{arm.replace('/', '_')}"
            stage = omni.usd.get_context().get_stage()
            att = PhysxSchema.PhysxPhysicsAttachment.Define(stage, path)
            att.GetActor0Rel().SetTargets([Sdf.Path(cloth_path(name))])
            att.GetActor1Rel().SetTargets([Sdf.Path(self._hold_of[arm])])
            # the cloth's points by where they lie in the mesh at rest; the
            # hold's by where they are now in its frame, the tool's
            att.CreatePoints0Attr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(
                [(spec.nodes[i][0], spec.nodes[i][1], 0.0) for i in patch], dtype=np.float32)))
            att.CreatePoints1Attr().Set(Vt.Vec3fArray.FromNumpy(np.asarray(in_hold, dtype=np.float32)))
            att.CreateCollisionFilterIndices0Attr().Set(patch)
            att.CreateFilterType0Attr().Set("Vertices")
            self.held[arm] = (name, path)
            return name
        return None

    def _let_go(self, arm: str) -> None:
        import omni.usd

        _, path = self.held.pop(arm)
        omni.usd.get_context().get_stage().RemovePrim(path)


__all__ = ["DEVICE", "Cloths", "check", "cloth_path", "hold_path", "spawn"]
