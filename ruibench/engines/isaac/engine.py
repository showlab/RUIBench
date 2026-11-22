"""Isaac Sim backend -- the engine RUIBench reports results on.

Why Isaac and not a fast rasteriser: the models under test are frontier APIs
that were never fine-tuned on our images.  Whatever we render *is* the
observation, so low-fidelity images would measure domain gap rather than
ability.  Photorealistic RTX rendering is therefore a requirement of the
experiment, not a nicety -- and it is the platform RoboLab and RoboDojo are
built on, which keeps those comparisons open.

One shared controller for all four interfaces: an absolute-pose differential IK
controller.  "Change the interface, not the controller" is the premise of the
whole design, so the controller configuration is built once, here.

Two Isaac facts that shape this file:

* ``SimulationApp`` must start before any ``omni``/``isaaclab``/``pxr`` import,
  so every such import is inside a function;
* the IK controller takes commands in the robot's **base frame**, while
  RUIBench speaks world poses everywhere else.  The conversion happens here and
  nowhere else.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ...core.errors import ConfigError, EngineNotAvailable
from ...core.types import (
    ArmState, CameraFrame, ObjectState, Pose, RobotState, WorldState,
)
from ..base import (ARRIVAL_POS_TOL, ARRIVAL_ROT_TOL, GRIP_CLOSING, HELD_MIN_GAP, Engine,
                    EngineCapabilities,
                    SceneSpec, collides, joint_program)
from ..mock import footprint_of
from . import app as isaac_app

#: The IK controller every interface shares.
IK_CONTROLLER = {"command_type": "pose", "use_relative_mode": False, "ik_method": "dls"}

DEFAULT_SIM_HZ = 120.0
DEFAULT_CONTROL_HZ = 20.0

#: RUIBench's "end-effector pose" is the tool centre point (each robot model's
#: ``tcp_from_body``, shared with every backend), because task code gives
#: object positions and a grasp happens where the fingertips meet.  Controlling
#: the hand frame instead put every target 10 cm off, and drove the fingertips
#: into the table.

#: A gripper command takes at least this long, however quickly the fingers
#: appear to stop -- they start from rest.
GRIPPER_MIN_SECONDS = 0.3
#: ...and it has finished once the opening stopped changing: less than this
#: much movement over this long.
GRIPPER_STALL_WINDOW = 0.1          # s
GRIPPER_STALL_TOLERANCE = 0.0005    # m, summed over both fingers

#: A body counts as seen once at least this many pixels of it are in the image.
#: A mark flush with a face still grazes a pixel or two along its edge from
#: almost anywhere; nobody can read a colour from that, and an evidence test
#: that fired on it would call hidden evidence visible.
MIN_VISIBLE_PIXELS = 16

#: RTX settings for every rendered frame (Isaac Lab ``RenderCfg`` fields).
#: FXAA, not the default DLSS: DLSS rebuilds each frame from earlier ones, and
#: after a block moved it still showed a translucent copy where the block had
#: been 52 frames later.  FXAA works on one frame; what is left after a jump is
#: a faint shadow that the passes below clear.
#: Ambient occlusion darkens where things meet (v2): without it nothing sits
#: on the table.  The raw RTX settings are ``environment.RTX_SETTINGS``.
RENDER_SETTINGS: Dict[str, object] = {"antialiasing_mode": "FXAA", "enable_ambient_occlusion": True}

#: Renders per observation.  RTX still carries lighting from frame to frame,
#: and this engine renders on demand, long after the previous frame.  Measured
#: on WK1 with FXAA (mean error, 0-255, where things moved, against the frame
#: 52 passes on): after the arm moved, 4.8 on the first pass, 2.5 on the
#: fourth, 1.5 on the eighth; after an episode reset put blocks elsewhere, 6.0,
#: 5.2 and 3.8 -- so the first frame of an episode gets many more.  About 40 ms
#: a pass.
RENDER_PASSES = 8
RESET_PASSES = 30
#: When frames come often (video is recorded every 0.1 s) the renderer's
#: history is recent: one pass per this much simulated time since the last.
PASS_EVERY = 0.05
#: What the cameras see (``visible_objects``, for the tasks that score seeing:
#: DS1-DS3) is worked out at most this often in simulated time; the pictures
#: the model is shown are rendered fresh every time.  Per control step (0.05
#: s), for two cameras, it was two thirds of a DS episode's wall time -- 695 s
#: for 39 s of DS3 on a shared GPU (v13)
VISIBILITY_EVERY = 0.2

#: Speed limits of the shared controller.  The commanded pose moves toward the
#: target no faster than this; the same numbers as the mock engine, so a task's
#: timing does not change character between backends.
MAX_LINEAR_SPEED = 0.25      # m/s
#: the reset's pre-roll to an ``ArmStart`` (``IsaacEngine._preroll``): the
#: hand comes down onto what it holds from this far back along the tool, is
#: "there" this close, closes for this long with the body pinned, and settles
#: for this long once it is let go
PREROLL_ABOVE = 0.06         # m
PREROLL_ARRIVED = 0.003      # m
PREROLL_MAX_SECONDS = 5.0
PREROLL_CLOSE_SECONDS = 0.6
PREROLL_SETTLE_SECONDS = 0.6
MAX_ANGULAR_SPEED = 1.5      # rad/s
#: An arm sent to its folded rest -- the station arm's all-zero pose, the
#: model's ``start_quat`` -- goes there in joint space, as a real arm's
#: "home" does: its joints turn together, the farthest at this speed.  In
#: Cartesian space the hand had to turn from pointing down to pointing
#: forward on the way, and the IK swung the arm out over the table before it
#: folded (WK4, the advisor twice: "folds forward, then comes back").
HOME_JOINT_SPEED = 1.2       # rad/s
#: v11 (9/24), the advisor, of WK4 once more: going home stalled and jerked.
#: It was three moves -- up, back over the table, home -- each stopping at its
#: end, and the first could not be reached (the hand 14 cm up at the far
#: reach, fingers down, is past the wrist's limit: it slid 3.5 cm and stood
#: 1.4 s).  Home is now one path in joint space, a cubic still at both ends:
#: from where the arm is to rest, and for an arm reaching further out than its
#: model's ``home_via`` posture, through that posture without stopping -- the
#: hand rises 4 cm on the way instead of 10 (``research/v11_x5_smooth_home.py``).
#: Its length sets its time: a cubic's fastest joint moves ``HOME_PEAK`` times
#: its mean speed.
HOME_VIA_MARGIN = 0.15       # rad further out than the posture, in a joint it sets
HOME_PEAK = 1.5
#: a target this close to the folded rest is the rest
HOME_POS_TOL, HOME_ROT_TOL = 0.01, 0.05


def _slerp_towards(q0, q1, max_angle: float):
    """Rotate unit quaternion q0 toward q1 by at most ``max_angle`` radians."""
    import math

    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:                                 # take the short way round
        q1, dot = tuple(-v for v in q1), -dot
    dot = min(1.0, dot)
    angle = 2.0 * math.acos(dot)
    if angle <= max_angle or angle < 1e-9:
        return tuple(q1)
    frac = max_angle / angle
    theta = math.acos(dot)
    s = math.sin(theta)
    w0 = math.sin((1.0 - frac) * theta) / s
    w1 = math.sin(frac * theta) / s
    q = [w0 * a + w1 * b for a, b in zip(q0, q1)]
    n = math.sqrt(sum(v * v for v in q))
    return tuple(v / n for v in q)


def hand_to_tcp(hand: Pose, model=None) -> Pose:
    """The tool centre point when the grasp body is at ``hand``."""
    from ..robots import FR3, tcp_from_body

    return tcp_from_body(hand, model or FR3)


def tcp_to_hand(tcp: Pose, model=None) -> Pose:
    """The grasp body's pose that puts the tool centre point at ``tcp``."""
    from ..robots import FR3, body_from_tcp

    return body_from_tcp(tcp, model or FR3)


def _step_towards(cur: Pose, goal: Pose, dt: float) -> Pose:
    """One control tick of bounded-speed motion from ``cur`` toward ``goal``."""
    import math

    d = [g - c for c, g in zip(cur.pos, goal.pos)]
    dist = math.sqrt(sum(v * v for v in d))
    step = MAX_LINEAR_SPEED * dt
    pos = goal.pos if dist <= step else tuple(c + v * step / dist for c, v in zip(cur.pos, d))
    quat = _slerp_towards(cur.quat, goal.quat, MAX_ANGULAR_SPEED * dt)
    return Pose(pos, quat)


class IsaacEngine(Engine):
    capabilities = EngineCapabilities(
        name="isaac", concurrent_arms=True, springs=True, actuated_joints=True,
        rendering=True, mesh_assets=True, max_camera_size=2048,
        photoreal=True, results_grade=True, dynamics=True, cloth=True,
    )

    def __init__(self, headless: bool = True, device: str = "cpu",
                 sim_hz: float = DEFAULT_SIM_HZ, control_hz: float = DEFAULT_CONTROL_HZ,
                 render_every_step: bool = False, render_passes: Optional[int] = None,
                 track_visibility: bool = False, gpu: Optional[int] = None,
                 render_settings: Optional[dict] = None) -> None:
        """``render_every_step`` renders once per *control* step (20 Hz), not per
        physics step -- used for throughput measurement and for recording video.

        ``device`` is the **physics** device, and it defaults to the CPU.  GPU
        PhysX is built for thousands of parallel environments; RUIBench runs one,
        with one or two robots, and on the GPU every step paid ~65 ms in kernel
        launches and host syncs.  Rendering stays on the GPU either way.  A
        scene with a cloth in it is the exception, whatever is asked for here:
        PhysX simulates cloth on the graphics card only (``isaac/cloth.py``).
        """
        self.headless = headless
        self.device = device
        self.sim_hz = sim_hz
        self.control_hz = control_hz
        self.render_passes = int(render_passes or RENDER_PASSES)
        #: Isaac Lab ``RenderCfg`` fields, over ``RENDER_SETTINGS``
        self.render_settings = {**RENDER_SETTINGS, **(render_settings or {})}
        self.gpu = gpu
        self.render_every_step = render_every_step
        #: Active Perception needs per-object visibility, which costs a render
        #: per query.  The other six suites do not, so it is off by default.
        self.track_visibility = track_visibility
        self._scene_spec: Optional[SceneSpec] = None
        self._sim = None
        self._scene = None
        self._ik: Dict[str, object] = {}
        self._entity: Dict[str, object] = {}     # arm -> resolved SceneEntityCfg
        self._jacobi_idx: Dict[str, int] = {}
        self._targets: Dict[str, Pose] = {}
        self._commanded: Dict[str, Pose] = {}     # where the arm is told to be *now*
        #: arms going home in joint space: arm -> the joint targets commanded now
        self._homing: Dict[str, object] = {}
        #: arms going home: arm -> the path (``_plan_home``)
        self._home_plan: Dict[str, dict] = {}
        #: arms going to joint targets the same way (``set_arm_joints``): arm -> the targets
        self._joint_goal: Dict[str, object] = {}
        self._grippers: Dict[str, float] = {}
        self._instruction, self._instruction_version = "", 0
        self._events: List[str] = []
        #: colours an episode has changed (a dipped pH strip): the state reports them
        self._colors: Dict[str, object] = {}
        self._visibility_cache: Dict[tuple, Tuple[str, ...]] = {}
        #: what segmentation calls each labelled thing (lower-cased) -> its name
        self._label_names: Dict[str, str] = {}
        self._rendered_at: Optional[float] = None
        #: when segmentation alone was last brought up to date (one pass)
        self._segmented_at: Optional[float] = None
        self._finger_ids: Dict[str, list] = {}
        self._gripper_changed_at: Dict[str, float] = {}
        self._finger_history: Dict[str, list] = {}
        self._attached: Dict[str, Tuple[str, Pose]] = {}   # child -> (parent, offset)
        self._specs: Dict[str, object] = {}                # body name -> BodySpec
        self._carried: Dict[str, object] = {}              # body name -> the joint carrying it
        self._links: Dict[str, int] = {}                   # joint -> its link's body index
        self._arm_key: Dict[str, str] = {}                 # arm id -> its articulation
        self._arm_model: Dict[str, object] = {}            # arm id -> engines.robots model
        self._arm_isaac: Dict[str, object] = {}            # arm id -> isaac.robots entry
        self._coupled: List[tuple] = []     # (scene key, isaac entry, arm id) of coupled grippers
        #: the scene's cloths, if it has any (``isaac.cloth.Cloths``)
        self._cloths = None
        self._t0 = 0.0

    # -- lifecycle ---------------------------------------------------------
    def load(self, scene: SceneSpec, seed: int) -> None:
        """Build ``scene`` -- or, if its structure matches the loaded one, reuse it.

        Isaac scenes are built once per process and reset in place: poses and
        colours are written, nothing is torn down.  Tearing a stage down with
        cameras attached hung for 15+ minutes, and building one costs seconds, so
        a structural change raises :class:`SceneRebuildRequired` and the isolated
        runner moves on to a fresh worker process.
        """
        from ...core.errors import SceneRebuildRequired

        from ..cloth import cloths

        has_cloth = bool(cloths(scene))
        if has_cloth:
            from . import cloth as isaac_cloth

            isaac_cloth.check(scene)
        self._label_names = self._segmentation_names(scene)
        if self._scene is not None:
            if scene.structure_key() != self._scene_spec.structure_key():
                raise SceneRebuildRequired(
                    f"{self._scene_spec.name} -> {scene.name}: scene structure differs")
            self._scene_spec = scene
            self._write_initial_state(scene)
            self.reset()
            return

        isaac_app.launch(headless=self.headless, gpu=self.gpu)
        import isaaclab.sim as sim_utils
        import isaacsim.core.utils.stage as stage_utils
        from isaaclab.scene import InteractiveScene

        from . import environment
        from .scene_builder import TABLE_PRIM, build_scene_cfg, spawn_markers

        stage_utils.create_new_stage()
        self._scene_spec = scene
        if has_cloth:
            self.device = isaac_cloth.DEVICE      # cloth is solved on the graphics card only
        self._sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(
            dt=1.0 / self.sim_hz, device=self.device,
            render=sim_utils.RenderCfg(**self.render_settings)))
        self._apply_carb_settings()
        self._scene = InteractiveScene(build_scene_cfg(scene, self))
        spawn_markers(scene)
        environment.finish(stage_utils.get_current_stage(), TABLE_PRIM)
        self._label_parts(scene)
        self._add_camera_housings(scene)
        self._add_finger_pads(scene)
        self._paint_links(scene)
        self._filter_pairs(scene)
        if has_cloth:
            isaac_cloth.spawn(scene)
        self._sim.reset()
        self._apply_carb_settings()          # the preset applied at reset rewrites some
        environment.wait_for_assets(self._sim)
        self._build_controllers(scene)
        self._cloths = isaac_cloth.Cloths(self, scene) if has_cloth else None
        self._write_initial_state(scene)
        self.reset()

    def _apply_carb_settings(self) -> None:
        """Write the raw RTX settings again, now that the simulation context
        exists: its anti-aliasing preset rewrites some of them after reading
        ``RenderCfg.carb_settings``."""
        import carb

        from .environment import apply_rtx_settings

        apply_rtx_settings()
        settings = carb.settings.get_settings()
        for key, value in (self.render_settings.get("carb_settings") or {}).items():
            settings.set("/" + key.strip("/").replace(".", "/"), value)

    @staticmethod
    def _segmentation_names(scene: SceneSpec) -> Dict[str, str]:
        """Every labelled thing -- body, hand, mark -- by the name segmentation
        gives it back: lower-cased.  WK5's pans (``pan_L1``) and DS5's beakers
        (``beaker_A``) came back as ``pan_l1`` and ``beaker_a``, matched no body,
        and were never seen on Isaac (the RUIKit collaborator, bench-issue
        #19).  Two names that differ only in case cannot be told apart there."""
        from .robots import arm_key, station_arms
        from .scene_builder import gripper_label, marker_label

        names = [b.name for b in scene.bodies]
        names += [gripper_label(arm_key(r, a)) for r in scene.robots for a in station_arms(r)]
        names += [marker_label(m.name) for m in scene.markers]
        out: Dict[str, str] = {}
        for name in names:
            other = out.setdefault(name.lower(), name)
            if other != name:
                raise ConfigError(f"{scene.name}: {other!r} and {name!r} differ only in case, "
                                  "and segmentation cannot tell them apart")
        return out

    def _label_parts(self, scene: SceneSpec) -> None:
        """Label each robot's hand and fingers, and every mark, for
        segmentation, so how much of them a camera sees can be counted like any
        body: "the gripper is in view from the first frame" and "the mark the
        instruction names is in the picture" are checked, not assumed."""
        import omni.usd
        try:
            import Semantics
        except ImportError:                       # Isaac Sim < 4.2
            from pxr import Semantics
        from ..robots import model_of
        from .robots import arm_key, isaac_robot, station_arms
        from .scene_builder import gripper_label, marker_label, marker_path

        stage = omni.usd.get_context().get_stage()
        parts = [(f"/World/envs/env_0/{arm_key(r, a)}/{link}", gripper_label(arm_key(r, a)))
                 for r in scene.robots for a in station_arms(r)
                 for link in isaac_robot(model_of(r).name).gripper_links]
        parts += [(marker_path(m.name), marker_label(m.name)) for m in scene.markers]
        for path, label in parts:
            prim = stage.GetPrimAtPath(path)
            if not prim.IsValid():
                continue
            sem = Semantics.SemanticsAPI.Apply(prim, "class_part")
            sem.CreateSemanticTypeAttr().Set("class")
            sem.CreateSemanticDataAttr().Set(label)

    def _filter_pairs(self, scene: SceneSpec) -> None:
        """Bodies that pass through each other (``base.passes``): a PhysX
        filtered pair on each, before the simulation starts."""
        import omni.usd
        from pxr import UsdPhysics

        from ..base import passes
        from .scene_builder import art_key, body_key, carried_bodies

        stage = omni.usd.get_context().get_stage()
        carried = carried_bodies(scene)

        def path(body: str) -> str:
            if body in carried:
                return f"/World/envs/env_0/{art_key(carried[body].name)}/link"
            return f"/World/envs/env_0/{body_key(body)}"

        bodies = list(scene.bodies)
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                if not passes(a, b):
                    continue
                prim = stage.GetPrimAtPath(path(a.name))
                other = stage.GetPrimAtPath(path(b.name))
                if not (prim.IsValid() and other.IsValid()):
                    raise ConfigError(f"cannot let {a.name} and {b.name} pass: no prim")
                UsdPhysics.FilteredPairsAPI.Apply(prim).CreateFilteredPairsRel().AddTarget(other.GetPath())

    def _add_finger_pads(self, scene: SceneSpec) -> None:
        """Rubber on the fingers (``IsaacRobot.finger_pads``)."""
        import isaaclab.sim as sim_utils

        from ..robots import model_of
        from .robots import arm_key, isaac_robot, station_arms

        for robot in scene.robots:
            isaac = isaac_robot(model_of(robot).name)
            if not isaac.finger_pads:
                continue
            links, static, dynamic = isaac.finger_pads
            material = f"/World/Materials/{isaac.model}_finger_pad"
            sim_utils.spawn_rigid_body_material(material, sim_utils.RigidBodyMaterialCfg(
                static_friction=static, dynamic_friction=dynamic, restitution=0.0,
                friction_combine_mode="max"))
            for arm in station_arms(robot):
                for link in links:
                    sim_utils.bind_physics_material(
                        f"/World/envs/env_0/{arm_key(robot, arm)}/{link}", material)

    def _paint_links(self, scene: SceneSpec) -> None:
        """The real robot's finish on each link (``IsaacRobot.link_materials``),
        bound over what its converted description carries."""
        import isaaclab.sim as sim_utils
        import omni.usd
        from pxr import UsdShade

        from ..robots import model_of
        from .environment import nvidia_dir
        from .robots import arm_key, isaac_robot, station_arms

        stage = omni.usd.get_context().get_stage()
        made = set()
        for robot in scene.robots:
            isaac = isaac_robot(model_of(robot).name)
            for link, finish in isaac.link_materials.items():
                if ":" in finish:                 # one of ours: "plastic:dark"
                    from .finishes import material as finish_material

                    material = finish_material(finish)
                else:                             # an NVIDIA base material
                    material = "/World/Materials/" + finish.replace("/", "_")
                    if material not in made:
                        sim_utils.spawn_from_mdl_file(material, sim_utils.MdlFileCfg(
                            mdl_path=f"{nvidia_dir()}/Materials/Base/{finish}.mdl"))
                        made.add(material)
                bound = UsdShade.Material(stage.GetPrimAtPath(material))
                for arm in station_arms(robot):
                    # on the link itself, stronger than descendants: the URDF
                    # importer binds its colour that way one level above the
                    # mesh, and the binding nearest the root wins
                    root = stage.GetPrimAtPath(f"/World/envs/env_0/{arm_key(robot, arm)}/{link}")
                    if root.IsValid():
                        UsdShade.MaterialBindingAPI.Apply(root).Bind(
                            bound, UsdShade.Tokens.strongerThanDescendants)

    def _add_camera_housings(self, scene: SceneSpec) -> None:
        """A dark box where each wrist camera is, part of the hand's rigid body:
        on the real robot the camera is a solid thing, and without one the other
        arm passed straight through it (BC2, v0)."""
        import omni.usd
        from pxr import Gf, Sdf, UsdGeom, UsdPhysics, UsdShade

        from ...core.geometry import compose
        from ..robots import model_of
        from .robots import arm_key, isaac_robot, station_arms

        stage = omni.usd.get_context().get_stage()
        for robot in scene.robots:
            model = model_of(robot)
            if not model.camera_housing:
                continue
            boxes = [("camera_housing", model.camera_housing)]
            if model.camera_bracket:
                boxes.append(("camera_bracket", model.camera_bracket))
            body = isaac_robot(model.name).grasp_body
            for (name, (centre, half)), arm in ((b, a) for b in boxes for a in station_arms(robot)):
                where = compose(model.tcp_from_body, Pose(tuple(centre)))
                path = f"/World/envs/env_0/{arm_key(robot, arm)}/{body}/{name}"
                cube = UsdGeom.Cube.Define(stage, path)
                cube.CreateSizeAttr(2.0)
                xform = UsdGeom.Xformable(cube)
                xform.AddTranslateOp().Set(Gf.Vec3d(*where.pos))
                xform.AddScaleOp().Set(Gf.Vec3f(*half))
                cube.CreateDisplayColorAttr([Gf.Vec3f(0.08, 0.08, 0.09)])
                UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
                material = UsdShade.Material.Define(stage, path + "_material")
                shader = UsdShade.Shader.Define(stage, path + "_material/Shader")
                shader.CreateIdAttr("UsdPreviewSurface")
                shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.08, 0.08, 0.09))
                material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
                UsdShade.MaterialBindingAPI.Apply(cube.GetPrim()).Bind(material)

    def _place_markers(self, scene: SceneSpec) -> None:
        """Move each mark's prim to this episode's pose."""
        import omni.usd
        from pxr import Gf

        from .scene_builder import marker_path

        stage = omni.usd.get_context().get_stage()
        for m in scene.markers:
            prim = stage.GetPrimAtPath(marker_path(m.name))
            prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(*m.pose.pos))
            orient = prim.GetAttribute("xformOp:orient")
            quat = Gf.Quatd if orient.GetTypeName() == "quatd" else Gf.Quatf
            orient.Set(quat(float(m.pose.quat[0]), Gf.Vec3d(*m.pose.quat[1:])
                            if quat is Gf.Quatd else Gf.Vec3f(*m.pose.quat[1:])))

    def _body(self, name: str):
        """(position, task orientation, linear velocity) of a body, whether it
        is a rigid object or the link a joint carries."""
        from .scene_builder import art_key, body_key, task_quat

        spec = self._specs[name]
        if self._cloths is not None and name in self._cloths.bodies:
            cloth = self._cloths.state(name)          # the box round it
            return tuple(cloth.pose.pos), tuple(cloth.pose.quat), tuple(cloth.lin_vel)
        joint = self._carried.get(name)
        if joint is None:
            data = self._scene[body_key(name)].data
            pos, quat = data.root_pos_w[0].tolist(), data.root_quat_w[0].tolist()
            vel = data.root_lin_vel_w[0].tolist()
        else:
            art = self._scene[art_key(joint.name)]
            state = art.data.body_state_w[0, self._link_index(joint.name)].tolist()
            pos, quat, vel = state[0:3], state[3:7], state[7:10]
        return (tuple(float(v) for v in pos), task_quat(spec, quat),
                tuple(float(v) for v in vel))

    def _link_index(self, joint: str) -> int:
        from .scene_builder import art_key

        if joint not in self._links:
            self._links[joint] = self._scene[art_key(joint)].find_bodies("link")[0][0]
        return self._links[joint]

    def _write_initial_state(self, scene: SceneSpec) -> None:
        """Write the episode's initial poses and colours into the live scene."""
        import torch
        from ...core.geometry import relative_pose
        from ..base import attachments, joint_start, rides
        from .scene_builder import art_key, body_key, carried_bodies, joint_gains, world_quat

        spec = {b.name: b for b in scene.bodies}
        self._specs = spec
        self._carried = carried_bodies(scene)
        self._attached = {
            child: (parent, relative_pose(Pose(spec[parent].pose.pos, world_quat(spec[parent])),
                                          Pose(spec[child].pose.pos, world_quat(spec[child]))))
            for child, parent in attachments(scene.bodies).items()}
        origin = self._scene.env_origins[0]
        for body in scene.bodies:
            if body.shape == "cloth":
                continue                  # its particles are written below
            pose = torch.tensor([list(body.pose.pos) + list(world_quat(body))],
                                device=self.device, dtype=torch.float32)
            pose[:, :3] += origin
            joint = self._carried.get(body.name)
            if joint is not None:
                # the joint's zero is where the body's spec puts it
                art = self._scene[art_key(joint.name)]
                art.write_root_pose_to_sim(pose)
                value = torch.tensor([[joint_start(joint)]], device=self.device)
                art.write_joint_state_to_sim(value, torch.zeros_like(value))
                art.set_joint_position_target(value)
                # an episode may have changed the drive (a pendulum let go)
                self.set_joint_drive(joint.name, *joint_gains(joint))
            else:
                obj = self._scene[body_key(body.name)]
                obj.write_root_pose_to_sim(pose)
                if not (body.fixed or rides(body)):   # PhysX rejects velocities on kinematic bodies
                    obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))
            if body.color and not body.asset and (body.shape != "compound"
                                                  or body.attrs.get("recolorable")):
                self.set_color(body.name, body.color)
        from .robots import arm_key, station_arms

        for rspec in scene.robots:
            for arm in station_arms(rspec):
                robot = self._scene[arm_key(rspec, arm)]
                robot.write_joint_state_to_sim(robot.data.default_joint_pos.clone(),
                                               robot.data.default_joint_vel.clone())
                # an arm no task commands holds its ready pose
                robot.set_joint_position_target(robot.data.default_joint_pos.clone())
        if self._cloths is not None:
            self._cloths.write_initial(scene)
        self._place_markers(scene)

    def _build_controllers(self, scene: SceneSpec) -> None:
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
        from isaaclab.managers import SceneEntityCfg

        from ..robots import model_of
        from .robots import arm_key, isaac_robot, station_arms

        self._coupled = []
        for rspec in scene.robots:
            model = model_of(rspec)
            isaac = isaac_robot(model.name)
            if isaac.finger_coupling is not None:
                # every arm of the station, commanded or idle, holds its fingers
                self._coupled += [(arm_key(rspec, arm), isaac, f"{rspec.name}/{arm}")
                                  for arm in station_arms(rspec)]
            for arm in rspec.arms:
                arm_id = f"{rspec.name}/{arm}"
                key = arm_key(rspec, arm)
                robot = self._scene[key]
                entity = SceneEntityCfg(key, joint_names=[isaac.arm_joints],
                                        body_names=[isaac.grasp_body])
                entity.resolve(self._scene)
                self._arm_key[arm_id] = key
                self._arm_model[arm_id] = model
                self._arm_isaac[arm_id] = isaac
                self._ik[arm_id] = DifferentialIKController(
                    DifferentialIKControllerCfg(**IK_CONTROLLER), num_envs=1,
                    device=self.device)
                self._entity[arm_id] = entity
                # for a fixed-base robot the jacobian index is one less than the
                # body index: the root body is not in the returned jacobians
                self._jacobi_idx[arm_id] = (entity.body_ids[0] - 1 if robot.is_fixed_base
                                            else entity.body_ids[0])

    def reset(self) -> WorldState:
        self._scene.reset()
        if self._scene_spec is not None:
            self._write_initial_state(self._scene_spec)
        self._scene.write_data_to_sim()
        self._sim.step(render=False)
        self._scene.update(1.0 / self.sim_hz)
        # marks painted on a body that starts off its joint's zero (DI2's
        # engine on its wagon) go with it before the first picture
        self._carry_attached()
        if self._cloths is not None:
            self._cloths.place_holds()
        self._events = []
        self._colors = {}
        self._visibility_cache = {}
        self._rendered_at = None
        self._segmented_at = None
        self._instruction, self._instruction_version = "", 0
        for ik in self._ik.values():
            ik.reset()
        st = self.state()
        self._targets = {a: st.arm(a).eef_pose for a in st.arm_ids}
        self._commanded = dict(self._targets)
        self._homing = {}
        self._home_plan = {}
        self._joint_goal = {}
        self._grippers = {a: 0.0 for a in st.arm_ids}
        if self._scene_spec is not None and self._scene_spec.arm_starts:
            self._preroll(self._scene_spec)
            st = self.state()
            self._targets = {a: st.arm(a).eef_pose for a in st.arm_ids}
            self._commanded = dict(self._targets)
        self._t0 = float(self._sim.current_time)
        st = self.state()
        self._gripper_changed_at = {}
        self._finger_history = {}
        return st

    def _preroll(self, scene: SceneSpec) -> None:
        """Take each arm with an ``ArmStart`` there before the clock starts:
        above its start pose with the hand open, down onto it, the fingers
        closed on what it holds -- which stays pinned where the scene puts it
        until they have -- and then let go into the fingers.  Everything
        before the episode's first frame, so the first frame shows the hands
        already holding (DI5, CL1)."""
        import torch

        from ...core.geometry import rotate
        from .scene_builder import body_key, world_quat

        starts = [s for s in scene.arm_starts if s.arm in self._targets]
        pinned = [self._specs[s.holding] for s in starts if s.holding]
        origin = self._scene.env_origins[0]
        dt = 1.0 / self.sim_hz
        per_control = max(1, int(round(self.sim_hz / self.control_hz)))

        def pin() -> None:
            for body in pinned:
                obj = self._scene[body_key(body.name)]
                pose = torch.tensor([list(body.pose.pos) + list(world_quat(body))],
                                    device=self.device, dtype=torch.float32)
                pose[:, :3] += origin
                obj.write_root_pose_to_sim(pose)
                obj.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.device))

        def run(seconds: float, pinning: bool, until_there: bool = False) -> None:
            for i in range(max(1, int(round(seconds * self.sim_hz)))):
                for arm, goal in self._targets.items():
                    self._commanded[arm] = _step_towards(self._commanded[arm], goal, dt)
                self._apply_targets()
                self._drive_coupled_fingers()
                if pinning:
                    pin()
                self._scene.write_data_to_sim()
                self._sim.step(render=False)
                self._scene.update(dt)
                self._carry_attached()
                if until_there and i % per_control == 0 and all(
                        math.dist(self.eef_pose(s.arm).pos, self._targets[s.arm].pos)
                        < PREROLL_ARRIVED for s in starts):
                    return

        for s in starts:
            # back along the tool's own axis: the hand comes onto the handle
            # the way it would to take it up
            back = rotate(s.pose.quat, (0.0, 0.0, -PREROLL_ABOVE))
            self._targets[s.arm] = Pose(tuple(p + b for p, b in zip(s.pose.pos, back)), s.pose.quat)
        run(PREROLL_MAX_SECONDS, pinning=True, until_there=True)
        for s in starts:
            self._targets[s.arm] = s.pose
        run(PREROLL_MAX_SECONDS, pinning=True, until_there=True)
        run(0.3, pinning=True)
        for s in starts:
            self._grippers[s.arm] = 1.0
        run(PREROLL_CLOSE_SECONDS, pinning=True)
        run(PREROLL_SETTLE_SECONDS, pinning=False)

    def close(self) -> None:
        """Drop the scene, keep the app.

        ``SimulationApp`` can be created **once per process**, and starting it is
        the expensive part (tens of minutes cold, tens of seconds warm).  A run
        is thousands of episodes, so the app outlives every one of them and only
        the stage is torn down here.
        """
        try:
            if self._sim is not None:
                self._sim.stop()
                self._sim.clear_all_callbacks()
                self._sim.clear_instance()
        except Exception:
            pass
        self._scene = None
        self._sim = None
        self._cloths = None
        self._ik.clear()
        self._entity.clear()
        self._finger_ids.clear()
        self._visibility_cache = {}
        self._rendered_at = None
        self._segmented_at = None

    # -- time --------------------------------------------------------------
    @property
    def sim_time(self) -> float:
        """Seconds since this episode's reset.

        The stage outlives episodes, so the simulator's own clock keeps running;
        everything downstream -- budgets, event times, the log -- expects each
        episode to start at zero, as it does on every other backend.
        """
        if self._sim is None:
            return 0.0
        return float(self._sim.current_time) - self._t0

    def advance(self, seconds: float) -> None:
        """Step physics, re-solving IK at the control rate.

        Physics runs at ``sim_hz`` and the controller at ``control_hz``; nothing
        here consults the wall clock, so an episode replays identically
        regardless of how fast the machine is.
        """
        if seconds <= 0 or self._sim is None:
            return
        per_control = max(1, int(round(self.sim_hz / self.control_hz)))
        steps = max(1, int(round(seconds * self.sim_hz)))
        dt = 1.0 / self.sim_hz
        for i in range(steps):
            control_step = (i % per_control == 0)
            # IK is solved on every physics step against a commanded pose that
            # moves toward the target at bounded speed.  Solving only at the
            # control rate, with the whole remaining error folded into one joint
            # update, made the arm overshoot and swing behind its own base.
            for arm, goal in self._targets.items():
                if arm not in self._homing:
                    self._commanded[arm] = _step_towards(self._commanded[arm], goal, dt)
            self._apply_targets()
            for joint in self._carried.values():
                if joint.oscillate:
                    self.set_articulation_target(joint.name, joint_program(joint, self.sim_time))
            self._drive_coupled_fingers()
            self._scene.write_data_to_sim()
            # Physics only.  Rendering happens on demand -- a frame is produced
            # when the harness asks for one, which is also what makes the
            # observation timestamp mean something.  `render_every_step` means
            # once per *control* step, not once per physics step: rendering six
            # times between two commands measures the renderer, not the task.
            self._sim.step(render=self.render_every_step and control_step)
            self._scene.update(dt)
            self._carry_attached()
            if control_step and self._cloths is not None:
                self._cloths.step()          # a hand that has closed on a cloth takes it
        # the frame is now stale -- _ensure_rendered compares times.  Not
        # cleared: a cleared stamp means "nothing rendered since the reset",
        # which asks for the long warm-up

    def _carry_attached(self) -> None:
        """Move visual-only marks with the bodies they are painted on."""
        if not self._attached:
            return
        import torch
        from ...core.geometry import compose, quat_mul
        from .scene_builder import body_key, upright_quat

        for child, (parent, offset) in self._attached.items():
            pos, quat, _ = self._body(parent)
            spec = self._specs[parent]
            pose = compose(Pose(pos, quat_mul(quat, upright_quat(spec))), offset)
            self._scene[body_key(child)].write_root_pose_to_sim(torch.tensor(
                [list(pose.pos) + list(pose.quat)], device=self.device, dtype=torch.float32))

    def _apply_targets(self) -> None:
        import torch
        from isaaclab.utils.math import subtract_frame_transforms

        for arm_id, tcp_target in self._commanded.items():
            if arm_id in self._homing:
                self._home_step(arm_id)
                continue
            # the IK tracks the grasp body
            target = tcp_to_hand(tcp_target, self._arm_model[arm_id])
            robot = self._scene[self._arm_key[arm_id]]
            entity = self._entity[arm_id]
            ik = self._ik[arm_id]

            # RUIBench speaks world poses; the IK controller wants the base frame
            root = robot.data.root_state_w[:, 0:7]
            goal = torch.tensor([list(target.pos) + list(target.quat)],
                                device=self.device, dtype=torch.float32)
            goal_pos_b, goal_quat_b = subtract_frame_transforms(
                root[:, 0:3], root[:, 3:7], goal[:, 0:3], goal[:, 3:7])
            ik.set_command(torch.cat([goal_pos_b, goal_quat_b], dim=-1))

            jacobian = robot.root_physx_view.get_jacobians()[
                :, self._jacobi_idx[arm_id], :, entity.joint_ids]
            ee_w = robot.data.body_state_w[:, entity.body_ids[0], 0:7]
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root[:, 0:3], root[:, 3:7], ee_w[:, 0:3], ee_w[:, 3:7])
            joint_pos = robot.data.joint_pos[:, entity.joint_ids]
            desired = ik.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos)
            robot.set_joint_position_target(desired, joint_ids=entity.joint_ids)
            self._apply_gripper(robot, arm_id)

    def _rest_pose(self, arm: str) -> Optional[Pose]:
        """The folded rest of an arm whose model has one (``start_quat``: the
        station's ARX X5), whatever it held at the start."""
        from types import SimpleNamespace

        from ..robots import arm_start_pose

        if not getattr(self._arm_model.get(arm), "start_quat", None):
            return None
        robot = next(r for r in self._scene_spec.robots if r.name == arm.split("/")[0])
        return arm_start_pose(SimpleNamespace(table_height=self._scene_spec.table_height), robot,
                              arm.split("/")[1])

    def _plan_home(self, arm_id: str, q0) -> dict:
        """The way home from joints ``q0`` (``HOME_VIA_MARGIN``)."""
        robot = self._scene[self._arm_key[arm_id]]
        rest = robot.data.default_joint_pos[:, self._entity[arm_id].joint_ids].clone()
        return self._plan_joints(arm_id, q0, rest, dict(self._arm_model[arm_id].home_via))

    def _plan_joints(self, arm_id: str, q0, goal, posture: dict) -> dict:
        """A way in joint space from ``q0`` to ``goal``, through ``posture``
        where the arm reaches further out than it: its knots -- where each is
        along the path, the joints there, their rate there (none at the ends;
        through the posture, the whole move's direction) -- and how many
        physics steps it takes."""
        import torch

        robot = self._scene[self._arm_key[arm_id]]
        ids = self._entity[arm_id].joint_ids
        points = [q0.clone()]
        if posture:
            names = [robot.joint_names[i] for i in ids]
            via = torch.tensor([[posture.get(n, 0.0) for n in names]], device=q0.device,
                               dtype=q0.dtype)
            sets = via.abs() > 1e-6
            if bool(((q0.abs() - via.abs() > HOME_VIA_MARGIN) & sets).any()):
                points.append(via)
        points.append(goal)
        lengths = [float((b - a).abs().max()) for a, b in zip(points[:-1], points[1:])]
        total = sum(lengths)
        chord = goal - q0
        knots, at = [], 0.0
        for k, q in enumerate(points):
            ends = k in (0, len(points) - 1)
            knots.append((at / total if total > 0 else 1.0, q, torch.zeros_like(q) if ends else chord))
            if k < len(lengths):
                at += lengths[k]
        steps = max(1, int(round(HOME_PEAK * total / HOME_JOINT_SPEED * self.sim_hz)))
        return {"knots": knots, "steps": steps, "step": 0}

    @staticmethod
    def _home_at(plan: dict, u: float):
        """The joints ``u`` of the way along a home path: the cubic between
        the two knots round it (a Hermite piece, the knots' rates matched)."""
        knots = plan["knots"]
        for k in range(len(knots) - 1):
            (u0, p0, v0), (u1, p1, v1) = knots[k], knots[k + 1]
            if u <= u1 or k == len(knots) - 2:
                h = max(u1 - u0, 1e-9)
                t = min(1.0, max(0.0, (u - u0) / h))
                t2, t3 = t * t, t * t * t
                return ((2 * t3 - 3 * t2 + 1) * p0 + (t3 - 2 * t2 + t) * h * v0
                        + (-2 * t3 + 3 * t2) * p1 + (t3 - t2) * h * v1)
        return knots[-1][1]

    def set_arm_joints(self, arm: str, joints) -> Pose:
        """Joint targets, inside the joints' limits, reached the way an arm goes
        home: a joint-space path at the shared joint speed (v1.17)."""
        import torch

        from ..kinematics import arm_frame, forward

        robot = self._scene[self._arm_key[arm]]
        ids = self._entity[arm].joint_ids
        q0 = robot.data.joint_pos[:, ids].clone()
        if len(joints) != q0.shape[1]:
            raise ValueError(f"{arm} has {q0.shape[1]} joints, not {len(joints)}")
        lim = robot.data.soft_joint_pos_limits[0, ids]
        goal = torch.tensor([[float(v) for v in joints]], device=q0.device, dtype=q0.dtype)
        goal = torch.max(torch.min(goal, lim[:, 1]), lim[:, 0])
        self._homing[arm] = q0
        self._joint_goal[arm] = goal
        self._home_plan[arm] = self._plan_joints(arm, q0, goal, {})
        robot_name, _, side = arm.partition("/")
        rspec = next(r for r in self._scene_spec.robots if r.name == robot_name)
        self._targets[arm] = forward(self._arm_model[arm],
                                     arm_frame(rspec, side, self._scene_spec.table_height),
                                     goal[0].tolist())
        return self._targets[arm]

    def arm_joints(self, arm: str):
        return self._joint_readings(arm)

    def _home_step(self, arm_id: str) -> None:
        """One physics step of an arm going home, along its path."""
        robot = self._scene[self._arm_key[arm_id]]
        ids = self._entity[arm_id].joint_ids
        plan = self._home_plan.get(arm_id)
        if plan is None:
            plan = self._home_plan[arm_id] = self._plan_home(arm_id, self._homing[arm_id])
        plan["step"] = min(plan["steps"], plan["step"] + 1)
        cmd = self._home_at(plan, plan["step"] / plan["steps"])
        self._homing[arm_id] = cmd
        robot.set_joint_position_target(cmd, joint_ids=ids)
        self._apply_gripper(robot, arm_id)

    def _home_reached(self, arm: str) -> bool:
        robot = self._scene[self._arm_key[arm]]
        ids = self._entity[arm].joint_ids
        rest = robot.data.default_joint_pos[:, ids]
        plan = self._home_plan.get(arm)
        return ((plan is None or plan["step"] >= plan["steps"])
                and float((self._homing[arm] - rest).abs().max()) < 1e-6
                and float((robot.data.joint_pos[:, ids] - rest).abs().max()) < 0.02)

    def _drive_coupled_fingers(self) -> None:
        """One physics step of every coupled gripper (``FingerCoupling``)."""
        import torch

        for key, isaac, arm_id in self._coupled:
            c = isaac.finger_coupling
            robot = self._scene[key]
            ids, opens = self._fingers(robot, arm_id, isaac)
            signs = [1.0 if v > 0 else -1.0 for v in opens]
            closing = self._grippers.get(arm_id, 0.0)
            if closing >= GRIP_CLOSING:
                pos = robot.data.joint_pos[0, ids].tolist()
                mean = sum(p * s for p, s in zip(pos, signs)) / len(pos)
                opening = max(0.0, mean - c.squeeze / c.stiffness)
            else:
                opening = abs(opens[0]) * (1.0 - closing)
            robot.set_joint_position_target(
                torch.tensor([[s * opening for s in signs]], device=self.device), joint_ids=ids)

    def _apply_gripper(self, robot, arm_id: str) -> None:
        import torch

        if self._arm_isaac[arm_id].finger_coupling is not None:
            return                          # driven every step by _drive_coupled_fingers
        ids, opens = self._fingers(robot, arm_id)
        if not ids:
            return
        frac = 1.0 - self._grippers.get(arm_id, 0.0)
        robot.set_joint_position_target(
            torch.tensor([[v * frac for v in opens]], device=self.device), joint_ids=ids)

    def _fingers(self, robot, arm_id: str, isaac=None):
        """(joint ids, each finger's open value) -- resolved once per arm."""
        hit = self._finger_ids.get(arm_id)
        if hit is None:
            isaac = isaac or self._arm_isaac[arm_id]
            ids, names = robot.find_joints(isaac.finger_joints)
            hit = (ids, [isaac.finger_open[n] for n in names])
            self._finger_ids[arm_id] = hit
        return hit

    # -- observation -------------------------------------------------------
    def state(self) -> WorldState:

        objects: Dict[str, ObjectState] = {}
        # Visibility costs a render, so only pay for it when a task actually
        # asks -- the Active Perception suite does, the other six do not.
        visible = ({cam.name: set(self.visible_objects(cam.name))
                    for cam in self._scene_spec.cameras}
                   if (self._scene_spec and self.track_visibility) else {})
        arm_ids = self._scene_spec.arm_ids if self._scene_spec else ()
        gaps = {a: self._finger_gap(a) for a in arm_ids}
        holders = {a: self._holding(a, gaps[a]) for a in arm_ids}
        for body in (self._scene_spec.bodies if self._scene_spec else ()):
            if self._cloths is not None and body.name in self._cloths.bodies:
                objects[body.name] = self._cloths.state(body.name, visible_from=(
                    tuple(c for c, names in visible.items() if body.name in names)
                    if self.track_visibility else None))
                continue
            # (w,x,y,z), with the asset's stand-up rotation turned back out
            pos, quat, vel = self._body(body.name)
            held_by = next((a for a, held in holders.items() if held == body.name), None)
            objects[body.name] = ObjectState(
                name=body.name, pose=Pose(pos, quat), size=body.size,
                category=body.category, color=self._colors.get(body.name, body.color),
                held_by=held_by, lin_vel=vel,
                visible_from=(tuple(c for c, names in visible.items() if body.name in names)
                              if self.track_visibility else None),
                attrs=dict(body.attrs),
                footprint=footprint_of(body),
            )

        robots: Dict[str, RobotState] = {}
        for rspec in (self._scene_spec.robots if self._scene_spec else ()):
            arms = {}
            for arm in rspec.arms:
                arm_id = f"{rspec.name}/{arm}"
                arms[arm] = ArmState(
                    name=arm_id, eef_pose=self.eef_pose(arm_id),
                    gripper=self._grippers.get(arm_id, 0.0), holding=holders.get(arm_id),
                    opening=None if gaps.get(arm_id) is None else round(gaps[arm_id], 4),
                    # moving toward its target; the gripper's own settling is
                    # stateful to query, so it is left to the executor
                    busy=arm_id in self._targets and not self.arm_at_target(arm_id),
                    joints=self._joint_readings(arm_id),
                )
            robots[rspec.name] = RobotState(rspec.name, arms)

        return WorldState(sim_time=self.sim_time, objects=objects, robots=robots,
                          articulations=self._joint_values(), instruction=self._instruction,
                          instruction_version=self._instruction_version,
                          events_fired=tuple(self._events))

    def _joint_values(self) -> Dict[str, float]:
        from .scene_builder import art_key

        return {j.name: float(self._scene[art_key(j.name)].data.joint_pos[0, 0])
                for j in self._carried.values()}

    def _ensure_rendered(self) -> None:
        """Produce a fresh frame if the last one is older than the current instant.

        Rendering is by far the most expensive thing this engine does, so it
        happens once per instant that anyone actually looks, rather than once
        per physics step.
        """
        now = round(self.sim_time, 6)
        if self._rendered_at == now:
            return
        if self._rendered_at is None:
            passes = max(self.render_passes, RESET_PASSES)
        else:
            passes = min(self.render_passes,
                         max(1, math.ceil((now - self._rendered_at) / PASS_EVERY - 1e-9)))
        for _ in range(passes):
            self._sim.render()
        self._scene.update(0.0)
        self._rendered_at = now
        self._visibility_cache = {}

    def _ensure_segmented(self) -> None:
        """Bring the cameras' segmentation up to the current instant.  One pass
        does it -- labels do not converge the way lighting does -- and it is
        not a picture anyone is shown, so the next picture still gets its own
        passes (``_ensure_rendered``)."""
        now = round(self.sim_time, 6)
        if now in (self._rendered_at, self._segmented_at):
            return
        if self._rendered_at is None:
            self._ensure_rendered()            # an episode's first frame: the long warm-up
            return
        self._sim.render()
        self._scene.update(0.0)
        self._segmented_at = now

    def render(self, camera: str) -> CameraFrame:
        from .scene_builder import camera_key

        self._ensure_rendered()
        cam = self._scene[camera_key(camera)]
        rgb = cam.data.output["rgb"][0]
        return CameraFrame(camera=camera, width=int(rgb.shape[1]), height=int(rgb.shape[0]),
                           captured_at=self.sim_time, image=rgb, depth=self._depth(cam))

    @staticmethod
    def _depth(cam):
        """The camera's depth image (metres along the optical axis, HxW), if it
        was asked for one.  Read from the camera's own annotator: Isaac Lab's
        tiled camera passes a CPU pipeline's buffers through uint8
        (``tiled_camera.py`` casts every numpy buffer), which leaves depth in
        whole metres.  One camera a render product here, so the tile is the
        image.  What the renderer hits nothing with is infinitely far."""
        if "depth" not in cam.cfg.data_types:
            return None
        import numpy as np

        raw = cam._annotators["depth"].get_data()
        if isinstance(raw, dict):
            raw = raw["data"]
        if hasattr(raw, "numpy"):
            raw = raw.numpy()
        depth = np.array(raw, dtype=np.float32).reshape(cam.cfg.height, cam.cfg.width)
        return depth

    def visible_objects(self, camera: str) -> Tuple[str, ...]:
        """Which bodies the renderer actually shows from this camera.

        Read off the semantic-segmentation channel, not guessed from geometry:
        the Active Perception suite rests entirely on the agent not being told
        about what it has not looked at, so this has to be the real answer.  A
        body needs ``MIN_VISIBLE_PIXELS`` of itself in view to count.

        Cached per simulated instant -- the answer cannot change without time
        passing, and reading a GPU tensor on every predicate evaluation would
        dominate the episode.
        """
        key = (camera, math.floor(round(self.sim_time, 6) / VISIBILITY_EVERY))
        hit = self._visibility_cache.get(key)
        if hit is not None:
            return hit
        from ..base import legibility_reach, legible_from

        eye = self.camera_position(camera)
        cam = next(c for c in self._scene_spec.cameras if c.name == camera)
        reach = legibility_reach(self._scene_spec, cam)
        specs = {b.name: b for b in self._scene_spec.bodies}
        out = []
        for name, n in sorted(self.pixel_counts(camera).items()):
            body = specs.get(name)
            if n < MIN_VISIBLE_PIXELS or body is None:
                continue
            pos, quat, _ = self._body(name)
            if legible_from(body, pos, quat, eye, reach):
                out.append(name)
        out = tuple(out)
        # only the current stretch of time is useful, for every camera
        self._visibility_cache = {k: v for k, v in self._visibility_cache.items() if k[1] == key[1]}
        self._visibility_cache[key] = out
        return out

    def camera_pose(self, camera: str) -> Pose:
        """Where a camera is now, in the optical convention (``base.project``).
        Computed, not read back: Isaac Lab's tiled camera records its pose once,
        at start-up, and a wrist camera moves.  The sensors are put where these
        say (``scene_builder``), so the two agree."""
        from ..base import external_camera_pose, wrist_camera_pose

        spec = next(c for c in self._scene_spec.cameras if c.name == camera)
        if spec.kind != "wrist":
            return spec.pose or external_camera_pose(self._scene_spec)
        arm = spec.attached_to or self._scene_spec.arm_ids[0]
        return wrist_camera_pose(self.eef_pose(arm), self._arm_model[arm])

    def camera_position(self, camera: str) -> Tuple[float, float, float]:
        return self.camera_pose(camera).pos

    def object_masks(self, camera: str) -> Dict[str, object]:
        """Each labelled body's pixels in this camera's segmentation now."""
        import numpy as np

        from .scene_builder import camera_key

        self._ensure_segmented()
        cam = self._scene[camera_key(camera)]
        seg = cam.data.output.get("semantic_segmentation")
        info = (cam.data.info or {}).get("semantic_segmentation") or {}
        if seg is None or not info:
            return {}
        ids = np.asarray(seg[0].detach().cpu().numpy() if hasattr(seg[0], "detach") else seg[0])
        ids = ids.reshape(ids.shape[0], ids.shape[1])
        out: Dict[str, object] = {}
        for raw_id, label in info.get("idToLabels", {}).items():
            name = label.get("class") if isinstance(label, dict) else str(label)
            if not name or name.upper() in ("BACKGROUND", "UNLABELLED"):
                continue
            try:
                mask = ids == int(raw_id)
            except (TypeError, ValueError):
                continue
            if mask.any():
                name = self._label_names.get(name.lower(), name)
                out[name] = mask if name not in out else (out[name] | mask)
        return out

    def pixel_counts(self, camera: str) -> Dict[str, int]:
        """How many pixels of each labelled body this camera sees right now."""
        from .scene_builder import camera_key

        self._ensure_segmented()
        cam = self._scene[camera_key(camera)]
        seg = cam.data.output.get("semantic_segmentation")
        info = (cam.data.info or {}).get("semantic_segmentation") or {}
        if seg is None or not info:
            return {}
        ids, counts = seg[0].unique(return_counts=True)
        seen = dict(zip((int(i) for i in ids.tolist()), counts.tolist()))
        out: Dict[str, int] = {}
        for raw_id, label in info.get("idToLabels", {}).items():
            try:
                n = seen.get(int(raw_id), 0)
            except (TypeError, ValueError):
                continue
            name = label.get("class") if isinstance(label, dict) else str(label)
            if n and name and name.upper() not in ("BACKGROUND", "UNLABELLED"):
                name = self._label_names.get(name.lower(), name)
                out[name] = out.get(name, 0) + int(n)
        return out

    # -- control -----------------------------------------------------------
    def set_arm_target(self, arm: str, pose: Pose) -> None:
        self._targets[arm] = pose
        if self._joint_goal.pop(arm, None) is not None:     # a pose now, not joints
            self._homing.pop(arm, None)
            self._home_plan.pop(arm, None)
            self._commanded[arm] = self.eef_pose(arm)
        rest = self._rest_pose(arm)
        home = (rest is not None and pose.distance_to(rest) <= HOME_POS_TOL
                and pose.angle_to(rest) <= HOME_ROT_TOL)
        if home and arm not in self._homing:
            robot = self._scene[self._arm_key[arm]]
            self._homing[arm] = robot.data.joint_pos[:, self._entity[arm].joint_ids].clone()
            self._home_plan[arm] = self._plan_home(arm, self._homing[arm])
        elif not home and arm in self._homing:
            # the IK takes over from where the hand is
            del self._homing[arm]
            self._home_plan.pop(arm, None)
            self._commanded[arm] = self.eef_pose(arm)

    def stop_arm(self, arm: str) -> None:
        """Hold where the arm is now -- target and command both, so a cancelled
        motion does not keep interpolating toward the old goal."""
        here = self.eef_pose(arm)
        self._homing.pop(arm, None)
        self._home_plan.pop(arm, None)
        self._joint_goal.pop(arm, None)
        self._targets[arm] = here
        self._commanded[arm] = here

    def set_gripper(self, arm: str, value: float) -> None:
        value = float(min(1.0, max(0.0, value)))
        if value != self._grippers.get(arm):
            self._gripper_changed_at[arm] = self.sim_time
            self._finger_history[arm] = []
        self._grippers[arm] = value

    def arm_at_target(self, arm: str, pos_tol: float = ARRIVAL_POS_TOL,
                      rot_tol: float = ARRIVAL_ROT_TOL) -> bool:
        if arm in self._joint_goal:
            # there once the path is run and the hand is where those joints put it
            plan = self._home_plan.get(arm)
            if plan is not None and plan["step"] < plan["steps"]:
                return False
        elif arm in self._homing:
            return self._home_reached(arm)
        cur = self.eef_pose(arm)                   # not state(): this is called every step
        tgt = self._targets[arm]
        return cur.distance_to(tgt) <= pos_tol and cur.angle_to(tgt) <= rot_tol

    def eef_pose(self, arm: str) -> Pose:
        """The tool centre point, not the hand frame."""
        robot = self._scene[self._arm_key[arm]]
        idx = self._entity[arm].body_ids[0]
        p = robot.data.body_state_w[0, idx, :7].tolist()
        return hand_to_tcp(Pose(tuple(p[:3]), tuple(p[3:7])), self._arm_model[arm])

    def gripper_settled(self, arm: str) -> bool:
        """A gripper command is finished when the fingers have stopped moving.

        Not when they reach the commanded opening: closing on an object stalls
        them short of it, and that stall *is* the grasp.  And not on
        instantaneous velocity either: fingers pressed against an object twitch,
        and a velocity threshold missed the stall in 2 of 6 grasps.  So: the
        opening changed by less than half a millimetre over the last 0.1 s.
        """
        now = self.sim_time
        if now - self._gripper_changed_at.get(arm, -1e9) < GRIPPER_MIN_SECONDS:
            return False
        robot = self._scene[self._arm_key[arm]]
        ids, _ = self._fingers(robot, arm)
        if not ids:
            return True
        opening = float(robot.data.joint_pos[0, ids].abs().sum().item())
        history = self._finger_history.setdefault(arm, [])
        history.append((now, opening))
        del history[:-16]
        past = [o for (ts, o) in history if now - ts >= GRIPPER_STALL_WINDOW]
        return bool(past) and abs(opening - past[-1]) < GRIPPER_STALL_TOLERANCE

    def launch(self, body: str, velocity) -> None:
        import torch

        from .scene_builder import body_key

        obj = self._scene[body_key(body)]
        obj.write_root_velocity_to_sim(torch.tensor([[*map(float, velocity), 0.0, 0.0, 0.0]],
                                                    device=self.device))

    def set_articulation_target(self, joint: str, value: float) -> None:
        import torch
        from .scene_builder import art_key

        spec = next((j for j in self._carried.values() if j.name == joint), None)
        if spec is None:
            raise ConfigError(f"no joint {joint!r} in this scene")
        lo, hi = spec.limits
        self._scene[art_key(joint)].set_joint_position_target(
            torch.tensor([[min(max(value, lo), hi)]], device=self.device))

    def set_joint_drive(self, joint: str, stiffness: float, damping: float) -> None:
        import torch
        from .scene_builder import art_key

        if not any(j.name == joint for j in self._carried.values()):
            raise ConfigError(f"no joint {joint!r} in this scene")
        art = self._scene[art_key(joint)]
        art.write_joint_stiffness_to_sim(torch.tensor([[float(stiffness)]], device=self.device))
        art.write_joint_damping_to_sim(torch.tensor([[float(damping)]], device=self.device))

    def _joint_readings(self, arm_id: str):
        """The arm's joint angles (radians, base to wrist), as its encoders read
        them: proprioception an agent is entitled to (v1.13)."""
        entity = self._entity.get(arm_id) if hasattr(self, "_entity") else None
        key = self._arm_key.get(arm_id)
        if entity is None or key is None:
            return None
        ids = entity.joint_ids
        return tuple(round(float(v), 4) for v in self._scene[key].data.joint_pos[0, ids].tolist())

    # -- grasping ----------------------------------------------------------
    def _finger_gap(self, arm: str) -> Optional[float]:
        """How far apart the fingers are (m), off their joints: each finger's
        travel from closed, added up."""
        key = self._arm_key.get(arm)
        if key is None:
            return None
        robot = self._scene[key]
        ids, _ = self._fingers(robot, arm)
        if not ids:
            return None
        return float(robot.data.joint_pos[0, ids].abs().sum().item())

    def _holding(self, arm: str, gap: Optional[float] = None) -> Optional[str]:
        """What this gripper holds: a cloth it has taken a patch of
        (``isaac.cloth``), else a body (``_holding_rigid``)."""
        if self._cloths is not None and self._cloths.holding(arm):
            return self._cloths.holding(arm)
        return self._holding_rigid(arm, gap)

    def _holding_rigid(self, arm: str, gap: Optional[float] = None) -> Optional[str]:
        """Which body this gripper holds.

        The gripper is commanded closed, the fingers have stopped on something
        (at least ``HELD_MIN_GAP`` apart), and the tool centre point is inside
        a body's box, give or take a margin -- the rule the mock engine uses
        (``base.in_grasp``), so a bar held by one end counts as held.  Without
        the fingers' say, a closed hand pushed onto a doll called it held from
        then on, and so did fingers the doll had slipped out of (bench-issue
        #21).  As in the mock, a button is pressed, never held, and nor is a
        link on a joint: WK2's raft, tapped with a closed hand, was reported
        held until it sailed out of the fingers (v13).
        """
        from ..base import in_grasp

        if self._grippers.get(arm, 0.0) < GRIP_CLOSING or self._scene_spec is None:
            return None
        gap = self._finger_gap(arm) if gap is None else gap
        if gap is not None and gap < HELD_MIN_GAP:
            return None                          # closed on nothing
        hand = self.eef_pose(arm).pos               # the TCP: where a grasp happens
        best, best_d = None, math.inf
        for body in self._scene_spec.bodies:
            if body.fixed or not collides(body) or body.attrs.get("pressable") \
                    or body.attrs.get("carried_by") or body.shape == "cloth":
                continue
            pos, quat, _ = self._body(body.name)
            pose = Pose(pos, quat)
            if not in_grasp(body, pose, hand):
                continue
            d = math.dist(hand, pose.pos)
            if d < best_d:
                best, best_d = body.name, d
        return best

    # -- event hooks -------------------------------------------------------
    def set_instruction(self, text: str, initial: bool = False) -> None:
        self._instruction = text
        if not initial:
            self._instruction_version += 1

    def note_event(self, name: str) -> None:
        self._events.append(name)

    def set_look(self, look=None) -> None:
        """Another table and another light for a demonstration film
        (``environment.apply_look``); None puts the benchmark's own back.
        Textures and light attributes only: the stage stays as it is."""
        import isaacsim.core.utils.stage as stage_utils

        from . import environment
        from .scene_builder import TABLE_PRIM

        self._look = environment.apply_look(stage_utils.get_current_stage(), TABLE_PRIM, look)
        environment.wait_for_assets(self._sim)
        # what was seen and rendered was seen in the other light
        self._visibility_cache = {}
        self._rendered_at = None
        self._segmented_at = None

    def set_color(self, body: str, color: str) -> None:
        """Recolour a primitive body by writing its shader's diffuse colour.

        Isaac Lab spawns each primitive with its own preview-surface material at
        ``<body>/geometry/material``, so this is one attribute write: cheap, and
        safe between episodes, which is what lets colour be a per-seed property
        rather than part of the scene's structure.
        """
        import omni.usd
        from pxr import Gf, Sdf

        from ..base import color_rgb as rgb

        self._colors[body] = color
        from .scene_builder import art_key, body_key

        stage = omni.usd.get_context().get_stage()
        root = f"/World/envs/env_0/{body_key(body)}"
        # Isaac Lab's shape spawners put the material under geometry/; the
        # built bin and lid put it at the body root, a carried body on its link
        candidates = (f"{root}/geometry/material/Shader", f"{root}/material/Shader",
                      f"{root}/material0/Shader")
        if body in self._carried:
            candidates = (f"/World/envs/env_0/{art_key(self._carried[body].name)}"
                          f"/link/material/Shader",)
        shader = next((stage.GetPrimAtPath(c) for c in candidates
                       if stage.GetPrimAtPath(c).IsValid()), None)
        if shader is None:
            raise ConfigError(f"set_color({body}): no preview-surface shader under {root}")
        # a finish (v2) is OmniPBR, shared: the body's material references it,
        # so this write is the body's own override -- paint, plastic and glaze
        # take the colour, fabric its dye; wood and metal keep their own
        from ..finishes import is_finish, linear, split

        # the palette is sRGB, as it should look; the renderer is linear
        name, value = "inputs:diffuseColor", linear(rgb(color))
        if shader.GetAttribute("inputs:glass_color"):
            name = "inputs:glass_color"                    # a glass: its tint
        elif shader.GetAttribute("inputs:diffuse_color_constant") or shader.GetAttribute("inputs:diffuse_tint"):
            kind = split(color)[0] if is_finish(color) else "paint"
            if kind in ("wood", "metal", "paper"):
                return
            name = "inputs:diffuse_tint" if kind == "fabric" else "inputs:diffuse_color_constant"
        attr = shader.GetAttribute(name)
        if not attr:
            attr = shader.CreateAttribute(name, Sdf.ValueTypeNames.Color3f)
        attr.Set(Gf.Vec3f(*value))
        glow = shader.GetAttribute("inputs:emissive_color")      # a flame glows in its colour
        if glow:
            glow.Set(Gf.Vec3f(*value))
