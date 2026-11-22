"""SceneSpec -> an Isaac Lab InteractiveScene.

Primitives are spawned directly; anything with ``BodySpec.asset`` is resolved
through :mod:`ruibench.engines.isaac.assets` and referenced from the Isaac asset
server.  Articulations (drawer, slide, spring) become USD joints with a PhysX
drive -- which is the Isaac form of the rule carried over from the MuJoCo plan:

    **a moving object is driven by a joint, never teleported.**

A kinematic rigid body in PhysX has the same defect a MuJoCo mocap body has: it
pushes other bodies but takes no force back, so anything it carries behaves
inconsistently.  DI2's sliding bowl and DI3's pushed plate ride joint drives.

Everything Isaac is imported inside functions -- and note the ordering rule that
forces it: ``isaaclab`` pulls in ``carb``, which does not exist until
``SimulationApp`` has started.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from ...core.errors import ConfigError
from ...core.geometry import quat_conj, quat_mul
from ...core.types import Pose
from ..base import (
    HORIZONTAL_APERTURE, ArticulationSpec, BodySpec, CameraSpec, SceneSpec,
    camera_apertures, camera_focal_length, collides, color_rgb as rgb, external_camera_pose,
    rides,
)

ENV_NS = "{ENV_REGEX_NS}"

#: SeattleLabTable: origin at its top surface, about 1.05 m tall.  Rotated a
#: quarter turn about z, as in Isaac Lab's lift task, so its long side runs
#: away from the robot.
TABLE_HEIGHT = 1.05
TABLE_ROT = (0.7071068, 0.0, 0.0, 0.7071068)
#: how far in front of the robot bases the table's centre sits
TABLE_AHEAD = 0.45
#: Measured, not assumed: with the table origin at z = table_height a dropped
#: cube came to rest on a surface at table_height - 0.003.  Raising the table
#: by that much makes RUIBench's world convention -- the work surface is at
#: exactly z = table_height -- hold in Isaac too.
TABLE_SURFACE_CORRECTION = 0.003
#: every task body's caps (``body_cfg``): how fast PhysX may push it out of an
#: overlap (m/s: the robots' own is 5; a toy pushed out faster than 1 m/s is
#: thrown, not separated) and what one contact may give it in a step (N s: a
#: 100 g toy held against 70 N of squeeze for a 1/120 s step takes 0.6)
MAX_DEPENETRATION, MAX_CONTACT_IMPULSE = 1.0, 5.0


#: The station table: a slab 1.1 m deep and 1.4 m wide, its near edge 15 cm
#: behind the arm bases, with an oak veneer top on aluminium legs
#: (``environment``).
STATION_TABLE = "station_table"
STATION_TABLE_SIZE = (1.10, 1.40, 0.05)
STATION_TABLE_AHEAD = 0.40
TABLE_PRIM = "/World/envs/env_0/table"

#: Marks live under one prim each, in the environment's frame.
MARKER_ROOT = "/World/envs/env_0/Markers"
#: ...and float this far above the work surface: the table's visible top is
#: TABLE_SURFACE_CORRECTION above where things rest, and a mark under it is
#: not drawn at all.
MARKER_LIFT = TABLE_SURFACE_CORRECTION + 0.0005


def table_center_x(spec) -> float:
    from ..robots import model_of

    xs = [r.base_pose.pos[0] for r in spec.robots] or [0.0]
    ahead = STATION_TABLE_AHEAD if spec.table == STATION_TABLE else TABLE_AHEAD
    if spec.table == STATION_TABLE and spec.robots and model_of(spec.robots[0]).ego_camera:
        ahead = model_of(spec.robots[0]).table_ahead
    return sum(xs) / len(xs) + ahead


def gripper_label(robot: str) -> str:
    return f"{robot}_gripper"


def marker_path(name: str) -> str:
    return f"{MARKER_ROOT}/{name}"


def marker_label(name: str) -> str:
    return f"marker_{name}"


def art_key(joint: str) -> str:
    return f"art_{joint}"


def carried_bodies(spec: SceneSpec) -> Dict[str, ArticulationSpec]:
    """body name -> the joint that carries it."""
    return {j.parent: j for j in spec.articulations if j.parent}


AXES = {(1.0, 0.0, 0.0): "X", (0.0, 1.0, 0.0): "Y", (0.0, 0.0, 1.0): "Z"}


def link_parts(body: BodySpec) -> tuple:
    """A carried body's shapes, in its own frame, for the articulation spawner."""
    from .spawners import bin_boxes, lid_boxes

    if body.shape == "compound":
        return tuple(body.attrs["parts"])
    if body.shape == "bin":
        boxes = bin_boxes(body.size, float(body.attrs.get("wall", 0.01)),
                          body.attrs.get("container") == "inverted")
    elif body.shape == "lid":
        boxes = lid_boxes(body.size, float(body.attrs["plate"]), tuple(body.attrs["handle_bar"]))
    elif body.shape == "cylinder":
        return (("cylinder", (0.0, 0.0, 0.0), body.size[0], body.size[2]),)
    elif body.shape == "box":
        boxes = {"box": ((0.0, 0.0, 0.0), tuple(body.size))}
    else:
        raise ConfigError(f"a joint cannot carry a {body.shape!r} ({body.name})")
    handle = body.attrs.get("handle")
    if handle:
        boxes["handle"] = (tuple(handle[0]), tuple(handle[1]))
    return tuple(("box", centre, half) for centre, half in boxes.values())


def joint_gains(joint: ArticulationSpec):
    """(stiffness, damping): driven joints track their target hard; free ones
    only have friction-like damping, and a spring pulls toward its target."""
    if joint.actuated:
        return 2.0e4, 2.0e3
    return joint.stiffness, joint.damping


def _articulation_cfg(joint: ArticulationSpec, body: BodySpec, spec: Optional[SceneSpec] = None):
    """A carried body as the moving link of a one-joint articulation -- its
    base fixed to the world, or (a mounted joint) to the link of the joint
    that carries the body it stands on."""
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    from .finishes import visual_cfg
    from .spawners import slide_cfg_class

    if joint.kind not in ("slide", "hinge"):
        raise ConfigError(f"joint {joint.name!r}: a joint slides or turns, not {joint.kind!r}")
    axis = AXES.get(tuple(float(v) for v in joint.axis))
    if axis is None:
        raise ConfigError(f"joint {joint.name!r}: the axis must be +x, +y or +z")
    mount = {}
    if joint.mounted_on:
        from ...core.geometry import relative_pose
        from ...core.types import Pose

        carrier = carried_bodies(spec).get(joint.mounted_on)
        if carrier is None:
            raise ConfigError(f"joint {joint.name!r} is mounted on {joint.mounted_on!r}, which no "
                              "joint carries: mount it on a jointed body, or on nothing")
        base = next(b for b in spec.bodies if b.name == joint.mounted_on)
        rel = relative_pose(Pose(tuple(base.pose.pos), world_quat(base)),
                            Pose(tuple(body.pose.pos), world_quat(body)))
        # the concrete prim: a USD relationship takes no ENV_REGEX_NS (one env)
        mount = {"mount": f"/World/envs/env_0/{art_key(carrier.name)}/link",
                 "mount_pos": tuple(rel.pos), "mount_rot": tuple(rel.quat)}
    spawn = slide_cfg_class()(
        parts=link_parts(body), joint_name=joint.name, kind=joint.kind, axis=axis,
        limits=tuple(joint.limits), pivot=tuple(joint.pivot),
        mass=body.mass, label=body.name, friction=body.attrs.get("friction"),
        visual_material=visual_cfg(body.color), **mount)
    stiffness, damping = joint_gains(joint)
    return ArticulationCfg(
        prim_path=f"{ENV_NS}/{art_key(joint.name)}", spawn=spawn,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=tuple(body.pose.pos), rot=world_quat(body), joint_pos={joint.name: joint.initial}),
        actuators={"joint": ImplicitActuatorCfg(joint_names_expr=[joint.name],
                                                stiffness=stiffness, damping=damping,
                                                effort_limit=200.0, velocity_limit=1.0)},
    )


def body_key(name: str) -> str:
    """Scene-entity key for a body.  Prefixed so a body called "table" or
    "robot0" cannot collide with the scene's own entities."""
    return f"obj_{name}"


def camera_key(name: str) -> str:
    return f"cam_{name}"


def spawn_markers(spec: SceneSpec) -> None:
    """Draw every mark as flat strokes that collide with nothing.

    One prim per mark carries its pose, so a new episode moves that prim and
    nothing else (``IsaacEngine._place_markers``); the strokes keep their
    offsets, which is why a mark's shape and text belong to the scene's
    structure.
    """
    import math

    import isaaclab.sim as sim_utils
    import isaacsim.core.utils.prims as prim_utils

    from ..base import STROKE_HEIGHT, STROKE_WIDTH, marker_strokes
    from ..finishes import linear

    if spec.markers:
        prim_utils.create_prim(MARKER_ROOT, "Xform")
    for m in spec.markers:
        root = marker_path(m.name)
        prim_utils.create_prim(root, "Xform", translation=m.pose.pos, orientation=m.pose.quat)
        material = f"{root}/material"
        sim_utils.spawn_preview_surface(
            material, sim_utils.PreviewSurfaceCfg(diffuse_color=linear(rgb(m.color)), roughness=0.7))
        for i, ((x1, y1), (x2, y2)) in enumerate(marker_strokes(m)):
            yaw = math.atan2(y2 - y1, x2 - x1)
            path = f"{root}/stroke{i}"
            sim_utils.spawn_cuboid(
                path, sim_utils.CuboidCfg(size=(math.hypot(x2 - x1, y2 - y1) + STROKE_WIDTH,
                                                STROKE_WIDTH, STROKE_HEIGHT)),
                translation=((x1 + x2) / 2, (y1 + y2) / 2, MARKER_LIFT + STROKE_HEIGHT / 2),
                orientation=(math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
            sim_utils.bind_visual_material(path, material)


def build_scene_cfg(spec: SceneSpec, cfg):
    """Assemble the InteractiveSceneCfg for ``spec``, as a live class."""
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab.utils import configclass

    from . import environment
    from .assets import get

    fields: Dict[str, object] = {}

    # Layout follows Isaac Lab's own Franka tasks: the table's origin is its top
    # surface, the robot stands on that surface, and the floor is one table
    # height below.  RUIBench keeps its world convention (work surface at
    # z = table_height), so everything is expressed relative to that.
    floor_z = spec.table_height - TABLE_HEIGHT
    # the ground plane only collides: ``environment.finish`` hides its grid,
    # and the room's own floor is what the cameras see
    fields["ground"] = AssetBaseCfg(
        prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, floor_z)),
    )
    fields.update(environment.room_cfgs((table_center_x(spec), 0.0), floor_z))
    if spec.table == STATION_TABLE:
        sx, sy, sz = STATION_TABLE_SIZE
        fields["table"] = AssetBaseCfg(
            prim_path=f"{ENV_NS}/table",
            spawn=sim_utils.CuboidCfg(
                size=(sx, sy, sz), collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=environment.tabletop_material()),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(table_center_x(spec), 0.0, spec.table_height - sz / 2)))
        fields.update(environment.leg_cfgs(ENV_NS, (table_center_x(spec), 0.0), STATION_TABLE_SIZE,
                                           spec.table_height, floor_z))
    elif spec.table:
        fields["table"] = AssetBaseCfg(
            prim_path=f"{ENV_NS}/table",
            spawn=sim_utils.UsdFileCfg(usd_path=get(spec.table).resolve()),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=(table_center_x(spec), 0.0,
                     spec.table_height + TABLE_SURFACE_CORRECTION),
                rot=TABLE_ROT),
        )

    for robot in spec.robots:
        fields.update(_robot_cfgs(robot, spec.table_height))

    carried = carried_bodies(spec)
    for body in spec.bodies:
        if body.shape == "cloth":
            continue                      # not a rigid body: ``isaac.cloth.spawn`` puts it on the stage
        if body.name in carried:
            fields[art_key(carried[body.name].name)] = _articulation_cfg(carried[body.name], body, spec)
        else:
            fields[body_key(body.name)] = _rigid_cfg(body)

    for cam in spec.cameras:
        fields[camera_key(cam.name)] = _camera_cfg(spec, cam)

    loose = [j.name for j in spec.articulations if j.parent is None]
    if loose:
        raise ConfigError("a joint must carry a body; these carry none: " + ", ".join(loose))

    fields["__annotations__"] = {k: type(v) for k, v in fields.items()}
    fields["num_envs"] = 1
    fields["env_spacing"] = 4.0
    return configclass(type("RUIBenchSceneCfg", (InteractiveSceneCfg,), fields))()


def _robot_cfgs(rspec, table_height: float) -> Dict[str, object]:
    """Every arm of a robot, as articulations standing on the work surface, and
    the pedestals they stand on.

    ``RobotSpec.base_pose.z`` is read **relative to the work surface**.  The
    Isaac Franka has no pedestal: placed at z = 0 it stands on the floor, a full
    table height below where the task happens, at the edge of its reach.  A
    Piper station builds both its arms whether or not a task commands them.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.assets import AssetBaseCfg

    from ..robots import arm_base, model_of
    from .environment import LEG_MATERIAL, nvidia_dir
    from .robots import arm_key, isaac_robot, station_arms

    model = model_of(rspec)
    isaac = isaac_robot(model.name)
    out: Dict[str, object] = {}
    for arm in station_arms(rspec):
        key = arm_key(rspec, arm)
        x, y, z = arm_base(rspec, arm, table_height)
        out[key] = isaac.articulation(f"{ENV_NS}/{key}", (x, y, z), tuple(rspec.base_pose.quat),
                                      isaac.ready)
        if model.pedestal:
            hx, hy, hz = model.pedestal
            out[f"{key}_pedestal"] = AssetBaseCfg(
                prim_path=f"{ENV_NS}/{key}_pedestal",
                spawn=sim_utils.CuboidCfg(
                    size=(2 * hx, 2 * hy, 2 * hz),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.MdlFileCfg(
                        mdl_path=LEG_MATERIAL.replace("{NVIDIA}", nvidia_dir()))),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(x, y, z - hz)))
    return out


def _rigid_cfg(body: BodySpec):
    import isaaclab.sim as sim_utils
    from isaaclab.assets import RigidObjectCfg

    from .assets import get, primitives
    from .finishes import visual_cfg

    # ``angular_damping`` stands in for rolling resistance, which PhysX does
    # not have: a rubber duck lying on its side rocked in a drawer for good.
    # Every task body's way out of an overlap, and what one contact can push,
    # are capped (v15, the RUIKit collaborator's audit): uncapped, a body a
    # finger had pressed into another could be shot across the room -- a
    # "success" the physics audit (``scoring/audit.py``) would also have to
    # catch.  Nothing an agent commands can change these, nor gravity or
    # kinematics, during an episode: no interface reaches the engine
    still = body.fixed or rides(body)                # moved by the engine, never by physics
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=still,
                                             disable_gravity=still,
                                             angular_damping=body.attrs.get("angular_damping"),
                                             max_depenetration_velocity=MAX_DEPENETRATION,
                                             max_contact_impulse=MAX_CONTACT_IMPULSE)
    mass = sim_utils.MassPropertiesCfg(mass=body.mass)
    collision = sim_utils.CollisionPropertiesCfg(collision_enabled=collides(body))
    # A name declared a primitive is a chosen shape, not a missing mesh: Isaac's
    # YCB subset has no plate and no ball, so those are built.
    if body.asset and body.asset not in primitives():
        entry = get(body.asset)
        if entry.physics == "added":
            from .spawners import physics_usd_cfg_class
            spawn = physics_usd_cfg_class()(
                usd_path=entry.resolve(), rigid_props=rigid, mass_props=mass,
                collision_props=collision,
                collision_approximation=entry.collision_approximation,
                semantic_tags=[("class", body.name)])
        else:
            spawn = sim_utils.UsdFileCfg(usd_path=entry.resolve(),
                                         rigid_props=rigid, mass_props=mass,
                                         semantic_tags=[("class", body.name)])
    else:
        shared = dict(rigid_props=rigid, mass_props=mass,
                      collision_props=collision,
                      visual_material=visual_cfg(body.color),
                      semantic_tags=[("class", body.name)])
        if body.shape == "bin":
            from .spawners import open_bin_cfg_class
            spawn = open_bin_cfg_class()(
                half_extents=tuple(body.size), wall=float(body.attrs.get("wall", 0.01)),
                inverted=body.attrs.get("container") == "inverted",
                rigid_props=rigid, mass_props=mass,
                visual_material=visual_cfg(body.color),
                semantic_tags=[("class", body.name)])
        elif body.shape == "lid":
            from .spawners import lid_cfg_class
            spawn = lid_cfg_class()(
                half_extents=tuple(body.size), plate=float(body.attrs["plate"]),
                handle=tuple(body.attrs["handle_bar"]), rigid_props=rigid, mass_props=mass,
                visual_material=visual_cfg(body.color),
                semantic_tags=[("class", body.name)])
        elif body.shape == "compound":
            from .spawners import compound_cfg_class
            spawn = compound_cfg_class()(
                parts=tuple(body.attrs["parts"]), rigid_props=rigid, mass_props=mass,
                friction=body.attrs.get("friction"), solid=not rides(body),
                semantic_tags=[("class", body.name)])
        elif body.shape == "box":
            spawn = sim_utils.CuboidCfg(size=tuple(2 * s for s in body.size), **shared)
        elif body.shape == "cylinder":
            spawn = sim_utils.CylinderCfg(radius=body.size[0], height=2 * body.size[2], **shared)
        elif body.shape == "sphere":
            spawn = sim_utils.SphereCfg(radius=body.size[0], **shared)
        else:
            raise ConfigError(f"unsupported shape {body.shape!r} for body {body.name!r}")

    return RigidObjectCfg(
        prim_path=f"{ENV_NS}/{body_key(body.name)}", spawn=spawn,
        init_state=RigidObjectCfg.InitialStateCfg(pos=tuple(body.pose.pos),
                                                  rot=world_quat(body)),
    )


def upright_quat(body: BodySpec) -> Tuple[float, float, float, float]:
    """The rotation that stands this body's asset up (identity for primitives)."""
    if not body.asset or body.attrs.get("raw_orientation"):
        return (1.0, 0.0, 0.0, 0.0)
    from .assets import get, primitives
    if body.asset in primitives():
        return (1.0, 0.0, 0.0, 0.0)
    return get(body.asset).upright_quat


def world_quat(body: BodySpec):
    """Task orientation -> simulator orientation: task rotation, then stand up."""
    return quat_mul(tuple(body.pose.quat), upright_quat(body))


def task_quat(body: BodySpec, world):
    """Simulator orientation -> the orientation task code sees (standing = identity)."""
    return quat_mul(tuple(world), quat_conj(upright_quat(body)))


#: Both camera placements are engine-independent (``engines.base``): the wrist
#: mount is Isaac Lab's own (an earlier hand-written orientation rendered a
#: blank image), the external pose is ours (Isaac Lab's stacking-task table
#: camera stood 0.3 m from the sorting bins and cut them off at the edges).



#: A wrist camera's near clipping plane.  Its housing -- the dark box round
#: it, and the X5's own D405 -- reaches at most 2.4 cm along what it sees
#: (the FR3's 2.2); nearer than the plane a thing is cut open.  At 5 cm, a
#: flower pot held in DS2 came out cut, its inside showing (v10); DS3's
#: detector box on the handle was cut into a black block (v7)
WRIST_NEAR = 0.026


def _camera_cfg(spec: SceneSpec, cam: CameraSpec):
    """Cameras carry a semantic-segmentation channel as well as RGB.

    Visibility is not a nicety for us: the Active Perception suite rests
    entirely on the agent genuinely not knowing what it has not looked at, so
    "is this object in view" has to be answered by the renderer.

    The external camera sits past the far edge of the work area, facing back
    toward the robot, so one image shows both the workspace and the arm.  A
    camera spec with an explicit pose (the snapshot check's top-down view) is
    taken as given, in the same optical convention.
    """
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg

    data_types = ["rgb", "semantic_segmentation"] + (["depth"] if cam.depth else [])
    across, down = camera_apertures(spec, cam)
    near = WRIST_NEAR if cam.kind == "wrist" else 0.05
    pinhole = sim_utils.PinholeCameraCfg(focal_length=camera_focal_length(spec, cam),
                                         focus_distance=400.0,
                                         horizontal_aperture=across, vertical_aperture=down,
                                         clipping_range=(near, 5.0))

    if cam.kind == "wrist":
        from ...core.geometry import compose
        from ..robots import model_of
        from .robots import arm_key, isaac_robot

        arm_id = cam.attached_to or spec.arm_ids[0]
        robot = next(r for r in spec.robots if r.name == arm_id.split("/")[0])
        model = model_of(robot)
        mount = compose(model.tcp_from_body, model.camera_from_tcp)   # in the grasp body's frame
        pos, rot = tuple(mount.pos), tuple(mount.quat)
        key = arm_key(robot, arm_id.split("/")[1])
        return TiledCameraCfg(
            prim_path=f"{ENV_NS}/{key}/{isaac_robot(model.name).grasp_body}/{cam.name}",
            offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
            data_types=data_types, spawn=pinhole,
            width=cam.width, height=cam.height, colorize_semantic_segmentation=False,
        )

    pose = cam.pose or external_camera_pose(spec)
    pos, rot = tuple(pose.pos), tuple(pose.quat)
    return TiledCameraCfg(
        prim_path=f"{ENV_NS}/{cam.name}",
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention="ros"),
        data_types=data_types, spawn=pinhole,
        width=cam.width, height=cam.height, colorize_semantic_segmentation=False,
    )
