"""What the Isaac backend needs to build and drive each robot model.

The engine-independent half (mounts, tool frame, camera, hand envelope) is
``engines/robots.py``.  This half says which USD to spawn, which joints are the
arm and which the fingers, where the arm is at rest, and how the joints are
driven.  Nothing here is imported before the simulator starts.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Mapping, Optional, Tuple

from ...core.errors import ConfigError
from ...core.paths import data_dir


#: Where fetched robot descriptions, scans and converted USD live: third_party/
#: in a source checkout (git-ignored), the user's cache otherwise
#: (``core/paths.py``).
THIRD_PARTY = Path(os.environ.get("RUIBENCH_THIRD_PARTY")
                   or data_dir("third_party", "third_party"))

#: The ARX X5 description: ``arx5_description`` (BSD-3-Clause, ARX Robotics) as
#: collected in CarlsonGYC/robot-descriptions-arx, pinned.  Its meshes are glTF,
#: which the URDF importer does not read: ``fetch_arx5`` writes them out as OBJ
#: and points a copy of the URDF at them.
ARX5_REPO = "https://raw.githubusercontent.com/CarlsonGYC/robot-descriptions-arx"
ARX5_COMMIT = "71b069961eafe3d71524d199d99ac24c752b348a"
ARX5_DIR = THIRD_PARTY / "arx5"
ARX5_URDF = ARX5_DIR / "arx5_obj.urdf"
ARX5_USD_DIR = THIRD_PARTY / "assets_usd/arx5"
#: the darkest a part of the X5 is drawn (its description's 0.03 read as a
#: silhouette in shadow)
ARX5_DARKEST = 0.25


@dataclass(frozen=True)
class FingerCoupling:
    """Two fingers driven as one, the way a parallel gripper's linkage moves
    them.

    The URDF drives each finger on its own.  Pressed against an object, both
    drives sit at their force limit, their pushes cancel, and nothing holds
    the object between them: carried sideways at 30 cm/s, a ring slid 12 mm
    along the fingers and missed its peg (2026-09-17).  Closing, both fingers
    are therefore told, every physics step, to go to ``squeeze / stiffness``
    inside their *mean* opening: each presses with ``squeeze`` and the
    difference between them is pulled to zero at the full drive stiffness --
    through the simulator's implicit drive.  (Computing the forces here
    instead was unstable: a 30 g finger on a 4000 N/m spring outruns a 120 Hz
    step.)  Opening, both go to the open position.
    """

    squeeze: float = 8.0              # N per finger
    stiffness: float = 2000.0         # N/m, the finger drive's


@dataclass(frozen=True)
class IsaacRobot:
    model: str
    arm_joints: str                      # joint-name regex
    finger_joints: str
    grasp_body: str
    #: links labelled for segmentation, so the render check can count them
    gripper_links: Tuple[str, ...]
    #: joint positions an arm starts from and returns to when idle
    ready: Mapping[str, float]
    #: each finger joint's value when open; closed is 0
    finger_open: Mapping[str, float]
    #: builds the ArticulationCfg for one arm
    articulation: Callable[..., object] = field(repr=False, default=None)
    #: None: each finger tracks its own position target
    finger_coupling: Optional[FingerCoupling] = None
    #: rubber on the finger links: (links, static friction, dynamic friction),
    #: combined with what they touch by taking the larger.  None: the default
    #: material (0.5).  No friction holds a round thing against turning about
    #: the fingers' axis -- the pads touch it along a line, and torsional
    #: friction (a contact patch radius) did not help either: a rolling pin
    #: held by one round handle still tilted 30 degrees on Isaac.  Such
    #: handles get a square collider instead (``engines.parts.grip``).
    finger_pads: Optional[Tuple[Tuple[str, ...], float, float]] = None
    #: link -> NVIDIA base material (``Materials/Base/...``, no extension) or
    #: one of our finishes ("plastic:dark"), bound over whatever the converted
    #: description carries.  Empty: as converted.
    link_materials: Mapping[str, str] = field(default_factory=dict)


#: Isaac Sim ships it (``Robots/Franka/FR3/fr3.usd``, the Franka hand
#: included); Isaac Lab 2.1 has no configuration for it, so this is the Panda's
#: high-PD one with FR3 names.  Its joints are the Panda's under an ``fr3_``
#: prefix, and the hand is the same Franka hand.
FR3_USD = "{NUCLEUS}/Robots/Franka/FR3/fr3.usd"


def _fr3_cfg(prim_path: str, pos, rot, ready):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    from .assets import nucleus_dir

    # like the Panda's high-PD configuration: gravity off for the links and stiff
    # joints, so the differential IK is tracked closely.  Implicit actuators take
    # their speed limit from ``velocity_limit_sim``; the fingers' targets are
    # set by the engine (``FingerCoupling``), which presses with 8 N per finger.
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=FR3_USD.replace("{NUCLEUS}", nucleus_dir()),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,
                                                         max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12,
                solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot),
                                                   joint_pos=dict(ready)),
        actuators={
            "shoulder": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[1-4]"], effort_limit=87.0,
                                            velocity_limit_sim=2.0, stiffness=400.0, damping=80.0),
            "forearm": ImplicitActuatorCfg(joint_names_expr=["fr3_joint[5-7]"], effort_limit=12.0,
                                           velocity_limit_sim=2.5, stiffness=400.0, damping=80.0),
            "hand": ImplicitActuatorCfg(joint_names_expr=["fr3_finger_joint.*"], effort_limit=20.0,
                                        velocity_limit_sim=0.1, stiffness=2000.0, damping=100.0),
        },
        soft_joint_pos_limit_factor=1.0,
    )


FR3 = IsaacRobot(
    model="fr3",
    arm_joints="fr3_joint[1-7]",
    finger_joints="fr3_finger_joint.*",
    grasp_body="fr3_hand",
    gripper_links=("fr3_hand", "fr3_leftfinger", "fr3_rightfinger"),
    # the tool centre point 30 cm in front of the base and 30 cm over the table,
    # straight down -- where the fixed camera sees the whole hand and the wrist
    # camera, behind the hand, looks ahead over the work area.  Solved by driving there on
    # Isaac (research/fr3_probe.py); the elbow (joint 4) is the joint nearest its
    # limit, 0.20 rad, and every move out from here opens it
    ready={"fr3_joint1": 0.0, "fr3_joint2": -0.7921, "fr3_joint3": 0.0, "fr3_joint4": -2.8390,
           "fr3_joint5": 0.0, "fr3_joint6": 2.0469, "fr3_joint7": 0.7854,
           "fr3_finger_joint.*": 0.04},
    finger_open={"fr3_finger_joint1": 0.04, "fr3_finger_joint2": 0.04},
    articulation=_fr3_cfg,
    finger_coupling=FingerCoupling(),
    finger_pads=(("fr3_leftfinger", "fr3_rightfinger"), 1.2, 1.0),
)


def arx5_usd() -> str:
    """The ARX X5 as USD, converted from the fetched URDF the first time and
    again whenever the URDF changes (needs the simulator running).  Until v10
    a USD once made was kept: v9's move of the
    wrist camera and its bracket reached the URDF and the sensor, never the
    body drawn -- the D405 stood 4 cm in front of its own lens, hidden only
    by the 5 cm near clip."""
    import hashlib

    usd = ARX5_USD_DIR / "arx5.usd"
    stamp = ARX5_USD_DIR / "urdf.sha256"
    if not ARX5_URDF.exists():
        raise ConfigError(f"the ARX X5 description is missing ({ARX5_URDF}); fetch it with "
                          f"ruibench.engines.isaac.robots.fetch_arx5() ({ARX5_COMMIT[:12]})")
    digest = hashlib.sha256(ARX5_URDF.read_bytes()).hexdigest()
    if usd.exists() and stamp.exists() and stamp.read_text().strip() == digest:
        return str(usd)
    from isaaclab.sim.converters import UrdfConverter, UrdfConverterCfg

    ARX5_USD_DIR.mkdir(parents=True, exist_ok=True)
    conv = UrdfConverter(UrdfConverterCfg(
        asset_path=str(ARX5_URDF), usd_dir=str(ARX5_USD_DIR), usd_file_name="arx5.usd",
        force_usd_conversion=True, make_instanceable=False, fix_base=True,
        merge_fixed_joints=True, collider_type="convex_hull", self_collision=False,
        joint_drive=UrdfConverterCfg.JointDriveCfg(
            gains=UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=80.0))))
    stamp.write_text(digest + "\n")
    return conv.usd_path


def _arx5_cfg(prim_path: str, pos, rot, ready):
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg

    # gravity off, stiff joints, the fingers set by the engine
    # (``FingerCoupling``: 8 N each, 10 cm/s)
    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=arx5_usd(),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(disable_gravity=True,
                                                         max_depenetration_velocity=5.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False, solver_position_iteration_count=12,
                solver_velocity_iteration_count=1)),
        init_state=ArticulationCfg.InitialStateCfg(pos=tuple(pos), rot=tuple(rot),
                                                   joint_pos=dict(ready)),
        actuators={
            "arm": ImplicitActuatorCfg(joint_names_expr=["joint[1-6]"], effort_limit=100.0,
                                       stiffness=400.0, damping=80.0),
            "gripper": ImplicitActuatorCfg(joint_names_expr=["gripper_joint", "joint8"],
                                           effort_limit=20.0, velocity_limit_sim=0.1,
                                           stiffness=2000.0, damping=100.0),
        },
    )


#: Ready pose: RoboDojo's -- every joint at 0, the arm folded low beside its
#: base, the tool pointing forward 15.6 cm over the table.  Until v5 (9/22)
#: it stood with its elbow 36 cm up and the hand over the bench, and in the
#: head camera the two hands covered the work.  Straight down, the X5 cannot
#: hold its tool higher than 18 cm over its own base -- joint 4 turns only 90
#: degrees each way -- so a task's first move turns the hand down.
ARX5 = IsaacRobot(
    model="arx5",
    arm_joints="joint[1-6]",
    finger_joints="gripper_joint|joint8",
    grasp_body="link6",
    gripper_links=("link6", "link7", "link8"),
    ready={"joint1": 0.0, "joint2": 0.0, "joint3": 0.0, "joint4": 0.0,
           "joint5": 0.0, "joint6": 0.0, "gripper_joint": 0.044, "joint8": 0.044},
    finger_open={"gripper_joint": 0.044, "joint8": 0.044},
    articulation=_arx5_cfg,
    finger_coupling=FingerCoupling(),
    finger_pads=(("link7", "link8"), 1.2, 1.0),
    # The real arm is black anodised aluminium.  Neither the description's own
    # colours (``fetch_arx5``) nor NVIDIA's dark metals survive the room's
    # light: both drew the arm as a flat black silhouette in the head camera.
    # A dark satin finish of ours keeps it dark and keeps its shape.
    link_materials={link: "plastic:dark" for link in (
        "base_link", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "link8")},
)

ROBOTS: Dict[str, IsaacRobot] = {"fr3": FR3, "arx5": ARX5}


def isaac_robot(model_name: str) -> IsaacRobot:
    try:
        return ROBOTS[str(model_name).lower()]
    except KeyError:
        raise ConfigError(f"no Isaac configuration for robot model {model_name!r}") from None


def arm_key(robot, arm: str) -> str:
    """The scene key of the articulation an arm belongs to.  A Panda is one
    articulation named after its robot; a station has one per arm."""
    from ..robots import model_of

    if len(model_of(robot).arm_mounts) == 1:
        return robot.name
    return f"{robot.name}_{arm}"


def station_arms(robot):
    """Every arm a robot has, commanded or not: all of them are built."""
    from ..robots import model_of

    return tuple(model_of(robot).arm_mounts)


#: the X5's own meshes for the arm, the R5 gripper it ships with (as in the
#: description), and the wrist camera's bracket
ARX5_MESHES = ("x5/base_link", "x5/link1", "x5/link2", "x5/link3", "x5/link4", "x5/link5",
               "r5/link6", "r5/link7", "r5/link8", "camera_stand/stand_d405")


def fetch_arx5(dest: Path = ARX5_DIR) -> Path:
    """Download the pinned ARX X5 description (~11 MB) and make it importable:
    every glTF mesh written out as OBJ, and a copy of the URDF pointing at them.

    Each glTF mesh is written out part by part, one OBJ per material, and
    drawn as that many visuals in its link with the description's own colours
    -- near-black anodising, dark grey, and light grey bands -- as URDF
    materials: one material over a whole link drew the arm as a flat black
    silhouette in the head camera (9/22).

    Two changes to the URDF besides the paths.  Each finger's collider is the
    convex hull of the finger itself, without the carriage it slides on (that
    one reaches across the whole gripper), instead of the description's 8 x 4
    x 6.4 cm box, which stops 5 mm short of the fingers' inner faces -- closed,
    the boxes leave a 1 cm gap, so nothing thinner could be held.  And the
    RealSense D405 on the bracket, whose mesh lives in another package, is
    drawn as its own collision box."""
    import re
    import urllib.request

    import trimesh

    if ARX5_URDF.exists():
        return ARX5_URDF
    base = f"{ARX5_REPO}/{ARX5_COMMIT}/arx5_description"
    (dest / "obj").mkdir(parents=True, exist_ok=True)
    urdf = dest / "arx5.urdf"
    for rel, out in [("urdf/arx5.urdf", urdf), ("LICENSE", dest / "LICENSE")] + [
            (f"meshes/{m}.glb", dest / "meshes" / f"{m}.glb") for m in ARX5_MESHES]:
        if not out.exists():
            out.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(f"{base}/{rel}", out)
    parts = {}                           # obj stem -> [(part obj, rgba)]
    for m in ARX5_MESHES:
        stem = m.replace("/", "_")
        mesh = trimesh.load(dest / "meshes" / f"{m}.glb", force="mesh")
        mesh.export(dest / "obj" / f"{stem}.obj", include_texture=False)     # colliders
        scene = trimesh.load(dest / "meshes" / f"{m}.glb")
        for k, node in enumerate(scene.graph.nodes_geometry):
            transform, name = scene.graph[node]
            part = scene.geometry[name].copy()
            part.apply_transform(transform)
            rgba = getattr(part.visual.material, "baseColorFactor", None)
            rgba = tuple(float(v) / 255 for v in rgba) if rgba is not None else (0.1, 0.1, 0.1, 1.0)
            # the near-black anodising lifted to a dark grey: the arm on the
            # head camera's shadow side was a shapeless black blob
            rgba = tuple(max(c, ARX5_DARKEST) for c in rgba[:3]) + tuple(rgba[3:])
            path = dest / "obj" / f"{stem}_part{k}.obj"
            part.export(path, include_texture=False)
            parts.setdefault(stem, []).append((path, rgba))
    text = urdf.read_text(encoding="utf-8")
    text = re.sub(r"package://arx5_description/meshes/(\w+)/(\w+)\.glb",
                  lambda g: str(dest / "obj" / f"{g.group(1)}_{g.group(2)}.obj"), text)

    def coloured(block):
        """One <visual> per part of the mesh it draws, each in its colour."""
        found = re.search(r'filename="[^"]*/obj/(\w+)\.obj"', block.group(0))
        if not found or found.group(1) not in parts:
            return block.group(0)
        out = []
        for path, rgba in parts[found.group(1)]:
            v = re.sub(r'filename="[^"]*"', f'filename="{path}"', block.group(0))
            colour = " ".join(f"{c:.3f}" for c in rgba)
            v = v.replace("</visual>", f'<material name="arx_{path.stem}">'
                                       f'<color rgba="{colour}"/></material></visual>')
            out.append(v)
        return "\n".join(out)

    text = re.sub(r"<visual>.*?</visual>", coloured, text, flags=re.S)
    for finger in ("link7", "link8"):
        # the finger without its carriage, which rides inside the gripper body
        tip = trimesh.load(dest / "meshes" / f"r5/{finger}.glb", force="mesh")
        hull = trimesh.PointCloud(tip.vertices[tip.vertices[:, 0] > -0.004]).convex_hull
        hull.export(dest / "obj" / f"r5_{finger}_finger.obj")
        text = re.sub(
            rf'(<link name="{finger}">.*?)<collision[^>]*>.*?</collision>',
            lambda g, f=finger: g.group(1) + (
                f'<collision><geometry><mesh filename="{dest / "obj" / f"r5_{f}_finger.obj"}"/>'
                f'</geometry></collision>'), text, count=1, flags=re.S)
    text = re.sub(r'(<link name="camera">\s*<visual>.*?<geometry>)\s*<mesh [^>]*/>',
                  r'\1<box size="0.0231 0.0421 0.042"/>', text, count=1, flags=re.S)
    text = raise_wrist_camera(text)
    ARX5_URDF.write_text(text, encoding="utf-8")
    return ARX5_URDF


def raise_wrist_camera(text: str) -> str:
    """The D405 moved up the wrist by ``robots.ARX5_CAMERA_RAISE``, bracket
    and all, and turned 25 degrees toward the fingers, not 30
    (``robots.ARX5.camera_from_tcp`` is where the picture is taken from; this
    is the housing the head camera sees) (v9).  Moved up without its bracket,
    the camera had the bracket in front of it, filling half its picture."""
    from ..robots import ARX5_CAMERA_RAISE

    moves = (('<origin xyz="0.045 0 0.021"/>',
              f'<origin xyz="{0.045 - ARX5_CAMERA_RAISE:.4f} 0 0.021"/>'),
             ('<origin rpy="0 0.523 0" xyz="0.0385 0 0.0475"/>',
              '<origin rpy="0 0.4363 0" xyz="0.0385 0 0.0475"/>'))
    for old, new in moves:
        if text.count(old) != 1:
            raise ConfigError("the ARX X5 description's camera mount is not where it was")
        text = text.replace(old, new)
    return text
