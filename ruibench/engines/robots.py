"""The robots RUIBench runs on -- what tasks, checks and engines share about them.

Engine-independent.  A robot model says where its arms stand, where the tool
centre point sits on the grasp body, where the wrist camera is, where each arm
starts, how wide the gripper opens, and what space the hand takes up around a
straight-down grasp.  The Isaac backend adds what only it needs -- asset,
joint names, drives -- in ``engines/isaac/robots.py``.

The models the tasks use:

* ``fr3`` -- one Franka Research 3 at the table's back edge: every
  single-arm task.
* ``arx5`` -- the two-arm station as RoboDojo has it: two ARX X5 arms 0.6 m
  apart on the table, a head camera between them: WK4, CL1, DI1, DI5.

Two more are kept for the engine checks and older logs: ``panda`` (one
Franka Panda, the v0 tasks) and ``piper`` (the AgileX Piper station the
two-arm tasks used before 9/22).

Poses follow ``core.types``: positions in metres, quaternions (w, x, y, z).
The *tool frame* has z pointing out of the fingers and y along the fingers'
travel, so ``GRASP_DOWN`` (180 degrees about x) is a straight-down grasp with
the fingers opening along the world's y axis.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

from ..core.errors import ConfigError
from ..core.types import GRASP_DOWN, Pose, Quat, Vec3


@dataclass(frozen=True)
class HandBox:
    """One part of the hand around a straight-down grasp, in the tool frame at
    yaw 0 (x across the fingers' travel, y along it): its footprint, the height
    of its lowest point above the tool centre point (negative: below), and its
    own height."""

    name: str
    x: Tuple[float, float]
    y: Tuple[float, float]
    bottom: float
    height: float
    #: a closed finger: ``y`` is measured outward from the side of whatever is
    #: held, and the box is mirrored to the other side
    from_side: bool = False


@dataclass(frozen=True)
class RobotModel:
    name: str
    #: where each arm's base stands, from the robot's base pose: (x, y) on the
    #: table and the height above it (a pedestal)
    arm_mounts: Mapping[str, Vec3]
    #: half extents of the block each arm base stands on, or None
    pedestal: Optional[Vec3]
    #: the tool frame in the grasp body's frame
    tcp_from_body: Pose
    #: the wrist camera (optical convention: it looks along its +z, image x
    #: right, y down) in the tool frame
    camera_from_tcp: Pose
    #: a box around the wrist camera, in the tool frame (centre, half extents),
    #: so the other arm cannot pass through it; None: no housing
    camera_housing: Optional[Tuple[Vec3, Vec3]]
    #: where each arm's tool centre point starts: (x, y) from its base, z above
    #: the table -- pointing straight down unless ``start_quat`` says otherwise
    start_tcp: Vec3
    #: the gripper's full opening (m)
    max_opening: float
    #: the hand with its fingers open, around a straight-down grasp
    open_hand: Tuple[HandBox, ...]
    #: the hand holding something: fingers closed
    closed_hand: Tuple[HandBox, ...]
    #: the wrist and forearm above the hand, as a column: (half width, height)
    wrist_column: Tuple[float, float]
    #: the fixed camera: eye and target from the commanded arms' centre --
    #: (x, y) from their bases, z above the table.  None: fitted instead
    camera_eye: Optional[Vec3]
    camera_target: Optional[Vec3]
    #: where a straight-down grasp can reach, from an arm's base: forward
    #: (min, max), sideways (max |y|), and height above the table (max)
    reach: Tuple[float, float, float, float]
    #: a fitted fixed camera (``engines.base.fit_camera``): how far it looks
    #: down, and the points it must show around every commanded arm -- (x, y)
    #: from the arm's base, z above the table
    camera_pitch: Optional[float] = None
    camera_frame: Tuple[Vec3, ...] = ()
    #: the fixed camera's lens (mm, on the shared 20.955 mm sensor)
    camera_focal_length: float = 24.0
    #: the wrist cameras' lens (mm, on the same sensor): 24 mm, 47 degrees
    #: across, unless the robot's own camera is known to see wider
    wrist_focal_length: float = 24.0
    #: the bracket holding the wrist camera out from the hand, drawn and solid
    #: like the housing (tool frame: centre, half extents); None: none
    camera_bracket: Optional[Tuple[Vec3, Vec3]] = None
    #: a posture an arm reaching further out goes home by -- (joint, angle)
    #: pairs; the Isaac engine's home path passes through it without stopping
    home_via: Tuple[Tuple[str, float], ...] = ()
    #: an egocentric fixed camera instead: (ahead of the arms' base line, height
    #: above the table), between the arms, looking forward ``camera_pitch`` down
    ego_camera: Optional[Tuple[float, float]] = None
    #: the gripper where it starts -- (x, y) from the arm's base, z above the
    #: table -- which every fixed camera shows, a task's own frame or not
    camera_hand: Tuple[Vec3, ...] = ()
    #: the fixed camera's own picture size (width, height), which a run does
    #: not change: a real head camera's sensor.  None: the run's size
    camera_size: Optional[Tuple[int, int]] = None
    #: how far the middle of the station table is ahead of the arms' base line
    table_ahead: float = 0.40
    #: the fixed camera's apertures (horizontal, vertical) in mm, when its
    #: sensor is not the shared 20.955 mm one with square pixels
    camera_apertures: Optional[Tuple[float, float]] = None
    #: the tool's orientation where it starts; None: straight down
    start_quat: Optional[Quat] = None
    #: what ``task_info`` says of this robot where the station's general words do
    #: not fit it -- a real robot's own wrist, cameras and sensors -- as (key,
    #: text) pairs over ``ruikit.env.station.Station.conventions``; () for none
    conventions: Tuple[Tuple[str, str], ...] = ()

    def hand_bottom(self) -> float:
        return min(b.bottom for b in self.open_hand if b.name == "hand")


# -- Franka Research 3 (v2) --------------------------------------------------
#: The lab's single-arm robot, standing on the table (2026-09-18).  Its hand is
#: the Franka hand the Panda has, measured on Isaac's FR3 USD
#: (research/fr3_hand_geometry.py): the fingertips are 11.23 cm along the
#: hand's z, so the tool centre point -- 8 mm above them, as on the Piper -- is
#: at 10.4 cm; the fingers are 2.6 cm thick and open 8 cm between them.  The
#: hand's envelope is the Panda's, moved to that tool centre point.
#:
#: The wrist camera is a RealSense D405 on a bracket on the hand's robot
#: side: 8.5 cm above the tool centre point and 7.5 cm behind its middle,
#: looking along the fingers tilted 20 degrees toward them -- ahead, away
#: from the robot -- through the X5's D405 lens (the middle square of its
#: 87 x 58 degrees, about 60).  The fingers are in the picture's bottom
#: corners open and at its bottom middle closed, what they hold between them.
#: v17 (9/30, the user: the fingers showed only closed): flat against the
#: hand's far side, 6.5 cm up and 4.5 cm out, turned 8 degrees, behind a
#: 47-degree lens, the open fingers were 4 to 6.6 cm to the side of a picture
#: 3 cm wide at their tips, and the hand's own corner stood between them and
#: the camera.  Out on a bracket on the far side it saw them, but stood
#: between the fixed camera and what the hand held (hidden 60-75% in nine
#: tasks); behind the hand, the hand is in the way of nothing
#: (research/v17_wrist_view.py, v17_mount_screen.py, v17_wrist_render.py).
#: Isaac Lab's visuomotor mount, 25 cm up above the flange, looked down on
#: most of the table from the starting pose and saw what the Active
#: Perception tasks hide (research/fr3_*.py, 2026-09-18).
#:
#: The fixed camera faces the arm from beyond the table, fitted to its work
#: area -- 26 to 82 cm in front of the base (all it reaches), 30 cm to either
#: side; a 5 cm thing 45 cm out is 26 px across -- and
#: to the whole hand where it starts, 30 cm up and 30 cm in front of the base,
#: so the gripper is in the first picture.
_FR3_TCP = 0.104
_FR3_START = (0.30, 0.0, 0.30)
_FR3_FRAME = tuple(
    [(dx, dy, dz) for dx in (0.26, 0.82) for dy in (-0.30, 0.30) for dz in (0.0, 0.08)]
    + [(_FR3_START[0] + dx, dy, _FR3_START[2] + dz)
       for dx in (-0.05, 0.05) for dy in (-0.11, 0.11) for dz in (-0.01, 0.17)])
FR3 = RobotModel(
    name="fr3",
    arm_mounts={"right": (0.0, 0.0, 0.0)},
    pedestal=None,
    tcp_from_body=Pose((0.0, 0.0, _FR3_TCP)),
    camera_from_tcp=Pose((-0.075, 0.0, -0.085), (0.696364, -0.122788, 0.122788, -0.696364)),
    camera_housing=((-0.075, 0.0, -0.085), (0.012, 0.021, 0.021)),
    # from the hand's side (3.2 cm from its middle) to the housing's
    camera_bracket=((-0.0475, 0.0, -0.085), (0.0155, 0.015, 0.003)),
    wrist_focal_length=18.0,
    # where the ready pose (engines/isaac/robots.py) puts the tool centre point
    start_tcp=_FR3_START,
    max_opening=0.08,
    open_hand=(
        HandBox("hand", (-0.032, 0.032), (-0.104, 0.104), 0.038, 0.075),
        HandBox("camera", (-0.087, -0.063), (-0.021, 0.021), 0.064, 0.042),
        HandBox("camera", (-0.063, -0.032), (-0.015, 0.015), 0.082, 0.006),
        HandBox("finger", (-0.010, 0.010), (0.040, 0.066), -0.008, 0.049),
        HandBox("finger", (-0.010, 0.010), (-0.066, -0.040), -0.008, 0.049),
    ),
    # closed on what it holds: a finger on each side of it (the Panda's v0
    # box between them stood where the held thing is)
    closed_hand=(
        HandBox("hand", (-0.032, 0.032), (-0.104, 0.104), 0.038, 0.075),
        HandBox("camera", (-0.087, -0.063), (-0.021, 0.021), 0.064, 0.042),
        HandBox("camera", (-0.063, -0.032), (-0.015, 0.015), 0.082, 0.006),
        HandBox("finger", (-0.010, 0.010), (0.0, 0.026), -0.008, 0.049, from_side=True),
    ),
    wrist_column=(0.045, 0.25),
    camera_eye=None,
    camera_target=None,
    reach=(0.25, 0.85, 0.40, 0.60),
    camera_pitch=math.radians(35),
    camera_frame=_FR3_FRAME,
    # the fingers and the bottom of the hand, 30 cm up: in the first picture
    camera_hand=tuple((_FR3_START[0] + dx, dy, _FR3_START[2] + dz)
                      for dx in (-0.03, 0.03) for dy in (-0.05, 0.05) for dz in (-0.01, 0.07)),
)


# -- ARX X5 station (v4) --------------------------------------------------------
#: RoboDojo's dual-arm platform, copied as it is (9/22: the Pipers' 20 cm
#: pedestals and 66 degree head camera hid much of the bench behind the arms).
#: Two ARX X5 arms stand straight on the table, 0.6 m apart, 10 cm in from its
#: back edge; the head camera is RoboDojo's -- see ``ego_camera``.
#:
#: From the description (``arx5_description``, BSD-3, ARX Robotics; measured on
#: its meshes by research/arx5_fk.py, 2026-09-22):
#:
#: * the tool points along link6's x.  The fingertips are 15.76 cm along it and
#:   the tool centre point 8 mm above them, as on the other hands.  The wrist
#:   camera (a RealSense D405 on ARX's bracket) sits on link6's +z; the tool
#:   frame is turned so that side is the tool's +x, away from the robot at yaw
#:   0 -- the posture the arm reaches forward in, joint 6 near zero.
#: * each finger's inner face travels 0 to 44 mm from the middle: 88 mm open.
#:   A finger tapers from 6 x 2.6 cm at its root to 2 x 0.8 cm at its tip.
#: * the gripper body is 17.4 cm long along the fingers' travel and 6 cm
#:   across, 7 to 10 cm along link6; the wrist above it a 5.7 cm cylinder.
_ARX5_TCP = 0.150
#: a finger in three tiers from its tip up: half width across its travel,
#: thickness out from its inner face, lowest point above the tool centre point,
#: height
_ARX5_FINGER = ((0.011, 0.008, -0.008, 0.021),
                (0.019, 0.015, 0.013, 0.030),
                (0.030, 0.027, 0.043, 0.027))
#: v9 (9/24), the advisor: the wrist camera sat too close to the fingers --
#: DI5's showed a mallet's handle and nothing else.  It is moved 4 cm up the
#: wrist on its bracket (``isaac/robots.raise_wrist_camera``; further, the
#: bracket would run into link5) and turned 25 degrees toward the fingers,
#: not 30: the hand and what it holds low in the picture, the bench ahead of
#: them above
ARX5_CAMERA_RAISE = 0.04
_ARX5_BODY = (
    HandBox("hand", (-0.030, 0.030), (-0.087, 0.087), 0.070, 0.030),
    HandBox("hand", (-0.029, 0.029), (-0.029, 0.029), 0.100, 0.050),
    HandBox("camera", (0.045, 0.090), (-0.021, 0.021), 0.050 + ARX5_CAMERA_RAISE, 0.050),
)
#: at rest the arm is folded as RoboDojo keeps it -- every joint at 0 -- the
#: tool pointing forward 15.6 cm over the table, 24.8 cm ahead of the base
_ARX5_START = (0.248, 0.0, 0.156)
ARX5 = RobotModel(
    name="arx5",
    arm_mounts={"left": (0.0, 0.30, 0.0), "right": (0.0, -0.30, 0.0)},
    pedestal=None,
    # the tool frame in link6: its z along link6's x, its x along link6's z
    tcp_from_body=Pose((_ARX5_TCP, 0.0, 0.0), (0.0, 0.707107, 0.0, 0.707107)),
    # the D405, raised up the wrist (``ARX5_CAMERA_RAISE``), looking along the
    # fingers 25 degrees toward them
    camera_from_tcp=Pose((0.0685, 0.0, -0.0665 - ARX5_CAMERA_RAISE),
                         (0.690391, -0.152839, -0.152839, 0.690391)),
    camera_housing=((0.0685, 0.0, -0.0665 - ARX5_CAMERA_RAISE), (0.012, 0.021, 0.021)),
    # where the ready pose (engines/isaac/robots.py) puts the tool centre point
    start_tcp=_ARX5_START,
    # folded: link6 as the base is, so the tool points forward, its x up
    start_quat=(0.0, 0.707107, 0.0, 0.707107),
    # the fingers down 24 cm in front of the base, 14 cm over the table: the
    # arm as it presses DI1's buttons (``research/v10_x5_home_path.py``)
    home_via=(("joint1", 0.0), ("joint2", 1.449), ("joint3", 1.321), ("joint4", -1.435),
              ("joint5", 0.0), ("joint6", 0.0)),
    max_opening=0.088,
    open_hand=_ARX5_BODY + tuple(
        HandBox("finger", (-hx, hx), (sign * 0.044, sign * (0.044 + t))[::sign], bottom, h)
        for hx, t, bottom, h in _ARX5_FINGER for sign in (1, -1)),
    closed_hand=_ARX5_BODY + tuple(
        HandBox("finger", (-hx, hx), (0.0, t), bottom, h, from_side=True)
        for hx, t, bottom, h in _ARX5_FINGER),
    wrist_column=(0.035, 0.30),
    camera_eye=None,
    camera_target=None,
    #: straight down, 18 to 48 cm ahead and at most 15 cm up: the wrist (joint
    #: 4, +-90 degrees) cannot point the tool down from much higher -- 18 cm at
    #: best, 28 cm ahead, with joint 4 on its limit; at 15 cm it is 0.16 rad off
    reach=(0.18, 0.48, 0.35, 0.15),
    #: RoboDojo's head camera, exactly: between the arms, 4 cm ahead of their
    #: base line and 54.3 cm over the table, **60 degrees down**; an Orbbec
    #: Gemini 345Lg, 10 mm on 22.212 x 14.266 mm (96 x 71 degrees), 640 x 480.
    #: RoboDojo turns the camera 30 degrees about x from the USD camera's own
    #: axis, which looks straight down: 30 from vertical (camera_manager.py).
    #: Read as 30 below the horizon (v4, 9/22) the picture was half wall, the
    #: bench far off and the hands in front of it.
    camera_pitch=math.radians(60),
    camera_focal_length=10.0,
    camera_apertures=(22.212, 14.266),
    #: the wrist's RealSense D405 sees 87 x 58 degrees; its middle square
    #: about 60 (v9: at 47, DI5's showed the mallet's handle and little else)
    wrist_focal_length=18.0,
    ego_camera=(0.04, 0.543),
    camera_size=(640, 480),
    #: RoboDojo's table: 1.1 m deep, the arms 10 cm in from its back edge
    table_ahead=0.45,
    camera_hand=tuple((_ARX5_START[0] + dx, dy, _ARX5_START[2] + dz)
                      for dx in (-0.03, 0.03) for dy in (-0.09, 0.09) for dz in (-0.01, 0.10)),
)

MODELS: Dict[str, RobotModel] = {"fr3": FR3, "arx5": ARX5}


def register_model(model: RobotModel) -> RobotModel:
    """Add a robot defined outside this module to ``MODELS``: the real
    station's (``engines.franka.model``), measured on the lab's robot and kept
    with its engine, out of the frozen simulation."""
    MODELS[model.name.lower()] = model
    return model


def robot_model(name: str) -> RobotModel:
    try:
        return MODELS[str(name).lower()]
    except KeyError:
        raise ConfigError(f"unknown robot model {name!r}; have {sorted(MODELS)}") from None


def model_of(robot) -> RobotModel:
    """The model of a ``RobotSpec``."""
    return robot_model(robot.model)


def arm_base(robot, arm: str, table_height: float) -> Vec3:
    """Where an arm's base is in the world (the top of its pedestal)."""
    model = model_of(robot)
    if arm not in model.arm_mounts:
        raise ConfigError(f"{model.name} has no arm {arm!r}; it has {sorted(model.arm_mounts)}")
    bx, by, bz = robot.base_pose.pos
    mx, my, mz = model.arm_mounts[arm]
    return (bx + mx, by + my, table_height + bz + mz)


def arm_start_pose(scene, robot, arm: str) -> Pose:
    """Where an arm's tool centre point starts -- pointing straight down, or
    as the model's rest pose holds it (the ARX X5 folded, tool forward) --
    unless the scene starts it elsewhere (``SceneSpec.arm_starts``)."""
    start = getattr(scene, "arm_start", None)
    given = start(f"{robot.name}/{arm}") if start else None
    if given is not None:
        return given.pose
    model = model_of(robot)
    x, y, _ = arm_base(robot, arm, scene.table_height)
    sx, sy, sz = model.start_tcp
    return Pose((x + sx, y + sy, scene.table_height + sz), model.start_quat or GRASP_DOWN)


def wrist_camera_pose(tcp: Pose, model: RobotModel = FR3) -> Pose:
    """Where the wrist camera is when the tool centre point is at ``tcp``."""
    from ..core.geometry import compose

    return compose(tcp, model.camera_from_tcp)


def tcp_for_wrist_camera(camera: Pose, model: RobotModel = FR3) -> Pose:
    """The tool pose that puts the wrist camera at ``camera``."""
    from ..core.geometry import compose, inverse

    return compose(camera, inverse(model.camera_from_tcp))


def body_from_tcp(tcp: Pose, model: RobotModel) -> Pose:
    """The grasp body's pose when the tool centre point is at ``tcp``."""
    from ..core.geometry import compose, inverse

    return compose(tcp, inverse(model.tcp_from_body))


def tcp_from_body(body: Pose, model: RobotModel) -> Pose:
    """The tool centre point when the grasp body is at ``body``."""
    from ..core.geometry import compose

    return compose(body, model.tcp_from_body)


def natural_yaw(base: Vec3, target: Vec3, yaw: float) -> float:
    """The one of ``yaw`` and ``yaw + pi`` -- the same grasp for a parallel
    gripper -- closer to the arm's reach direction from ``base`` to ``target``,
    so the wrist stays inside its limits."""
    heading = math.atan2(target[1] - base[1], target[0] - base[0])
    best = None
    for cand in (yaw, yaw + math.pi):
        d = (cand - heading + math.pi) % (2 * math.pi) - math.pi
        if best is None or abs(d) < abs(best[0]):
            best = (d, cand)
    return (best[1] + math.pi) % (2 * math.pi) - math.pi


def within_reach(robot, arm: str, table_height: float, point: Vec3) -> bool:
    """Whether a straight-down grasp at ``point`` is inside the arm's reach
    (the model's coarse box, from the measured workspace)."""
    model = model_of(robot)
    bx, by, _ = arm_base(robot, arm, table_height)
    near, far, side, high = model.reach
    dx, dy = point[0] - bx, point[1] - by
    return near <= dx <= far and abs(dy) <= side and point[2] - table_height <= high


__all__ = [
    "HandBox", "RobotModel", "FR3", "ARX5", "MODELS", "register_model", "robot_model",
    "model_of", "arm_base", "arm_start_pose", "wrist_camera_pose", "tcp_for_wrist_camera",
    "body_from_tcp", "tcp_from_body", "natural_yaw", "within_reach",
]
