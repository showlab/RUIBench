"""The real station's robot: the lab's Franka Emika Panda with the Franka Hand.

Its arm is the FR3's kinematic chain (the two share every link length), held
to the Panda's own joint limits, and its hand is the FR3 model's hand
(``engines.robots.FR3``: the same Franka Hand).  What differs from the
simulated FR3 is what was measured on the lab's robot:

* the joint limits are the Polymetis server's (``franka_hardware.yaml``: the
  Panda's less 0.1 rad) less the 0.2 rad its safety controller pushes back
  from: a target inside them is reached, one outside is not commanded.  Joint 6
  stops short of pi, where the wrist's axes come into line and the robot's
  Cartesian controller swings joints 3, 5 and 7 faster than they may go
  (``route``: it put the robot into a reflex on 2026-11-02);
* the tool centre point is the Franka Hand's own, 10.34 cm along the hand
  from the flange that Polymetis reports (``panda_link8``), the hand turned
  45 degrees on it;
* the cameras are the station's RealSenses, each picture rectified to this
  model's lens (``engines.franka.cameras``): the fixed D435 delivers 640 x 480
  through a 21.5 mm lens on the shared sensor (52 x 40 degrees), as much as
  its picture holds about its principal point; the wrist D405 delivers the
  simulated FR3's square picture behind its 18 mm lens (60 degrees);
* the wrist camera's mount is the calibration's (``calibration.json``); the
  value here is nominal, for the planners that place the wrist camera;
* the arm starts about where the lab's HOME has it, joints mid-range: the
  simulated FR3's start, 30 cm out and 30 cm over this table, has joint 4 at its
  limit and the elbow pushed out to the side;
* ``conventions``: what ``task_info`` says of this robot where the simulated
  station's words do not fit it -- how far the wrist turns, where the wrist
  camera is, what the sensors' errors are.

The model and its kinematics are registered on import, outside the frozen
simulation (``engines.robots.register_model``): they change with the robot's
calibration, which no simulation result depends on.
"""

from __future__ import annotations

import math
from dataclasses import replace

from ...core.types import Pose
from .. import kinematics, robots

#: the Franka Hand's tool centre point along the hand from the flange (Franka's
#: own figure; the simulated FR3's, 10.4 cm, was measured on its USD)
TCP_FROM_FLANGE_M = 0.1034
#: the Franka Hand's body from the tool centre point, in the tool's own frame: x across its
#: thickness, y along the line between the fingertips, z the way the fingers point
HAND_BODY_M = ((-0.035, 0.035), (-0.105, 0.105), (-0.10, -0.045))
#: the hand sits on the flange turned this far about the flange's axis
#: (``panda_hand_joint``): at joint 7 = 45 degrees its fingers open along y
HAND_ON_FLANGE_RAD = -math.pi / 4

#: the lab's HOME (``franka_server/go_home_client.py``)
HOME_JOINTS = (0.0, -0.5058451, 0.0, -2.6068573, 0.0, 2.0711833, 0.8071)

#: the Polymetis server's joint limits (``franka_hardware.yaml``), and the
#: margin its safety controller starts pushing back from (``safety_controller``)
SERVER_LOWER = (-2.80, -1.66, -2.80, -2.97, -2.80, 0.08, -2.80)
SERVER_UPPER = (2.80, 1.66, 2.80, -0.17, 2.80, 3.65, 2.80)
SAFETY_MARGIN_RAD = 0.2
#: the server's limit on each joint's speed (``franka_hardware.yaml``, rad/s)
SERVER_SPEED = (2.075, 2.075, 2.075, 2.075, 2.51, 2.51, 2.51)
#: how far joint 6 stays from pi.  A hand pointing down over the table has joint 6
#: near 3 where the arm is folded most (2.95 at 40 cm out and 3 cm up), and the wrist
#: gets worse slowly up to there: the Jacobian's smallest singular value is 0.20 at
#: 2.8 and 0.12 at 3.0, and it goes on falling past pi, which the first calibration
#: crossed (``route``)
WRIST_CLEAR_RAD = 0.14

_UPPER = tuple(round(v - SAFETY_MARGIN_RAD, 4) for v in SERVER_UPPER)

PANDA_KINEMATICS = replace(
    kinematics.KINEMATICS["fr3"],
    joints=tuple(f"panda_joint{i}" for i in range(1, 8)),
    lower=tuple(round(v + SAFETY_MARGIN_RAD, 4) for v in SERVER_LOWER),
    upper=_UPPER[:5] + (round(math.pi - WRIST_CLEAR_RAD, 4),) + _UPPER[6:])

#: how far a straight line of the hand may take the joints.  The robot's controller
#: chooses them along one, so a line may go half way into the margin the safety
#: controller pushes back from (joint 4's elbow does, going straight down from the
#: start), and no further
LINE_LOWER = tuple(round(v + SAFETY_MARGIN_RAD / 2, 4) for v in SERVER_LOWER)
LINE_UPPER = tuple(max(u, round(v - SAFETY_MARGIN_RAD / 2, 4)) if j != 5 else u
                   for j, (u, v) in enumerate(zip(PANDA_KINEMATICS.upper, SERVER_UPPER)))

#: what task_info says of this robot instead of the simulated station's words
#: (``ruikit.env.station.Station.conventions``).  The wrist's travel is worked
#: out from ``PANDA_KINEMATICS`` over the reach: joint 7 within its limits, and
#: joint 1 turned toward the hand taking its share
CONVENTIONS = (
    ("orientation",
        "rpy_deg=[roll,pitch,yaw] relative to the gripper pointing straight down; yaw>0 turns "
        "the hand counter-clockwise seen from above, pitch>0 swings the fingertips forward, "
        "roll>0 swings them left; the wrist turns about 300 degrees in all, not a full circle: "
        "from yaw 0, counter-clockwise to between 150 and 180 and clockwise to between -65 and "
        "-140 (least on the robot's left); a move turns the shorter way and stops where that "
        "travel ends (blocked: residual_deg says by how much) -- the heading may still be "
        "reached the other way round, through a yaw in between"),
    ("wrist_camera",
     "a RealSense D405 on the hand, about 8 cm back along the tool and 8 cm to its side (away "
     "from the robot's base at yaw 0), looking along it; it moves with the hand, so ask "
     "camera_look for calibration to get its exact pose at that moment; at yaw 0 the top of its "
     "picture is toward the robot's base and its right is the station's +y; an open hand's "
     "fingertips just show in its top corners and the point between them lies just above the "
     "picture, so what the hand holds is seen from the fixed camera"),
    ("estimates",
        "this is a real robot: no tool knows where things really are, everything is measured "
        "from this station's own pictures. Detection is an open-vocabulary detector on the "
        "picture: it can miss a thing, merge two or give the wrong word, and it finds what you "
        "name in your own words ('yellow lemon', 'white plate') better than everything at once; "
        "it reports no part of the robot. Positions and sizes come from the RealSense depth "
        "cameras: only the side of a thing the camera sees is measured, so a pose from the fixed "
        "camera across the table and one from the wrist camera above can differ by a "
        "centimetre or two (go by the nearer camera); a thing lower than a few millimetres is "
        "not told from the table; depth is noisier on dark, shiny and see-through things and "
        "reads nothing along the edges of things; each camera's calibration is the robot's "
        "own, good to a few millimetres; what the pictures show is real, and touching "
        "something tells where it is"),
)

PANDA = replace(
    robots.FR3,
    name="panda",
    tcp_from_body=Pose((0.0, 0.0, TCP_FROM_FLANGE_M)),
    start_tcp=(0.38, 0.0, 0.29),
    # nominal: as the calibration's first run's pictures put it (2026-11-02), about
    # 9 cm back along the tool and 8 cm off it, looking along it, the picture's top
    # toward the robot at yaw 0 -- the calibration has the mount
    camera_from_tcp=Pose((0.077, 0.005, -0.089), (0.6915, 0.0510, -0.0680, -0.7170)),
    camera_focal_length=21.5,
    camera_apertures=(20.955, 15.716),
    camera_size=(640, 480),
    wrist_focal_length=18.0,
    conventions=CONVENTIONS,
)

robots.register_model(PANDA)
kinematics.register_kinematics(PANDA.name, PANDA_KINEMATICS)

__all__ = ["HOME_JOINTS", "LINE_LOWER", "LINE_UPPER", "PANDA", "PANDA_KINEMATICS", "TCP_FROM_FLANGE_M",
           "HAND_ON_FLANGE_RAD", "SERVER_LOWER", "SERVER_UPPER", "SERVER_SPEED", "SAFETY_MARGIN_RAD", "WRIST_CLEAR_RAD"]
