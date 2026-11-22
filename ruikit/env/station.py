"""The conventions every tool speaks, in one place.

* **Station frame** -- origin on the table top, under the robot base (one arm)
  or midway between the two bases (two arms); x forward (away from the robot),
  y left, z up; metres.  RUIBench's world differs from it by a translation only.
* **Orientation** -- ``rpy_deg = [roll, pitch, yaw]`` relative to "gripper
  pointing straight down, fingers opening along y".  yaw > 0 turns the hand
  counter-clockwise seen from above; pitch > 0 swings the fingertips forward
  (+x); roll > 0 swings them left (+y).  Applied roll, then pitch, then yaw,
  about the station axes.  [0, 0, 0] is the ordinary top-down grasp.
* **Arms** -- ``right`` on the single-arm station; ``left`` and ``right`` on the
  two-arm one.
* **Cameras** -- single arm: ``front`` (fixed) and ``wrist``; two arms: ``head``
  (fixed), ``wrist_left``, ``wrist_right``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

from ruibench.core.geometry import quat_conj, quat_mul
from ruibench.core.types import GRASP_DOWN, Pose, Quat, Vec3

from ruikit.core.result import ToolError


def _axis_quat(axis: Sequence[float], angle: float) -> Quat:
    s = math.sin(angle / 2.0)
    return (math.cos(angle / 2.0), axis[0] * s, axis[1] * s, axis[2] * s)


def _normalized(q: Quat) -> Quat:
    n = math.sqrt(sum(v * v for v in q)) or 1.0
    q = tuple(v / n for v in q)
    return q if q[0] >= 0 else tuple(-v for v in q)


def quat_matrix(q: Quat):
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def quat_from_rpy_deg(rpy_deg: Optional[Sequence[float]]) -> Quat:
    """Station-convention [roll, pitch, yaw] (degrees) -> world quaternion (wxyz)."""
    if rpy_deg is None:
        return GRASP_DOWN
    roll, pitch, yaw = (math.radians(float(v)) for v in rpy_deg)
    rel = quat_mul(_axis_quat((0, 0, 1), yaw),
                   quat_mul(_axis_quat((0, 1, 0), -pitch), _axis_quat((1, 0, 0), roll)))
    return _normalized(quat_mul(rel, GRASP_DOWN))


#: within this of pitch +-90 degrees (as |cos pitch|: about 0.6 degrees) roll
#: and yaw turn about one axis, and neither alone means anything: the reading
#: puts the turn in yaw, roll 0 (v2.3.4).  The X5's hands start there, and
#: their sag made the reading jump between [0, 90, 0], [180, 90, 180] and
#: [90, 90, -90] -- so the same relative turn from it pointed the hand down,
#: up, or left it as it was
GIMBAL_LOCK = 0.01


def rpy_deg_from_quat(q: Quat) -> Tuple[float, float, float]:
    """World quaternion -> station-convention [roll, pitch, yaw] in degrees."""
    rel = quat_mul(tuple(q), quat_conj(GRASP_DOWN))
    m = quat_matrix(_normalized(rel))
    # rel = Rz(yaw) Ry(b) Rx(roll) with b = -pitch
    b = -math.asin(max(-1.0, min(1.0, m[2][0])))
    if abs(math.cos(b)) > GIMBAL_LOCK:
        roll = math.atan2(m[2][1], m[2][2])
        yaw = math.atan2(m[1][0], m[0][0])
    else:                                    # gimbal lock: put it all in yaw
        roll = 0.0
        yaw = math.atan2(-m[0][1], m[1][1])
    return (round(math.degrees(roll), 3), round(math.degrees(-b), 3),
            round(math.degrees(yaw), 3))


#: what the agent is told about the estimates under ``perception: model``
FROM_PIXELS = (
    "as on a real robot, no tool knows where things really are: everything is measured from "
    "this station's own pictures. Detection is an open-vocabulary detector on the picture: it "
    "can miss a thing, merge two or give the wrong word, and it finds what you name in your own "
    "words ('blue ring', 'red cross') better than everything at once; it reports no part of the "
    "robot. Positions and sizes come from the depth camera: only the side of a thing the camera "
    "sees is measured, so a pose from the fixed camera across the table and one from the wrist "
    "camera above can differ by a centimetre or two (go by the nearer camera), and a thing "
    "lower than a few millimetres is not told from the table; depth has about 1 mm + 0.3 % of the distance of noise a pixel "
    "and no reading along the edges of things; every camera's calibration is off by about 3 mm "
    "and 0.3 degrees (the angle alone moves a point a metre away by 5 mm); looking again from "
    "the same place gives the same answer, so repeating a look removes nothing; what the "
    "pictures show is real, and touching something tells where it is")


#: how high an arm can hold its tool pointing straight down, by distance from its
#: base -- (distance m, height m over the table), measured on Isaac for the ARX X5
#: with robot_move_to, a binary search to 5 mm at each distance; both arms,
#: straight ahead or 35 degrees out, the hand turned 0 or 90 degrees, all alike.
#: The model's envelope (a flat 15 cm) is optimistic far out: a skill hovering
#: 10 cm over a fork on WK4's tray, 47 cm out, stopped 7 cm short.  A robot
#: without a table here has no ceiling
CEILINGS: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "arx5": ((0.22, 0.170), (0.26, 0.175), (0.30, 0.175), (0.34, 0.165), (0.38, 0.150),
             (0.42, 0.125), (0.45, 0.100), (0.47, 0.075)),
}
#: kept below the measured ceiling, for the hand's own sway and the 5 mm search
CEILING_MARGIN_M = 0.01


@dataclass(frozen=True)
class ArmInfo:
    name: str                 # what the agent says: "right"
    arm_id: str               # what the engine says: "robot0/right"
    max_opening: float        # m, fingers fully open
    finger_reach: float = 0.0  # m, how far the fingertips reach below the tool centre point


class Station:
    """Station frame, arm and camera names for one scene."""

    def __init__(self, scene) -> None:
        from ruibench.engines import robots

        self.scene = scene
        robot = scene.robots[0]
        self.model = robots.model_of(robot)
        self.table_height = float(scene.table_height)
        bases = [robots.arm_base(robot, a, scene.table_height) for a in robot.arms]
        self.origin: Vec3 = (sum(b[0] for b in bases) / len(bases),
                             sum(b[1] for b in bases) / len(bases), self.table_height)
        self.arms: Dict[str, ArmInfo] = {
            a: ArmInfo(a, f"{robot.name}/{a}", float(self.model.max_opening),
                       -min(b.bottom for b in self.model.open_hand if b.name == "finger"))
            for a in robot.arms}
        self.two_arms = len(self.arms) > 1
        self.cameras: Dict[str, str] = {}
        for cam in scene.cameras:
            if cam.kind == "wrist":
                side = (cam.attached_to or "").split("/")[-1]
                name = f"wrist_{side}" if self.two_arms else "wrist"
            else:
                name = "head" if self.two_arms else "front"
            self.cameras.setdefault(name, cam.name)
        self._scene_camera_to_name = {v: k for k, v in self.cameras.items()}

    def ceiling(self, arm: str, xy) -> float:
        """How high (station frame, metres over the table) the arm can hold its
        tool pointing straight down over ``xy``, less ``CEILING_MARGIN_M``: lower
        the farther out it reaches (``CEILINGS``); infinite for a robot without a
        measured table.  The robot's own kinematics -- nothing of the scene."""
        from ruibench.engines import robots

        table = CEILINGS.get(self.model.name)
        if not table:
            return math.inf
        robot = self.scene.robots[0]
        base = self.to_station(robots.arm_base(robot, self.arm(arm).name, self.table_height))
        r = math.hypot(xy[0] - base[0], xy[1] - base[1])
        if r <= table[0][0]:
            z = table[0][1]
        else:
            z = table[-1][1] + (r - table[-1][0]) * (table[-1][1] - table[-2][1]) \
                / (table[-1][0] - table[-2][0])
            for (r0, z0), (r1, z1) in zip(table, table[1:]):
                if r0 <= r <= r1:
                    z = z0 + (r - r0) * (z1 - z0) / (r1 - r0)
                    break
        return max(0.0, z - CEILING_MARGIN_M)

    def kinematics(self, arm: str):
        """(the robot model, the arm's base frame in the world, its joints) --
        the robot's own kinematics (``ruibench.engines.kinematics``),
        nothing of the scene; None for a robot without them."""
        from ruibench.engines.kinematics import arm_frame, arm_kinematics

        kin = arm_kinematics(self.model.name)
        if kin is None:
            return None
        return (self.model, arm_frame(self.scene.robots[0], self.arm(arm).name,
                                      self.table_height), kin)

    @staticmethod
    def too_high(ceiling: float) -> str:
        """What plan_ik and the skills say of a point over ``ceiling``."""
        return f"higher than {ceiling:.3f} m above the table this far from the base"

    # -- names ---------------------------------------------------------------
    def arm(self, name: str) -> ArmInfo:
        try:
            return self.arms[name]
        except KeyError:
            raise ToolError("invalid_args", f"no arm {name!r} on this station",
                            hint=f"arms here: {sorted(self.arms)}") from None

    def default_arm(self) -> str:
        return "right" if "right" in self.arms else next(iter(self.arms))

    def camera(self, name: str) -> str:
        try:
            return self.cameras[name]
        except KeyError:
            raise ToolError("invalid_args", f"no camera {name!r} on this station",
                            hint=f"cameras here: {sorted(self.cameras)}") from None

    def camera_name(self, scene_camera: str) -> str:
        return self._scene_camera_to_name.get(scene_camera, scene_camera)

    # -- frames ----------------------------------------------------------------
    def to_world(self, p: Sequence[float]) -> Vec3:
        return (p[0] + self.origin[0], p[1] + self.origin[1], p[2] + self.origin[2])

    def to_station(self, p: Sequence[float]) -> Vec3:
        return (p[0] - self.origin[0], p[1] - self.origin[1], p[2] - self.origin[2])

    def pose_to_world(self, xyz: Sequence[float], rpy_deg: Optional[Sequence[float]]) -> Pose:
        return Pose(self.to_world(xyz), quat_from_rpy_deg(rpy_deg))

    def describe_pose(self, pose: Pose) -> dict:
        return {"xyz_m": [round(v, 4) for v in self.to_station(pose.pos)],
                "rpy_deg": [round(v, 1) for v in rpy_deg_from_quat(pose.quat)]}

    def conventions(self, perception: str = "synthetic") -> dict:
        out = self._conventions()
        if perception == "model":
            out["estimates"] = FROM_PIXELS
        # a robot's own words where the station's do not fit it: a real robot's
        # wrist, cameras and sensors (``RobotModel.conventions``)
        out.update(dict(self.model.conventions))
        return out

    def _conventions(self) -> dict:
        # how far a wrist turns, measured on Isaac with the hand pointing down;
        # without it, agents could not tell whether half a circle was one move,
        # two, or none
        turn = ("each wrist turns about 170 degrees in all: from yaw 0 the left hand about 60 "
                "degrees counter-clockwise and 100 clockwise, the right hand about 90 and 65; a "
                "turn past that stops there (blocked: residual_deg says by how much), and the "
                "fingers close along the same line at the heading half a circle away"
                if self.two_arms else
                "the wrist turns about 345 degrees in all, not a full circle: from yaw 0, "
                "counter-clockwise to 180 and clockwise to between -100 and -150 (least on the "
                "robot's left); a move turns the shorter way and stops where that travel ends "
                "(blocked: residual_deg says by how much) -- the heading may still be reached "
                "the other way round, through a yaw in between")
        return {
            "frame": "station: origin on the table top "
                     + ("midway between the two arm bases" if self.two_arms
                        else "under the robot base")
                     + "; x forward (away from the robot), y left, z up; metres",
            "orientation": "rpy_deg=[roll,pitch,yaw] relative to the gripper pointing straight "
                           "down; yaw>0 turns the hand counter-clockwise seen from above, "
                           "pitch>0 swings the fingertips forward, roll>0 swings them left; "
                           + turn,
            "pixels": "u to the right, v down, in the delivered image of that frame_id",
            "arms": sorted(self.arms),
            "cameras": sorted(self.cameras),
            "gripper_max_opening_m": {a.name: a.max_opening for a in self.arms.values()},
            "gripper": "the tcp (the pose robot_state reports and the moves take) is the point "
                       "between the fingertips, about 1 cm above their ends; at rpy_deg [0, 0, 0] "
                       "the fingers open and close along the station's y axis, and yaw turns that "
                       "axis with the hand; each finger is about 2.6 cm thick along that axis "
                       "(fully open, their outer faces are about 13 cm apart), 2 cm wide across "
                       "it, and reaches from about 1 cm below the tcp to 5 cm above it",
            "wrist_camera": "on a bracket on the hand, above and behind the tcp, tilted toward the "
                            "fingers; it moves with the hand, so ask camera_look for calibration "
                            "to get its exact pose at that moment; open fingertips show in the "
                            "bottom corners of its picture, closed fingers and what they hold at "
                            "the bottom centre",
            "estimates": "as on a real robot, no tool knows where things really are: detection "
                         "boxes, depth, poses and sizes all come from the robot's own estimate of "
                         "the scene, where each thing sits off by its own fixed error -- about "
                         "5 mm in place (sometimes 1-1.5 cm), 3 degrees in heading, 5 % in size "
                         "-- and every camera's calibration is off by about 3 mm and 0.3 degrees (the "
                         "angle alone moves a point a metre away by 5 mm); all told, a position "
                         "read through the fixed camera across the table can be 1-2 cm off, "
                         "through the wrist camera up close a few millimetres, and two cameras "
                         "can disagree by 1-1.5 cm; all the tools "
                         "share these errors, so checking one tool against another, or looking "
                         "again, does not remove them; depth adds about 1 mm + 0.3 % of the "
                         "distance per pixel, and it meets boxes that stand in for things (one "
                         "box for a doll, several for a board with holes or a tray); what the "
                         "pictures show is real, and touching something tells where it is",
            "one_reply": "several tool calls in one reply start in order: a move keeps running "
                         "until it is done while the calls after it answer at once, so a picture "
                         "taken in the same reply shows the world before the move -- look in the "
                         "next reply; an arm and its gripper are one resource, so a gripper "
                         "command and a move of the same arm cannot share a reply; with the "
                         "tool interfaces a reply starts at most one action per arm (move, step, "
                         "point, gripper) and no action after a robot_wait -- two arms may act "
                         "together, and any looking may come along; a later call can read a "
                         "picture taken earlier in the same reply as 'latest.<camera>' (for "
                         "example camera_look, then vision_detect on 'latest.wrist')",
        }
