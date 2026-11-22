"""The end-effector pose: the gripper sent to an absolute position and orientation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ruibench.core.geometry import rotate
from ruikit.actions.base import ActionAbstraction
from ruikit.actions.common import _arm
from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool


@dataclass(frozen=True)
class MoveToParams:
    xyz_m: Tuple[float, float, float] = param(
        "target gripper position (between the fingertips), station frame, metres")
    arm: str = param("which arm: 'right', or 'left'/'right' on the two-arm station",
                     default="right")
    rpy_deg: Optional[Tuple[float, float, float]] = param(
        "target orientation [roll, pitch, yaw] in degrees relative to pointing straight down; "
        "null is [0, 0, 0] -- straight down at yaw 0, not the hand's present turn", default=None)
    approach_m: Optional[float] = param(
        "if set, first stop this far back along the gripper's axis, then go straight in",
        default=None, minimum=0.0, maximum=0.3)
    wait: bool = param("true: return when the motion ends; false: return a handle at once",
                       default=True)


@tool(name="robot_move_to",
      data=MOTION_DATA,
      family="act", params=MoveToParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"arm": "right", "xyz_m": [0.45, -0.1, 0.12]},
                {"arm": "right", "xyz_m": [0.45, -0.1, 0.03], "rpy_deg": [0, 0, 90],
                 "approach_m": 0.08}],
      description="""Move one arm's gripper to an absolute position (station frame, metres: x
      forward, y left, z up from the table top) and orientation (rpy_deg relative to pointing
      straight down; null is [0, 0, 0], so a hand turned earlier turns back to yaw 0 unless
      you give its yaw again). The shared controller moves at a fixed,
      limited speed and does not plan around obstacles: a move that runs into something stops
      and reports 'blocked' with how far it got. Blocks until the gripper arrives (at most 20 s)
      unless wait=false; the calls after it in the same reply answer without waiting, so look in the next reply. It does not open or close the gripper -- use
      robot_gripper.""")
def robot_move_to(ctx, p: MoveToParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    target = ctx.station.pose_to_world(p.xyz_m, p.rpy_deg)
    poses = [target]
    if p.approach_m:
        axis = rotate(target.quat, (0.0, 0.0, 1.0))
        poses = [target.translated(tuple(-p.approach_m * a for a in axis)), target]
    handle = ctx.move(arm, poses, f"move_to {list(p.xyz_m)}")
    return ctx.motion_result(handle, p.wait)


ABSTRACTION = ActionAbstraction(
    "pose", "位姿", "a target, as coordinates: 'the hand to this position and heading'",
    ("robot_move_to", "robot_gripper"), "Robocurve; Inspect Robots move_to; asimfish's Waypoint")
