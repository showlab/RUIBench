"""The continuous delta: the gripper moved by a translation and a rotation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

from ruibench.core.geometry import rotate
from ruibench.core.types import Pose
from ruikit.actions.base import ActionAbstraction
from ruikit.actions.common import _arm
from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool
from ruikit.env.station import quat_from_rpy_deg, rpy_deg_from_quat


@dataclass(frozen=True)
class MoveDeltaParams:
    arm: str = param("which arm", default="right")
    dxyz_m: Tuple[float, float, float] = param("translation [dx, dy, dz] in metres",
                                               default=(0.0, 0.0, 0.0))
    drpy_deg: Tuple[float, float, float] = param(
        "rotation [droll, dpitch, dyaw] in degrees, added to the current rpy_deg",
        default=(0.0, 0.0, 0.0))
    frame: Literal["station", "tool"] = param(
        "'station': dxyz along the station axes; 'tool': along the gripper's own axes "
        "(x, y across the hand, z toward the fingertips)", default="station")
    wait: bool = param("true: return when the motion ends; false: return a handle",
                       default=True)


@tool(name="robot_move_delta",
      data=MOTION_DATA,
      family="act", params=MoveDeltaParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"arm": "right", "dxyz_m": [0, 0, -0.05]},
                {"arm": "right", "drpy_deg": [0, 0, 45]}],
      description="""Move one arm's gripper relative to where it is now: a translation in
      metres (along the station axes, or along the gripper's own axes with frame='tool') and a
      rotation in degrees added to its current rpy_deg. While the hand is still settling onto
      the previous move's target (it trails by a millimetre or two for about half a second),
      the move is taken from that target, so repeated moves add up exactly. The gripper travels in a straight line
      at a fixed speed with no obstacle avoidance, 'blocked' if it runs into something. Blocks
      until done (at most 20 s) unless wait=false. It does not open or close the gripper.""")
def robot_move_delta(ctx, p: MoveDeltaParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    ctx.check_free(arm)
    now = ctx.base_pose(arm)
    d = p.dxyz_m
    if p.frame == "tool":
        d = rotate(now.quat, d)
    if any(p.drpy_deg):
        rpy = rpy_deg_from_quat(now.quat)
        quat = quat_from_rpy_deg(tuple(a + b for a, b in zip(rpy, p.drpy_deg)))
    else:
        quat = now.quat                     # no turn: the hand as it is, not as read
    target = Pose(tuple(a + b for a, b in zip(now.pos, d)), quat)
    handle = ctx.move(arm, [target], f"move_delta {list(p.dxyz_m)} {list(p.drpy_deg)}")
    return ctx.motion_result(handle, p.wait)


ABSTRACTION = ActionAbstraction(
    "delta", "增量", "a small move, in numbers: 'along x by 2 cm'",
    ("robot_move_delta", "robot_gripper"), "Embody's direct control; asimfish's Delta")
