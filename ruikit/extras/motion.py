"""Ways of moving the arm outside the design space: several waypoints in one call, a chunk
of deltas, back home."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional, Tuple

from ruibench.core.types import Pose
from ruikit.actions.common import _arm
from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool
from ruikit.env.station import quat_from_rpy_deg, rpy_deg_from_quat
from ruikit.harness.components.base import Component


@dataclass(frozen=True)
class Waypoint:
    xyz_m: Tuple[float, float, float] = param("position, station frame, metres")
    rpy_deg: Optional[Tuple[float, float, float]] = param("orientation; null = [0, 0, 0], straight down at yaw 0",
                                                          default=None)
    gripper: Optional[Literal["open", "close"]] = param(
        "open or close the fingers on arrival at this waypoint", default=None)


@dataclass(frozen=True)
class WaypointsParams:
    points: List[Waypoint] = param("waypoints in order (at most 12)", min_items=1, max_items=12)
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the last waypoint is reached; false: a handle",
                       default=True)


@tool(name="robot_waypoints",
      data=MOTION_DATA,
      family="act", params=WaypointsParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"points": [{"xyz_m": [0.4, 0, 0.15]},
                            {"xyz_m": [0.4, 0, 0.03], "gripper": "close"},
                            {"xyz_m": [0.4, 0, 0.15]}]}],
      description="""Send one arm through up to 12 waypoints in order (station frame, metres;
      orientation relative to straight down), opening or closing the fingers on arrival at any
      waypoint. The whole list runs open-loop on the shared fixed-speed controller: it does not
      stop to look, does not avoid obstacles, and aborts at the first blocked leg. Blocks until
      the last waypoint (at most 20 s in total) unless wait=false; the calls after it in the same reply answer without waiting, so look in the next reply.""")
def robot_waypoints(ctx, p: WaypointsParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    poses, grips = [], {}
    for i, wp in enumerate(p.points):
        poses.append(ctx.station.pose_to_world(wp.xyz_m, wp.rpy_deg))
        if wp.gripper:
            grips[i] = 1.0 if wp.gripper == "close" else 0.0
    handle = ctx.move(arm, poses, f"waypoints x{len(poses)}", gripper_after=grips)
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class ChunkParams:
    actions: List[List[float]] = param(
        "T rows of [dx, dy, dz, droll, dpitch, dyaw] (metres and degrees per step, station "
        "axes), optionally with a 7th value: >= 0.5 closes the fingers, < 0.5 opens them",
        min_items=1, max_items=64)
    arm: str = param("which arm", default="right")
    space: Literal["tool_delta", "joint_delta", "joint_abs"] = param(
        "action space; this station supports tool_delta", default="tool_delta")
    hz: float = param("rate the rows are played at", default=10.0, minimum=1.0, maximum=50.0)
    wait: bool = param("true: return when the chunk ends; false: a handle", default=True)


@tool(name="robot_chunk",
      data=MOTION_DATA,
      family="act", params=ChunkParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"actions": [[0.01, 0, 0, 0, 0, 0], [0.01, 0, 0, 0, 0, 0, 1]], "hz": 10}],
      description="""Play an open-loop chunk of small relative moves on one arm, one row per
      tick at hz: each row is [dx, dy, dz, droll, dpitch, dyaw] in metres and degrees along
      the station axes, with an optional 7th gripper value (>= 0.5 closes). This is how a
      VLA's proposed action chunk or a yad-use/asimfish style delta sequence is executed. The
      rows accumulate from where the gripper is when the chunk starts; nothing is re-planned
      while it plays. Joint-space chunks are not supported on these stations.""")
def robot_chunk(ctx, p: ChunkParams) -> ToolResult:
    if p.space != "tool_delta":
        raise ToolError("not_supported", f"{p.space} chunks need joint-space control, which "
                        "this station's controller does not take", hint="use space='tool_delta'")
    arm = _arm(ctx, p.arm)
    ctx.check_free(arm)
    now = ctx.base_pose(arm)
    pos, rpy = list(now.pos), list(rpy_deg_from_quat(now.quat))
    poses, grips = [], {}
    for i, row in enumerate(p.actions):
        if len(row) not in (6, 7):
            raise ToolError("invalid_args", f"actions[{i}] has {len(row)} values; need 6 or 7")
        pos = [a + b for a, b in zip(pos, row[:3])]
        rpy = [a + b for a, b in zip(rpy, row[3:6])]
        poses.append(Pose(tuple(pos), quat_from_rpy_deg(tuple(rpy))))
        if len(row) == 7:
            grips[i] = 1.0 if row[6] >= 0.5 else 0.0
    handle = ctx.move(arm, poses, f"chunk x{len(poses)} @ {p.hz:g} Hz", gripper_after=grips)
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class HomeParams:
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the arm is home", default=True)


@tool(name="robot_home",
      data=MOTION_DATA,
      family="act", params=HomeParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"arm": "right"}],
      description="""Send one arm back to its start pose above the table, pointing straight
      down, with the fingers as they are. Useful to clear the view of the fixed camera or to
      recover from an awkward pose. Same controller and blocking rules as robot_move_to; it
      does not open or close the gripper.""")
def robot_home(ctx, p: HomeParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    start = ctx.home_pose(arm)
    handle = ctx.move(arm, [Pose(start.pos, quat_from_rpy_deg(None))], "home")
    return ctx.motion_result(handle, p.wait)


COMPONENT = Component("motion", "多点运动", "several moves in one call: waypoints, a chunk, back home",
                      ("robot_waypoints", "robot_chunk", "robot_home"))
