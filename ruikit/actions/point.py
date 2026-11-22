"""The image point: the gripper sent to a pixel of a camera frame.

The pixel is lifted with that frame's depth; how high above it the hand
stops and, when needed, the hand's orientation are two more arguments, the
same for every task -- a point alone cannot carry six degrees of freedom."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from ruibench.core.types import Pose
from ruikit.actions.base import ActionAbstraction
from ruikit.actions.common import _arm
from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool
from ruikit.env.station import quat_from_rpy_deg


@dataclass(frozen=True)
class MoveToPixelParams:
    frame_id: str = param("the frame the pixel is in (from camera_look); or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    u: float = param("pixel column (0 at the left edge)")
    v: float = param("pixel row (0 at the top edge)")
    arm: str = param("which arm", default="right")
    height_m: float = param("mode 'surface': how far above the surface seen at that pixel "
                            "to stop; mode 'plane': the height above the table where the "
                            "pixel's ray is taken (metres)", default=0.0, minimum=-0.05,
                            maximum=0.5)
    mode: Literal["surface", "plane"] = param(
        "'surface': the first surface the pixel's ray meets, then height_m above it; "
        "'plane': the point where the ray crosses height_m above the table (for the middle "
        "of an opening, or a point in the air)", default="surface")
    rpy_deg: Optional[Tuple[float, float, float]] = param(
        "gripper orientation [roll, pitch, yaw] in degrees relative to pointing straight down; "
        "null is [0, 0, 0] -- straight down at yaw 0", default=None)
    wait: bool = param("true: return when the motion ends; false: return a handle",
                       default=True)


@tool(name="robot_move_to_pixel",
      data=MOTION_DATA + "; and lifted_xyz_m, the point the pixel was lifted to",
      family="act", params=MoveToPixelParams, controls="arm",
      source="controller", reads=("frames", "depth"), returns="result_or_handle",
      cost=Cost("physical"),
      examples=[{"frame_id": "f3.front", "u": 250, "v": 310, "height_m": 0.1},
                {"frame_id": "f5.wrist", "u": 256, "v": 256, "height_m": 0.0,
                 "rpy_deg": [0, 0, 45]},
                {"frame_id": "f3.front", "u": 300, "v": 200, "height_m": 0.15, "mode": "plane"}],
      description="""Point at a pixel of a frame and move the gripper there. In mode
      'surface' (default) the pixel is lifted with that frame's depth -- the first surface its
      ray meets, as the camera saw it then -- and the gripper goes height_m above that point
      (0 = touching it). In mode 'plane' the gripper goes to where the pixel's ray crosses
      height_m above the table -- for the middle of a container's opening or a point in the
      air. The gripper points straight down unless rpy_deg says otherwise. Use the fixed camera
      for the whole table and a wrist camera for close work. The gripper travels in a straight
      line at a fixed speed with no obstacle avoidance and the result says 'blocked' if it runs
      into something; blocks until done (at most 20 s) unless wait=false; the calls after it in the same reply answer without waiting, so look in the next reply. To go
      straight up or
      down, point again at the same pixel of the same frame with another height_m: the point
      is the one that frame saw, so it stays put. It does not open or close the gripper.""")
def robot_move_to_pixel(ctx, p: MoveToPixelParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    frame = ctx.frame(p.frame_id)
    if p.mode == "plane":
        point = ctx.perception.ray_at_height(frame, p.u, p.v, p.height_m)
        target = Pose(point, quat_from_rpy_deg(p.rpy_deg))
    else:
        lifted = ctx.perception.lift(frame, p.u, p.v)
        point = ctx.station.to_world(lifted["xyz_m"])
        target = Pose((point[0], point[1], point[2] + p.height_m), quat_from_rpy_deg(p.rpy_deg))
    handle = ctx.move(arm, [target], f"move_to_pixel ({p.u:g}, {p.v:g}) of {frame.id}")
    result = ctx.motion_result(handle, p.wait)
    if not hasattr(result, "finalize"):
        result.data["lifted_xyz_m"] = [round(c, 4) for c in ctx.station.to_station(point)]
    return result


ABSTRACTION = ActionAbstraction(
    "point", "指点", "a target, as a pixel: 'the hand to this point of the picture'",
    ("robot_move_to_pixel", "robot_gripper"), "Gemini Robotics-ER pointing; MOKA / PIVOT")
