"""What a skill or a planner is pointed at: an object id, a pixel or a point, and where
the perception backend puts it."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple
from ruibench.interfaces.commands import MoveTo, SetGripper  # noqa: F401 (session chunks)
from ruikit.core.result import ToolError
from ruikit.core.params import param


@dataclass(frozen=True)
class Target:
    frame_id: Optional[str] = param("frame of the object id or pixel; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply", default=None)
    id: Optional[str] = param("a frame-local object id from vision_detect", default=None)
    u: Optional[float] = param("or a pixel column in that frame", default=None)
    v: Optional[float] = param("and its row", default=None)
    xyz_m: Optional[Tuple[float, float, float]] = param("or a point, station frame, metres",
                                                        default=None)


    @classmethod
    def from_text(cls, text: str) -> dict:
        """A bare object id (``"f3.front#2"``, ``"latest.wrist#1"``) names its
        frame too: programs write ``skill.pick(target="f1.front#1")`` again and
        again, and would otherwise fail on it."""
        frame, sep, _ = text.partition("#")
        if not sep or not frame:
            raise ValueError("a target given as text is an object id such as 'f3.front#2'; "
                             "give a pixel as {frame_id, u, v} and a point as {xyz_m}")
        return {"frame_id": frame, "id": text}


def _resolve(ctx, t: Target):
    """(station xyz of the target's top, yaw_deg or None, extent or None)."""
    if t.xyz_m is not None:
        return tuple(t.xyz_m), None, None
    if not t.frame_id:
        raise ToolError("invalid_args", "give target.frame_id with an id or a pixel, or target.xyz_m")
    frame = ctx.frame(t.frame_id)
    if t.id is not None:
        pose = ctx.perception.pose(frame, t.id)
        x, y, z = pose["xyz_m"]
        ext = pose["extent_m"]
        return (x, y, z + ext[2] / 2), pose["yaw_deg"], ext
    if t.u is not None and t.v is not None:
        lifted = ctx.perception.lift(frame, t.u, t.v)
        return tuple(lifted["xyz_m"]), None, None
    raise ToolError("invalid_args", "target needs an id or a pixel (u, v) with its frame_id")
