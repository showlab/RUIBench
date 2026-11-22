"""What every action abstraction offers besides its moves: the gripper, a wait and a stop."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Literal, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool


def _arm(ctx, arm: Optional[str]) -> str:
    return ctx.station.arm(arm or ctx.station.default_arm()).name


@dataclass(frozen=True)
class GripperParams:
    command: Optional[Literal["open", "close"]] = param(
        "'close' closes the fingers on whatever is between them; 'open' opens fully",
        default=None)
    width_m: Optional[float] = param("instead of command: open the fingers to this width, metres",
                                     default=None, minimum=0.0, maximum=0.1)
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the fingers have settled", default=True)


@tool(name="robot_gripper",
      data="handle, status, tcp {xyz_m, rpy_deg}, gripper_closed, holding_something",
      family="act", params=GripperParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"command": "close"}, {"command": "open", "arm": "left"},
                {"width_m": 0.04}],
      description="""Open or close one arm's fingers: command='close' closes them on whatever
      is between them with a fixed force, command='open' opens fully, or width_m opens them to
      that width. The result says whether the fingers closed on something (not what). Blocks
      until the fingers settle unless wait=false. It does not move the arm.""")
def robot_gripper(ctx, p: GripperParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    if (p.command is None) == (p.width_m is None):
        raise ToolError("invalid_args", "give exactly one of command or width_m")
    if p.command is not None:
        value = 1.0 if p.command == "close" else 0.0
        label = p.command
    else:
        full = ctx.station.arm(arm).max_opening
        value = min(0.49, max(0.0, 1.0 - p.width_m / full))
        label = f"open to {p.width_m:.3f} m"
    handle = ctx.grip(arm, value, label)
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class StopParams:
    arm: Optional[str] = param("stop this arm; null stops every arm", default=None)
    handle: Optional[str] = param("or stop just this action", default=None)


@tool(name="robot_stop",
      data="stopped (the handles stopped), requested, at_rest",
      family="act", params=StopParams, controls="arm",
      source="controller", returns="result", cost=Cost("declared", 0.05),
      examples=[{"arm": "right"}, {"handle": "h7"}, {}],
      description="""Stop an arm (or one action, or everything) now: the running motion halts
      where it is, queued motions are dropped, and nothing stopped ever resumes -- a late
      result from an old command cannot move the arm again. The fingers stay as they are (a
      held object is not dropped). Returns the stopped handles and whether the arm is at
      rest.""")
def robot_stop(ctx, p: StopParams) -> ToolResult:
    arm = _arm(ctx, p.arm) if p.arm else None
    stopped = ctx.stop(arm, p.handle)
    return ToolResult(data={"stopped": stopped, "requested": True, "at_rest": True},
                      text=f"stopped {', '.join(stopped) if stopped else 'nothing (idle)'}")


@dataclass(frozen=True)
class WaitParams:
    seconds: Optional[float] = param("wait this long (simulated seconds)", default=None,
                                     minimum=0.0, maximum=60.0)
    handles: Optional[List[str]] = param("or wait for these action handles; ['running'] waits "
                                         "for every action running now, e.g. a move started "
                                         "with wait=false earlier in this reply", default=None)
    mode: Literal["all", "any"] = param("with handles: wait for all of them or the first",
                                        default="all")
    until_event: bool = param("or wait until the next event (new instruction, message) "
                              "that you have not been given yet", default=False)
    max_seconds: float = param("give up waiting after this long", default=20.0,
                               minimum=0.0, maximum=60.0)


@tool(name="robot_wait",
      data="waited_s; with handles also handles: [{handle, status, label}]; with until_event also new_events and events: [{t, type, text}]",
      family="act", params=WaitParams, controls="none",
      source="controller", returns="result", cost=Cost("physical"),
      examples=[{"seconds": 1.5}, {"handles": ["h3", "h4"], "mode": "all"},
                {"until_event": True, "max_seconds": 10}],
      description="""Let the world run without new commands: for a number of simulated
      seconds, until given action handles finish (all or any), or until the next event arrives
      (a new instruction or a message from the scene), at most max_seconds. The next event is
      one you have not been given yet: an event that came while you were thinking, or while
      your program was doing something else, ends the wait at once, and the wait returns it
      (data.events). The world never pauses for you, so waiting is also how you time an
      action to something that moves. It does not move the robot.""")
def robot_wait(ctx, p: WaitParams) -> ToolResult:
    chosen = sum(x is not None and x is not False for x in (p.seconds, p.handles, p.until_event))
    if chosen != 1:
        raise ToolError("invalid_args", "give exactly one of seconds, handles or until_event")
    t0 = ctx.sim_time
    if p.seconds is not None:
        ctx.advance(p.seconds)
        return ToolResult(data={"waited_s": round(ctx.sim_time - t0, 3)},
                          text=f"waited {p.seconds:g} s")
    if p.handles is not None:
        states = ctx.wait_handles(p.handles, p.mode, p.max_seconds)
        return ToolResult(data={"handles": states, "waited_s": round(ctx.sim_time - t0, 3)},
                          text=", ".join(f"{s['handle']} {s['status']}" for s in states))
    events = ctx.wait_event(p.max_seconds)
    return ToolResult(data={"waited_s": round(ctx.sim_time - t0, 3), "new_events": len(events),
                            "events": events},
                      text=f"{len(events)} new event(s)" if events else
                      f"no event within {p.max_seconds:g} s")
