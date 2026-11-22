"""Proprioception: the robot's own readings -- where its hands are, how far its fingers are
open, its joints."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import Cost, tool


@dataclass(frozen=True)
class StateParams:
    arm: Optional[str] = param("which arm; null reports every arm", default=None)


@tool(name="robot_state",
      data="arms: [{arm, tcp {xyz_m, rpy_deg}, gripper_opening_m, gripper_closed, holding_something, running (a handle, or null), and joints_deg (base to wrist) where the robot reports them}]",
      family="observe", params=StateParams, reads=("proprioception",),
      source="sensor", cost=Cost("declared", 0.0),
      examples=[{"arm": "right"}, {}],
      description="""Read the robot's proprioception: for each arm the gripper's position and
      orientation (station frame: metres, rpy_deg relative to pointing straight down), how far
      the fingers are open, whether they are closed on something (they stopped at least 2 mm
      apart: the gripper's own reading), its joint angles in degrees from the base to the
      wrist, and the handle of any action still running. It does not say what is held, and it
      reports no forces or torques -- weigh things with the tools the scene provides. Costs no
      simulated time.""")
def robot_state(ctx, p: StateParams) -> ToolResult:
    arms = [p.arm] if p.arm else sorted(ctx.station.arms)
    states = [ctx.arm_state(a) for a in arms]
    text = "; ".join(f"{s['arm']}: at {s['tcp']['xyz_m']} rpy {s['tcp']['rpy_deg']}, "
                     f"{'holding something' if s['holding_something'] else 'empty'}"
                     for s in states)
    return ToolResult(data={"arms": states}, text=text)


#: fingers commanded closed that stopped at least this far apart have stopped
#: on something: what a real gripper's encoders tell (the engines' own rule,
#: ``ruibench.engines.base.HELD_MIN_GAP``; the thinnest thing taken is 4 mm)
HOLD_MIN_GAP_M = 0.002


def holding_something(st) -> bool:
    """Whether the fingers are closed on something, from the arm's readings
    alone: commanded closed, and stopped apart.  An engine that cannot measure
    the fingers (the mock) says what its own bookkeeping says -- a test double."""
    if st.gripper < 0.5:
        return False
    opening = getattr(st, "opening", None)
    if opening is None:
        return st.holding is not None
    return float(opening) >= HOLD_MIN_GAP_M
