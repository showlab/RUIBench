"""Planning: where the hand can reach, from the robot's own model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component


@dataclass(frozen=True)
class ReachParams:
    xyz_m: Tuple[float, float, float] = param("a gripper position, station frame, metres")
    arm: str = param("which arm", default="right")


@tool(name="plan_ik",
      data="reachable, limits (what stands in the way, in words), envelope_m {near, far, side, high}, "
           "ceiling_m (how high the hand can be held straight down over that spot; null: only "
           "the envelope's high)",
      family="compute", params=ReachParams, source="controller",
      cost=Cost("declared", 0.01), examples=[{"xyz_m": [0.5, 0.0, 0.1]}],
      description="""Check whether one arm can reach a position with the gripper pointing
      straight down, from the arm's reach envelope (forward and sideways distance from its
      base, and height) and how high it can hold the hand that far out (the ARX X5 reaches
      lower the farther out). Returns reachable true/false and, when not, which limit is
      exceeded. It does not return joint angles (these stations are driven by gripper poses) and does
      not account for obstacles or tilted grips.""")
def plan_ik(ctx, p: ReachParams) -> ToolResult:
    from ruibench.engines import robots

    info = ctx.station.arm(p.arm)
    robot = ctx.station.scene.robots[0]
    world = ctx.station.to_world(p.xyz_m)
    ok = robots.within_reach(robot, info.name, ctx.station.table_height, world)
    near, far, side, high = ctx.station.model.reach
    base = robots.arm_base(robot, info.name, ctx.station.table_height)
    dx, dy, dz = (world[0] - base[0], world[1] - base[1], world[2] - ctx.station.table_height)
    why = []
    if dx < near:
        why.append(f"closer than {near:.2f} m in front of the base")
    if dx > far:
        why.append(f"farther than {far:.2f} m from the base")
    if abs(dy) > side:
        why.append(f"more than {side:.2f} m to the side")
    if dz > high:
        why.append(f"higher than {high:.2f} m above the table")
    ceiling = ctx.station.ceiling(info.name, p.xyz_m[:2])
    if dz > ceiling and dz <= high:
        why.append(ctx.station.too_high(ceiling))
        ok = False
    return ToolResult(data={"reachable": bool(ok), "limits": why,
                            "envelope_m": {"near": near, "far": far, "side": side, "high": high},
                            "ceiling_m": round(ceiling, 3) if math.isfinite(ceiling) else None},
                      text="reachable" if ok else "not reachable: " + "; ".join(why or ["outside"]))


COMPONENT = Component("reach", "可达性", "can the arm get there", ("plan_ik",))
