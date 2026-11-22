"""The joints: angles from the robot's own inverse kinematics, and joint targets."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

from ruikit.actions.base import ActionAbstraction
from ruikit.actions.common import _arm
from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool


@dataclass(frozen=True)
class JointsParams:
    joints_deg: List[float] = param("target angle for each joint, degrees, base to wrist -- "
                                    "robot_state's joints_deg")
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the motion ends; false: a handle", default=True)


_LIMIT_SLACK = math.radians(0.05)


@tool(name="robot_move_joints",
      data=MOTION_DATA,
      family="act", params=JointsParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"joints_deg": [0, -45, 0, -163, 0, 117, 45]},
                {"arm": "left", "joints_deg": [10, 60, 50, -70, 0, 20]}],
      description="""Move one arm's joints to target angles, degrees, base to wrist -- the order
      and the zero of robot_state's joints_deg (the FR3 has 7 joints, the ARX X5 6). Every joint
      goes its own way at a shared joint speed and all arrive together, so the hand's path
      between is not a straight line. An angle outside its joint's limits is refused, with the
      limits. The controller does not plan around obstacles: a move that runs into something
      stops and reports 'blocked' with how far it got. Blocks until the arm arrives unless
      wait=false. It does not open or close the gripper -- use robot_gripper.""")
def robot_move_joints(ctx, p: JointsParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    kinematics = ctx.station.kinematics(arm)
    if kinematics is None:
        raise ToolError("not_supported", "this station's arms take no joint targets")
    _, _, kin = kinematics
    if len(p.joints_deg) != len(kin.joints):
        raise ToolError("invalid_args", f"the {arm} arm has {len(kin.joints)} joints, base to "
                        f"wrist; got {len(p.joints_deg)} angles", hint="robot_state gives them all")
    q = [math.radians(float(v)) for v in p.joints_deg]
    for k, (v, lo, hi) in enumerate(zip(q, kin.lower, kin.upper), start=1):
        # an angle that rounds onto a limit is at it: the limits are given to 0.1
        # degree, and robot_solve_ik's angles at a limit round past it (CL4 on Isaac)
        if not lo - _LIMIT_SLACK <= v <= hi + _LIMIT_SLACK:
            raise ToolError("unreachable", f"joint {k} at {math.degrees(v):.1f} deg is outside its "
                            f"limits [{math.degrees(lo):.1f}, {math.degrees(hi):.1f}] deg",
                            data={"limits_deg": [[round(math.degrees(a), 1), round(math.degrees(b), 1)]
                                                 for a, b in zip(kin.lower, kin.upper)]})
    q = [min(max(v, lo), hi) for v, lo, hi in zip(q, kin.lower, kin.upper)]
    handle = ctx.move_joints(arm, q, f"move_joints {[round(float(v), 1) for v in p.joints_deg]}")
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class SolveIKParams:
    xyz_m: Tuple[float, float, float] = param(
        "gripper position (between the fingertips), station frame, metres")
    rpy_deg: Optional[Tuple[float, float, float]] = param(
        "orientation [roll, pitch, yaw] in degrees relative to pointing straight down; "
        "null is [0, 0, 0]", default=None)
    arm: str = param("which arm", default="right")


@tool(name="robot_solve_ik",
      data="reachable, joints_deg (base to wrist; when not reachable, the nearest it found), "
           "residual_mm and residual_deg (how far those joints leave the gripper), ceiling_m "
           "(how high the hand can be held straight down over that spot; null: no such limit)",
      family="compute", params=SolveIKParams, source="controller",
      cost=Cost("declared", 0.01),
      examples=[{"xyz_m": [0.45, -0.1, 0.05]}, {"xyz_m": [0.4, 0.0, 0.1], "rpy_deg": [0, 0, 90]}],
      description="""Joint angles that put one arm's gripper at a position (station frame,
      metres: x forward, y left, z up from the table top) and orientation (rpy_deg relative to
      pointing straight down): the robot's own inverse kinematics, inside the joint limits,
      starting from where the joints are now, so the answer is the nearest way there. Says
      whether the pose is reachable; when it is not, gives the nearest joints it found and how
      far they leave the gripper. Angles in degrees, base to wrist, as robot_move_joints takes
      them. It does not move the robot and knows nothing of what is on the table.""")
def robot_solve_ik(ctx, p: SolveIKParams) -> ToolResult:
    from ruibench.engines.kinematics import forward, inverse

    info = ctx.station.arm(p.arm)
    kinematics = ctx.station.kinematics(info.name)
    if kinematics is None:
        raise ToolError("not_supported", "this station's arms take no joint targets")
    model, base, kin = kinematics
    goal = ctx.station.pose_to_world(p.xyz_m, p.rpy_deg)
    sol = inverse(model, base, goal, seed=ctx.arm_joints(info.name) or kin.ready)
    got = forward(model, base, sol.joints)
    ceiling = ctx.station.ceiling(info.name, p.xyz_m[:2])
    joints = [round(math.degrees(q), 2) for q in sol.joints]
    miss_mm, miss_deg = 1000 * got.distance_to(goal), math.degrees(got.angle_to(goal))
    return ToolResult(data={"reachable": bool(sol.ok), "joints_deg": joints,
                            "residual_mm": round(miss_mm, 2), "residual_deg": round(miss_deg, 2),
                            "ceiling_m": round(ceiling, 3) if math.isfinite(ceiling) else None},
                      text=(f"reachable: joints {joints} deg" if sol.ok else
                            f"not reachable: the nearest joints leave the gripper "
                            f"{miss_mm:.0f} mm and {miss_deg:.0f} deg off"))


ABSTRACTION = ActionAbstraction(
    "joint", "关节", "the arm's joints, as angles: 'joint 2 to -40 degrees' (with the "
    "robot's own inverse kinematics to find them)",
    ("robot_solve_ik", "robot_move_joints", "robot_gripper"),
    "joint targets, as Inspect Robots and yad-use offer them")
