"""The semantic skill: a sub-goal handed to a scripted skill.

Skills (``skill_pick`` / ``place`` / ``press`` / ``push`` / ``move`` / ``turn``)
are the SayCan / ROS-MCP / vendor-skill kind: parameterised, closed within
themselves, built only from what the frame shows (a pose estimate, a lifted
pixel) and the shared controller.

A pick comes from above, or from the side (v2.5): the hand level, its fingers
closing up and down on an edge that overhangs -- a thing lying flat that must
end up standing, like CR4's graphics card.  A turn is about the vertical, or a
tip about the level line across the hand, square to the line between the
fingertips, so that what the fingers hold turns with the hand -- a present
shown bottom first to a camera, like DS1's.  Every other skill points the
fingers down, as it always did.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Optional

from ruibench.core.geometry import quat_mul
from ruibench.core.types import Pose
from ruikit.actions.base import ActionAbstraction
from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool
from ruikit.env.station import quat_from_rpy_deg, rpy_deg_from_quat
from ruikit.perception.targets import Target, _resolve

HOVER_M = 0.10
LIFT_M = 0.12
GRIP_BELOW_TOP_M = 0.025
#: air left under the fingertips over what the object stands on (estimated)
FINGERTIP_CLEARANCE_M = 0.002
#: where the arm cannot rise as high as a skill would, it hovers and lifts as
#: high as it can -- but never closer than this over the target's top
MIN_CLEAR_M = 0.02
#: a pick from the side: it comes down level this far short of the target, back
#: along the way the fingers point, and backs out as far once it has lifted; the
#: tool centre point goes this far past the target, so the fingers close on a
#: strip of the edge; and it lifts this high (CR4's own reference: 7 cm, 13 mm, 10 cm)
SIDE_BACK_M = 0.07
SIDE_IN_M = 0.013
SIDE_LIFT_M = 0.10
#: how long the fingers close on an edge before the lift
SIDE_SETTLE_S = 0.5


def _world(ctx, xyz, yaw_deg=0.0) -> Pose:
    return ctx.station.pose_to_world(xyz, (0.0, 0.0, yaw_deg or 0.0))


def _level(ctx, xyz, heading_deg: float) -> Pose:
    """The hand level, its fingers pointing toward ``heading_deg`` and closing up
    and down: the hand pointing down at yaw ``heading_deg + 90``, tipped -90
    (``skill_turn``'s ``tip``)."""
    return ctx.station.pose_to_world(xyz, (-90.0, 0.0, heading_deg + 90.0))


def _tipped(quat, angle_deg: float):
    """``quat`` turned by ``angle_deg`` about the hand's own x axis: the level line
    across the hand, square to the line between the fingertips."""
    half = math.radians(angle_deg) / 2.0
    return quat_mul(quat, (math.cos(half), math.sin(half), 0.0, 0.0))


def _base_xy(ctx, arm: str):
    from ruibench.engines import robots

    station = ctx.station
    base = station.to_station(robots.arm_base(station.scene.robots[0], station.arm(arm).name,
                                              station.table_height))
    return base[0], base[1]


def _reach_along(ext, box_yaw_deg: Optional[float], heading_deg: float) -> float:
    """How far a box of extents ``ext`` (along its own x and y, turned to
    ``box_yaw_deg``) reaches from its middle toward ``heading_deg``."""
    a = math.radians(heading_deg - (box_yaw_deg or 0.0))
    return ext[0] / 2 * abs(math.cos(a)) + ext[1] / 2 * abs(math.sin(a))


def _up(ctx, arm: str, xy, want: float, low: float) -> float:
    """``want`` (station z), but no higher than the arm can hold its hand
    straight down over ``xy`` (``Station.ceiling``) -- and, lowered so, never
    below ``low``.  The ARX X5 reaches lower the farther out: the pick hovering
    10 cm over a fork 47 cm out on WK4 stopped short.  An arm without a
    ceiling keeps ``want``."""
    return max(min(want, ctx.station.ceiling(arm, xy)), min(low, want))


def _not_over(ctx, arm: str, xy, z: float, top: float) -> None:
    """Refuse a goal the agent set the height of (``top`` + its height_m) when
    it is over the arm's ceiling there -- in plan_ik's words -- rather than
    quietly lower it."""
    ceiling = ctx.station.ceiling(arm, xy)
    if z > ceiling:
        room = ceiling - top
        hint = (f"height_m at most {math.floor(room * 1000) / 1000:.3f} here" if room >= 0
                else "the target itself is that high")
        raise ToolError("unreachable", "not reachable: " + ctx.station.too_high(ceiling),
                        hint=hint, data={"ceiling_m": round(ceiling, 3)})


@dataclass(frozen=True)
class PickParams:
    target: Target = param("what to pick: an object id ('f3.front#2', or {frame_id, id}), a pixel {frame_id, u, v}, or a point {xyz_m}")
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the pick is over", default=True)
    yaw_deg: Optional[float] = param(
        "turn the hand to this heading instead of the skill's own choice; the fingers close "
        "across it (at 0 they close along y). From the side: the way the fingers point "
        "(0 = forward +x, 90 = left +y)", default=None)
    approach: Literal["above", "side"] = param(
        "'above': down onto it from above; 'side': level from the side, the fingers closing "
        "up and down on an edge", default="above")


@tool(name="skill_pick",
      data=MOTION_DATA,
      family="delegate", params=PickParams, controls="arm",
      source="controller", reads=("frames", "depth"), returns="result_or_handle",
      cost=Cost("physical"),
      examples=[{"target": {"frame_id": "f3.front", "id": "f3.front#2"}}],
      description="""Pick something up with one arm, as a complete skill: open the fingers,
      approach from above, grip a little below the object's top, close and lift 12 cm (far
      out, where the arm cannot rise that high, as high as it can). The target is a visible
      object id, a pixel, or a point; everything the skill knows comes from that frame's
      estimate, so a stale frame means a stale target. With an object id it
      grips at the middle of the estimate's box, fingers across its narrower side, never so
      low that the fingertips meet what a flat object lies on: right for a box-like thing,
      and for another shape (a cross, a figure with its arms out) wherever that middle and
      that side happen to fall -- give yaw_deg, or a pixel or a point on the part to grip,
      to choose yourself. With a pixel or a point it grips there, 2 cm below it but never
      with the fingertips in the table, at yaw_deg (0 if not given). approach='side' takes
      it from the side instead -- an edge that overhangs, a flat thing that must end up
      standing: the hand comes down level 7 cm short of the target, its fingers pointing
      toward yaw_deg (if not given: from the arm's base toward the target) and closing up
      and down, moves in until its tool centre is 13 mm past the target (an object id: past
      the near side of the estimate's box, at the box's middle height; a pixel or a point:
      at its height), closes, lifts 10 cm and backs out 7 cm. The hand stays level:
      skill_turn with axis 'tip' and +90 turns the fingertips down, and every other skill
      points them down on its way. Level, the hand's body stands across the fingers'
      travel, so low over the table it meets the table. It does not check what it caught;
      the result only says whether the fingers hold something.""")
def skill_pick(ctx, p: PickParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    top, yaw, ext = _resolve(ctx, p.target)
    check = ctx.params.get("check")
    if check and p.wait:                    # +check: the pick, then a look (v2.5.2)
        from ruikit.actions.checks import checked_pick

        return checked_pick(ctx, arm, p, (top, yaw, ext), check)
    return ctx.motion_result(_pick_handle(ctx, arm, p, top, yaw, ext), p.wait)


def _pick_handle(ctx, arm: str, p: PickParams, top, yaw, ext):
    """The pick's motion from an estimate of its target, started."""
    if p.approach == "side":
        return _side_pick(ctx, arm, p, top, yaw, ext)
    if ext is not None:
        # at the estimate's heading the fingers close across its second extent.
        # Where that is too wide for them and the first is narrower: a quarter
        # turn.  Never just because the first is too wide: across the long side
        # of a fork, a plank or a doll lying down, the open fingers come down on
        # the thing itself.
        if ext[0] <= ext[1] and ext[1] > ctx.station.arm(arm).max_opening:
            yaw = (yaw or 0.0) + 90.0
        grip_z = top[2] - min(GRIP_BELOW_TOP_M, ext[2] / 2)
        # the fingertips reach below the tool centre point (8 mm on the FR3):
        # half a 1 cm coin's height put them into the table, and every first
        # grip of DS4's coins on Isaac stopped short
        floor = top[2] - ext[2] + ctx.station.arm(arm).finger_reach + FINGERTIP_CLEARANCE_M
        grip_z = max(grip_z, floor)
    else:
        grip_z = top[2] - 0.02
    # nor ever with the fingertips in the table top (z = 0 of the station frame),
    # whatever the estimate says is under the thing: for a pixel on a doll 1.4 cm
    # thick, 2 cm below it was 6 mm under the table, and the pick stopped short
    # without closing
    grip_z = max(grip_z, ctx.station.arm(arm).finger_reach + FINGERTIP_CLEARANCE_M)
    if p.yaw_deg is not None:
        yaw = p.yaw_deg
    x, y = top[0], top[1]
    low = top[2] + MIN_CLEAR_M
    hover, grip, lift = (_world(ctx, (x, y, _up(ctx, arm, (x, y), top[2] + HOVER_M, low)), yaw),
                         _world(ctx, (x, y, grip_z), yaw),
                         _world(ctx, (x, y, _up(ctx, arm, (x, y), grip_z + LIFT_M, low)), yaw))
    return ctx.sequence(arm, [[("grip", 0.0), ("move", hover), ("move", grip)],
                              [("grip", 1.0)], [("move", lift)]], "pick", lift)


def _side_pick(ctx, arm: str, p: PickParams, top, yaw, ext):
    """A pick from the side, as CR4's reference takes the card: over the start
    pointing down, turned level there, down beside the target, in, closed, up
    and back out -- the hand level from the start to the end."""
    x, y, z = top
    if p.yaw_deg is not None:
        heading = p.yaw_deg
    else:
        bx, by = _base_xy(ctx, arm)
        heading = math.degrees(math.atan2(y - by, x - bx))
    h = math.radians(heading)
    hx, hy = math.cos(h), math.sin(h)
    if ext is not None:                     # the near side of the box, at its middle height
        reach = _reach_along(ext, yaw, heading)
        x, y, z = x - hx * reach, y - hy * reach, z - ext[2] / 2
    at = (x + hx * SIDE_IN_M, y + hy * SIDE_IN_M, z)
    back = (x - hx * SIDE_BACK_M, y - hy * SIDE_BACK_M, z)
    low = z + MIN_CLEAR_M
    high = (back[0], back[1], _up(ctx, arm, back[:2], z + SIDE_LIFT_M, low))
    lifted = (at[0], at[1], _up(ctx, arm, at[:2], z + SIDE_LIFT_M, low))
    out = (lifted[0] - hx * SIDE_BACK_M, lifted[1] - hy * SIDE_BACK_M, lifted[2])
    end = _level(ctx, out, heading)
    return ctx.sequence(arm, [[("grip", 0.0), ("move", _world(ctx, high, heading + 90.0)),
                               ("move", _level(ctx, high, heading)),
                               ("move", _level(ctx, back, heading)),
                               ("move", _level(ctx, at, heading))],
                              [("grip", 1.0)], SIDE_SETTLE_S,
                              [("move", _level(ctx, lifted, heading)), ("move", end)]],
                        "pick from the side", end)


@dataclass(frozen=True)
class PlaceParams:
    target: Target = param("where to put it: an object id ('f3.front#2', or {frame_id, id}), a pixel {frame_id, u, v}, or a point {xyz_m}")
    arm: str = param("which arm", default="right")
    height_m: float = param("release this far above the target's top surface", default=0.03,
                            minimum=0.0, maximum=0.3)
    yaw_deg: Optional[float] = param("turn the hand to this heading before releasing",
                                     default=None)
    wait: bool = param("true: return when the place is over", default=True)


@tool(name="skill_place",
      data=MOTION_DATA,
      family="delegate", params=PlaceParams, controls="arm",
      source="controller", reads=("frames", "depth"), returns="result_or_handle",
      cost=Cost("physical"),
      examples=[{"target": {"frame_id": "f4.front", "id": "f4.front#1"}, "height_m": 0.04}],
      description="""Put down what one arm holds, as a complete skill: move above the target,
      descend until the gripper is height_m above the target's top surface, open the fingers,
      and rise 10 cm (far out, where the arm cannot rise that high, as high as it can). The
      target is a visible object id (placed on its top), a pixel, or a point. It does not know
      how big the held object is -- choose height_m so it clears -- and it does not check
      where the object ends up. A release point higher than the arm reaches there is refused,
      with how high it reaches.""")
def skill_place(ctx, p: PlaceParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    top, yaw, ext = _resolve(ctx, p.target)
    check = ctx.params.get("check")
    if check and p.wait:                    # +check: the place, then a look (v2.5.2)
        from ruikit.actions.checks import TARGET_R_M, checked_place

        r = max(TARGET_R_M, min(ext[0], ext[1]) / 2) if ext is not None else TARGET_R_M
        return checked_place(ctx, arm, p, top, r, check)
    return ctx.motion_result(_place_handle(ctx, arm, p, top), p.wait)


def _place_handle(ctx, arm: str, p: PlaceParams, top):
    """The place's motion over a target's top, started."""
    now = rpy_deg_from_quat(ctx.base_pose(arm).quat)
    yaw = p.yaw_deg if p.yaw_deg is not None else now[2]
    x, y = top[0], top[1]
    z = top[2] + p.height_m
    _not_over(ctx, arm, (x, y), z, top[2])
    above = _world(ctx, (x, y, _up(ctx, arm, (x, y), z + HOVER_M, z + MIN_CLEAR_M)), yaw)
    down = _world(ctx, (x, y, z), yaw)
    return ctx.sequence(arm, [[("move", above), ("move", down)], [("grip", 0.0)],
                              [("move", above)]], "place", above)


@dataclass(frozen=True)
class PressParams:
    target: Target = param("what to press or tap: an object id ('f3.front#2', or {frame_id, id}), a pixel {frame_id, u, v}, or a point {xyz_m}")
    arm: str = param("which arm", default="right")
    depth_m: float = param("how far below the target's top to push", default=0.01,
                           minimum=0.0, maximum=0.05)
    hold_s: float = param("how long to hold it down", default=0.0, minimum=0.0, maximum=10.0)
    wait: bool = param("true: return when the press is over", default=True)


@tool(name="skill_press",
      data=MOTION_DATA,
      family="delegate", params=PressParams, controls="arm",
      source="controller", reads=("frames", "depth"), returns="result_or_handle",
      cost=Cost("physical"),
      examples=[{"target": {"frame_id": "f2.head", "id": "f2.head#4"}, "hold_s": 0.5},
                {"target": {"xyz_m": [0.4, 0.1, 0.05]}, "arm": "left"}],
      description="""Press, tap or strike something from above with one arm: close the
      fingers, come down on the target's top, push depth_m further, hold for hold_s seconds,
      and lift off 8 cm (far out, where the arm cannot rise that high, as high as it can).
      For buttons, toys to tap, xylophone keys and the like. The target is an object id, a
      pixel, or a point. Two arms pressing together: call it for each arm in
      the same reply, or with wait=false and robot_wait.""")
def skill_press(ctx, p: PressParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    top, _, _ = _resolve(ctx, p.target)
    x, y = top[0], top[1]
    above = _world(ctx, (x, y, _up(ctx, arm, (x, y), top[2] + 0.08, top[2] + MIN_CLEAR_M)))
    down = _world(ctx, (x, y, top[2] - p.depth_m))
    steps = [[("grip", 1.0), ("move", above), ("move", down)]]
    if p.hold_s > 0:
        steps.append(p.hold_s)
    steps.append([("move", above)])
    handle = ctx.sequence(arm, steps, "press", above)
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class PushParams:
    target: Target = param("what to push: an object id ('f3.front#2', or {frame_id, id}), a pixel {frame_id, u, v}, or a point {xyz_m}")
    direction_deg: float = param("push direction in the table plane: 0 = forward (+x), "
                                 "90 = left (+y)")
    distance_m: float = param("how far to push", minimum=0.0, maximum=0.3)
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the push is over", default=True)


@tool(name="skill_push",
      data=MOTION_DATA,
      family="delegate", params=PushParams, controls="arm",
      source="controller", reads=("frames", "depth"), returns="result_or_handle",
      cost=Cost("physical"),
      examples=[{"target": {"frame_id": "f3.front", "id": "f3.front#1"}, "direction_deg": 90,
                 "distance_m": 0.1}],
      description="""Push something along the table with one arm's closed fingers: go down
      beside the target on the side opposite the push direction, low over the table, move
      distance_m in that direction, then lift off (10 cm, or as high as the arm can there).
      direction_deg is measured in the table plane from forward (+x) toward left (+y). It
      does not grasp and does not check where the object ends up.""")
def skill_push(ctx, p: PushParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    top, _, ext = _resolve(ctx, p.target)
    rad = math.radians(p.direction_deg)
    dx, dy = math.cos(rad), math.sin(rad)
    back = (ext[0] / 2 if ext else 0.03) + 0.03
    z = max(0.015, top[2] - (ext[2] / 2 if ext else 0.02))
    start = (top[0] - dx * back, top[1] - dy * back, z)
    end = (start[0] + dx * p.distance_m, start[1] + dy * p.distance_m, z)
    yaw = p.direction_deg + 90.0
    low = top[2] + MIN_CLEAR_M
    lifted = _world(ctx, (end[0], end[1], _up(ctx, arm, end[:2], z + 0.1, low)), yaw)
    over = _world(ctx, (start[0], start[1], _up(ctx, arm, start[:2], z + 0.1, low)), yaw)
    handle = ctx.sequence(arm, [[("grip", 1.0), ("move", over),
                                 ("move", _world(ctx, start, yaw)), ("move", _world(ctx, end, yaw)),
                                 ("move", lifted)]], "push", lifted)
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class SkillMoveParams:
    target: Target = param("where to: an object id ('f3.front#2', or {frame_id, id}) -- above its top -- a pixel {frame_id, u, v} or a point {xyz_m}")
    arm: str = param("which arm", default="right")
    height_m: float = param("stop this far above the target's top (a point: above the point)",
                            default=0.0, minimum=0.0, maximum=0.4)
    yaw_deg: Optional[float] = param("the hand's heading on arrival; null keeps the present one",
                                     default=None)
    wait: bool = param("true: return when the move is over", default=True)


@tool(name="skill_move",
      data=MOTION_DATA,
      family="delegate", params=SkillMoveParams, controls="arm",
      source="controller", reads=("frames", "depth"), returns="result_or_handle",
      cost=Cost("physical"),
      examples=[{"target": {"frame_id": "f3.front", "id": "f3.front#2"}, "height_m": 0.08},
                {"target": {"xyz_m": [0.45, -0.05, 0.12]}}],
      description="""Move one arm's hand to a target, as a skill: above a visible object's top
      by height_m, or to a pixel's lifted point or a point (station frame, metres), the
      fingers pointing down at yaw_deg (or as they are; a hand left level by a side pick or a
      tip turns back down on the way). It carries whatever the hand holds
      and neither opens nor closes the fingers; it goes in a straight line at the shared
      controller's speed and stops 'blocked' where something is in the way. A goal higher
      than the arm reaches there is refused, with how high it reaches. For carrying a held
      thing somewhere, or bringing the wrist camera close to look.""")
def skill_move(ctx, p: SkillMoveParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    top, _, _ = _resolve(ctx, p.target)
    yaw = p.yaw_deg if p.yaw_deg is not None else rpy_deg_from_quat(ctx.base_pose(arm).quat)[2]
    _not_over(ctx, arm, top[:2], top[2] + p.height_m, top[2])
    goal = _world(ctx, (top[0], top[1], top[2] + p.height_m), yaw)
    handle = ctx.move(arm, [goal], f"move to {[round(v, 3) for v in top]}")
    return ctx.motion_result(handle, p.wait)


@dataclass(frozen=True)
class SkillTurnParams:
    angle_deg: float = param("how far to turn the hand about the vertical: positive "
                             "counter-clockwise seen from above (axis 'tip': as the "
                             "description says)", minimum=-180.0, maximum=180.0)
    arm: str = param("which arm", default="right")
    wait: bool = param("true: return when the turn is over", default=True)
    axis: Literal["vertical", "tip"] = param(
        "'vertical': about the vertical; 'tip': about the level line across the hand, square "
        "to the line between the fingertips", default="vertical")


@tool(name="skill_turn",
      data=MOTION_DATA,
      family="delegate", params=SkillTurnParams, controls="arm",
      source="controller", returns="result_or_handle", cost=Cost("physical"),
      examples=[{"angle_deg": 90}, {"angle_deg": -45, "arm": "left"},
                {"angle_deg": -90, "axis": "tip"}],
      description="""Turn one arm's hand where it is, about the vertical, by angle_deg --
      screwing and unscrewing, turning a held thing to fit. The fingers stay as they are, so
      a hand closed on something turns it with it. The wrist turns about 345 degrees in all
      (an X5's about 170): a turn past its end stops there, 'blocked', and residual_deg says
      by how much; then let go, turn back, take hold again. axis='tip' tips the hand instead,
      about the level line across it, square to the line between the fingertips, so a held
      thing turns with the hand rather than swinging between the fingers: from pointing down
      at heading yaw, -90 makes the hand level, pointing toward yaw - 90 with its fingers
      closing up and down (as after a side pick pointing there), and +90 turns it back down
      -- a held thing shown to a camera from another side, a flat thing taken from the side
      left hanging from the fingers.""")
def skill_turn(ctx, p: SkillTurnParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    now = ctx.base_pose(arm)
    if p.axis == "tip":
        goal = Pose(now.pos, _tipped(now.quat, p.angle_deg))
        handle = ctx.move(arm, [goal], f"tip {p.angle_deg:+g} deg")
        return ctx.motion_result(handle, p.wait)
    rpy = rpy_deg_from_quat(now.quat)
    goal = Pose(now.pos, quat_from_rpy_deg((rpy[0], rpy[1], rpy[2] + p.angle_deg)))
    handle = ctx.move(arm, [goal], f"turn {p.angle_deg:+g} deg")
    return ctx.motion_result(handle, p.wait)


ABSTRACTION = ActionAbstraction(
    "skill", "技能", "a whole act, ready-made: 'pick it up', 'put it there'",
    ("skill_pick", "skill_place", "skill_press", "skill_push", "skill_move", "skill_turn",
     "robot_gripper"),
    "SayCan; ROS-MCP", kind="act")
