"""The discrete step: named moves of a few fixed sizes.

The vocabulary is Show-Harness's -- a move along an axis, a turn, grasp,
release -- spoken as an enum argument rather than JSON text, with two
additions without which some tasks would be impossible for this abstraction
rather than hard: tilt steps (no other step tips the hand, and the two-arm
station's rest pose points the fingers forward) and a micro step (a ring onto
its peg, a domino on its mark: 1 cm steps fall short or overshoot).  The same
step may repeat up to five times in one reply, so a long straight move does
not cost a reply every 4 cm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from ruibench.core.types import Pose
from ruikit.actions.base import ActionAbstraction
from ruikit.actions.common import _arm
from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import MOTION_DATA, Cost, tool
from ruikit.env.station import quat_from_rpy_deg, rpy_deg_from_quat

#: the semantic-step vocabulary and its default sizes (Show-Harness: 2 / 4 cm,
#: 0.15 rad); an interface may narrow the vocabulary and change the sizes
STEP_DIRECTIONS = {"forward": (1, 0, 0), "back": (-1, 0, 0), "left": (0, 1, 0),
                   "right": (0, -1, 0), "up": (0, 0, 1), "down": (0, 0, -1)}


STEP_TURNS = {"rotate_ccw": (0, 0, 1), "rotate_cw": (0, 0, -1),
              "tilt_forward": (0, 1, 0), "tilt_back": (0, -1, 0),
              "tilt_left": (1, 0, 0), "tilt_right": (-1, 0, 0)}


STEP_DEFAULTS = {"micro_m": 0.005, "fine_m": 0.02, "coarse_m": 0.04,
                 "micro_deg": 3.0, "fine_deg": 8.6, "coarse_deg": 17.2}


StepMove = Literal["forward", "back", "left", "right", "up", "down", "rotate_ccw", "rotate_cw",
                   "tilt_forward", "tilt_back", "tilt_left", "tilt_right", "grasp", "release"]


@dataclass(frozen=True)
class StepParams:
    move: StepMove = param(
        "one step: forward/back/left/right/up/down move the gripper along the station axes; "
        "rotate_ccw/rotate_cw turn it about the vertical; tilt_* tip the fingertips; "
        "grasp/release close/open the fingers")
    arm: str = param("which arm", default="right")
    size: Literal["micro", "fine", "coarse"] = param(
        "step size: micro for the last millimetres, fine, or coarse", default="fine")
    n: int = param("take the same step this many times in a row (1-5), without stopping to "
                   "look in between", default=1, minimum=1, maximum=5)


@tool(name="robot_step",
      data=MOTION_DATA + "; and step_sizes {micro_m, fine_m, coarse_m, micro_deg, fine_deg, coarse_deg}; "
           "and pushed {frames, note}: the cameras' pictures after the step, which come with it",
      family="act", params=StepParams, controls="arm",
      source="controller", returns="result", cost=Cost("physical"),
      examples=[{"move": "down", "size": "coarse"}, {"move": "grasp"},
                {"move": "left", "arm": "left", "size": "fine"}],
      description="""Move one arm by one discrete step, Show-Harness style. Directions are the
      station axes (forward = away from the robot, left = +y, up = +z). Step sizes are fixed by
      the configuration and reported in the result (by default 0.5 / 2 / 4 cm for micro / fine /
      coarse, and 3 / 8.6 / 17.2 degrees for rotate and tilt); grasp closes and release opens
      the fingers; n repeats the same step up to 5 times in a row. The reply comes back when
      its step is done and, on the step interface, brings what the cameras see then (other
      calls in the same reply ran before the step finished). To move both arms together,
      step each arm in the same reply. Steps add up exactly: while the hand is still settling onto the last step (it
      trails by a millimetre or two for about half a second) the next one starts from that
      step's target. There is no absolute target: look again after a few steps to see where
      you are.""")
def robot_step(ctx, p: StepParams) -> ToolResult:
    arm = _arm(ctx, p.arm)
    allowed = ctx.params.get("moves")
    if allowed and p.move not in allowed:
        raise ToolError("invalid_args", f"step {p.move!r} is not part of this vocabulary",
                        hint=f"steps here: {', '.join(allowed)}")
    sizes = {**STEP_DEFAULTS, **{k: v for k, v in ctx.params.items() if k in STEP_DEFAULTS}}
    if p.move in ("grasp", "release"):
        if p.n != 1:
            raise ToolError("invalid_args", f"{p.move} cannot be repeated", hint="use n=1")
        handle = ctx.grip(arm, 1.0 if p.move == "grasp" else 0.0, p.move)
        return _with_sizes(ctx.motion_result(handle, True), sizes)
    ctx.check_free(arm)
    now = ctx.base_pose(arm)
    if p.move in STEP_DIRECTIONS:
        k = sizes[f"{p.size}_m"] * p.n
        d = tuple(k * c for c in STEP_DIRECTIONS[p.move])
        target = now.translated(d)
    else:
        deg = sizes[f"{p.size}_deg"] * p.n
        rpy = rpy_deg_from_quat(now.quat)
        turn = tuple(deg * c for c in STEP_TURNS[p.move])
        target = Pose(now.pos, quat_from_rpy_deg(tuple(a + b for a, b in zip(rpy, turn))))
    label = f"step {p.move} ({p.size}{f' x{p.n}' if p.n > 1 else ''})"
    handle = ctx.move(arm, [target], label)
    return _with_sizes(ctx.motion_result(handle, True), sizes)


def _with_sizes(result: ToolResult, sizes: dict) -> ToolResult:
    if hasattr(result, "finalize"):
        inner = result._finalize
        result._finalize = lambda: _with_sizes(inner(), sizes)
        return result
    result.data["step_sizes"] = {k: round(v, 4) for k, v in sizes.items()}
    return result


#: the sizes every interface over the step abstraction uses.  Show-Harness steps
#: 2 / 4 cm and 0.15 rad and shrinks the step near the target with an
#: adaptive-step plugin; here the vocabulary carries a micro step instead: the
#: same reach, one fixed rule for every task
STEP_SIZES = {"micro_m": 0.0025, "fine_m": 0.01, "coarse_m": 0.04,
              "micro_deg": 2.0, "fine_deg": 5.0, "coarse_deg": 15.0}

ABSTRACTION = ActionAbstraction(
    "step", "语义步", "a small move, in words: 'one step left', 'grasp'",
    ("robot_step",), "Show-Harness; RT-H", params={"robot_step": STEP_SIZES},
    hand_back="each_step")
