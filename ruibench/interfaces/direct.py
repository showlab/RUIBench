"""I0 -- Direct.  A harness test double, **not** one of the four interfaces.

It hands the caller privileged world state and accepts the shared command
vocabulary verbatim as JSON.  Two uses, both required by the design:

* **Layer 1 of the three-layer solvability check** (v0 §7): a privileged
  reference controller reading simulator ground truth shows that the scene and
  the success criterion are valid at all, *before* any interface is blamed.
* an end-to-end path for the harness -- clock, events, executor, scorer,
  reports -- while I1-I4 are being written elsewhere.

``is_benchmark_interface`` is False, and the reporting layer refuses to put
``I0`` on a leaderboard.
"""

from __future__ import annotations

import json
from typing import List

from ..core.types import Pose
from .base import (
    Expressivity, Interface, InterfaceContract, ModelTurn, ParsedAction, TurnContext,
)
from .commands import (
    Cancel, Command, Done, MoveDelta, MoveTo, SetGripper, Wait,
)

_CODECS = {
    "move_to": lambda d: MoveTo(d["arm"], Pose(tuple(d["pos"]), tuple(d.get("quat", (1, 0, 0, 0)))),
                                float(d.get("speed", 1.0))),
    "move_delta": lambda d: MoveDelta(d["arm"], tuple(d.get("dpos", (0, 0, 0))),
                                      tuple(d.get("drot", (0, 0, 0))), d.get("frame", "world")),
    "gripper": lambda d: SetGripper(d["arm"], float(d["value"])),
    "wait": lambda d: Wait(float(d["seconds"]), d.get("arm")),
    "cancel": lambda d: Cancel(d.get("arm"), d.get("handle_id")),
    "done": lambda d: Done(d.get("note", "")),
}


def encode(commands: List[Command]) -> str:
    """Serialise commands into the JSON I0 accepts."""
    out = []
    for c in commands:
        if isinstance(c, MoveTo):
            out.append({"op": "move_to", "arm": c.arm, "pos": list(c.pose.pos),
                        "quat": list(c.pose.quat), "speed": c.speed})
        elif isinstance(c, MoveDelta):
            out.append({"op": "move_delta", "arm": c.arm, "dpos": list(c.dpos),
                        "drot": list(c.drot), "frame": c.frame})
        elif isinstance(c, SetGripper):
            out.append({"op": "gripper", "arm": c.arm, "value": c.value})
        elif isinstance(c, Wait):
            out.append({"op": "wait", "seconds": c.seconds, **({"arm": c.arm} if c.arm else {})})
        elif isinstance(c, Cancel):
            out.append({"op": "cancel", "arm": c.arm, "handle_id": c.handle_id})
        elif isinstance(c, Done):
            out.append({"op": "done", "note": c.note})
        else:  # pragma: no cover - defensive
            raise TypeError(f"cannot encode {c!r}")
    return json.dumps(out)


class DirectInterface(Interface):
    id = "I0_direct"
    name = "direct (harness test double)"
    source = "RUIBench harness"
    is_benchmark_interface = False

    @property
    def contract(self) -> InterfaceContract:
        return InterfaceContract(
            position=Expressivity.ABSOLUTE, orientation=Expressivity.ABSOLUTE,
            gripper=Expressivity.ABSOLUTE, blocking=False, cancellable=True,
            concurrent_arms=True, camera_control="explicit",
            services=("detect", "segment", "depth_at", "pose_of", "ground_truth"),
            feedback_delivery="inline", reentry="per_step",
            notes="privileged; excluded from every leaderboard",
        )

    def system_prompt(self, ctx: TurnContext) -> str:
        return "Direct command channel: emit a JSON list of RUIBench commands."

    def observe(self, ctx: TurnContext) -> ModelTurn:
        return ModelTurn(
            turn_index=ctx.turn_index,
            prompt=ctx.instruction,
            requested_at=ctx.sim_time,
            observed_at=ctx.observation.sim_time,
        )

    def parse(self, raw: str, ctx: TurnContext) -> ParsedAction:
        try:
            items = json.loads(raw or "[]")
            return ParsedAction(commands=[_CODECS[d["op"]](d) for d in items])
        except Exception as exc:
            return ParsedAction(parse_error=f"{type(exc).__name__}: {exc}")
