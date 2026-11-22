"""The reference through the continuous delta: robot_move_delta."""

from __future__ import annotations

from ruibench.core.types import Pose
from ruibench.interfaces.commands import (
    Cancel,
    SetGripper,
)
from ruikit.diagnostics.reference.base import Expressor, _arm_name, _grip_args
from ruikit.diagnostics.reference.grounding import _targets, _with_notes
from ruikit.env.station import rpy_deg_from_quat


class DeltaExpressor(Expressor):
    """The delta abstraction: each target becomes the move from where the hand
    is now -- the translation and the turn to add -- read from the hand's own
    pose, as an agent reads it with robot_state."""
    style = "delta"

    @staticmethod
    def delta_to(ep, arm: str, target: Pose) -> dict:
        from ruikit.env.context import ToolContext

        now = ToolContext(ep, "robot_move_delta").base_pose(arm)
        a, b = rpy_deg_from_quat(now.quat), rpy_deg_from_quat(target.quat)
        turn = [((y - x + 180.0) % 360.0) - 180.0 for x, y in zip(a, b)]
        return {"arm": arm, "dxyz_m": [round(t - n, 5) for t, n in zip(target.pos, now.pos)],
                "drpy_deg": [round(v, 3) for v in turn]}

    def express(self, ep, acts):
        for row in self.rounds(acts):
            calls = []
            for c in row:
                if isinstance(c, Cancel):
                    calls.append(("robot_stop", {"arm": _arm_name(c.arm)} if c.arm else {}))
                elif isinstance(c, SetGripper):
                    calls.append(("robot_gripper", _grip_args(ep, c)))
                else:
                    calls.append(("robot_move_delta",
                                  self.delta_to(ep, _arm_name(c.arm), self.target_of(ep, c))))
            ep.call_batch(calls)
            if ep.over:
                return


class GroundedDelta(_with_notes(DeltaExpressor)):
    """The delta abstraction on estimates: looks, finds and estimates as the
    pose interface (``rui.pose/v1``) does, then moves by the difference between
    the estimate and where the hand reads it is."""

    def on_commands(self, ep, cmds):
        targets = _targets(self, ep, cmds)
        if targets:
            self.grounder.prepare(targets, with_state=True)

    def express(self, ep, acts):
        g = self.grounder
        for row in self.rounds(acts):
            calls, notes = [], []
            for c in row:
                if isinstance(c, Cancel):
                    calls.append(("robot_stop", {"arm": _arm_name(c.arm)} if c.arm else {}))
                elif isinstance(c, SetGripper):
                    calls.append(("robot_gripper", _grip_args(ep, c)))
                else:
                    got = g.ground(self.target_of(ep, c))
                    if got is None:
                        self.unexpressed.append("nothing estimated to place a move by")
                        continue
                    notes.append((len(calls), got.note))
                    calls.append(("robot_move_delta", self.delta_to(ep, _arm_name(c.arm), got.pose)))
            for k, text in notes:
                self.note(ep, text, k)
            if calls:
                ep.call_batch(calls)
            if ep.over:
                return
