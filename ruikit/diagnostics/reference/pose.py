"""The reference through the end-effector pose: robot_move_to."""

from __future__ import annotations

from ruibench.interfaces.commands import (
    Cancel,
    SetGripper,
)
from ruikit.diagnostics.reference.base import Expressor, _arm_name, _grip_args
from ruikit.diagnostics.reference.grounding import _targets, _with_notes


class PoseExpressor(Expressor):
    style = "pose"

    def express(self, ep, acts):
        for row in self.rounds(acts):
            calls = []
            for c in row:
                if isinstance(c, Cancel):
                    calls.append(("robot_stop", {"arm": _arm_name(c.arm)} if c.arm else {}))
                elif isinstance(c, SetGripper):
                    calls.append(("robot_gripper", _grip_args(ep, c)))
                else:
                    pose = self.target_of(ep, c)
                    d = ep.station.describe_pose(pose)
                    calls.append(("robot_move_to", {"arm": _arm_name(c.arm), "xyz_m": d["xyz_m"],
                                                    "rpy_deg": d["rpy_deg"]}))
            ep.call_batch(calls)
            if ep.over:
                return


class GroundedPose(_with_notes(PoseExpressor)):
    def on_commands(self, ep, cmds):
        targets = _targets(self, ep, cmds)
        if targets:
            self.grounder.prepare(targets)

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
                    d = ep.station.describe_pose(got.pose)
                    notes.append((len(calls), got.note))
                    calls.append(("robot_move_to", {"arm": _arm_name(c.arm), "xyz_m": d["xyz_m"],
                                                    "rpy_deg": d["rpy_deg"]}))
            for k, text in notes:
                self.note(ep, text, k)
            if calls:
                ep.call_batch(calls)
            if ep.over:
                return
