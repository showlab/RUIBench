"""The reference through the discrete step: walks of robot_step."""

from __future__ import annotations

import math
from typing import Dict, List

from ruibench.interfaces.commands import (
    Cancel,
    MoveDelta,
    MoveTo,
    SetGripper,
)
from ruikit.diagnostics.reference.base import Expressor, _arm_name
from ruikit.diagnostics.reference.grounding import _targets, _with_notes
from ruikit.env.station import rpy_deg_from_quat


class StepExpressor(Expressor):
    """Walk to each commanded pose in discrete steps, reading back the gripper
    pose the step results report -- the way a careful person would step it:
    up first, then turn the wrist (at the higher of the two heights, so the
    fingers never sweep through what was just put down), then across, and
    down last, never more than a millimetre below the target."""

    style = "step"
    max_steps_per_target = 160
    #: a walk that has not got closer in this many steps stops where it is: at
    #: the edge of an arm's reach one step trades position for tilt (the X5 low
    #: and far out, Isaac), and a person stepping it would stop there too
    stall_steps = 12
    SIZES = ("micro", "fine", "coarse")

    def express(self, ep, acts):
        from ruikit.actions.step import STEP_DEFAULTS

        sizes = {**STEP_DEFAULTS, **ep.profile.tool_params("robot_step")}
        for row in self.rounds(acts):
            targets = {}
            for c in row:
                arm = _arm_name(getattr(c, "arm", "") or "right")
                if isinstance(c, Cancel):
                    ep.call("robot_stop", {"arm": arm} if c.arm else {})
                elif isinstance(c, SetGripper):
                    ep.call_batch([("robot_step", {"arm": arm,
                                                   "move": "grasp" if c.value >= 0.5 else "release"})])
                else:
                    targets[arm] = self.target_of(ep, c)
            for arm, target in targets.items():
                self._walk_starts(ep, arm, target)
            best = {arm: (math.inf, 0) for arm in targets}
            # an arm whose turn steps keep failing goes on by position alone:
            # where the wrist cannot turn (the X5 far out, Isaac), trying again
            # sent the hand diving to the table
            failed_turns: Dict[str, int] = {}
            for _ in range(self.max_steps_per_target):
                if ep.over:
                    return
                moves = {}
                for arm, target in list(targets.items()):
                    move = self._next_step(ep, arm, target, sizes,
                                           turn=failed_turns.get(arm, 0) < 2)
                    if move is None:
                        del targets[arm]
                    else:
                        moves[arm] = move
                downs = [a for a, (m, _) in moves.items() if m == "down"]
                if 0 < len(downs) < len(moves):
                    # a chord, a two-handed press: an arm ready to come down waits
                    # for the other to line up, and then both come down together
                    moves = {a: mv for a, mv in moves.items() if a not in downs}
                elif len(downs) > 1:
                    size = min((sz for _, sz in moves.values()), key=self.SIZES.index)
                    moves = {a: ("down", size) for a in moves}   # in the same steps
                for arm in list(moves):                  # (CL1: two bars within 0.4 s)
                    err = self._error(ep, arm, targets[arm])
                    low, since = best[arm]
                    best[arm] = (err, 0) if err < low - 0.2 else (low, since + 1)
                    if best[arm][1] >= self.stall_steps:
                        self.unexpressed.append(f"step walk stalled {err:.1f} (mm + deg) short "
                                                f"of the target ({arm})")
                        del targets[arm], moves[arm]
                if not moves:
                    if not targets:
                        break
                    continue
                results = ep.call_batch([("robot_step", {"arm": a, "move": m, "size": sz})
                                         for a, (m, sz) in moves.items()])
                for (a, (m, _)), r in zip(moves.items(), results):
                    if m.startswith(("tilt", "rotate")) and (r.is_error or r.data.get("status") == "failed"):
                        failed_turns[a] = failed_turns.get(a, 0) + 1
            if targets:
                self.unexpressed.append(f"step walk did not converge for {sorted(targets)}")

    def _walk_starts(self, ep, arm, target) -> None:
        """A walk toward ``target`` begins (the grounded expressor notes why)."""

    @staticmethod
    def _now(ep, arm):
        return ep.engine.state().arm(ep.station.arm(arm).arm_id).eef_pose

    @classmethod
    def _error(cls, ep, arm, target) -> float:
        """How far from the target: millimetres plus degrees."""
        now = cls._now(ep, arm)
        rn, rt = rpy_deg_from_quat(now.quat), rpy_deg_from_quat(target.quat)
        turn = max(abs(((t - n + 180) % 360) - 180) for n, t in zip(rn, rt))
        return 1000 * math.dist(now.pos, target.pos) + turn

    @classmethod
    def _size(cls, dist, sizes, unit):
        for size in reversed(cls.SIZES):
            if dist >= sizes[f"{size}_{unit}"] * 0.75:
                return size
        return "micro"

    @classmethod
    def _next_step(cls, ep, arm, target, sizes, turn: bool = True):
        now = cls._now(ep, arm)
        rn, rt = rpy_deg_from_quat(now.quat), rpy_deg_from_quat(target.quat)
        drot = [((t - n + 180) % 360) - 180 for n, t in zip(rn, rt)]
        d = [t - n for n, t in zip(now.pos, target.pos)]
        tol = sizes["micro_m"] / 2 + 0.0003          # the nearest a step can get
        if d[2] > tol:                                # 1. up
            return ("up", cls._size(d[2], sizes, "m"))
        # 2. turn -- roll (tilt_left), pitch (tilt_forward), yaw -- where the hand is
        for i, (pos_move, neg_move) in enumerate((("tilt_left", "tilt_right"),
                                                  ("tilt_forward", "tilt_back"),
                                                  ("rotate_ccw", "rotate_cw"))):
            if turn and abs(drot[i]) > sizes["micro_deg"] / 2 + 0.2:
                return (pos_move if drot[i] > 0 else neg_move, cls._size(abs(drot[i]), sizes, "deg"))
        for i in (0, 1):                              # 3. across
            if abs(d[i]) > tol:
                move = (("forward", "back"), ("left", "right"))[i][0 if d[i] > 0 else 1]
                return (move, cls._size(abs(d[i]), sizes, "m"))
        if d[2] < 0:                                  # 4. down, never a millimetre past it
            # (WK2: a coarse step 2.5 mm under the cabbage's grasp counted as a tap)
            for size in reversed(cls.SIZES):
                if sizes[f"{size}_m"] <= -d[2] + 0.001:
                    return ("down", size)
        return None


class GroundedStep(_with_notes(StepExpressor)):
    """Looks, reads where the hand is and estimates once a turn, then steps."""

    def on_commands(self, ep, cmds):
        targets = _targets(self, ep, cmds)
        if targets:
            self.grounder.prepare(targets, with_state=True)

    def express(self, ep, acts):
        g = self.grounder
        grounded = []
        for c in acts:
            if isinstance(c, (MoveTo, MoveDelta)):
                got = g.ground(self.target_of(ep, c))
                if got is None:
                    self.unexpressed.append("nothing estimated to place a move by")
                    continue
                grounded.append(MoveTo(c.arm, got.pose))
                self._pending.append(got.note)
            else:
                grounded.append(c)
        super().express(ep, grounded)

    def run(self, ep):
        self._pending: List[str] = []
        return super().run(ep)

    def _walk_starts(self, ep, arm, target):
        if self._pending:
            self.note(ep, "目标 " + str(ep.station.describe_pose(target)["xyz_m"]) + " "
                      + self._pending.pop(0))
