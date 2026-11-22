"""The reference through the joints: robot_solve_ik and robot_move_joints, along
the pose controller's straight lines."""

from __future__ import annotations

import math
from typing import List

from ruibench.core.types import Pose
from ruibench.interfaces.commands import (
    Cancel,
    SetGripper,
)
from ruikit.diagnostics.reference.base import Expressor, _arm_name, _grip_args
from ruikit.diagnostics.reference.grounding import _targets, _with_notes
from ruikit.diagnostics.reference.program import _Out


class JointExpressor(Expressor):
    """The joint abstraction, along the pose controller's straight lines: a move
    whose joint-space path would take the hand more than ``TOL_M`` off the
    straight line is cut in half, and the halves again; every waypoint's joints
    are asked of the robot's own inverse kinematics in one reply, then taken
    one waypoint a reply (the arms of a round together) -- the nearest joints
    where a pose is out of reach, as an agent would try them.  Over a program,
    a row's solves and moves are one program, the joints passed on inside it,
    as the pose program puts a whole pick into one.  A joint move between two
    far poses swings the hand along a curve, which knocked WK3's dominoes over
    on Isaac; cut so, it goes where the pose interface goes."""
    style = "joint"
    TOL_M = 0.005
    MAX_DEPTH = 6

    def _steps(self, ep, c, target: Pose) -> List[Pose]:
        from ruibench.engines.kinematics import forward, inverse

        model, base, kin = ep.station.kinematics(_arm_name(c.arm))
        now = ep.engine.state().arm(c.arm).eef_pose
        q_now = ep.engine.arm_joints(c.arm) or kin.ready
        out: List[Pose] = []

        def go(a: Pose, qa, b: Pose, depth: int):
            qb = inverse(model, base, b, seed=qa).joints
            if depth < self.MAX_DEPTH and a.distance_to(b) > 2 * self.TOL_M:
                off = max(_off_line(forward(model, base, [x + (y - x) * t for x, y in zip(qa, qb)]).pos,
                                    a.pos, b.pos) for t in (0.25, 0.5, 0.75))
                if off > self.TOL_M:
                    mid = Pose(tuple((x + y) / 2 for x, y in zip(a.pos, b.pos)),
                               _slerp(a.quat, b.quat, 0.5))
                    qm = go(a, qa, mid, depth + 1)
                    return go(mid, qm, b, depth + 1)
            out.append(b)
            return qb

        go(now, q_now, target, 0)
        return out

    def targets(self, ep, row) -> dict:
        """Each move's target pose, by the command (the grounded expressor's estimate)."""
        return {id(c): self.target_of(ep, c) for c in row
                if not isinstance(c, (Cancel, SetGripper))}

    def express(self, ep, acts):
        for row in self.rounds(acts):
            goals = self.targets(ep, row)
            steps = {k: self._steps(ep, c, goals[id(c)]) for k, c in enumerate(row)
                     if id(c) in goals}
            asks = []
            for k, poses in steps.items():
                for pose in poses:
                    d = ep.station.describe_pose(pose)
                    asks.append(("robot_solve_ik", {"arm": _arm_name(row[k].arm),
                                                    "xyz_m": d["xyz_m"], "rpy_deg": d["rpy_deg"]}))
            program = hasattr(ep, "call_rounds")
            if program:
                # a program solves and moves in one run, the joints passed on inside it
                ref = iter(_Out(j, "joints_deg") for j in range(len(asks)))
                joints = {k: [next(ref) for _ in poses] for k, poses in steps.items()}
            else:
                solved = iter(ep.call_batch(asks) if asks else [])
                if ep.over:
                    return
                joints = {k: [(next(solved).data or {}).get("joints_deg") for _ in poses]
                          for k, poses in steps.items()}
            rounds = [[]]
            for k, c in enumerate(row):
                if isinstance(c, Cancel):
                    rounds[0].append(("robot_stop", {"arm": _arm_name(c.arm)} if c.arm else {}))
                elif isinstance(c, SetGripper):
                    rounds[0].append(("robot_gripper", _grip_args(ep, c)))
            for i in range(max((len(v) for v in joints.values()), default=0)):
                if i >= len(rounds):
                    rounds.append([])
                for k, js in joints.items():
                    if i < len(js) and js[i] is not None:
                        rounds[i].append(("robot_move_joints", {"arm": _arm_name(row[k].arm),
                                                                "joints_deg": js[i]}))
            rounds = [r for r in rounds if r]
            if program:                                     # one program: all of it at once
                ep.call_rounds([asks] + rounds if asks else rounds)
            else:
                for calls in rounds:
                    ep.call_batch(calls)
                    if ep.over:
                        return
            if ep.over:
                return


def _off_line(p, a, b) -> float:
    """How far point ``p`` is from the segment ``a``-``b``."""
    d = [y - x for x, y in zip(a, b)]
    dd = sum(v * v for v in d)
    t = 0.0 if dd == 0 else max(0.0, min(1.0, sum((x - y) * v for x, y, v in zip(p, a, d)) / dd))
    return math.dist(p, [x + t * v for x, v in zip(a, d)])


def _slerp(a, b, t: float):
    """The orientation ``t`` of the way from quaternion ``a`` to ``b``."""
    dot = sum(x * y for x, y in zip(a, b))
    if dot < 0:
        b, dot = tuple(-x for x in b), -dot
    if dot > 0.9995:
        q = tuple(x + (y - x) * t for x, y in zip(a, b))
    else:
        th = math.acos(min(1.0, dot))
        sa, sb = math.sin((1 - t) * th) / math.sin(th), math.sin(t * th) / math.sin(th)
        q = tuple(sa * x + sb * y for x, y in zip(a, b))
    n = math.sqrt(sum(x * x for x in q))
    return tuple(x / n for x in q)


class GroundedJoint(_with_notes(JointExpressor)):
    """The joint abstraction on estimates: looks, finds and estimates as the
    pose interface (``rui.pose/v1``) does, then goes to each estimated target
    the joint reference's way -- the straight line in steps, each step's joints
    from the inverse kinematics."""

    def on_commands(self, ep, cmds):
        targets = _targets(self, ep, cmds)
        if targets:
            self.grounder.prepare(targets)

    def targets(self, ep, row):
        out = {}
        for k, c in enumerate(row):
            if isinstance(c, (Cancel, SetGripper)):
                continue
            got = self.grounder.ground(self.target_of(ep, c))
            if got is None:
                self.unexpressed.append("nothing estimated to place a move by")
                continue
            self.note(ep, got.note, k, got.pose)
            out[id(c)] = got.pose
        return out
