"""A physics audit of a trajectory, before it is scored (v15).

Success is read off the physical trajectory, so a physics fault -- a body shot
out of an overlap, a step that blew up, a joint pushed past its stop -- can
pass for a success, or hand one to an agent that found the fault (EmbodiedSWE,
2026-09: coding agents on Isaac Lab scored by forcing and writing state).  The
audit flags what no honest trajectory does; ``Scorer.score`` counts a flagged
episode as a failure until someone has looked at it.

What it looks for, snapshot to snapshot:

* ``nan``: a pose, a velocity, a hand or a joint that is not a number;
* ``joint``: a task joint past its stop by more than ``JOINT_SLACK``;
* ``too_fast``: a free body covering more than ``MAX_SPEED`` -- the arms move
  at well under 1 m/s, and a toy knocked off the table lands at 4 m/s;
* ``jumped``: a body at rest before and after that moved ``JUMP`` or more in
  between with no hand near it -- a teleport, as physics never does.

The last two need dynamics (``meta["dynamics"]``): the mock engine, which has
none, puts a released thing straight down on what is under it.  Interpenetration
is not tested on boxes -- a ring round a peg and an egg under a cup overlap
their neighbours' boxes by design; what it does to a scene, an ejection, is
caught as ``too_fast``, and the objects' depenetration speed is capped on
Isaac (``engines/isaac/scene_builder``).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Mapping, Sequence, Tuple

#: m/s: faster than this a free body is not carried, pushed or dropped
MAX_SPEED = 6.0
#: m or rad past a joint's stop
JOINT_SLACK = 0.02
#: a body at rest (slower than ``AT_REST`` at both ends of a snapshot) that
#: moved this far in it, with no hand within ``HAND_NEAR``, jumped
JUMP, AT_REST, HAND_NEAR = 0.03, 0.05, 0.12
#: at most this many flags of one kind are listed
MAX_LISTED = 5


def _finite(values: Iterable[float]) -> bool:
    return all(math.isfinite(float(v)) for v in values)


def audit(traj: Sequence, joint_limits: Mapping[str, Tuple[float, float]] = None,
          fixed: Iterable[str] = (), dynamics: bool = True) -> List[str]:
    """The flags a trajectory earns (empty: nothing wrong)."""
    joint_limits = dict(joint_limits or {})
    fixed = set(fixed)
    flags: dict = {}

    def flag(kind: str, what: str, t: float) -> None:
        listed = flags.setdefault(kind, [])
        if len(listed) < MAX_LISTED and not any(w == what for w, _ in listed):
            listed.append((what, round(t, 2)))

    prev = None
    for w in traj:
        t = w.sim_time
        for name, o in w.objects.items():
            if not (_finite(o.pose.pos) and _finite(o.pose.quat) and _finite(o.lin_vel)):
                flag("nan", name, t)
        for arm in w.arm_ids:
            a = w.arm(arm)
            if not (_finite(a.eef_pose.pos) and _finite(a.eef_pose.quat) and math.isfinite(a.gripper)):
                flag("nan", arm, t)
        for joint, value in w.articulations.items():
            if not math.isfinite(float(value)):
                flag("nan", joint, t)
                continue
            lo, hi = joint_limits.get(joint, (-math.inf, math.inf))
            if value < lo - JOINT_SLACK or value > hi + JOINT_SLACK:
                flag("joint", joint, t)
        if dynamics and prev is not None and t > prev.sim_time:
            dt = t - prev.sim_time
            hands = [w.arm(a).eef_pose.pos for a in w.arm_ids] + \
                    [prev.arm(a).eef_pose.pos for a in prev.arm_ids]
            for name, o in w.objects.items():
                before = prev.objects.get(name)
                if before is None or name in fixed or not (_finite(o.pose.pos) and _finite(before.pose.pos)):
                    continue
                attrs = o.attrs or {}
                if attrs.get("carried_by") or attrs.get("attached_to") or attrs.get("collides") is False:
                    continue
                step = math.dist(o.pose.pos, before.pose.pos)
                if step / dt > MAX_SPEED:
                    flag("too_fast", name, t)
                elif (step >= JUMP and not o.is_held and not before.is_held
                      and o.speed < AT_REST and before.speed < AT_REST
                      and all(math.dist(h, p) > HAND_NEAR for h in hands
                              for p in (o.pose.pos, before.pose.pos))):
                    flag("jumped", name, t)
        prev = w
    return [f"{kind}:{what}@{at}" for kind in sorted(flags) for what, at in flags[kind]]
