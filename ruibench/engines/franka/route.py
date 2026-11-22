"""What a straight line of the hand asks of the joints, on the robot's own terms.

Polymetis turns each Cartesian setpoint into joints from where the joints are
(``RobotInterface.update_desired_ee_pose``, ``inverse_kinematics`` in its
``pinocchio_wrapper.cpp``): damped least squares with no limits, no restarts and
nothing to keep the arm's redundancy in hand.  So where the joints go along a
line is decided by where they start, and only following the line finds it out.
Asking a solver for each point alone does not: it finds *some* posture inside
the limits for every point while the robot, on its own way, runs into a limit or
near the wrist's singularity, where joints 3, 5 and 7 swing fast for a small
motion of the hand.  (2026-11-02, the calibration's first run: such a check let
views through that asked joint 3 for 2.9 rad/s, against the robot's 2.075, and
the robot stopped in a reflex.)

``along`` follows a line as the robot would, and says

* how far it may go: it stops where the hand would leave what is allowed, where a
  joint would pass a line's limit (``model.LINE_*``: joint 6's keeps it from the
  wrist's singularity), and where the joints would ask for more than the rig lets a move
  slow down for;
* how long it has to take, for no joint to pass ``joint_speed_rad_s``.

The motion is minimum-jerk in time along the whole line, so its fastest instant
is 1.875 times its mean, and the time here counts the line's most demanding step
as if it fell on that instant: an upper bound on every joint's speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Optional, Sequence, Tuple

from ...core.geometry import relative_pose
from ...core.types import Pose
from ..kinematics import (
    KINEMATICS, ArmKinematics, _chain, _cross, _mat_mul, _quat_matrix, _rotvec, _solve, _transpose,
)
from .arm import min_jerk, slerp
from .model import LINE_LOWER, LINE_UPPER
from .rig import Motion

#: a line is followed this finely: about what a control period's setpoints do at
#: full speed, so that a joint's speed over a step is what the controller would ask
STEP_M = 0.005
STEP_RAD = math.radians(1.5)
#: Polymetis' damping, and its answer as near as it asks (``inverse_kinematics``);
#: a step this small settles in this many iterations
DAMPING = 1e-6
ITERATIONS = 3
SOLVED_M = 1e-4
SOLVED_RAD = 1e-3
#: the executor calls a move blocked that went less than this in a second, and
#: a minimum-jerk move starts slowly: what its first second has to show, the
#: arm's own lag in it (``executor.BLOCKED_TOLERANCE``, ``BLOCKED_TURN``, 1.5 times)
FIRST_SECOND_M = 0.0015
FIRST_SECOND_RAD = math.radians(1.5)


@dataclass(frozen=True)
class Route:
    """How far a line goes (``along``)."""
    #: the furthest point it may go to, or None when not even its first step
    end: Optional[Pose]
    #: where the robot's joints are there
    joints: Tuple[float, ...]
    #: how long the line to ``end`` has to take, at the least
    seconds: float
    #: why it stops short; "" when it goes all the way
    stopped: str = ""

    @property
    def whole(self) -> bool:
        return self.end is not None and not self.stopped


def solve(kin: ArmKinematics, tool: Pose, q: Sequence[float],
          goal: Pose) -> Tuple[Tuple[float, ...], bool]:
    """The joints the robot's inverse kinematics comes to from ``q`` for the
    ``tool`` frame at ``goal`` (base frame), and whether it gets there.  Its limits
    are not its concern, nor the arm's redundancy: the smallest motion of the joints
    that closes the error, whichever frame the error is read in."""
    goal_r = _quat_matrix(goal.quat)
    q = list(q)
    for i in range(ITERATIONS + 1):
        (r, p), axes = _chain(kin, q, tool)
        ep = [g - c for g, c in zip(goal.pos, p)]
        er = _rotvec(_mat_mul(goal_r, _transpose(r)))
        if i == ITERATIONS:
            break
        cols = [tuple(a + b for a, b in zip(v, _cross(w, p))) + tuple(w) for w, v in axes]
        jjt = [[sum(c[a] * c[b] for c in cols) + (DAMPING if a == b else 0.0) for b in range(6)]
               for a in range(6)]
        y = _solve(jjt, ep + list(er))
        q = [v + sum(c[a] * y[a] for a in range(6)) for v, c in zip(q, cols)]
    return tuple(q), (math.sqrt(sum(e * e for e in ep)) < SOLVED_M
                      and math.sqrt(sum(e * e for e in er)) < SOLVED_RAD)


def _past_limit(a: Sequence[float], b: Sequence[float]) -> Optional[int]:
    """The joint that going from ``a`` to ``b`` takes on past a line's limit: an arm
    that stands past one may come back."""
    for j, (x, y, lo, hi) in enumerate(zip(a, b, LINE_LOWER, LINE_UPPER)):
        if (y < lo and y < x) or (y > hi and y > x):
            return j
    return None


def along(model, base: Pose, q: Sequence[float], start: Pose, goal: Pose, motion: Motion,
          allowed: Optional[Callable[[Pose], bool]] = None) -> Route:
    """Follow the tool centre point's straight line from ``start`` to ``goal`` (world)
    from joints ``q``, as the robot would (the module docstring).  ``allowed`` says
    of a point of the line whether the hand may be there."""
    kin = KINEMATICS[model.name]
    tool = model.tcp_from_body
    now, on = solve(kin, tool, q, relative_pose(base, start))    # what the arm has yet to close
    q = now if on else tuple(q)
    dist, turn = start.distance_to(goal), start.angle_to(goal)
    if dist < 1e-9 and turn < 1e-9:
        return Route(goal, q, 0.0)
    steps = max(1, math.ceil(max(dist / STEP_M, turn / STEP_RAD)))
    at_speed = 1.875 * max(dist / motion.max_speed_m_s, turn / motion.max_turn_rad_s)
    end: Optional[Pose] = None
    seconds, biggest = 0.0, 0.0
    stopped = ""
    for i in range(1, steps + 1):
        f = i / steps
        p = Pose(tuple(a + (b - a) * f for a, b in zip(start.pos, goal.pos)),
                 slerp(start.quat, goal.quat, f))
        if allowed is not None and not allowed(p):
            stopped = "the hand would leave what it is allowed"
            break
        q1, solved = solve(kin, tool, q, relative_pose(base, p))
        if not solved:
            stopped = "the robot's inverse kinematics would not follow"
            break
        past = _past_limit(q, q1)
        if past is not None:
            stopped = f"joint {past + 1} would pass its limit"
            break
        step = max(abs(b - a) for a, b in zip(q, q1))
        unhurried = max(motion.min_move_seconds, at_speed * f)
        needs = max(unhurried, 1.875 * i * max(biggest, step) / motion.joint_speed_rad_s)
        if needs > motion.max_slowdown * unhurried or needs > motion.max_move_seconds:
            stopped = f"the joints would need more than {needs:.1f} s for it"
            break
        moved = min_jerk(1.0 / needs)
        if needs > 1.0 and moved * dist * f < FIRST_SECOND_M and moved * turn * f < FIRST_SECOND_RAD:
            stopped = f"the joints would keep it too slow for its first second ({needs:.1f} s)"
            break
        q, end, seconds, biggest = q1, p, needs, max(biggest, step)
    return Route(end, tuple(q), seconds, stopped)


__all__ = ["Route", "along", "solve"]
