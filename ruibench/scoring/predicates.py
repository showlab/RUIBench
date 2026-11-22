"""Predicates over privileged world state.

Two levels, on purpose:

* :class:`StatePredicate` -- true or false of a single snapshot;
* :class:`TrajectoryPredicate` -- true or false of a whole episode.

Success criteria are **trajectory** predicates, because "never did the thing it
was not supposed to do" cannot be checked from a terminal state, and because
the design forbids a second metric for constraint violations: what must not
happen goes into the success criterion itself (v0 §2.1).

Everything here is engine-agnostic and side-effect free, so scoring a stored
log twice gives the same answer -- a Gate B acceptance criterion.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, FrozenSet, Optional, Sequence, Tuple

from ..core.types import ObjectState, Vec3, WorldState

#: How far above its container's rim an object's centre may be and still count
#: as inside.  A cube resting on another cube in a bin is in it; the third cube
#: of a tower, balanced above the rim, is not.
RIM_TOLERANCE = 0.03


def inside(obj: ObjectState, container: ObjectState, z_tol: float = RIM_TOLERANCE) -> bool:
    """``obj`` rests in ``container``: over its footprint, not above its rim.

    The one definition of "in the bin": the scorer, the placement feedback and
    the scripted references all use it, so they can never disagree about
    whether something has been put away.
    """
    return (not obj.is_held and container.contains_xy(obj.pose.pos)
            and obj.pose.pos[2] <= container.pose.pos[2] + container.size[2] + z_tol)

Trajectory = Sequence[WorldState]


# ==========================================================================
# state predicates
# ==========================================================================
class StatePredicate(ABC):
    @abstractmethod
    def holds(self, w: WorldState) -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...

    def __call__(self, w: WorldState) -> bool:
        return self.holds(w)

    def __and__(self, other: "StatePredicate") -> "And":
        return And((self, other))

    def __or__(self, other: "StatePredicate") -> "Or":
        return Or((self, other))

    def __invert__(self) -> "Not":
        return Not(self)


@dataclass
class And(StatePredicate):
    parts: Tuple[StatePredicate, ...]

    def holds(self, w): return all(p.holds(w) for p in self.parts)
    def describe(self): return "(" + " and ".join(p.describe() for p in self.parts) + ")"


@dataclass
class Or(StatePredicate):
    parts: Tuple[StatePredicate, ...]

    def holds(self, w): return any(p.holds(w) for p in self.parts)
    def describe(self): return "(" + " or ".join(p.describe() for p in self.parts) + ")"


@dataclass
class Not(StatePredicate):
    part: StatePredicate

    def holds(self, w): return not self.part.holds(w)
    def describe(self): return f"not {self.part.describe()}"


@dataclass
class On(StatePredicate):
    """``obj`` rests on top of ``support``: within its footprint and just above it."""

    obj: str
    support: str
    xy_margin: float = 0.0
    z_tol: float = 0.02

    def holds(self, w: WorldState) -> bool:
        a, b = w.obj(self.obj), w.obj(self.support)
        if a.is_held:
            return False
        if not b.contains_xy(a.pose.pos, self.xy_margin):
            return False
        dz = a.pose.pos[2] - (b.pose.pos[2] + b.size[2])
        return -self.z_tol <= dz <= a.size[2] * 2 + self.z_tol

    def describe(self): return f"On({self.obj}, {self.support})"


@dataclass
class In(StatePredicate):
    """``obj`` is inside container ``container`` (footprint + below its rim)."""

    obj: str
    container: str
    z_tol: float = RIM_TOLERANCE

    def holds(self, w: WorldState) -> bool:
        return inside(w.obj(self.obj), w.obj(self.container), self.z_tol)

    def describe(self): return f"In({self.obj}, {self.container})"


@dataclass
class NearPoint(StatePredicate):
    """Centre within ``eps`` of a world point.  This is the Fine-Grained knob:
    ``eps`` is a declared field, and it is told to the agent -- otherwise the
    suite measures the error distribution instead of whether the model adapts
    its behaviour to the required precision (v0 §2)."""

    obj: str
    point: Vec3
    eps: float
    planar: bool = True

    def holds(self, w: WorldState) -> bool:
        p = w.obj(self.obj).pose
        d = p.planar_distance_to(self.point) if self.planar else p.distance_to(self.point)
        return d <= self.eps

    def describe(self): return f"NearPoint({self.obj}, eps={self.eps:g}m)"


@dataclass
class Held(StatePredicate):
    obj: str
    arm: Optional[str] = None

    def holds(self, w: WorldState) -> bool:
        h = w.obj(self.obj).held_by
        return h is not None and (self.arm is None or h == self.arm)

    def describe(self): return f"Held({self.obj}{', ' + self.arm if self.arm else ''})"


@dataclass
class AtRest(StatePredicate):
    obj: str
    speed_tol: float = 0.01

    def holds(self, w: WorldState) -> bool:
        o = w.obj(self.obj)
        return (not o.is_held) and o.speed <= self.speed_tol

    def describe(self): return f"AtRest({self.obj})"


@dataclass
class RelativeStatic(StatePredicate):
    """``obj`` is not moving *relative to* ``ref``.

    DI2's success criterion: a ball inside a bowl that is itself sliding is
    still a success.  Absolute rest would be the wrong test (v0 §2).
    """

    obj: str
    ref: str
    speed_tol: float = 0.02

    def holds(self, w: WorldState) -> bool:
        a, b = w.obj(self.obj), w.obj(self.ref)
        rel = math.sqrt(sum((x - y) ** 2 for x, y in zip(a.lin_vel, b.lin_vel)))
        return (not a.is_held) and rel <= self.speed_tol

    def describe(self): return f"RelativeStatic({self.obj}, {self.ref})"


@dataclass
class JointOpen(StatePredicate):
    joint: str
    threshold: float = 0.1

    def holds(self, w: WorldState) -> bool:
        return w.articulations.get(self.joint, 0.0) >= self.threshold

    def describe(self): return f"JointOpen({self.joint} >= {self.threshold:g})"


@dataclass
class Upright(StatePredicate):
    """Body z-axis within ``tol`` radians of world up."""

    obj: str
    tol: float = 0.26          # ~15 degrees

    def holds(self, w: WorldState) -> bool:
        qw, qx, qy, qz = w.obj(self.obj).pose.quat
        up_z = 1 - 2 * (qx * qx + qy * qy)
        return math.acos(max(-1.0, min(1.0, up_z))) <= self.tol

    def describe(self): return f"Upright({self.obj})"


@dataclass
class VisibleFrom(StatePredicate):
    obj: str
    camera: str

    def holds(self, w: WorldState) -> bool:
        return w.obj(self.obj).seen(self.camera)

    def describe(self): return f"VisibleFrom({self.obj}, {self.camera})"


@dataclass
class InstructionIs(StatePredicate):
    """Which instruction version was in force.  Used by DI4 to score against the
    instruction that was actually current at that moment."""

    version: int

    def holds(self, w: WorldState) -> bool:
        return w.instruction_version == self.version

    def describe(self): return f"InstructionVersion=={self.version}"


@dataclass
class Custom(StatePredicate):
    fn: Callable[[WorldState], bool]
    label: str = "custom"

    def holds(self, w): return bool(self.fn(w))
    def describe(self): return self.label


# ==========================================================================
# trajectory predicates
# ==========================================================================
class TrajectoryPredicate(ABC):
    @abstractmethod
    def evaluate(self, traj: Trajectory) -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...

    def __and__(self, other): return AllOfT((self, other))
    def __or__(self, other): return AnyOfT((self, other))
    def __invert__(self): return NotT(self)


@dataclass
class AllOfT(TrajectoryPredicate):
    parts: Tuple[TrajectoryPredicate, ...]

    def evaluate(self, traj): return all(p.evaluate(traj) for p in self.parts)
    def describe(self): return "(" + " and ".join(p.describe() for p in self.parts) + ")"


@dataclass
class AnyOfT(TrajectoryPredicate):
    parts: Tuple[TrajectoryPredicate, ...]

    def evaluate(self, traj): return any(p.evaluate(traj) for p in self.parts)
    def describe(self): return "(" + " or ".join(p.describe() for p in self.parts) + ")"


@dataclass
class NotT(TrajectoryPredicate):
    part: TrajectoryPredicate

    def evaluate(self, traj): return not self.part.evaluate(traj)
    def describe(self): return f"not {self.part.describe()}"


@dataclass
class Finally(TrajectoryPredicate):
    """True if the state predicate holds in the final snapshot."""

    pred: StatePredicate

    def evaluate(self, traj): return bool(traj) and self.pred.holds(traj[-1])
    def describe(self): return f"Finally {self.pred.describe()}"


@dataclass
class Ever(TrajectoryPredicate):
    pred: StatePredicate

    def evaluate(self, traj): return any(self.pred.holds(w) for w in traj)
    def describe(self): return f"Ever {self.pred.describe()}"


@dataclass
class Never(TrajectoryPredicate):
    """The form every "must not do X" constraint takes.  Kept inside the success
    criterion rather than as a second metric (v0 §2.1)."""

    pred: StatePredicate

    def evaluate(self, traj): return not any(self.pred.holds(w) for w in traj)
    def describe(self): return f"Never {self.pred.describe()}"


@dataclass
class Before(TrajectoryPredicate):
    """The state predicate became true within ``seconds`` of the start (and
    holds at the end): a task with a time limit."""

    pred: StatePredicate
    seconds: float

    def evaluate(self, traj: Trajectory) -> bool:
        if not traj or not self.pred.holds(traj[-1]):
            return False
        t0 = traj[0].sim_time
        return any(self.pred.holds(w) for w in traj if w.sim_time - t0 <= self.seconds + 1e-9)

    def describe(self): return f"{self.pred.describe()} within {self.seconds:g}s"


@dataclass
class KnownBefore(TrajectoryPredicate):
    """The hidden answer found out before it was acted on.  Replaying the
    trajectory, ``narrow`` keeps the candidates a snapshot leaves open after what
    the world showed in it (a strip's colour, a balance's tilt, a detector's
    lights), and at the first snapshot of ``act`` (the answer set down, lifted, put
    in the dish) only ``answer`` may be left.  Ruling all the others out counts
    as much as testing the answer itself -- an agent may deduce it -- and what
    the agent made of what it was shown is not asked, only that the world showed
    it.  True when ``act`` never comes: the rest of the criterion judges that.
    ``narrow(world, left, memo)``: ``memo`` is the replay's own, for what a rule
    needs to remember from one snapshot to the next."""

    answer: str
    candidates: Tuple[str, ...]
    narrow: Callable[[WorldState, FrozenSet[str], dict], FrozenSet[str]]
    act: StatePredicate
    what: str

    def evaluate(self, traj: Trajectory) -> bool:
        left, memo = frozenset(self.candidates), {}
        for w in traj:
            left = self.narrow(w, left, memo)
            if self.act.holds(w):
                return left == frozenset((self.answer,))
        return True

    def describe(self) -> str:
        return f"{self.what} found out before it was acted on"


@dataclass
class HoldsFor(TrajectoryPredicate):
    """The predicate stays true for ``seconds`` of continuous simulated time.

    "stable for 3 s", "at rest for 2 s", "relatively static for 2 s".
    """

    pred: StatePredicate
    seconds: float

    def evaluate(self, traj: Trajectory) -> bool:
        return self.longest_run(traj) >= self.seconds - 1e-9

    def longest_run(self, traj: Trajectory) -> float:
        best = run_start = 0.0
        running = False
        for w in traj:
            if self.pred.holds(w):
                if not running:
                    running, run_start = True, w.sim_time
                best = max(best, w.sim_time - run_start)
            else:
                running = False
        return best

    def describe(self): return f"{self.pred.describe()} for {self.seconds:g}s"
