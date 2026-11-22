"""Placements: when a carried object comes to rest in a bin.

A pure function of the trajectory, shared by the feedback channel (which emits
one bit per placement) and by the scorer (which judges the last k of them).
Keeping one definition is what guarantees the bit the model received and the
score the episode got describe the same event.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from ..core.types import ObjectState, WorldState
from .predicates import Trajectory, TrajectoryPredicate, inside


@dataclass(frozen=True)
class Placement:
    """One object coming to rest in one bin."""

    obj: str
    bin: str
    sim_time: float
    index: int                      # order of placement within the episode


def bin_of(world: WorldState, obj: ObjectState, bins: Sequence[str]) -> Optional[str]:
    """Which bin an object is resting in, if any."""
    return next((name for name in bins if inside(obj, world.obj(name))), None)


def segment_placements(traj: Sequence[WorldState], objects: Sequence[str],
                       bins: Sequence[str], settle_seconds: float = 0.5,
                       speed_tol: float = 0.02) -> List[Placement]:
    """Every time an object that had been carried comes to rest in a bin.

    Only objects that were actually picked up count: an object knocked into a
    bin by accident is not a placement the agent chose.  An object placed twice
    counts twice, in order -- which is how "the last four placements" is
    defined.
    """
    carried: Dict[str, bool] = {}
    rest_since: Dict[str, float] = {}
    placements: List[Placement] = []
    for w in traj:
        for name in objects:
            o = w.objects.get(name)
            if o is None:
                continue
            if o.is_held:
                carried[name] = True
                rest_since.pop(name, None)
                continue
            if not carried.get(name):
                continue
            where = bin_of(w, o, bins)
            if where is None or o.speed > speed_tol:
                rest_since.pop(name, None)
                continue
            start = rest_since.setdefault(name, w.sim_time)
            if w.sim_time - start >= settle_seconds:
                placements.append(Placement(name, where, w.sim_time, len(placements)))
                carried[name] = False
                rest_since.pop(name, None)
    return placements


@dataclass
class LastKCorrect(TrajectoryPredicate):
    """The last ``k`` placements all went to the bin the hidden rule names.

    IC1's success criterion.  Judged on *placements*, in order, so an agent that
    probes early and gets the tail right succeeds, and one that never learns
    does not -- whatever it declares.
    """

    objects: Sequence[str]
    bins: Sequence[str]
    rule: Callable[[ObjectState], str]
    k: int = 4
    settle_seconds: float = 0.5

    def evaluate(self, traj: Trajectory) -> bool:
        placed = segment_placements(traj, self.objects, self.bins, self.settle_seconds)
        if len(placed) < self.k or not traj:
            return False
        final = traj[-1]
        return all(self.rule(final.obj(p.obj)) == p.bin for p in placed[-self.k:])

    def describe(self) -> str:
        return f"last {self.k} placements correct under the hidden rule"
