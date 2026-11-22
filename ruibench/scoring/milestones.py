"""Milestones: the progress score, and the trigger points events fire on.

Three to four per task (v0 §5.1).  They serve two purposes at once, which is
why they live in their own module: they give partial credit, and they are the
**physical** anchor an event script fires on, instead of a model-call count.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ..core.types import WorldState
from .predicates import StatePredicate, Trajectory


@dataclass(frozen=True)
class Milestone:
    name: str
    predicate: StatePredicate
    #: milestones are ordered; reaching #3 implies credit for #1 and #2
    ordered: bool = True

    def describe(self) -> str:
        return f"{self.name}: {self.predicate.describe()}"


@dataclass
class MilestoneTracker:
    """Records the first time each milestone became true."""

    milestones: Tuple[Milestone, ...]
    hit_at: dict = None

    def __post_init__(self) -> None:
        if self.hit_at is None:
            self.hit_at = {}

    def update(self, w: WorldState) -> List[str]:
        newly = []
        for m in self.milestones:
            if m.name in self.hit_at:
                continue
            if m.predicate.holds(w):
                self.hit_at[m.name] = w.sim_time
                newly.append(m.name)
        return newly

    @property
    def hit(self) -> Tuple[str, ...]:
        return tuple(self.hit_at)

    @property
    def progress(self) -> float:
        """Furthest ordered milestone reached, divided by the total.

        Ordered milestones use the deepest one reached rather than the count,
        so a task that is briefly satisfied out of order cannot inflate it.
        """
        if not self.milestones:
            return 0.0
        ordered = [m for m in self.milestones if m.ordered]
        if ordered:
            deepest = 0
            for i, m in enumerate(ordered, start=1):
                if m.name in self.hit_at:
                    deepest = i
            return deepest / len(self.milestones)
        return len(self.hit_at) / len(self.milestones)


def replay(milestones: Sequence[Milestone], traj: Trajectory) -> MilestoneTracker:
    """Re-derive milestone times from a stored trajectory.

    Used by ``ruibench rescore`` to check the Gate B criterion that re-scoring
    the same log gives the same answer.
    """
    tracker = MilestoneTracker(tuple(milestones))
    for w in traj:
        tracker.update(w)
    return tracker
