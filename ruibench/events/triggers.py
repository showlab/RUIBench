"""When an event fires.

The rule that shapes this module: **main-board events fire on physical
milestones, never on "the k-th model call"** (TODO §4).  Tying an event to a
call count makes a fast interface meet a different world from a slow one, which
is exactly the confound the Responsiveness suite exists to avoid.

``ModelCallTrigger`` exists anyway -- some diagnostics want it -- but it carries
``main_board_allowed = False`` and the task validator rejects any card that
uses it outside a diagnostic condition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Optional

from ..core.types import WorldState


class Trigger(ABC):
    #: may a main-board task use this trigger?
    main_board_allowed: bool = True

    @abstractmethod
    def ready(self, world: WorldState, ctx: "TriggerContext") -> bool: ...

    @abstractmethod
    def describe(self) -> str: ...


@dataclass
class TriggerContext:
    """What a trigger may consult besides the world state."""

    milestones_hit: frozenset = frozenset()
    model_calls: int = 0
    episode_start: float = 0.0


@dataclass
class SimTimeTrigger(Trigger):
    """Fire at a fixed simulated time since episode start."""

    seconds: float

    def ready(self, world: WorldState, ctx: TriggerContext) -> bool:
        return world.sim_time - ctx.episode_start >= self.seconds

    def describe(self) -> str:
        return f"t >= {self.seconds:g}s"


@dataclass
class MilestoneTrigger(Trigger):
    """Fire once a physical milestone has been reached -- the preferred form.

    Every interface then meets the event at the same point *in the task*,
    whatever its re-entry period.
    """

    milestone: str

    def ready(self, world: WorldState, ctx: TriggerContext) -> bool:
        return self.milestone in ctx.milestones_hit

    def describe(self) -> str:
        return f"milestone {self.milestone!r}"


@dataclass
class PredicateTrigger(Trigger):
    """Fire when an arbitrary predicate over the world state holds."""

    predicate: Callable[[WorldState], bool]
    label: str = "predicate"

    def ready(self, world: WorldState, ctx: TriggerContext) -> bool:
        return bool(self.predicate(world))

    def describe(self) -> str:
        return self.label


@dataclass
class ModelCallTrigger(Trigger):
    """Fire on the k-th model call.  **Diagnostics only.**"""

    calls: int
    main_board_allowed: bool = False

    def ready(self, world: WorldState, ctx: TriggerContext) -> bool:
        return ctx.model_calls >= self.calls

    def describe(self) -> str:
        return f"model call #{self.calls} (diagnostic only)"


@dataclass
class AllOf(Trigger):
    """Conjunction; useful for 'once the arm is close *and* t > x'."""

    parts: tuple

    def __post_init__(self) -> None:
        self.main_board_allowed = all(p.main_board_allowed for p in self.parts)

    def ready(self, world: WorldState, ctx: TriggerContext) -> bool:
        return all(p.ready(world, ctx) for p in self.parts)

    def describe(self) -> str:
        return " and ".join(p.describe() for p in self.parts)
