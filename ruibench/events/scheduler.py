"""The event scheduler.

One scheduler per episode.  It is ticked by the runner after every physics
advance, fires each event at most once, and writes an auditable record of
*when* each one fired in simulated time -- which is what makes the Gate B
criterion "event / clock / scorer ordering can be re-checked" meaningful.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

from ..core.errors import ConfigError
from ..core.types import WorldState
from ..engines.base import Engine
from .actions import EventAction
from .triggers import Trigger, TriggerContext


@dataclass
class Event:
    name: str
    trigger: Trigger
    action: EventAction
    once: bool = True
    #: diagnostic events are allowed to use forbidden triggers
    diagnostic: bool = False


@dataclass
class EventRecord:
    name: str
    sim_time: float
    trigger: str
    detail: str

    def as_dict(self) -> dict:
        return {"name": self.name, "sim_time": self.sim_time,
                "trigger": self.trigger, "detail": self.detail}


class EventScheduler:
    def __init__(self, events: Sequence[Event], episode_start: float = 0.0) -> None:
        for e in events:
            if not e.diagnostic and not e.trigger.main_board_allowed:
                raise ConfigError(
                    f"event {e.name!r} uses trigger {e.trigger.describe()}, which is not "
                    "allowed on the main board; mark the event diagnostic=True or use a "
                    "physical-milestone trigger (TODO §4)"
                )
        self._events = list(events)
        self._fired: set[str] = set()
        self._episode_start = episode_start
        self.records: List[EventRecord] = []

    def tick(self, engine: Engine, world: WorldState, milestones_hit: Sequence[str],
             model_calls: int) -> List[EventRecord]:
        ctx = TriggerContext(frozenset(milestones_hit), model_calls, self._episode_start)
        fired: List[EventRecord] = []
        for e in self._events:
            if e.once and e.name in self._fired:
                continue
            if not e.trigger.ready(world, ctx):
                continue
            detail = e.action.apply(engine)
            if hasattr(engine, "note_event"):
                engine.note_event(e.name)       # type: ignore[attr-defined]
            rec = EventRecord(e.name, world.sim_time, e.trigger.describe(), detail)
            self.records.append(rec)
            fired.append(rec)
            self._fired.add(e.name)
        return fired

    @property
    def pending(self) -> Tuple[str, ...]:
        return tuple(e.name for e in self._events if e.name not in self._fired)

    def fired_at(self, name: str) -> Optional[float]:
        for r in self.records:
            if r.name == name:
                return r.sim_time
        return None
