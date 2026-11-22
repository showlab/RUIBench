"""Toys that answer: scripted parts of the world that react to the robot.

An :class:`~ruibench.events.scheduler.Event` fires once, at a milestone.  Some
toys answer every time something happens: a monkey that eats the fruit it
likes and refuses the rest, a rocket that lifts off while both buttons are
down, an opponent who puts down an O after every X.  A reactor is ticked after
every physics slice with the engine and the world state.  It changes the world
only through the engine's event hooks (driving joints, changing colours), may
say something -- a one-bit message delivered on the same path as the
In-Context suite's feedback channel -- and records everything it does with its
simulated time, as the scheduler does for events.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from ..core.types import WorldState
from ..engines.base import Engine
from .actions import EventAction
from .feedback import FeedbackMessage
from .scheduler import EventRecord


class Reactor(ABC):
    #: How long the world keeps going after the agent says it is done, for a
    #: toy whose show starts then (WK3's pendulum is let go); 0: no show.
    finale_seconds: float = 0.0

    def __init__(self) -> None:
        self.messages: List[FeedbackMessage] = []
        self.records: List[EventRecord] = []

    @abstractmethod
    def tick(self, engine: Engine, world: WorldState) -> None:
        """Look at the world; act and speak if it calls for it."""

    def finale(self, engine: Engine, world: WorldState) -> None:
        """The agent has said it is done: start the show (``finale_seconds``)."""

    def act(self, engine: Engine, world: WorldState, name: str, action: EventAction) -> None:
        detail = action.apply(engine)
        if hasattr(engine, "note_event"):
            engine.note_event(name)                  # type: ignore[attr-defined]
        self.records.append(EventRecord(name, world.sim_time, "reactor", detail))

    def note(self, engine: Engine, world: WorldState, name: str, detail: str = "") -> None:
        """Record that something happened, changing nothing (a bar played)."""
        if hasattr(engine, "note_event"):
            engine.note_event(name)                  # type: ignore[attr-defined]
        self.records.append(EventRecord(name, world.sim_time, "reactor", detail))

    def say(self, world: WorldState, obj: str, signal: str, text: str) -> None:
        self.messages.append(FeedbackMessage(obj, signal, world.sim_time, text))
