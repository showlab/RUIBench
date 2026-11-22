"""Three run conditions, one rule: the world advances while the model thinks.

Each clock answers a single question -- *how much simulated time passes during
one model call?* -- and nothing else in the harness reads a wall clock
(design doc v0 §3.2, details §1.3).

============  ==========================  ===================================
clock         simulated time per call     what it is for
============  ==========================  ===================================
zero_latency  0                           separates "thought wrong" from "too slow"
controlled    a fixed, swept L            latency pressure, comparable across models
measured      the real API latency        deployment as it is today
============  ==========================  ===================================

Two boundaries that are easy to violate by accident:

* when sweeping L, **do not** rescale object speeds to match -- that turns a
  latency experiment back into a difficulty experiment;
* within one run, every model faces the same clock.  Never tune the world to a
  model's measured latency; that changes the task instead of measuring the model.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..core.registry import Registry


class Clock(ABC):
    name: str = ""

    @abstractmethod
    def sim_seconds_for(self, wall_seconds: float, turn_index: int = 0) -> float:
        """Simulated seconds that elapse while a call taking ``wall_seconds`` runs."""

    def describe(self) -> str:
        return self.name


@dataclass
class ZeroLatencyClock(Clock):
    """The world pauses while the model thinks.  A diagnostic reference, never
    a deployment claim."""

    name: str = "zero_latency"

    def sim_seconds_for(self, wall_seconds: float, turn_index: int = 0) -> float:
        return 0.0


@dataclass
class ControlledLatencyClock(Clock):
    """A fixed latency for everyone, so the same world faces every model."""

    latency_seconds: float = 20.0
    name: str = "controlled"

    def sim_seconds_for(self, wall_seconds: float, turn_index: int = 0) -> float:
        return float(self.latency_seconds)

    def describe(self) -> str:
        return f"controlled(L={self.latency_seconds:g}s)"


@dataclass
class MeasuredClock(Clock):
    """Real wall-clock latency of the real API call, passed straight through."""

    name: str = "measured"
    floor_seconds: float = 0.0

    def sim_seconds_for(self, wall_seconds: float, turn_index: int = 0) -> float:
        return max(self.floor_seconds, float(wall_seconds))


CLOCKS: Registry = Registry("clock")
CLOCKS.register("zero_latency", ZeroLatencyClock)
CLOCKS.register("controlled", ControlledLatencyClock)
CLOCKS.register("measured", MeasuredClock)


def make_clock(name: str, **kwargs) -> Clock:
    cls = CLOCKS.get(name)
    if name == "controlled" and "latency_seconds" not in kwargs:
        kwargs["latency_seconds"] = 20.0
    return cls(**kwargs)


__all__ = ["Clock", "ZeroLatencyClock", "ControlledLatencyClock", "MeasuredClock",
           "CLOCKS", "make_clock"]
