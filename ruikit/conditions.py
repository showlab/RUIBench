"""The evaluation conditions: how the model's thinking is charged, and what
perception may know.

Thinking delay (the clock)
    8s          every reply costs 8 s of simulated time: the arm holds still
                while the scene goes on (the main setting); any ``<N>s`` likewise
    paused      thinking costs nothing: the world waits for the model
    measured    each reply costs what the model really took

Information
    pixels      the perception tools answer from the cameras' pictures (the main setting)
    privileged  they answer with the simulator's own values, a control
                (``@privileged`` on the interface id)
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: the main setting's clock
DEFAULT_CLOCK = "8s"
INFORMATION = ("pixels", "privileged")


def clock_settings(clock: str) -> dict:
    """The episode clock (``ruikit.env.episode.Episode``) of a condition's clock."""
    if clock == "paused":
        return {"clock": "zero"}
    if clock == "measured":
        return {"clock": "measured"}
    m = re.fullmatch(r"(\d+(?:\.\d+)?)s", str(clock))
    if not m:
        raise ValueError(f"no clock {clock!r}: <N>s (8s, the main setting), paused or measured")
    return {"clock": "fixed", "think_seconds": float(m.group(1)), "think_per": "reply"}


def clock_seconds(clock: str) -> float:
    """What a reply costs on ``clock``: its seconds, or 0 (paused, measured)."""
    return float(clock_settings(clock).get("think_seconds", 0.0))


@dataclass(frozen=True)
class Condition:
    """One evaluation condition: a clock and what perception may know."""
    clock: str = DEFAULT_CLOCK
    information: str = "pixels"

    def __post_init__(self) -> None:
        clock_settings(self.clock)                 # a clock that is none: said here
        if self.information not in INFORMATION:
            raise ValueError(f"information must be one of {INFORMATION}, not {self.information!r}")

    def episode_clock(self) -> dict:
        return clock_settings(self.clock)

    def interface(self, interface_id: str) -> str:
        """The interface id under this condition: ``@privileged`` added for the control."""
        if self.information == "privileged":
            if "@" in interface_id:
                raise ValueError(f"privileged information for {interface_id!r}, which sets its "
                                 "own perception")
            return f"{interface_id}@privileged"
        return interface_id
