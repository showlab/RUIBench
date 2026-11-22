"""An action abstraction: one way of telling the arm what to do, and the tools
that move it."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple


@dataclass(frozen=True)
class ActionAbstraction:
    """One way of telling the arm what to do: the tools that move it.

    ``kind`` is ``"move"`` for the abstractions that move the hand a step at a
    time -- joints, a delta, a step, a pose, a pixel -- and ``"act"`` for those
    that hand over a whole act (the skills).  An interface offering several
    abstractions at once (RUI-Free) lists the moves first, then the gripper
    they share, then the acts."""
    name: str
    zh: str
    en: str
    tools: Tuple[str, ...]
    #: the published work it stands for
    reproduces: str = ""
    kind: str = "move"
    #: settings of its tools that every interface over it fixes
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: when control comes back after one of its actions, one action a reply:
    #: when the move is done, or after each step (the semantic step)
    hand_back: str = "when_done"
