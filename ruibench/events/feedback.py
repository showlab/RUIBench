"""The one-bit feedback channel of the In-Context Learning suite.

Rules this module exists to enforce (v0 §2, In-Context Learning controls):

* **one bit** -- "correct" or "wrong", never why, never the rule;
* **after the object settles** -- not at release, not while it is still moving,
  so the signal cannot be read as a hint about where it is about to land;
* **one shared implementation**, delivered identically to every interface --
  the program interface included, or the suite would be comparing information
  rather than action languages.

The channel watches the trajectory, so it needs nothing from the model and
nothing from the engine beyond world state.  The same segmentation is reused by
the scorer, which is what keeps scoring a pure function of the log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.types import ObjectState, WorldState
from ..scoring.placements import Placement, bin_of, segment_placements

CORRECT = "correct"
WRONG = "wrong"


@dataclass(frozen=True)
class FeedbackMessage:
    obj: str
    signal: str                     # CORRECT | WRONG -- one bit, nothing else
    sim_time: float
    #: the words, when a toy says it its own way ("The monkey ate it.")
    text: str = ""

    def as_text(self) -> str:
        """What the model sees.  Deliberately terse: no object name would leak
        nothing, but naming the bin would leak the rule."""
        return self.text or f"{self.obj}: {self.signal}"

    def as_dict(self) -> dict:
        out = {"obj": self.obj, "signal": self.signal, "sim_time": self.sim_time}
        if self.text:
            out["text"] = self.text
        return out


@dataclass
class FeedbackChannel:
    """Emits one bit per settled placement, judged against a hidden rule.

    ``rule`` maps an object to the bin it belongs in.  It is known only to the
    evaluation process; nothing about it is ever sent except the bit.
    """

    objects: Tuple[str, ...]
    bins: Tuple[str, ...]
    rule: Callable[[ObjectState], str]
    settle_seconds: float = 0.5
    messages: List[FeedbackMessage] = field(default_factory=list)
    _history: List[WorldState] = field(default_factory=list, repr=False)
    _seen: int = 0

    def tick(self, world: WorldState) -> List[FeedbackMessage]:
        """Feed one snapshot; return any bits that became due."""
        self._history.append(world)
        placements = segment_placements(self._history, self.objects, self.bins,
                                        self.settle_seconds)
        new: List[FeedbackMessage] = []
        for p in placements[self._seen:]:
            obj = self._history[-1].obj(p.obj)
            ok = self.rule(obj) == p.bin
            msg = FeedbackMessage(p.obj, CORRECT if ok else WRONG, p.sim_time)
            self.messages.append(msg)
            new.append(msg)
        self._seen = len(placements)
        # the segmentation only needs recent history once a placement is logged
        if len(self._history) > 400 and not new:
            self._history = self._history[-200:]
            self._seen = len(segment_placements(self._history, self.objects,
                                                self.bins, self.settle_seconds))
        return new

    def delivered(self) -> Tuple[str, ...]:
        return tuple(m.as_text() for m in self.messages)
