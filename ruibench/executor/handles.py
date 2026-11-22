"""Action handles and the two waits.

Every interface gets handles, including the ones that only make sequential
calls -- otherwise they could not express even *serial* two-arm work, let alone
concurrent (design doc v0 §7, TODO §4).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable, List, Optional

#: ids for a handle made outside an executor (an executor numbers its own from h1)
_ids = itertools.count(1)


class HandleStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    DONE = "done"
    CANCELLED = "cancelled"
    FAILED = "failed"
    REJECTED = "rejected"


@dataclass
class ActionHandle:
    """One submitted action.  Its timestamps are the raw material of the
    mechanism figure: queueing delay, execution time, leftover motion after a
    change (v0 §5.3)."""

    arm: str
    kind: str
    submitted_at: float
    id: str = field(default_factory=lambda: f"h{next(_ids)}")
    status: HandleStatus = HandleStatus.QUEUED
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    generation: int = 0
    detail: str = ""

    @property
    def done(self) -> bool:
        return self.status in (HandleStatus.DONE, HandleStatus.CANCELLED,
                               HandleStatus.FAILED, HandleStatus.REJECTED)

    @property
    def succeeded(self) -> bool:
        return self.status is HandleStatus.DONE

    def as_dict(self) -> dict:
        return {"id": self.id, "arm": self.arm, "kind": self.kind,
                "status": self.status.value, "submitted_at": self.submitted_at,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "generation": self.generation, "detail": self.detail}


def all_done(handles: Iterable[ActionHandle]) -> bool:
    return all(h.done for h in handles)


def any_done(handles: Iterable[ActionHandle]) -> bool:
    hs = list(handles)
    return (not hs) or any(h.done for h in hs)
