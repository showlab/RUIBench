"""The diagnostic oracle: what the scripted reference reads of the simulator,
never offered to an agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ruikit.env.episode import Episode


class Oracle:
    """What only the diagnostic references may read."""

    def __init__(self, episode: Episode) -> None:
        self._ep = episode

    def world(self):
        return self._ep.engine.state()

    @property
    def watched(self):
        return tuple(self._ep._watched)

    def said(self):
        return self._ep._said()

    def executor_busy(self) -> bool:
        return self._ep.executor.busy()

    @property
    def instance(self):
        return self._ep.inst
