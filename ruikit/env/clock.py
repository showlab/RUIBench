"""Think time: how the model's replies are charged as simulated time."""

from __future__ import annotations

from typing import Optional

#: an external client's call after at least this long a pause starts a new
#: model turn under the ``fixed`` clock
NEW_TURN_GAP_SECONDS = 0.5


class EpisodeClock:
    """Turns and think time: the clock of the evaluation condition, charged per reply."""

    def model_turn(self, think_seconds: Optional[float] = None) -> None:
        """A loop announces a new model turn.  Under ``fixed`` the turn costs
        ``think_seconds`` of world time; under ``measured`` the caller passes
        what the model actually took (a proxy's generation time)."""
        self.model_turns += 1
        self._turn_marked = True
        if self.clock == "fixed" and self.think_per == "turn":
            self._pending_think += self.think_seconds
        elif self.clock == "measured" and think_seconds is not None:
            self._pending_think += float(think_seconds)

    def idle_reply(self) -> None:
        """A loop got a reply that called no tool.  Under ``fixed`` per reply,
        a reply costs ``think_seconds`` whether it calls a tool or not: the
        world goes on that long now, and the episode ends if its time is up.
        (The ``measured`` clock counts such a reply in the gap before the next
        call; per turn, the loop announces turns itself.)"""
        if self.over or not (self.clock == "fixed" and self.think_per == "reply"):
            return
        self.advance(self.think_seconds)
        stop = self._check_budget()
        if stop:
            self.finish(stop)

    def endpoint_wait(self, seconds: float) -> None:
        """The loop spent ``seconds`` waiting on the model's endpoint (a rate
        limit, a retry, a token), not thinking: the ``measured`` clock leaves
        them out of the gap it charges next."""
        self._waited += max(0.0, float(seconds))

    def _think_charge(self, now: float) -> float:
        think, self._pending_think = self._pending_think, 0.0
        waited, self._waited = self._waited, 0.0
        gap = (now - self._last_return_wall) if self._last_return_wall is not None else 0.0
        if self.clock == "fixed" and self.think_per == "reply":
            think += self.think_seconds
        elif self.clock == "measured" and not self._turn_marked:
            think += max(0.0, gap - waited)
        elif self.clock == "fixed" and not self._turn_marked and gap >= NEW_TURN_GAP_SECONDS:
            self.model_turns += 1
            think += self.think_seconds
        self._turn_marked = False
        return think
