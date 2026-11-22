"""The world between calls: advanced in snapshot slices -- motions, events, toys, feedback and
sessions ticking together -- and the events it produces, stamped on each result."""

from __future__ import annotations

from typing import Callable, List

from ruibench.runner import heartbeat
from ruibench.runner.episode import (
    SNAPSHOT_DT,
)
from ruikit.core.result import ToolResult


class EpisodeWorld:
    """The world between calls: advanced in snapshot slices -- motions, events, toys, feedback and
    sessions ticking together -- and the events it produces, stamped on each result."""

    def advance(self, seconds: float) -> None:
        """Advance the world in snapshot slices: motions, events, toys,
        feedback and running sessions all tick; the trajectory records it."""
        remaining = max(0.0, float(seconds))
        while remaining > 1e-9:
            dt = min(SNAPSHOT_DT, remaining)
            for s in list(self.sessions):
                s.tick(self, dt)
            self.executor.tick(dt)
            # a sign of life for a watchdog (a no-op unless one set it up; at most 1/s)
            heartbeat.beat(heartbeat.SIM, sim_time=round(self.sim_time, 1))
            world = self.engine.state()
            self.tracker.update(world)
            self.scheduler.tick(self.engine, world, self.tracker.hit, self.model_turns)
            if self._reactor is not None:
                self._reactor.tick(self.engine, world)
            if self._feedback is not None:
                self._feedback.tick(world)
            self.traj.append(self.engine.state())
            if self._recorder is not None:
                self._recorder.tick(self.engine, self._scene_cameras)
            self._collect_events()
            remaining -= dt

    def _run_until(self, done: Callable[[], bool], cap_seconds: float) -> bool:
        waited = 0.0
        while not done() and waited < cap_seconds - 1e-9:
            self.advance(SNAPSHOT_DT)
            waited += SNAPSHOT_DT
        return done()

    # -- events ------------------------------------------------------------
    def _said(self) -> list:
        out = list(getattr(self._feedback, "messages", ()))
        out += list(getattr(self._reactor, "messages", ()))
        return sorted(out, key=lambda m: m.sim_time)

    def _collect_events(self) -> None:
        world = self.traj[-1]
        if world.instruction_version != self._instruction_version:
            self._instruction_version = world.instruction_version
            self._events.append({"t": round(world.sim_time, 2), "type": "instruction",
                                 "version": world.instruction_version,
                                 "text": world.instruction})
        said = self._said()
        for msg in said[self._delivered:]:
            self._events.append({"t": round(msg.sim_time, 2), "type": "message",
                                 "text": msg.as_text()})
        self._delivered = len(said)

    def events(self, since: int = 0) -> List[dict]:
        return list(self._events[since:])

    def _stamp(self, result: ToolResult) -> ToolResult:
        new = self._events[self._events_seen:]
        self._events_seen = len(self._events)
        result.events = list(new)
        result.sim_time = self.sim_time
        result.budget = self.budget_left()
        return result
