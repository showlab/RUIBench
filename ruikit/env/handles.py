"""Action handles: what an agent holds of a motion it started, and per-arm
sequences of moves and dwells."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional, Sequence

from ruibench.core.types import Pose
from ruibench.executor.handles import ActionHandle, HandleStatus
from ruikit.core.result import ToolError

if TYPE_CHECKING:
    from ruikit.env.episode import Episode

#: a blocking action returns after this much simulated time at most, with its
#: handle still running (client time-outs: Codex waits 60 s by default)
BLOCKING_CAP_SECONDS = 20.0


@dataclass
class Handle:
    """An action the agent can wait for or stop.  One tool call may drive
    several engine actions (an approach and a descent, a list of waypoints);
    the agent sees one handle."""

    id: str
    arm: str
    label: str
    parts: List[ActionHandle]
    generation: int
    created_at: float
    target: Optional[Pose] = None
    #: a multi-step motion (commands and dwells) still being fed to the executor
    sequence: Optional["MotionSequence"] = None

    @property
    def status(self) -> str:
        if self.sequence is not None and not self.sequence.finished:
            if self.sequence.cancelled:
                return "cancelled"
            return "running"
        states = [p.status for p in self.parts]
        if any(s is HandleStatus.CANCELLED for s in states):
            return "cancelled"
        if any(s in (HandleStatus.FAILED, HandleStatus.REJECTED) for s in states):
            return "failed"
        if all(s is HandleStatus.DONE for s in states):
            return "done"
        return "running"

    @property
    def done(self) -> bool:
        return self.status != "running"

    @property
    def detail(self) -> str:
        return next((p.detail for p in self.parts if p.detail), "")


class MotionSequence:
    """Steps for one arm, fed to the executor as the previous one finishes:
    a list of commands, or a dwell in seconds (hold still -- the executor's own
    wait is global, a dwell is per arm).  Ticked with the world, so two arms'
    sequences run side by side and a press held for half a second on each is a
    real simultaneous hold."""

    def __init__(self, episode: "Episode", handle: "Handle", steps: List[Any]) -> None:
        self.ep = episode
        self.handle = handle
        self.steps = list(steps)
        self.current: List[ActionHandle] = []
        self.dwell_until: Optional[float] = None
        self.finished = False
        self.cancelled = False
        self.arm = handle.arm
        # the executor's own count of cancels on this arm, as it stood when the
        # sequence began: a step submitted after a stop is turned away there too
        self.gen = episode.executor.generation(episode.station.arm(handle.arm).arm_id)
        self._start_next()

    def _start_next(self) -> None:
        while self.steps:
            step = self.steps.pop(0)
            if isinstance(step, (int, float)):
                self.dwell_until = self.ep.sim_time + float(step)
                return
            parts = [h for h in (self.ep.executor.submit(c, generation=self.gen) for c in step)
                     if h is not None]
            self.handle.parts.extend(parts)
            self.current = parts
            if parts:
                return
        self.finished = True
        if self in self.ep.sessions:
            self.ep.sessions.remove(self)

    def tick(self, episode: "Episode", dt: float) -> None:
        if self.finished:
            return
        if any(p.status in (HandleStatus.FAILED, HandleStatus.CANCELLED, HandleStatus.REJECTED)
               for p in self.current):
            self.finished = True
            self.close(episode)
            return
        if self.dwell_until is not None:
            if episode.sim_time + 1e-9 >= self.dwell_until:
                self.dwell_until = None
                self._start_next()
            return
        if all(p.done for p in self.current):
            self._start_next()

    def request_stop(self, episode: "Episode") -> None:
        self.cancelled = True
        self.finished = True
        self.close(episode)

    def close(self, episode: "Episode") -> None:
        self.finished = True
        if self in episode.sessions:
            episode.sessions.remove(self)


class EpisodeMotion:
    """Starting, sequencing and stopping motions on the episode's arms."""

    # -- motion --------------------------------------------------------------
    def start_motion(self, arm: str, commands: Sequence[Any], label: str,
                     target: Optional[Pose] = None, generation: Optional[int] = None) -> Handle:
        """Submit one motion's commands.  ``generation``: the executor's count of
        cancels on this arm when the caller decided on it (a session); None, now."""
        info = self.station.arm(arm)
        parts = []
        for cmd in commands:
            h = self.executor.submit(cmd, generation=generation)
            if h is None:
                continue
            if h.status is HandleStatus.REJECTED:
                detail = str(getattr(h, "detail", "") or "")
                if detail.startswith("stale generation"):
                    # planned before a stop on this arm: nothing from before a stop moves it
                    raise ToolError("stale_handle", f"the arm was stopped after this was planned "
                                                    f"({detail})")
                raise ToolError("tool_failed", f"the controller rejected the command: {detail}")
            parts.append(h)
        hid = f"h{next(self._handle_ids)}"
        handle = Handle(hid, info.name, label, parts, self._generation[info.name],
                        self.sim_time, target)
        self.handles[hid] = handle
        return handle

    def start_sequence(self, arm: str, steps: Sequence[Any], label: str,
                       target: Optional[Pose] = None) -> Handle:
        info = self.station.arm(arm)
        hid = f"h{next(self._handle_ids)}"
        handle = Handle(hid, info.name, label, [], self._generation[info.name], self.sim_time,
                        target)
        self.handles[hid] = handle
        seq = MotionSequence(self, handle, list(steps))
        handle.sequence = seq
        if not seq.finished:
            self.sessions.append(seq)
        return handle

    def arm_busy(self, arm: str) -> Optional[Handle]:
        for h in reversed(list(self.handles.values())):
            if h.arm == arm and not h.done:
                return h
        return None

    def stop(self, arm: Optional[str] = None, handle: Optional[str] = None) -> List[str]:
        arms = [arm] if arm else list(self.station.arms)
        if handle is not None:
            h = self.handles.get(handle)
            if h is None:
                raise ToolError("stale_handle", f"no handle {handle!r}")
            arms = [h.arm]
        stopped = []
        for a in arms:
            self.executor.cancel(self.station.arm(a).arm_id)
            self._generation[a] += 1
            for h in self.handles.values():
                if h.arm == a and not h.done:
                    for p in h.parts:
                        if not p.done:
                            p.status = HandleStatus.CANCELLED
                            p.finished_at = self.sim_time
                    stopped.append(h.id)
            for s in list(self.sessions):
                if getattr(s, "arm", None) == a:
                    s.request_stop(self)
        return stopped
