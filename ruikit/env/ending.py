"""The end of an episode: asked for by the agent or a limit, then scored from the physical
trajectory and logged."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class Outcome:
    """How an episode ended: why, whether it succeeded, how far it got, its score and metrics."""
    reason: str
    success: bool
    progress: float
    score: dict
    metrics: dict


class EpisodeEnd:
    """The end of an episode: asked for by the agent or a limit, then scored from the physical
    trajectory and logged."""

    # -- the end ---------------------------------------------------------------
    def request_done(self, report: str) -> None:
        self._done_report = report or ""

    def give_up(self, reason: str) -> None:
        self._given_up = reason or ""

    def finish(self, reason: str) -> Outcome:
        if self.outcome is not None:
            return self.outcome
        for s in list(self.sessions):
            try:
                s.close(self)
            except Exception:
                pass
        if self.sandbox is not None:
            self.sandbox.close()
        if self._recorder is not None:
            self._recorder.final(self.engine, self._scene_cameras)
        metrics = {"tool_calls": self.tool_calls, "replies": self.replies,
                   "model_turns": self.model_turns,
                   "sim_seconds": round(self.sim_time, 3),
                   "wall_seconds": round(time.time() - self._wall_start, 3),
                   "usd": round(self.usd, 5), "frames": len(self.frames),
                   "actions": len(self.executor.history), "leaks": len(self.leaks)}
        scoring = {
            "timed_out": reason in ("timeout", "wall_timeout"),
            "budget_exhausted": reason in ("budget_calls", "budget_turns", "budget_replies"),
            "parse_errors": 0,
            # the physics audit's motion checks need a physics engine: the mock
            # drops a released thing straight down, which would read as a jump
            "dynamics": bool(getattr(getattr(self.engine, "capabilities", None), "dynamics", True)),
            "leaks": len(self.leaks),
        }
        # a leak voids the run like a physics flag does -- judged by the scorer,
        # so ``ruibench rescore`` reaches the same verdict from the log alone
        score = self.inst.scorer.score(self.traj, {**scoring, "metrics": metrics})
        self._scoring, self._score = scoring, score.as_dict()
        self.outcome = Outcome(reason=reason, success=bool(score.success),
                               progress=float(score.progress), score=score.as_dict(),
                               metrics=metrics)
        self._write_log()
        return self.outcome

    def close(self) -> None:
        if not self.over:
            self.finish("closed")
        if self._own_engine:
            self.engine.close()
