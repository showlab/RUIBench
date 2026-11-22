"""Turning a stored trajectory into a score.

The scorer is a **pure function of the log**: same trajectory in, same numbers
out, no clocks, no randomness, no engine.  That is what makes two of the Gate B
acceptance criteria checkable -- "re-scoring the same log agrees with itself"
and "the ordering of event, clock and scorer can be re-derived".

Success is judged from the physical trajectory and the task rules, never from
the model reporting that it is finished (v0 §2.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from .audit import audit
from .failure import FailureLabel
from .milestones import Milestone, replay
from .predicates import Trajectory, TrajectoryPredicate
from .state_machine import LegalMoveMachine, MoveRecord


@dataclass
class EpisodeScore:
    """One row of the results table."""

    task_id: str
    level: str
    success: bool
    progress: float
    failure: FailureLabel = FailureLabel.NONE
    milestones_hit: Mapping[str, float] = field(default_factory=dict)
    violations: List[str] = field(default_factory=list)
    illegal_moves: List[dict] = field(default_factory=list)
    metrics: Mapping[str, float] = field(default_factory=dict)
    coverage_gap: Optional[str] = None
    #: what the physics audit found (``scoring/audit.py``); any flag makes the
    #: episode a failure until someone has looked at it
    flags: List[str] = field(default_factory=list)

    @property
    def counted(self) -> bool:
        """Coverage gaps are excluded from success rates entirely."""
        return self.coverage_gap is None

    def as_dict(self) -> dict:
        return {"task_id": self.task_id, "level": self.level, "success": self.success,
                "progress": round(self.progress, 4), "failure": self.failure.value,
                "milestones_hit": dict(self.milestones_hit), "violations": list(self.violations),
                "illegal_moves": list(self.illegal_moves), "metrics": dict(self.metrics),
                "coverage_gap": self.coverage_gap, "flags": list(self.flags)}


@dataclass
class Scorer:
    """Assembled from a task's definition; owns no state."""

    task_id: str
    level: str
    success: TrajectoryPredicate
    milestones: Tuple[Milestone, ...] = ()
    #: named "must not happen" predicates.  They are part of the success
    #: criterion (``success and not any(violation)``), not a second metric.
    violations: Tuple[Tuple[str, TrajectoryPredicate], ...] = ()
    machine: Optional[LegalMoveMachine] = None
    #: the scene's joints' stops and its fixed bodies, for the physics audit
    #: (``tasks.instance.build`` fills them in from the scene)
    joint_limits: Mapping[str, Tuple[float, float]] = field(default_factory=dict)
    fixed: Tuple[str, ...] = ()

    def score(self, traj: Trajectory, meta: Optional[Mapping[str, object]] = None) -> EpisodeScore:
        """``meta``: ``timed_out``, ``budget_exhausted``, ``parse_errors``,
        ``metrics``, ``coverage_gap``, ``dynamics`` -- does the engine have
        physics (the audit's motion checks need it; default yes) -- and
        ``leaks``: how many times a tool showed the agent one of the task's
        secrets (RUIKit's leak scan); any makes the episode void."""
        return self._score(traj, meta)

    def _score(self, traj: Trajectory, meta: Optional[Mapping[str, object]] = None) -> EpisodeScore:
        meta = dict(meta or {})
        if meta.get("coverage_gap"):
            return EpisodeScore(self.task_id, self.level, False, 0.0,
                                FailureLabel.NOT_EXPRESSIBLE,
                                coverage_gap=str(meta["coverage_gap"]))

        tracker = replay(self.milestones, traj)
        broken = [name for name, pred in self.violations if not pred.evaluate(traj)]
        illegal: List[MoveRecord] = self.machine.violations(traj) if self.machine else []
        flags = audit(traj, self.joint_limits, self.fixed, bool(meta.get("dynamics", True)))
        leaks = int(meta.get("leaks", 0) or 0)
        if leaks > 0:
            flags = [f"secret_leak:{leaks}"] + flags
        ok = bool(traj) and self.success.evaluate(traj) and not broken and not illegal and not flags

        failure = FailureLabel.NONE
        if not ok:
            if leaks > 0:
                failure = FailureLabel.SECRET_LEAK
            elif flags:
                failure = FailureLabel.PHYSICS_FLAG
            elif broken or illegal:
                failure = FailureLabel.VIOLATION
            elif meta.get("timed_out"):
                failure = FailureLabel.TIMEOUT
            elif meta.get("budget_exhausted"):
                failure = FailureLabel.BUDGET
            elif meta.get("parse_errors"):
                failure = FailureLabel.DECISION_ERROR
            else:
                failure = FailureLabel.EXECUTION_FAILURE

        return EpisodeScore(
            task_id=self.task_id, level=self.level, success=ok,
            progress=1.0 if ok else tracker.progress, failure=failure,
            milestones_hit=dict(tracker.hit_at), violations=broken,
            illegal_moves=[m.as_dict() for m in illegal],
            metrics=dict(meta.get("metrics", {})), flags=flags,
        )

    def describe(self) -> str:
        lines = [f"{self.task_id} [{self.level}]",
                 f"  success: {self.success.describe()}"]
        for m in self.milestones:
            lines.append(f"  milestone {m.describe()}")
        for name, pred in self.violations:
            lines.append(f"  must hold: {name} = {pred.describe()}")
        if self.machine:
            lines.append(f"  legal moves: {len(self.machine.locations)} declared locations")
        return "\n".join(lines)
