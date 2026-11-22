"""Aggregating episode logs into the numbers the paper reports.

Three columns in the main table (v0 §5.1): success rate, progress score, cost.
Plus interface sensitivity, and a coverage-gap column that is deliberately kept
*out* of the success rates.

Two statistical habits enforced here rather than remembered:

* every rate carries a **Wilson interval**.  At 20 episodes per cell, a 50%
  rate has a 95% interval of roughly +-22 points; pooling a suite's three tasks
  to 60 gets it to about +-13.  Those intervals describe this fixed test set,
  not generalisation to "tasks like these".
* comparisons between interfaces are **paired on the seed**.  Same world, same
  initial state, different action language.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..tasks import SUITES, SUITE_ORDER, get as get_task


@dataclass(frozen=True)
class Rate:
    successes: int
    total: int

    @property
    def value(self) -> float:
        return self.successes / self.total if self.total else 0.0

    @property
    def interval(self) -> Tuple[float, float]:
        return wilson(self.successes, self.total)

    def __str__(self) -> str:
        lo, hi = self.interval
        return f"{self.value:.0%} [{lo:.0%}, {hi:.0%}] n={self.total}"


def wilson(successes: int, total: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval.  Never a normal approximation: at n=20 with 0 or
    20 successes the normal interval has zero width, which is nonsense."""
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denom = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denom
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Cell:
    """One (task or suite) x interface cell."""

    successes: int = 0
    total: int = 0
    progress_sum: float = 0.0
    usd: float = 0.0
    sim_seconds: float = 0.0
    model_calls: int = 0
    coverage_gaps: int = 0
    #: episodes our infrastructure lost (a crashed simulator worker).  Like
    #: coverage gaps they are counted, reported, and excluded from every rate.
    harness_errors: int = 0
    failures: Dict[str, int] = field(default_factory=lambda: defaultdict(int))

    @property
    def rate(self) -> Rate:
        return Rate(self.successes, self.total)

    @property
    def progress(self) -> float:
        return self.progress_sum / self.total if self.total else 0.0

    @property
    def usd_per_episode(self) -> float:
        return self.usd / self.total if self.total else 0.0


@dataclass
class Results:
    """Everything the report layer needs, derived from a list of episode logs."""

    cells: Dict[Tuple[str, str], Cell] = field(default_factory=dict)   # (task, interface)
    engines: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    conditions: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    by_seed: Dict[Tuple[str, str, str, int], bool] = field(default_factory=dict)
    levels: Dict[Tuple[str, str, str], Cell] = field(default_factory=dict)  # (task, iface, level)

    # -- construction ------------------------------------------------------
    @classmethod
    def from_logs(cls, logs: Iterable[Mapping]) -> "Results":
        r = cls()
        for log in logs:
            task, iface = log["task_id"], log["interface"]
            level, seed = log.get("level", "easy"), int(log.get("seed", 0))
            score = log.get("score", {})
            r.engines[log.get("engine", "?")] += 1
            r.conditions[log.get("condition", "main")] += 1

            cell = r.cells.setdefault((task, iface), Cell())
            lcell = r.levels.setdefault((task, iface, level), Cell())
            for c in (cell, lcell):
                if score.get("coverage_gap"):
                    c.coverage_gaps += 1          # never counted in the rate
                    continue
                if score.get("failure") == "harness_error":
                    c.harness_errors += 1         # our bug, not the model's
                    continue
                c.total += 1
                c.successes += int(bool(score.get("success")))
                c.progress_sum += float(score.get("progress", 0.0))
                m = log.get("metrics", {}) or {}
                c.usd += float(m.get("usd", 0.0))
                c.sim_seconds += float(m.get("sim_seconds", 0.0))
                c.model_calls += int(m.get("model_calls", 0))
                c.failures[str(score.get("failure", "none"))] += 1
            if not score.get("coverage_gap") and score.get("failure") != "harness_error":
                r.by_seed[(task, iface, level, seed)] = bool(score.get("success"))
        return r

    # -- views -------------------------------------------------------------
    @property
    def interfaces(self) -> List[str]:
        return sorted({i for _, i in self.cells})

    @property
    def tasks(self) -> List[str]:
        return sorted({t for t, _ in self.cells})

    def suite_cell(self, suite: str, interface: str) -> Cell:
        out = Cell()
        for (task, iface), cell in self.cells.items():
            if iface != interface or get_task(task).suite != suite:
                continue
            out.successes += cell.successes
            out.total += cell.total
            out.progress_sum += cell.progress_sum
            out.usd += cell.usd
            out.coverage_gaps += cell.coverage_gaps
            out.harness_errors += cell.harness_errors
        return out

    def heatmap(self) -> Dict[str, Dict[str, Cell]]:
        """The paper's main figure: four suites x four interfaces."""
        return {suite: {i: self.suite_cell(suite, i) for i in self.interfaces}
                for suite in SUITE_ORDER}

    def interface_sensitivity(self, suite: str) -> Optional[float]:
        """Best interface minus worst, per suite.

        One number for "how much does the interface matter here".  Report it
        with the caveat it deserves: at 20 episodes per cell, max-minus-min is
        biased upward by sampling noise, so prefer the pre-registered pairwise
        differences below for any claim.
        """
        rates = [c.rate.value for c in self.heatmap()[suite].values() if c.total]
        return max(rates) - min(rates) if len(rates) >= 2 else None

    def paired_difference(self, task: str, a: str, b: str) -> Optional[dict]:
        """Seed-paired difference between two interfaces on one task."""
        shared = [(lv, s) for (t, i, lv, s) in self.by_seed
                  if t == task and i == a and (t, b, lv, s) in self.by_seed]
        if not shared:
            return None
        wins = sum(1 for lv, s in shared
                   if self.by_seed[(task, a, lv, s)] and not self.by_seed[(task, b, lv, s)])
        losses = sum(1 for lv, s in shared
                     if not self.by_seed[(task, a, lv, s)] and self.by_seed[(task, b, lv, s)])
        n = len(shared)
        return {"task": task, "a": a, "b": b, "n": n, "a_wins": wins, "b_wins": losses,
                "difference": (wins - losses) / n}

    def knob_curve(self, task: str, interface: str) -> Dict[str, Rate]:
        """Success rate at each level -- the easy/hard pair that is this
        benchmark's whole single-factor claim."""
        return {lv: cell.rate for (t, i, lv), cell in sorted(self.levels.items())
                if t == task and i == interface}

    # -- guards ------------------------------------------------------------
    def leaderboard_blockers(self) -> List[str]:
        """Reasons these results must not be published as a leaderboard."""
        from ..engines import ENGINES

        problems = []
        for engine, n in self.engines.items():
            name = {"MockEngine": "mock", "IsaacEngine": "isaac"}.get(engine, engine)
            try:
                grade = ENGINES.get(name)().capabilities.results_grade
            except Exception:
                grade = False
            if not grade:
                problems.append(
                    f"{n} episodes came from the {name!r} engine, which is not "
                    "results-grade: frontier models are judged on what they see, so "
                    "only the photorealistic backend produces a leaderboard number")
        for iface in self.interfaces:
            if iface.startswith("I0"):
                problems.append(f"{iface} is a harness test double, not a benchmark interface")
        return problems


def summary_table(results: Results) -> List[dict]:
    """Main table rows: one per suite x interface."""
    rows = []
    for suite, per_iface in results.heatmap().items():
        for iface, cell in per_iface.items():
            if not cell.total and not cell.coverage_gaps and not cell.harness_errors:
                continue
            lo, hi = cell.rate.interval
            rows.append({
                "suite": suite, "suite_name": SUITES[suite][1], "interface": iface,
                "success_rate": cell.rate.value, "ci_low": lo, "ci_high": hi,
                "n": cell.total, "progress": cell.progress,
                "usd_per_episode": cell.usd_per_episode, "coverage_gaps": cell.coverage_gaps,
                "harness_errors": cell.harness_errors,
            })
    return rows
