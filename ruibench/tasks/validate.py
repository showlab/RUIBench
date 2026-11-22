"""Design rules, enforced.

The design documents contain a set of promises that are easy to break quietly
during implementation.  Each one below is a promise turned into a check, so
breaking it fails ``ruibench validate`` instead of surviving into the paper.

Run it in CI.  It needs no simulator.
"""

from __future__ import annotations

from ..core.config import DEFAULT_T_REF_SECONDS

from dataclasses import dataclass
from typing import List, Sequence

from ..engines.isaac.assets import load_catalogue, primitives
from .loader import load_all
from .spec import ABILITIES, ABILITY_WEIGHTS, SUITES, SplitRole, Status, TaskSpec

#: suites of six on the board, one held-out template each
TASKS_PER_SUITE = 6
#: two-arm tasks on the board, at the least: WK4, CL1, DI1, DI5, CR6
TWO_ARM_TASKS = 5
#: how well a mission's origin is backed (v4): a named paradigm with children's
#: data, a known game or school practical, or our own
SOURCE_GRADES = ("A", "B", "C")

#: every task draws on at least this many basic abilities (v3 §0)
MIN_ABILITIES = 3


@dataclass
class Finding:
    level: str        # "error" | "warning"
    where: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level}] {self.where}: {self.message}"


#: a hint's length, characters, before it is a manual (v5)
HINT_MAX = 320


def validate(tasks: Sequence[TaskSpec] | None = None) -> List[Finding]:
    tasks = list(tasks if tasks is not None else load_all())
    catalogue = load_catalogue()
    built = primitives()
    out: List[Finding] = []
    by_suite = {s: [t for t in tasks if t.suite == s] for s in SUITES}

    # -- shape of the benchmark -------------------------------------------
    for suite, group in by_suite.items():
        prefix = SUITES[suite][0]
        live = [t for t in group if t.split_role is not SplitRole.HOLDOUT]
        held = [t for t in group if t.split_role is SplitRole.HOLDOUT]
        if len(live) != TASKS_PER_SUITE:
            out.append(Finding("error", suite, f"expected {TASKS_PER_SUITE} main-board tasks, "
                               f"found {len(live)}"))
        if len(held) != 1:
            out.append(Finding("error", suite,
                               f"expected exactly 1 held-out template, found {len(held)}"))
        if sum(1 for t in live if t.split_role is SplitRole.DEV) != 1:
            out.append(Finding("error", suite, "expected exactly 1 dev task per suite"))
        for t in group:
            if not t.id.startswith(prefix):
                out.append(Finding("error", t.id, f"a task is named after its suite: "
                                   f"{prefix!r} for {suite}"))
    prefixes = {p for p, _, _ in SUITES.values()}
    for t in tasks:
        if t.id[:2] not in prefixes or not t.id[2:].isdigit():
            out.append(Finding("error", t.id, f"an id is a suite's prefix {sorted(prefixes)} and "
                               "a number"))
    if len({t.id for t in tasks}) != len(tasks):
        out.append(Finding("error", "benchmark", "two cards share an id"))
    two_arm = [t.id for t in tasks if t.arms == 2 and t.split_role is not SplitRole.HOLDOUT]
    if len(two_arm) < TWO_ARM_TASKS:
        out.append(Finding("error", "benchmark", f"expected at least {TWO_ARM_TASKS} two-arm "
                           f"tasks on the board, found {two_arm}"))

    # -- per-task discipline ----------------------------------------------
    for t in tasks:
        if set(t.knob.levels) != {"easy", "hard"}:
            out.append(Finding("error", t.id, "a knob must have exactly two levels"))
        if not t.confounds:
            out.append(Finding(
                "error", t.id,
                "no confounds listed.  One config field does not make a single-factor "
                "experiment; list what else changes with the knob, or stop claiming it "
                "is a controlled contrast (v0 §1.1)"))
        if t.split_role is not SplitRole.HOLDOUT:
            if not t.gap:
                out.append(Finding("error", t.id, "no gap: what does the task leave out on purpose? (v3 §0)"))
            if not t.gap_filled:
                out.append(Finding("error", t.id,
                                   "no gap_filled control: the same task with the gap filled in (v3 §0)"))
            unknown = sorted(set(t.abilities) - set(ABILITIES))
            if unknown:
                out.append(Finding("error", t.id, f"unknown abilities {unknown}; have {sorted(ABILITIES)}"))
            bad = sorted(k for k, w in t.abilities.items() if w not in ABILITY_WEIGHTS)
            if bad:
                out.append(Finding("error", t.id, f"abilities {bad} must be primary or secondary"))
            if len(t.abilities) < MIN_ABILITIES:
                out.append(Finding("error", t.id, f"draws on {len(t.abilities)} basic abilities; "
                                   f"every task draws on at least {MIN_ABILITIES} (v3 §0)"))
            if "primary" not in t.abilities.values():
                out.append(Finding("error", t.id, "no primary ability"))
            if ("BI" in t.abilities) != (t.arms == 2):
                out.append(Finding("error", t.id, "a two-arm task, and only one, draws on BI"))
            grade = t.source_grade
            if grade not in SOURCE_GRADES:
                out.append(Finding("error", t.id,
                                   "no source: where does the mission come from? (the card's "
                                   "source: grade A, B or C, with refs for A and B)"))
            elif grade in ("A", "B") and not t.source.get("refs"):
                out.append(Finding("error", t.id, f"source grade {grade} but no refs"))
            elif grade == "C":
                out.append(Finding("warning", t.id,
                                   "source grade C: a mission of our own, with no published "
                                   "paradigm or known game behind it to cite"))
            if t.hint and t.gap_filled and t.gap_filled.lower().rstrip(".") in t.hint.lower():
                out.append(Finding("error", t.id,
                                   "the hint gives away the gap: it says what the gap_filled "
                                   "control says (v5 -- a hint tells how things work, never "
                                   "what the task leaves out)"))
            if len(t.hint) > HINT_MAX:
                out.append(Finding("warning", t.id, f"the hint runs to {len(t.hint)} characters; "
                                   f"a few plain sentences ({HINT_MAX} at most) read best"))
            if grade == "A" and not t.source.get("children"):
                out.append(Finding("error", t.id,
                                   "source grade A but no children's result to compare with"))
            if t.status is Status.IMPLEMENTED and not t.blind:
                out.append(Finding(
                    "warning", t.id,
                    "no blind control yet: an agent that never fills the gap, to show it "
                    "does no better than chance"))
        if t.status is Status.DEFERRED and t.split_role is not SplitRole.HOLDOUT:
            out.append(Finding("error", t.id, "deferred tasks must be split_role: holdout"))
        if t.split_role is SplitRole.HOLDOUT and t.status is not Status.DEFERRED:
            out.append(Finding("error", t.id,
                               "held-out templates must stay status: deferred -- implementing one "
                               "spends the benchmark's only clean extrapolation check"))
        if t.status is Status.DEFERRED and t.impl:
            out.append(Finding("error", t.id, "a deferred task must not name an implementation"))
        if t.status is Status.IMPLEMENTED and not t.impl:
            out.append(Finding("error", t.id, "status is implemented but no impl module is named"))
        if t.status is Status.IMPLEMENTED and not t.failure_example:
            out.append(Finding("error", t.id,
                               "an implemented task names a failure_example: the agent whose "
                               "failure the showcase shows"))
        for label, spec in (("failure_example", t.failure_example), ("blind", t.blind)):
            if not spec:
                continue
            from ..core.errors import ConfigError
            from ..policies.agents import parse_agent_spec
            try:
                parse_agent_spec(spec)
            except ConfigError as exc:
                out.append(Finding("error", t.id, f"{label}: {exc}"))
        if t.abilities.get("SP") == "primary" and t.status is Status.IMPLEMENTED \
                and not t.knob.calibrated:
            out.append(Finding(
                "warning", t.id,
                f"knob levels are provisional: set for an assumed response time of "
                f"{DEFAULT_T_REF_SECONDS:g} s (RunConfig.t_ref_seconds); measure the "
                f"interfaces' effective response time and set T_ref to it before "
                f"reporting results"))
        for logical in t.catalogue_assets:
            if logical in built:
                continue                 # a chosen primitive, not a missing mesh
            try:
                entry = catalogue[logical]
            except KeyError:
                out.append(Finding(
                    "error", t.id,
                    f"catalogue asset {logical!r} is neither in the Isaac asset catalogue nor "
                    "declared a primitive; add it to ruibench/engines/isaac/assets.yaml rather "
                    "than naming a USD path in a card"))
                continue
            if not entry.verified:
                out.append(Finding(
                    "warning", t.id,
                    f"asset {logical!r} has not been resolved on the Isaac asset server; "
                    "run `ruibench doctor --assets` before building on it"))

    # -- scene geometry ----------------------------------------------------
    implemented = [t for t in tasks if t.status is Status.IMPLEMENTED and t.impl]
    out += _clearance_findings(implemented)
    out += _sightline_findings(implemented)

    # -- benchmark-wide totals --------------------------------------------
    live_total = sum(1 for t in tasks if t.split_role is not SplitRole.HOLDOUT)
    if live_total != TASKS_PER_SUITE * len(SUITES):
        out.append(Finding("error", "benchmark",
                           f"expected {TASKS_PER_SUITE * len(SUITES)} main-board tasks, found {live_total}"))

    # -- the freeze: what has changed since the benchmark was frozen ---------
    out += _freeze_findings()
    return out


def _freeze_findings() -> List[Finding]:
    """One warning per file (and T_ref) that differs from the frozen
    fingerprint: results on the test split taken before and after the change
    do not compare (``ruibench freeze --check``)."""
    from ..core.errors import ConfigError
    from . import freeze

    record = freeze.load()
    if record is None:
        return []
    try:
        lines = freeze.drift()
    except ConfigError:                 # an installed package: nothing to compare
        return []
    shown = [Finding("warning", f"frozen {record['name']}", line) for line in lines[:12]]
    if len(lines) > 12:
        shown.append(Finding("warning", f"frozen {record['name']}", f"... and {len(lines) - 12} more"))
    return shown


#: Seeds every implemented task is built at for the geometric checks.
CLEARANCE_SEEDS = range(10)


def _clearance_findings(tasks: Sequence[TaskSpec]) -> List[Finding]:
    """Every object a task needs picked up can be approached straight down, and
    every joint nothing drives can move.

    Built scenes, not cards: layouts are drawn per seed, so a blocked grasp is
    usually a property of some seeds only.
    """
    from ..engines.cloth import scene_problems as cloth_problems
    from .clearance import (arm_start_problems, check_scene, falls_through, joint_sweeps,
                            out_of_reach, resting_on_table)
    from .instance import build

    out: List[Finding] = []
    for t in tasks:
        blocked: dict = {}
        stuck: dict = {}
        far: dict = {}
        falls: dict = {}
        starts: dict = {}
        cloth: dict = {}
        for level in sorted(t.knob.levels):
            for seed in CLEARANCE_SEEDS:
                scene = build(t.id, level, seed).scene
                for line in cloth_problems(scene):
                    cloth.setdefault(line, []).append(f"{level}/{seed}")
                for line in check_scene(scene):
                    blocked.setdefault(line, []).append(f"{level}/{seed}")
                for line in resting_on_table(scene) + joint_sweeps(scene):
                    stuck.setdefault(line, []).append(f"{level}/{seed}")
                for line in arm_start_problems(scene):
                    starts.setdefault(line, []).append(f"{level}/{seed}")
                for line in out_of_reach(scene):
                    far.setdefault(line.split(" at ")[0], []).append(f"{level}/{seed}")
                for line in falls_through(scene):
                    falls.setdefault(line, []).append(f"{level}/{seed}")
        for line, where in cloth.items():
            out.append(Finding("error", t.id, f"a cloth's scene: {line} (at {_where(where)}); "
                                              "see engines/cloth.py (scene_problems)"))
        for line, where in falls.items():
            out.append(Finding("error", t.id,
                               f"it will fall through the table on Isaac: {line} (at {_where(where)}); "
                               "fix it in place or carry it -- see ruibench/tasks/clearance.py"))
        for line, where in blocked.items():
            out.append(Finding(
                "error", t.id,
                f"a straight-down grasp is blocked: {line} (at {_where(where)}). "
                "The hand is 20 cm long; see ruibench/tasks/clearance.py"))
        for line, where in stuck.items():
            out.append(Finding(
                "error", t.id,
                f"a joint may stick: {line} (at {_where(where)}); "
                "see ruibench/tasks/clearance.py"))
        for name, where in far.items():
            out.append(Finding(
                "error", t.id,
                f"{name} starts out of every commanded arm's reach (at {_where(where)}); "
                "see engines/robots.py (reach)"))
        for line, where in starts.items():
            out.append(Finding("error", t.id, f"an arm's start cannot be set up: {line} "
                                              f"(at {_where(where)}); see engines/base.py ArmStart"))
    return out


def _where(where: Sequence[str]) -> str:
    return ", ".join(where[:3]) + (" ..." if len(where) > 3 else "")


def _sightline_findings(tasks: Sequence[TaskSpec]) -> List[Finding]:
    """The fixed camera sees every object at the start and under the hand."""
    from . import sightlines
    from .instance import build

    out: List[Finding] = []
    for t in tasks:
        hidden: dict = {}
        for level in sorted(t.knob.levels):
            for seed in CLEARANCE_SEEDS:
                for line in sightlines.check_scene(build(t.id, level, seed).scene):
                    hidden.setdefault(line.split(":")[0] + ":" + line.split(" ", 2)[-1], []).append(
                        (f"{level}/{seed}", line))
        for key, where in hidden.items():
            more = " ..." if len(where) > 3 else ""
            out.append(Finding(
                "error", t.id,
                f"the fixed camera cannot see enough: {where[0][1]} "
                f"(at {', '.join(w for w, _ in where[:3])}{more}); see ruibench/tasks/sightlines.py"))
    return out


def errors(findings: Sequence[Finding]) -> List[Finding]:
    return [f for f in findings if f.level == "error"]
