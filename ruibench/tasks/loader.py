"""Reading task cards.

One YAML file per suite: six tasks on the board and one held-out template.  The loader is strict: an unknown
field, a missing knob level or a card that breaks a design rule fails here,
loudly, rather than halfway through a 100-hour run.
"""

from __future__ import annotations

import functools
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Tuple

import yaml

from ..core.errors import ConfigError
from .spec import SUITES, AssetClass, Knob, SplitRole, Status, TaskSpec

CARDS_DIR = Path(__file__).parent / "cards"

_ALLOWED = {
    "id", "name", "summary", "knob", "success", "milestones", "confounds",
    "gap", "gap_filled", "blind", "abilities", "formerly", "asset_class", "base_env", "assets", "catalogue_assets",
    "arms", "status",
    "split_role", "impl", "negative_controls", "failure_example", "risks", "notes", "source",
    "hint", "secrets",
}


def _task_from_dict(suite: str, d: Mapping) -> TaskSpec:
    unknown = set(d) - _ALLOWED
    if unknown:
        raise ConfigError(f"{d.get('id', '?')}: unknown card fields {sorted(unknown)}")
    for req in ("id", "name", "summary", "knob", "success"):
        if req not in d:
            raise ConfigError(f"{d.get('id', '?')}: card is missing {req!r}")

    k = d["knob"]
    levels = dict(k.get("levels", {}))
    if set(levels) != {"easy", "hard"}:
        raise ConfigError(
            f"{d['id']}: a knob must have exactly the levels easy and hard, got {sorted(levels)}"
        )
    knob = Knob(field=k["field"], description=k.get("description", ""), levels=levels,
                unit=k.get("unit", ""), calibrated=bool(k.get("calibrated", False)))

    return TaskSpec(
        id=d["id"], suite=suite, name=d["name"], summary=d["summary"], knob=knob,
        success=d["success"],
        milestones=tuple(d.get("milestones", ())),
        confounds=tuple(d.get("confounds", ())),
        gap=d.get("gap", "") or "", gap_filled=d.get("gap_filled", "") or "",
        blind=d.get("blind", "") or "", abilities=dict(d.get("abilities") or {}),
        formerly=d.get("formerly", "") or "",
        asset_class=AssetClass(d.get("asset_class", "A")),
        base_env=d.get("base_env", ""), assets=tuple(d.get("assets", ())),
        catalogue_assets=tuple(d.get("catalogue_assets", ())),
        arms=int(d.get("arms", 1)),
        status=Status(d.get("status", "planned")),
        split_role=SplitRole(d.get("split_role", "main")),
        impl=d.get("impl", ""),
        negative_controls=tuple(d.get("negative_controls", ())),
        failure_example=d.get("failure_example", "") or "",
        risks=tuple(d.get("risks", ())), notes=d.get("notes", "") or "",
        source=dict(d.get("source") or {}),
        hint=" ".join(str(d.get("hint") or "").split()),
        secrets=" ".join(str(d.get("secrets") or "").split()),
    )


def load_suite(path: Path) -> List[TaskSpec]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    suite = data["suite"]
    if suite not in SUITES:
        raise ConfigError(f"{path.name}: unknown suite {suite!r}; expected one of {sorted(SUITES)}")
    return [_task_from_dict(suite, t) for t in data.get("tasks", [])]


@functools.lru_cache(maxsize=1)
def load_all(cards_dir: Optional[str] = None) -> Tuple[TaskSpec, ...]:
    """Every card, in paper order (suite order, then task id)."""
    directory = Path(cards_dir) if cards_dir else CARDS_DIR
    tasks: List[TaskSpec] = []
    for suite in SUITES:
        path = directory / f"{suite}.yaml"
        if not path.exists():
            raise ConfigError(f"missing task cards for suite {suite!r} at {path}")
        tasks.extend(load_suite(path))
    return tuple(tasks)


def get(task_id: str) -> TaskSpec:
    for t in load_all():
        if t.id == task_id:
            return t
    raise ConfigError(f"unknown task {task_id!r}; try `ruibench tasks`")


def suite_key(suite: str) -> str:
    """A suite by its key ("knowledge"), its prefix ("WK") or its name ("World
    Knowledge"), any case."""
    want = suite.strip().lower()
    for key, (prefix, name, _) in SUITES.items():
        if want in (key, prefix.lower(), name.lower()):
            return key
    raise ConfigError(f"unknown suite {suite!r}; have "
                      + ", ".join(f"{p} ({k})" for k, (p, _, _) in SUITES.items()))


def by_suite(suite: Optional[str] = None) -> Dict[str, List[TaskSpec]]:
    out: Dict[str, List[TaskSpec]] = {s: [] for s in SUITES}
    for t in load_all():
        out[t.suite].append(t)
    if not suite:
        return out
    key = suite_key(suite)
    return {key: out[key]}


def main_board() -> Tuple[TaskSpec, ...]:
    """The 24 tasks that carry the leaderboard (dev + main, not held out)."""
    return tuple(t for t in load_all() if t.split_role is not SplitRole.HOLDOUT)


def held_out() -> Tuple[TaskSpec, ...]:
    """The four seventh tasks: designed, never implemented, never tuned against."""
    return tuple(t for t in load_all() if t.split_role is SplitRole.HOLDOUT)
