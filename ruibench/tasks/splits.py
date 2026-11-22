"""The seed splits: which missions (task x level x seed) are for development,
which are held for testing, and which are spare.

Every task is procedural -- a seed lays its mission out -- so a split is a
list of seeds per task and level, kept in ``splits.json`` beside the cards:
15 for development, 20 for testing, 5 spare (design v0).  A seed goes in only
if its mission passes the geometric checks ``ruibench validate`` runs on
seeds 0-9 (the grasp is clear, everything is within reach and in sight,
nothing falls through the table) and, where the scripted reference has been
run on it on Isaac, the reference solved it: a mission nobody can solve would
score the task, not the model.

    python -m ruibench.tasks.splits [ISAAC_LOGS ...]     # write splits.json
    ruibench run DI4 --engine isaac --split dev          # run a split

The file says whether it is frozen (``ruibench freeze``, ``freeze.py``).
Until it is, it may be regenerated; once frozen, regenerating it is refused,
and changing it -- or the tasks, knobs, scorers or prompts it was frozen
with -- invalidates results taken on the test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from ..core.errors import ConfigError

PATH = Path(__file__).with_name("splits.json")
SIZES = (("dev", 15), ("test", 20), ("spare", 5))
SPLITS = tuple(name for name, _ in SIZES)
#: how far past the first 40 seeds to look for missions that pass
SEARCH = 80


def mission_problems(task: str, level: str, seed: int) -> List[str]:
    """What is wrong with one mission's layout, by the checks ``validate``
    runs (``research`` may add more; an empty list is a mission to keep)."""
    from . import sightlines
    from .clearance import (arm_start_problems, check_scene, falls_through, joint_sweeps,
                            out_of_reach, resting_on_table)
    from .instance import build

    try:
        scene = build(task, level, seed).scene
    except Exception as exc:                 # a seed that does not build is itself a finding
        return [f"does not build: {exc!r}"]
    return (check_scene(scene) + resting_on_table(scene) + joint_sweeps(scene)
            + arm_start_problems(scene) + out_of_reach(scene) + falls_through(scene)
            + sightlines.check_scene(scene))


def reference_results(paths: Iterable[str]) -> Dict[tuple, bool]:
    """(task, level, seed) -> did the scripted reference solve it, from Isaac
    episode logs (a seed run more than once must have been solved every time;
    an episode the harness lost says nothing)."""
    from .instance import build

    references: Dict[str, str] = {}

    def reference_name(task: str, level: str) -> str:
        if task not in references:
            inst = build(task, level, 0)
            references[task] = inst.reference_factory(inst).name
        return references[task]

    out: Dict[tuple, bool] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            log = json.loads(line)
            if log.get("engine") != "IsaacEngine" or log.get("interface") != "I0_direct" \
                    or (log.get("score") or {}).get("failure") == "harness_error":
                continue
            # the reference itself -- not a negative control or a stress agent
            if log.get("agent") != reference_name(log["task_id"], log["level"]):
                continue
            key = (log["task_id"], log["level"], int(log["seed"]))
            out[key] = out.get(key, True) and bool(log["score"].get("success"))
    return out


def make(tasks: Optional[Sequence[str]] = None, solved: Mapping[tuple, bool] = (),
         progress=None) -> dict:
    """The splits: per task and level, the first seeds that pass, in order,
    dealt out dev, test, spare.  ``solved``: what the Isaac reference did on
    the seeds it was run on."""
    from .loader import main_board

    solved = dict(solved)
    need = sum(n for _, n in SIZES)
    out: dict = {"frozen": False,
                 "sizes": dict(SIZES),
                 "criteria": "the geometric checks of `ruibench validate`, and the scripted "
                             "reference solving the mission on Isaac where it was run",
                 "tasks": {}}
    for spec in main_board():
        if tasks and spec.id not in tasks:
            continue
        levels = {}
        for level in sorted(spec.knob.levels):
            kept, dropped, verified = [], {}, 0
            for seed in range(SEARCH):
                if len(kept) == need:
                    break
                problems = mission_problems(spec.id, level, seed)
                if solved.get((spec.id, level, seed)) is False:
                    problems.append("the scripted reference failed it on Isaac")
                if problems:
                    dropped[str(seed)] = problems[0]
                    continue
                kept.append(seed)
                verified += (spec.id, level, seed) in solved
            if len(kept) < need:
                raise ConfigError(f"{spec.id} {level}: only {len(kept)} good missions in "
                                  f"seeds 0-{SEARCH - 1}")
            dealt, i = {}, 0
            for name, n in SIZES:
                dealt[name] = kept[i:i + n]
                i += n
            levels[level] = {**dealt, "isaac_verified": verified,
                             "left_out": dropped}
            if progress:
                progress(f"{spec.id} {level}: {verified}/{need} solved on Isaac, "
                         f"{len(dropped)} left out {sorted(dropped, key=int)}")
        out["tasks"][spec.id] = levels
    return out


def load(path: Path = PATH) -> dict:
    if not path.exists():
        raise ConfigError(f"no seed splits at {path}; `python -m ruibench.tasks.splits` writes them")
    return json.loads(path.read_text(encoding="utf-8"))


def seeds(task: str, level: str, split: str, data: Optional[dict] = None) -> List[int]:
    if split not in SPLITS:
        raise ConfigError(f"unknown split {split!r}; have {', '.join(SPLITS)}")
    data = data or load()
    try:
        return list(data["tasks"][task][level][split])
    except KeyError:
        raise ConfigError(f"no {split} seeds for {task} {level} in the splits file") from None


def main(argv: Sequence[str]) -> None:
    from ..core.errors import FrozenAssetError

    if PATH.exists() and load(PATH).get("frozen") and "--force" not in argv:
        raise FrozenAssetError(f"the splits are frozen ({load(PATH)['frozen']}): a new split voids "
                               "results on the test split -- freeze a new version instead, or "
                               "pass --force")
    argv = [a for a in argv if a != "--force"]
    solved = reference_results(argv)
    data = make(solved=solved, progress=print)
    PATH.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {PATH}")


if __name__ == "__main__":
    main(sys.argv[1:])
