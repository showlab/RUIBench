"""The options every command shares, spelled one way.

``ruibench`` and ``ruikit`` choose what to run with the same words:

    --task WK1 CL2 | --task CR | --task all     task ids, suites (WK CL DS DI CR), or every task
    --level easy hard                            the levels
    --seeds dev | test | spare | dev:N | test:N | 0-4 | 15,16
                                                 a split's seeds (the first N), a range, a list
    --clock paused | <N>s | measured             how thinking counts: not at all, N seconds a
                                                 reply (the paper's main setting: 8s), as it took
    --engine isaac | mock                        Isaac Sim, or the mock (no physics: checks only)
    --gpus 3 | 3,4                               the cards Isaac may use
    --out DIR                                    where the run goes

Older spellings (``--tasks``, ``--levels``, ``--split``, ``--gpu``, the clocks
``zero_latency`` / ``controlled`` / ``zero`` / ``fixed``) are still read and
not shown.  The interfaces, agents and models are RUIKit's
(``ruikit.cli_options``).
"""

from __future__ import annotations

import argparse
import re
from typing import List, Optional, Sequence, Tuple

LEVELS = ("easy", "hard")
ENGINES = ("isaac", "mock")
SPLITS = ("dev", "test", "spare")
SUITES = ("WK", "CL", "DS", "DI", "CR")


def add_tasks(p: argparse.ArgumentParser, required: bool = False, positional: bool = False) -> None:
    """``--task``: task ids, suites or ``all`` (several may follow, or the flag repeat)."""
    help_ = "task ids (WK1), suites (WK CL DS DI CR) or all"
    if positional:
        p.add_argument("task_args", nargs="*", metavar="TASK", help=help_)
    p.add_argument("--task", dest="tasks", nargs="+", action="extend", default=None,
                   required=required and not positional, help=help_)
    p.add_argument("--tasks", dest="tasks", nargs="+", action="extend", help=argparse.SUPPRESS)


def add_levels(p: argparse.ArgumentParser, default: Sequence[str] = LEVELS) -> None:
    p.add_argument("--level", dest="levels", nargs="+", action="extend", choices=LEVELS,
                   default=None, help=f"easy, hard (default: {' '.join(default)})")
    p.add_argument("--levels", dest="levels", nargs="+", action="extend", choices=LEVELS,
                   help=argparse.SUPPRESS)
    p.set_defaults(levels_default=list(default))


def add_seeds(p: argparse.ArgumentParser, default: Optional[str]) -> None:
    p.add_argument("--seeds", default=None,
                   help="dev | test | spare (a split's seeds), dev:N or test:N (its first N), "
                        f"0-4, 15,16{f' (default {default})' if default else ''}")
    p.add_argument("--split", choices=SPLITS, default=None, help=argparse.SUPPRESS)
    p.set_defaults(seeds_default=default)


def add_engine(p: argparse.ArgumentParser, default: str = "isaac") -> None:
    p.add_argument("--engine", default=default, choices=ENGINES,
                   help=f"isaac (Isaac Sim) or mock (no physics: for checks; its results are "
                        f"never scores) (default {default})")


def add_gpus(p: argparse.ArgumentParser) -> None:
    p.add_argument("--gpus", default="", help="Isaac: only these cards, e.g. 3 or 3,4")
    p.add_argument("--gpu", dest="gpus", type=str, help=argparse.SUPPRESS)


def add_clock(p: argparse.ArgumentParser, default: str) -> None:
    p.add_argument("--clock", default=None,
                   help=f"paused (thinking costs nothing), <N>s (each reply costs N seconds; "
                        f"the paper's main setting is 8s) or measured (default {default})")
    p.set_defaults(clock_default=default)


def add_out(p: argparse.ArgumentParser, default: Optional[str], what: str = "the run's folder") -> None:
    p.add_argument("--out", "-o", default=default,
                   help=what + (f" (default {default})" if default else ""))


# ---------------------------------------------------------------- reading them
def tasks_of(args) -> Optional[List[str]]:
    """The tasks asked for (``--task`` and any positional ones), or None."""
    names = list(getattr(args, "task_args", None) or []) + list(args.tasks or [])
    return names or None


def levels_of(args) -> List[str]:
    return list(dict.fromkeys(args.levels or args.levels_default))


def seeds_of(args) -> Optional[str]:
    """The seed spec: ``--seeds``, else the older ``--split``, else the default."""
    return args.seeds or args.split or args.seeds_default


def gpus_of(args) -> List[int]:
    return [int(g) for g in str(args.gpus or "").split(",") if g.strip()]


#: older clock names -> today's
OLD_CLOCKS = {"zero_latency": "paused", "zero": "paused", "measured": "measured"}


def clock_of(args) -> str:
    """The clock in today's words: ``paused``, ``<N>s`` or ``measured``."""
    return normal_clock(args.clock or args.clock_default)


def normal_clock(spec: str, latency_seconds: Optional[float] = None) -> str:
    spec = str(spec).strip()
    if spec in OLD_CLOCKS:
        return OLD_CLOCKS[spec]
    if spec in ("controlled", "fixed"):
        return f"{latency_seconds if latency_seconds is not None else 8:g}s"
    if spec == "paused" or re.fullmatch(r"\d+(\.\d+)?s", spec):
        return spec
    raise ValueError(f"no clock {spec!r}: paused, <N>s (8s) or measured")


def engine_clock(clock: str) -> Tuple[str, float]:
    """A clock as the episode runner reads it: (``zero_latency``, 0),
    (``controlled``, N) or (``measured``, 0)."""
    clock = normal_clock(clock)
    if clock == "paused":
        return "zero_latency", 0.0
    if clock == "measured":
        return "measured", 0.0
    return "controlled", float(clock[:-1])


def parse_seeds(spec: str, task: str, level: str, data: Optional[dict] = None) -> List[int]:
    """``dev``, ``test`` or ``spare`` (the split's seeds), ``dev:3`` (its first
    three), ``0-4``, ``0,3,7``."""
    from ruibench.tasks import splits

    name, _, n = spec.partition(":")
    if name in SPLITS:
        seeds = splits.seeds(task, level, name, data)
        return seeds[:int(n)] if n else seeds
    out: List[int] = []
    for part in spec.split(","):
        a, _, b = part.strip().partition("-")
        out += list(range(int(a), int(b) + 1)) if b else [int(a)]
    return out


def expand_tasks(names: Sequence[str], ids: Sequence[str]) -> List[str]:
    """Task ids, a suite standing for its tasks (``CR``: CR1 ... CR6), ``all``
    for every one of ``ids``."""
    out: List[str] = []
    for name in names:
        if name.lower() == "all":
            hits = list(ids)
        else:
            hits = [t for t in ids if t == name or (name.isalpha() and t.rstrip("0123456789") == name)]
        if not hits:
            raise ValueError(f"no task {name!r}: a task id (WK1), a suite (WK, CL, DS, DI, CR) or all")
        out += [t for t in hits if t not in out]
    return out
