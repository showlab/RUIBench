"""The tool-solvability table: interface x task x level.

For each cell, the task's scripted reference decides and the interface's tools
express (``ruikit.diagnostics.reference``).  A failure means the interface cannot
do the task *at all*, and points at the tool that is missing: interfaces may
differ in how well they do a task, but none should be completely unable to.

    python -m ruikit.diagnostics.coverage --tasks WK1 DS2 --levels easy hard --seeds 0
    .venv/bin/python -m ruikit.diagnostics.coverage --engine isaac --workers 4 --out runs/x/cells.jsonl
    .venv/bin/python -m ruikit.diagnostics.coverage --engine isaac --seeds test:1 --think-seconds 8 \
        --gpus 3,3,3,3 --out runs/check/cells.jsonl     # the check before the experiments

On Isaac the cells run in worker processes, as the benchmark's own runs do
(``ruibench/runner/isolated.py``): one simulator per process, on its own GPU,
for one cell (``--per-worker 1``, the default: a reused stage keeps state
from the episodes before it, so a cell's outcome would depend on the order;
``--per-worker N`` keeps it for N cells of one task and level, which share a
scene structure); a worker that crashes or hangs is replaced and the cell it was on
tried again, and after ``MAX_ATTEMPTS`` recorded as a harness error -- never as
the interface failing.  As the batch runner does (``ruikit.experiments.batch``): each
worker sees only its own card (``CUDA_VISIBLE_DEVICES``; the cloth of CR6 is
solved on the first card a process sees), Isaac starts are ``LAUNCH_GAP``
apart across every runner on the machine, none while the load is over
``MAX_LOAD`` of the cores, and none without a vision server.  A rerun into the
same ``--out`` goes on where the last one stopped.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from ruikit.diagnostics.reference import EXPRESSORS, GROUNDED, style_of
from ruikit.experiments.preflight import vision_check
from ruikit.experiments.provenance import stamp
from ruikit.report.clips import RECORD_EVERY, RECORD_SIZE, _film, clip_dir

#: the main table's interfaces: each action abstraction one action a reply, and a program
BASE_PROFILES = ("rui.joint/v1", "rui.delta/v1", "rui.step/v1", "rui.pose/v1", "rui.point/v1",
                 "rui.skill/v1", "rui.code/v1")


@dataclass
class Cell:
    task: str
    level: str
    seed: int
    profile: str
    success: bool
    progress: float
    reason: str
    tool_calls: int
    sim_seconds: float
    wall_seconds: float
    unexpressed: List[str]
    error: str = ""
    leaks: int = 0
    replies: int = 0
    think_seconds: float = 0.0          # each reply's model time, charged on the clock
    engine: str = ""
    #: the milestones reached, in order, and the task-level's limits -- how far a
    #: failure got, and whether the reference's own replies x T_ref would fit
    milestones: List[str] = field(default_factory=list)
    reply_limit: int = 0
    sim_limit: float = 0.0
    #: what ran it (``ruikit.experiments.provenance``)
    provenance: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return dict(self.__dict__)


def run_cell(task: str, level: str, seed: int, profile: str, engine="mock",
             max_tool_calls: int = 6000, log_dir: Optional[str] = None,
             record: Optional[str] = None, oracle_coordinates: bool = False,
             think_seconds: float = 0.0, reference: Optional[str] = None) -> Cell:
    """One cell.  ``engine``: a name, or an engine to reuse (Isaac: one per
    process); a scene that needs a new one raises ``SceneRebuildRequired``.
    ``record``: film it into its own folder under this one (``clip_dir``):
    frames, the pictures the agent was shown, the logs, ``video.mp4`` and
    ``clip.json``.  The agent gets every coordinate from the interface's
    perception tools (``grounded``); ``oracle_coordinates``: from the
    simulator, to ask only whether the tools can act the task.
    ``think_seconds``: every reply first costs this much simulated time (the
    model thinking while the world goes on); 0 stops the clock between replies."""
    from ruibench.core.errors import SceneRebuildRequired
    from ruikit.env import Budget, Episode

    t0 = time.time()
    style = style_of(profile)
    engine_name = engine if isinstance(engine, str) else type(engine).__name__.replace("Engine", "").lower()
    film = {}
    if record:
        clip = clip_dir(record, task, level, seed, profile)
        if clip.exists():
            shutil.rmtree(clip)                  # a retried clip starts clean
        film = dict(record_dir=str(clip / "frames"), record_every=RECORD_EVERY,
                    record_size=RECORD_SIZE, image_dir=str(clip / "images"), play_out=True,
                    agent=f"reference through {profile}")
        log_dir = str(clip)
    clock = dict(clock="fixed", think_seconds=think_seconds, think_per="reply") if think_seconds else {}
    try:
        # the task's limits end it, not the real time the cloth or a long script takes
        ep = Episode(task, level=level, seed=seed, profile=profile, engine=engine,
                     budget=Budget(max_tool_calls=max_tool_calls, max_wall_seconds=86400.0),
                     log_dir=log_dir, end_at_success=True, **clock, **film)
    except SceneRebuildRequired:
        raise
    except Exception as exc:
        return Cell(task, level, seed, profile, False, 0.0, "setup_error", 0, 0.0,
                    time.time() - t0, [], f"{type(exc).__name__}: {exc}",
                    think_seconds=think_seconds, engine=engine_name)
    ex = (EXPRESSORS if oracle_coordinates else GROUNDED)[style]()
    try:
        if reference:                            # another way than the task's own (``alter_reference``)
            from ruikit.diagnostics.alter_reference import install

            install(reference, ep.inst)
        out = ex.run(ep)
        reason = out.reason
        if not out.success and out.score.get("failure"):
            # the run stopped for one reason and was scored a failure for another
            # (a violation, a milestone short, a physics flag): say both
            why = out.score["failure"]
            if out.score.get("violations"):
                why += ":" + ",".join(out.score["violations"])
            if out.score.get("flags"):
                why += ":" + ",".join(out.score["flags"])
            reason = f"{reason}/{why}"
        hit = (out.score or {}).get("milestones_hit") or {}
        cell = Cell(task, level, seed, profile, out.success, out.progress, reason,
                    out.metrics["tool_calls"], out.metrics["sim_seconds"], time.time() - t0,
                    ex.unexpressed[:5], leaks=out.metrics.get("leaks", 0),
                    replies=out.metrics.get("replies", 0), think_seconds=think_seconds,
                    engine=engine_name,
                    milestones=[m for m, _ in sorted(hit.items(), key=lambda kv: kv[1])],
                    reply_limit=int(ep.budget.max_replies or 0),
                    sim_limit=float(ep.budget.max_sim_seconds or 0.0))
    except SceneRebuildRequired:
        raise
    except Exception as exc:
        return Cell(task, level, seed, profile, False, 0.0, "harness_error", ep.tool_calls,
                    ep.sim_time, time.time() - t0, ex.unexpressed[:5],
                    "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)[-3:]),
                    replies=ep.replies, think_seconds=think_seconds, engine=engine_name)
    finally:
        ep.close()
    if record:
        _film(clip_dir(record, task, level, seed, profile), cell, getattr(ex, "notes", []),
              ep.budget.as_dict())
    return cell


def table(cells: Sequence[Cell], profiles: Sequence[str] = BASE_PROFILES) -> str:
    """A markdown table: one row per task x level, one column per interface."""
    rows: Dict[tuple, Dict[str, List[Cell]]] = {}
    for c in cells:
        rows.setdefault((c.task, c.level), {}).setdefault(c.profile, []).append(c)
    head = "| task | level | " + " | ".join(p.split("/")[0] for p in profiles) + " |"
    out = [head, "|" + "---|" * (len(profiles) + 2)]
    for (task, level), by in rows.items():
        marks = []
        for p in profiles:
            cs = by.get(p, [])
            if not cs:
                marks.append("")
                continue
            ok = sum(c.success for c in cs)
            lost = sum(c.reason == "harness_error" for c in cs)
            marks.append(f"{ok}/{len(cs)}" + ("" if ok == len(cs) else " ✗")
                         + (f" ({lost} lost)" if lost else ""))
        out.append(f"| {task} | {level} | " + " | ".join(marks) + " |")
    return "\n".join(out)


def _say(cell: Cell) -> str:
    flag = "ok " if cell.success else "-- "
    return (f"{flag}{cell.task} {cell.level:4s} s{cell.seed} {cell.profile:13s} {cell.reason:13s} "
            f"replies={cell.replies:4d} sim={cell.sim_seconds:7.1f}s "
            f"wall={cell.wall_seconds:5.1f}s {cell.unexpressed[:1]} "
            f"{cell.error.strip().splitlines()[-1] if cell.error else ''}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    from ruibench.tasks.instance import implemented
    from ruikit.diagnostics.coverage_workers import (
        _worker,
        idle_gpus,
        keys_for,
        run_isolated,
    )

    from ruibench import options as O
    from ruikit import cli_options as K

    ap = argparse.ArgumentParser(prog="ruikit coverage", description=__doc__.split("\n\n")[0])
    O.add_tasks(ap)
    O.add_levels(ap)
    O.add_seeds(ap, "0")
    K.add_interfaces(ap, "the interfaces, by the paper's names (with the aids) or by id (default "
                         "the bare per-action ones and the program over the pose)")
    O.add_engine(ap, "mock")
    O.add_clock(ap, "paused")
    O.add_gpus(ap)
    ap.add_argument("--out", "-o", default=None, help="write every cell as JSON lines here")
    ap.add_argument("--log-dir", default=None,
                    help="write each episode's logs here (ruibench rescore reads *.episode.jsonl)")
    ap.add_argument("--workers", type=int, default=4,
                    help="Isaac: at most this many simulators at once (one per GPU)")
    ap.add_argument("--per-worker", type=int, default=None,
                    help="Isaac: cells a simulator runs before a fresh one (default 1: a reused "
                         "stage keeps state from the episodes before)")
    ap.add_argument("--record", default=None,
                    help="film every cell into its own folder here (the RUIKit showcase)")
    ap.add_argument("--oracle-coordinates", action="store_true",
                    help="send the reference's own coordinates instead of estimating them "
                         "with the interface's perception tools (does its acting suffice?)")
    ap.add_argument("--think-seconds", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--reference", default=None,
                    help="an alternative to the task's reference "
                         "(ruikit.diagnostics.alter_reference); give its one task in --task")
    ap.add_argument("--worker", default=None, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        _worker(args.worker)
    try:
        tasks = O.expand_tasks(O.tasks_of(args), list(implemented())) if O.tasks_of(args) \
            else list(implemented())
        clock = O.clock_of(args)
        args.profiles = K.interface_ids(args.interfaces) or list(BASE_PROFILES)
    except ValueError as exc:
        ap.error(str(exc))
    if args.think_seconds is None:
        if clock == "measured":
            ap.error("coverage runs the reference, which does not think: --clock paused or <N>s")
        args.think_seconds = 0.0 if clock == "paused" else float(clock[:-1])
    args.levels, args.seeds = O.levels_of(args), [O.seeds_of(args)]
    keys = keys_for(tasks, args.levels, args.seeds, args.profiles)
    if args.engine == "isaac":
        if not args.out:
            ap.error("--engine isaac needs --out (workers append to it; a rerun resumes)")
        why = vision_check(args.profiles)
        if why:
            print(f"cannot run on Isaac: {why}", file=sys.stderr)
            return 2
        gpus = O.gpus_of(args) or idle_gpus()
        gpus = gpus[:max(1, args.workers)]
        if not gpus:
            print("no GPU with room", file=sys.stderr)
            return 1
        print(f"{len(keys)} cells on GPUs {gpus}", flush=True)
        cells = run_isolated(keys, args.engine, args.out, gpus, args.log_dir,
                             say=lambda s: print(s, flush=True), record=args.record,
                             oracle_coordinates=args.oracle_coordinates,
                             think_seconds=args.think_seconds, reference=args.reference,
                             **({"per_worker": args.per_worker} if args.per_worker else {}))
    else:
        cells = []
        for task, level, seed, profile in keys:
            cell = run_cell(task, level, seed, profile, args.engine, log_dir=args.log_dir,
                            record=args.record, oracle_coordinates=args.oracle_coordinates,
                            think_seconds=args.think_seconds, reference=args.reference)
            cell.provenance = stamp(1, None)         # an engine of its own, in this process
            cells.append(cell)
            print(_say(cell), flush=True)
            if args.out:
                with open(args.out, "a") as fh:
                    fh.write(json.dumps(cell.as_dict()) + "\n")
    print()
    print(table(cells, args.profiles))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
