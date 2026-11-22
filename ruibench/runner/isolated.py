"""Running episodes in worker processes that are allowed to die.

Why this exists: the Isaac stack can abort with a native heap corruption that
does not reproduce -- the same scene crashed once and ran cleanly the next time.
A benchmark that runs thousands of episodes cannot let one of those take the
whole run down, and must never let one be scored as a model failure.

So a run is a queue of cells.  A worker process takes cells off the queue,
keeping one simulator alive across many episodes (starting Isaac is the
expensive part), and appends each finished episode to the log the moment it
completes.  If the worker dies, the parent starts a new one for whatever is
left.  The cell that was in flight is retried; after ``max_attempts`` it is
recorded as a ``harness_error`` -- a fact about our infrastructure, reported in
its own column, excluded from every success rate.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..core.config import RunConfig
from . import heartbeat
from .log import EpisodeLog, read_jsonl, write_jsonl

Cell = Tuple[str, str, int, str]            # task, level, seed, condition

#: A worker that needs a different scene structure exits with this code.  It is
#: not a crash: the parent starts a fresh worker and charges no retry.
REBUILD_EXIT = 75


@dataclass
class IsolatedRun:
    cells: List[Cell]
    config: RunConfig
    agent: str = "reference"                # "reference" or a negative-control name
    max_attempts: int = 3
    python: str = sys.executable
    env: Dict[str, str] = field(default_factory=dict)
    #: Watchdog.  A worker that stops making progress is killed and its
    #: in-flight episode retried -- a hang becomes a bounded, recoverable failure
    #: instead of a run that silently stops.
    startup_seconds: float = 1200.0         # before the first heartbeat (a cold Isaac start)
    stall_seconds: float = 180.0            # no progress while simulating
    agent_stall_seconds: float = 900.0      # no progress while waiting on a model API
    log_dir: Optional[str] = None           # where worker stdout/stderr go


def _key(task: str, level: str, seed: int, condition: str) -> str:
    return f"{task}|{level}|{seed}|{condition}"


def run_isolated(job: IsolatedRun, out: str | Path,
                 progress: Optional[Callable[[str], None]] = None) -> List[dict]:
    """Run every cell, surviving worker crashes.  Returns the logs as dicts."""
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    attempts: Dict[str, int] = {}
    done: Dict[str, dict] = {}
    if out.exists():                        # resume: never redo a finished cell
        for log in read_jsonl(out):
            done[_key(log["task_id"], log["level"], log["seed"], log.get("condition", "main"))] = log

    say = progress or (lambda s: None)
    worker_no = [0]
    while True:
        todo = [c for c in job.cells if _key(*c) not in done]
        if not todo:
            break
        in_flight = _key(*todo[0])
        attempts[in_flight] = attempts.get(in_flight, 0) + 1
        if attempts[in_flight] > job.max_attempts:
            record = _harness_error(todo[0], job, attempts[in_flight] - 1)
            write_jsonl(out, [record], append=True)
            done[in_flight] = record.as_dict()
            say(f"gave up on {in_flight} after {job.max_attempts} crashed attempts")
            continue

        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
            json.dump({"cells": todo, "config": job.config.as_dict(),
                       "agent": job.agent, "out": str(out)}, fh)
            spec = fh.name
        worker_no[0] += 1
        log_dir = Path(job.log_dir) if job.log_dir else out.parent / f"{out.stem}.workers"
        log_dir.mkdir(parents=True, exist_ok=True)
        worker_log = log_dir / f"worker-{worker_no[0]:03d}.log"
        beat_file = log_dir / f"worker-{worker_no[0]:03d}.heartbeat"
        env = dict(os.environ, **job.env)
        env[heartbeat.ENV_VAR] = str(beat_file)
        started = time.time()
        with open(worker_log, "w") as log_fh:
            proc = subprocess.Popen([job.python, "-m", "ruibench.runner.isolated", spec],
                                    env=env, stdout=log_fh, stderr=subprocess.STDOUT)
            stalled = _watch(proc, beat_file, started, job, say)
        os.unlink(spec)
        returncode = proc.returncode

        finished = 0
        for log in read_jsonl(out) if out.exists() else []:
            key = _key(log["task_id"], log["level"], log["seed"], log.get("condition", "main"))
            if key not in done:
                done[key] = log
                finished += 1
                mark = "ok " if log["score"].get("success") else "-- "
                say(f"{mark} {log['task_id']} {log['level']:4s} seed={log['seed']} "
                    f"{log['score'].get('failure')}")
        if stalled:
            say(f"worker stalled ({stalled}) after {finished} episode(s) -- killed; "
                f"see {worker_log}")
        elif returncode == REBUILD_EXIT:
            # a scene change, not a crash: the in-flight cell has not been tried
            attempts[in_flight] -= 1
            say(f"new scene structure after {finished} episode(s); starting a fresh worker")
        elif returncode != 0:
            say(f"worker died (exit {returncode}) after {finished} episode(s) in "
                f"{time.time() - started:.0f} s: {_last_error(worker_log)} -- restarting")
    return [done[_key(*c)] for c in job.cells]


#: How often a running worker's progress is printed.  A long episode (six
#: pick-and-places) takes minutes; silence for that long reads as a hang.
PROGRESS_EVERY = 30.0


def _watch(proc: subprocess.Popen, beat_file: Path, started: float, job: IsolatedRun,
           say: Callable[[str], None]) -> Optional[str]:
    """Wait for the worker; kill it if it stops making progress.

    Returns a description of the stall, or None if the worker ended on its own.
    """
    last_report = time.time()
    while True:
        try:
            proc.wait(timeout=2.0)
            return None
        except subprocess.TimeoutExpired:
            pass
        beat = heartbeat.read(str(beat_file))
        now = time.time()
        if now - last_report >= PROGRESS_EVERY:
            last_report = now
            say("   ... " + _describe(beat, now - started))
        if beat is None:
            if now - started > job.startup_seconds:
                reason = f"no heartbeat {job.startup_seconds:.0f} s after start"
                break
            continue
        limit = job.agent_stall_seconds if beat.get("phase") == heartbeat.AGENT else job.stall_seconds
        if beat["age"] > limit:
            reason = f"no progress for {beat['age']:.0f} s in phase {beat.get('phase')}"
            break
    proc.kill()
    proc.wait()
    return reason


def _describe(beat: Optional[dict], elapsed: float) -> str:
    if beat is None:
        return f"starting the simulator ({elapsed:.0f} s)"
    where = beat.get("task") or beat.get("finished") or beat.get("stage") or "working"
    parts = [f"{where}"]
    if beat.get("seed") is not None:
        parts.append(f"seed {beat['seed']}")
    if beat.get("turn") is not None:
        parts.append(f"turn {beat['turn']}")
    if beat.get("sim_time") is not None:
        parts.append(f"sim {beat['sim_time']} s")
    parts.append(f"{beat.get('phase', '?')}, last progress {beat['age']:.0f} s ago")
    return " · ".join(parts)


def _last_error(worker_log: Path) -> str:
    """The most informative line of a dead worker's log."""
    try:
        lines = worker_log.read_text(errors="replace").strip().splitlines()
    except OSError:
        return "?"
    for line in reversed(lines):
        if "Error" in line or "Exception" in line:
            return line.strip()[:160]
    return (lines[-1].strip()[:160] if lines else "?")


def _harness_error(cell: Cell, job: IsolatedRun, attempts: int) -> EpisodeLog:
    task, level, seed, condition = cell
    log = EpisodeLog(run_id=job.config.run_id, task_id=task, level=level, seed=seed,
                     interface=job.config.interface, agent=job.agent,
                     engine=job.config.engine, clock=job.config.clock,
                     config=job.config.as_dict(), condition=condition)
    log.stamp("start")
    log.score = {"task_id": task, "level": level, "success": False, "progress": 0.0,
                 "failure": "harness_error", "milestones_hit": {}, "violations": [],
                 "illegal_moves": [], "metrics": {"crashed_attempts": attempts},
                 "coverage_gap": None}
    log.stamp("end")
    return log


# ---------------------------------------------------------------- stopping
#: argv (after the interpreter) of the processes ``ruibench stop`` ends
_STOPPABLE = re.compile(
    r"^(-m )?ruibench(\.cli)? (run|doctor|gallery)\b"
    r"|^-m ruibench\.(runner\.isolated|engines\.isaac\.checks|reports\.gallery)\b")


def is_stoppable(argv: Sequence[str]) -> bool:
    """Is this command line one of ours?  Only an interpreter or the console
    script counts: a shell whose command line merely mentions ``ruibench run``
    is not one (``pkill -f`` once killed the shell that ran it)."""
    if not argv:
        return False
    exe = os.path.basename(argv[0])
    if exe == "ruibench":
        rest = "ruibench " + " ".join(argv[1:])
    elif exe.startswith("python"):
        args = [a for a in argv[1:] if a != "-u"]
        if args and os.path.basename(args[0]) == "ruibench":     # python .../bin/ruibench run
            rest = "ruibench " + " ".join(args[1:])
        else:
            rest = " ".join(args)
    else:
        return False
    return bool(_STOPPABLE.search(rest))


def stop_workers(root: Optional[Path] = None) -> List[int]:
    """Kill the RUIBench runs, workers, doctors and Isaac children whose
    working directory is under ``root`` -- this checkout by default,
    ``Path("/")`` for every one on this account -- except this process and
    its ancestors.  A second clone on the same account (a fresh-install
    test, a collaborator's copy) is someone else's work.  Linux only (reads
    /proc)."""
    import signal

    proc = Path("/proc")
    if not proc.is_dir():
        return []
    root = (root or Path(__file__).resolve().parents[2]).resolve()

    def started_in(entry: Path) -> bool:
        try:
            cwd = (entry / "cwd").resolve()
        except OSError:
            return False
        return cwd == root or root in cwd.parents
    spared = set()
    pid = os.getpid()
    while pid > 1:
        spared.add(pid)
        try:
            pid = int((proc / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    stopped = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) in spared:
            continue
        try:
            argv = (entry / "cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if is_stoppable([a for a in argv if a]) and started_in(entry):
            try:
                os.kill(int(entry.name), signal.SIGKILL)
                stopped.append(int(entry.name))
            except OSError:
                pass
    return stopped


# ---------------------------------------------------------------- worker side
def _worker(spec_path: str) -> None:
    from ..core.config import BudgetConfig
    from ..core.errors import SceneRebuildRequired
    from ..policies import build_agent
    from .harness import RunPlan, run_plan

    spec = json.loads(Path(spec_path).read_text())
    raw = dict(spec["config"])
    raw["budget"] = BudgetConfig(**raw["budget"])
    raw["cameras"] = tuple(raw["cameras"])
    config = RunConfig(**raw)
    agent = spec["agent"]

    def factory(inst):
        return build_agent(agent, inst)

    crash_on = os.environ.get("RUIBENCH_TEST_CRASH_ON")   # test hook: "TASK|level|seed"
    hang_on = os.environ.get("RUIBENCH_TEST_HANG_ON")     # test hook: "TASK|level|seed"
    heartbeat.beat(heartbeat.SIM, stage="worker started")
    for task, level, seed, condition in spec["cells"]:
        key = f"{task}|{level}|{seed}"
        if crash_on and crash_on == key:
            os.abort()                        # stand-in for a native heap corruption
        if hang_on and hang_on == key:
            while True:                       # stand-in for a hung simulator
                time.sleep(1)
        plan = RunPlan([task], levels=(level,), seeds=(seed,), interface=config.interface,
                       conditions=(condition,))
        try:
            results = run_plan(plan, factory, config=config, keep_engine=True)
        except SceneRebuildRequired:
            _exit(REBUILD_EXIT)               # never tear an Isaac stage down in-process
        # appended as soon as it finishes, so a crash later loses nothing
        write_jsonl(spec["out"], [r.log for r in results], append=True)
        heartbeat.beat(heartbeat.SIM, force=True, finished=key)
    _exit(0)


def _exit(code: int) -> None:
    """Leave without running interpreter teardown.

    Once Isaac has started, a normal interpreter exit runs its shutdown hooks,
    and those hung for over thirteen minutes after an ordinary exception.  So a
    worker never exits normally: results are already on disk, the log already
    has the traceback, and the process just ends.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    try:
        _worker(sys.argv[1])
    except BaseException:                     # noqa: BLE001 -- anything at all
        import traceback
        traceback.print_exc()
        _exit(1)
