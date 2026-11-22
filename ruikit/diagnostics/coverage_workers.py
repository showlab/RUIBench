"""Running coverage cells on Isaac: one worker per card, resumable, restarted when it stalls."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from ruikit.diagnostics.coverage import Cell, _say, run_cell
from ruikit.experiments.machine import (
    LAUNCH_GAP,
    MAX_LOAD,
    PER_WORKER,
    busy,
    last_launch,
    stamp_launch,
)
from ruikit.experiments.plans import parse_seeds

# --------------------------------------------------------------------------
# Isaac: worker processes, one simulator each
# --------------------------------------------------------------------------
#: a card with this much memory free takes one Isaac process (as the benchmark's fan-out)
NEED_MIB = 16000


#: before a worker's first sign of life (a cold Isaac start compiles shaders; a
#: loaded machine takes longer -- 1200 s killed sound starts), and between two
#: once it runs
STARTUP_SECONDS, STALL_SECONDS = 1500.0, 600.0


MAX_ATTEMPTS = 3


#: a worker leaves with this when the next cell needs another scene structure
REBUILD_EXIT = 75


Key = Tuple[str, str, int, str]


def idle_gpus(need_mib: int = NEED_MIB) -> List[int]:
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                          "--format=csv,noheader,nounits"], capture_output=True, text=True)
    free = []
    for line in out.stdout.strip().splitlines():
        index, left = (v.strip() for v in line.split(","))
        if int(left) >= need_mib:
            free.append(int(index))
    return free


def _may_start() -> bool:
    """May an Isaac worker start now: a minute after any runner's last start,
    and the machine's load under its share of the cores (``ruikit.experiments.batch``)."""
    return time.time() - last_launch() >= LAUNCH_GAP and not busy(MAX_LOAD)


def keys_for(tasks: Sequence[str], levels: Sequence[str], seeds: Sequence[str],
             profiles: Sequence[str]) -> List[Key]:
    """Every cell: ``seeds`` are numbers, or the split's seeds of each task and
    level (``test:1``, its first test seed; ``dev:3``), as the batch runner takes them."""
    from ruibench.tasks import splits

    data = splits.load()
    keys = []
    for t in tasks:
        for lv in levels:
            numbers: List[int] = []
            for spec in seeds:
                numbers += [s for s in parse_seeds(str(spec), t, lv, data) if s not in numbers]
            keys += [(t, lv, s, p) for s in numbers for p in profiles]
    return keys


def _lost(key: Key, attempts: int) -> Cell:
    task, level, seed, profile = key
    return Cell(task, level, seed, profile, False, 0.0, "harness_error", 0, 0.0, 0.0, [],
                f"the worker died or hung {attempts} times on this cell")


def run_isolated(keys: Sequence[Key], engine: str, out: str, gpus: Sequence[int],
                 log_dir: Optional[str] = None, say=print,
                 record: Optional[str] = None, oracle_coordinates: bool = False,
                 think_seconds: float = 0.0, per_worker: int = PER_WORKER,
                 reference: Optional[str] = None) -> List[Cell]:
    """Every cell, in worker processes that may die: one per entry of ``gpus``
    (a card named twice takes two), each taking up to ``per_worker`` cells of
    one task and level -- by default one, so that no cell's outcome depends on
    what ran before it in the same simulator."""
    from ruibench.runner import heartbeat

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = out_path.parent / f"{out_path.stem}.workers"
    work.mkdir(parents=True, exist_ok=True)
    done: Dict[Key, Cell] = {}
    if out_path.exists():                     # resume: never redo a finished cell
        for line in out_path.read_text().splitlines():
            c = Cell(**json.loads(line))
            done[(c.task, c.level, c.seed, c.profile)] = c
    groups: Dict[Tuple[str, str], List[Key]] = {}
    for k in keys:
        if k not in done:
            groups.setdefault((k[0], k[1]), []).append(k)
    step = max(1, int(per_worker))
    queue = [g[i:i + step] for g in groups.values() for i in range(0, len(g), step)]
    attempts: Dict[Key, int] = {}
    running: Dict[int, dict] = {}              # slot -> the worker in it
    n = 0

    def collect(w: dict) -> None:
        part = Path(w["part"])
        newest: Dict[Key, Cell] = {}           # a file's last row of a cell is its answer
        for line in part.read_text().splitlines() if part.exists() else []:
            c = Cell(**json.loads(line))
            newest[(c.task, c.level, c.seed, c.profile)] = c
        for k, c in newest.items():
            if k not in done:
                done[k] = c
                with out_path.open("a") as fh:
                    fh.write(json.dumps(c.as_dict()) + "\n")
                say(_say(c))

    isaac = engine == "isaac"
    while queue or running:
        for slot, gpu in enumerate(gpus):
            if slot in running or not queue:
                continue
            if isaac and not _may_start():
                break
            todo = [k for k in queue.pop(0) if k not in done]
            if not todo:
                continue
            n += 1
            stem = work / f"w{n:03d}-gpu{gpu}-{todo[0][0]}-{todo[0][1]}"
            # names repeat from run to run (a rerun into the same --out): a part file
            # or heartbeat of that name is an earlier run's, and must not answer for
            # this worker
            for old in (Path(f"{stem}.cells.jsonl"), Path(f"{stem}.heartbeat")):
                old.unlink(missing_ok=True)
            spec = {"cells": todo, "engine": engine, "out": f"{stem}.cells.jsonl",
                    "per_worker": step, "log_dir": log_dir, "record": record,
                    "oracle_coordinates": oracle_coordinates, "think_seconds": think_seconds,
                    "reference": reference}
            Path(f"{stem}.spec.json").write_text(json.dumps(spec))
            env = dict(os.environ, RUIBENCH_ISAAC_GPU=str(gpu))
            if isaac:                         # its own card only (CR6's cloth takes the first)
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), RUIBENCH_ISAAC_GPU="0")
            env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
            env[heartbeat.ENV_VAR] = f"{stem}.heartbeat"
            with open(f"{stem}.log", "w") as fh:
                proc = subprocess.Popen([sys.executable, "-m", "ruikit.diagnostics.coverage",
                                         "--worker", f"{stem}.spec.json"],
                                        stdout=fh, stderr=subprocess.STDOUT, env=env)
            running[slot] = {"proc": proc, "todo": todo, "part": spec["out"], "gpu": gpu,
                             "beat": env[heartbeat.ENV_VAR], "started": time.time()}
            if isaac:
                stamp_launch(time.time())
            say(f"   gpu {gpu}: {todo[0][0]} {todo[0][1]}, {len(todo)} cells (pid {proc.pid})")
        time.sleep(2.0)
        for slot, w in list(running.items()):
            gpu = w["gpu"]
            code = w["proc"].poll()
            if code is None:
                beat = heartbeat.read(w["beat"])
                idle = (time.time() - w["started"] if beat is None else beat["age"])
                if idle > (STARTUP_SECONDS if beat is None else STALL_SECONDS):
                    say(f"   gpu {gpu}: no progress for {idle:.0f} s -- killing the worker")
                    w["proc"].kill()
                    w["proc"].wait()
                    code = -9
                else:
                    continue
            del running[slot]
            collect(w)
            left = [k for k in w["todo"] if k not in done]
            if not left:
                continue
            k = left[0]                       # the cell the worker was on
            if code != REBUILD_EXIT or k == w["todo"][0]:
                attempts[k] = attempts.get(k, 0) + 1
                if attempts[k] >= MAX_ATTEMPTS:
                    done[k] = _lost(k, attempts[k])
                    with out_path.open("a") as fh:
                        fh.write(json.dumps(done[k].as_dict()) + "\n")
                    say(_say(done[k]))
                    left = left[1:]
                else:
                    say(f"   gpu {gpu}: worker ended ({code}) on {k}; trying again")
            if left:
                queue.insert(0, left)
    return [done[k] for k in keys if k in done]


def _worker(spec_path: str) -> None:
    """One simulator, the cells of one scene structure, then leave without the
    interpreter's teardown (Isaac's shutdown hooks can hang for minutes)."""
    from ruibench.core.errors import SceneRebuildRequired
    from ruibench.engines import make_engine
    from ruibench.runner import heartbeat
    from ruikit.experiments import provenance

    spec = json.loads(Path(spec_path).read_text())
    heartbeat.beat(heartbeat.SIM, stage="worker started")
    code = 0
    try:
        provenance.launch()                   # the code and interpreter, as started
        engine = make_engine(spec["engine"])
        crash_on = os.environ.get("RUIKIT_TEST_CRASH_ON")     # test hook: "TASK|level|seed|profile"
        for index, (task, level, seed, profile) in enumerate(spec["cells"], 1):
            heartbeat.beat(heartbeat.SIM, force=True, task=f"{task} {level} {profile}", seed=seed)
            if crash_on == f"{task}|{level}|{seed}|{profile}":
                os.abort()                    # stand-in for a native crash in the simulator
            try:
                cell = run_cell(task, level, seed, profile, engine=engine,
                                log_dir=spec.get("log_dir"), record=spec.get("record"),
                                oracle_coordinates=bool(spec.get("oracle_coordinates")),
                                think_seconds=float(spec.get("think_seconds") or 0.0),
                                reference=spec.get("reference"))
            except SceneRebuildRequired:
                code = REBUILD_EXIT
                break
            cell.provenance = provenance.stamp(index, spec.get("per_worker"))
            with open(spec["out"], "a") as fh:
                fh.write(json.dumps(cell.as_dict()) + "\n")
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
