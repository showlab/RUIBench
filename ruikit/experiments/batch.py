"""Many episodes in one go: agents and models x tasks x levels x seeds x interfaces.

    ruikit run --plan paper --dry-run                       # every episode the paper's plan asks for
    ruikit run --plan paper --experiment rui --agent claude-code --model claude-opus-5-5 \
        --form program                                      # the part with these settings
    ruikit run --plan paper --report                        # the tables so far
    ruikit run --agent minimal-loop --model openai:<model> --task DI2 DI3 --level easy \
        --seeds dev:3 --interface pose program-pose --out runs/b1
    ruikit run ... --plugins my/tools.py --interface my/interface.yaml

(``ruikit run`` is this module: ``python -m ruikit.experiments.batch``.)  The
options are spelled as everywhere (``ruibench.options``, ``ruikit.cli_options``);
interfaces go by the paper's names (``program-skill``: the RUI with its aids)
or by id.

**A plan** (``--plan``) lists the experiments: for each, the rows -- a
framework and a model, with their seeds and clock (``8s``, the default;
``paused``; ``measured``) -- the interfaces (columns), the tasks and levels,
and who runs which rows (``docs/ruikit/experiments.md``).  A
cell -- framework, model, clock, task, level, seed, interface -- is run once,
whichever experiment asks for it, and lands in the shared layout of
``ruikit.experiments.results``:
``OUT/<framework>-<model>[@<clock>]/<task>_<level>_s<seed>_<interface>/result.json``.
The part of a plan to run is chosen by setting: ``--experiment``,
``--agent``, ``--model``, ``--form`` (per-action, program, free),
``--abstraction`` (joint ... skill), ``--perception`` (pixels, privileged),
``--interface``, ``--task`` (ids, suites or all), ``--level`` and ``--clock``;
each omitted, all of it.  Everyone's parts go into one ``OUT`` (or are copied
into one), and ``--report`` builds each experiment's tables from all of it
(``ruikit.report.tables``).

**The frameworks.**  ``loop`` is our simple loop (``ruikit.harness.minimal_loop``):
worker processes run its episodes, by default one episode per simulator: a
reused Isaac stage keeps the previous episode's physics layout and renderer
state, so outcomes would depend on the order.  ``--per-worker N`` lets a worker
keep its simulator for N episodes of one task and level -- faster (starting
Isaac takes about two minutes), but no longer independent of the order.
Workers leave without Isaac's slow shutdown.  ``codex`` and
``claude-code`` are the ready-made agents (``ruikit.harness.session_agent``): one
episode a process -- a session served to a fresh agent -- at most
``--max-agents`` of each at once.

Episodes run seed by seed: every task, level, interface and agent once before
any is run again, so a first full pass is in early.

Safe to stop and start again: an episode with a ``result.json`` is never run
again, and one a live worker is still on is left to it (Ctrl-C stops the
runner, not its workers).  Not a result, and tried again (``MAX_ATTEMPTS``
times, then ``lost.json``): a worker that crashed or went quiet, and an
episode the model's API ended (a rate limit past the retries, the gateway
down, an agent's usage limit).  A used-up quota stops new launches; so do
``MAX_BROKEN`` different episodes in a row without a result.

On Isaac, as the machine's other runners do: every worker on its own
card with ``MIN_FREE_MB`` free (``CUDA_VISIBLE_DEVICES``: left to default,
Isaac lands on GPU 0, which another user's job keeps full), at most
``--max-isaac`` Isaac apps of this user's in all, and starts ``LAUNCH_GAP``
seconds apart across every runner (an Isaac start takes ~40 cores for a
minute, a running one about 6).  What an Isaac app is short of is cores, not
cards: no worker starts while the machine's load is over ``--max-load`` of
its cores, whatever the count.  No file's timestamp is ever compared with
the clock: this disk stamps files hours ahead of it.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from ruikit.conditions import DEFAULT_CLOCK
from ruikit.core.paths import ROOT
from ruikit.experiments import provenance
from ruikit.experiments import results as cells
from ruikit.experiments.machine import (
    LAUNCH_GAP,
    MAX_LOAD,
    PER_WORKER,
    busy,
    free_mb,
    last_launch,
    other_isaac,
    pick_gpu,
    stamp_launch,
)
from ruikit.experiments.plans import (
    ACTIONS,
    FORMS,
    INTERFACES,
    PERCEPTIONS,
    Experiment,
    Job,
    _older_clock,
    all_tasks,
    chunks,
    describe,
    expand_tasks,
    label,
    load_plan,
    plan,
    select,
)
from ruikit.experiments.preflight import sandbox_check, vision_check
from ruikit.experiments.worker import (
    AGENT_EPISODE_SECONDS,
    EPISODE_SECONDS,
    REBUILD_EXIT,
    _read,
    _worker,
    _write,
    state,
)
from ruikit.harness.base import harnesses
from ruikit.report.tables import write_reports

MAX_ATTEMPTS = 3
MAX_BROKEN = 3
#: a worker's patience: before its first beat (a cold start compiles shaders),
#: while the simulator works, and while the model is answering (retries and
#: rate limits included)
STARTUP_SECONDS, SIM_STALL, AGENT_STALL = 1200.0, 600.0, 1800.0
#: ready-made agents at once, of each kind (their subscriptions' limits)
MAX_AGENTS = 4
#: an episode the model's API ended waits this long before it is tried again
API_COOLDOWN = 300.0
POLL = 2.0


# --------------------------------------------------------------------------
# the runner
# --------------------------------------------------------------------------
def _say_row(job: Job, row: dict) -> str:
    if "api_error" in row:
        return f"api  {job.tag()}: {row['api_error'].splitlines()[0][:140]}"
    flag = "ok  " if row.get("success") else "--  "
    return (f"{flag}{job.tag()}: {row.get('reason')} replies={row.get('replies')} "
            f"sim={row.get('sim_seconds')}s wall={row.get('wall_seconds')}s")


def run(models: Sequence[str], tasks: Sequence[str], levels: Sequence[str], seeds: str,
        profiles: Sequence[str], out: str, engine: str = "isaac", workers: int = 2,
        per_worker: int = PER_WORKER, max_isaac: int = 8,
        think_seconds: Optional[float] = None, base_url: Optional[str] = None,
        retry_lost: bool = False, ping: bool = True, plugins: Sequence[str] = (), say=print,
        max_load: float = MAX_LOAD, harness: str = "loop", clock: Optional[str] = None,
        **kw) -> int:
    """Run what is left of one grid (models x tasks x levels x seeds x
    profiles); 0 when nothing is left to run.  ``plugins``: files or modules
    whose tools the profiles list (``ruikit.core.plugins``).  The clock: ``clock``
    (default ``8s``), or the older ``think_seconds`` a reply (0: measured)."""
    profiles = [str(Path(p).resolve()) if p.endswith((".yaml", ".yml")) and Path(p).exists()
                else p for p in profiles]
    try:
        jobs = plan(models, tasks, levels, seeds, profiles, harness,
                    clock or (_older_clock({"think_seconds": think_seconds})
                              if think_seconds is not None else DEFAULT_CLOCK))
    except Exception as exc:
        say(f"cannot plan: {type(exc).__name__}: {exc}")
        return 2
    return run_jobs(jobs, out, engine=engine, workers=workers, per_worker=per_worker,
                    max_isaac=max_isaac, base_url=base_url, retry_lost=retry_lost, ping=ping,
                    plugins=plugins, say=say, max_load=max_load, **kw)


def stale_results(jobs, out_path: Path) -> int:
    """How many of these done cells hold a result that does not compare with the
    version frozen now (``provenance.not_comparable``): the runner skips a cell
    with a result, so such a cell is never run again unless moved aside."""
    expected = provenance.expected_version()
    n = 0
    for j in jobs:
        row = _read(j.folder(out_path) / "result.json")
        if row and provenance.not_comparable(row, expected):
            n += 1
    return n


def run_jobs(jobs: Sequence[Job], out: str, engine: str = "isaac", workers: int = 2,
             per_worker: int = PER_WORKER, max_isaac: int = 8,
             base_url: Optional[str] = None, retry_lost: bool = False, ping: bool = True,
             plugins: Sequence[str] = (), say=print, max_load: float = MAX_LOAD,
             max_agents: int = MAX_AGENTS, gpus: Sequence[int] = (), record: bool = False,
             experiments: Sequence[Experiment] = ()) -> int:
    """Run the jobs not done yet; 0 when nothing is left to run."""
    from ruikit.core import plugins as plugin_files
    from ruikit.interfaces import load_profile

    # the workers start in the repository: paths given from elsewhere are made whole
    plugins = [str(Path(p).resolve()) if Path(p).exists() else p for p in plugins]
    try:
        plugin_files.load(plugins)
        for p in sorted({j.profile for j in jobs}):
            load_profile(p)
    except Exception as exc:                  # a missing tool, a bad file: say so before any start
        say(f"cannot use the profiles: {type(exc).__name__}: {exc}")
        return 2
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    if retry_lost:
        for j in jobs:
            (j.folder(out_path) / "lost.json").unlink(missing_ok=True)
    states = {j: state(j, out_path) for j in jobs}
    todo = [j for j in jobs if states[j] == "todo"]
    counts = {s: sum(v == s for v in states.values()) for s in ("done", "running", "lost", "todo")}
    say(f"{len(jobs)} episodes planned: {counts['done']} done, {counts['running']} running "
        f"elsewhere, {counts['lost']} lost, {counts['todo']} to run")
    stale = stale_results([j for j in jobs if states[j] == "done"], out_path)
    if stale:
        say(f"  {stale} of the done hold results that do not compare with "
            f"{provenance.expected_version()} and are not run again: `python -m "
            f"ruikit.experiments.upgrade {out} --go` scores again what it can and moves the "
            "rest aside to be run")
    if not todo:
        return 0
    say(provenance.describe(per_worker))
    if record:
        say("filming every episode: video.mp4 beside its result.json")
    if engine == "isaac":
        why = (vision_check(sorted({j.profile for j in todo}))
               or sandbox_check(sorted({j.profile for j in todo})))
        if why:
            say(f"cannot run on Isaac: {why}")
            return 2
    if ping:
        from ruikit.models.gateway import Gateway

        for m in sorted({j.model for j in todo if harnesses()[j.harness].in_process}):
            try:
                Gateway(m, base_url=base_url).ask("Reply with exactly: OK")
            except Exception as exc:          # a ToolError: no token, no route, no quota
                say(f"{m} does not answer: {exc}")
                return 2
        for h in sorted({j.harness for j in todo if not harnesses()[j.harness].in_process}):
            why = harnesses()[h].preflight()
            if why:
                say(f"{h} cannot run: {why}")
                return 2
    if engine == "isaac":
        # a ready-made agent counts only once it is shown to get the session's tools
        # and nothing else, with its CLI as installed here (``ruikit.harness.selftest``)
        from ruikit.harness.selftest import ensure

        for h, m in sorted({(j.harness, j.model) for j in todo
                            if not harnesses()[j.harness].in_process}):
            why = harnesses()[h].preflight() or ensure(h, m, out_path / "selftest", say=say)
            if why:
                say(f"{h} {m} cannot run: {why}")
                return 2
    isaac = engine == "isaac"
    work = out_path / ".workers"
    work.mkdir(exist_ok=True)
    queue: List[dict] = [{"jobs": c, "after": 0.0} for c in chunks(todo, max(1, per_worker))]
    attempts: Dict[Job, int] = {}
    running: Dict[int, dict] = {}
    last_start: Dict[int, float] = {}
    stop, n = "", 0
    broken: set = set()                       # episodes without a result since the last result

    def launch(chunk: List[Job], gpu: Optional[int]) -> None:
        nonlocal n
        n += 1
        stem = work / f"w{n:03d}-{os.getpid()}-{chunk[0].task}-{chunk[0].level}"
        spec = {"jobs": [asdict(j) for j in chunk], "engine": engine,
                "out": str(out_path.resolve()), "per_worker": per_worker,
                "base_url": base_url, "plugins": list(plugins), "record": record}
        Path(f"{stem}.spec.json").write_text(json.dumps(spec))
        Path(f"{stem}.heartbeat").unlink(missing_ok=True)
        env = dict(os.environ, OMNI_KIT_ACCEPT_EULA=os.environ.get("OMNI_KIT_ACCEPT_EULA", "YES"),
                   RUIBENCH_HEARTBEAT=f"{stem}.heartbeat",
                   PYTHONPATH=os.pathsep.join(filter(None, [str(ROOT), os.environ.get("PYTHONPATH")])))
        if gpu is not None:                   # Isaac: its card, its temporary files beside the runs
            (ROOT / "runs" / "tmp").mkdir(parents=True, exist_ok=True)
            env.update(CUDA_VISIBLE_DEVICES=str(gpu), RUIBENCH_ISAAC_GPU="0",
                       TMPDIR=str(ROOT / "runs" / "tmp"))
        module = harnesses()[chunk[0].harness].worker_module
        with open(f"{stem}.log", "w") as fh:
            proc = subprocess.Popen([sys.executable, "-m", module, "--worker",
                                     f"{stem}.spec.json"], cwd=str(ROOT), stdout=fh,
                                    stderr=subprocess.STDOUT, env=env,
                                    start_new_session=True)   # Ctrl-C stops the runner, not them
        now = time.time()
        running[proc.pid] = {"proc": proc, "jobs": chunk, "gpu": gpu, "started": now,
                             "beat": f"{stem}.heartbeat", "log": f"{stem}.log",
                             "harness": chunk[0].harness}
        if gpu is not None:
            last_start[gpu] = now
            stamp_launch(now)
        say(f"   start {chunk[0].task} {chunk[0].level}: "
            + (f"{len(chunk)} episodes" if harnesses()[chunk[0].harness].in_process
               else chunk[0].tag())
            + (f" on gpu {gpu}" if gpu is not None else "") + f" (pid {proc.pid})")

    def pool(harness: str) -> str:
        return "codex" if harness.startswith("codex") else harness    # one subscription

    def room(entry: dict) -> bool:
        """May this entry's framework start one more now?"""
        h = entry["jobs"][0].harness
        mine = sum(1 for w in running.values() if pool(w["harness"]) == pool(h))
        return mine < (workers if harnesses()[h].in_process else max_agents)

    def where() -> Optional[int]:
        """The card for the next worker (-1 without Isaac), or None: not now.  The
        shared clock is read first and again last: the slower checks between
        (nvidia-smi) would otherwise lose every free minute to a quicker runner."""
        if not isaac:
            return -1
        if time.time() - last_launch() < LAUNCH_GAP:
            return None
        if len(running) + other_isaac(list(running)) >= max_isaac:
            return None
        if busy(max_load):
            return None
        now = time.time()
        if now - last_launch() < LAUNCH_GAP:
            return None
        return pick_gpu(free_mb(), [] if gpus else [w["gpu"] for w in running.values()],
                        last_start, now, only=gpus)

    def stalled(pid: int, w: dict) -> str:
        from ruibench.runner import heartbeat

        beat = heartbeat.read(w["beat"])
        if beat is None:
            idle = time.time() - w["started"]
            return f"no sign of life for {idle:.0f} s" if idle > STARTUP_SECONDS else ""
        limit = AGENT_STALL if beat.get("phase") == heartbeat.AGENT else SIM_STALL
        if beat["age"] > limit:
            return f"no progress for {beat['age']:.0f} s ({beat.get('phase')})"
        cap = EPISODE_SECONDS if harnesses()[w["harness"]].in_process else AGENT_EPISODE_SECONDS
        for j in w["jobs"]:
            info = _read(j.folder(out_path) / "running.json")
            if info.get("pid") == pid and time.time() - float(info.get("t", 0)) > cap:
                return f"{j.tag()} over {cap // 60:.0f} min"
        return ""

    def charge(j: Job, why: str) -> bool:
        """One more try used; False when that was the last (``lost.json``)."""
        attempts[j] = attempts.get(j, 0) + 1
        if attempts[j] < MAX_ATTEMPTS:
            return True
        f = j.folder(out_path)
        f.mkdir(parents=True, exist_ok=True)
        _write(f / "lost.json", {**asdict(j), "attempts": attempts[j], "last": why[-800:]})
        say(f"lost {j.tag()} after {attempts[j]} tries")
        return False

    def settle(w: dict, code: int) -> None:
        """A worker has left: its results, the episode it stopped on, the rest."""
        nonlocal stop
        jobs = w["jobs"]
        k = 0
        while k < len(jobs) and (jobs[k].folder(out_path) / "result.json").exists():
            say(_say_row(jobs[k], _read(jobs[k].folder(out_path) / "result.json")))
            broken.clear()
            k += 1
        if k == len(jobs):
            return
        j, rest = jobs[k], jobs[k + 1:]
        api = _read(j.folder(out_path) / "api_error.json")
        if api:                                   # the model's API ended it: later, again
            say(_say_row(j, api))
            broken.add(j)
            if api.get("quota"):                  # the account's, not the cell's: no try used
                stop = stop or f"the quota of {j.model} is used up (or its login failed)"
                queue.append({"jobs": [j], "after": time.time() + API_COOLDOWN})
            elif charge(j, api.get("api_error", "")):
                queue.append({"jobs": [j], "after": time.time() + API_COOLDOWN})
        elif code == REBUILD_EXIT and k > 0:      # another scene structure: a fresh worker
            rest = [j] + rest
        else:                                     # it crashed, hung, or could not start
            broken.add(j)
            why = f"worker ended ({code}): " + _tail(w["log"]).strip().replace("\n", " | ")[-400:]
            say(f"   {j.tag()}: {why}")
            if charge(j, why):
                rest = [j] + rest
        if rest:
            queue.insert(0, {"jobs": rest, "after": 0.0})
        if len(broken) >= MAX_BROKEN and not stop:
            stop = f"{len(broken)} episodes in a row without a result"

    try:
        while (queue and not stop) or running:
            while not stop:
                now = time.time()
                ready = [e for e in queue if e["after"] <= now and room(e)]
                if not ready:
                    break
                gpu = where()
                if gpu is None:
                    break
                queue.remove(ready[0])
                chunk = [j for j in ready[0]["jobs"] if state(j, out_path) == "todo"]
                if chunk:
                    launch(chunk, None if gpu == -1 else gpu)
            for pid, w in list(running.items()):
                code = w["proc"].poll()
                if code is None:
                    why = stalled(pid, w)
                    if not why:
                        continue
                    say(f"   pid {pid}: {why} -- stopping it")
                    _stop_tree(w["proc"])
                    code = -9
                del running[pid]
                settle(w, code)
            if stop and queue:
                say(f"not starting more: {stop}")
                queue.clear()
            time.sleep(POLL)
    except KeyboardInterrupt:
        say("stopped; the workers finish the episodes they are on "
            f"(pids {', '.join(map(str, running)) or 'none'}). Run again to go on.")
        return 130
    say(write_reports(out_path, experiments))
    left = sum(state(j, out_path) == "todo" for j in jobs)
    return 0 if not left else 1


def _stop_tree(proc) -> None:
    """Stop a worker and what it started (a session's simulator, an agent): by
    process group, then for good -- an Isaac app can ignore SIGTERM."""
    import signal

    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except OSError:
            pass
        try:
            proc.wait(timeout=20)
            return
        except subprocess.TimeoutExpired:
            continue


def _tail(path: str, n: int = 1500) -> str:
    try:
        return Path(path).read_text(errors="replace")[-n:]
    except OSError:
        return ""


# --------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    from ruibench import options as O
    from ruikit import cli_options as K

    ap = argparse.ArgumentParser(prog="ruikit run", description=__doc__.split("\n\n")[0])
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    O.add_out(ap, None, "the run's folder: results, logs, report.md (default runs/<plan>, "
                        "or runs/ruikit without one)")
    ap.add_argument("--plan", default=None,
                    help="a plan of experiments: a name in ruikit/configs/experiments (paper) or "
                         "a YAML file")
    ap.add_argument("--experiment", dest="experiments", nargs="+", action="extend", default=[],
                    help="with --plan: these experiments")
    ap.add_argument("--experiments", dest="experiments", nargs="+", action="extend",
                    help=argparse.SUPPRESS)
    K.add_agent(ap, many=True)
    K.add_model(ap, many=True)
    K.add_interfaces(ap)
    K.add_forms(ap, FORMS, ACTIONS)
    ap.add_argument("--perception", nargs="+", default=[], choices=PERCEPTIONS + ("model",),
                    help="with --plan: pixels (the main setting) or privileged (the control)")
    O.add_tasks(ap)
    O.add_levels(ap)
    O.add_seeds(ap, None)
    O.add_clock(ap, DEFAULT_CLOCK)
    ap.add_argument("--owners", nargs="+", default=[], help=argparse.SUPPRESS)
    ap.add_argument("--agents", dest="rows", nargs="+", default=[],
                    help="with --plan: these rows, as agent:model (codex:gpt-6-astra)")
    ap.add_argument("--plugins", nargs="+", default=[],
                    help="files or modules with tools the interfaces list (outside the library)")
    O.add_engine(ap, "isaac")
    O.add_gpus(ap)
    ap.add_argument("--workers", type=int, default=2,
                    help="minimal-loop episodes at once (default 2)")
    ap.add_argument("--max-agents", type=int, default=MAX_AGENTS,
                    help=f"codex and claude-code episodes at once, of each (default {MAX_AGENTS})")
    ap.add_argument("--per-worker", type=int, default=PER_WORKER,
                    help=f"episodes a worker runs before a fresh one (default {PER_WORKER})")
    ap.add_argument("--max-isaac", type=int, default=8,
                    help="this user's Isaac apps on the machine, everyone's runs counted (default 8)")
    ap.add_argument("--max-load", type=float, default=MAX_LOAD,
                    help="start no Isaac app while the machine's 1-minute load is over this share "
                         f"of its cores (default {MAX_LOAD})")
    ap.add_argument("--think-seconds", type=float, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--record", action="store_true",
                    help="film each episode (video.mp4 in its folder); a plan's record says its own")
    ap.add_argument("--no-record", action="store_true",
                    help="film nothing, whatever the plan says: a quick trial, whose results "
                         "do not compare with filmed ones")
    ap.add_argument("--base-url", default=None, help="the model channel's endpoint, if not the default")
    ap.add_argument("--retry-lost", action="store_true", help="run lost episodes again")
    ap.add_argument("--no-ping", action="store_true", help="skip the one-line check of each model")
    ap.add_argument("--dry-run", action="store_true", help="print the episodes and stop")
    ap.add_argument("--report", action="store_true", help="only write the reports of --out")
    args = ap.parse_args(argv)
    if args.worker:
        _worker(args.worker)
    try:
        agents, models = K.agents_of(args), K.models_of(args)
        clock = O.clock_of(args) if args.clock else None
    except ValueError as exc:
        ap.error(str(exc))
    out = args.out or str(Path("runs") / (Path(args.plan).stem if args.plan else "ruikit"))
    experiments = load_plan(args.plan) if args.plan else []
    chosen = [e for e in experiments if not args.experiments or e.name in args.experiments]
    if args.report:
        print(write_reports(Path(out), chosen))
        return 0
    seeds = O.seeds_of(args)
    if experiments:
        if args.think_seconds is not None or seeds is not None:
            ap.error("a plan says each row's seeds: choose its part with --experiment, --agent, "
                     "--model, --form, --abstraction, --perception, --interface, --task, "
                     "--level and --clock")
        try:
            jobs = select(chosen, columns=args.interfaces or (), agents=args.rows,
                          owners=args.owners, harnesses=agents, models=models,
                          forms=args.forms, actions=args.actions, perceptions=args.perception,
                          tasks=O.tasks_of(args) or (), levels=args.levels or (),
                          clocks=[clock] if clock else ())
        except ValueError as exc:
            ap.error(str(exc))
    else:
        if args.forms or args.actions or args.perception or args.rows or args.owners \
                or args.experiments:
            ap.error("--experiment, --form, --abstraction, --perception and --agents choose from "
                     "a plan (--plan); without one give --interface")
        if len(agents) > 1:
            ap.error("without --plan, one --agent")
        if not models:
            ap.error("without --plan, give --model (openai:<name>, anthropic:<name>, local:<name>; "
                     "for codex and claude-code their model)")
        try:
            tasks = expand_tasks(O.tasks_of(args)) if O.tasks_of(args) else all_tasks()
            profiles = K.interface_ids(args.interfaces) or list(INTERFACES)
        except ValueError as exc:
            ap.error(str(exc))
        if clock is None:
            clock = (_older_clock({"think_seconds": args.think_seconds})
                     if args.think_seconds is not None else DEFAULT_CLOCK)
        jobs = plan(models, tasks, O.levels_of(args), seeds or "dev:1", profiles,
                    (agents or ["loop"])[0], clock)
    record = (args.record or any(e.record for e in chosen)) and not args.no_record
    if args.dry_run:
        for k, c in enumerate(chunks(jobs, args.per_worker), 1):
            print(f"worker job {k}: {c[0].task} {c[0].level}, "
                  + ", ".join(f"s{j.seed} {label(j.profile)} {j.agent}" for j in c))
        for line in describe(jobs):
            print(line)
        print(f"{len(jobs)} episodes" + (", each filmed" if record else ", not filmed"))
        return 0
    return run_jobs(jobs, out, engine=args.engine, workers=args.workers,
                    per_worker=args.per_worker, max_isaac=args.max_isaac, max_load=args.max_load,
                    base_url=args.base_url, retry_lost=args.retry_lost,
                    ping=not args.no_ping, plugins=args.plugins, max_agents=args.max_agents,
                    gpus=O.gpus_of(args), record=record, experiments=chosen)

if __name__ == "__main__":
    sys.exit(main())
