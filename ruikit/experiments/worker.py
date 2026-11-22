"""Workers: one process runs a chunk of jobs, and the files that say what each job came to."""

from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from ruikit.conditions import clock_settings
from ruikit.experiments import provenance
from ruikit.experiments import results as cells
from ruikit.experiments.plans import (
    Job,
)

#: an episode's own limits on the harness side, as the session server sets them
#: for the ready-made agents: the task's limits (replies, simulated time) are what
#: end an episode -- not the real time a slow model takes, which the fixed clock
#: does not see, nor a count of tool calls (a reply may hold several)
EPISODE_BUDGET = {"max_tool_calls": 6000, "max_wall_seconds": 86400.0}


QUOTA_WORDS = ("quota is used up", "quota_exceeded", "HTTP 402")


#: a worker leaves with these: another scene structure is needed (no retry
#: charged); the model's API ended an episode; the quota is used up
REBUILD_EXIT, API_EXIT, QUOTA_EXIT = 75, 78, 77


def state(job: Job, out: Path) -> str:
    """``done``, ``lost``, ``running`` (a live worker is on it) or ``todo``."""
    f = job.folder(out)
    if (f / "result.json").exists():
        return "done"
    if (f / "lost.json").exists():
        return "lost"
    try:
        info = json.loads((f / "running.json").read_text())
        if info.get("host", socket.gethostname()) != socket.gethostname():
            # another machine's worker: judged by the time it wrote inside the file
            if time.time() - float(info.get("t", 0)) < AGENT_EPISODE_SECONDS:
                return "running"
        elif _alive(int(info.get("pid", 0))):
            return "running"
    except (OSError, ValueError):
        pass
    return "todo"


#: the modules a worker runs as; the first two are the old package layout's: their
#: workers may share an OUT with ours and must not be taken for dead
WORKER_MODULES = ("ruikit.agents.batch", "ruikit.agents.harness",
                  "ruikit.experiments.batch", "ruikit.harness.session_agent")


def _alive(pid: int) -> bool:
    """A live batch worker (a recycled pid running something else is not)."""
    try:
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
    except OSError:
        return False
    return any(m in cmd for m in WORKER_MODULES) and "--worker" in cmd


def _write(path: Path, data: dict) -> None:
    """Whole or not at all: a reader never sees half a file."""
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str))
    os.replace(tmp, path)


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


# --------------------------------------------------------------------------
# a worker: one simulator, a few episodes of one task and level
# --------------------------------------------------------------------------
def run_job(job: Job, out: Path, engine, spec: dict, index: int = 1) -> dict:
    """One episode into its folder: ``result.json`` when it is a result,
    ``api_error.json`` when the model's API ended it.  ``index``: the episode's
    place in this simulator process (1: a fresh one)."""
    from ruibench.runner import heartbeat
    from ruikit.core.result import ToolError
    from ruikit.env import Budget, Episode
    from ruikit.harness.minimal_loop import ModelAgent
    from ruikit.report import clips

    folder = job.folder(out)
    if folder.exists():
        shutil.rmtree(folder)                 # an earlier try's logs are not this one's
    folder.mkdir(parents=True)
    t0 = time.time()
    _write(folder / "running.json", {"pid": os.getpid(), "host": socket.gethostname(), "t": t0,
                                     **asdict(job)})
    heartbeat.beat(heartbeat.SIM, force=True, job=job.stem, model=job.model)
    clock = clock_settings(job.clock)
    # filmed: a frame every RECORD_EVERY simulated seconds, and the episode ends as an
    # unfilmed one does (no play-out after a success).  Filming renders the cameras
    # between the agent's looks, which changes the pictures it is shown (not the
    # physics): a plan films all its episodes or none
    film = (dict(record_dir=str(folder / "frames"), record_every=clips.RECORD_EVERY,
                 record_size=clips.RECORD_SIZE) if spec.get("record") else {})
    try:
        ep = Episode(job.task, level=job.level, seed=job.seed, profile=job.profile, engine=engine,
                     log_dir=str(folder), agent=f"{job.model} through RUIKit (batch)",
                     budget=Budget(**EPISODE_BUDGET), **clock, **film)
    except BaseException:                     # a scene to rebuild, a broken card: no episode
        (folder / "running.json").unlink(missing_ok=True)
        raise
    try:
        agent = ModelAgent(job.model, base_url=spec.get("base_url"),
                           transcript=str(folder / f"{job.stem}.model.jsonl"),
                           max_output_tokens=MAX_OUTPUT_TOKENS)
        if getattr(agent.gateway, "thinking", False):    # a model thinking: all of it kept
            agent.gateway.trace = folder / f"{job.stem}.thinking.jsonl"
        if LOG_REQUESTS:                                  # every request and answer, whole
            agent.gateway.requests_log = folder / f"{job.stem}.requests.jsonl"
        result = agent.run(ep)
    except ToolError as exc:                  # the model's API: not a measurement
        row = {**asdict(job), "api_error": str(exc)[:800],
               "quota": any(w in str(exc) for w in QUOTA_WORDS)}
        _write(folder / "api_error.json", row)
        return row
    finally:
        ep.close()                            # the episode; the engine stays the worker's
        (folder / "running.json").unlink(missing_ok=True)
    score = result.score or {}
    wall = round(time.time() - t0, 1)         # the episode's, not the video's encoding
    # the frames themselves too, of every episode or of a success (``RUIKIT_KEEP_FRAMES``)
    keep = KEEP_FRAMES == "all" or (KEEP_FRAMES == "success" and bool(result.success))
    video = clips.film(folder, keep_frames=keep) if film else None
    row = cells.result_row(
        harness=job.harness, model=job.model, task=job.task, level=job.level, seed=job.seed,
        profile=job.profile, folder=folder, clock=job.clock, engine=spec["engine"],
        model_id=agent.gateway.model, success=result.success, reason=result.reason,
        progress=result.progress, failure=score.get("failure"),
        replies=result.metrics.get("replies"), sim_seconds=result.metrics.get("sim_seconds"),
        wall_seconds=wall, usage=agent.usage, filmed=bool(film), video=video,
        provenance=provenance.stamp(index, spec.get("per_worker")))
    # how the model was asked: its thinking, output cap, sampling (the server's own unless
    # set here) and reply timeout -- none of it in the frozen code, so said with each result
    gw = agent.gateway
    row["model_settings"] = {"thinking": bool(getattr(gw, "thinking", False)),
                             "max_output_tokens": agent.max_output_tokens,
                             "sampling": gw.sampling or "server defaults",
                             "timeout_s": gw.timeout_s}
    if gw.provider == "qwencloud" or gw.passback or LOG_REQUESTS or KEEP_FRAMES:
        row["model_settings"].update(channel=gw.provider, stream=gw.stream,
                                     passback_thinking=gw.passback,
                                     requests_logged=gw.requests_log is not None)
        if gw.plan is not None:                   # answers by endpoint: Token Plan, pay-as-you-go
            row["model_settings"]["endpoints"] = dict(gw.endpoints)
        row["frames_kept"] = bool(film) and keep
    _write(folder / "result.json", row)
    return row


def _worker(spec_path: str) -> None:
    """The jobs of a spec, then leave without the interpreter's teardown
    (Isaac's shutdown can hold ~20 cores for half an hour)."""
    from ruibench.core.errors import SceneRebuildRequired
    from ruibench.engines import make_engine
    from ruibench.runner import heartbeat
    from ruikit.core import plugins

    spec = json.loads(Path(spec_path).read_text())
    code = 0
    try:
        provenance.launch()                             # the code and interpreter, as started
        plugins.load(spec.get("plugins") or ())         # their tools, before any profile
        engine = make_engine(spec["engine"])
        heartbeat.beat(heartbeat.SIM, stage="simulator up")    # until here: STARTUP_SECONDS
        crash_on = os.environ.get("RUIKIT_TEST_CRASH_ON")     # test hook: a Job.tag()
        for index, raw in enumerate(spec["jobs"], 1):
            job = Job(**raw)
            if crash_on == job.tag():
                os.abort()                    # stand-in for a native crash in the simulator
            try:
                row = run_job(job, Path(spec["out"]), engine, spec, index)
            except SceneRebuildRequired:
                code = REBUILD_EXIT
                break
            print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
            if "api_error" in row:
                code = QUOTA_EXIT if row["quota"] else API_EXIT
                break
    except Exception:
        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


#: one episode, at most, before the worker is taken for hung (its own limits end
#: it long before, on a working machine; a slow model on a long task, or the
#: cloth of CR6, can take hours of real time); a ready-made agent thinks longer --
#: DI5's hard level, two arms filmed at 7-10 % of real time, ran 4 h and was
#: stopped at the 4 h it had
EPISODE_SECONDS, AGENT_EPISODE_SECONDS = 3 * 3600, 6 * 3600
#: a thinking model in the minimal loop takes minutes a reply: its episode may take longer
#: (``RUIKIT_EPISODE_SECONDS``), and its reply more than the 4,096 tokens a reply without
#: thinking needs (``RUIKIT_MAX_OUTPUT_TOKENS``; its thinking counts in it)
EPISODE_SECONDS = float(os.environ.get("RUIKIT_EPISODE_SECONDS") or EPISODE_SECONDS)
MAX_OUTPUT_TOKENS = int(os.environ.get("RUIKIT_MAX_OUTPUT_TOKENS") or 4096)
#: every request and its whole answer to ``<episode>.requests.jsonl``, the pictures sent to
#: ``inputs/`` (``RUIKIT_LOG_REQUESTS=1``); a filmed episode's frames kept beside its video,
#: of every episode or of a success only (``RUIKIT_KEEP_FRAMES=all|success``)
LOG_REQUESTS = os.environ.get("RUIKIT_LOG_REQUESTS", "0") not in ("", "0", "false")
KEEP_FRAMES = os.environ.get("RUIKIT_KEEP_FRAMES", "")
