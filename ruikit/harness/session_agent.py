"""Ready-made agents -- Codex, Claude Code -- one episode each, as the batch runs them.

    python -m ruikit.harness.session_agent episode --harness codex --model gpt-5.6-sol \\
        --task WK1 --level easy --seed 15 --profile rui.pose/v1+aids --out runs/x --gpu 3
    python -m ruikit.harness.session_agent check codex        # can it run on this machine?
    python -m ruikit.harness.session_agent selftest claude-code --model claude-opus-5-5
    python -m ruikit.experiments.batch --plan exp.yaml ...  # many: codex / claude-code rows

One episode: a session (``ruikit.serve.session``: Isaac on its own card, the
clock, the task's limits), then a fresh agent with a clean context, hooked to
the session over MCP (``ruikit.serve.mcp_stdio --session``):

- **clean**: it starts in an empty folder of its own outside the repository,
  named by nothing but a random id, which is deleted with everything in it
  when the episode ends.  It reads no instruction file (CLAUDE.md, AGENTS.md),
  no memory, settings, hooks, skills or earlier session: Codex gets a home of
  its own (``CODEX_HOME``: only the login, linked), Claude Code loads none of
  the user's settings and keeps no session on disk.  It gets the frozen prompt
  (``PROMPT``) and nothing else; its environment holds no key, token or path
  of the repository;
- **no shell, no files**: the session's tools are all it has (Codex: its shell,
  file, image, web and sub-agent tools off; Claude Code: ``--tools ""``);
  pictures come back inline, as they do to the simple loop;
- **counted alike**: every tool call is one reply, ``batch`` several calls as
  one reply -- as the simple loop's model sends several calls in one reply;
- **the same model settings**: each model reasons at high effort, set
  explicitly so that it does not change with the CLI; every result records it;
- **checked**: before an agent's first episode on Isaac, ``selftest``
  (``ruikit.harness.selftest``) shows that this agent, at the version
  installed here, gets the session's tools and nothing else.

When the episode ends the agent has ``GRACE_SECONDS`` to give its summary;
an agent that stops first ends the episode (``agent_stopped``).  Then the cell
gets its ``result.json`` (``ruikit.experiments.results``) with the agent's token use
(``ruikit.experiments.usage``) and the audit of its transcript
(``ruikit.experiments.rules_audit``).  A run that does not count -- the agent cut
off by a usage limit or an API error before the episode ended, or a rule broken --
leaves ``api_error.json`` instead, and the batch runs it again with a new agent: so
does an agent that did not get the session's tools, made no call to them, or was
reached by anything besides its prompt and the session's results.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

from ruikit.conditions import DEFAULT_CLOCK
from ruikit.core.paths import ROOT
from ruikit.experiments import provenance
from ruikit.experiments import results as cells
from ruikit.harness.base import Harness
from ruikit.harness.claude_code import (
    CLAUDE_LOGIN,
    EFFORT,
    claude_binary,
    claude_command,
    claude_leftovers,
    claude_start,
)
from ruikit.harness.codex import (
    CODEX_LOGIN,
    CODEX_MODES,
    _keep_codex_record,
    codex_binary,
    codex_effort,
    codex_home,
)
from ruikit.harness.mcp import PYTHON

#: an agent's home and work folder: never in the repository
AGENT_HOMES = Path(os.environ.get("RUIKIT_AGENT_HOMES")
                   or Path.home() / ".cache" / "ruikit-agents")
STARTUP_SECONDS = 1500.0
#: after the episode has ended, the agent's time to give its summary
GRACE_SECONDS = 180.0

#: the prompt of every ready-made agent, word for word; the task's own lines are
#: filled in as the simple loop's model is told them (``ruikit.harness.minimal_loop``)
PROMPT = """You are the control agent of a robot at a table in a simulated lab (Isaac Sim), in RUIBench task {task} ({level}), through the interface {interface}. You act ONLY through the tools of the MCP server "ruibench", and you see ONLY the pictures its tools return. Treat it like a real robot: nobody tells you where anything is. You have to find things and estimate their positions with the tools; estimates can be off by several millimetres, so check your work with the cameras when it matters.

Task: {instruction}
Hint: {hint}
Arms: {arms}.
Limits: at most {replies} replies and {seconds:g} s of simulated time. {clock}

How replies count: every tool call you make is one reply and counts against the limit. The tool "batch" sends several independent calls as ONE reply; they start in order and run together where they can (moves on different arms at the same time), and the reply comes back when every blocking one is done (task_info says what one reply may hold).

Rules (this is a fairness test of the interface, please keep them strictly):
- Use only the ruibench tools. You have no shell and no files; do not try to reach anything outside these tools.
- Start with task_info (the station's frames, units, gripper, cameras, what one reply may hold), and read the descriptions of the tools you intend to use.
- Be efficient: every reply counts. Batch independent calls. But do verify with the cameras when it matters (e.g., after grasping, after placing), instead of blindly repeating moves.
- Reaching the goal does not end the episode. When you believe the task is done, call task_done with a one-line report: the episode is judged on the world as it is then. If you are stuck, call task_done and say why. The episode also ends by itself when a limit is reached (a result then says THE EPISODE HAS ENDED).

When the episode has ended, reply with a brief summary: what you did, whether you think it worked, how many replies you used, and what about this interface or its tools made the task harder than it needed to be (missing tools, confusing or unhelpful results)."""

#: the prompt on the real station (``ruibench.engines.franka``): the simulated
#: lab's prompt, word for word but for what is real -- the robot, its cameras'
#: estimates, the clock -- and the person who stands by it
REAL_PROMPT = """You are the control agent of a real robot, a Franka arm at a table in our lab, in RUIBench task {task} ({level}), through the interface {interface}. You act ONLY through the tools of the MCP server "ruibench", and you see ONLY the pictures its tools return. Nobody tells you where anything is. You have to find things and estimate their positions with the tools; the estimates come from real cameras and can be off by several millimetres, so check your work with the cameras when it matters. A person stands by the robot for safety only: they will not help or answer.

Task: {instruction}
Hint: {hint}
Arms: {arms}.
Limits: at most {replies} replies and {seconds:g} s of time, by the wall clock. {clock}

How replies count: every tool call you make is one reply and counts against the limit. The tool "batch" sends several independent calls as ONE reply; they start in order and run together where they can (moves on different arms at the same time), and the reply comes back when every blocking one is done (task_info says what one reply may hold).

Rules (this is a fairness test of the interface, please keep them strictly):
- Use only the ruibench tools. You have no shell and no files; do not try to reach anything outside these tools.
- Start with task_info (the station's frames, units, gripper, cameras, what one reply may hold), and read the descriptions of the tools you intend to use.
- Be efficient: every reply counts. Batch independent calls. But do verify with the cameras when it matters (e.g., after grasping, after placing), instead of blindly repeating moves.
- Reaching the goal does not end the episode. When you believe the task is done, call task_done with a one-line report: the episode is judged on the world as it is then. If you are stuck, call task_done and say why. The episode also ends by itself when a limit is reached (a result then says THE EPISODE HAS ENDED).

When the episode has ended, reply with a brief summary: what you did, whether you think it worked, how many replies you used, and what about this interface or its tools made the task harder than it needed to be (missing tools, confusing or unhelpful results)."""


def preflight(harness: str) -> Optional[str]:
    """Why this harness cannot run here, or None."""
    if harness in CODEX_MODES:
        if not codex_binary():
            return "no Codex CLI (set RUIKIT_CODEX)"
        if not CODEX_LOGIN.exists():
            return f"Codex is not logged in ({CODEX_LOGIN} is missing): run codex login"
        return None
    if harness == "claude-code":
        if not claude_binary():
            return "no Claude Code CLI (set RUIKIT_CLAUDE)"
        if not CLAUDE_LOGIN.exists():
            return f"Claude Code is not logged in ({CLAUDE_LOGIN} is missing): run claude, then /login"
        return None
    return f"no ready-made agent {harness!r} (codex, codex-code, claude-code)"


class SessionAgent(Harness):
    """A ready-made agent -- Codex, in either mode, or Claude Code -- with a
    clean context, reaching its own episode's session server over MCP."""

    worker_module = "ruikit.harness.session_agent"
    in_process = False

    def __init__(self, name: str) -> None:
        self.name = name

    def preflight(self) -> Optional[str]:
        return preflight(self.name)

    def run(self, job, out: Path, engine: str = "isaac", **options) -> dict:
        return run_episode(self.name, job.model, job.task, job.level, job.seed, job.profile,
                           str(out), engine=engine, clock=job.clock, **options)


def agent_command(harness: str, model: str, prompt: str, home: Path, folder: Path):
    """(argv, env) of a fresh agent on this session."""
    work, tmp = home / "work", home / "tmp"
    work.mkdir(parents=True, exist_ok=True)
    tmp.mkdir(exist_ok=True)
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL",
                    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy",
                    "no_proxy", "SSL_CERT_FILE")}
    env["TMPDIR"] = str(tmp)                   # its own: deleted with its home
    if harness in CODEX_MODES:
        env["CODEX_HOME"] = str(codex_home(home, folder, model, CODEX_MODES[harness]))
        argv = [codex_binary(), "exec", "--json", "-m", model, "--skip-git-repo-check",
                "--ignore-rules", "-C", str(work), prompt]
        return argv, env
    argv, more = claude_command(model, prompt, folder)
    env.update(more)
    return argv, env


@lru_cache(maxsize=None)
def cli_version(harness: str) -> str:
    """The agent's CLI as installed here (``--version``), for its results."""
    binary = codex_binary() if harness in CODEX_MODES else claude_binary()
    try:
        done = subprocess.run([binary, "--version"], capture_output=True, text=True, timeout=60,
                              stdin=subprocess.DEVNULL)
        return ((done.stdout or done.stderr).strip().splitlines() or ["unknown"])[0][:80]
    except (OSError, subprocess.SubprocessError, TypeError):
        return "unknown"


def effort_of(harness: str, model: str) -> str:
    return codex_effort(model) if harness in CODEX_MODES else EFFORT


def prompt_for(folder: Path, template: Optional[str] = None) -> str:
    from ruikit.serve.session import _ask

    info = _ask(folder, {"op": "info"})
    if info.get("error"):
        raise RuntimeError(f"the session does not answer: {info['error']}")
    lim = info["limits"]
    template = template or (REAL_PROMPT if info.get("real") else PROMPT)
    prompt = template.format(task=info["task"], level=info["level"], interface=info["interface"],
                           instruction=info["instruction"], hint=info.get("hint") or "-",
                           arms=", ".join(info.get("arms") or []), replies=lim["replies"],
                           seconds=float(lim["simulated_seconds"]),
                           clock=lim["clock"][0].upper() + lim["clock"][1:] + ".")
    # an interface with an operation guide (``+guide``): it comes last
    return prompt + (f"\n\n{info['guide']}" if info.get("guide") else "")


def _stop(proc, grace: float = 20.0) -> None:
    """A child, for good: SIGTERM, then SIGKILL (an Isaac app can ignore the
    first).  The children stay in this process's group, so a batch that stops
    this worker stops them too."""
    if proc is None or proc.poll() is not None:
        return
    for fn in (proc.terminate, proc.kill):
        try:
            fn()
        except OSError:
            pass
        try:
            proc.wait(timeout=grace)
            return
        except subprocess.TimeoutExpired:
            continue


def _write(path: Path, data: dict) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, default=str, indent=1))
    os.replace(tmp, path)


def run_episode(harness: str, model: str, task: str, level: str, seed: int, profile: str,
                out: str, engine: str = "isaac", clock: str = DEFAULT_CLOCK,
                record: bool = False, gpu: Optional[int] = None, say=print,
                template: Optional[str] = None, plant: Optional[Dict[str, str]] = None) -> dict:
    """One episode with a fresh agent, into the cell's folder of the shared
    layout; returns what was written there (``result.json`` or ``api_error.json``).
    ``template`` and ``plant`` are the self-test's: another prompt, and files put
    in the agent's home (by their paths in it) before it starts."""
    from ruibench.runner import heartbeat
    from ruikit.serve.session import _ask, _state

    provenance.launch()                             # the code and interpreter, as started
    folder = Path(out) / cells.agent_dir(harness, model, clock) / cells.stem(task, level, seed,
                                                                            profile)
    done = _state(folder) if folder.exists() else {}
    if done.get("status") == "over" and (folder / "agent.jsonl").exists() \
            and not (folder / "result.json").exists() and not (folder / "api_error.json").exists():
        # an earlier try ran the episode to its end and died collecting it: collect it.
        # Not one it did collect, and found not counted (the rules broken, the agent cut
        # off): that is run again by a new agent (v2.3.2: collected again, it was the
        # same each time, and after three tries lost)
        t = float(done.get("wall_seconds") or 0.0)
        return _collect(harness, model, task, level, seed, profile, folder, clock,
                        engine, agent_code=None, wall_seconds=t, agent_seconds=None,
                        filmed=record)
    if (folder.exists() and done.get("status") != "over" and (folder / "agent.jsonl").exists()
            and not (folder / "result.json").exists() and _ended(folder)):
        # an earlier try's episode ended, but its session was stopped before it wrote its
        # state (filming a long episode): finish it from its logs and collect it, not run
        # it again at its whole cost (v2.3.6)
        (folder / "api_error.json").unlink(missing_ok=True)
        # its span on the files' own clock: the file server's clock is not this
        # machine's (v2.3.6: 7 h 51 min apart, and a salvaged episode took -6 h)
        started, logs = folder / "prompt.txt", [p.stat().st_mtime for p in folder.glob("*.jsonl")]
        span = max(logs) - started.stat().st_mtime if started.exists() and logs else 0.0
        _settle(folder, record, time.time() - max(span, 0.0))
        wall = float(_state(folder).get("wall_seconds") or 0.0)
        return _collect(harness, model, task, level, seed, profile, folder, clock,
                        engine, agent_code=None, wall_seconds=wall, agent_seconds=None,
                        filmed=record)
    if folder.exists():
        _keep_try(folder)
        shutil.rmtree(folder)                       # an earlier try's files are not this one's
    folder.mkdir(parents=True)
    t0 = time.time()
    _write(folder / "running.json", {"pid": os.getpid(), "host": socket.gethostname(), "t": t0,
                                     "harness": harness, "model": model})
    # PYTHONFAULTHANDLER: a session that hangs can be made to write its threads' stacks
    env = dict(os.environ, OMNI_KIT_ACCEPT_EULA="YES", PYTHONPATH=str(ROOT), PYTHONFAULTHANDLER="1")
    # the batch watches this worker's heartbeat: the session must not write the same file
    # (two writers race on it, and one of them dies)
    env.pop("RUIBENCH_HEARTBEAT", None)
    if gpu is not None:                             # else the batch has chosen the card
        env.update(CUDA_VISIBLE_DEVICES=str(gpu), RUIBENCH_ISAAC_GPU="0")
    server = subprocess.Popen(
        [PYTHON, "-m", "ruikit.serve.session", "start", "--dir", str(folder), "--task", task,
         "--level", level, "--seed", str(seed), "--profile", profile, "--engine", engine,
         "--clock", clock, "--agent",
         f"{harness} {model} (clean context) over MCP"] + (["--record"] if record else [])
        + ["--serve"], cwd=str(ROOT), env=env, stdout=open(folder / "server.log", "w"),
        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    # named by nothing it could learn from: its working folder's path is in its context
    agent, home = None, AGENT_HOMES / f"agent-{uuid.uuid4().hex[:12]}"
    try:
        while _state(folder).get("status") not in ("ready", "failed"):
            heartbeat.beat(heartbeat.SIM, stage="session starting")
            if server.poll() is not None or time.time() - t0 > STARTUP_SECONDS:
                raise RuntimeError(f"the session did not start: {_state(folder).get('error')}")
            time.sleep(2.0)
        if _state(folder).get("status") == "failed":
            raise RuntimeError(f"the session failed: {_state(folder).get('error')}")
        prompt = prompt_for(folder, template)
        (folder / "prompt.txt").write_text(prompt)
        argv, agent_env = agent_command(harness, model, prompt, home, folder)
        for rel, text in (plant or {}).items():
            (home / rel).parent.mkdir(parents=True, exist_ok=True)
            (home / rel).write_text(text)
        t_agent = time.time()
        agent = subprocess.Popen(argv, cwd=str(home / "work"), env=agent_env,
                                 stdout=open(folder / "agent.jsonl", "w"),
                                 stderr=open(folder / "agent.err", "w"),
                                 stdin=subprocess.DEVNULL)
        say(f"   {harness} {model} on {folder.name} (pid {agent.pid})")
        ended_at = None
        while True:
            heartbeat.beat(heartbeat.AGENT, agent=harness)
            over = _state(folder).get("status") == "over" or server.poll() is not None
            if over and ended_at is None:
                ended_at = time.time()
            if agent.poll() is not None:
                break
            if ended_at is not None and time.time() - ended_at > GRACE_SECONDS:
                break                                   # the summary takes too long: done anyway
            time.sleep(5.0)
        agent_code = agent.poll()
        _stop(agent)
        if (server.poll() is None and _state(folder).get("status") == "ready"
                and not _ended(folder)):
            try:                                        # the agent stopped first
                _ask(folder, {"op": "stop"}, timeout=STOP_ANSWER_SECONDS)
            except socket.timeout:                      # the session hangs: keep its stacks
                _dump_stacks(server)
            except OSError:                             # or the episode was just ending by itself
                pass
        _wait_finished(server)                          # it scores, films and leaves
        _stop(server)
        _settle(folder, record, t0)
    except BaseException:
        _stop(agent)
        _stop(server)
        (folder / "running.json").unlink(missing_ok=True)
        raise
    finally:
        _keep_codex_record(home, folder)
        shutil.rmtree(home, ignore_errors=True)
        for left in claude_leftovers(home / "work"):
            shutil.rmtree(left, ignore_errors=True)
    return _collect(harness, model, task, level, seed, profile, folder, clock, engine,
                    agent_code=agent_code, wall_seconds=round(time.time() - t0, 1),
                    agent_seconds=round(time.time() - t_agent, 1), filmed=record)


#: how long a session may take to finish an episode that has ended -- score it, and
#: film it: a long filmed episode's video took over 5 min on a loaded machine, and a
#: session stopped before writing its state had its episode run again (v2.3.6)
FINISH_SECONDS = 1800.0


def _ended(folder: Path) -> bool:
    """The episode has ended: its own log holds its outcome (written as it
    ends, before the session scores and films it)."""
    from ruikit.serve.session_server import _outcome_row

    return bool(_outcome_row(folder).get("outcome"))


def _wait_finished(server: subprocess.Popen) -> None:
    """While the session lives, up to ``FINISH_SECONDS``: it finishes on its own."""
    from ruibench.runner import heartbeat

    t0 = time.time()
    while server.poll() is None and time.time() - t0 < FINISH_SECONDS:
        heartbeat.beat(heartbeat.SIM, stage="session finishing")
        time.sleep(2.0)


def _settle(folder: Path, record: bool, t0: float) -> None:
    """An episode that ended but whose session stopped before it wrote its
    state -- stopped while filming -- is finished here, from its logs: it
    counts, and is not run again by a new agent (v2.3.6: 172 of Claude Code's
    episodes on v2.3.4-v2.3.5 were run again so, each its whole cost)."""
    from ruikit.serve.session import _state
    from ruikit.serve.session_server import finalize

    if _state(folder).get("status") == "over" or not _ended(folder):
        return
    try:
        finalize(folder, film=record, wall_seconds=time.time() - t0)
    except Exception as exc:                        # pragma: no cover - its state says why not
        print(f"   could not finish {folder.name} from its logs: {type(exc).__name__}: {exc}",
              file=sys.stderr)


#: a session that does not answer its stop within this long hangs: its threads'
#: stacks go into its server.log before it is stopped (v2.3.4: four sessions hung
#: on a program's call, and the stop then waited 30 min more on each)
STOP_ANSWER_SECONDS = 60.0
#: what is kept of a try not counted when the cell is run again in a clean folder
KEPT_OF_A_TRY = ("agent.jsonl", "agent.err", "codex-rollout.jsonl", "server.log", "session.json",
                 "api_error.json", "prompt.txt", "notes.json")


def _dump_stacks(server: subprocess.Popen) -> None:
    """A hung session's threads' stacks, into its server.log: it runs with
    PYTHONFAULTHANDLER, which writes them on SIGABRT before it ends."""
    try:
        server.send_signal(signal.SIGABRT)
        server.wait(timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        pass


def _keep_try(folder: Path) -> None:
    """An earlier try with a transcript and no result -- not counted, or stopped
    over its time (killed: it wrote nothing more) -- kept in ``<agent>/.tries/``,
    since the cell runs again in a clean folder (v2.3.4: the record of each hung
    session was lost to the retry that wiped it)."""
    if not (folder / "agent.jsonl").exists() and not (folder / "codex-rollout.jsonl").exists():
        return
    if (folder / "result.json").exists():
        return
    keep = folder.parent / ".tries" / f"{folder.name}.{int(time.time())}"
    keep.mkdir(parents=True, exist_ok=True)
    # and the episode's own logs, from which an episode that ended can still be scored
    logs = [p.name for p in folder.glob(f"{folder.name}*.jsonl")]
    for name in (*KEPT_OF_A_TRY, *logs):
        if (folder / name).exists():
            shutil.copy2(folder / name, keep / name)


def _collect(harness: str, model: str, task: str, level: str, seed: int, profile: str,
             folder: Path, clock: str, engine: str, agent_code, wall_seconds,
             agent_seconds, filmed: bool = False) -> dict:
    """The cell's ``result.json`` -- or ``api_error.json`` for a run that does
    not count -- from what the episode and the agent left in its folder.
    ``filmed``: the episode was filmed (which changes the pictures it shows)."""
    from ruikit.experiments.rules_audit import audit_transcript
    from ruikit.experiments.usage import claude_usage, codex_usage
    from ruikit.serve.session import _state

    state = _state(folder)
    codex = harness in CODEX_MODES
    transcript = folder / ("codex-rollout.jsonl" if codex else "agent.jsonl")
    usage = (codex_usage(transcript) if codex else claude_usage(transcript)) or {}
    audit = (audit_transcript(str(transcript), own=str(folder)) if transcript.exists() else None)
    broken = list((audit or {}).get("flagged") or [])
    broken += [{"reached it": text} for text in (audit or {}).get("outside") or []]
    said = _tail(folder / "agent.err") + _tail(folder / "agent.jsonl", 4000)
    # cut off: the episode ended because the agent stopped, and it stopped on an
    # error -- a usage limit, the API -- not by its own choice
    cut = state.get("reason") in (None, "agent_stopped") and (
        agent_code not in (0, None) or _failed(harness, folder / "agent.jsonl")
        or bool(usage.get("rate_limit_reached")))
    # not the agent's doing: it did not get the session's tools, or never used them
    unarmed = (None if codex else claude_start(folder / "agent.jsonl")) or (
        "the agent made no call to the session" if not (audit or {}).get("session_calls")
        else None)
    (folder / "running.json").unlink(missing_ok=True)
    if cut or broken or unarmed or state.get("status") != "over":
        why = ("the agent broke the rules: " + json.dumps(broken)[:600] if broken else
               f"the agent was not set up: {unarmed}" if unarmed and not cut else
               f"the agent was cut off (exit {agent_code}): " + said.strip()[-600:])
        row = {"harness": harness, "model": model, "task": task, "level": level, "seed": seed,
               "profile": profile, "api_error": why, "quota": cut and _quota(said, usage)}
        _write(folder / "api_error.json", row)
        return row
    row = cells.result_row(
        harness=harness, model=model, task=task, level=level, seed=seed, profile=profile,
        folder=folder, clock=clock, engine=engine, reason=state.get("reason"),
        success=state.get("success"), replies=state.get("replies"),
        sim_seconds=state.get("sim_seconds"), wall_seconds=wall_seconds,
        agent_seconds=agent_seconds, usage=usage, agent_exit=agent_code,
        tool_mode=CODEX_MODES.get(harness, "direct"), agent_cli=cli_version(harness),
        effort=effort_of(harness, model), filmed=bool(filmed),
        video="video.mp4" if (folder / "video.mp4").exists() else None,
        audit={"tool_uses": (audit or {}).get("tool_uses"), "flagged": 0},
        summary=_summary(harness, folder / "agent.jsonl"),
        provenance=provenance.stamp(1, 1))          # its session started its own simulator
    _write(folder / "result.json", row)
    if row.get("video"):                            # the frames are in the video now
        shutil.rmtree(folder / "frames", ignore_errors=True)
    return row


def _failed(harness: str, path: Path) -> bool:
    """Did the agent's own output end on an error (rather than a summary)?"""
    for line in reversed(_tail(path, 200000).splitlines()):
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if harness == "claude-code" and d.get("type") == "result":
            return bool(d.get("is_error"))
        if harness in CODEX_MODES and d.get("type") in ("turn.failed", "error"):
            return True
        if harness in CODEX_MODES and d.get("type") == "turn.completed":
            return False
    return False


def _quota(said: str, usage: dict) -> bool:
    """The agent was cut off by its account, not by the cell: a usage limit, or
    a login that failed (v2.3.6: an expired login cut off every agent for 18 h,
    and each cell used up its tries and was lost), or a workspace's spend cap
    (v2.3.10: Codex's, unrecognised, had the runner start episodes for 27 rounds)."""
    low = said.lower()
    return bool(usage.get("rate_limit_reached")) or any(w in low for w in (
        "usage limit", "usage_limit", "quota", "credit balance", "limit reached", "spend cap",
        "failed to authenticate", "oauth session expired", "not logged in", "please run /login"))


def _tail(path: Path, n: int = 2000) -> str:
    try:
        return path.read_text(errors="replace")[-n:]
    except OSError:
        return ""


def _summary(harness: str, path: Path) -> str:
    """The agent's last words: its summary of the episode."""
    last = ""
    for line in _tail(path, 400000).splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if harness == "claude-code" and d.get("type") == "result":
            last = str(d.get("result") or last)
        elif harness in CODEX_MODES and d.get("type") == "item.completed":
            item = d.get("item") or {}
            if item.get("type") in ("agent_message", "assistant_message"):
                last = str(item.get("text") or last)
    return last[-3000:]


# ----------------------------------------------------------------------------
def _worker(spec_path: str) -> None:
    """The batch's worker for a ready-made agent: one job, then leave."""
    from ruikit.experiments.worker import API_EXIT, QUOTA_EXIT

    spec = json.loads(Path(spec_path).read_text())
    job = spec["jobs"][0]
    code = 0
    try:
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))   # stop the session and agent
        row = run_episode(job["harness"], job["model"], job["task"], job["level"], job["seed"],
                          job["profile"], spec["out"], engine=spec.get("engine", "isaac"),
                          clock=job.get("clock") or DEFAULT_CLOCK,
                          record=bool(spec.get("record")))
        print(json.dumps(row, ensure_ascii=False, default=str), flush=True)
        if "api_error" in row:
            code = QUOTA_EXIT if row.get("quota") else API_EXIT
    except Exception:
        import traceback

        traceback.print_exc()
        code = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worker", help=argparse.SUPPRESS)
    sub = ap.add_subparsers(dest="cmd")
    ep = sub.add_parser("episode", help="one episode with a fresh agent")
    ep.add_argument("--harness", required=True, choices=("codex", "codex-code", "claude-code"))
    ep.add_argument("--model", required=True, help="codex: gpt-5.6-sol, gpt-6-astra; "
                                                   "claude-code: claude-opus-5-5")
    ep.add_argument("--task", required=True)
    ep.add_argument("--level", default="easy")
    ep.add_argument("--seed", type=int, default=0)
    ep.add_argument("--profile", required=True)
    ep.add_argument("--out", required=True)
    ep.add_argument("--engine", default="isaac")
    ep.add_argument("--clock", default=DEFAULT_CLOCK,
                    help="8s (the main setting; any <N>s), paused or measured")
    ep.add_argument("--gpu", type=int, default=None)
    ep.add_argument("--record", action="store_true")
    ck = sub.add_parser("check", help="can this framework run here")
    ck.add_argument("harness", choices=("codex", "codex-code", "claude-code"))
    sf = sub.add_parser("selftest", help="does this agent get the session's tools and nothing "
                                         "else (ruikit.harness.selftest)")
    sf.add_argument("harness", choices=("codex", "codex-code", "claude-code"))
    sf.add_argument("--model", required=True)
    sf.add_argument("--out", default=str(ROOT / "runs" / "selftest"))
    args = ap.parse_args(argv)
    if args.worker:
        _worker(args.worker)
    if args.cmd == "check":
        why = preflight(args.harness)
        print(why or f"{args.harness}: ready")
        return 1 if why else 0
    if args.cmd == "selftest":
        from ruikit.harness.selftest import selftest

        why = preflight(args.harness)
        if why:
            print(why)
            return 1
        verdict = selftest(args.harness, args.model, Path(args.out))
        print(json.dumps({k: v for k, v in verdict.items() if k != "self_report"}, indent=1,
                         ensure_ascii=False))
        print("PASSED: only the session's tools" if verdict["passed"]
              else "FAILED: " + "; ".join(verdict["failures"]))
        return 0 if verdict["passed"] else 1
    if args.cmd == "episode":
        row = run_episode(args.harness, args.model, args.task, args.level, args.seed,
                          args.profile, args.out, engine=args.engine,
                          clock=args.clock, record=args.record, gpu=args.gpu)
        print(json.dumps(row, ensure_ascii=False, default=str))
        return 0 if "api_error" not in row else 1
    ap.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
