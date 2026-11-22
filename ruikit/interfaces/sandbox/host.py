"""The server side of ``code_run``: a sandboxed program process per episode.

The agent's code never runs in the tool server's process -- the CaP-X lesson:
its code namespace held the raw simulator (``env``) and ground truth (``obs``),
and a same-process Python sandbox can always be climbed out of
(``__subclasses__``, ``sys._getframe``).  Instead:

* **bubblewrap** (``bwrap``) runs the program in fresh namespaces: no network,
  no view of the repository, the task cards, the scorer, ``runs/`` or the home
  directory.  Only the Python interpreter, numpy and the child script are
  mounted, read-only; ``/tmp`` is an empty tmpfs.  Without bwrap the process
  still runs apart, with resource limits and an empty environment, and every
  result says ``isolation: "process"`` so it can be kept off a leaderboard.
* **the only way out is a tool call**: each call the program makes is one RPC
  answered by the episode, validated, charged and logged like any other.
* **the process is frozen between runs** (SIGSTOP): the namespace persists, but
  nothing computes while the agent is not running a program.  Its compute time
  inside a run -- not the time spent in tool calls -- is charged to the
  simulated clock.
"""

from __future__ import annotations

import json
import os
import select
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

CHILD = Path(__file__).with_name("child.py")
#: wall-clock cap on the program's own compute in one run (tool calls excluded)
RUN_COMPUTE_SECONDS = 30.0
#: wall-clock cap on one whole run, tool calls included
RUN_TOTAL_SECONDS = 1800.0


def _limits() -> None:
    import resource

    for what, value in ((resource.RLIMIT_AS, 4 << 30), (resource.RLIMIT_NOFILE, 256),
                        (resource.RLIMIT_FSIZE, 16 << 20), (resource.RLIMIT_CPU, 3600)):
        try:
            resource.setrlimit(what, (value, value))
        except (ValueError, OSError):
            pass


def _numpy_dir() -> Optional[Path]:
    try:
        import numpy
    except ImportError:
        return None
    return Path(numpy.__file__).resolve().parent


def bwrap_command(python: str) -> Optional[List[str]]:
    bwrap = shutil.which("bwrap")
    if not bwrap:
        return None
    base = Path(sys.base_prefix).resolve()
    cmd = [bwrap, "--unshare-all", "--die-with-parent",
           "--ro-bind", "/usr", "/usr"]
    for link in ("/lib", "/lib64", "/bin", "/sbin", "/lib32", "/libx32"):
        p = Path(link)
        if p.is_symlink():
            cmd += ["--symlink", os.readlink(link), link]
        elif p.exists():
            cmd += ["--ro-bind", link, link]
    if not str(base).startswith("/usr/"):
        cmd += ["--ro-bind", str(base), str(base)]
    npdir = _numpy_dir()
    if npdir is not None:
        cmd += ["--ro-bind", str(npdir), "/pkgs/numpy"]
        libs = npdir.parent / "numpy.libs"
        if libs.exists():
            cmd += ["--ro-bind", str(libs), "/pkgs/numpy.libs"]
    cmd += ["--ro-bind", str(CHILD), "/sandbox/child.py",
            "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/tmp",
            "--clearenv", "--setenv", "HOME", "/tmp", "--setenv", "PYTHONHASHSEED", "0",
            python, "-I", "-B", "/sandbox/child.py"]
    return cmd


def _base_python() -> str:
    exe = Path(sys.base_prefix) / "bin" / f"python{sys.version_info.major}.{sys.version_info.minor}"
    return str(exe if exe.exists() else Path(sys.executable).resolve())


class SandboxError(Exception):
    pass


class Sandbox:
    """One program process for one episode."""

    def __init__(self, isolation: str = "auto") -> None:
        self.isolation_wanted = isolation
        self.isolation = "none"
        self.proc: Optional[subprocess.Popen] = None

    # -- lifecycle -------------------------------------------------------------
    def _spawn(self) -> None:
        python = _base_python()
        cmd = None
        if self.isolation_wanted in ("auto", "bwrap"):
            cmd = bwrap_command(python)
            if cmd is None and self.isolation_wanted == "bwrap":
                raise SandboxError("bwrap is not installed")
        if cmd is not None:
            self.isolation = "bwrap"
            env = {}
        else:
            self.isolation = "process"
            cmd = [python, "-I", "-B", str(CHILD)]
            env = {"HOME": "/tmp", "PYTHONHASHSEED": "0"}
            npdir = _numpy_dir()
            if npdir is not None:
                env["RUIKIT_NUMPY"] = str(npdir.parent)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, text=True, bufsize=1, env=env,
                                     cwd="/tmp", start_new_session=True, preexec_fn=_limits)

    def _signal(self, sig) -> None:
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, sig)
            except ProcessLookupError:
                pass

    def close(self) -> None:
        if self.proc is None:
            return
        self._signal(signal.SIGCONT)
        self._signal(signal.SIGKILL)
        try:
            self.proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None

    # -- one run -----------------------------------------------------------------
    def run(self, code: str, tools: Dict[str, List[str]],
            on_call: Callable[[str, dict], dict], custom_args: Optional[dict] = None,
            compute_cap: float = RUN_COMPUTE_SECONDS) -> dict:
        """Run ``code`` (or call a custom tool's ``run`` with ``custom_args``);
        answer each tool call with ``on_call``.  Returns stdout, the value of
        ``result`` (or the tool's return), an error traceback, the frames the
        program asked to show, the compute seconds to charge, and the isolation."""
        if self.proc is None or self.proc.poll() is not None:
            self._spawn()
        self._signal(signal.SIGCONT)
        msg = {"op": "invoke" if custom_args is not None else "run", "code": code,
               "tools": tools}
        if custom_args is not None:
            msg["args"] = custom_args
        compute = 0.0
        started = time.time()
        try:
            self._send(msg)
            while True:
                t0 = time.time()
                line = self._read(compute_cap - compute, started)
                compute += time.time() - t0
                reply = json.loads(line)
                if reply.get("op") == "call":
                    result = on_call(reply.get("tool", ""), reply.get("args") or {})
                    self._send({"op": "result", "id": reply.get("id"), "result": result})
                elif reply.get("op") == "done":
                    reply["compute_s"] = round(compute, 4)
                    reply["isolation"] = self.isolation
                    return reply
        except SandboxError as exc:
            self.close()
            return {"stdout": "", "value": None, "error": str(exc), "shown": [],
                    "compute_s": round(compute, 4), "isolation": self.isolation}
        finally:
            self._signal(signal.SIGSTOP)

    def _send(self, msg: dict) -> None:
        self._still_there()
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise SandboxError(f"the program process died: {self._stderr()}") from exc

    def _read(self, budget: float, started: float) -> str:
        self._still_there()
        if budget <= 0:
            raise SandboxError(f"the program exceeded its {RUN_COMPUTE_SECONDS:g} s compute limit")
        if time.time() - started > RUN_TOTAL_SECONDS:
            raise SandboxError("the program ran too long")
        ready, _, _ = select.select([self.proc.stdout], [], [], budget)
        if not ready:
            raise SandboxError(f"the program exceeded its {RUN_COMPUTE_SECONDS:g} s compute limit")
        line = self.proc.stdout.readline()
        if not line:
            raise SandboxError(f"the program process died: {self._stderr()}")
        return line

    def _still_there(self) -> None:
        """The program's process can be closed while it waits on a tool: the
        call ended the episode, or a program run inside it failed and closed
        this sandbox.  Writing to it would crash the host (AttributeError) and
        take the whole run with it, so the call fails instead."""
        if self.proc is None:
            raise SandboxError("the program's process was stopped while it ran (the episode "
                               "ended, or a program inside it failed)")

    def _stderr(self) -> str:
        try:
            if self.proc and self.proc.poll() is not None:
                return (self.proc.stderr.read() or "")[-800:]
        except Exception:
            pass
        return ""
