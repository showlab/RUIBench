"""Can this machine run the benchmark -- the checks, as a command.

The design docs open with a checklist that must be run before any task code is
written.  Under the Isaac plan that checklist is:

  A1  Isaac Sim imports and a SimulationApp starts headless
  A2  two arms accept two independent IK targets in the same step -- concurrently
  A3  photorealistic rendering throughput at 512x512, two cameras
  A4  every catalogue asset resolves on the asset server

Every check reports pass / fail / skipped with a reason, and the whole thing is
written to ``runs/gate-a.json`` in a source checkout, so the answers are kept
instead of living in somebody's terminal history (in the user's cache directory
for an installed package).
"""

from __future__ import annotations

import json
import sys
import platform
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core.paths import data_dir

#: Where the report goes: runs/ in a source checkout, the user's cache
#: otherwise -- never the current directory, wherever that happens to be.
REPORT_PATH = data_dir("doctor", "runs") / "gate-a.json"
#: Where an interpreter without Isaac writes.  Kept apart so a development
#: environment can never overwrite the report of the machine that runs results.
DEV_REPORT_PATH = data_dir("doctor", "runs") / "gate-a.dev.json"


@dataclass
class Check:
    id: str
    name: str
    status: str = "skipped"          # pass | fail | skipped
    detail: str = ""
    data: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        mark = {"pass": "PASS", "fail": "FAIL", "skipped": "SKIP"}[self.status]
        line = f"  [{mark}] {self.id} {self.name}"
        return f"{line}\n         {self.detail}" if self.detail else line


def python_and_gpu() -> Check:
    c = Check("A0", "python and GPU")
    data = {"python": platform.python_version(), "platform": platform.platform()}
    try:
        import subprocess
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=20)
        data["gpus"] = [g.strip() for g in out.stdout.strip().splitlines() if g.strip()]
    except Exception as exc:
        data["gpus"] = []
        data["gpu_error"] = str(exc)
    c.data = data
    ok = data["python"].startswith(("3.10", "3.11")) and bool(data["gpus"])
    c.status = "pass" if ok else "fail"
    c.detail = (f"python {data['python']}, "
                f"{len(data['gpus'])} GPU(s): {', '.join(data['gpus'][:2]) or 'none'}"
                + ("" if ok else "  <- Isaac Sim needs Python 3.10/3.11 and an RTX GPU"))
    return c


def isaac_import() -> Check:
    """A1: is Isaac Sim installed, and may it start?

    The doctor never starts the simulator in its own process -- a simulation app
    can be created once per process and does not tear down cleanly -- so this
    only checks that it *could*.  The subprocess checks below prove it does.
    """
    c = Check("A1", "Isaac Sim installed and licensed to start")
    from .engines.isaac import app
    if not app.is_available():
        c.status = "fail"
        c.detail = "isaacsim is not installed in this interpreter; see `ruibench doctor --hint`"
        return c
    if not app.eula_accepted():
        c.status = "fail"
        c.detail = ("isaacsim is installed but NVIDIA's EULA has not been accepted. "
                    "Read the terms, then `export OMNI_KIT_ACCEPT_EULA=YES`. "
                    "RUIBench does not accept it for you.")
        return c
    try:
        import importlib.metadata as md
        c.data = {"isaacsim": md.version("isaacsim")}
    except Exception:
        c.data = {}
    c.status = "pass"
    c.detail = f"isaacsim {c.data.get('isaacsim', '?')} installed; EULA accepted"
    return c


def isaac_lab() -> Check:
    """A1b: Isaac Lab, which supplies robot configs, cameras and the IK controller."""
    c = Check("A1b", "Isaac Lab installed")
    import importlib.util
    missing = [m for m in ("isaaclab", "isaaclab_assets") if importlib.util.find_spec(m) is None]
    if missing:
        c.status = "fail"
        c.detail = f"missing: {', '.join(missing)}; see `ruibench doctor --hint`"
        return c
    c.status = "pass"
    c.detail = ("isaaclab and isaaclab_assets installed (importable only after the app "
                "starts -- they pull in carb)")
    return c


def isaac_ready() -> Optional[str]:
    from .engines.isaac import app
    if not app.is_available():
        return "isaacsim is not installed; see `ruibench doctor --hint`"
    if not app.eula_accepted():
        return "blocked by A1: NVIDIA's EULA has not been accepted"
    return None


def run_isaac_child(module: str, *args: str, timeout: int = 900) -> dict:
    """Run ``python -m module args...`` -- a process that starts Isaac, prints
    one ``RESULT {json}`` line and leaves -- and return that result.

    A fresh process each time: the app starts once per process, a stage with
    cameras must not be torn down in-process, and a native crash stays there.
    """
    import os
    import subprocess
    import sys

    env = dict(os.environ)
    env.setdefault("RUIBENCH_ISAAC_GPU", "0")
    env.setdefault("CUDA_VISIBLE_DEVICES", env["RUIBENCH_ISAAC_GPU"])
    try:
        proc = subprocess.run([sys.executable, "-m", module, *args],
                              capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"pass": False, "error": f"timed out after {timeout} s"}
    for line in reversed(proc.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT "):])
    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
    return {"pass": False, "error": f"exit {proc.returncode}, no result: {' | '.join(tail)[:400]}"}


def _run_isaac_check(name: str, timeout: int = 900) -> dict:
    """Run one hardware check in a fresh process and return its RESULT."""
    return run_isaac_child("ruibench.engines.isaac.checks", name, timeout=timeout)


def _subprocess_check(cid: str, name: str, check: str, describe) -> Check:
    c = Check(cid, name)
    blocked = isaac_ready()
    if blocked:
        c.detail = blocked
        return c
    result = _run_isaac_check(check)
    c.data = result
    c.status = "pass" if result.get("pass") else "fail"
    c.detail = result.get("error") or describe(result)
    return c


def controller_reach() -> Check:
    """A1d: the shared controller reaches the poses tasks use.

    Checked first because the whole design rests on it: change the interface,
    not the controller.  It failed three ways before it passed -- solving IK at
    the control rate went unstable, controlling the hand frame instead of the
    fingertips put targets 10 cm off, and starting from Isaac Lab's tucked
    default pose stalled joints at their limits.
    """
    return _subprocess_check(
        "A1d", "shared controller reaches task poses", "ik",
        lambda r: f"worst position error {r['worst_pos_err_mm']} mm over "
                  f"{len(r['targets'])} targets (limit 10 mm)")


def dual_arm_concurrency() -> Check:
    """A2: two arms, two goals, commanded in the same step -- both move at once.

    If it fails, the two-arm tasks (WK4, CL1, DI1, DI5) cannot run as
    designed: each needs both arms moving at once.
    """
    return _subprocess_check(
        "A2", "two arms move concurrently", "dual_arm",
        lambda r: f"after 1 s the arms had moved {r['moved_at_1s_cm'][0]} and "
                  f"{r['moved_at_1s_cm'][1]} cm; final errors {r['final_err_mm'][0]} and "
                  f"{r['final_err_mm'][1]} mm")


def render_throughput() -> Check:
    """A3: what rendering the observation costs -- this sets the concurrency plan."""
    return _subprocess_check(
        "A3", "render throughput at 512x512, two cameras", "render",
        lambda r: f"{r['ms_per_control_step']} ms per control step with both cameras "
                  f"rendered every step ({r['realtime_factor']}x realtime, an upper bound); "
                  f"agentview sees {r['visible_from_agentview']}")


def table_surface() -> Check:
    """A3b: the work surface is where the world convention says it is."""
    return _subprocess_check(
        "A3b", "work surface at table_height", "table",
        lambda r: f"dropped cubes rest on a surface at {r['surface_z']} (design 0.800)")


def asset_catalogue() -> Check:
    """A4: every catalogue path resolves on the asset server.

    Over HTTP -- the public asset root is plain HTTPS, so no simulator is needed.
    """
    c = Check("A4", "Isaac asset catalogue resolves")
    from .engines.isaac import assets
    try:
        results = assets.verify()
    except Exception as exc:
        c.status = "fail"
        c.detail = f"{type(exc).__name__}: {exc}"
        return c
    assets.mark_verified(results)
    missing = sorted(k for k, ok in results.items() if not ok)
    c.data = {"results": results, "missing": missing,
              "primitives": sorted(assets.primitives())}
    c.status = "fail" if missing else "pass"
    c.detail = (f"{len(results) - len(missing)}/{len(results)} resolved"
                + (f"; missing: {', '.join(missing)}" if missing else "")
                + f"; built from primitives by design: {', '.join(sorted(assets.primitives()))}")
    return c


def warp_pin() -> Check:
    """A1c: pip's warp must match the warp Kit bundles.

    Two different warp native libraries in one process corrupted the heap
    intermittently -- 3 of 5 two-arm runs aborted with warp-lang 1.17, 0 of 8
    after pinning 1.5.0.  Cheap to check, expensive to rediscover.
    """
    c = Check("A1c", "warp-lang matches Kit's bundled warp")
    try:
        import importlib.metadata as md
        import pathlib
        pip_version = md.version("warp-lang")
    except Exception:
        c.status = "skipped"
        c.detail = "warp-lang is not installed"
        return c
    bundled = None
    try:
        import isaacsim
        root = pathlib.Path(isaacsim.__file__).parent / "extscache"
        for d in root.glob("omni.warp.core-*"):
            bundled = d.name.split("-")[1].split("+")[0]
    except Exception:
        pass
    c.data = {"pip": pip_version, "kit": bundled}
    if bundled is None:
        c.status = "skipped"
        c.detail = f"pip warp-lang {pip_version}; Kit's bundled warp not found"
    elif pip_version == bundled:
        c.status = "pass"
        c.detail = f"warp-lang {pip_version} == Kit's omni.warp.core {bundled}"
    else:
        c.status = "fail"
        c.detail = (f"pip warp-lang {pip_version} != Kit's {bundled}. Two warp native "
                    f"libraries in one process corrupt the heap intermittently; "
                    f"`pip install warp-lang=={bundled}`")
    return c


def run(checks: Optional[List[str]] = None, write: bool = True,
        on_result=None, on_start=None) -> dict:
    """Run the checks in order, reporting each one as soon as it finishes.

    The hardware checks take tens of seconds each; printing only at the end
    made a healthy run indistinguishable from a hung one.
    """
    wanted = set(checks or [])
    all_checks = [python_and_gpu, warp_pin, isaac_import, isaac_lab, controller_reach,
                  dual_arm_concurrency, render_throughput, table_surface,
                  asset_catalogue]
    results = []
    for fn in all_checks:
        if wanted and fn.__name__ not in wanted:
            continue
        if on_start:
            on_start(fn)
        c = fn()
        results.append(c)
        if on_result:
            on_result(c)
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engine_plan": "isaac",
        "checks": [asdict(c) for c in results],
        "summary": {s: sum(1 for c in results if c.status == s)
                    for s in ("pass", "fail", "skipped")},
    }
    from .engines.isaac import app
    path = REPORT_PATH if app.is_available() else DEV_REPORT_PATH
    report["interpreter"] = sys.executable
    report["written_to"] = str(path)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report
