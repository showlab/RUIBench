"""Progress heartbeats, so a hung simulator is noticed and killed.

A worker writes a small JSON file whenever it makes progress; the parent
watches its modification time.  The beat comes from the thread doing the work,
never from a helper thread -- a helper would keep beating while the main thread
is stuck, which is exactly the case this exists to catch.

Two phases, two patience levels: stepping physics should never go quiet for
long, while waiting on a model API can legitimately take a minute or more.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

ENV_VAR = "RUIBENCH_HEARTBEAT"
_last = 0.0

SIM = "sim"          # stepping the world, loading a scene
AGENT = "agent"      # waiting for a model to answer


def beat(phase: str = SIM, min_interval: float = 1.0, **info) -> None:
    """Record progress.  Cheap: at most one write per ``min_interval`` seconds,
    except that a change of phase is always written."""
    global _last
    path = os.environ.get(ENV_VAR)
    if not path:
        return
    now = time.time()
    if now - _last < min_interval and info.get("force") is None and phase == _phase():
        return
    _last = now
    payload = {"t": now, "phase": phase, "pid": os.getpid(), **info}
    tmp = f"{path}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)                  # atomic: a reader never sees half a file
    _set_phase(phase)


_current_phase = SIM


def _phase() -> str:
    return _current_phase


def _set_phase(phase: str) -> None:
    global _current_phase
    _current_phase = phase


def read(path: str) -> Optional[dict]:
    """The latest beat, with ``age`` in seconds.

    Age comes from the timestamp *inside* the beat, written by the worker's own
    clock -- never from the file's modification time.  On the NFS home here the
    server's clock runs hours apart from the host's, so mtime-based ages came
    out negative and the watchdog could never see a stall.
    """
    try:
        with open(path) as fh:
            data = json.load(fh)
        data["age"] = time.time() - float(data["t"])
        return data
    except (OSError, ValueError, KeyError, TypeError):
        return None
