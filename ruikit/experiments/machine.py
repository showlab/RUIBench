"""The machine: free GPU memory, the load, other Isaac apps, and the launch clock
shared by every runner."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Dict, Optional, Sequence

from ruikit.core.paths import ROOT

#: one RUIKit episode's Isaac takes ~3 GB of GPU memory
MIN_FREE_MB = 7000


#: episodes one Isaac process runs: one, since a reused stage keeps PhysX's build
#: layout and the renderer's state, so an episode's outcome would depend on what
#: ran before it (starting Isaac and building the stage takes about two minutes)
PER_WORKER = 1


#: seconds between two starts on one card, and between any two starts by any
#: runner on this machine (the time is written into the files, never read from
#: their stamps; gate_c.py keeps the second file)
STAGGER, LAUNCH_GAP = 60.0, 60.0


#: no new Isaac app while the 1-minute load is over this share of the cores:
#: a start takes about 40 of them, and the machine has other users
MAX_LOAD = 0.8


#: where the machine's runners keep their shared start clock -- the repository's
#: runs/tmp, or ``RUIBENCH_LAUNCH_DIR`` for a copy of the code run beside it
LAUNCH_DIR = Path(os.environ.get("RUIBENCH_LAUNCH_DIR") or ROOT / "runs" / "tmp")


LAUNCH_FILES = (LAUNCH_DIR / "isaac.last_launch", LAUNCH_DIR / "gate_c.last_launch")


# --------------------------------------------------------------------------
# the machine: cards, other Isaac apps, the shared launch clock
# --------------------------------------------------------------------------
def free_mb() -> Dict[int, int]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
    out = {}
    for line in q.strip().splitlines():
        try:
            i, used, total = (int(v) for v in line.split(","))
        except ValueError:
            continue
        out[i] = total - used
    return out


def pick_gpu(free: Dict[int, int], mine: Sequence[int], last_start: Dict[int, float],
             now: float, only: Sequence[int] = ()) -> Optional[int]:
    """The card with the most memory free, at least MIN_FREE_MB, none of ours on
    it, and none of ours started on it in the last STAGGER seconds -- among
    ``only``, when given (then several of ours may share a card)."""
    ok = [g for g, mb in sorted(free.items(), key=lambda kv: -kv[1])
          if mb >= MIN_FREE_MB and g not in mine and now - last_start.get(g, -1e18) >= STAGGER
          and (not only or g in only)]
    return ok[0] if ok else None


def _ppid(pid: int) -> int:
    try:
        return int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
    except (OSError, ValueError, IndexError):
        return 0


def _kit_app(pid: int) -> bool:
    """Is this process a simulator: has it loaded Kit (``libcarb``, its core)?
    A GPU also carries what is none -- the vision server of perception from
    pixels, a local language model -- and counting those would leave a batch
    a slot or two short of ``--max-isaac``."""
    try:
        with open(f"/proc/{pid}/maps") as maps:
            return any("libcarb.so" in line for line in maps)
    except OSError:
        return False                      # gone, or not ours to read


def other_isaac(ours: Sequence[int]) -> int:
    """This user's Isaac apps on a GPU that are not under one of ``ours``: the
    sweep's, a session's, another runner's.  Ours are counted by launch: an
    Isaac app takes a minute to reach the GPU."""
    pids = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                          capture_output=True, text=True).stdout.split()
    n = 0
    for pid in pids:
        try:
            pid = int(pid)
            if os.stat(f"/proc/{pid}").st_uid != os.getuid():
                continue
        except (OSError, ValueError):
            continue
        if not _kit_app(pid):
            continue
        p = pid
        while p > 1 and p not in ours:
            p = _ppid(p)
        n += p not in ours
    return n


def busy(share: float) -> bool:
    """Is the machine's 1-minute load over ``share`` of its cores?"""
    try:
        return os.getloadavg()[0] > share * (os.cpu_count() or 1)
    except OSError:
        return False


def last_launch() -> float:
    """When any runner last started an Isaac app (the time inside the files)."""
    best = 0.0
    for f in LAUNCH_FILES:
        try:
            best = max(best, float(f.read_text().strip() or 0))
        except (OSError, ValueError):
            pass
    return best


def stamp_launch(now: float) -> None:
    for f in LAUNCH_FILES:
        try:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(f"{now}\n")
        except OSError:
            pass
