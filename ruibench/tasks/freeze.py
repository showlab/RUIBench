"""Freezing the benchmark for the test split.

Results on the test split compare only while everything that shapes an
episode stays as it was: the missions (task cards and their code), the
scoring, the engines, the interfaces and the Harnesses (RUIKit's profiles and
tools, the simple loop, and how a ready-made agent is set up, served and
audited) and ``T_ref``.  ``ruibench freeze NAME`` records a fingerprint of those
files, and ``T_ref``, in ``frozen.json`` beside the splits, and marks the
splits frozen.  From then on ``ruibench validate`` lists every file that
differs from the fingerprint, and the splits can no longer be regenerated.
A change made on purpose is a new version: freeze again under a new name, and
results taken on the old version stay with the old version.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from ..core.errors import ConfigError

PATH = Path(__file__).with_name("frozen.json")
#: what shapes an episode, relative to the checkout
FINGERPRINTED = (
    "ruibench/core/*.py",
    "ruibench/tasks/*.py", "ruibench/tasks/impl/*.py", "ruibench/tasks/cards/*.yaml",
    "ruibench/scoring/*.py", "ruibench/events/*.py", "ruibench/executor/*.py",
    "ruibench/engines/*.py", "ruibench/engines/isaac/*.py", "ruibench/engines/isaac/assets.yaml",
    "ruibench/runner/episode.py",
    # RUIKit: the interfaces, every tool, the episode and the perception; the
    # Harnesses -- the simple loop, and a ready-made agent's prompt, what it is
    # given and its self-test; the session an agent reaches and the words it reads
    # about its tools; and the audit that decides which agents' runs count
    "ruikit/configs/interfaces.yaml", "ruikit/core/*.py", "ruikit/env/*.py",
    "ruikit/actions/*.py", "ruikit/interfaces/**/*.py", "ruikit/perception/**/*.py",
    "ruikit/harness/**/*.py", "ruikit/extras/**/*.py", "ruikit/serve/*.py",
    "ruikit/experiments/rules_audit.py",
    # the minimal loop's conversation: what of its history the model is sent, in
    # either endpoint's shape (v2.3.5: a full context cuts it)
    "ruikit/models/conversation.py", "ruikit/models/chat_completions.py",
    "ruikit/models/responses.py",
)
#: never fingerprinted: the mock engine never produces a result, and fixing it
#: should not void the test results (v1.2).  What the Isaac engine shares with
#: it -- ``in_grasp`` -- lives in ``engines/base.py``.  Nor is the real station
#: (``engines/franka/``, ``tasks/real/``, ``ruikit/experiments/real.py``; v2.4):
#: its robot, calibration and tables change with the lab, no simulation result
#: depends on them, and each real result records its commit
UNFINGERPRINTED = ("ruibench/engines/mock.py",)


def _root() -> Path:
    from ..core.paths import checkout

    root = checkout()
    if root is None:
        raise ConfigError("the fingerprint is taken over a source checkout; this is an installed "
                          "package")
    return root


def fingerprint(root: Optional[Path] = None) -> Dict[str, str]:
    """``{path: sha256}`` of every fingerprinted file."""
    root = root or _root()
    out = {}
    for pattern in FINGERPRINTED:
        for path in sorted(root.glob(pattern)):
            rel = path.relative_to(root).as_posix()
            if path.is_file() and rel not in UNFINGERPRINTED:
                out[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def load(path: Optional[Path] = None) -> Optional[dict]:
    path = path or PATH
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def freeze(name: str, path: Optional[Path] = None, splits_path: Optional[Path] = None) -> dict:
    """Record the fingerprint and ``T_ref`` under ``name``, and mark the splits
    frozen with that name."""
    from ..core.config import DEFAULT_T_REF_SECONDS
    from . import splits

    path = path or PATH
    record = {"name": name, "date": date.today().isoformat(),
              "t_ref_seconds": DEFAULT_T_REF_SECONDS, "files": fingerprint()}
    path.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    splits_path = splits_path or splits.PATH
    data = splits.load(splits_path)
    data["frozen"] = name
    splits_path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return record


def drift(path: Optional[Path] = None, now: Optional[Dict[str, str]] = None) -> List[str]:
    """What differs from the frozen fingerprint, one line each ([] if nothing
    is frozen)."""
    from ..core.config import DEFAULT_T_REF_SECONDS

    record = load(path)
    if record is None:
        return []
    now = fingerprint() if now is None else now
    was = record["files"]
    out = [f"changed: {p}" for p in sorted(was) if p in now and now[p] != was[p]]
    out += [f"removed: {p}" for p in sorted(was) if p not in now]
    out += [f"added: {p}" for p in sorted(now) if p not in was]
    if record.get("t_ref_seconds") != DEFAULT_T_REF_SECONDS:
        out.append(f"T_ref: {record.get('t_ref_seconds')} s when frozen, {DEFAULT_T_REF_SECONDS} s now")
    return out
