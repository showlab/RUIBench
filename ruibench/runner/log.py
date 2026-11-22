"""The episode log.

Everything needed to re-score an episode without re-running it, and everything
needed to reproduce it: model id and snapshot date, inference config, prompt
hash, service version, sensor permissions, date (details §4.3).

Logs are JSON Lines: one episode per line, so a 7,440-episode run is a single
file that streams.
"""

from __future__ import annotations

import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence

#: 2 (v15): the trajectory holds the whole world state (``runner/snapshot``),
#: and ``scoring`` what else the scorer was told -- enough to score again
SCHEMA_VERSION = "ruibench-log/2"


@dataclass
class EpisodeLog:
    run_id: str
    task_id: str
    level: str
    seed: int
    interface: str
    agent: str
    engine: str
    clock: str
    config: Mapping[str, Any] = field(default_factory=dict)
    #: the instruction the episode started with (changes are in ``events``)
    instruction: str = ""
    #: the hint told with it (``TaskInstance.hint``)
    hint: str = ""
    #: ``SceneSpec.summary()``: enough to draw the episode from its log alone
    scene: Mapping[str, Any] = field(default_factory=dict)
    #: what the engine actually ran with -- rates, device -- as opposed to
    #: what ``config`` asked for
    engine_setup: Mapping[str, Any] = field(default_factory=dict)
    turns: List[dict] = field(default_factory=list)
    events: List[dict] = field(default_factory=list)
    handles: List[dict] = field(default_factory=list)
    feedback: List[dict] = field(default_factory=list)
    milestones_hit: Mapping[str, float] = field(default_factory=dict)
    trajectory: List[dict] = field(default_factory=list)
    score: Mapping[str, Any] = field(default_factory=dict)
    #: what the scorer was told besides the trajectory (timed out, budget
    #: used up, parse errors, whether the engine has physics)
    scoring: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, float] = field(default_factory=dict)
    #: the task's own numbers, reported beside success (``TaskInstance.extra_metrics``)
    task_metrics: Mapping[str, Any] = field(default_factory=dict)
    #: recorded camera frames: {t, camera, kind, path relative to config.record_dir}
    frames: List[dict] = field(default_factory=list)
    #: the demonstrations played before the episode: {caption, seconds, frames}
    demonstrations: List[dict] = field(default_factory=list)
    condition: str = "main"          # main | zero_latency | matched_control | negative_control
    started_at: str = ""
    finished_at: str = ""
    schema: str = SCHEMA_VERSION
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def stamp(self, when: str = "start") -> None:
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        if when == "start":
            self.started_at = now
            from ..core.canary import CANARY_GUID

            self.provenance = {"python": platform.python_version(),
                               "platform": platform.platform(), "canary": CANARY_GUID}
        else:
            self.finished_at = now

    def as_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, default=str)


def write_jsonl(path: str | Path, logs: Iterable[EpisodeLog], append: bool = False) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a" if append else "w", encoding="utf-8") as fh:
        for log in logs:
            fh.write(log.to_json() + "\n")
    return path


def read_jsonl(path: str | Path) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)
