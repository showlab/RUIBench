"""Clips of episodes: the video, the notes and what each clip shows."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Tuple

if TYPE_CHECKING:
    from ruikit.diagnostics.coverage import Cell

#: a filmed episode: a frame every 0.2 simulated seconds, 480 pixels wide; its video
#: plays at least twice real time, and faster when that would make it longer than
#: VIDEO_SECONDS
RECORD_EVERY, RECORD_SIZE, VIDEO_SECONDS = 0.2, 480, 75.0


def clip_dir(root, task: str, level: str, seed: int, profile: str) -> Path:
    return Path(root) / f"{task}-{level}-s{seed}-{profile.split('/')[0].replace('.', '-')}"


def _video(folder: Path) -> Tuple[Optional[Path], float, float, dict]:
    """The episode's video from its own log (``video.mp4`` in its folder): the
    video (None when nothing was filmed), its speed, the demonstrations' lead
    and the log."""
    from ruibench.reports.video import DEMO_GAP, write_video

    log = json.loads(next(Path(folder).glob("*.episode.jsonl")).read_text().splitlines()[0])
    cfg = log.get("config") or {}
    if not Path(cfg.get("record_dir") or ".").is_dir() and (Path(folder) / "frames").is_dir():
        # a folder copied or moved since: its frames are beside the log
        log = {**log, "config": {**cfg, "record_dir": str(Path(folder) / "frames")}}
    lead = sum(float(d.get("seconds", 0.0)) + DEMO_GAP for d in log.get("demonstrations") or ())
    end = lead + max((float(f["t"]) for f in log.get("frames") or () if "demonstration" not in f),
                     default=0.0)
    speed = max(2.0, end / VIDEO_SECONDS)
    video = None
    if log.get("frames"):
        video = write_video(log, Path(folder) / "video.mp4", speed=speed, max_seconds=None)
    return video, speed, lead, log


def film(folder: Path, keep_frames: bool = False) -> Optional[str]:
    """A batch episode's video, ``video.mp4`` in its folder; its frames are removed
    once it is written.  The video's name, or None when nothing was filmed (an
    engine without pictures)."""
    if not any(Path(folder).glob("*.episode.jsonl")):
        return None
    video = _video(Path(folder))[0]
    if video is not None and not keep_frames:
        shutil.rmtree(Path(folder) / "frames", ignore_errors=True)
    return video.name if video is not None else None


def _film(clip: Path, cell: Cell, notes=(), budget=None) -> None:
    """The clip's video and summary, from the episode's own log."""
    video, speed, lead, log = _video(clip)
    (clip / "notes.json").write_text(json.dumps(list(notes), ensure_ascii=False))
    summary = {**cell.as_dict(), "video": video.name if video else None, "speed": round(speed, 4),
               "budget": budget or {}, "replies": (log.get("metrics") or {}).get("replies"),
               "lead": round(lead, 3), "instruction": log.get("instruction", ""),
               "hint": log.get("hint", ""), "demonstrations": log.get("demonstrations") or [],
               "engine": log.get("engine"), "score": log.get("score"), "agent": log.get("agent")}
    (clip / "clip.json").write_text(json.dumps(summary, ensure_ascii=False))
