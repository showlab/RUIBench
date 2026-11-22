"""The episode viewer: what the robot saw and did, second by second.

    ruibench view runs/x.jsonl              # -> runs/x.viewer.html
    ruibench view runs/x.jsonl --embed      # frames inlined: one portable file

One self-contained page, like the report, fetching nothing from the network.
For every episode it replays a plan view of the table drawn from the logged
trajectory -- so it works for any engine, the mock included -- beside the
camera frames the model was shown when the run was recorded on Isaac, on one
clock with a trace of model turns, arm actions, milestones and events.  It is
the tool for finding out why an episode went the way it did; the page for
showing what each task is, is ``ruibench showcase``.
"""

from __future__ import annotations

import ast
import base64
import html as _html
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

from ..engines.base import color_hex

STATIC = Path(__file__).with_name("static")

#: Only for pages published to the web: the local file keeps to system fonts.
WEB_FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
             '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
             '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600;700'
             '&family=IBM+Plex+Sans:wght@400;500;600&display=swap">')

_SET_INSTRUCTION = re.compile(r"^set_instruction\((.*)\)$", re.S)


def asset(name: str) -> str:
    return (STATIC / name).read_text(encoding="utf-8")


# ------------------------------------------------------------------ data
def scene_for_view(summary: Mapping) -> dict:
    """``SceneSpec.summary()`` in the shape the page draws from."""
    robots = [{"name": r["name"], "base": r["base"]} for r in summary.get("robots", [])]
    arm_ids = [f"{r['name']}/{a}" for r in summary.get("robots", []) for a in r.get("arms", [])]
    bodies = [{"name": b["name"], "shape": b["shape"], "size": b["size"], "pos": b["pos"],
               "yaw": b.get("yaw", 0.0), "color": color_hex(b.get("color")),
               "category": b.get("category"), "fixed": bool(b.get("fixed")),
               "container": bool(b.get("container")), "wall": b.get("wall"),
               "footprint": b.get("footprint"), "attachedTo": b.get("attached_to")}
              for b in summary.get("bodies", [])]
    markers = [{**m, "color": color_hex(m.get("color"))} for m in summary.get("markers", [])]
    return {"tableHeight": summary.get("table_height", 0.8), "robots": robots,
            "armIds": arm_ids, "bodies": bodies, "markers": markers}


def _scene_of(log: Mapping) -> dict:
    if log.get("scene"):
        return log["scene"]
    try:                                    # logs written before scenes were logged
        from ..tasks.instance import build
        return build(log["task_id"], log["level"], int(log["seed"])).scene.summary()
    except Exception:
        return {}


def _commands(raw) -> object:
    try:
        parsed = json.loads(raw)
    except (TypeError, ValueError):
        return raw if raw else []
    return parsed if isinstance(parsed, list) else [parsed]


def _new_instruction(detail: str) -> Optional[str]:
    m = _SET_INSTRUCTION.match(detail or "")
    if not m:
        return None
    try:
        return str(ast.literal_eval(m.group(1)))
    except (ValueError, SyntaxError):
        return m.group(1)


def _trajectory(snapshots: Sequence[Mapping], scene: Mapping, stride: int = 1) -> dict:
    if not snapshots:
        return {"t": [0.0], "objects": {}, "arms": {}}
    if stride > 1:                                   # keep the last snapshot: the end state
        snapshots = list(snapshots[::stride]) + ([snapshots[-1]] if (len(snapshots) - 1) % stride else [])
    arm_ids = list(snapshots[0].get("arms", {}))
    initial_yaw = {b["name"]: b.get("yaw", 0.0) for b in scene.get("bodies", [])}
    objects: Dict[str, List[list]] = {n: [] for n in snapshots[0].get("objects", {})}
    arms: Dict[str, List[list]] = {a: [] for a in arm_ids}
    for snap in snapshots:
        for name, rows in objects.items():
            o = snap["objects"].get(name)
            if o is None:
                rows.append(rows[-1] if rows else [0.0, 0.0, 0.0, 0.0, 0])
                continue
            holder = o.get("held_by")
            held = arm_ids.index(holder) + 1 if holder in arm_ids else 0
            x, y, z = o["pos"]
            rows.append([round(x, 4), round(y, 4), round(z, 4),
                         round(o.get("yaw", initial_yaw.get(name, 0.0)), 3), held]
                        # a cloth: the box round it as it lies at this moment
                        + ([round(v, 4) for v in o["size"][:2]] if "size" in o else []))
        for arm, rows in arms.items():
            a = snap["arms"][arm]
            rows.append([round(v, 4) for v in a["pos"]] + [round(a.get("gripper", 0.0), 3)])
    return {"t": [s["t"] for s in snapshots], "objects": objects, "arms": arms}


def _frames(log: Mapping, out_dir: Path, embed: bool, max_frames: Optional[int]) -> dict:
    if max_frames == 0:
        return {}
    root = Path((log.get("config") or {}).get("record_dir") or ".")
    by_camera: Dict[str, List[dict]] = {}
    for rec in log.get("frames") or []:
        if "demonstration" in rec:
            continue                   # played before the episode's clock started
        by_camera.setdefault(rec["camera"], []).append(rec)
    out = {}
    for camera, recs in by_camera.items():
        recs.sort(key=lambda r: r["t"])
        if max_frames and len(recs) > max_frames:
            keep = [r for r in recs if r["kind"] == "observation"][:max_frames]
            rest = [r for r in recs if r["kind"] != "observation"]
            room = max_frames - len(keep)
            if room > 0 and rest:
                stride = len(rest) / room
                keep += [rest[int(i * stride)] for i in range(min(room, len(rest)))]
            recs = sorted(keep, key=lambda r: r["t"])
        rows = []
        for rec in recs:
            file = root / rec["path"]
            if not file.exists():
                continue
            if embed:
                src = "data:image/jpeg;base64," + base64.b64encode(file.read_bytes()).decode("ascii")
            else:
                src = Path(os.path.relpath(file.resolve(), out_dir.resolve())).as_posix()
            rows.append([rec["t"], src, rec["kind"]])
        if rows:
            out[camera] = rows
    return out


def episode_key(log: Mapping) -> str:
    """The id an episode goes by in the viewer and in video file names."""
    return "-".join(str(p) for p in (log["task_id"], log["level"], f"s{log['seed']}",
                                     log.get("interface"), log.get("condition", "main"),
                                     log.get("agent")))


def episode_data(log: Mapping, out_dir: Path, embed: bool = False,
                 max_frames: Optional[int] = None, traj_stride: int = 1) -> dict:
    """One episode in the shape the page reads."""
    from ..engines import ENGINES
    from ..tasks import SUITES, get as get_task

    task = log["task_id"]
    try:
        spec = get_task(task)
    except Exception:
        spec = None
    summary = _scene_of(log)
    scene = scene_for_view(summary)
    traj = _trajectory(log.get("trajectory") or [], summary, traj_stride)
    if not scene["armIds"]:
        scene["armIds"] = list(traj["arms"])
    engine = ((log.get("engine_setup") or {}).get("name")
              or {"IsaacEngine": "isaac", "MockEngine": "mock"}.get(log.get("engine"), log.get("engine")))
    try:
        grade = bool(ENGINES.get(engine)().capabilities.results_grade)
    except Exception:
        grade = False
    score = log.get("score") or {}
    actions = [{"arm": a["arm"], "kind": a["kind"], "status": a["status"],
                "t0": a.get("started_at") if a.get("started_at") is not None else a.get("submitted_at", 0.0),
                "t1": a.get("finished_at"), "detail": a.get("detail", "")}
               for a in log.get("handles") or [] if a.get("status") != "rejected"]
    turns = [{"i": t["turn"], "obs": t["observed_at"], "req": t["requested_at"],
              "ret": t["returned_at"], "age": t.get("observation_age", 0.0),
              "cmds": _commands(t.get("raw_response"))} for t in log.get("turns") or []]
    events = [{"name": e["name"], "t": e["sim_time"], "detail": e.get("detail", ""),
               "newInstruction": _new_instruction(e.get("detail", ""))}
              for e in log.get("events") or []]
    feedback = [{"obj": f["obj"], "signal": f["signal"], "t": f["sim_time"]}
                for f in log.get("feedback") or []]
    ends = [traj["t"][-1], (log.get("metrics") or {}).get("sim_seconds") or 0.0]
    ends += [a["t1"] for a in actions if a["t1"] is not None]
    return {
        "id": episode_key(log),
        "task": task,
        "name": spec.name if spec else task,
        "suiteName": SUITES[spec.suite][1] if spec else "",
        "knob": ({"field": spec.knob.field, "value": spec.knob.levels.get(log["level"]),
                  "description": spec.knob.description} if spec else None),
        "level": log["level"], "seed": log["seed"], "interface": log.get("interface"),
        "agent": log.get("agent"), "condition": log.get("condition", "main"),
        "engine": engine, "resultsGrade": grade,
        "instruction": log.get("instruction") or "",
        "outcome": {"success": bool(score.get("success")), "progress": score.get("progress", 0.0),
                    "failure": score.get("failure", "none"),
                    "coverageGap": score.get("coverage_gap"),
                    "milestones": dict(score.get("milestones_hit") or {}),
                    "milestoneCount": len(spec.milestones) if spec else
                    len(score.get("milestones_hit") or {})},
        "metrics": dict(log.get("metrics") or {}),
        "taskMetrics": dict(log.get("task_metrics") or {}),
        "scene": scene, "traj": traj, "turns": turns, "actions": actions,
        "events": events, "feedback": feedback,
        "frames": _frames(log, out_dir, embed, max_frames),
        "duration": round(max(ends), 3),
    }


def render(logs: Sequence[Mapping], title: str = "episode viewer", out_dir: str | Path = ".",
           embed: bool = False, max_frames: Optional[int] = None,
           videos: Optional[Mapping[str, str]] = None, web_fonts: bool = False,
           fragment: bool = False, page_title: Optional[str] = None,
           traj_stride: int = 1) -> str:
    """The page.  ``title`` labels the run in the top bar; ``page_title``
    (default "RUIBench · <title>") names the browser tab.  ``videos`` maps
    episode ids to video URLs.  ``fragment`` leaves out the document wrapper
    (for hosts that supply their own), ``web_fonts`` links the IBM Plex faces,
    ``traj_stride`` thins trajectories for a lighter page."""
    out_dir = Path(out_dir)
    episodes, seen = [], {}
    for log in logs:
        ep = episode_data(log, out_dir, embed=embed, max_frames=max_frames,
                          traj_stride=traj_stride)
        n = seen.get(ep["id"], 0)
        seen[ep["id"]] = n + 1
        if videos and ep["id"] in videos and not n:
            ep["video"] = videos[ep["id"]]      # a video is one run's: the first of that id
        if n:
            ep["id"] += f"-{n + 1}"
        episodes.append(ep)
    payload = json.dumps({"title": title, "episodes": episodes},
                         separators=(",", ":")).replace("</", "<\\/")
    head = (f"<title>{_html.escape(page_title or f'RUIBench · {title}')}</title>"
            + (WEB_FONTS if web_fonts else "")
            + f"<style>{asset('tokens.css')}{asset('viewer.css')}</style>")
    body = ('<div id="app"></div>'
            f'<script id="ruibench-data" type="application/json">{payload}</script>'
            f"<script>{asset('viewer.js')}</script>")
    if fragment:
        return head + body
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"{head}</head><body>{body}</body></html>")


def find_videos(logs: Sequence[Mapping], video_dir: str | Path, out_dir: str | Path) -> dict:
    """Episode id -> path (relative to ``out_dir``) for the videos that exist."""
    from .video import video_name

    video_dir, out_dir = Path(video_dir), Path(out_dir)
    found = {}
    for log in logs:
        file = video_dir / video_name(episode_key(log))
        if file.exists():
            found[episode_key(log)] = Path(os.path.relpath(file.resolve(), out_dir.resolve())).as_posix()
    return found


def write(logs: Sequence[Mapping], out: str | Path, **kwargs) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(logs, out_dir=out.parent, **kwargs), encoding="utf-8")
    return out
