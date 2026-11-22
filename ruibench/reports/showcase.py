"""The showcase: every task, what the agent is given, and the reference
succeeding at both levels.

    ruibench showcase                  # record what is missing (Isaac), then the page
    ruibench showcase --page-only      # just the page, from what is recorded

Four suites, six tasks each, in paper order.  A task opens with what the
agent under test is given -- the task prompt, the hint (a film shown first,
instructions that change during the episode; most tasks have none) and the
first frame from every camera it sees -- and then its scripted reference
succeeding at the easy level, and at the hard one (v13, 9/25: the hard levels
are part of what others will use; until then, while the tasks changed week by
week, the easy level alone).  No failure clips yet.  A task not built yet says
so.  Everything lives in one directory:

    runs/showcase/CL3-success.jsonl  .frames/  .mp4     the easy level
    runs/showcase/CL3-hard.jsonl     .frames/  .mp4     the hard level
    runs/showcase/index.html
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from .viewer import WEB_FONTS, _new_instruction, asset

DEFAULT_DIR = "runs/showcase"
LEVEL, SEED = "easy", 0
#: seeds tried, in order, for a clip whose agent can be lucky on seed 0
SEEDS = tuple(range(10))
#: the clips: the reference succeeding at the easy level ("success") and at
#: the hard one
ROLES = ("success", "hard")
#: recording for video: a frame every 0.1 simulated seconds, each camera at
#: its own size (640 x 480 for the station's head camera; the first frame is
#: shown at that size too)
RECORD_EVERY, RECORD_SIZE = 0.1, 640
#: how many frames of a film the hint shows
FILM_FRAMES = 4
#: cameras by what they are called on the page
CAMERA_NAMES = {"agentview": "fixed camera", "wrist": "wrist camera",
                "wrist_left": "left wrist", "wrist_right": "right wrist"}


def level_of(role: str) -> str:
    return "hard" if role == "hard" else LEVEL


def agent_for(task, role: str) -> str:
    return "reference" if role in ("success", "hard") else task.failure_example


def seed_for(task, role: str) -> int:
    """Seed 0 -- unless the clip's agent does not show its role there.  DS2's
    blind guess lifts the first cup, and at seed 0 the block is under it.  The
    first seed on which the mock engine shows the role is taken instead: which
    cup hides the block, and so what a guess finds, comes from the seed, not
    from the physics."""
    from ..core.config import RunConfig
    from ..policies.agents import build_agent
    from ..runner import RunPlan, run_plan

    cfg = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
    spec = agent_for(task, role)
    for seed in SEEDS:
        [result] = run_plan(RunPlan([task.id], levels=(level_of(role),), seeds=(seed,)),
                            agent_factory=lambda inst: build_agent(spec, inst), config=cfg)
        if shows_its_role(json.loads(result.log.to_json()), role):
            return seed
    return SEED


def log_path(out_dir: str | Path, task_id: str, role: str) -> Path:
    return Path(out_dir) / f"{task_id}-{role}.jsonl"


def video_path(out_dir: str | Path, task_id: str, role: str) -> Path:
    return Path(out_dir) / f"{task_id}-{role}.mp4"


def level_text(knob, level: str) -> str:
    """What a level sets the knob to, in words: "space width 5.6 cm", "7 layers",
    "latching"."""
    value = knob.levels.get(level)
    if value is None or isinstance(value, str):
        return str(value or "")
    field, unit = knob.field.replace("_", " "), knob.unit or ""
    if unit == "m":
        size = f"{value * 100:g} cm" if value >= 0.01 else f"{value * 1000:g} mm"
        return f"{field} {size}"
    if unit == "ratio":
        return f"{field} ×{value:g}"
    if unit == "T_ref":
        return f"{field} {value:g} × T_ref"
    noun = unit[:-1] if value == 1 and unit.endswith("s") else unit
    return f"{value:g} {noun}".strip()


def read_log(path: Path) -> Optional[dict]:
    """The last episode in a file, if any."""
    if not path.exists():
        return None
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return json.loads(lines[-1]) if lines else None


def shows_its_role(log: Optional[Mapping], role: str) -> bool:
    """A success clip must succeed; a failure clip must fail for the task's
    reasons -- not because the harness broke."""
    if not log:
        return False
    score = log.get("score") or {}
    if role in ("success", "hard"):
        return bool(score.get("success"))
    return not score.get("success") and score.get("failure") not in ("harness_error", None)


def _later(log: Mapping) -> List[str]:
    """The instructions the episode changes to, in order."""
    out = []
    for event in log.get("events") or []:
        new = _new_instruction(event.get("detail", ""))
        if new:
            out.append(new)
    return out


def first_frames(log: Mapping) -> dict:
    """{camera: frame path, relative to the record directory}: the first
    observation the agent was given, every camera of it."""
    out: dict = {}
    for rec in sorted((r for r in log.get("frames") or () if r.get("kind") == "observation"),
                      key=lambda r: r["t"]):
        out.setdefault(rec["camera"], rec["path"])
    return out


def film_frames(log: Mapping, index: int, camera: str = "agentview",
                count: int = FILM_FRAMES) -> List[str]:
    """``count`` frames of one film, evenly spaced, first and last included."""
    rows = sorted((r for r in log.get("frames") or ()
                   if r.get("demonstration") == index and r.get("camera") == camera),
                  key=lambda r: r["t"])
    if len(rows) <= count:
        return [r["path"] for r in rows]
    if count == 1:
        return [rows[0]["path"]]
    step = (len(rows) - 1) / (count - 1)
    return [rows[round(k * step)]["path"] for k in range(count)]


def _frame_src(log: Mapping, rel: str, out_dir: Path) -> str:
    root = Path((log.get("config") or {}).get("record_dir") or ".")
    return Path(os.path.relpath((root / rel).resolve(), out_dir.resolve())).as_posix()


def _src(file: Path, out_dir: Path, videos: Optional[Mapping[str, str]]) -> str:
    if videos is not None:
        return videos.get(file.name, "")
    return Path(os.path.relpath(file.resolve(), out_dir.resolve())).as_posix()


def _credits(tasks: Sequence) -> str:
    """The footer: who made the scanned and modelled things in the pictures,
    read from the asset catalogue (CC BY asks to be credited wherever the
    things are shown)."""
    from ..engines.isaac.assets import credits
    from ..engines.isaac.environment import ambientcg_ids, poly_haven_ids
    from ..tasks.instance import assets_shown

    esc = html.escape

    def link(text: str, url: str) -> str:
        return (f'<a href="{esc(url)}" target="_blank" rel="noopener">{esc(text)}</a>'
                if url else esc(text))

    lines = []
    for c in credits(assets_shown(t.id for t in tasks)):
        if len(c.items) == 1 and c.items[0][1]:
            what = link(c.source, c.items[0][1])       # one model, its own page
        else:
            what = f"{esc(c.source)}: " + ", ".join(link(n, u) for n, u in c.items)
        lines.append(f'<li>{what}<span class="lic">{esc(c.licence)}</span></li>')
    room = ", ".join(link(i, f"https://polyhaven.com/a/{i}") for i in poly_haven_ids())
    lines.append(f'<li>Poly Haven, the room light and the materials: {room}'
                 f'<span class="lic">CC0</span></li>')
    grounds = ", ".join(link(i, f"https://ambientcg.com/view?id={i}") for i in ambientcg_ids())
    lines.append(f'<li>ambientCG, materials: {grounds}<span class="lic">CC0</span></li>')
    return ('<footer class="credits"><h2>Credits</h2>'
            '<p>Objects scanned or modelled by others, used under the licence named.</p>'
            f'<ul>{"".join(lines)}</ul></footer>')


def _given(task, log: Optional[Mapping], out_dir: Path) -> str:
    """The panel "what the agent gets": the task prompt, the hint and the
    first frame, read off the recorded episode."""
    esc = html.escape
    if not log:
        return (f'<section class="given"><h4>What the agent gets</h4>'
                f'<p class="pending">{esc(task.summary)}</p></section>')
    rows = [f'<div class="row"><dt>Task prompt</dt>'
            f'<dd class="prompt">{esc(log.get("instruction") or "")}</dd></div>']
    hints = []
    told = log.get("hint") or task.hint
    if told:
        hints.append(f'<p class="told">{esc(told)}</p>')
    for i, demo in enumerate(log.get("demonstrations") or ()):
        picture = bool(demo.get("picture"))
        strip = "".join(f'<img src="{esc(_frame_src(log, rel, out_dir))}" alt="" loading="lazy">'
                        for rel in film_frames(log, i, count=1 if picture else FILM_FRAMES))
        seconds = float(demo.get("seconds") or 0.0)
        shown = "A picture is shown first" if picture else f"A film is shown first ({seconds:.0f} s)"
        hints.append(f'<p>{shown}, captioned <q>{esc(demo.get("caption") or "")}</q>:</p>'
                     + (f'<div class="film{" picture" if picture else ""}">{strip}</div>'
                        if strip else ""))
    later = _later(log)
    if later:
        shown = "".join(f"<li>{esc(t)}</li>" for t in later[:6])
        more = (f'<li class="more">and {len(later) - 6} more</li>' if len(later) > 6 else "")
        hints.append(f'<p>The instruction changes as the episode goes on:</p>'
                     f'<ol class="later">{shown}{more}</ol>')
    rows.append('<div class="row"><dt>Hint</dt><dd>'
                + ("".join(hints) or '<span class="none-given">none</span>') + '</dd></div>')
    frames = first_frames(log)
    main = frames.pop("agentview", None)
    tiles = []
    if main:
        tiles.append(f'<figure class="cam main"><img src="{esc(_frame_src(log, main, out_dir))}" '
                     f'alt="{esc(task.id)}: the fixed camera\'s first frame" loading="lazy">'
                     f'<figcaption>{CAMERA_NAMES["agentview"]}</figcaption></figure>')
    side = [(cam, frames[cam]) for cam in ("wrist_left", "wrist_right", "wrist") if cam in frames]
    if side:
        tiles.append('<div class="wrists">' + "".join(
            f'<figure class="cam"><img src="{esc(_frame_src(log, rel, out_dir))}" '
            f'alt="{esc(task.id)}: the {CAMERA_NAMES.get(cam, cam)}\'s first frame" loading="lazy">'
            f'<figcaption>{CAMERA_NAMES.get(cam, cam)}</figcaption></figure>'
            for cam, rel in side) + '</div>')
    rows.append('<div class="row"><dt>First frame</dt><dd><div class="frames">'
                + "".join(tiles) + '</div></dd></div>')
    return (f'<section class="given"><h4>What the agent gets</h4>'
            f'<dl>{"".join(rows)}</dl></section>')


def render(data_dir: str | Path, out_dir: str | Path | None = None,
           videos: Optional[Mapping[str, str]] = None, web_fonts: bool = False,
           fragment: bool = False, title: str = "RUIBench tasks") -> str:
    """The page.  ``videos`` maps video file names to URLs (for a page published
    somewhere the files are not next to it)."""
    from ..tasks.loader import load_all, main_board
    from ..tasks.spec import SUITE_ORDER, SUITES, SplitRole, Status

    data_dir = Path(data_dir)
    out_dir = Path(out_dir or data_dir)
    esc = html.escape
    sections = []
    built = 0
    built_tasks = []
    for suite in SUITE_ORDER:
        _, suite_name, where = SUITES[suite]
        cards = []
        for task in (t for t in load_all() if t.suite == suite
                     and t.split_role is not SplitRole.HOLDOUT):
            arms = "two arms" if task.arms == 2 else "one arm"
            head = (f'<header class="task-head"><h3><span class="id">{esc(task.id)}</span> '
                    f'{esc(task.name)}</h3><span class="arms">{arms}</span></header>')
            if task.status is not Status.IMPLEMENTED:
                cards.append(f'<article class="task todo">{head}'
                             f'<p class="summary">{esc(task.summary)}</p>'
                             f'<p class="state">not built yet</p></article>')
                continue
            built += 1
            built_tasks.append(task)
            log = read_log(log_path(data_dir, task.id, "success"))
            ok = shows_its_role(log, "success")
            video = video_path(data_dir, task.id, "success")
            what = level_text(task.knob, LEVEL)
            label = f"{LEVEL}{f' · {what}' if what else ''} · the scripted reference succeeds"
            seed = (log or {}).get("seed", SEED)
            if seed != SEED:
                label += f" · seed {seed}"
            if ok and video.exists():
                clip = (f'<figure class="clip success"><video src="{esc(_src(video, out_dir, videos))}" '
                        f'controls muted loop playsinline preload="metadata"></video>'
                        f'<figcaption>✓ {esc(label)}</figcaption></figure>')
            else:
                clip = (f'<figure class="clip success missing"><div class="none">'
                        f'not recorded yet</div><figcaption>{esc(label)}</figcaption></figure>')
            clip += _hard_clip(task, log, data_dir, out_dir, videos)
            cards.append(f'<article class="task">{head}{_given(task, log if ok else None, out_dir)}'
                         f'{clip}</article>')
        sections.append(
            f'<section class="suite" id="{suite}"><div class="suite-head">'
            f'<h2>{esc(suite_name)}</h2><p>{esc(where)}</p>'
            f'</div><div class="tasks">{"".join(cards)}</div></section>')
    body = (f'<div class="showcase"><header class="top"><b>RUIBench</b>'
            f'<span>{built} of {len(main_board())} tasks built · for each: what the agent under '
            f'test is given, then the scripted reference succeeding at the easy level and at '
            f'the hard one</span>'
            f'</header><main>{"".join(sections)}</main>{_credits(built_tasks)}</div>')
    head = (f"<title>{esc(title)}</title>" + (WEB_FONTS if web_fonts else "")
            + f"<style>{asset('tokens.css')}{asset('showcase.css')}</style>")
    if fragment:
        return head + body
    from ..core.canary import CANARY_GUID, NOTICE

    return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<!-- {NOTICE} --><meta name="ruibench-canary" content="{CANARY_GUID}">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            f"{head}</head><body>{body}</body></html>")


def _hard_clip(task, easy_log: Optional[Mapping], data_dir: Path, out_dir: Path,
               videos: Optional[Mapping[str, str]]) -> str:
    """The reference at the hard level, with the instruction when it differs."""
    esc = html.escape
    log = read_log(log_path(data_dir, task.id, "hard"))
    ok = shows_its_role(log, "hard")
    video = video_path(data_dir, task.id, "hard")
    what = level_text(task.knob, "hard")
    label = f"hard{f' · {what}' if what else ''} · the scripted reference succeeds"
    seed = (log or {}).get("seed", SEED)
    if seed != SEED:
        label += f" · seed {seed}"
    said = (log or {}).get("instruction")
    note = (f'<p class="hard-said">Instruction at the hard level: '
            f'<q>{esc(said)}</q></p>'
            if ok and said and said != (easy_log or {}).get("instruction") else "")
    if ok and video.exists():
        return (f'<figure class="clip success hard"><video src="{esc(_src(video, out_dir, videos))}" '
                f'controls muted loop playsinline preload="metadata"></video>'
                f'<figcaption>✓ {esc(label)}</figcaption>{note}</figure>')
    return (f'<figure class="clip success hard missing"><div class="none">not recorded yet</div>'
            f'<figcaption>{esc(label)}</figcaption></figure>')


def write(data_dir: str | Path = DEFAULT_DIR, out: str | Path | None = None, **kwargs) -> Path:
    out = Path(out or Path(data_dir) / "index.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(data_dir, out.parent, **kwargs), encoding="utf-8")
    return out


def missing(tasks: Sequence, data_dir: str | Path, roles: Sequence[str] = ROLES) -> List[tuple]:
    """(task, role) pairs whose clip is not recorded, or does not show its role."""
    return [(t, role) for t in tasks for role in roles
            if not shows_its_role(read_log(log_path(data_dir, t.id, role)), role)]


def record(task, role: str, data_dir: str | Path, engine: str = "isaac",
           progress=None, seed: Optional[int] = None) -> Optional[dict]:
    """Run one clip's episode, recorded for video, and encode the video.
    ``seed``: the mission to film, instead of the first one on which the mock
    engine shows the role (the two engines can disagree about a lucky guess)."""
    from ..core.config import RunConfig
    from ..runner import RunPlan
    from ..runner.isolated import IsolatedRun, run_isolated
    from .video import write_video

    path = log_path(data_dir, task.id, role)
    if path.exists() and not shows_its_role(read_log(path), role):
        path.unlink()                      # a stale clip is redone, not resumed
    # played out: the clip shows the reference putting its tools down and
    # folding its arms away after the task is done
    cfg = RunConfig(run_id="showcase", engine=engine,
                    record_dir=str(path.with_suffix(".frames")),
                    record_every=RECORD_EVERY, record_size=RECORD_SIZE, play_out=True)
    plan = RunPlan(task_ids=[task.id], levels=(level_of(role),),
                   seeds=(seed_for(task, role) if seed is None else seed,))
    job = IsolatedRun(cells=plan.cells(), config=cfg, agent=agent_for(task, role))
    run_isolated(job, path, progress=progress)
    log = read_log(path)
    if log and log.get("frames"):
        write_video(log, video_path(data_dir, task.id, role))
    return log
