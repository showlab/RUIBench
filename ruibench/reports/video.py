"""Episode videos: what the cameras saw, as a file anyone can play.

    ruibench video runs/x.jsonl            # -> runs/x.videos/<episode>.mp4

Made from the frames a recorded run saved (``ruibench run --record``; use
``--record-every 0.1`` for smooth motion).  Each video frame shows the fixed
camera large and the wrist cameras beside it, each at its own aspect, under a
caption -- task, level, agent, outcome, simulated time -- and over a
subtitle: what the agent is being told, a new instruction marked as it comes,
and what the world says back.  The last frame holds with the verdict.  H.264
in MP4, which every browser plays; encoded by the system ``ffmpeg``.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

CAMERA_ORDER = ("agentview", "wrist", "wrist_left", "wrist_right", "wrist2")
#: a beat between one demonstration and the next, and before the episode
DEMO_GAP = 0.5
CAPTION_H = 52
HOLD_SECONDS = 1.5
FONT_DIRS = ("/usr/share/fonts/truetype/dejavu", "/usr/share/fonts/dejavu",
             "/Library/Fonts", "C:/Windows/Fonts")

#: colours of the caption bar -- the viewer's tokens, dark theme
INK, MUTED, GROUND = (231, 235, 240), (142, 152, 165), (13, 16, 21)
OK, FAIL, ACCENT = (69, 194, 139), (255, 123, 97), (255, 92, 157)
#: the subtitle under the pictures (v8, 9/24 -- the advisor: in a task where
#: new input comes mid-way, DI4's calls, a video that does not show it cannot
#: be followed): the instruction the agent has now, marked NEW for a while
#: when it changes, and a message from the world for a while after it is said
NEW_SECONDS, SAID_SECONDS = 3.0, 2.5
#: the instruction's lines: as many as the longest one takes, up to this many
#: (CL6's runs to three), and a line under them for what is said
SUBTITLE_LINES, LINE_H, SAID_H = 4, 18, 22


def video_name(episode_id: str) -> str:
    """A file name for an episode's video (ids carry agent names like
    ``ref+misperceive(0.03 m)``)."""
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", episode_id).strip("_") + ".mp4"


def have_ffmpeg() -> bool:
    return shutil.which("ffmpeg") is not None


def _font(size: int, bold: bool = False):
    from PIL import ImageFont

    names = ("DejaVuSansCondensed-Bold.ttf", "DejaVuSans-Bold.ttf") if bold else \
        ("DejaVuSansCondensed.ttf", "DejaVuSans.ttf")
    for folder in FONT_DIRS:
        for name in names:
            path = Path(folder) / name
            if path.exists():
                return ImageFont.truetype(str(path), size)
    return ImageFont.load_default()


def _frames_by_camera(log: Mapping) -> dict:
    """{camera: [(video time, file)]}.  Demonstrations come first, one after
    another, and the episode follows: its frames are shifted by their length."""
    root = Path((log.get("config") or {}).get("record_dir") or ".")
    starts, t0 = {}, 0.0
    for i, demo in enumerate(log.get("demonstrations") or ()):
        starts[i] = t0
        t0 += float(demo.get("seconds", 0.0)) + DEMO_GAP
    out: dict = {}
    for rec in log.get("frames") or []:
        file = root / rec["path"]
        if not file.exists():
            continue
        if "demonstration" in rec:
            t = starts.get(rec["demonstration"], 0.0) + rec["t"]
        else:
            t = t0 + rec["t"]
        out.setdefault(rec["camera"], []).append((t, file))
    for rows in out.values():
        rows.sort(key=lambda r: r[0])
    return out


def _latest(rows: Sequence[Tuple[float, Path]], t: float) -> Optional[Path]:
    best = None
    for ft, file in rows:
        if ft <= t + 1e-9:
            best = file
        else:
            break
    return best or (rows[0][1] if rows else None)


def _caption(log: Mapping, t: float, width: int, final: bool):
    from PIL import Image, ImageDraw

    score = log.get("score") or {}
    ok = bool(score.get("success"))
    bar = Image.new("RGB", (width, CAPTION_H), GROUND)
    draw = ImageDraw.Draw(bar)
    title = f"{log['task_id']} · {log['level']} · seed {log['seed']}"
    lead = sum(float(d.get("seconds", 0.0)) + DEMO_GAP for d in log.get("demonstrations") or ())
    watching = t < lead
    agent = "the demonstration shown first" if watching else str(log.get("agent", ""))
    draw.text((12, 7), title, font=_font(17, bold=True), fill=INK)
    draw.text((12, 29), agent, font=_font(13), fill=MUTED)
    clock = "demo" if watching else f"t = {t - lead:5.1f} s"
    clock_font = _font(16, bold=True)
    w = draw.textlength(clock, font=clock_font)
    draw.text((width - w - 12, 7), clock, font=clock_font, fill=ACCENT)
    verdict = "success" if ok else str(score.get("failure", "failed")).replace("_", " ")
    if final:
        verdict = ("✓ " if ok else "✗ ") + verdict
    vfont = _font(13, bold=final)
    vw = draw.textlength(verdict, font=vfont)
    draw.text((width - vw - 12, 29), verdict, font=vfont, fill=OK if ok else FAIL)
    return bar


def _told(log: Mapping):
    """(instruction changes [(episode time, text)], messages [(episode time,
    text)]) -- the instruction the episode began with at time 0."""
    from .viewer import _new_instruction

    changes = [(0.0, str(log.get("instruction") or ""))]
    for e in log.get("events") or ():
        text = _new_instruction(e.get("detail") or "")
        if text is not None:
            changes.append((float(e.get("sim_time", 0.0)), text))
    said = [(float(m.get("sim_time", 0.0)), str(m["text"])) for m in log.get("feedback") or ()
            if m.get("text")]
    return sorted(changes), sorted(said)


def _wrap(draw, text: str, font, width: float, lines: int) -> List[str]:
    out, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=font) <= width or not line:
            line = trial
        else:
            out.append(line)
            line = word
    out.append(line)
    if len(out) > lines:
        out = out[:lines]
        out[-1] = out[-1].rstrip(".,;") + " …"
    return out


def _texts_left(draw) -> float:
    return 12 + draw.textlength("FILM", font=_font(12, bold=True)) + 10


def subtitle_lines(log: Mapping, width: int, told) -> int:
    """The lines the subtitle gives the instruction in a ``width`` wide
    video: as many as the longest instruction or film caption takes."""
    from PIL import Image, ImageDraw

    draw = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    texts = [text for _, text in told[0]]
    texts += [str(d.get("caption", "")) for d in log.get("demonstrations") or ()]
    room = width - _texts_left(draw) - 12
    return max(len(_wrap(draw, text, _font(14, bold=True), room, SUBTITLE_LINES))
               for text in texts)


def subtitle_height(lines: int) -> int:
    return 6 + LINE_H * lines + SAID_H


def _subtitle(log: Mapping, t: float, width: int, told, lines: int = 2) -> "object":
    """The bar under the pictures at video time ``t``: the instruction in
    ``lines`` lines, what is said under it."""
    from PIL import Image, ImageDraw

    height = subtitle_height(lines)
    bar = Image.new("RGB", (width, height), GROUND)
    draw = ImageDraw.Draw(bar)
    demos = log.get("demonstrations") or ()
    at = 0.0
    for d in demos:                          # the demonstrations come first
        if t < at + float(d.get("seconds", 0.0)) + DEMO_GAP:
            tag, text, colour = "FILM", str(d.get("caption", "")), MUTED
            break
        at += float(d.get("seconds", 0.0)) + DEMO_GAP
    else:
        te = t - at
        changes, said = told
        when, text = max((c for c in changes if c[0] <= te + 1e-9), default=changes[0])
        new = when > 0 and te - when < NEW_SECONDS
        tag, colour = ("NEW", ACCENT) if new else ("TASK", MUTED)
        heard = [m for m in said if 0 <= te - m[0] < SAID_SECONDS]
        if heard:
            line_font = _font(13, bold=True)
            draw.text((12, height - 20), "“" + heard[-1][1] + "”", font=line_font, fill=OK)
    tag_font = _font(12, bold=True)
    draw.text((12, 6), tag, font=tag_font, fill=colour)
    left = _texts_left(draw)
    body = _font(14, bold=tag == "NEW")
    for k, line in enumerate(_wrap(draw, text, body, width - left - 12, lines)):
        draw.text((left, 4 + LINE_H * k), line, font=body, fill=INK)
    return bar


#: what each camera is called on its tile
CAMERA_LABELS = {"agentview": "fixed camera", "wrist": "wrist", "wrist_left": "left wrist",
                 "wrist_right": "right wrist"}
GAP = 4


def _layout(frames: Mapping[str, Sequence[Tuple[float, Path]]], order: Sequence[str],
            size: int) -> Tuple[Tuple[int, int], dict]:
    """(canvas size, {camera: (x, y, w, h)}): the first camera -- its long
    side ``size``, its own aspect (the station's head camera is 4:3) -- and
    the rest stacked in a column beside it, sharing its height."""
    from PIL import Image

    def aspect(cam: str) -> float:
        with Image.open(frames[cam][0][1]) as img:
            return img.width / img.height

    main, rest = order[0], list(order[1:])
    a = aspect(main)
    mw, mh = (size, round(size / a)) if a >= 1 else (round(size * a), size)
    boxes = {main: (0, CAPTION_H, mw, mh)}
    if not rest:
        return (mw, CAPTION_H + mh), boxes
    # one wrist camera stands as tall as the fixed one; two share its height
    cell_h = (mh - GAP * (len(rest) - 1)) // len(rest)
    col_w = max(round(cell_h * aspect(c)) for c in rest)
    top = CAPTION_H + (mh - (cell_h * len(rest) + GAP * (len(rest) - 1))) // 2
    for k, cam in enumerate(rest):
        w = round(cell_h * aspect(cam))
        boxes[cam] = (mw + GAP + (col_w - w) // 2, top + k * (cell_h + GAP), w, cell_h)
    return (mw + GAP + col_w, CAPTION_H + mh), boxes


def compose(log: Mapping, size: int = 320) -> List[Tuple[float, "object"]]:
    """(time, RGB image) for every recorded instant: the fixed camera large,
    the wrist cameras beside it, each tile named."""
    from PIL import Image, ImageDraw

    cams = _frames_by_camera(log)
    order = [c for c in CAMERA_ORDER if c in cams] + sorted(set(cams) - set(CAMERA_ORDER))
    if not order:
        return []
    (width, height), boxes = _layout(cams, order, size)
    label_font = _font(12, bold=True)
    times = sorted({t for rows in cams.values() for t, _ in rows})
    out = []
    for t in times:
        canvas = Image.new("RGB", (width, height), GROUND)
        draw = ImageDraw.Draw(canvas)
        for cam in order:
            file = _latest(cams[cam], t)
            if file is None:
                continue
            x, y, w, h = boxes[cam]
            with Image.open(file) as img:
                canvas.paste(img.convert("RGB").resize((w, h)), (x, y))
            label = CAMERA_LABELS.get(cam, cam)
            lw = draw.textlength(label, font=label_font)
            draw.rectangle((x, y + h - 20, x + lw + 12, y + h), fill=GROUND)
            draw.text((x + 6, y + h - 18), label, font=label_font, fill=MUTED)
        out.append((t, canvas))
    return out


#: a video plays for at most about this long: a long episode plays faster
MAX_VIDEO_SECONDS = 45.0


def write_video(log: Mapping, out: str | Path, fps: int = 20, speed: float = 2.0,
                size: int = 320, max_seconds: float = MAX_VIDEO_SECONDS) -> Optional[Path]:
    """Encode one episode at ``speed`` times real time -- faster, if that
    would make it longer than ``max_seconds``.  None when it has no frames."""
    import numpy as np
    from PIL import Image

    frames = compose(log, size)
    if not frames:
        return None
    if not have_ffmpeg():
        raise RuntimeError("ffmpeg is not on PATH; install it to write videos")
    from ..core.canary import NOTICE
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    width, height = frames[0][1].size
    told = _told(log)
    lines = subtitle_lines(log, width, told)
    under = subtitle_height(lines)
    height += under
    width, height = width - width % 2, height - height % 2
    end = frames[-1][0]
    if max_seconds and end / speed > max_seconds:
        speed = end / max_seconds
    count = int(end / speed * fps) + 1
    hold = int(HOLD_SECONDS * fps)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p",
           "-metadata", f"comment={NOTICE}",
           "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        k = 0
        for i in range(count + hold):
            t = min(i * speed / fps, end)
            while k + 1 < len(frames) and frames[k + 1][0] <= t + 1e-9:
                k += 1
            frame = frames[k][1]
            image = Image.new("RGB", (frame.width, frame.height + under), GROUND)
            image.paste(frame, (0, 0))
            image.paste(_caption(log, t, image.width, final=i >= count), (0, 0))
            image.paste(_subtitle(log, t, image.width, told, lines), (0, frame.height))
            proc.stdin.write(np.asarray(image.crop((0, 0, width, height)), dtype="uint8").tobytes())
    finally:
        proc.stdin.close()
        code = proc.wait()
    if code != 0:
        raise RuntimeError(f"ffmpeg exited with {code} for {out}")
    return out


def write_videos(logs: Iterable[Mapping], out_dir: str | Path, **kwargs) -> List[Path]:
    """One video per recorded episode, named by :func:`video_name`."""
    from .viewer import episode_key

    written = []
    for log in logs:
        path = write_video(log, Path(out_dir) / video_name(episode_key(log)), **kwargs)
        if path is not None:
            written.append(path)
    return written
