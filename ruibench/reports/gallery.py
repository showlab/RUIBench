"""The render check: how every mission of every task starts, as Isaac draws it.

    ruibench gallery                      # -> runs/gallery/ (images + index.html)

One child process per task (every level and seed of a task shares one scene
structure, so one simulator start covers them all), each printing one
``RESULT`` line like the doctor's checks.  Besides the pictures it counts, in
the fixed camera, the pixels of the gripper and of every mark on the table:
both have to be in view from the first frame; and a picture that is nearly
all one colour -- a camera pressed against something -- is flagged.  ``index.html`` is a contact
sheet of the pictures.

    python -m ruibench.reports.gallery TASK OUT_DIR LEVELS SEEDS SIZE   # the child
"""

from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path
from typing import List, Sequence

MANIFEST = "gallery.json"


def image_name(task: str, level: str, seed: int, camera: str) -> str:
    return f"{task}-{level}-s{seed}-{camera}.jpg"


def seed_list(text: str) -> List[int]:
    out: List[int] = []
    for part in str(text).split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


#: a picture this much of one colour is a camera pressed against something
BLANK_SHARE = 0.95


def blank(image, tolerance: int = 12) -> bool:
    """Is (nearly) the whole picture one colour?"""
    import numpy as np

    pixels = np.asarray(image.convert("RGB"), dtype=np.int16).reshape(-1, 3)
    middle = np.median(pixels, axis=0)
    near = np.all(np.abs(pixels - middle) <= tolerance, axis=1)
    return float(near.mean()) >= BLANK_SHARE


def render_task(task: str, out_dir: str, levels: str = "easy,hard", seeds: str = "0",
                size: str = "384") -> dict:
    """Child process: render every mission of one task from both cameras, and
    count the gripper's and each mark's pixels in the fixed camera."""
    from ..engines import make_engine
    from ..engines.isaac.robots import arm_key
    from ..engines.isaac.scene_builder import gripper_label, marker_label
    from ..runner.recorder import to_image
    from ..tasks.instance import build

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    engine = make_engine("isaac")
    saved, gripper, marks, blanks = [], {}, {}, []
    for level in levels.split(","):
        for seed in seed_list(seeds):
            inst = build(task, level, seed)
            engine.load(inst.scene, seed=seed)
            engine.advance(1.0)                   # let everything settle
            for camera in (c.name for c in inst.scene.cameras):
                image = to_image(engine.render(camera).image, int(size))
                if image is not None:
                    file = out / image_name(task, level, seed, camera)
                    image.save(file, quality=88)
                    saved.append(file.name)
                    if blank(image):
                        blanks.append(file.name)
            counts = engine.pixel_counts("agentview")
            where = f"{level}/{seed}"
            # every commanded hand, by its own label (a station labels each arm)
            for robot in inst.scene.robots:
                for arm in robot.arms:
                    key = f"{where} {arm}" if len(robot.arms) > 1 else where
                    gripper[key] = counts.get(gripper_label(arm_key(robot, arm)), 0)
            for m in inst.scene.markers:
                marks[f"{where} {m.name}"] = counts.get(marker_label(m.name), 0)
    return {"pass": bool(saved) and not blanks and min(gripper.values(), default=0) > 0
            and min(marks.values(), default=1) > 0,
            "saved": len(saved), "blank": blanks, "gripper_pixels_in_agentview": gripper,
            "mark_pixels_in_agentview": marks}


def write_manifest(out_dir: str | Path, tasks: Sequence[str], levels: Sequence[str],
                   seeds: Sequence[int], size: int) -> Path:
    path = Path(out_dir) / MANIFEST
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"tasks": list(tasks), "levels": list(levels),
                                "seeds": list(seeds), "size": size}, indent=1))
    return path


def contact_sheet(out_dir: str | Path) -> Path:
    """``index.html``: every rendered start, a row per task and level."""
    from .viewer import asset

    out_dir = Path(out_dir)
    meta = json.loads((out_dir / MANIFEST).read_text())
    rows = []
    for task in meta["tasks"]:
        for level in meta["levels"]:
            cells = "".join(
                f'<figure><img src="{html.escape(image_name(task, level, seed, "agentview"))}" '
                f'alt="{task} {level} seed {seed}" loading="lazy"><figcaption>seed {seed}</figcaption></figure>'
                for seed in meta["seeds"]
                if (out_dir / image_name(task, level, seed, "agentview")).exists())
            rows.append(f'<h2>{html.escape(task)} · {html.escape(level)}</h2><div class="row">{cells}</div>')
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>RUIBench · mission starts</title>'
            f"<style>{asset('tokens.css')}"
            "body{margin:0;padding:16px 24px;background:var(--ground);color:var(--ink);font-family:var(--font-ui)}"
            "h2{font-family:var(--font-label);font-size:var(--step-1);margin:18px 0 6px}"
            ".row{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:6px}"
            "figure{margin:0}img{width:100%;border-radius:4px;display:block}"
            "figcaption{font-size:var(--step--1);color:var(--muted)}</style></head>"
            f"<body>{''.join(rows)}</body></html>")
    path = out_dir / "index.html"
    path.write_text(page, encoding="utf-8")
    return path


if __name__ == "__main__":
    try:
        from ..engines.isaac import app

        app.launch(headless=True)
        result = render_task(*sys.argv[1:])
        print("RESULT " + json.dumps(result), flush=True)
    except Exception:
        import traceback
        print("RESULT " + json.dumps({"pass": False, "error": traceback.format_exc()[-800:]}),
              flush=True)
    finally:
        sys.stdout.flush()
        os._exit(0)                            # never tear an Isaac stage down in-process
