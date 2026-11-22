#!/usr/bin/env python3
"""The picture of the real CL3, which ``task_demo`` shows: the lab's house, as it last stood.

    .venv/bin/python scripts/real_cl3_card.py

Drawn as ``ruibench.tasks.impl.cl3_house`` draws its card: the house from the front, as the fixed
camera sees it, to scale; each block in its colour inside a dark outline, each edge a little off
straight, on the same cream.  It stands on the mat, with no board.  As large as the fixed
camera's picture.  Writes ``ruibench/tasks/real/cards/CL3.png``.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(__file__).resolve().parents[1] / "ruibench" / "tasks" / "real" / "cards" / "CL3.png"
#: the fixed camera's picture, and the centimetres it stands for here
SIZE = (640, 480)
CARD_CM = (24.0, 18.0)
#: as simulation's card (``cl3_house``: its colours, PENCIL, LINE)
CREAM = (245, 240, 224)
PENCIL = (33, 28, 26)
LINE_CM = 0.22
#: the house's own centimetres, measured on the table: (colour, left from the middle, bottom,
#: width, height) of each block; the roof a triangle.  No ground under it: simulation's card lost
#: its line for one, read as a plank to lay first (v2.3.1)
BLOCKS = (("orange", -6.9, 0.0, 4.2, 5.4), ("white", 2.7, 0.0, 4.2, 5.4),
          ("salmon", -4.45, 5.4, 8.9, 1.5))
ROOF = ("teal", -2.85, 6.9, 5.7, 2.6)
COLOURS = {"orange": (240, 130, 30), "white": (236, 240, 240), "salmon": (232, 130, 95),
           "teal": (125, 205, 180)}
#: how much larger than the house the drawing is, and where its middle stands
SCALE = 1.4
FOOT_CM = (12.0, 2.3)
#: drawn larger than it is kept, for smooth lines
SAMPLES = 4


def px(cm: float) -> float:
    return cm * SIZE[0] / CARD_CM[0] * SAMPLES


def wobbled(a, b, k: int):
    """An edge a little off straight, the same way every time."""
    s = math.sin(7.3 * k)
    (ax, ay), (bx, by) = a, b
    return [(ax + 0.05 * s, ay - 0.05 * s), (bx - 0.05 * s, by + 0.05 * s)]


def draw_block(d: ImageDraw.ImageDraw, corners, colour, k: int) -> None:
    """A block's outline in house centimetres, onto the picture."""
    def on_card(p):
        return (px(FOOT_CM[0] + SCALE * p[0]), px(CARD_CM[1] - FOOT_CM[1] - SCALE * p[1]))

    d.polygon([on_card(p) for p in corners], fill=colour)
    for j, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])):
        d.line([on_card(p) for p in wobbled(a, b, 10 * k + j)], fill=PENCIL,
               width=round(px(LINE_CM)))


def box(left: float, bottom: float, width: float, height: float):
    return [(left, bottom), (left + width, bottom), (left + width, bottom + height),
            (left, bottom + height)]


def main() -> None:
    card = Image.new("RGB", (SIZE[0] * SAMPLES, SIZE[1] * SAMPLES), CREAM)
    d = ImageDraw.Draw(card)
    for k, (colour, *shape) in enumerate(BLOCKS, 1):
        draw_block(d, box(*shape), COLOURS[colour], k)
    colour, left, bottom, width, height = ROOF
    draw_block(d, [(left, bottom), (left + width, bottom), (left + width / 2, bottom + height)],
               COLOURS[colour], len(BLOCKS) + 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    card.resize(SIZE, Image.LANCZOS).save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
