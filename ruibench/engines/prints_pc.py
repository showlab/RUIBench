"""Printed faces for CR4's computer parts (v1.12, 2026-10-05).

A collaborator on the first clips of CR4: the card and the motherboard look
"a bit cartoon" -- they were flat colours, a green slab, a black one with a
grey ring for its fan, a navy box, bright pink foam.  What a real one's
surfaces show is drawn here, each to the size of the face it goes on
(``SIZES``, in millimetres), and laid on that face as any other print is
(``parts.printed``, ``prints.DESIGNS``):

    mainboard:         a Mini-ITX board from above, 170 mm square: green
                       solder mask, copper under it, traces and vias, the
                       small parts, the silk-screen's outlines and names, a
                       solid-state drive lying on it
    gpu_back:          a low-profile card's back: a black board, the steel
                       cross behind its processor, a serial-number sticker
    gpu_shroud:        its face: the shroud, the fan in its well, the
                       heatsink's fins behind a grille
    gpu_fins:          the heatsink's fins from above the card's top edge
    dimm:              a memory module's side
    fins:              a round heatsink's fins, to wrap round it
    atx:               the board's power connector from above
    carton:<face>      a corrugated carton's ``top`` (its flaps taped shut)
                       or ``side``; carton_end:<face>, the end with the
                       shipping ``label`` or the ``plain`` one
    foam_pad:          pink anti-static foam from above

Every name, number and mark on them is made up.  Nothing here is random
between runs: each design draws from its own seeded generator.
"""

from __future__ import annotations

import math
import random
from typing import Dict, Tuple

#: a design -> (width, height) in millimetres it is drawn for
SIZES: Dict[str, Tuple[float, float]] = {
    "mainboard": (170.0, 170.0), "gpu_back": (150.0, 69.0), "gpu_shroud": (134.0, 61.0),
    "gpu_fins": (126.0, 8.0), "dimm": (80.0, 24.0), "fins": (181.0, 16.0), "atx": (32.0, 20.0),
    "carton": (160.0, 140.0), "carton_end": (140.0, 140.0), "foam_pad": (140.0, 90.0)}

#: the colours the parts beside a print are painted, so that an edge matches
#: its face: a board's edge, the carton's own board, the foam (sRGB, 0-1)
PCB_GREEN = (14 / 255, 82 / 255, 50 / 255)
PCB_BLACK = (21 / 255, 23 / 255, 27 / 255)
KRAFT_RGB = (178 / 255, 140 / 255, 98 / 255)
FOAM_RGB = (236 / 255, 174 / 255, 190 / 255)


class _Later:
    """A module imported when it is first used: the core imports neither
    numpy nor PIL (``prints.py`` imports them inside its functions)."""

    def __init__(self, name: str) -> None:
        self._name, self._module = name, None

    def __getattr__(self, attr):
        if self._module is None:
            from importlib import import_module

            self._module = import_module(self._name)
        return getattr(self._module, attr)


np, Image, ImageDraw = _Later("numpy"), _Later("PIL.Image"), _Later("PIL.ImageDraw")


def _font(size: int, name: str = "DejaVuSansMono-Bold"):
    from .prints import _font as font

    return font(name, size)


def _cloud(shape, cell: int, seed: int):
    """Smooth noise in [-1, 1]: a coarse random grid, enlarged."""
    h, w = shape
    coarse = np.random.default_rng(seed).random((max(2, h // cell + 2), max(2, w // cell + 2)))
    img = Image.fromarray((coarse * 255).astype("uint8")).resize((w, h), Image.BICUBIC)
    return np.asarray(img, dtype="float32") / 127.5 - 1.0


def _grain(shape, seed: int, amount: float):
    return np.random.default_rng(seed).normal(0.0, amount, shape).astype("float32")


def _lit(img, factor):
    """An image, each pixel's brightness scaled by ``factor``."""
    a = np.asarray(img, dtype="float32") * factor[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


# -- a board, drawn in its own millimetres ------------------------------------------
MASK = (14, 82, 50)              # solder mask over bare laminate
POUR = (17, 91, 56)              # ... over copper
TRACE = (34, 122, 76)
GAP = (9, 62, 38)
SILK = (236, 236, 228)
GOLD = (212, 176, 88)
TIN = (196, 198, 194)
CHIP = (24, 24, 27)
MARK = (96, 98, 100)
SHADOW = (8, 50, 30)             # what stands on the board darkens the mask beside it
BOARD_MM, BOARD_S = 170.0, 8


class Board:
    """Drawing on a board in its own millimetres: x up the image, y to its left
    (a print's top is the part's +x, its right the part's -y)."""

    def __init__(self, w_mm, h_mm, scale, ground):
        self.w, self.h, self.s = w_mm, h_mm, scale
        self.img = Image.new("RGB", (round(w_mm * scale), round(h_mm * scale)), ground)
        self.d = ImageDraw.Draw(self.img)

    def p(self, x, y):
        return ((self.w / 2 - y) * self.s, (self.h / 2 - x) * self.s)

    def px(self, mm):
        return max(1, round(mm * self.s))

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=0.0, radius=0.0):
        (ax, ay), (bx, by) = self.p(x1, y1), self.p(x0, y0)
        box = [min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)]
        kw = dict(fill=fill, outline=outline, width=self.px(width) if outline else 0)
        if radius:
            self.d.rounded_rectangle(box, radius=self.px(radius), **kw)
        else:
            self.d.rectangle(box, **kw)

    def poly(self, pts, fill=None, outline=None):
        self.d.polygon([self.p(x, y) for x, y in pts], fill=fill, outline=outline)

    def line(self, pts, fill, width):
        self.d.line([self.p(x, y) for x, y in pts], fill=fill, width=self.px(width), joint="curve")

    def disc(self, x, y, r, fill=None, outline=None, width=0.0):
        cx, cy = self.p(x, y)
        rr = r * self.s
        self.d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=fill, outline=outline,
                       width=self.px(width) if outline else 0)

    def text(self, x, y, s, size_mm, fill=SILK, turn=0, name="DejaVuSansMono-Bold"):
        """Text centred at (x, y); ``turn`` in quarter turns counter-clockwise on the image."""
        f = _font(self.px(size_mm), name)
        box = f.getbbox(s)
        tile = Image.new("L", (box[2] - box[0] + 4, box[3] - box[1] + 4), 0)
        ImageDraw.Draw(tile).text((2 - box[0], 2 - box[1]), s, font=f, fill=255)
        if turn:
            tile = tile.rotate(90 * turn, expand=True)
        cx, cy = self.p(x, y)
        at = (round(cx - tile.width / 2), round(cy - tile.height / 2))
        self.img.paste(Image.new("RGB", tile.size, fill), at, tile)


def bus(b, pts, n, pitch, width, fill=TRACE):
    """``n`` traces side by side along a path of horizontal, vertical and 45-degree runs."""
    for k in range(n):
        off = (k - (n - 1) / 2) * pitch
        path = []
        for i, (x, y) in enumerate(pts):
            # the offset is perpendicular to the mean of the two runs meeting here
            (x0, y0), (x1, y1) = pts[max(i - 1, 0)], pts[min(i + 1, len(pts) - 1)]
            dx, dy = x1 - x0, y1 - y0
            norm = math.hypot(dx, dy) or 1.0
            path.append((x - dy / norm * off, y + dx / norm * off))
        b.line(path, fill, width)


def chip(b, x, y, w, h, legs=0, label=None, pitch=1.27):
    """A black package, its legs along the two long sides, its mark."""
    b.rect(x - w / 2 - 0.25, y - h / 2 - 0.25, x + w / 2 - 0.25, y + h / 2 - 0.25, fill=SHADOW)
    if legs:
        n = max(2, int((h if h >= w else w) / pitch))
        for k in range(n):
            t = (k - (n - 1) / 2) * pitch
            if h >= w:
                b.rect(x - w / 2 - 0.8, y + t - 0.25, x + w / 2 + 0.8, y + t + 0.25, fill=TIN)
            else:
                b.rect(x + t - 0.25, y - h / 2 - 0.8, x + t + 0.25, y + h / 2 + 0.8, fill=TIN)
    b.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=CHIP)
    b.rect(x - w / 2 + 0.3, y - h / 2 + 0.3, x + w / 2 - 0.3, y + h / 2 - 0.3, outline=(40, 40, 44), width=0.12)
    if label and min(w, h) >= 3.0:
        b.text(x, y, label, min(1.5, min(w, h) * 0.26), fill=MARK, turn=0 if h >= w else 1)
    b.disc(x + w / 2 - 0.9, y + h / 2 - 0.9, 0.3, fill=MARK)


def passive(b, x, y, along_y=True, kind="cap", size=(1.6, 0.8)):
    """A chip capacitor (tan) or resistor (black), tinned at both ends."""
    l, w = size
    body = (176, 138, 96) if kind == "cap" else (30, 30, 32)
    if along_y:
        b.rect(x - w / 2, y - l / 2, x + w / 2, y + l / 2, fill=TIN)
        b.rect(x - w / 2, y - l / 2 + l * 0.22, x + w / 2, y + l / 2 - l * 0.22, fill=body)
    else:
        b.rect(x - l / 2, y - w / 2, x + l / 2, y + w / 2, fill=TIN)
        b.rect(x - l / 2 + l * 0.22, y - w / 2, x + l / 2 - l * 0.22, y + w / 2, fill=body)


def header(b, x, y, nx, ny, pitch=2.54):
    """A pin header: gold pins in a black shroud, its outline in silk."""
    w, h = nx * pitch, ny * pitch
    b.rect(x - w / 2 - 0.5, y - h / 2 - 0.5, x + w / 2 + 0.5, y + h / 2 + 0.5, outline=SILK, width=0.18)
    b.rect(x - w / 2, y - h / 2, x + w / 2, y + h / 2, fill=(26, 26, 28))
    for i in range(nx):
        for j in range(ny):
            b.rect(x - w / 2 + (i + 0.5) * pitch - 0.32, y - h / 2 + (j + 0.5) * pitch - 0.32,
                   x - w / 2 + (i + 0.5) * pitch + 0.32, y - h / 2 + (j + 0.5) * pitch + 0.32, fill=GOLD)


# -- the motherboard ------------------------------------------------------------------
def mainboard(_: str = ""):
    """The motherboard from above, its x up the picture: the slot's place along
    the top, the memory's down the right, the processor's in the lower middle,
    the regulators along the bottom -- where the task's parts stand on it."""
    b = Board(BOARD_MM, BOARD_MM, BOARD_S, MASK)
    rng = random.Random(1717)
    # -- copper under the mask: pours round the processor and along the slot, their edges a dark gap
    for pts in ([(-84, -12), (-62, -12), (-62, 56), (-84, 56)],                       # the regulators' strip
                [(-60, -48), (4, -48), (4, -14), (-60, -14)],                         # between memory and processor
                [(4, -84), (84, -84), (84, -62), (56, -62), (56, 42), (4, 42)],       # the slot's side
                [(-84, -84), (2, -84), (2, -74), (-84, -74)],
                [(14, 46), (56, 46), (56, 84), (14, 84)]):
        b.poly(pts, fill=POUR, outline=GAP)
    # -- traces.  Memory: a wide bus from the socket down to the two memory sockets, with its wiggles
    for x0 in (-56, -40, -24, -8):
        n = 9
        for k in range(n):
            x = x0 + k * 0.62
            wig = [(x, -16)]
            y = -20.0
            for _ in range(rng.randint(0, 3)):
                wig += [(x, y), (x + 0.3, y - 0.6), (x + 0.3, y - 1.8), (x, y - 2.4)]
                y -= 3.4
            wig += [(x, -50)]
            b.line(wig, TRACE, 0.22)
    # the slot's lanes: pairs from the processor's side across to the slot, fanning to its length
    for k in range(16):
        y_slot = -52 + k * 5.4
        y_cpu = -6 + k * 1.3
        pair = [(4, y_cpu), (10, y_cpu), (10 + abs(y_slot - y_cpu) * 0.5 + 6, y_slot), (37, y_slot)]
        for off in (-0.22, 0.22):
            b.line([(x, y + off) for x, y in pair], TRACE, 0.2)
        b.disc(37.2, y_slot - 0.22, 0.28, fill=GOLD)
        b.disc(37.2, y_slot + 0.22, 0.28, fill=GOLD)
    # the storage and front connectors: buses along the front edge and up the right side
    bus(b, [(-70, -79), (40, -79), (46, -73), (72, -73)], 10, 0.5, 0.2)
    bus(b, [(66, -60), (66, 30), (60, 36), (60, 44)], 8, 0.55, 0.22)
    bus(b, [(30, 46), (30, 20), (20, 10), (6, 10)], 7, 0.5, 0.2)
    bus(b, [(-50, 60), (-50, 66), (10, 66), (16, 60)], 6, 0.5, 0.2)
    # vias, sprinkled where traces change layer
    for _ in range(520):
        x, y = rng.uniform(-82, 82), rng.uniform(-82, 82)
        b.disc(x, y, 0.3, fill=(150, 152, 120))
        b.disc(x, y, 0.13, fill=(10, 40, 26))

    # -- the processor's socket (the cooler stands over it)
    b.rect(-52, -2, -8, 42, fill=(58, 58, 60), outline=(120, 122, 124), width=0.5)
    b.rect(-46, 4, -14, 36, fill=(190, 160, 96))
    b.text(-30, 20, "LGA", 3.0, fill=(90, 80, 50))
    b.d.ellipse([*b.p(-30 + 31.5, 20 + 31.5), *b.p(-30 - 31.5, 20 - 31.5)], outline=SILK, width=b.px(0.2))

    # -- the regulators, left of the processor: a choke, its two transistors and its capacitors, six times
    for k in range(6):
        y = -6 + k * 10.6
        b.rect(-73.6, y - 3.6, -66.4, y + 3.6, fill=SHADOW, radius=0.8)
        b.rect(-73.4, y - 3.4, -66.2, y + 3.8, fill=(52, 53, 56), radius=0.8)            # the choke
        b.rect(-72.0, y - 2.0, -67.6, y + 2.4, fill=(70, 71, 74), radius=0.4)
        b.text(-69.8, y + 0.2, "R22", 1.5, fill=(150, 152, 154), turn=1)
        chip(b, -79.5, y - 2.2, 4.2, 3.2, label=None)
        chip(b, -79.5, y + 2.2, 4.2, 3.2, label=None)
        b.rect(-82.4, y - 3.2, -81.6, y - 1.2, fill=TIN)
        b.rect(-82.4, y + 1.2, -81.6, y + 3.2, fill=TIN)
        for dy in (-3.0, -1.0, 1.0, 3.0):
            passive(b, -63.6, y + dy + 0.4, along_y=False)
    b.text(-76.5, 58.2, "VCORE", 1.6)
    # along the top of the socket: capacitors (their cans stand here) and a controller
    for x in (-48, -38, -28, -18):
        b.disc(x, 55, 3.5, outline=SILK, width=0.18)
    chip(b, -6, 51, 5, 5, legs=0, label="PWM")
    for i in range(6):
        passive(b, -11.5 + (i % 3) * 1.3 - 1.0, 46.5 + (i // 3) * 2.2, along_y=True, kind="res")

    # -- memory: the two sockets' outlines and names (the sockets stand on them)
    for y, name in ((-55, "DDR5_A1"), (-66, "DDR5_B1")):
        b.rect(-76, y - 4.6, 16, y + 4.6, outline=SILK, width=0.2)
        b.text(22.5, y, name, 1.5, turn=1)
    for i in range(26):
        passive(b, -74 + i * 3.4, -60.5, along_y=True, kind="cap" if i % 3 else "res")
    for i in range(18):
        passive(b, -70 + i * 4.6, -72.6, along_y=False, kind="cap")

    # -- the long slot: its outline, its name, the row of little capacitors every lane has
    b.rect(36.5, -61, 53.5, 41, outline=SILK, width=0.22)
    b.text(57.3, -10, "PCIEX16_1", 2.0, turn=1)
    for k in range(32):
        passive(b, 33.6, -53 + k * 2.75, along_y=False, size=(1.3, 0.65))

    # -- a solid-state drive lying between processor and slot: its connector, its screw
    b.rect(8.5, -46, 30.5, 34, fill=SHADOW)
    b.rect(8, -45.5, 30, 34.5, fill=(18, 20, 24))
    b.rect(8, 34.5, 30, 40.5, fill=(34, 34, 36))                                         # the connector
    for k in range(20):
        b.rect(9 + k * 1.0, 33.8, 9.5 + k * 1.0, 35.2, fill=GOLD)
    # its label: dark, with a band of blue -- beside the long white slot nothing else is long and white
    b.rect(10.5, -28, 27.5, 26, fill=(44, 46, 52))
    b.rect(10.5, 18, 27.5, 26, fill=(28, 60, 130))
    b.text(19, 22, "NVMe SSD", 2.0, fill=(240, 240, 240), turn=1)
    b.text(19, 4, "1TB", 4.2, fill=(214, 216, 220), turn=1)
    b.text(24, -8, "M.2 2280  PCIe 4.0 x4", 1.3, fill=(150, 153, 158), turn=1, name="DejaVuSansMono")
    b.text(13.5, -8, "S/N 7C2A41K0093", 1.2, fill=(150, 153, 158), turn=1, name="DejaVuSansMono")
    for k in range(34):                                                                  # a bar code
        w = rng.choice((0.25, 0.25, 0.5, 0.75))
        b.rect(15.6, -26 + k * 0.6, 17.6 + 2.6, -26 + k * 0.6 + w * 0.5, fill=(190, 192, 196))
    b.disc(19, -43.2, 2.3, fill=TIN)
    b.disc(19, -43.2, 1.3, fill=(120, 122, 122))
    b.line([(19 - 0.9, -43.2), (19 + 0.9, -43.2)], (70, 72, 72), 0.25)
    b.line([(19, -43.2 - 0.9), (19, -43.2 + 0.9)], (70, 72, 72), 0.25)
    b.text(3.2, -30, "M2_1", 1.6, turn=1)

    # -- the chipset, under its heatsink (which stands on this square), and its little parts
    b.rect(16, 48, 44, 76, outline=SILK, width=0.2)
    for i in range(10):
        passive(b, 47.2, 49.5 + i * 2.7, along_y=False)
    # -- the battery's holder, the clock crystal, the firmware chip
    b.disc(70, 12, 10.6, outline=SILK, width=0.22)
    b.text(70, 25.2, "BAT1  CR2032", 1.5)
    chip(b, 72, -8, 5.2, 5.4, legs=1, label="BIOS")
    b.rect(63.2, -11.6, 66.4, -4.4, fill=TIN, radius=1.0)                                 # a crystal's can
    b.text(64.8, -8, "25.000", 1.0, fill=(90, 92, 92), turn=1, name="DejaVuSansMono")
    chip(b, 72, -22, 7, 7, legs=0, label="SIO")
    for i in range(8):
        passive(b, 78.5, -26 + i * 2.1, along_y=False, kind="res")
    chip(b, 70, -40, 4, 5, legs=1)
    # -- headers along the front edge and the right
    header(b, 70, -52, 2, 5)
    b.text(70, -44.2, "F_PANEL", 1.4)
    header(b, -40, -80.5, 10, 2)
    b.text(-40, -76.6, "F_USB32", 1.4)
    header(b, -8, -80.5, 5, 2)
    b.text(-8, -76.6, "F_AUDIO", 1.4)
    header(b, -70, -80.5, 4, 1)
    b.text(-70, -77.6, "CPU_FAN", 1.4)
    for x, name in ((30, "SATA6G_1"), (48, "SATA6G_2")):
        b.rect(x - 7.5, -84, x + 7.5, -70, outline=SILK, width=0.2)
        b.text(x, -67.8, name, 1.4)
    # -- power: the big connector's outline and name (it stands here), the processor's own by the rear
    b.rect(59, 43, 81, 77, outline=SILK, width=0.22)
    b.text(56.6, 60, "ATX_PWR1", 1.6, turn=1)
    b.rect(-50, 71, -30, 83, outline=SILK, width=0.2)
    b.text(-40, 68.8, "EPS_12V", 1.4)
    # -- the rear connectors' outlines
    for x0, x1, name in ((-26, -8, "HDMI_DP"), (-4, 14, "AUDIO")):
        b.rect(x0, 70, x1, 84, outline=SILK, width=0.2)
    # -- the board's name, its marks
    b.text(-7.5, -31, "ITX170-M", 3.4)
    b.text(-2.2, -24.5, "REV 1.02", 1.5)
    # more of the board's wiring: the memory's second run, the slot's clocks, the rear connectors
    bus(b, [(-58, -73.2), (0, -73.2)], 14, 0.42, 0.18)
    bus(b, [(34, 44), (34, 64), (26, 72), (16, 72)], 6, 0.5, 0.2)
    bus(b, [(-6, 46), (-6, 58), (-14, 66), (-26, 66)], 8, 0.45, 0.18)
    bus(b, [(56, -66), (80, -66)], 9, 0.5, 0.2)
    bus(b, [(74, 30), (74, 40), (80, 46), (80, 56)], 5, 0.55, 0.22)
    for k in range(7):                                    # test points, in a row
        b.disc(-16 + k * 2.6, -12.4, 0.7, fill=GOLD)
    for x, y in ((78.5, 78.5), (78.5, -78.5), (-78.5, -78.5), (-78.5, 78.5)):            # mounting holes
        b.disc(x, y, 3.6, fill=TIN)
        for k in range(8):
            a = k * math.pi / 4
            b.disc(x + 2.75 * math.cos(a), y + 2.75 * math.sin(a), 0.32, fill=(120, 122, 120))
        b.disc(x, y, 1.9, fill=(46, 40, 30))
    # scattered small parts
    for _ in range(90):
        x, y = rng.uniform(-80, 80), rng.uniform(-80, 80)
        if (-62 < x < 2 and -12 < y < 52) or (34 < x < 56 and -62 < y < 42) or (6 < x < 32 and -48 < y < 42):
            continue
        if ((x < -52 and y > 56) or (58 < x < 82 and 42 < y < 78) or (14 < x < 46 and 46 < y < 78)
                or (-78 < x < 18 and -72 < y < -48)):
            continue
        passive(b, x, y, along_y=rng.random() < 0.5, kind=rng.choice(("cap", "cap", "res")))
    # -- the laminate's weave, a little unevenness, the mask's fine grain
    a = np.asarray(b.img, dtype="float32")
    h, w = a.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    weave = 1.0 + 0.012 * (np.sin(xx * 2 * math.pi / 5.0) + np.sin(yy * 2 * math.pi / 5.0))
    factor = weave * (1.0 + 0.05 * _cloud((h, w), 160, 3)) + _grain((h, w), 4, 0.012)
    return Image.fromarray(np.clip(a * factor[..., None], 0, 255).astype("uint8"))


# -- the graphics card ----------------------------------------------------------------
BLACK_MASK = (21, 23, 27)
BLACK_POUR = (27, 30, 35)
BLACK_TRACE = (40, 44, 51)
CARD_W, CARD_H, CARD_S = 150.0, 69.0, 10
TAB_MID, TAB_LEN = -22.5, 85.0           # the gold edge under the board: its middle along the card, its length


class Face(Board):
    """Drawing on an upright face in its own millimetres: u to the right, v up."""

    def p(self, u, v):
        return ((self.w / 2 + u) * self.s, (self.h / 2 - v) * self.s)


def bar_code(f, u0, v0, u1, v1, rng, ink=(26, 26, 28), step=1.0):
    """A bar code: bars of four widths, ``step`` scaling them."""
    u = u0
    while u < u1 - 0.3 * step:
        w = rng.choice((0.2, 0.2, 0.4, 0.6)) * step
        f.rect(u, v0, min(u + w, u1), v1, fill=ink)
        u += w + rng.choice((0.2, 0.2, 0.4)) * step


def gpu_back(_: str = ""):
    """The card's back: a black board, the retention bracket behind its processor,
    the rows of capacitors there, a serial-number sticker.  Seen from behind:
    the bracket's end at the left."""
    f = Face(CARD_W, CARD_H, CARD_S, BLACK_MASK)
    rng = random.Random(2204)
    # copper areas and the lanes running down to the gold edge
    f.poly([(-74, -33), (-30, -33), (-30, 33), (-74, 33)], fill=BLACK_POUR)
    f.poly([(24, -33), (74, -33), (74, 33), (24, 33)], fill=BLACK_POUR)
    for k in range(40):
        u_edge = TAB_MID - TAB_LEN / 2 + 3 + k * 2.0
        u_chip = -18 + k * 0.62
        for off in (-0.16, 0.16):
            f.line([(u_edge + off, -34.5), (u_edge + off, -27), (u_chip + off, -18 - (k % 5) * 0.6),
                    (u_chip + off, -12)], BLACK_TRACE, 0.16)
    for k in range(18):
        v = -8 + k * 0.9
        f.line([(8, v), (20, v), (26 + k * 0.5, v + 8), (60, v + 8)], BLACK_TRACE, 0.18)
    for _ in range(360):
        u, v = rng.uniform(-73, 73), rng.uniform(-33, 33)
        f.disc(u, v, 0.28, fill=(98, 100, 98))
        f.disc(u, v, 0.12, fill=(12, 12, 14))
    # behind the processor: its field of capacitors, and the steel cross that holds the cooler on
    cu, cv = -6.0, 4.0
    for i in range(11):
        for j in range(11):
            if 3 <= i <= 7 and 3 <= j <= 7 and (i + j) % 2:
                continue
            u, v = cu - 12.5 + i * 2.5, cv - 12.5 + j * 2.5
            f.rect(u - 0.55, v - 0.9, u + 0.55, v + 0.9, fill=TIN)
            f.rect(u - 0.55, v - 0.45, u + 0.55, v + 0.45, fill=(150, 116, 80))
    steel, steel_dark, steel_light = (150, 153, 156), (96, 99, 102), (188, 191, 194)
    arm = 23.5
    for su, sv in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        tip = (cu + su * arm, cv + sv * arm)
        f.d.line([f.p(cu + su * 13.0, cv + sv * 13.0), f.p(*tip)], fill=steel_dark, width=f.px(6.4))
        f.d.line([f.p(cu + su * 13.0, cv + sv * 13.0), f.p(*tip)], fill=steel, width=f.px(5.6))
        f.disc(*tip, 3.4, fill=steel_dark)
        f.disc(*tip, 3.0, fill=steel)
        f.disc(*tip, 1.9, fill=(70, 72, 75))
        f.line([(tip[0] - 1.3, tip[1]), (tip[0] + 1.3, tip[1])], (36, 37, 40), 0.35)
        f.line([(tip[0], tip[1] - 1.3), (tip[0], tip[1] + 1.3)], (36, 37, 40), 0.35)
    f.rect(cu - 15.2, cv - 15.2, cu + 15.2, cv + 15.2, outline=steel_dark, width=3.4)
    f.rect(cu - 14.8, cv - 14.8, cu + 14.8, cv + 14.8, outline=steel, width=2.6)
    f.rect(cu - 14.6, cv - 14.6, cu + 14.6, cv + 14.6, outline=steel_light, width=0.3)
    # the memory chips' backs, the regulators' side
    for u, v, w, h in ((-44, 18, 12, 10), (-44, -8, 12, 10), (32, 20, 12, 10)):
        for i in range(5):
            for j in range(4):
                uu, vv = u - w / 2 + 1.2 + i * 2.4, v - h / 2 + 1.2 + j * 2.5
                f.rect(uu - 0.45, vv - 0.8, uu + 0.45, vv + 0.8, fill=TIN)
                f.rect(uu - 0.45, vv - 0.4, uu + 0.45, vv + 0.4, fill=(150, 116, 80))
    for k in range(7):
        u = 34 + k * 5.4
        f.rect(u - 2.0, -22, u + 2.0, -14, fill=(34, 35, 39), outline=(58, 60, 64), width=0.15)
        f.rect(u - 2.0, -24.2, u + 2.0, -22.6, fill=TIN)
        for j in range(3):
            f.rect(u - 0.45, -9 + j * 2.6 - 0.8, u + 0.45, -9 + j * 2.6 + 0.8, fill=TIN)
            f.rect(u - 0.45, -9 + j * 2.6 - 0.4, u + 0.45, -9 + j * 2.6 + 0.4, fill=(150, 116, 80))
    # the cooler's four other screws, the bracket's two
    for u, v in ((-66, 28), (-66, -26), (68, 28), (68, -26), (22, 30), (-34, -28)):
        f.disc(u, v, 2.6, fill=(70, 72, 75))
        f.disc(u, v, 1.9, fill=steel)
        f.line([(u - 1.1, v - 1.1), (u + 1.1, v + 1.1)], (48, 50, 53), 0.3)
        f.line([(u - 1.1, v + 1.1), (u + 1.1, v - 1.1)], (48, 50, 53), 0.3)
    # its sticker: a serial number and a bar code; the marks every board carries
    f.rect(36, 4, 70, 15, fill=(240, 240, 236), radius=0.5)
    bar_code(f, 38, 9.4, 68, 13.8, rng)
    f.text(53, 6.4, "S/N  GC4F0-771-02845", 1.5, fill=(30, 30, 34), name="DejaVuSansMono")
    f.text(-55, -28.5, "GFX-LP4  REV 2.1", 1.9, fill=(206, 208, 204))
    f.text(54, -30.5, "94V-0   E252619", 1.3, fill=(150, 152, 150), name="DejaVuSansMono")
    f.rect(-71.5, -6, -61.5, 6, outline=(206, 208, 204), width=0.18)
    f.text(-66.5, 0, "PCB", 1.4, fill=(206, 208, 204))
    a = np.asarray(f.img, dtype="float32")
    h, w = a.shape[:2]
    factor = 1.0 + 0.06 * _cloud((h, w), 140, 5) + _grain((h, w), 6, 0.02)
    return Image.fromarray(np.clip(a * factor[..., None], 0, 255).astype("uint8"))


SHROUD_W, SHROUD_H = 134.0, 61.0
FAN_AT, FAN_R = (-8.0, 0.0), 27.0        # on the shroud's face, seen from the front (the bracket to the right)


def fan(f, cu, cv, r, hub, blades, sweep=58.0, sticker=True):
    """A fan seen from its face: its well, its blades swept back, its hub and the hub's label."""
    f.disc(cu, cv, r + 1.6, fill=(58, 59, 63))
    f.disc(cu, cv, r + 0.9, fill=(34, 35, 38))
    f.disc(cu, cv, r, fill=(9, 9, 11))
    for k in range(blades):
        a0 = 2 * math.pi * k / blades
        lead, trail, glint = [], [], []
        steps = 16
        for i in range(steps + 1):
            t = i / steps
            rr = hub * 0.9 + (r - 0.9 - hub * 0.9) * t
            # the blade leans back as it goes out, and widens
            lean = math.radians(sweep) * t ** 1.2
            half = math.radians(5.0 + 14.0 * t ** 0.75)
            lead.append((cu + rr * math.cos(a0 - lean + half), cv + rr * math.sin(a0 - lean + half)))
            trail.append((cu + rr * math.cos(a0 - lean - half), cv + rr * math.sin(a0 - lean - half)))
            glint.append((cu + rr * math.cos(a0 - lean + half * 0.45), cv + rr * math.sin(a0 - lean + half * 0.45)))
        f.d.polygon([f.p(*q) for q in lead + trail[::-1]], fill=(40, 41, 45))
        # the light catches the half of the blade behind its leading edge
        f.d.polygon([f.p(*q) for q in lead + glint[::-1]], fill=(54, 56, 61))
        f.d.line([f.p(*q) for q in lead], fill=(96, 98, 104), width=f.px(0.4))
    f.disc(cu, cv, hub, fill=(30, 31, 34))
    f.disc(cu, cv, hub - 0.5, fill=(40, 41, 45))
    if sticker:
        rs = hub - 1.8
        f.disc(cu, cv, rs, fill=(176, 179, 184))
        # a brushed-metal label: bright and dark sectors
        x0, y0 = f.p(cu - rs, cv + rs)
        x1, y1 = f.p(cu + rs, cv - rs)
        for a_deg, tone in ((20, 222), (65, 150), (110, 205), (200, 226), (245, 148), (290, 200)):
            f.d.pieslice([x0, y0, x1, y1], a_deg - 14, a_deg + 14, fill=(tone, tone + 2, tone + 6))
        f.disc(cu, cv, rs, outline=(110, 112, 116), width=0.25)
        f.disc(cu, cv, rs * 0.33, fill=(60, 62, 66))
        f.disc(cu, cv, rs * 0.2, fill=(130, 133, 138))


def gpu_shroud(_: str = ""):
    """The card's face: a black shroud, its fan in a round well, the heatsink's
    fins showing through a grille at the bracket's end.  Seen from the front:
    the bracket's end at the right."""
    f = Face(SHROUD_W, SHROUD_H, CARD_S, (27, 27, 30))
    poly = f.poly
    # the shroud's facets: raised panels either side of the fan, a chamfer along the top and bottom
    poly([(-67, 30.5), (67, 30.5), (64, 26.5), (-64, 26.5)], (40, 41, 45))
    poly([(-67, -30.5), (67, -30.5), (64, -26.5), (-64, -26.5)], (17, 17, 19))
    poly([(-67, 30.5), (-64, 26.5), (-64, -26.5), (-67, -30.5)], (33, 34, 37))
    poly([(67, 30.5), (64, 26.5), (64, -26.5), (67, -30.5)], (21, 21, 24))
    poly([(-62, 24), (-40, 24), (-46, -24), (-62, -24)], (33, 33, 37))
    poly([(-62, 24), (-40, 24), (-40.6, 23.2), (-61.2, 23.2)], (52, 53, 57))
    poly([(22, 24), (62, 24), (62, -24), (28, -24)], (33, 33, 37))
    poly([(22, 24), (62, 24), (61.2, 23.2), (22.8, 23.2)], (52, 53, 57))
    # an accent: a satin strip across the far panel
    poly([(-60, 6), (-43.3, 6), (-43.9, 2.2), (-60, 2.2)], (146, 149, 154))
    poly([(-60, 6), (-43.3, 6), (-43.4, 5.4), (-60, 5.4)], (196, 199, 204))
    # the grille at the bracket's end: fins behind slanted bars
    gx0, gx1, gy0, gy1 = 31.0, 59.0, -19.0, 19.0
    f.rect(gx0, gy0, gx1, gy1, fill=(12, 12, 14), radius=1.2)
    u = gx0 + 0.9
    while u < gx1 - 0.6:
        f.rect(u, gy0 + 0.8, u + 0.42, gy1 - 0.8, fill=(168, 171, 176))
        f.rect(u + 0.42, gy0 + 0.8, u + 0.62, gy1 - 0.8, fill=(96, 98, 102))
        u += 1.25
    for k in range(5):
        v = gy0 + 3.8 + k * 7.6
        poly([(gx0, v - 1.0), (gx1, v + 1.6), (gx1, v + 3.4), (gx0, v + 0.8)], (30, 30, 33))
        poly([(gx0, v + 0.5), (gx1, v + 3.1), (gx1, v + 3.4), (gx0, v + 0.8)], (58, 59, 63))
    f.rect(gx0, gy0, gx1, gy1, outline=(50, 51, 55), width=0.35, radius=1.2)
    fan(f, FAN_AT[0], FAN_AT[1], FAN_R, 9.0, 9)
    # screws at the shroud's corners
    for u, v in ((-63.5, 27.5), (-63.5, -27.5), (63.5, 27.5), (63.5, -27.5), (-36, 0), (24.5, 0)):
        f.disc(u, v, 1.5, fill=(12, 12, 14))
        f.disc(u, v, 1.05, fill=(74, 76, 80))
        f.line([(u - 0.7, v), (u + 0.7, v)], (30, 30, 33), 0.22)
    a = np.asarray(f.img, dtype="float32")
    h, w = a.shape[:2]
    factor = 1.0 + 0.05 * _cloud((h, w), 120, 8) + _grain((h, w), 9, 0.03)
    return Image.fromarray(np.clip(a * factor[..., None], 0, 255).astype("uint8"))


# -- a memory module ------------------------------------------------------------------
DIMM_W, DIMM_H, DIMM_S = 80.0, 24.0, 12


def dimm(_: str = ""):
    """A memory module's side: eight chips on a black board, its label across four of them."""
    f = Face(DIMM_W, DIMM_H, DIMM_S, BLACK_MASK)
    rng = random.Random(64)
    f.rect(-40, -12, 40, -8.6, fill=(26, 60, 40))
    for k in range(62):                                                      # the gold contacts
        u = -38.6 + k * 1.25
        if 2.0 < u < 5.0:
            continue                                                         # the key's notch
        f.rect(u, -12, u + 0.75, -9.0, fill=GOLD)
    f.rect(2.4, -12, 4.6, -9.6, fill=(8, 8, 9))
    for k in range(8):
        u = -34.5 + k * 9.86
        f.rect(u - 3.9, -5.2, u + 3.9, 7.6, fill=(12, 12, 14))
        f.rect(u - 3.7, -5.0, u + 3.7, 7.4, fill=(30, 30, 33))
        f.text(u, 1.6, "D5", 1.5, fill=(92, 94, 98))
        for j in range(4):
            f.rect(u - 3.0 + j * 2.0 - 0.3, -7.3, u - 3.0 + j * 2.0 + 0.3, -6.3, fill=(150, 116, 80))
    f.rect(-19.6, -4.4, 19.6, 6.8, fill=(186, 189, 194), radius=0.4)          # a silver label
    f.text(-8.5, 3.2, "16GB DDR5-5600", 1.9, fill=(26, 26, 30))
    f.text(-9.6, -0.2, "CL46 1.1V  UDIMM", 1.2, fill=(60, 60, 66), name="DejaVuSansMono")
    bar_code(f, 6.5, -2.8, 18.2, 5.2, rng)
    f.text(-9.4, -2.8, "S/N 4K07A1193", 1.1, fill=(60, 60, 66), name="DejaVuSansMono")
    f.disc(-37.6, 9.4, 1.1, fill=(8, 8, 9))
    f.disc(37.6, 9.4, 1.1, fill=(8, 8, 9))
    a = np.asarray(f.img, dtype="float32")
    h, w = a.shape[:2]
    return Image.fromarray(np.clip(a * (1.0 + _grain((h, w), 11, 0.02))[..., None], 0, 255).astype("uint8"))


# -- the card's carton ----------------------------------------------------------------
KRAFT = (178, 140, 98)
TAPE = (196, 160, 112)
CARTON_W, CARTON_H, CARTON_S = 160.0, 140.0, 6


def kraft(w_mm, h_mm, seed, flutes_along_u=True):
    """Corrugated board: kraft paper, its fibres, the flutes showing faintly through the liner."""
    f = Face(w_mm, h_mm, CARTON_S, KRAFT)
    h, w = f.img.size[1], f.img.size[0]
    yy, xx = np.mgrid[0:h, 0:w]
    along = yy if flutes_along_u else xx
    flutes = 1.0 + 0.022 * np.sin(along * 2 * math.pi / (3.4 * CARTON_S))
    streaks = (np.random.default_rng(seed).random((h // 2, w // 24 + 2)) * 255).astype("uint8")
    fibres = np.asarray(Image.fromarray(streaks).resize((w, h), Image.BICUBIC), dtype="float32") / 255.0
    factor = flutes * (0.94 + 0.10 * fibres) * (1.0 + 0.05 * _cloud((h, w), 150, seed + 1))
    factor = factor + _grain((h, w), seed + 2, 0.018)
    f.img = _lit(f.img, factor)
    f.d = ImageDraw.Draw(f.img)
    return f


def tape(f, u0, v0, u1, v1, seed):
    """Packing tape: glossy brown film, lighter where the light lies on it."""
    (ax, ay), (bx, by) = f.p(u0, v1), f.p(u1, v0)
    x0, y0, x1, y1 = (round(v) for v in (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by)))
    a = np.asarray(f.img, dtype="float32").copy()
    h, w = y1 - y0, x1 - x0
    film = np.array(TAPE, dtype="float32")[None, None, :] * (1.0 + 0.06 * _cloud((h, w), 60, seed))[..., None]
    sheen = np.zeros((h, w), dtype="float32")
    if w >= h:
        sheen += 0.10 * np.exp(-((np.arange(h)[:, None] - h * 0.3) / (h * 0.12)) ** 2)
    else:
        sheen += 0.10 * np.exp(-((np.arange(w)[None, :] - w * 0.3) / (w * 0.12)) ** 2)
    a[y0:y1, x0:x1] = 0.22 * a[y0:y1, x0:x1] + 0.78 * film * (1.0 + sheen)[..., None]
    a[y0:y0 + 1, x0:x1] *= 0.86
    a[y1 - 1:y1, x0:x1] *= 0.86
    f.img = Image.fromarray(np.clip(a, 0, 255).astype("uint8"))
    f.d = ImageDraw.Draw(f.img)


INK_BOX = (38, 34, 30)


def arrows_up(f, u, v, size):
    """The 'this way up' mark: two arrows on a line."""
    for du in (-size * 0.55, size * 0.55):
        x = u + du
        pts = [(x, v + size), (x - size * 0.42, v + size * 0.35), (x - size * 0.16, v + size * 0.35),
               (x - size * 0.16, v - size * 0.7), (x + size * 0.16, v - size * 0.7),
               (x + size * 0.16, v + size * 0.35), (x + size * 0.42, v + size * 0.35)]
        f.poly(pts, fill=INK_BOX)
    f.rect(u - size * 1.05, v - size * 0.98, u + size * 1.05, v - size * 0.82, fill=INK_BOX)


def glass(f, u, v, size):
    """The 'fragile' mark: a stemmed glass, cracked."""
    pts = [(-0.5, 1.0), (0.5, 1.0), (0.32, 0.1), (0.07, -0.05), (0.07, -0.75), (0.4, -0.9),
           (-0.4, -0.9), (-0.07, -0.75), (-0.07, -0.05), (-0.32, 0.1)]
    f.poly([(u + size * a, v + size * b) for a, b in pts], fill=INK_BOX)
    crack = [(0.05, 0.85), (-0.12, 0.45), (0.1, 0.3)]
    f.d.line([f.p(u + size * a, v + size * b) for a, b in crack], fill=KRAFT, width=f.px(size * 0.07))


def carton(face: str = "side"):
    """A long face of the carton the card came in: its ``top``, the flaps taped
    shut along their seam, or a ``side`` with its handling marks."""
    f = kraft(CARTON_W, CARTON_H, {"top": 31, "side": 33}[face], flutes_along_u=face != "top")
    if face == "top":
        f.line([(-80, 0), (80, 0)], (112, 84, 54), 0.5)                    # where the flaps meet
        tape(f, -80, -24, 80, 24, 5)
        return f.img
    arrows_up(f, -44, 2, 15)
    f.text(-44, -22, "THIS SIDE UP", 3.6, fill=INK_BOX)
    f.text(40, 4, "HANDLE WITH CARE", 3.6, fill=INK_BOX)
    f.text(40, -4, "ESD SENSITIVE DEVICE", 2.8, fill=INK_BOX, name="DejaVuSansMono")
    return f.img


def carton_end(face: str = "plain"):
    """An end of the carton, the tape's end coming down over it: the one with
    the shipping ``label`` and the marks, or the ``plain`` one."""
    f = kraft(140.0, 140.0, {"label": 41, "plain": 42}[face])
    rng = random.Random(77)
    tape(f, -24, 70 - 36, 24, 70, 6)
    if face == "plain":
        f.text(0, 2, "LP-GFX 4G", 7.0, fill=INK_BOX)
        f.text(0, -10, "MODEL GC4F0  ·  1 PC", 3.4, fill=INK_BOX, name="DejaVuSansMono")
        f.rect(-30, -44, 30, -28, outline=INK_BOX, width=0.7)
        f.text(0, -36, "KEEP DRY", 4.0, fill=INK_BOX)
        return f.img
    f.rect(-62, -58, 4, -12, fill=(244, 243, 238), radius=0.6)              # the shipping label
    f.rect(-62, -21, 4, -12, fill=(40, 40, 44))
    f.text(-29, -16.5, "FRAGILE  ELECTRONICS", 3.0, fill=(244, 243, 238))
    f.text(-47, -26.5, "SHIP TO", 2.0, fill=(60, 60, 66))
    for k, line in enumerate(("RUI LAB, BENCH 4", "ROBOTICS BUILDING", "ROOM 1.11")):
        f.text(-36, -32 - k * 4.2, line, 2.6, fill=(30, 30, 34))
    bar_code(f, -58, -55.5, -24, -48, rng, step=1.8)
    f.text(-9, -52, "1 / 1", 3.4, fill=(30, 30, 34))
    arrows_up(f, 40, -30, 12)
    f.text(40, -49, "THIS SIDE UP", 3.2, fill=INK_BOX)
    glass(f, 50, 14, 9)
    f.rect(37, 2, 63, 27, outline=INK_BOX, width=0.9)
    f.text(-28, 18, "GRAPHICS CARD", 5.2, fill=INK_BOX)
    f.text(-28, 10, "LOW PROFILE  ·  QTY 1", 2.8, fill=INK_BOX, name="DejaVuSansMono")
    return f.img


# -- anti-static foam, a heatsink's fins, the power connector --------------------------
FOAM = (236, 174, 190)


def foam(w_mm, h_mm, seed):
    """Pink anti-static foam from above: its fine cells, its edge pressed a little darker."""
    s = 6
    w, h = round(w_mm * s), round(h_mm * s)
    rng = np.random.default_rng(seed)
    coarse = (rng.random((h // 3 + 1, w // 3 + 1)) * 255).astype("uint8")
    cells = np.asarray(Image.fromarray(coarse).resize((w, h), Image.BILINEAR), dtype="float32") / 255.0
    factor = (0.93 + 0.12 * cells) * (1.0 + 0.04 * _cloud((h, w), 120, seed + 1)) + _grain((h, w), seed + 2, 0.02)
    yy, xx = np.mgrid[0:h, 0:w]
    edge = np.minimum.reduce([xx, yy, w - 1 - xx, h - 1 - yy]).astype("float32")
    factor *= 0.93 + 0.07 * np.clip(edge / (2.0 * s), 0, 1)
    a = np.array(FOAM, dtype="float32")[None, None, :] * factor[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def foam_pad(_: str = ""):
    return foam(140.0, 90.0, 51)


# ---------------------------------------------------------------------------------- the power connector's top
def atx(_: str = ""):
    """The board's power connector from above: natural nylon, two rows of square sockets."""
    f = Face(32.0, 20.0, 16, (232, 226, 206))
    for i in range(10):
        for j in range(2):
            u, v = -13.5 + i * 3.0, -4.2 + j * 8.4
            # every other socket is keyed: its corners rounded
            f.rect(u - 1.25, v - 1.25, u + 1.25, v + 1.25, fill=(206, 199, 178), radius=0.3 if (i + j) % 2 else 0.0)
            f.rect(u - 0.95, v - 0.95, u + 0.95, v + 0.95, fill=(46, 42, 36), radius=0.25 if (i + j) % 2 else 0.0)
            f.rect(u - 0.3, v - 0.3, u + 0.3, v + 0.3, fill=(150, 126, 70))
    f.rect(-15.6, -9.6, 15.6, 9.6, outline=(204, 196, 172), width=0.4)
    f.rect(-3.2, -1.0, 3.2, 1.0, fill=(214, 207, 186))                      # the latch's ridge
    a = np.asarray(f.img, dtype="float32")
    h, w = a.shape[:2]
    return Image.fromarray(np.clip(a * (1.0 + _grain((h, w), 13, 0.012))[..., None], 0, 255).astype("uint8"))




def fins(_: str = ""):
    """A round heatsink's fins, as a band to wrap round it: thin aluminium
    plates standing side by side, the dark between them."""
    s = 8
    w, h = round(SIZES["fins"][0] * s), round(SIZES["fins"][1] * s)
    xx = np.arange(w, dtype="float32")[None, :].repeat(h, axis=0)
    yy = np.arange(h, dtype="float32")[:, None].repeat(w, axis=1)
    pitch = w / 72.0                                    # 72 fins round it
    phase = (xx % pitch) / pitch
    plate = np.where(phase < 0.46, 1.0, 0.0)
    tone = 46 + plate * (150 + 40 * np.sin(phase / 0.46 * math.pi))
    tone *= 0.82 + 0.18 * (yy / h)                      # darker toward the board
    tone[: round(0.9 * s)] = 96                         # the base plate's rim, top and bottom
    tone[-round(0.9 * s):] = 70
    a = np.stack([tone * 0.985, tone * 0.99, tone], axis=-1)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


def gpu_fins(_: str = ""):
    """The card's heatsink from above its top edge: the fins' tops, side by
    side along the card, two heat pipes crossing them."""
    w_mm, h_mm = SIZES["gpu_fins"]
    s = 10
    w, h = round(w_mm * s), round(h_mm * s)
    xx = np.arange(w, dtype="float32")[None, :].repeat(h, axis=0)
    yy = np.arange(h, dtype="float32")[:, None].repeat(w, axis=1)
    pitch = 1.6 * s
    phase = (xx % pitch) / pitch
    tone = np.where(phase < 0.42, 150.0 + 60.0 * np.sin(phase / 0.42 * math.pi), 34.0)
    for v_mm in (2.4, 5.6):                             # the pipes, nickel-plated copper
        d = np.abs(yy - v_mm * s) / (0.9 * s)
        tone = np.where(d < 1.0, 120.0 + 90.0 * (1.0 - d ** 2), tone)
    tone *= 0.9 + 0.1 * (yy / h)
    a = np.stack([tone * 0.985, tone * 0.99, tone], axis=-1)
    return Image.fromarray(np.clip(a, 0, 255).astype("uint8"))


DESIGNS = {"mainboard": mainboard, "gpu_back": gpu_back, "gpu_shroud": gpu_shroud, "gpu_fins": gpu_fins,
           "dimm": dimm, "fins": fins, "atx": atx, "carton": carton, "carton_end": carton_end,
           "foam_pad": foam_pad}

__all__ = ["SIZES", "DESIGNS", "PCB_GREEN", "PCB_BLACK", "KRAFT_RGB", "FOAM_RGB"]
