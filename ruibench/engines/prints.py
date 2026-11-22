"""Printed faces: labels, tags and cards drawn as images (v8, 2026-09-24).

Until v8 every label was lettering built from thin boxes (``parts.text``):
legible, but it looked like it (the advisor: DS6's labels "are not
beautiful").  A *print* is a flat face showing an image drawn here from a
short spec (``parts.printed``):

    label:<formula>    a reagent card: the formula, its subscripts set as
                       subscripts, and the salt's name in small print
    letter:<X>         a beaker's label: one letter
    tag:<name>         a gift tag: "To" and a name in a script hand
    sticker:<name>     a label stuck under a present: "To" and a name in
                       print, big enough to read from across the table
    wrap:<paper>       wrapping paper (``WRAPS``): one face of a present
    cloth:<cloth>      a tablecloth (``CLOTHS``): gingham, its weave's grain
    towel:<colour>     a tea towel (``TOWELS``): unbleached cotton, three
                       woven stripes along two of its edges, a hem all round
    tee:<colour>       a child's striped T-shirt (``TEES``), lying flat: the
                       knit, its stripes, the neckband and the hems' stitching
    blank:             plain paper: a label's back

and, each to the size of the face it goes on, what CR4's computer parts show
(``prints_pc.py``: ``mainboard:``, ``gpu_back:``, ``gpu_shroud:``, ``gpu_fins:``,
``dimm:``, ``fins:``, ``atx:``, ``carton:<face>``, ``carton_end:<face>``,
``foam_pad:``).

Drawn with PIL in fonts every install has (matplotlib's DejaVu), the script
hand a pinned download (Dancing Script, SIL OFL 1.1), and cached by spec.
The Isaac finish ``paper:<spec>`` shows the image; everything else sees a
white card.  An image is as wide as its design's ``ASPECT`` times its height:
a print's face must be too, or it is stretched.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Dict, Tuple

#: bump to redraw every cached image after a change to a design
VERSION = 2
#: pixels per millimetre of face, at the size a design is drawn for
PX_PER_MM = 24
#: a design -> (width, height) in millimetres it is drawn for; a print of
#: another size is the same image scaled
SIZES: Dict[str, Tuple[float, float]] = {"label": (80.0, 30.0), "letter": (24.0, 24.0),
                                         "tag": (48.0, 30.0), "blank": (10.0, 10.0),
                                         "sticker": (52.0, 34.0), "wrap": (60.0, 60.0),
                                         "cloth": (152.0, 100.0), "towel": (300.0, 300.0),
                                         "tee": (260.0, 330.0)}
#: what the salts are called, for the small print under a formula
NAMES = {"NaCl": "sodium chloride", "KCl": "potassium chloride", "LiCl": "lithium chloride",
         "CuCl₂": "copper(II) chloride", "CuCl2": "copper(II) chloride",
         "SrCl₂": "strontium chloride", "CaCl₂": "calcium chloride", "BaCl₂": "barium chloride",
         "H₂O": "distilled water"}
SUBSCRIPTS = {chr(0x2080 + d): str(d) for d in range(10)}

INK = (28, 30, 36)
CARD = (250, 248, 242)


def aspect(spec: str) -> float:
    w, h = SIZES[spec.split(":", 1)[0]]
    return w / h


def _cache() -> Path:
    from ..core.paths import data_dir

    root = Path(os.environ.get("RUIBENCH_PRINTS") or data_dir("prints", "runs/cache/prints"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def _font(name: str, size: int):
    from PIL import ImageFont

    if name == "script":
        from .isaac.environment import fetched

        font = ImageFont.truetype(fetched("script_font"), size)
        font.set_variation_by_name("SemiBold")
        return font
    import matplotlib

    folder = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    return ImageFont.truetype(str(folder / f"{name}.ttf"), size)


def image(spec: str) -> str:
    """The path of ``spec``'s image, drawn the first time."""
    key = hashlib.sha256(f"{VERSION}|{spec}".encode()).hexdigest()[:16]
    design = spec.split(":", 1)[0]
    path = _cache() / f"{design}-{key}.png"
    if not path.exists():
        img = DESIGNS[design](spec.split(":", 1)[1])
        tmp = path.with_name(f".{path.name}.{os.getpid()}.part")
        img.save(tmp, format="PNG")
        os.replace(tmp, path)
    return str(path)


def _canvas(design: str, ground=CARD):
    from PIL import Image, ImageDraw

    w, h = SIZES[design]
    img = Image.new("RGB", (round(w * PX_PER_MM), round(h * PX_PER_MM)), ground)
    return img, ImageDraw.Draw(img)


def _mm(v: float) -> int:
    return round(v * PX_PER_MM)


def _formula_runs(formula: str):
    """(text, subscript?) runs: digits after a letter, and subscript digits,
    are subscripts."""
    runs = []
    for ch in formula:
        sub = ch in SUBSCRIPTS or (ch.isdigit() and runs and runs[-1][0][-1:].isalpha())
        ch = SUBSCRIPTS.get(ch, ch)
        if runs and runs[-1][1] == sub:
            runs[-1] = (runs[-1][0] + ch, sub)
        else:
            runs.append((ch, sub))
    return runs


def _label(formula: str):
    """A reagent card: a thin frame, the formula large, the name under it."""
    img, draw = _canvas("label")
    w, h = img.size
    inset, line = _mm(1.6), _mm(0.35)
    draw.rounded_rectangle((inset, inset, w - inset, h - inset), radius=_mm(1.8),
                           outline=(46, 72, 110), width=line)
    big, small = _font("DejaVuSans-Bold", _mm(17)), _font("DejaVuSans-Bold", _mm(10.5))
    runs = _formula_runs(formula)
    widths = [draw.textlength(t, font=small if sub else big) for t, sub in runs]
    x = (w - sum(widths)) / 2
    base = _mm(18.0)                                  # the formula's baseline
    for (t, sub), tw in zip(runs, widths):
        font = small if sub else big
        y = base + (_mm(3.2) if sub else 0)
        draw.text((x, y), t, font=font, fill=INK, anchor="ls")
        x += tw
    name = NAMES.get(formula, "")
    if name:
        font = _font("DejaVuSans", _mm(3.6))
        draw.text((w / 2, _mm(25.2)), name, font=font, fill=(70, 76, 88), anchor="ms")
    return img


def _letter(letter: str):
    """A beaker's label: a letter in a thin frame."""
    img, draw = _canvas("letter", (252, 252, 250))
    w, h = img.size
    inset = _mm(1.2)
    draw.rounded_rectangle((inset, inset, w - inset, h - inset), radius=_mm(2.0),
                           outline=(40, 44, 52), width=_mm(0.4))
    draw.text((w / 2, h / 2 + _mm(0.6)), letter, font=_font("DejaVuSans-Bold", _mm(16)), fill=INK,
              anchor="mm")
    return img


def _tag(name: str):
    """A gift tag: cream card, a deckled frame, the punched hole with its
    ring, "To" and the name in a script hand."""
    img, draw = _canvas("tag", (246, 238, 222))
    w, h = img.size
    gold = (176, 138, 64)
    inset = _mm(2.0)
    draw.rounded_rectangle((inset, inset, w - inset, h - inset), radius=_mm(2.5), outline=gold,
                           width=_mm(0.45))
    draw.rounded_rectangle((inset + _mm(1.0), inset + _mm(1.0), w - inset - _mm(1.0),
                            h - inset - _mm(1.0)), radius=_mm(2.0), outline=gold, width=_mm(0.15))
    cx, cy, r = _mm(7.5), h / 2, _mm(2.2)
    draw.ellipse((cx - r - _mm(0.9), cy - r - _mm(0.9), cx + r + _mm(0.9), cy + r + _mm(0.9)),
                 fill=(226, 206, 160))
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(120, 96, 60))
    left = _mm(13.0)
    draw.text((left, _mm(10.5)), "To", font=_font("DejaVuSerif-Italic", _mm(4.2)),
              fill=(110, 84, 48), anchor="ls")
    font = _font("script", _mm(13))
    room = w - left - _mm(6.0)
    size = _mm(13)
    while draw.textlength(name, font=font) > room and size > _mm(6):
        size -= _mm(0.5)
        font = _font("script", size)
    draw.text((left + room / 2, _mm(22.5)), name, font=font, fill=(96, 26, 36), anchor="ms")
    return img


def _sticker(name: str):
    """A white label under a present: a red frame, "To", the name in bold."""
    img, draw = _canvas("sticker", (250, 250, 247))
    w, h = img.size
    red = (178, 36, 48)
    inset = _mm(1.8)
    draw.rounded_rectangle((inset, inset, w - inset, h - inset), radius=_mm(3.0), outline=red,
                           width=_mm(0.8))
    draw.text((w / 2, _mm(9.5)), "To", font=_font("DejaVuSerif-Italic", _mm(5.2)), fill=red,
              anchor="ms")
    font, size = _font("DejaVuSans-Bold", _mm(15)), _mm(15)
    while draw.textlength(name, font=font) > w - _mm(10) and size > _mm(8):
        size -= _mm(0.5)
        font = _font("DejaVuSans-Bold", size)
    draw.text((w / 2, _mm(27.5)), name, font=font, fill=INK, anchor="ms")
    return img


#: wrapping papers: ground, and what is printed on it
WRAPS = {"red_dots": ((176, 30, 46), (250, 246, 240)),
         "blue_stars": ((30, 52, 112), (232, 190, 88)),
         "green_stripes": ((34, 104, 68), (246, 242, 230)),
         "kraft_hearts": ((198, 162, 118), (184, 38, 52)),
         "purple_snow": ((98, 60, 146), (244, 244, 252))}


def _star(draw, cx, cy, r, fill):
    import math

    pts = []
    for k in range(10):
        a = -math.pi / 2 + k * math.pi / 5
        rr = r if k % 2 == 0 else r * 0.42
        pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
    draw.polygon(pts, fill=fill)


def _heart(draw, cx, cy, r, fill):
    draw.ellipse((cx - r, cy - r * 0.9, cx + 0.05 * r, cy + 0.15 * r), fill=fill)
    draw.ellipse((cx - 0.05 * r, cy - r * 0.9, cx + r, cy + 0.15 * r), fill=fill)
    draw.polygon([(cx - r * 0.96, cy - r * 0.25), (cx + r * 0.96, cy - r * 0.25), (cx, cy + r * 1.05)],
                 fill=fill)


def _flake(draw, cx, cy, r, fill, width):
    import math

    for k in range(6):
        a = k * math.pi / 3
        ex, ey = cx + r * math.cos(a), cy + r * math.sin(a)
        draw.line((cx, cy, ex, ey), fill=fill, width=width)
        for t in (0.55,):
            mx, my = cx + t * r * math.cos(a), cy + t * r * math.sin(a)
            for da in (-0.6, 0.6):
                draw.line((mx, my, mx + 0.35 * r * math.cos(a + da), my + 0.35 * r * math.sin(a + da)),
                          fill=fill, width=width)


def _wrap(paper: str):
    """One face of a present in ``paper``: a repeat a few centimetres across."""
    ground, ink = WRAPS[paper]
    img, draw = _canvas("wrap", ground)
    w, h = img.size
    step = _mm(12)
    if paper == "green_stripes":
        for k in range(-w, 2 * w, _mm(8)):
            draw.line((k, 0, k + h, h), fill=ink, width=_mm(1.2))
            draw.line((k + _mm(2.2), 0, k + _mm(2.2) + h, h), fill=(214, 178, 92), width=_mm(0.4))
        return img
    for row, y in enumerate(range(-step, h + step, step)):
        for x in range(-step, w + step, step):
            cx = x + (step / 2 if row % 2 else 0)
            if paper == "red_dots":
                r = _mm(2.4)
                draw.ellipse((cx - r, y - r, cx + r, y + r), fill=ink)
            elif paper == "blue_stars":
                _star(draw, cx, y, _mm(3.4), ink)
            elif paper == "kraft_hearts":
                _heart(draw, cx, y, _mm(2.6), ink)
            elif paper == "purple_snow":
                _flake(draw, cx, y, _mm(3.6), ink, _mm(0.5))
    return img


#: tablecloths: (the coloured thread, the white); CLOTH_CHECKS along the
#: cloth's long side -- on DS2's 76 cm one, 2.7 cm checks, as a picnic
#: cloth's are
CLOTHS = {"gingham_red": ((190, 30, 38), (248, 246, 240))}
CLOTH_CHECKS = 28


def _cloth(cloth: str):
    """A cotton picnic cloth, gingham: stripes of coloured thread crossed with
    stripes of white, so white squares, pale ones where a coloured stripe
    crosses a white one and full ones where two coloured stripes cross --
    and the weave's grain over it, every other thread a shade darker."""
    import numpy as np
    from PIL import Image

    ink, ground = (np.array(c, float) for c in CLOTHS[cloth])
    w, h = (round(v * PX_PER_MM) for v in SIZES["cloth"])
    check = w / CLOTH_CHECKS
    across = (np.floor(np.arange(w) / check) % 2).astype(bool)[None, :]
    down = (np.floor(np.arange(h) / check) % 2).astype(bool)[:, None]
    both, one = (across & down)[..., None], (across ^ down)[..., None]
    img = np.where(both, ink, np.where(one, 0.5 * ink + 0.5 * ground, ground))
    thread = max(2, round(check / 14))
    weave = ((np.arange(w) // thread) % 2)[None, :] ^ ((np.arange(h) // thread) % 2)[:, None]
    img = img * (1.0 - 0.06 * weave[..., None])
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


#: a tea towel's stripes are
TOWELS = {"blue": (38, 78, 158), "red": (178, 40, 44), "green": (46, 122, 82)}
TOWEL_GROUND = (244, 240, 230)
#: drawn coarser than a label: it is 30 cm across, and nobody reads it
TOWEL_PX_PER_MM = 5
#: the stripes, from an edge inward: (where each begins, how wide), mm
TOWEL_STRIPES = ((22.0, 3.0), (29.0, 12.0), (45.0, 3.0))
TOWEL_HEM = 7.0


def _towel(colour: str):
    """A cotton tea towel, lying with its striped edges away from and toward
    whoever looks at it: unbleached ground, a broad stripe between two narrow
    ones woven in along each of those edges (they show on both sides, as woven
    stripes do), a hem turned and stitched all round, and the weave's grain
    over everything.  The image's x is the towel's own x (away from the
    robot), so the stripes are bands across the picture's width."""
    import numpy as np
    from PIL import Image

    ink, ground = np.array(TOWELS[colour], float), np.array(TOWEL_GROUND, float)
    w, h = (round(v * TOWEL_PX_PER_MM) for v in SIZES["towel"])
    mm = np.arange(w) / TOWEL_PX_PER_MM
    from_edge = np.minimum(mm, SIZES["towel"][0] - mm)
    striped = np.zeros(w, bool)
    for start, width in TOWEL_STRIPES:
        striped |= (from_edge >= start) & (from_edge < start + width)
    img = np.where(striped[None, :, None], ink, ground) * np.ones((h, w, 1))
    # the weave: every other thread a shade darker, both ways
    thread = max(2, round(0.5 * TOWEL_PX_PER_MM))
    weave = ((np.arange(w) // thread) % 2)[None, :] ^ ((np.arange(h) // thread) % 2)[:, None]
    img = img * (1.0 - 0.07 * weave[..., None])
    # the hem: a doubled edge, a little darker, and its line of stitches
    y_mm = np.arange(h) / TOWEL_PX_PER_MM
    edge = np.minimum(from_edge[None, :], np.minimum(y_mm, SIZES["towel"][1] - y_mm)[:, None])
    img = np.where((edge < TOWEL_HEM)[..., None], img * 0.94, img)
    stitch = (np.abs(edge - TOWEL_HEM) < 0.3)
    dashes = ((np.arange(w)[None, :] + np.arange(h)[:, None]) // round(2.4 * TOWEL_PX_PER_MM)) % 2 == 0
    img = np.where((stitch & dashes)[..., None], img * 0.80, img)
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


# -- a T-shirt ---------------------------------------------------------------------
#: a child's T-shirt lying flat, in millimetres: x from its hem (-130) to the
#: top of its shoulders, y across it (its left is +y).  One place for the cut:
#: the cloth is cut to :func:`tee_outline` (``tasks/impl/_garment.py``) and the
#: print is drawn to it
TEE = {
    "hem": -130.0, "top": 130.0,            # the hem; the shoulders at the neck
    "side": 100.0,                           # half the body's width
    "neck": 40.0, "scoop": 25.0,            # half the neck opening's width; how deep it dips in front
    "shoulder": 115.0,                       # where a shoulder ends, at the body's side (it slopes 15 mm)
    "armpit": 45.0,
    "cuff_top": (95.0, 165.0), "cuff_bottom": (35.0, 150.0),      # a sleeve's hem, (x, |y|)
}
#: the stripes' colour on an off-white knit
TEES = {"navy": (38, 54, 98), "red": (178, 46, 50), "green": (42, 112, 82)}
TEE_GROUND = (245, 242, 235)
TEE_PX_PER_MM = 5
#: a stripe and the white after it, mm; the neckband's width; how far in from
#: an edge a hem is stitched
TEE_STRIPE, TEE_BAND, TEE_STITCH = (6.0, 10.0), 9.0, 11.0
NECK_STEPS = 12


def tee_neck():
    """The front of the neck opening, right to left: (x, y) in mm."""
    import math

    return [(TEE["top"] - TEE["scoop"] * math.sin(math.pi * k / NECK_STEPS),
             -TEE["neck"] + 2 * TEE["neck"] * k / NECK_STEPS) for k in range(NECK_STEPS + 1)]


def tee_outline():
    """The T-shirt's edge as it lies flat, mm, going round from the right end
    of the hem: side, sleeve, shoulder, neck, and down the other side.  The
    middle of each sleeve's hem is a corner of it too (a hand takes it there)."""
    (cx, cy), (bx, by) = TEE["cuff_top"], TEE["cuff_bottom"]
    mid = ((cx + bx) / 2, (cy + by) / 2)
    right = [(TEE["hem"], -TEE["side"]), (TEE["armpit"], -TEE["side"]), (bx, -by), (mid[0], -mid[1]),
             (cx, -cy), (TEE["shoulder"], -TEE["side"])]
    left = [(x, -y) for x, y in reversed(right)]
    return right + tee_neck() + left


def _tee(colour: str):
    """A child's T-shirt of striped cotton jersey: stripes across the body
    and round the sleeves, a ribbed neckband in the stripes' colour, the hems
    turned and stitched twice, and the knit's fine ribs over everything.  The
    image is the box round the shirt as it lies flat -- its x the shirt's own
    (from the hem to the shoulders), its top the shirt's left sleeve -- and
    what is outside the shirt's edge is never seen."""
    import numpy as np
    from PIL import Image

    px = TEE_PX_PER_MM
    w, h = (round(v * px) for v in SIZES["tee"])
    x = (TEE["hem"] + (np.arange(w) + 0.5) / px)[None, :] * np.ones((h, 1))
    y = (SIZES["tee"][1] / 2 - (np.arange(h) + 0.5) / px)[:, None] * np.ones((1, w))
    ink, ground = np.array(TEES[colour], float), np.array(TEE_GROUND, float)

    # the stripes: across the body by how far up it; round a sleeve by how far along it
    (cx, cy), (bx, by) = TEE["cuff_top"], TEE["cuff_bottom"]
    arm_mid = ((TEE["shoulder"] + TEE["armpit"]) / 2, TEE["side"])
    cuff_mid = ((cx + bx) / 2, (cy + by) / 2)
    ax, ay = cuff_mid[0] - arm_mid[0], cuff_mid[1] - arm_mid[1]
    length = float(np.hypot(ax, ay))
    ax, ay = ax / length, ay / length
    along_sleeve = (x - arm_mid[0]) * ax + (np.abs(y) - arm_mid[1]) * ay
    period = sum(TEE_STRIPE)
    on_sleeve = np.abs(y) > TEE["side"]
    # the body's stripes counted from the hem; a sleeve's from its own hem inward
    phase = np.where(on_sleeve, (length - along_sleeve) + 4.0, x - TEE["hem"] + 4.0)
    striped = (phase % period) < TEE_STRIPE[0]
    img = np.where(striped[..., None], ink, ground)

    def distance(points):
        """How far each pixel is from a line through these points, mm."""
        best = np.full(x.shape, 1e9)
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            dx, dy = x1 - x0, y1 - y0
            t = np.clip(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy), 0.0, 1.0)
            best = np.minimum(best, np.hypot(x - (x0 + t * dx), y - (y0 + t * dy)))
        return best

    # the neckband: ribbing in the stripes' colour, along the front of the neck
    # and (the back of the neck, seen through the opening's edge) along the top
    neck = distance(tee_neck())
    band = neck < TEE_BAND
    ribs = (np.floor((y + 500.0) / 0.9) % 2) == 0
    img = np.where(band[..., None], ink * np.where(ribs, 1.0, 0.86)[..., None], img)
    seam = np.abs(neck - TEE_BAND) < 0.35                    # where the band is sewn on
    img = np.where(seam[..., None], img * 0.78, img)

    # the hems, turned and stitched twice: the body's, and each sleeve's
    hems = [[(TEE["hem"], -TEE["side"]), (TEE["hem"], TEE["side"])],
            [(cx, cy), (bx, by)], [(cx, -cy), (bx, -by)]]
    for line in hems:
        d = distance(line)
        img = np.where((d < TEE_STITCH)[..., None], img * 0.965, img)       # the doubled cloth, a shade darker
        for at in (TEE_STITCH, TEE_STITCH - 2.2):
            stitch = np.abs(d - at) < 0.3
            dashes = (np.floor((x * 0.8 + y) / 1.6) % 2) == 0
            img = np.where((stitch & dashes)[..., None], img * 0.72, img)
    # the shoulder seams and the seams the sleeves are set in by
    for sy in (-1.0, 1.0):
        for line in ([(TEE["top"], sy * TEE["neck"]), (TEE["shoulder"], sy * TEE["side"])],
                     [(TEE["shoulder"], sy * TEE["side"]), (TEE["armpit"], sy * TEE["side"])]):
            d = distance(line)
            img = np.where((d < 0.45)[..., None], img * 0.80, img)
            img = np.where((np.abs(d - 2.0) < 0.25)[..., None], img * 0.90, img)

    # the knit: fine ribs along the shirt's length, and its grain
    wale = (np.floor(y * px / 2.0) % 2) == 0
    img = img * np.where(wale, 1.0, 0.955)[..., None]
    rng = np.random.default_rng(7)
    img = img * (1.0 + 0.022 * rng.standard_normal((h, w, 1)))
    return Image.fromarray(np.clip(img, 0, 255).astype("uint8"))


def _blank(_: str):
    return _canvas("blank", (244, 243, 238))[0]


DESIGNS = {"label": _label, "letter": _letter, "tag": _tag, "blank": _blank,
           "sticker": _sticker, "wrap": _wrap, "cloth": _cloth, "towel": _towel, "tee": _tee}

# -- what CR4's computer parts show: drawn in their own module, named here
from . import prints_pc as _pc  # noqa: E402

SIZES.update(_pc.SIZES)
DESIGNS.update(_pc.DESIGNS)

__all__ = ["image", "aspect", "SIZES", "DESIGNS", "VERSION", "WRAPS", "CLOTHS", "TOWELS", "TEES",
           "TEE", "tee_outline", "tee_neck"]
