"""Flat garments for the folding task: their outlines and key points, the
folds a level asks for, where every point lies once the folds are made, how
far a cloth is from that, and the way a hand goes to make a fold.

* the goal (:func:`goal`): each fold mirrors what lies on its side of the
  crease;
* the score (:func:`fit`): how far the cloth is from the goal, the goal laid
  on it as well as turning, shifting and flipping can -- where on the table it
  ends is nobody's business, only its shape;
* the hand's way (:func:`fold_way`).

The cloth itself -- its points, the rule a hand takes it by, the fold that
stands in for physics -- is the engines' (``engines/cloth.py``).
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from ...engines.cloth import (CELL, MARK_EVERY, PINCH_ACROSS, PINCH_ALONG, ClothSpec, Line, Vec2,
                              fold_flat, inside, nearest)
from ...engines.prints import TEE, tee_outline


@dataclass(frozen=True)
class Fold:
    """One fold: its crease, the key points that go over it -- what the hands
    take -- and the key points those are laid on, one for one."""

    name: str
    crease: Line
    moved: Tuple[str, ...]
    onto: Tuple[str, ...]


@dataclass(frozen=True)
class Garment:
    name: str
    outline: Tuple[Vec2, ...]
    keypoints: Tuple[Tuple[str, Vec2], ...]
    folds: Tuple[Fold, ...]
    #: level -> the names of the folds it asks for, in order
    levels: Tuple[Tuple[str, Tuple[str, ...]], ...]
    cell: float = CELL

    def keypoint(self, name: str) -> Vec2:
        return dict(self.keypoints)[name]

    def fold(self, name: str) -> Fold:
        return next(f for f in self.folds if f.name == name)

    def asked(self, level: str) -> Tuple[Fold, ...]:
        return tuple(self.fold(n) for n in dict(self.levels)[level])

    def spec(self) -> ClothSpec:
        """What it is made of: its points, marks and key points."""
        return _spec(self)

    def marks(self) -> List[Vec2]:
        """Its marks as it lies flat, in its own frame."""
        spec = self.spec()
        return [spec.nodes[n] for n in spec.marks]


@functools.lru_cache(maxsize=None)
def _spec(g: Garment) -> ClothSpec:
    return cut(g.outline, g.keypoints, g.cell)


# -- a cloth cut to its outline ---------------------------------------------------------
#: a point this near the outline (in cells) is moved onto it; and no two points
#: are brought nearer to one another than this
ONTO, APART = 0.5, 0.45
#: a triangle smaller than this share of half a cell is left out
SLIVER = 0.15


def _foot(p: Vec2, a: Vec2, b: Vec2) -> Tuple[float, Vec2]:
    """(how far ``p`` is from the stretch a-b, the point of it nearest ``p``)."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    n = dx * dx + dy * dy
    s = 0.0 if n == 0.0 else min(1.0, max(0.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / n))
    q = (a[0] + s * dx, a[1] + s * dy)
    return math.dist(p, q), q


def to_outline(outline: Sequence[Vec2], p: Vec2) -> Tuple[float, Vec2]:
    """(how far ``p`` is from the outline, the point of the outline nearest it)."""
    n = len(outline)
    return min(_foot(p, outline[k], outline[(k + 1) % n]) for k in range(n))


def _area(a: Vec2, b: Vec2, c: Vec2) -> float:
    return ((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2


def cut(outline: Sequence[Vec2], keypoints: Sequence[Tuple[str, Vec2]] = (), cell: float = CELL,
        mark_every: int = MARK_EVERY, **material) -> ClothSpec:
    """A cloth cut to ``outline``: points a cell apart inside it, as
    ``engines/cloth.cloth_spec`` lays them, but its edge the outline's own --
    the points nearest the outline are moved onto it, each of its corners is
    a point, and no cell is left half out.  A grid alone gives a slanted edge
    steps of a cell, a neckline a staircase: a T-shirt cut with pinking
    shears.  Marks and key points as there."""
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    x0, y0 = min(xs), min(ys)
    nx, ny = round((max(xs) - x0) / cell), round((max(ys) - y0) / cell)
    at: dict = {}

    def near(place, q) -> bool:
        i, j = place
        return any(math.dist(at[(i + di, j + dj)], q) < APART * cell
                   for di in (-1, 0, 1) for dj in (-1, 0, 1)
                   if (di or dj) and (i + di, j + dj) in at)

    # the outline's own corners first: each takes the grid place it is nearest
    for v in outline:
        place = (round((v[0] - x0) / cell), round((v[1] - y0) / cell))
        if place not in at and not near(place, v):
            at[place] = (float(v[0]), float(v[1]))
    fixed = set(at)
    # then what is well inside, where the grid has it
    edge_in, edge_out = [], []
    for i in range(nx + 1):
        for j in range(ny + 1):
            if (i, j) in fixed:
                continue
            p = (x0 + i * cell, y0 + j * cell)
            d, q = to_outline(outline, p)
            if inside(outline, p):
                if d <= ONTO * cell:
                    edge_in.append(((i, j), p, q))
                else:
                    at[(i, j)] = p
            elif d < ONTO * cell:
                edge_out.append(((i, j), q))
    # then the points just inside the outline, moved onto it where that
    # crowds nothing; then those just outside it, where there is still room
    for place, p, q in edge_in:
        at[place] = p if near(place, q) else q
    for place, q in edge_out:
        if not near(place, q):
            at[place] = q
    # two triangles a cell, cut the way that leaves neither a sliver
    least = SLIVER * cell * cell / 2
    triangles = []

    def good(a, b, c) -> bool:
        pa, pb, pc = at[a], at[b], at[c]
        if _area(pa, pb, pc) < least:
            return False
        middle = ((pa[0] + pb[0] + pc[0]) / 3, (pa[1] + pb[1] + pc[1]) / 3)
        return inside(outline, middle, eps=1e-6)

    for i in range(nx):
        for j in range(ny):
            a, b, c, d = (i, j), (i + 1, j), (i + 1, j + 1), (i, j + 1)
            have = [k for k in (a, b, c, d) if k in at]
            if len(have) == 4:
                one = [tri for tri in ((a, b, c), (a, c, d)) if good(*tri)]
                other = [tri for tri in ((a, b, d), (b, c, d)) if good(*tri)]
                smallest = lambda tris: min((_area(*(at[k] for k in tri)) for tri in tris), default=0.0)
                pick = one if (len(one), smallest(one)) >= (len(other), smallest(other)) else other
                triangles += pick
            elif len(have) == 3 and good(*have):
                triangles.append(tuple(have))
    used = sorted({k for tri in triangles for k in tri})
    number = {k: n for n, k in enumerate(used)}
    nodes = [at[k] for k in used]
    keys = tuple((name, nearest(nodes, p)) for name, p in keypoints)
    marks = sorted({number[(i, j)] for (i, j) in used if i % mark_every == 0 and j % mark_every == 0}
                   | {n for _, n in keys})
    return ClothSpec(tuple(nodes), tuple(tuple(number[k] for k in tri) for tri in triangles),
                     tuple(marks), keys, cell=cell, **material)


# -- the garments ---------------------------------------------------------------------
def towel(size: float = 0.24) -> Garment:
    """A square tea towel.  Folded in half -- its far edge onto its near edge
    -- and in half again the same way.  24 cm: an X5's hand pointing down gets
    to the table no farther out than 48 cm, and 4 cm above it no farther than
    45 (10/5); a towel of 30 cm, turned a few degrees, has its far corners
    past that."""
    h = size / 2
    return Garment(
        name="towel",
        outline=((-h, -h), (h, -h), (h, h), (-h, h)),
        keypoints=(("near_right", (-h, -h)), ("far_right", (h, -h)),
                   ("far_left", (h, h)), ("near_left", (-h, h)),
                   ("mid_right", (0.0, -h)), ("mid_left", (0.0, h))),
        folds=(Fold("in_half", ((0.0, -h), (0.0, h)), ("far_left", "far_right"),
                    ("near_left", "near_right")),
               Fold("in_half_again", ((-h / 2, -h), (-h / 2, h)), ("mid_left", "mid_right"),
                    ("near_left", "near_right"))),
        levels=(("easy", ("in_half",)), ("hard", ("in_half", "in_half_again"))),
    )


def tshirt() -> Garment:
    """A child's T-shirt lying flat, its collar away from the robot, cut as
    ``engines/prints.TEE`` has it (the print is drawn to the same cut): a
    body 20 cm wide and 26 long, a scooped neck, shoulders that slope, short
    sleeves set in and slanting down to 33 cm across.  Folded: each sleeve in
    over the front, then the hem up to the shoulders -- taken by its two
    corners: two X5 hands pointing down side by side cannot come nearer than
    17 cm to one another (each is 17.4 cm long the way its fingers travel), so
    a hem 20 cm wide cannot be taken nearer its middle (tried, Isaac, 10/5: the
    hands stopped against one another 2.5 cm short of where they were sent)."""
    mm = 0.001
    outline = tuple((x * mm, y * mm) for x, y in tee_outline())
    hem, side, shoulder, armpit = (TEE[k] * mm for k in ("hem", "side", "shoulder", "armpit"))
    (cx, cy), (bx, by) = TEE["cuff_top"], TEE["cuff_bottom"]
    cuff = ((cx + bx) / 2 * mm, (cy + by) / 2 * mm)            # the middle of a sleeve's hem
    chest = (cuff[0], 2 * side - cuff[1])                      # where it comes to lie on the front
    fold = (hem + shoulder) / 2                                # the hem's corners go to the shoulders' ends
    return Garment(
        name="tshirt",
        outline=outline,
        keypoints=(("hem_right", (hem, -side)), ("hem_left", (hem, side)),
                   ("shoulder_right", (shoulder, -side)), ("shoulder_left", (shoulder, side)),
                   ("sleeve_right", (cuff[0], -cuff[1])), ("sleeve_left", cuff),
                   ("chest_right", (chest[0], -chest[1])), ("chest_left", chest)),
        folds=(Fold("sleeve_left_in", ((armpit, side), (shoulder, side)), ("sleeve_left",), ("chest_left",)),
               Fold("sleeve_right_in", ((armpit, -side), (shoulder, -side)), ("sleeve_right",),
                    ("chest_right",)),
               Fold("hem_up", ((fold, -side), (fold, side)), ("hem_left", "hem_right"),
                    ("shoulder_left", "shoulder_right"))),
        levels=(("easy", ("hem_up",)),
                ("hard", ("sleeve_left_in", "sleeve_right_in", "hem_up"))),
    )


GARMENTS = {"towel": towel, "tshirt": tshirt}


# -- folded, and how far from it ---------------------------------------------------
def goal(g: Garment, level: str, flat: Optional[Sequence[Vec2]] = None) -> Tuple[List[Vec2], List[int]]:
    """Where the cloth's points lie, and in which layer, once the level's
    folds are made.  ``flat``: the points of the cloth lying flat, in the
    garment's frame (its marks by default)."""
    flat = list(flat if flat is not None else g.marks())
    pts, layers = list(flat), [0] * len(flat)
    for f in g.asked(level):
        # the side that goes over is the one the fold's first key point lies on by now
        taken = pts[nearest(flat, g.keypoint(f.moved[0]))]
        pts, layers = fold_flat(pts, layers, f.crease, taken)
    return pts, layers


def fit(goal_points: Sequence[Vec2], actual: Sequence[Vec2]) -> Tuple[float, float]:
    """(mean, worst) distance between the cloth's points and the goal's, the
    goal laid on the cloth as well as turning, shifting and flipping it can.
    Flipped counts too: a towel folded far edge to near and one folded near
    edge to far are mirror images of one another, point for point, and
    either is the towel folded in half."""
    if not goal_points or len(goal_points) != len(actual):
        raise ValueError("as many goal points as cloth points, and at least one")
    return min(_fit(goal_points, actual), _fit([(-x, y) for x, y in goal_points], actual))


def _fit(goal_points: Sequence[Vec2], actual: Sequence[Vec2]) -> Tuple[float, float]:
    n = len(goal_points)
    gx = sum(p[0] for p in goal_points) / n
    gy = sum(p[1] for p in goal_points) / n
    ax = sum(p[0] for p in actual) / n
    ay = sum(p[1] for p in actual) / n
    s = c = 0.0
    for (px, py), (qx, qy) in zip(goal_points, actual):
        px, py, qx, qy = px - gx, py - gy, qx - ax, qy - ay
        c += px * qx + py * qy
        s += px * qy - py * qx
    turn = math.atan2(s, c)
    ct, st = math.cos(turn), math.sin(turn)
    total = worst = 0.0
    for (px, py), (qx, qy) in zip(goal_points, actual):
        px, py = px - gx, py - gy
        d = math.hypot(ct * px - st * py + ax - qx, st * px + ct * py + ay - qy)
        total += d
        worst = max(worst, d)
    return total / n, worst


# -- how a hand carries a fold over ------------------------------------------------
#: heights above the table of the hand's tool point: where it pinches (its
#: fingertips, 8 mm lower, just over the cloth), where it lets go, and where it
#: waits above before and after
PINCH, LAY, HOVER = 0.014, 0.022, 0.04
#: the hand takes the cloth this far in from the point it is to carry
INSET = 0.004
#: the way over is half a circle where the hand can be held that high, and no
#: higher than this anywhere (an X5's hand points down 10 cm over the table no
#: farther out than 39 cm: 10/5)
PEAK = 0.10
#: the points of the way, as angles round the crease
ROUND = (30.0, 60.0, 90.0, 120.0, 150.0)
#: where there is physics, the hand lets go this far short of where the point
#: is to lie -- and how far depends on which way its fingers open.  The bit of
#: cloth in them is left doubled back on the fold, and has to turn over.
#: Along the fold, the finger that opens ahead turns it over as it goes: the
#: hand stops where the cloth is not stretched.  (A sleeve let go 6 mm short
#: drew its shoulder 15 mm in after it; 18 mm short the shoulder stayed, and
#: the sleeve's end lay 1-3 mm from the chest point.)  Across the fold nothing
#: turns it over but the cloth's own spring, and that takes a cloth drawn
#: taut: a hem let go 18 mm short stayed doubled back, 25 mm short of the
#: shoulders; 14 mm short it turned over only as the hands left; 10 mm short,
#: by itself in half a second, and lay 4-7 mm past them (Isaac, 10/5)
SHORT_ALONG, SHORT_ACROSS = 0.018, 0.010


def _along_across(u: Vec2, travel: Vec2) -> Tuple[float, float]:
    return (abs(u[0] * travel[0] + u[1] * travel[1]), abs(u[0] * travel[1] - u[1] * travel[0]))


def short(u: Vec2, travel: Vec2 = (0.0, 1.0)) -> float:
    """How far short of its place a point is let go where there is physics,
    carried along ``u`` (a unit vector on the table) by a hand whose fingers
    travel along ``travel``."""
    along, across = _along_across(u, travel)
    return along * along * SHORT_ALONG + across * across * SHORT_ACROSS


def reach(u: Vec2, travel: Vec2 = (0.0, 1.0)) -> float:
    """How far the patch a closing hand takes reaches from the tool point
    along ``u`` (a unit vector on the table), the fingers travelling along
    ``travel``: the patch is wider across the fingers' travel than along it."""
    along, across = _along_across(u, travel)
    return across * PINCH_ACROSS + along * PINCH_ALONG


def one_move(ways: Sequence[Tuple[Vec2, Vec2]]) -> List[Tuple[Vec2, Vec2]]:
    """The ways of the hands that carry one edge -- each ``(from, to)`` --
    made one move: every point goes as far and the same way as they were to go
    on average, so the edge between the hands stays as long as it is.  (A
    T-shirt's shoulders, its sleeves folded in, lie 3 cm nearer one another
    than its hem is wide: hands sent each to its own shoulder gathered the
    hem between them, and it lay in a wave: Isaac, 10/5.)"""
    n = len(ways)
    dx = sum(b[0] - a[0] for a, b in ways) / n
    dy = sum(b[1] - a[1] for a, b in ways) / n
    return [(a, (a[0] + dx, a[1] + dy)) for a, _ in ways]


def fold_way(taken: Vec2, onto: Vec2, inset: float = INSET, travel: Vec2 = (0.0, 1.0),
             turned: bool = True, lay: float = LAY):
    """A hand's way with a point of the cloth that is to be laid on another:
    ``(pinch, over)`` -- where the tool point comes down to pinch, and the
    points it then goes through, the last of them where it lets go; all
    ``(x, y, height above the table)``.  ``lay``: how high it lets go -- as
    much higher than ``LAY`` as the cloth lies high there already.

    The hand pinches ``inset`` in from the point, toward where it is going.
    Where it lets go depends on what the bit of cloth in its fingers does.
    ``turned`` (no physics): it turns over with the rest -- the cloth folds
    along the perpendicular bisector of the way the patch's middle went
    (``engines/cloth.fold_by``) -- and the hand stops :func:`reach` short of
    where the point is to lie.  Not ``turned`` (Isaac): the hand does not
    turn over as it carries, so the bit in its fingers stays the way up it
    was and is left doubled back on the fold, a centimetre of it, to turn
    over once it is let go: the hand stops :func:`short` of the place.

    The way over goes round the crease: half a circle, flattened where the
    hand cannot be held as high as the fold is deep.  Never outside the
    circle: a hand that lifts the edge straight up first is farther from the
    crease than the cloth between them is long, and pulls the half that is
    to stay along with it -- of a towel already folded once, the upper layer
    of that half came 2 cm after the hand (Isaac, 10/5)."""
    dx, dy = onto[0] - taken[0], onto[1] - taken[1]
    far = math.hypot(dx, dy)
    if far < 1e-6:
        raise ValueError("a point cannot be folded onto itself")
    ux, uy = dx / far, dy / far
    back = reach((ux, uy), travel) if turned else short((ux, uy), travel)
    a = (taken[0] + inset * ux, taken[1] + inset * uy)
    b = (onto[0] - back * ux, onto[1] - back * uy)
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    half = math.dist(a, b) / 2
    peak = min(half, PEAK)
    over = []
    for degrees in ROUND:
        c, s = math.cos(math.radians(degrees)), math.sin(math.radians(degrees))
        z = max(peak * s, lay)
        over.append((mid[0] - half * c * ux, mid[1] - half * c * uy, z))
    over.append((b[0], b[1], lay))
    return (a[0], a[1], PINCH), over
