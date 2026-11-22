"""CL5 "tangram" -- In-Context Learning: what to do is in a picture.

Knob: the card (the silhouette with each piece's outline drawn in -> the
silhouette alone).

A picture card on the table with a black silhouette on it, and the seven
pieces of a wooden tangram -- two large triangles, a medium one, two small
ones, a square and a parallelogram -- lying either side of it, each with a
little knob in its middle, as toddlers' knob puzzles have.  Before the
episode, one picture: another card, finished -- its silhouette (another
figure) filled with the seven pieces.  "Finish your card the way I finished
mine."  What there is to do -- all seven pieces laid on the silhouette,
filling it exactly -- only the picture shows; how they go on this card's
figure is the agent's own puzzle.

v6 (2026-09-22): a picture, where there was a film of the seven pieces going
down one by one (the advisor: the finished card is enough, no demo needed).

The tangram is two centuries old (Slocum, *The Tangram Book*, 2003) and a
staple of early geometry teaching; Clements, Wilson & Sarama (2004) traced
how young children learn to compose shapes on exactly these puzzle cards --
first ones with every piece's outline drawn in, later the bare silhouette.
That is the knob.

Three silhouettes -- a cat, a fish, a boat -- each laid out piece by piece on
the tangram grid and checked there: no two pieces overlap, all sixteen units
are covered, and the parallelogram is never turned over (a knob puzzle's
pieces cannot be).  The film plays one, the episode another; the seed picks.

v4 (2026-09-22), replacing "the puzzle box" (Horner & Whiten's over-imitation
box): the advisor found it not exciting enough.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import quat_mul, yaw_of
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import PlacePosePolicy
from ...scoring import Custom, Finally, Milestone, Scorer, Upright
from ..instance import Demonstration, TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

# -- the pieces, on the tangram grid (units of U) --------------------------------------
#: one unit: the whole set makes a 12 cm square
U = 0.03
#: each kind in the standard square [0, 4]^2, counter-clockwise
CANON = {"large": ((0, 0), (4, 0), (2, 2)), "medium": ((4, 2), (4, 4), (2, 4)),
         "small": ((2, 2), (3, 1), (3, 3)), "square": ((2, 2), (3, 3), (2, 4), (1, 3)),
         "para": ((3, 1), (4, 0), (4, 2), (3, 3))}
#: the seven pieces: name, kind, paint
PIECES = (("large_red", "large", "paint:red"), ("large_blue", "large", "paint:blue"),
          ("medium_yellow", "medium", "paint:yellow"), ("small_green", "small", "paint:green"),
          ("small_purple", "small", "paint:purple"), ("square_orange", "square", "paint:orange"),
          ("para_teal", "para", "paint:teal"))
#: turning a piece by this much leaves it looking the same
SYMMETRY = {"large": 2 * math.pi, "medium": 2 * math.pi, "small": 2 * math.pi,
            "square": math.pi / 2, "para": math.pi}

#: the silhouettes, piece by piece (research/tangram_draw.py checked each)
SHAPES = {
    "cat": (("large", ((0, 0), (4, 0), (2, 2))), ("large", ((0, 0), (2, 2), (0, 4))),
            ("medium", ((2, 2), (4, 0), (4, 2))), ("square", ((0, 4), (1, 5), (0, 6), (-1, 5))),
            ("small", ((-1, 5), (0, 6), (-1, 7))), ("small", ((0, 6), (1, 5), (1, 7))),
            ("para", ((4, 1), (5, 0), (5, 2), (4, 3)))),
    "fish": (("large", ((0, 2), (4, 2), (2, 4))), ("large", ((0, 2), (2, 0), (4, 2))),
             ("medium", ((4, 2), (6, 2), (6, 4))), ("para", ((3, 1), (5, 1), (6, 2), (4, 2))),
             ("square", ((0, 2), (-1, 3), (-2, 2), (-1, 1))),
             ("small", ((1, 3), (2, 4), (0, 4))), ("small", ((1, 1), (0, 0), (2, 0)))),
    "boat": (("large", ((2, 2), (-2, 2), (0, 0))), ("large", ((0, 0), (4, 0), (2, 2))),
             ("medium", ((0, 2), (2, 2), (2, 4))), ("square", ((3, 1), (4, 2), (3, 3), (2, 2))),
             ("small", ((3, 3), (2, 4), (2, 2))), ("small", ((4, 0), (5, 1), (3, 1))),
             ("para", ((5, 1), (6, 2), (4, 2), (3, 1)))),
}

# -- the table -------------------------------------------------------------------------
CARD_AT, CARD_HALF, CARD_HZ = (0.33, 0.0), (0.15, 0.15), 0.002
CARD_TOP = TABLE + 2 * CARD_HZ
#: a piece's plate: 1 cm thick, 8 % smaller than its place on the card, so
#: that seven laid side by side leave each other 4 to 7 mm.  At 3 and 5 % a
#: piece set down a millimetre or two off shoved its neighbour (Isaac, 9/22)
PLATE_HZ, SHRINK = 0.005, 0.92
KNOB_R, KNOB_HZ = 0.008, 0.0125
#: the square post inside the knob the fingers close on (the knob itself
#: does not collide)
POST_HALF = 0.8 * KNOB_R
#: taken by the knob, 6 mm below its top: the fingertips then pass 6 mm over
#: the pieces already down
GRASP_H = PLATE_HZ + 2 * KNOB_HZ - 0.006
#: where the pieces start: four places down each side of the card, 13 cm
#: apart (a large triangle reaches 6.3 cm from its middle)
START_X, START_Y = (0.15, 0.28, 0.41, 0.54), 0.23
#: the easy card draws each piece's outline: its shape this much smaller
OUTLINED = 0.90
#: placed: a piece of the kind within this of a place, turned within this
#: (v2.3.4; 1.2 cm and 12 degrees before, which failed a piece turned 12.0
#: degrees and one 1.3 cm off, the silhouette otherwise filled)
NEAR, TURNED = 0.015, math.radians(15)
#: a piece in its place lies flat, face up: within this of the vertical (v2.3)
FLAT = math.radians(10)
FRAME = Frame((0.07, 0.61), (-0.30, 0.30), 0.05)
#: big pieces first: the last ones go into the gaps the others leave
ORDER = ("large", "medium", "para", "square", "small")
#: ...but after the two large triangles, an order whose open fingers stay
#: this far from every knob already down, where there is one (``fill_order``)
ORDER_CLEAR = 0.010


def _turn(p, k):
    x, y = p
    for _ in range(k % 4):
        x, y = -y, x
    return (x, y)


def _centroid(poly) -> Tuple[float, float]:
    return (sum(p[0] for p in poly) / len(poly), sum(p[1] for p in poly) / len(poly))


def local(kind: str) -> Tuple[Tuple[float, float], ...]:
    """A kind's corners about its middle, in metres, as it lies at heading 0."""
    cx, cy = _centroid(CANON[kind])
    return tuple(((x - cx) * U, (y - cy) * U) for x, y in CANON[kind])


def places(shape: str, polys=None) -> List[Tuple[str, Tuple[float, float], float]]:
    """The silhouette's seven places: (kind, middle on the table, heading) --
    of its tiling in ``SHAPES``, or of ``polys``, another tiling of it.

    The grid's up is away from the fixed camera's bottom edge -- toward the
    robot -- so the picture stands upright in it: grid x is the table's +y,
    grid y its -x.  The silhouette is centred on the card."""
    frame = SHAPES[shape]
    polys = frame if polys is None else polys
    xs = [p[0] for _, poly in frame for p in poly]
    ys = [p[1] for _, poly in frame for p in poly]
    mx, my = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    cx, cy = ahead(*CARD_AT)
    out = []
    for kind, poly in polys:
        gx, gy = _centroid(poly)
        rel = sorted((round(x - gx, 6), round(y - gy, 6)) for x, y in poly)
        c0x, c0y = _centroid(CANON[kind])
        base = [(x - c0x, y - c0y) for x, y in CANON[kind]]
        k = next(k for k in range(4)
                 if sorted((round(a, 6), round(b, 6)) for a, b in (_turn(p, k) for p in base)) == rel)
        # grid heading k quarter turns; the grid itself is turned a quarter
        # turn onto the table
        out.append((kind, (cx - (gy - my) * U, cy + (gx - mx) * U), (k + 1) * math.pi / 2))
    return out


def _flat(yaw: float):
    """A prism (a triangle in its x-z plane) laid flat, its apex along +y
    turned by ``yaw``."""
    lie = (math.cos(-math.pi / 4), math.sin(-math.pi / 4), 0.0, 0.0)
    return quat_mul((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), lie)


def triangle_part(corners, z: float, hz: float, color, collides=True):
    """A right isosceles triangle (corners counter-clockwise, right angle
    anywhere) as a flat prism ``2 hz`` thick at height ``z``."""
    n = len(corners)
    # the right angle is the corner opposite the longest side
    k = max(range(n), key=lambda i: math.dist(corners[(i + 1) % n], corners[(i + 2) % n]))
    apex, a, b = corners[k], corners[(k + 1) % n], corners[(k + 2) % n]
    mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
    d = (apex[0] - mid[0], apex[1] - mid[1])
    h = math.hypot(*d)
    centre = (mid[0] + d[0] / 2, mid[1] + d[1] / 2, z)
    return P.prism(centre, (math.dist(a, b) / 2, hz, h / 2), color,
                   rot=_flat(math.atan2(d[1], d[0]) - math.pi / 2), collides=collides)


def shape_parts(kind: str, corners, z: float, hz: float, color, collides=True) -> List:
    """A piece's shape from its corners (counter-clockwise, about any point)."""
    if kind == "square":
        cx, cy = _centroid(corners)
        side = math.dist(corners[0], corners[1])
        edge = math.atan2(corners[1][1] - corners[0][1], corners[1][0] - corners[0][0])
        return [P.box((cx, cy, z), (side / 2, side / 2, hz), color,
                      rot=(math.cos(edge / 2), 0.0, 0.0, math.sin(edge / 2)), collides=collides)]
    if kind == "para":
        # two small triangles, cut along the short diagonal
        a, b, c, d = corners
        cut = (a, c) if math.dist(a, c) < math.dist(b, d) else (b, d)
        if cut == (a, c):
            halves = ((a, b, c), (a, c, d))
        else:
            halves = ((b, c, d), (b, d, a))
        return [triangle_part(h, z, hz, color, collides) for h in halves]
    return [triangle_part(corners, z, hz, color, collides)]


def turned(corners, yaw: float, scale: float = 1.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return tuple((scale * (c * x - s * y), scale * (s * x + c * y)) for x, y in corners)


def piece(name: str, kind: str, color, xy, yaw: float) -> BodySpec:
    """A wooden piece with a knob in its middle: its frame at the plate's
    middle, turned ``yaw``."""
    corners = turned(local(kind), 0.0, SHRINK)
    parts = shape_parts(kind, corners, 0.0, PLATE_HZ, color)
    # the knob looks round; the fingers close on a square post inside it,
    # across two of its faces -- on a round one the piece swivelled in the
    # hand as soon as it touched a neighbour (Isaac, 9/22)
    knob = (0.0, 0.0, PLATE_HZ + KNOB_HZ)
    parts += [P.grip(knob, (POST_HALF, POST_HALF, KNOB_HZ)),
              P.cylinder(knob, KNOB_R, KNOB_HZ, "wood:ash", collides=False)]
    hx = max(abs(x) for x, _ in corners)
    hy = max(abs(y) for _, y in corners)
    return BodySpec(name, "compound", (hx, hy, PLATE_HZ),
                    Pose((xy[0], xy[1], TABLE + PLATE_HZ),
                         (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
                    category=f"tangram {kind}", color=color, mass=0.05,
                    attrs={"parts": tuple(parts), "footprint": "rect",
                           "grasp_height": GRASP_H, "grasp_yaw": 0.0,
                           # what the fingers close on: the knob, above the plate
                           "handle": ((0.0, 0.0, PLATE_HZ + KNOB_HZ), (KNOB_R, KNOB_R, KNOB_HZ))})


def card(shape: str, outlined: bool) -> List[BodySpec]:
    """The picture card, and the silhouette printed on it -- each piece's
    outline drawn in (the gaps between slightly smaller shapes) on the easy
    one.  Each printed shape is a body of its own, in a fixed order of kinds:
    only their places change with the silhouette, so every seed builds the
    same scene."""
    cx, cy = ahead(*CARD_AT)
    out = [BodySpec("card", "box", (*CARD_HALF, CARD_HZ), Pose((cx, cy, TABLE + CARD_HZ)),
                    category="picture card", color="paint:cream", fixed=True,
                    attrs={"graspable": False, "footprint": "rect"})]
    scale = OUTLINED if outlined else 1.012
    for k, (kind, (x, y), yaw) in enumerate(sorted(places(shape),
                                                   key=lambda p: ORDER.index(p[0]))):
        parts = shape_parts(kind, turned(local(kind), 0.0, scale), 0.0, 0.0003, "paint:dark",
                            collides=False)
        out.append(BodySpec(f"print{k + 1}", "compound", (0.02, 0.02, 0.0003),
                            Pose((x, y, TABLE + 2 * CARD_HZ + 0.0003),
                                 (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
                            category="printed silhouette", color="paint:dark", fixed=True,
                            attrs={"parts": tuple(parts), "graspable": False, "collides": False,
                                   "footprint": "rect"}))
    return out


def shapes_for(seed: int) -> Tuple[str, str]:
    """(the episode's silhouette, the film's) -- never the same."""
    names = sorted(SHAPES)
    rng = SeedBundle(seed).stream("tangram")
    mine = rng.choice(names)
    return mine, rng.choice([n for n in names if n != mine])


def start(seed: int, film: bool) -> Dict[str, Tuple[Tuple[float, float], float]]:
    """Where each piece starts and how it is turned."""
    rng = SeedBundle(seed).stream("tangram/film" if film else "tangram/start")
    spots = [(x, s * START_Y) for x in START_X for s in (1, -1)]
    rng.shuffle(spots)
    return {name: (ahead(*spots[k]), rng.uniform(-math.pi, math.pi))
            for k, (name, _, _) in enumerate(PIECES)}


def solved(shape: str) -> Dict[str, Tuple[Tuple[float, float], float]]:
    """Each piece in a place of its kind on ``shape``'s silhouette."""
    free = list(places(shape))
    out = {}
    for name, kind, _ in PIECES:
        k = next(i for i, (kd, _, _) in enumerate(free) if kd == kind)
        _, xy, yaw = free.pop(k)
        out[name] = (xy, yaw)
    return out


def scene_for(level: str, seed: int, film: bool):
    """The episode's card, the pieces beside it -- or (``film``) the
    picture's: the other figure's card, finished."""
    mine, films = shapes_for(seed)
    bodies = card(films if film else mine, outlined=level == "easy")
    where = solved(films) if film else start(seed, False)
    for name, kind, color in PIECES:
        xy, yaw = where[name]
        body = piece(name, kind, color, xy, yaw)
        if film:
            # lying on the card, not on the table
            body = replace(body, pose=Pose((xy[0], xy[1], CARD_TOP + PLATE_HZ), body.pose.quat))
        bodies.append(body)
    return station_scene(f"cl5_tangram_{level}", bodies, frame=FRAME)


def tilings(shape: str) -> List[list]:
    """The tilings of a silhouette the seven pieces fill it with: the one in
    ``SHAPES``, and each another split of a square two congruent right
    triangles make sharing their long side -- across its other diagonal.  The
    fish's two large triangles make such a square: laid across the other
    diagonal they fill the silhouette as well (v2.3.4: three of Claude Code's
    six hard-level fish did so, and counted as unfinished)."""
    polys = [(kind, [tuple(p) for p in poly]) for kind, poly in SHAPES[shape]]
    out = [polys]
    for i, (ka, a) in enumerate(polys):
        for j in range(i + 1, len(polys)):
            kb, b = polys[j]
            shared = set(a) & set(b)
            if ka != kb or len(a) != 3 or len(b) != 3 or len(shared) != 2:
                continue
            p, q = sorted(shared)
            if math.dist(p, q) < max(math.dist(u, v) for u in a for v in a) - 1e-9:
                continue                               # not their long side
            r, t = (set(a) - shared).pop(), (set(b) - shared).pop()
            other = list(polys)
            other[i], other[j] = (ka, [r, t, p]), (kb, [r, t, q])
            out.append(other)
    return out


def matching(w: WorldState, shape: str) -> List[Optional[str]]:
    """For each place of the silhouette's tiling that the pieces fill best
    (``tilings``), the piece lying in it, or None (``matching_places``)."""
    return max((matching_places(w, places(shape, polys)) for polys in tilings(shape)),
               key=lambda m: sum(1 for x in m if x))


def matching_places(w: WorldState, where) -> List[Optional[str]]:
    """For each place, the piece lying in it (or None): of the place's kind,
    within ``NEAR`` of its middle and ``TURNED`` of its heading, lying flat face
    up (``FLAT``: not turned over -- the parallelogram's mirror image is not the
    piece -- nor propped on another), not held -- each piece counted for one
    place only."""
    out: List[Optional[str]] = []
    used = set()
    for kind, xy, yaw in where:
        found = None
        for name, k, _ in PIECES:
            o = w.obj(name)
            if k != kind or name in used or o.is_held:
                continue
            d = (yaw_of(o.pose.quat) - yaw) % SYMMETRY[kind]
            if (math.dist(o.pose.pos[:2], xy) < NEAR and min(d, SYMMETRY[kind] - d) < TURNED
                    and o.pose.pos[2] < CARD_TOP + 0.03 and Upright(name, FLAT).holds(w)):
                found = name
                break
        if found:
            used.add(found)
        out.append(found)
    return out


def filled(w: WorldState, shape: str) -> int:
    return sum(1 for m in matching(w, shape) if m)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CL5")
    mine, films = shapes_for(seed)
    film = Demonstration(scene=scene_for(level, seed, film=True),
                         demonstrator=lambda s=films: TangramPolicy(s, film=True),
                         caption="My card, finished.", picture=True)
    done = Custom(lambda w: filled(w, mine) == len(PIECES), "the silhouette filled")
    milestones = (
        Milestone("a_piece_placed", Custom(lambda w: filled(w, mine) >= 1, "a piece in its place")),
        Milestone("half_placed", Custom(lambda w: filled(w, mine) >= 4, "four pieces in place")),
        Milestone("filled", done),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene_for(level, seed, film=False),
        instruction="Finish your card the way I finished mine.",
        scorer=Scorer(task_id="CL5", level=level, success=Finally(done), milestones=milestones),
        milestones=milestones,
        demonstrations=(film,),
        reference_factory=lambda inst, s=mine: TangramPolicy(s, film=False),
        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level],
                  # hidden from the agent: the goal (the gap-filled control)
                  "goal": f"lay all seven pieces on the silhouette ({mine}), filling it",
                  "max_model_calls": 90},
    )


#: the open hand around a knob: a finger 1 cm either side of the fingers'
#: line, reaching 6.6 cm out from the knob along it (the FR3's)
FINGER_HALF, FINGER_REACH = 0.010, 0.066


def finger_clearance(at, yaw: float, knobs) -> float:
    """How far the open fingers of a hand over ``at``, turned ``yaw``, stay
    from the nearest of ``knobs`` (negative: they would hit it).  A knob
    between the open fingers is in their way too: they sweep across it
    closing on the post and opening again -- counted as clear, it put every
    boat's fingers through a placed square's knob (Isaac, v13)."""
    ux, uy = -math.sin(yaw), math.cos(yaw)          # the fingers' line (tool y)
    best = math.inf
    for kx, ky in knobs:
        dx, dy = kx - at[0], ky - at[1]
        along, across = abs(dx * ux + dy * uy), abs(-dx * uy + dy * ux)
        best = min(best, max(across - FINGER_HALF - KNOB_R, along - FINGER_REACH - KNOB_R))
    return best


def fill_order(spots) -> List[int]:
    """In what order to fill a silhouette's places (``places``): big pieces
    first -- unless another order, still with the two large triangles
    first, lets the fingers setting each piece down miss every knob already
    down by ``ORDER_CLEAR``; then the first such, bigger pieces first.  A
    piece's square post lets the fingers close along two lines only, and
    with two neighbours down each line can pass a different knob: big pieces
    first, the cat's second small triangle had 2 mm, and one time in eight
    its fingers came down on a knob and it could not be set down.  No order
    gives the boat 1 cm, and the other one tried there lost three boats in
    nine to big pieces first's one (Isaac, v13)."""
    first = [i for i, spot in enumerate(spots) if spot[0] == ORDER[0]]
    rest = [i for i in range(len(spots)) if i not in first]
    big_first = sorted(range(len(spots)), key=lambda i: ORDER.index(spots[i][0]))

    def clear(i: int, down) -> float:
        _, at, yaw = spots[i]
        knobs = [spots[j][1] for j in down]
        return max(finger_clearance(at, yaw + k * math.pi / 2, knobs) for k in range(2))

    def worst(order) -> float:
        return min(clear(i, order[:n]) for n, i in enumerate(order) if n)

    roomy = [first + list(tail) for tail in itertools.permutations(rest)
             if worst(first + list(tail)) >= ORDER_CLEAR]
    return max(roomy, key=lambda order: [-ORDER.index(spots[i][0]) for i in order]) \
        if roomy and worst(big_first) < ORDER_CLEAR else big_first


class _KnobPlace(PlacePosePolicy):
    """Takes a piece by its knob -- across two faces of the square post in it
    -- choosing, of the four ways that square up, one whose open fingers miss
    every other piece's knob where the piece is taken and where it is set
    down (the knobs stand higher than the fingertips: on Isaac, fingers
    opening beside a piece already placed knocked it out of its place), and
    of those the one that keeps the wrist nearest straight.  Lets go only
    once the piece is square."""

    #: the other pieces' knobs, as the policy last saw them
    knobs: tuple = ()

    def _hand_yaw(self, yaw: float, at) -> float:
        err = (yaw - self.yaw) % self.symmetry
        turn_by = -err if err <= self.symmetry / 2 else self.symmetry - err

        def wrap(a: float) -> float:
            return (a + math.pi) % (2 * math.pi) - math.pi

        def score(g: float):
            clear = min(finger_clearance(at, g, self.knobs),
                        finger_clearance(self.point, g + turn_by, self.knobs))
            return (min(clear, 0.005), -max(abs(g), abs(wrap(g + turn_by))))

        return max((wrap(yaw + k * math.pi / 2) for k in range(4)), key=score)


class TangramPolicy(ScriptedPolicy):
    """The largest places first, each with the nearest free piece of its
    kind.  The reference fills its own card only if it has seen the picture;
    without it, it does not know what the card is for, and lays the pieces
    out in a row on it.  (``film``: as it would fill the picture's card --
    the picture itself is the finished card, and nothing is run for it.)"""

    name = "tangram"

    def __init__(self, shape: str, film: bool, arm: str = RIGHT) -> None:
        self.shape, self.film, self.arm = shape, film, arm
        self.reset()

    #: how many times it looks the card over for pieces knocked out of place
    CHECKS = 2

    def reset(self) -> None:
        self.queue: List[Tuple[str, Tuple[float, float, float], float, float]] = []
        self._sub = None
        self._planned = False
        self._checks = 0

    def _recheck(self, world: WorldState) -> None:
        """Every place without its piece gets the nearest piece of its kind
        that is not in a place of its own."""
        found = matching(world, self.shape)
        spare = {name: kind for name, kind, _ in PIECES if name not in found}
        for (kind, (x, y), yaw), m in zip(places(self.shape), found):
            if m is None:
                name = min((n for n, k in spare.items() if k == kind),
                           key=lambda n: math.dist(world.obj(n).pose.pos[:2], (x, y)))
                spare.pop(name)
                self.queue.append((name, (x, y, CARD_TOP), yaw, SYMMETRY[kind]))

    def _watched(self, ctx: TurnContext) -> bool:
        return self.film or bool(ctx.extras.get("demonstrations"))

    def _plan(self, world: WorldState, ctx: TurnContext) -> None:
        free = {name: kind for name, kind, _ in PIECES}
        if self._watched(ctx):
            spots = places(self.shape)
            for kind, (x, y), yaw in (spots[i] for i in fill_order(spots)):
                name = min((n for n, k in free.items() if k == kind),
                           key=lambda n: math.dist(world.obj(n).pose.pos[:2], (x, y)))
                free.pop(name)
                self.queue.append((name, (x, y, CARD_TOP), yaw, SYMMETRY[kind]))
        else:
            # onto the card, as they came, the small ones in the middle
            cx, cy = ahead(*CARD_AT)
            spots = [(-0.08, -0.08), (-0.08, 0.08), (0.08, -0.08), (0.08, 0.08), (0.0, -0.09),
                     (0.0, 0.0), (0.0, 0.09)]
            for (name, kind, _), (dx, dy) in zip(PIECES, spots):
                self.queue.append((name, (cx + dx, cy + dy, CARD_TOP),
                                   yaw_of(world.obj(name).pose.quat), SYMMETRY[kind]))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS

        if ctx.extras.get("executor_busy"):
            return []
        if not self._planned:
            self._plan(world, ctx)
            self._planned = True
        if not self.queue and self._watched(ctx) and self._checks < self.CHECKS:
            self._checks += 1
            self._recheck(world)
        if not self.queue:
            return [Wait(1.0), Done("that is how it goes" if self.film else "filled")]
        name, point, yaw, sym = self.queue[0]
        if self._sub is None:
            self._sub = _KnobPlace(name, point, yaw, arm=self.arm, symmetry=sym,
                                   tolerance=0.006, yaw_tolerance=math.radians(5),
                                   release_clearance=0.003, refine=True,
                                   release_tolerance=0.004)
        o = world.obj(name)
        self._sub.knobs = tuple(tuple(world.obj(n).pose.pos[:2]) for n, _, _ in PIECES if n != name)
        if not o.is_held and self._sub.placed(world, o):
            self.queue.pop(0)
            self._sub = None
            return [Wait(0.3)]
        if self._sub.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"could not place {name}")]
        return [c for c in self._sub.commands(world, ctx) if not isinstance(c, Done)] \
            or [Wait(0.3)]
