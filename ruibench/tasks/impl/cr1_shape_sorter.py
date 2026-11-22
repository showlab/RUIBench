"""CR1 "shape sorter" -- Contact-Rich Manipulation (the suite's development task).

Knob: how much room a block has in its hole (6 mm all round -> 2 mm).

A toddler's shape sorter: a wooden box whose lid has four holes -- a cross, a
square, a round one and a triangle -- and four blocks of those shapes on the
table beside it.  "Put each block into the box through the hole of its own
shape."  What to do is all in the instruction.  How is not: a block goes
through only when it is turned the way its hole is (all but the round one),
and at the hard level it has to be brought within 2 mm of the hole's middle,
which the eye alone does not give -- a block let go a little off rests on the
lid, and has to be taken up and squared again.

Each block must go through its own hole.  At the easy level the round hole
and the square one are wide enough for other blocks, so which hole a block
went through is recorded as it drops below the lid (the reactor), and a block
through another shape's hole fails the task: the box does not give it back.

The box is the one FG2 had (v1, 2026-09-18: the lid is tiles whose inner
faces are the holes' edges, so a hole is exactly where it is drawn; its
reference passed 25/25 on Isaac at 1 mm), with every hole now cut to its own
block and all four blocks on the table.

Örnkloo & von Hofsten (2007) gave 14- to 26-month-olds blocks of these cross
sections to fit into holes: from 22 months children solved it consistently,
and those who did had turned the block before the hand reached the hole.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import quat_mul, rotate, yaw_of
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...engines.robots import arm_base
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done
from ...policies.base import ScriptedPolicy
from ...policies.scripted import PlacePosePolicy
from ...scoring import Custom, Finally, HoldsFor, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import ONE_ARM, RIGHT, TABLE, Frame, ahead, station_scene

SORTER = "sorter"
#: the shapes, in the order of the lid's cells: a 2 x 2 grid of holes 7 cm apart
SHAPES = ("cross", "square", "round", "triangle")
CELLS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
CELL = 0.035                                  # half a cell
#: the box: 16 cm square and 7 cm tall -- lower than the palm of a hand taking
#: a block beside it (8.1 cm over the table) -- its walls and lid 6 mm thick
BOX_HALF = (0.08, 0.08, 0.035)
WALL, LID = 0.006, 0.006
BOX_FINISH, LID_FINISH, INSIDE = "wood:ash", "wood:plywood", (0.25, 0.22, 0.2)

#: the blocks, 5 cm tall, held 1.8 cm above their middles: the pads close on
#: the top 1.5 cm, and the lower end goes into the hole before they open.
#: Stained wood, the grain showing through (as WK1's rings)
BLOCK_HALF_HEIGHT, BLOCK_GRASP = 0.025, 0.018
CROSS_ARM = T.CROSS_ARM                       # a plus sign 4.4 cm across, arms 1.6 cm wide
SQUARE = 0.016                                # half its side
ROUND = 0.018                                 # its radius
TRIANGLE = 0.042                              # its side
FINISH = {"cross": "wood:stain_green", "square": "wood:stain_blue",
          "round": "wood:stain_red", "triangle": "wood:stain_yellow"}
#: a turn of this takes each block onto itself
SYMMETRY = {"cross": math.pi / 2, "square": math.pi / 2, "round": math.pi / 4,
            "triangle": 2 * math.pi / 3}

#: the knob: the gap between a block and the edge of its hole, all round
CLEARANCE = {"easy": 0.006, "hard": 0.002}

SORTER_AT = (0.30, 0.0)
#: where the blocks start, two on each side of the box and 12 cm from each
#: other: the hand taking one (20.8 cm long along its fingers' travel) is over
#: the table, or over a corner of the box, which is lower than it
SLOTS = ((0.19, -0.22), (0.31, -0.22), (0.19, 0.22), (0.31, 0.22))
JITTER = 0.012
#: the blocks lie straight -- the way the fingers close on one within this of
#: across the table, so that the fixed camera, looking back from the far
#: side, sees the block between the fingers (at 15 degrees the near finger
#: hides half of it, at 30 most of it) -- and it is the box that stands
#: askew, by a seed-drawn angle between these, either way
STRAIGHT = math.radians(8)
SKEW = (math.radians(26), math.radians(40))
#: ...so that every block but the round one starts at least this far from any
#: way it fits: a cross or a triangle put down as it lies never goes in (they
#: fit within 17 and 18 degrees at the easy level)
MIN_TURN = math.radians(20)
#: square the block over its hole with its bottom this far above the lid, then
#: lower it this far -- 4 mm into the 6 mm lid -- and let go.  Let go above the
#: lid, a block squeezed with 8 N a finger jumped 5 mm along the fingers as
#: they opened and caught on the hole's rim (FG2, Isaac, 1 mm, three tries)
RELEASE, INSERT = 0.003, 0.007
STABLE_SECONDS = 1.0

#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.10, 0.42), (-0.285, 0.285), 0.10)


# -- the box --------------------------------------------------------------------
def _tile(x0, x1, y0, y1, z, color, dx=0.0, dy=0.0) -> P.Part:
    return P.box(((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, z),
                 ((x1 - x0) / 2, (y1 - y0) / 2, LID / 2), color)


def _edge_tile(normal: float, offset: float, depth: float, half_len: float, along: float,
               z, color, dx=0.0, dy=0.0) -> P.Part:
    """A tile whose inner face lies on the line at distance ``offset`` along
    ``normal`` (an angle), reaching ``depth`` outward and ``half_len`` either
    way of the point ``along`` the line."""
    nx, ny = math.cos(normal), math.sin(normal)
    cx = nx * (offset + depth / 2) - ny * along + dx
    cy = ny * (offset + depth / 2) + nx * along + dy
    return P.box((cx, cy, z), (depth / 2, half_len, LID / 2), color, P.turn_z(normal))


ROUND_SIDES = 32


def triangle_inradius(side: float) -> float:
    return side / (2 * math.sqrt(3))


def hole_side(clearance: float) -> float:
    """The triangular hole's side: the block's, its edges moved out by the
    clearance."""
    return TRIANGLE + 2 * math.sqrt(3) * clearance


def _hole_tiles(shape: str, clearance: float, z: float, color, dx: float, dy: float):
    """The tiles of one cell of the lid, round a hole for ``shape`` cut
    ``clearance`` bigger than the block all round."""
    c = CELL
    t = lambda *a: _tile(*a, z, color, dx, dy)
    if shape == "cross":
        a, b = CROSS_ARM[0] + clearance, CROSS_ARM[1] + clearance
        out = [t(*sorted((sx * b, sx * c)), *sorted((sy * b, sy * c)))
               for sx in (-1, 1) for sy in (-1, 1)]
        for s in (-1, 1):
            out += [t(*sorted((s * a, s * c)), -b, b), t(-b, b, *sorted((s * a, s * c)))]
        return out
    if shape == "square":
        q = SQUARE + clearance
        return [t(q, c, -c, c), t(-c, -q, -c, c), t(-q, q, q, c), t(-q, q, -c, -q)]
    if shape == "round":
        r = ROUND + clearance
        f, half = min(c, r + 0.008), math.pi / ROUND_SIDES
        out = [t(f, c, -c, c), t(-c, -f, -c, c), t(-f, f, f, c), t(-f, f, -c, -f)]
        for k in range(ROUND_SIDES):
            ang = 2 * math.pi * k / ROUND_SIDES
            reach = max(f / max(abs(math.cos(a)), abs(math.sin(a)))
                        for a in (ang - half, ang, ang + half)) + 0.001
            out.append(_edge_tile(ang, r, reach - r, reach * math.tan(half) + 0.0005, 0.0,
                                  z, color, dx, dy))
        return out
    # a triangle, point toward the box's +x, its bounding box centred in the cell
    side = hole_side(clearance)
    r_in = triangle_inradius(side)
    x_base, x_apex = -1.5 * r_in, 1.5 * r_in
    out = [t(-c, c, side / 2, c), t(-c, c, -c, -side / 2), t(x_apex, c, -side / 2, side / 2),
           t(-c, x_base, -side / 2, side / 2)]
    # beside each slanted edge, the corner of the bounding box: a right triangle,
    # as two wedges that meet at the middle of the edge (each is isosceles there).
    # FG2's lid had one long tile along each edge, reaching past the corner; with
    # holes this size it reached into the next cell's hole
    for s in (-1, 1):
        out.append(_wedge((x_apex, s * side / 4), math.pi, side / 4, x_apex, z, color, dx, dy))
        out.append(_wedge((0.0, s * side / 2), -s * math.pi / 2, x_apex, side / 4, z, color, dx, dy))
    return out


def _wedge(base_mid, toward: float, half_base: float, height: float, z, color, dx, dy) -> P.Part:
    """A triangular tile: its base's middle at ``base_mid`` (in the cell), its
    point ``height`` from it in the direction ``toward`` (an angle)."""
    ux, uy = math.cos(toward), math.sin(toward)
    centre = (base_mid[0] + ux * height / 2 + dx, base_mid[1] + uy * height / 2 + dy, z)
    # a roof block laid flat (its point along +y, see ``block``), turned to point ``toward``
    rot = quat_mul(P.turn_z(toward - math.pi / 2), (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)))
    return P.prism(centre, (half_base, LID / 2, height / 2), color, rot=rot)


def shape_sorter(name: str, xy, clearance: float, yaw: float = 0.0) -> BodySpec:
    """The box, turned ``yaw``: a floor, four walls, and a lid of tiles with
    ``SHAPES``' holes in ``CELLS``."""
    hx, hy, hz = BOX_HALF
    w = WALL
    wall_h = (2 * hz - w - LID) / 2
    wall_z = (w - LID) / 2
    parts = [P.box((0, 0, -hz + w / 2), (hx, hy, w / 2), INSIDE),
             P.box((hx - w / 2, 0, wall_z), (w / 2, hy, wall_h), BOX_FINISH),
             P.box((-hx + w / 2, 0, wall_z), (w / 2, hy, wall_h), BOX_FINISH),
             P.box((0, hy - w / 2, wall_z), (hx - w, w / 2, wall_h), BOX_FINISH),
             P.box((0, -hy + w / 2, wall_z), (hx - w, w / 2, wall_h), BOX_FINISH)]
    z = hz - LID / 2
    edge = 2 * CELL
    parts += [_tile(edge, hx, -hy, hy, z, LID_FINISH), _tile(-hx, -edge, -hy, hy, z, LID_FINISH),
              _tile(-edge, edge, edge, hy, z, LID_FINISH),
              _tile(-edge, edge, -hy, -edge, z, LID_FINISH)]
    for shape, (sx, sy) in zip(SHAPES, CELLS):
        parts += _hole_tiles(shape, clearance, z, LID_FINISH, sx * CELL, sy * CELL)
    return BodySpec(name, "compound", BOX_HALF,
                    Pose((xy[0], xy[1], TABLE + hz), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
                    category="shape sorter", color=BOX_FINISH, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect",
                           # a block passes a hole it fits to a fifth of a millimetre
                           "fit_tolerance": 0.0002})


def lid_top(sorter) -> float:
    return sorter.pose.pos[2] + BOX_HALF[2]


def cell_centre(sorter, shape: str) -> Tuple[float, float]:
    """The middle of a shape's cell of the lid, the box turned as it stands."""
    sx, sy = CELLS[SHAPES.index(shape)]
    dx, dy, _ = rotate(sorter.pose.quat, (sx * CELL, sy * CELL, 0.0))
    return sorter.pose.pos[0] + dx, sorter.pose.pos[1] + dy


def in_box(sorter, point) -> Tuple[float, float]:
    """A point across the table in the box's own axes, from its middle."""
    dx, dy = point[0] - sorter.pose.pos[0], point[1] - sorter.pose.pos[1]
    yaw = yaw_of(sorter.pose.quat)
    return math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy


def cell_of(sorter, point) -> Optional[str]:
    """The shape whose cell of the lid a point is over; None: not over the lid."""
    x, y = in_box(sorter, point)
    if max(abs(x), abs(y)) > 2 * CELL:
        return None
    return SHAPES[CELLS.index((1 if x > 0 else -1, 1 if y > 0 else -1))]


# -- the blocks -----------------------------------------------------------------
def triangle_height() -> float:
    return TRIANGLE * math.sqrt(3) / 2


def block(shape: str, xy, yaw: float) -> BodySpec:
    """One block, standing, turned ``yaw``.  The hand takes it across its own
    heading -- the fingers close along its y: on the ends of a cross's arm, on
    two faces of the square, on two flats of the round one's eight-sided core,
    and on the triangle's base and the edge across from it (its point is along
    +y), where a squeeze holds it square."""
    h = BLOCK_HALF_HEIGHT
    finish = FINISH[shape]
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    attrs = {"kind": shape, "footprint": "rect", "grasp_height": BLOCK_GRASP, "grasp_yaw": yaw}
    if shape == "cross":
        a, b = CROSS_ARM
        parts = (P.box((0, 0, 0), (a, b, h), finish), P.box((0, 0, 0), (b, a, h), finish))
        size = (a, a, h)
    elif shape == "square":
        parts = (P.box((0, 0, 0), (SQUARE, SQUARE, h), finish),)
        size = (SQUARE, SQUARE, h)
    elif shape == "round":
        flat = ROUND / T.ROUND_OVER
        parts = T.octagon_core((flat, h)) + (P.lathe((0, 0, 0), "cylinder", ROUND, h, finish),)
        size = (ROUND, ROUND, h)
        attrs["footprint"] = "round"
    else:
        # a roof block laid on its side: the triangle in plan, its point along +y
        th = triangle_height()
        parts = (P.prism((0, 0, 0), (TRIANGLE / 2, h, th / 2), finish,
                         rot=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))),)
        size = (TRIANGLE / 2, th / 2, h)
    return BodySpec(shape, "compound", size, Pose((xy[0], xy[1], TABLE + h), quat),
                    category=f"{shape} block", color=finish, mass=0.03,
                    attrs={**attrs, "parts": tuple(parts)})


def fit_yaw(shape: str, clearance: float) -> float:
    """How far a block may be turned from square to its hole and still pass."""
    if shape == "cross":
        a, b = CROSS_ARM
        return math.asin(min(1.0, (b + clearance) / math.hypot(a, b))) - math.atan2(b, a)
    if shape == "square":
        return math.asin(min(1.0, (SQUARE + clearance) / (SQUARE * math.sqrt(2)))) - math.pi / 4
    if shape == "triangle":
        r = triangle_inradius(TRIANGLE)
        return math.pi / 3 - math.acos(min(1.0, (r + clearance) / (2 * r)))
    return math.pi


def target(sorter, shape: str, clearance: float, yaw_now: float) -> Tuple[Tuple[float, float], float]:
    """Where a block's middle goes, and turned how, to pass its hole: of the
    turns that fit, the one nearest ``yaw_now``.  A triangle's middle (of its
    bounding box) is not its centroid, so where it goes depends on the turn."""
    box_yaw = yaw_of(sorter.pose.quat)
    cx, cy = cell_centre(sorter, shape)
    if shape != "triangle":
        return (cx, cy), box_yaw
    # the block's point (its +y) toward the hole's (the box's +x)
    base = box_yaw - math.pi / 2
    sym = SYMMETRY[shape]
    err = (yaw_now - base) % sym
    yaw = yaw_now + (-err if err <= sym / 2 else sym - err)
    # the hole's centroid, and the block's middle from its own centroid
    r_hole = triangle_inradius(hole_side(clearance))
    hx, hy, _ = rotate(sorter.pose.quat, (-0.5 * r_hole, 0.0, 0.0))
    off = triangle_height() / 6
    return (cx + hx - math.sin(yaw) * off, cy + hy + math.cos(yaw) * off), base


def layout(seed: int) -> Tuple[float, Dict[str, Tuple[Tuple[float, float], float]]]:
    """This seed's mission: the box's turn (a number of quarter turns -- every
    seed builds the same box, and each hole is at each corner as often -- and
    its skew), and each block's place and heading."""
    group, k = divmod(seed, 4)
    corners = [0, 1, 2, 3]
    SeedBundle(group).stream("turns").shuffle(corners)
    rng = SeedBundle(seed).stream("blocks")
    box_yaw = corners[k] * math.pi / 2 + rng.choice((-1, 1)) * rng.uniform(*SKEW)
    slots = list(SLOTS)
    rng.shuffle(slots)
    blocks = {}
    for shape, (sx, sy) in zip(SHAPES, slots):
        x, y = ahead(sx + rng.uniform(-JITTER, JITTER), sy + rng.uniform(-JITTER, JITTER))
        blocks[shape] = ((x, y), _heading(shape, box_yaw, rng))
    return box_yaw, blocks


def _heading(shape: str, box_yaw: float, rng) -> float:
    """A block's heading at the start: straight to within ``STRAIGHT`` (a
    triangle's point toward either side), and ``MIN_TURN`` or more from every
    turn that fits -- which a round block has none of."""
    sym = SYMMETRY[shape]
    fits = box_yaw - (math.pi / 2 if shape == "triangle" else 0.0)
    while True:
        yaw = rng.uniform(-STRAIGHT, STRAIGHT) + (math.pi * rng.randrange(2) if shape == "triangle" else 0.0)
        off = (yaw - fits) % sym
        if shape == "round" or min(off, sym - off) >= MIN_TURN:
            return (yaw + math.pi) % (2 * math.pi) - math.pi


def build_scene(level: str, seed: int):
    box_yaw, blocks = layout(seed)
    sorter = shape_sorter(SORTER, ahead(*SORTER_AT), CLEARANCE[level], yaw=box_yaw)
    bodies = [block(shape, xy, yaw) for shape, (xy, yaw) in blocks.items()]
    return station_scene(f"cr1_shape_sorter_{level}", bodies + [sorter], frame=FRAME), sorter


# -- what counts ----------------------------------------------------------------
def through(w: WorldState, shape: str) -> bool:
    """A block let go, inside the box, below the lid -- inside in the box's own
    axes (v2.3: the world's, and the box stands turned, so a block in a corner
    of it was outside, and one on the table by a wall inside)."""
    b, box = w.obj(shape), w.obj(SORTER)
    inner = BOX_HALF[0] - WALL
    under = box.pose.pos[2] + BOX_HALF[2] - LID - BLOCK_HALF_HEIGHT
    x, y = in_box(box, b.pose.pos)
    return not b.is_held and abs(x) < inner and abs(y) < inner and b.pose.pos[2] < under


def went_in(shape: str, hole: str) -> str:
    return f"{shape}_through_{hole}"


def own_hole(w: WorldState, shape: str) -> bool:
    return went_in(shape, shape) in w.events_fired


def wrong_hole(w: WorldState) -> bool:
    return any(went_in(s, h) in w.events_fired for s in SHAPES for h in SHAPES if h != s)


def home(w: WorldState, shape: str) -> bool:
    return through(w, shape) and own_hole(w, shape)


class Sorter(Reactor):
    """Records which hole each block went through, as it drops below the lid:
    ``<shape>_through_<hole>``, once a block."""

    def __init__(self) -> None:
        super().__init__()
        self._seen: Dict[str, str] = {}

    def tick(self, engine: Engine, world: WorldState) -> None:
        box = world.obj(SORTER)
        for shape in SHAPES:
            if shape in self._seen or not through(world, shape):
                continue
            hole = cell_of(box, world.obj(shape).pose.pos) or "none"
            self._seen[shape] = hole
            self.note(engine, world, went_in(shape, hole))


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CR1")
    scene, sorter = build_scene(level, seed)
    clearance = CLEARANCE[level]

    def count(w: WorldState) -> int:
        return sum(home(w, s) for s in SHAPES)

    def still(w: WorldState) -> bool:
        return count(w) == len(SHAPES) and all(
            max(abs(v) for v in w.obj(s).lin_vel) < 0.01 for s in SHAPES)

    all_in = Custom(lambda w: count(w) == len(SHAPES), "every block in the box through its own hole")
    milestones = tuple(
        Milestone(name, Custom(lambda w, n=n: count(w) >= n, f"{n} in through their own holes"))
        for n, name in ((1, "one_block_in"), (2, "two_blocks_in"), (3, "three_blocks_in"),
                        (4, "all_four_in")))
    base = arm_base(ONE_ARM, "right", TABLE)
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Put each block into the box through the hole of its own shape.",
        scorer=Scorer(task_id="CR1", level=level,
                      success=Finally(all_in) & HoldsFor(Custom(still, "all in and still"),
                                                         STABLE_SECONDS),
                      milestones=milestones,
                      violations=(("own_hole_only",
                                   Never(Custom(wrong_hole,
                                                "a block through another shape's hole"))),)),
        milestones=milestones,
        reactor_factory=Sorter,
        reference_factory=lambda inst, lv=level: SorterPolicy(lv, base),
        metadata={"knob": spec.knob.field, "knob_value": clearance,
                  "hole_play_m": round(2 * clearance, 4),
                  "fit_yaw_deg": {s: round(math.degrees(fit_yaw(s, clearance)), 1)
                                  for s in SHAPES if s != "round"},
                  "max_model_calls": 120},
        extra_metrics={"tries": _tries},
    )


def _tries(traj) -> int:
    """How many times a block was let go."""
    n = 0
    for shape in SHAPES:
        held = [w.obj(shape).is_held for w in traj]
        n += sum(1 for a, b in zip(held, held[1:]) if a and not b)
    return n


# -- the reference --------------------------------------------------------------
class BlockPolicy(PlacePosePolicy):
    """One block: taken across its heading, turned square to its hole, lowered
    to just above the lid, corrected in place to a fraction of the hole's
    play, its end pushed into the hole, let go.  Done when it is in the box;
    left on the lid, taken up and tried again."""

    def __init__(self, shape: str, level: str, base, arm: str = RIGHT) -> None:
        clearance = CLEARANCE[level]
        super().__init__(shape, (0.0, 0.0, 0.0), 0.0, arm=arm, symmetry=SYMMETRY[shape],
                         release_clearance=RELEASE, tolerance=0.0,
                         yaw_tolerance=max(0.01, fit_yaw(shape, clearance) / 3), base=base,
                         refine=True, release_tolerance=clearance / 3, insert=INSERT)
        self.shape, self.clearance = shape, clearance
        #: the rule-blind play: the block put down as it lies, never turned
        self.blind = False

    def aim(self, world: WorldState) -> None:
        obj, sorter = world.obj(self.obj), world.obj(SORTER)
        yaw_now = yaw_of(obj.pose.quat)
        (x, y), yaw = target(sorter, self.shape, self.clearance, yaw_now)
        if self.blind:
            # over its hole, as it is: for a triangle, as if it were square to it
            (x, y), yaw = (cell_centre(sorter, self.shape) if self.shape != "triangle"
                           else (x, y)), yaw_now
        self.point, self.yaw = (x, y, lid_top(sorter)), yaw

    def placed(self, world: WorldState, obj) -> bool:
        return through(world, self.shape)

    def _heading_error(self, obj) -> float:
        return 0.0 if self.blind else super()._heading_error(obj)

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        self.aim(world)
        return super().commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        self.aim(world)
        return super().plan(world, ctx)


class SorterPolicy(ScriptedPolicy):
    """The four blocks, one after another, each through its own hole: the
    round one first (it needs no turning), the cross last.  ``ignore_rules``
    (the blind control): every block is put over its hole as it lies."""

    name = "shape_sorter"
    ORDER = ("round", "square", "triangle", "cross")

    def __init__(self, level: str, base, arm: str = RIGHT) -> None:
        self.level, self.base, self.arm = level, base, arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._blocks = {s: BlockPolicy(s, self.level, self.base, self.arm) for s in self.ORDER}
        self._left: set = set()

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        for shape in self.ORDER:
            if shape in self._left or (through(world, shape) and not world.obj(shape).is_held):
                continue
            policy = self._blocks[shape]
            policy.blind = self.ignore_rules
            cmds = policy.commands(world, ctx)
            if cmds and isinstance(cmds[0], Done):       # gave this one up
                self._left.add(shape)
                continue
            return cmds
        return [Done("every block is in the box" if not self._left
                     else f"left out: {', '.join(sorted(self._left))}")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        out: List[Command] = []
        for shape in self.ORDER:
            out += [c for c in self._blocks[shape].plan(world, ctx) if not isinstance(c, Done)]
        return out + [Done("planned")]
