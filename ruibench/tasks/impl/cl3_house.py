"""CL3 "build the house" -- In-Context Learning.

Knob: how many blocks the house on the card has (3 -> 7).

A set of wooden blocks lies on the table -- six painted cubes, two each of
red, blue and yellow, two planks (one bare ash, one painted green) and a red
roof -- beside a wooden board to build on, and a card on an easel at the
back, facing the camera, with a picture of the house.  "Build the house on the card, on the wooden board."
Easy: a gate, two cubes and a plank across them.  Hard (v3): a two-storey
house with a doorway and a chimney -- two door posts and the lintel across
them, two more cubes and the flat roof on those, and the chimney, a cube of
the third colour, at one end of the roof.  **The same nine blocks are on the
table at both levels** (the red roof block is never asked for since v3) --
only the card changes -- so the knob changes the number of steps and not the
clutter.  Which colours the card asks for is drawn per seed.  The order is
the structure's own (what holds up what); the instruction never states it.
The card is drawn by hand (v3): each block in its colour inside a dark,
not quite straight outline, smoke from the chimney.

Blocks of a colour are interchangeable: the card shows colours and shapes,
and the scorer reads the house as kinds in their places relative to each
other, the house anywhere on the board (v2.3.1).  The long blocks start lying
along the camera's line of sight and are turned a quarter on the way: the
Franka takes each across its width, its hand across the fixed camera's view
(v2; the house faces the camera).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import yaw_of
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, color_rgb
from ...engines.robots import arm_base
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done
from ...policies.base import ScriptedPolicy
from ...scoring import And, AtRest, Custom, Finally, HoldsFor, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import ONE_ARM, RIGHT, TABLE, Frame, ahead, station_scene

# -- the blocks -----------------------------------------------------------------------
CUBE = 0.02                                  # half side: 4 cm cubes
PLANK = (0.08, 0.02, 0.008)                  # 16 x 4 x 1.6 cm, long along its own x
#: the roof: a triangle 16 cm across and 5 cm high, 5 cm deep; its triangular
#: faces are where the fingers hold it
ROOF = (0.08, 0.025, 0.025)
COLOURS = ("red", "blue", "yellow")
PLANKS = {"ash plank": "wood:ash", "green plank": "paint:green"}
ROOF_FINISH = "paint:red"
#: name -> kind; two cubes of each colour
BLOCKS: Dict[str, str] = {
    **{f"cube_{c}_{k}": f"{c} cube" for c in COLOURS for k in "ab"},
    "plank_ash": "ash plank", "plank_green": "green plank", "roof": "roof",
}

# -- where things are, from the arm's base (``ahead``) --------------------------------
BOARD = "board"
BOARD_AT, BOARD_HALF = (0.40, 0.15), (0.06, 0.11, 0.005)
#: the cubes in two staggered rows of three at the back, 11 cm apart across
#: the view (nearer the fixed camera, the hand hid 62-73% of a cube it held);
#: the long blocks in front of them, lying along the line of sight -- the roof
#: 6 cm further out, so a hand taking a plank passes beside its ridge
CUBE_SPOTS = [(0.21, -0.27), (0.21, -0.16), (0.21, -0.05),
              (0.30, -0.215), (0.30, -0.105), (0.30, 0.005)]
PLANK_SPOTS = [(0.47, -0.26), (0.47, -0.14)]
ROOF_SPOT = (0.53, -0.02)
CARD = "card"
#: the card: 24 x 17.6 cm on a little easel behind the cubes on the left, its
#: bottom edge here, leaning back so that it faces the fixed camera square on
#: (the camera looks down 35 degrees).  Stood in front, it would hide 20 cm of
#: table behind it -- the planks and a row of cubes (v2 revision, 2026-09-18;
#: it was 13 x 13 cm, upright, the house drawn at 45%)
CARD_AT = (0.165, -0.20)
CARD_HALF = (0.003, 0.12, 0.088)
CARD_TILT = math.radians(35)
#: the picture on the card: the house from the front, this much smaller
CARD_SCALE = 0.8
#: the board as the card draws it: in the brown the board shows in the cameras'
#: pictures.  Drawn in its material's flat colour (plywood), it came out the light
#: tan of the ash plank (rendered 176/153/118, the plank 198/180/157, the board
#: itself 123/90/55): a thin pale strip under the house, read as a plank to lay
#: down first -- by every Claude Code agent of the paper's runs (v2.3.1)
DRAWN_BOARD = (0.48, 0.30, 0.12)

#: the house on the board: pillars this far either side of its middle
PILLAR_Y = 0.05
#: the chimney stands over the second storey's wall on this side
CHIMNEY_Y = PILLAR_Y
STOREY = 2 * CUBE + 2 * PLANK[2]
#: a slot is filled by a block of its kind this close to it (and a long
#: block turned across the board, within ``TURN``)
NEAR_XY, NEAR_Z, TURN = 0.015, 0.01, math.radians(15)
STABLE_SECONDS = 3.0


#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.050, 0.650), (-0.325, 0.300), 0.17)


class Slot(tuple):
    """(kind, (dx, dy, dz) from the board's top middle, whether it lies across)."""

    @property
    def kind(self) -> str:
        return self[0]

    @property
    def offset(self) -> Tuple[float, float, float]:
        return self[1]

    @property
    def across(self) -> bool:
        return self[2]


def slots_for(n: int, colours: Sequence[str], planks: Sequence[str]) -> List[Slot]:
    """The house, bottom up, as the order it has to go up in: a storey is two
    cubes and a plank across them -- the first storey's the doorway and its
    lintel; seven blocks put the chimney on the second's flat roof, over one
    of its walls."""
    out: List[Slot] = []
    for storey in range(2):
        z0 = storey * STOREY
        out += [Slot((f"{colours[storey]} cube", (0.0, -PILLAR_Y, z0 + CUBE), False)),
                Slot((f"{colours[storey]} cube", (0.0, PILLAR_Y, z0 + CUBE), False)),
                Slot((planks[storey], (0.0, 0.0, z0 + 2 * CUBE + PLANK[2]), True))]
    out.append(Slot((f"{colours[2]} cube", (0.0, CHIMNEY_Y, 2 * STOREY + CUBE), False)))
    return out[:n]


def design_for(seed: int) -> Tuple[List[str], List[str]]:
    """This seed's colours: the cubes of each storey, and its plank."""
    rng = SeedBundle(seed).stream("design")
    colours = list(COLOURS)
    rng.shuffle(colours)
    planks = list(PLANKS)
    rng.shuffle(planks)
    return colours, planks


def cube(name: str, xy) -> BodySpec:
    colour = BLOCKS[name].split()[0]
    # held by its upper half: the hand and the wrist camera then hide half of
    # it from the fixed camera, not three quarters
    return BodySpec(name, "box", (CUBE, CUBE, CUBE), Pose((xy[0], xy[1], TABLE + CUBE)),
                    category="block", color=f"paint:{colour}", mass=0.04,
                    attrs={"kind": BLOCKS[name], "footprint": "rect", "grasp_height": 0.01})


def plank(name: str, xy) -> BodySpec:
    return BodySpec(name, "box", PLANK, Pose((xy[0], xy[1], TABLE + PLANK[2])),
                    category="block", color=PLANKS[BLOCKS[name]], mass=0.05,
                    attrs={"kind": BLOCKS[name], "footprint": "rect", "grasp_yaw": 0.0})


def roof(name: str, xy) -> BodySpec:
    """Lying with its triangular faces across the view: the fingers close on
    them, half way up, and the palm clears the ridge by 1.3 cm."""
    parts = (P.prism((0.0, 0.0, 0.0), ROOF, ROOF_FINISH),)
    return BodySpec(name, "compound", ROOF, Pose((xy[0], xy[1], TABLE + ROOF[2])),
                    category="block", color=ROOF_FINISH, mass=0.06,
                    attrs={"parts": parts, "kind": "roof", "footprint": "rect", "grasp_yaw": 0.0,
                           "grasp_height": 0.0, "curved_top": True})


def board_body() -> BodySpec:
    x, y = ahead(*BOARD_AT)
    return BodySpec(BOARD, "box", BOARD_HALF, Pose((x, y, TABLE + BOARD_HALF[2])),
                    category="board", color="wood:plywood", fixed=True,
                    attrs={"graspable": False, "footprint": "rect"})


#: the drawing's pencil: dark outlines this thick, each edge a little off
#: straight -- a hand drew it
PENCIL = (0.13, 0.11, 0.1)
LINE = 0.0011
WOBBLE = (math.radians(1.5), 0.0005)


def card_bodies(slots: Sequence[Slot]) -> List[BodySpec]:
    """A card on an easel, facing the fixed camera, with the house drawn on it
    by hand, from the front, to scale: the board, then each block in its
    colour inside a pencil outline -- smoke from the chimney.  (Until v2.3.1 a
    green ground line ran under the board: under a house of blocks, read as a
    green plank to lay first.)
    Every coloured mark of the tallest house is always there, the ones this
    house does not use tucked inside the card, so both levels build the same
    bodies; the marks' colours are set per seed."""
    from ...core.geometry import quat_mul, rotate

    hx, hy, hz = CARD_HALF
    sc = CARD_SCALE
    # the card leans back: its up turns toward the robot, its face up toward the camera
    lean = (math.cos(-CARD_TILT / 2), 0.0, math.sin(-CARD_TILT / 2), 0.0)
    x0, y = ahead(*CARD_AT)
    up = rotate(lean, (0.0, 0.0, hz))
    centre = (x0 + up[0], y, TABLE + up[2] + 0.001)

    def on_card(local):                              # a point on the card, from its middle
        return rotate(lean, local)

    def drawn(dy, dz, half_w, half_h, colour, depth=0.0006, turn=0.0):
        """A flat shape drawn on the card's face, (dy, dz) from its middle."""
        rot = quat_mul(lean, P.turn_x(turn)) if turn else lean
        return P.box(on_card((hx + depth, dy, dz)), (0.0004, half_w, half_h), colour, rot=rot,
                     collides=False)

    base = -hz + 0.035                               # the drawn board's top, up the card
    # the easel's leg: from high on the card's back down to the table behind it
    top = on_card((-hx, 0.0, 0.4 * hz))
    foot = (top[0] - 0.05, 0.0, -up[2])
    leg_mid = tuple((a + b) / 2 for a, b in zip(top, foot))
    leg_len = math.dist(top, foot) / 2
    leg_turn = math.atan2(top[0] - foot[0], top[2] - foot[2])
    leg_rot = (math.cos(leg_turn / 2), 0.0, math.sin(leg_turn / 2), 0.0)
    board_z = base - sc * BOARD_HALF[2]
    board = [P.box((0.0, 0.0, 0.0), CARD_HALF, (0.96, 0.94, 0.88), rot=lean),
             P.box(leg_mid, (0.004, 0.012, leg_len), "wood:oak", rot=leg_rot, collides=False),
             drawn(0.0, board_z, sc * BOARD_HALF[1], sc * BOARD_HALF[2], DRAWN_BOARD)]
    shown = len(slots)
    tallest = slots_for(7, COLOURS, list(PLANKS))

    def outline(cy: float, cz: float, w: float, h: float, k: int) -> List:
        """A pencil outline round a w x h shape at (cy, cz) on the face, each
        edge a little off straight (the same way every time)."""
        out = []
        for j, (ey, ez, ew, eh) in enumerate(((cy, cz + h, w + LINE, LINE), (cy, cz - h, w + LINE, LINE),
                                              (cy - w, cz, LINE, h + LINE), (cy + w, cz, LINE, h + LINE))):
            wob = math.sin(7.3 * k + 2.1 * j)
            out.append((ey + WOBBLE[1] * wob, ez - WOBBLE[1] * wob, ew, eh, WOBBLE[0] * wob))
        return out

    # the drawn board's outline is on the card itself: every house has it
    for ey, ez, ew, eh, turn in outline(0.0, board_z, sc * BOARD_HALF[1], sc * BOARD_HALF[2], 0):
        board.append(drawn(ey, ez, ew, eh, PENCIL, depth=0.0014, turn=turn))
    # its size is the card's: the leg behind it is decoration
    out = [BodySpec(CARD, "compound", P.half_extents(board), Pose(centre),
                    category="card", color="white", fixed=True,
                    attrs={"parts": tuple(board), "graspable": False})]

    def placed(name: str, k: int, dy: float, dz: float, half, colour, parts=None):
        """A mark on the card's face, (dy, dz) from the drawn board's top in
        the house's own metres -- tucked inside the card if this house does
        not use it, so both levels have the same bodies."""
        local = (hx + 0.0007, sc * dy, base + sc * dz) if k < shown else (0.0, sc * dy, 0.0)
        pos = tuple(c + o for c, o in zip(centre, on_card(local)))
        attrs = {"collides": False, "graspable": False, "attached_to": CARD}
        if parts is None:
            return BodySpec(name, "box", half, Pose(pos, lean), category="mark", color=colour,
                            fixed=True, attrs=attrs)
        return BodySpec(name, "compound", P.half_extents(parts, solid_only=False), Pose(pos, lean),
                        category="mark", color=colour, fixed=True,
                        attrs={**attrs, "parts": tuple(parts)})

    for k, slot in enumerate(tallest):
        _, dy, dz = slot.offset
        hw, hh = (CUBE, CUBE) if slot.kind.endswith("cube") else (PLANK[0], PLANK[2])
        out.append(placed(f"card_mark{k}", k, dy, dz, (0.0006, sc * hw - 0.0007, sc * hh - 0.0007),
                          "white"))
        lines = [P.box((0.0007, ey, ez), (0.0004, ew, eh), PENCIL, rot=P.turn_x(turn), collides=False)
                 for ey, ez, ew, eh, turn in outline(0.0, 0.0, sc * hw, sc * hh, k + 1)]
        out.append(placed(f"card_line{k}", k, dy, dz, None, PENCIL, lines))
    # smoke from the chimney, puffs growing as they rise
    _, dy, dz = tallest[-1].offset
    puffs = [P.cylinder((0.0, 0.006 * (j + 1), 0.013 * j), r, 0.0004, (0.72, 0.72, 0.74),
                        rot=P.turn_y(math.pi / 2), collides=False)
             for j, r in enumerate((0.004, 0.0055, 0.007))]
    out.append(placed("card_smoke", len(tallest) - 1, dy, dz + CUBE + 0.008 / sc, None,
                      (0.72, 0.72, 0.74), puffs))
    return out


def mark_colours(slots: Sequence[Slot]) -> Dict[str, tuple]:
    """Each shown block's mark, in the block's colour."""
    out = {}
    for k, slot in enumerate(slots):
        if slot.kind.endswith("cube"):
            out[f"card_mark{k}"] = color_rgb(f"paint:{slot.kind.split()[0]}")
        elif slot.kind in PLANKS:
            out[f"card_mark{k}"] = color_rgb(PLANKS[slot.kind])
    return out


def build_scene(level: str, seed: int, slots: Sequence[Slot]):
    s = SeedBundle(seed)
    cubes = [n for n in BLOCKS if n.startswith("cube")]
    spots = list(CUBE_SPOTS)
    s.stream("cubes").shuffle(spots)
    bodies = [cube(n, ahead(*xy)) for n, xy in zip(cubes, spots)]
    planks = list(PLANK_SPOTS)
    s.stream("planks").shuffle(planks)
    bodies += [plank("plank_ash", ahead(*planks[0])), plank("plank_green", ahead(*planks[1]))]
    bodies.append(roof("roof", ahead(*ROOF_SPOT)))
    bodies.append(board_body())
    colours = mark_colours(slots)
    bodies += [replace(b, color=colours.get(b.name, b.color)) for b in card_bodies(slots)]
    return station_scene(f"lh1_house_{level}", bodies, frame=FRAME)


# -- reading the house ----------------------------------------------------------------
def board_top() -> Tuple[float, float, float]:
    x, y = ahead(*BOARD_AT)
    return (x, y, TABLE + 2 * BOARD_HALF[2])


def slot_pose(slot: Slot) -> Tuple[float, float, float]:
    bx, by, top = board_top()
    dx, dy, dz = slot.offset
    return (bx + dx, by + dy, top + dz)


def in_slot(obj, slot: Slot, shift: Tuple[float, float] = (0.0, 0.0)) -> bool:
    x, y, z = slot_pose(slot)
    x, y = x + shift[0], y + shift[1]
    if obj.is_held or obj.attrs.get("kind") != slot.kind:
        return False
    if math.hypot(obj.pose.pos[0] - x, obj.pose.pos[1] - y) > NEAR_XY or abs(obj.pose.pos[2] - z) > NEAR_Z:
        return False
    if slot.across:
        turn = (yaw_of(obj.pose.quat) - math.pi / 2) % math.pi
        return min(turn, math.pi - turn) < TURN
    return True


def _filled_at(world: WorldState, slots: Sequence[Slot],
               shift: Tuple[float, float]) -> List[Optional[str]]:
    used: set = set()
    out: List[Optional[str]] = []
    for slot in slots:
        hit = next((n for n in BLOCKS if n not in used and in_slot(world.obj(n), slot, shift)), None)
        if hit:
            used.add(hit)
        out.append(hit)
    return out


def filled(world: WorldState, slots: Sequence[Slot]) -> List[Optional[str]]:
    """The block in each slot, bottom up (None: empty); a block fills one slot.
    The house may stand anywhere on the board: the slots are read where they are,
    and moved across the board to each free block of the first slot's kind, and
    the reading that fills the most from the bottom up is the house's (v2.3.1: the
    card shows the house from the front, not how far back on the board it stands,
    and a house built as the card shows, 3 cm from the board's middle, was none)."""
    x0, y0, _ = slot_pose(slots[0])
    shifts = [(0.0, 0.0)] + [(o.pose.pos[0] - x0, o.pose.pos[1] - y0)
                             for o in (world.obj(n) for n in BLOCKS)
                             if o.attrs.get("kind") == slots[0].kind and not o.is_held]

    def rank(got: List[Optional[str]]) -> Tuple[int, int]:
        bottom = next((i for i, g in enumerate(got) if not g), len(got))
        return bottom, sum(map(bool, got))
    return max((_filled_at(world, slots, s) for s in shifts), key=rank)


def on_board(world: WorldState) -> List[str]:
    """Every free block standing on the board or on what is on it."""
    bx, by, top = board_top()
    out = []
    for n in BLOCKS:
        o = world.obj(n)
        if o.is_held:
            continue
        if (abs(o.pose.pos[0] - bx) < BOARD_HALF[0] + 0.01 and abs(o.pose.pos[1] - by) < BOARD_HALF[1] + 0.01
                and o.pose.pos[2] > top):
            out.append(n)
    return out


def house_is(slots: Sequence[Slot]) -> Custom:
    """Every slot filled, in its place, and nothing else on the board."""
    def check(w: WorldState) -> bool:
        got = filled(w, slots)
        return all(got) and sorted(on_board(w)) == sorted(got)
    return Custom(check, "the house on the card, on the board")


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CL3")
    n = int(spec.knob.levels[level])
    colours, planks = design_for(seed)
    slots = slots_for(n, colours, planks)
    scene = build_scene(level, seed, slots)
    house = house_is(slots)
    resting = And(tuple(AtRest(b) for b in BLOCKS))
    half = max(2, n // 2)

    def first(k: int) -> Custom:
        return Custom(lambda w, k=k: all(filled(w, slots)[:k]), f"the first {k} blocks in place")

    milestones = (
        Milestone("pillars", first(2)),
        Milestone("half_built", first(half)),
        Milestone("house_built", house),
        Milestone("stable", Custom(lambda w: house.holds(w) and resting.holds(w),
                                   "the house standing still")),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Build the house on the card, on the wooden board.",
        scorer=Scorer(task_id="CL3", level=level,
                      success=Finally(house) & HoldsFor(And((house, resting)), STABLE_SECONDS),
                      milestones=milestones),
        milestones=milestones,
        reference_factory=lambda inst, s=tuple(slots): HousePolicy(s),
        metadata={"knob": spec.knob.field, "knob_value": n,
                  "order": [slot.kind for slot in slots],
                  "board": [round(v, 4) for v in board_top()], "report_completion_ratio": True},
    )


class HousePolicy(ScriptedPolicy):
    """Fill the slots bottom up: for the lowest empty one, the nearest free
    block of its kind, put down square in its place.  A block that fell is
    put back before anything goes on top of it.

    ``ignore_rules`` (the blind control): never looks at the card -- builds a
    house of the same size in the blocks' own order (red, blue, yellow; the
    ash plank first), right one time in six."""

    name = "house"
    CLEARANCE = 0.004

    def __init__(self, slots: Sequence[Slot], arm: str = RIGHT) -> None:
        self._card = list(slots)
        self.arm = arm
        self.base = arm_base(ONE_ARM, arm.split("/")[-1], TABLE)
        self.ignore_rules = False
        self._subs: dict = {}

    @property
    def slots(self) -> List[Slot]:
        if self.ignore_rules:
            return slots_for(len(self._card), COLOURS, list(PLANKS))
        return self._card

    def reset(self) -> None:
        self._subs = {}

    def _block_for(self, world: WorldState, i: int, taken) -> Optional[str]:
        slot = self.slots[i]
        free = [n for n in BLOCKS if BLOCKS[n] == slot.kind and n not in taken]
        held = [n for n in free if world.obj(n).is_held]
        if held:
            return held[0]
        x, y, _ = slot_pose(slot)
        return min(free, key=lambda n: math.dist(world.obj(n).pose.pos[:2], (x, y)), default=None)

    def _sub(self, world: WorldState, i: int, taken):
        """The placing policy for slot ``i``, with the block it has chosen --
        chosen again if that block has gone into another slot."""
        from ...policies.scripted import PlacePosePolicy

        sub = self._subs.get(i)
        if sub is None or sub.obj in taken:
            name = self._block_for(world, i, taken)
            if name is None:
                return None
            slot = self.slots[i]
            x, y, z = slot_pose(slot)
            sub = PlacePosePolicy(name, (x, y, z - world.obj(name).size[2]),
                                  math.pi / 2 if slot.across else 0.0, arm=self.arm,
                                  symmetry=math.pi if slot.across else math.pi / 2, base=self.base,
                                  release_clearance=self.CLEARANCE)
            self._subs[i] = sub
        return sub

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS

        if ctx.extras.get("executor_busy"):
            return []
        got = filled(world, self.slots)
        taken = {n for n in got if n}
        for i, block in enumerate(got):
            if block:
                continue
            sub = self._sub(world, i, taken)
            if sub is None:
                return [Done(f"no block left for the {self.slots[i].kind}")]
            cmds = sub.commands(world, ctx)
            if sub.attempts > MAX_GRASP_ATTEMPTS:
                return [Done(f"could not place the {self.slots[i].kind}")]
            if any(isinstance(c, Done) for c in cmds):
                # placed as far as the sub-policy can tell, but not in the
                # slot by the scorer's reading: take it up again
                self._subs.pop(i, None)
                return []
            return cmds
        return [Done("house built")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import PlacePosePolicy

        out: List[Command] = []
        taken: set = set()
        for i, slot in enumerate(self.slots):
            name = self._block_for(world, i, taken)
            taken.add(name)
            x, y, z = slot_pose(slot)
            obj = world.obj(name)
            steps = PlacePosePolicy(name, (x, y, z - obj.size[2]), math.pi / 2 if slot.across else 0.0,
                                    arm=self.arm, symmetry=math.pi if slot.across else math.pi / 2,
                                    base=self.base, release_clearance=self.CLEARANCE).plan(world, ctx)
            out += [c for c in steps if not isinstance(c, Done)]
        return out + [Done("planned")]
