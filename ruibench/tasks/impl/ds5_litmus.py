"""DS5 "which one is the acid?" -- Discovery.

Knob: how many beakers, and how much pH paper (three beakers, paper to
spare -> five, and only four strips).

A row of clear glass beakers on the bench, lettered A, B, C..., each half full
of something as clear as water; a white tile with four pH strips lying on it;
a glass stirring rod standing beside a pot of water; a red mat (and, hard, a
blue one).  Easy: "One beaker holds an acid.  Put it on the red mat."  Hard:
"Put the acid on the red mat and the base on the blue mat."

Nothing tells the liquids apart but a test, and the test is done the way a lab
does it (v3.1): a strip is never put into a beaker -- that would spoil the
whole beaker -- but a drop is lifted out on the glass rod and touched to a
strip on the tile, which turns red for the acid, green for water, blue for the
base.  The rod is rinsed in the water pot between beakers; carried from one
beaker to the next unrinsed it still holds the first drop, and the strip reads
that one instead.  A strip reads once and keeps its colour.  What the colours
mean is in the hint since 10/4 ("red with an acid, blue with a base"): a
person at the bench would be told, and the task is to find out which beaker,
not to know litmus.

Hard: five beakers (one acid, one base, three water) and four strips -- one
beaker can never be tested, and when the four tests leave one of the two
unfound, it is the fifth: elimination, and remembering which strip read which
beaker.

v3 (2026-09-19), replacing the treasure chest: a Discovery task from the
chemistry lab.  The beakers' walls are glass drawn over a hidden square shell,
open at the top -- flat faces to grasp, room inside for the rod; the rod is a
glass cylinder over a hidden square core, stood in a block so it can be taken
from above; each pad is a mark painted on its strip, so the reactor can
recolour it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...events.actions import SetColor
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, KnownBefore, Milestone, Never, On, Scorer, Upright
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

LETTERS = "ABCDE"
BEAKERS = {"easy": 3, "hard": 5}
STRIPS = 4
#: what is in them: one acid, (hard) one base, the rest water
ACID, BASE, WATER = "acid", "base", "water"
#: what a pad turns in each -- universal indicator; and before it is used
PAD_COLOR = {ACID: (0.86, 0.14, 0.12), WATER: (0.36, 0.68, 0.26), BASE: (0.22, 0.24, 0.66)}
UNUSED = (0.97, 0.80, 0.42)
RED_MAT, BLUE_MAT = "red_mat", "blue_mat"
MAT_FOR = {ACID: RED_MAT, BASE: BLUE_MAT}

# -- a beaker: 5.2 cm across, 7 cm tall, filled to 4.5 cm --------------------------
BEAKER_R, BEAKER_HZ = 0.026, 0.035
GLASS = "glass:clear"
#: every liquid looks like water, the acid and the base too
LIQUID = "glass:water"
SHELL = 0.0015                  # the hidden square shell's walls
FLOOR = 0.004
LIQUID_H = 0.045
#: a beaker is held 1.5 cm above its middle: the fingertips on the glass, the
#: palm above the rim
BEAKER_GRASP = 0.015
BEAKER_MASS = 0.12

# -- a strip: 5.5 cm of paper lying flat on the tile, the pad at the near end ------
STRIP_HALF = (0.0275, 0.005, 0.0008)
PAD_HALF = (0.007, 0.0045, 0.0009)
STRIP_PITCH = 0.026

# -- the white tile the strips lie on ----------------------------------------------
TILE_HALF = (0.05, 0.062)
TILE_H = 0.006

# -- the glass rod: 14 cm, a glass cylinder over a hidden square core, standing
#    in a block with a square hole so it can be taken from above -----------------
ROD_LEN = 0.11
ROD_HZ = ROD_LEN / 2
#: the rod's glass (v9: 1 cm across and solid -- 8.4 mm of thin-walled glass
#: could hardly be seen), and the hidden core the fingers close on
ROD_R, ROD_CORE = 0.005, 0.0032
ROD_GRASP_BELOW_TOP = 0.016
ROD_FOOT = 0.006                 # its foot above the table, standing in the block
BLOCK_HALF = (0.022, 0.022)
BLOCK_H = 0.045
BLOCK_HOLE = 0.0055

# -- the pot of water the rod is rinsed in -----------------------------------------
POT_R, POT_HZ = 0.030, 0.026
POT_WATER = 0.032                # how deep it is, from the table

#: the layout, ahead of the station line: beakers in a row across at the back;
#: in front of them, right to left, the tile of strips, the water pot, the rod
#: in its block, and the mats
BEAKER_X, BEAKER_PITCH = 0.34, 0.10
TILE_AT = (0.125, -0.215)
POT_AT = (0.135, -0.098)
BLOCK_AT = (0.135, -0.005)
MAT_AT = {RED_MAT: (0.13, 0.095), BLUE_MAT: (0.13, 0.215)}
MAT_HALF = (0.045, 0.045)
FRAME = Frame((0.06, 0.40), (-0.30, 0.28), 0.14)

#: the hand's heights over the table: carrying the rod clear of the rims,
#: dipping its tip (8 mm off a beaker's floor), touching a pad, standing it back
CARRY_Z = TABLE + 0.25
DIP_FOOT = FLOOR + 0.008
POT_FOOT = 0.010
PAD_TOUCH = TILE_H + 2 * STRIP_HALF[2] + 0.0015
BLOCK_FOOT = ROD_FOOT
#: tipped further than this from upright: knocked over (spilt)
KNOCKED = math.radians(35)


def letters(level: str) -> str:
    return LETTERS[:BEAKERS[level]]


def beaker_name(letter: str) -> str:
    return f"beaker_{letter}"


def strip_name(i: int) -> str:
    return f"strip{i + 1}"


def pad_name(i: int) -> str:
    return f"strip{i + 1}_pad"


def contents(level: str, seed: int) -> Dict[str, str]:
    """Which beaker holds what: acid (and base) drawn per seed, water elsewhere."""
    lets = list(letters(level))
    SeedBundle(seed).stream("litmus/contents").shuffle(lets)
    out = {l: WATER for l in lets}
    out[lets[0]] = ACID
    if level == "hard":
        out[lets[1]] = BASE
    return out


def beaker_xy(level: str, letter: str) -> Tuple[float, float]:
    """A left to right as the robot faces them (+y is its left)."""
    n = BEAKERS[level]
    k = LETTERS.index(letter)
    return ahead(BEAKER_X, (n - 1) / 2 * BEAKER_PITCH - k * BEAKER_PITCH)


def strip_xy(i: int) -> Tuple[float, float]:
    """Where strip ``i`` lies on the tile, right to left as the robot faces it."""
    x, y = ahead(*TILE_AT)
    return (x, y + (i - (STRIPS - 1) / 2) * STRIP_PITCH)


def pad_xy(w: WorldState, i: int) -> Tuple[float, float]:
    return tuple(w.obj(pad_name(i)).pose.pos[:2])


def rod_top(w: WorldState) -> float:
    return w.obj("glass_rod").pose.pos[2] + ROD_HZ


def rod_tip(w: WorldState) -> Tuple[float, float, float]:
    """The rod's lower end, which carries the drop."""
    x, y, z = w.obj("glass_rod").pose.pos
    return (x, y, z - ROD_HZ)


def stuck_label(centre, r: float, width: float, spec: str) -> List[tuple]:
    """A printed label stuck round a glass of radius ``r`` on its -x side,
    facing the robot, and its plain back, seen through the glass (v8)."""
    x, y, z = centre
    face = P.printed((x - r - 0.0003, y, z), width, spec, P.turn_y(-math.pi / 2), wrap=r + 0.0003)
    hx, hy, _ = face[2]
    back = ("print", (x - r - 0.0001, y, z), (hx, hy, -(r + 0.0001)), P.turn_y(math.pi / 2),
            "paper:blank:", False)
    return [face, back]


def beaker(letter: str, xy) -> BodySpec:
    """Glass over a hidden square shell open at the top; the liquid a clear
    column inside; a white label with the letter, facing the robot."""
    r, hz = BEAKER_R, BEAKER_HZ
    parts = [P.grip((0, 0, -hz + FLOOR / 2), (r, r, FLOOR / 2))]
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = (SHELL / 2, r, hz) if sx else (r - SHELL, SHELL / 2, hz)
        parts.append(P.grip((sx * (r - SHELL / 2), sy * (r - SHELL / 2), 0), half))
    parts.append(P.vessel((0, 0, 0), r, hz, GLASS))
    liquid = LIQUID_H - 0.003
    parts.append(P.cylinder((0, 0, -hz + 0.003 + liquid / 2), r - 0.0022, liquid / 2, LIQUID,
                            collides=False))
    parts += stuck_label((0.0, 0.0, 0.004), r, 0.022, f"letter:{letter}")
    return BodySpec(beaker_name(letter), "compound", (r, r, hz), Pose((xy[0], xy[1], TABLE + hz)),
                    category="beaker", color=GLASS, mass=BEAKER_MASS,
                    attrs={"parts": tuple(parts), "footprint": "round", "grasp_height": BEAKER_GRASP,
                           "letter": letter})


def strip(i: int) -> Tuple[BodySpec, BodySpec]:
    """A pH strip lying flat on the tile, the pad (a mark the reactor recolours)
    at the end nearest the robot.  It is never picked up: the drop comes to it."""
    hx, hy, hz = STRIP_HALF
    x, y = strip_xy(i)
    z = TABLE + TILE_H + hz
    body = BodySpec(strip_name(i), "box", STRIP_HALF, Pose((x, y, z)),
                    category="pH strip", color="plastic:white", mass=0.002,
                    attrs={"footprint": "rect", "graspable": False})
    pad = BodySpec(pad_name(i), "box", PAD_HALF, Pose((x - hx + PAD_HALF[0], y, z + hz),),
                   category="pH pad", color=UNUSED,
                   attrs={"collides": False, "graspable": False, "attached_to": strip_name(i)})
    return body, pad


def tile(name: str = "white_tile") -> BodySpec:
    """The white porcelain tile the strips are read on, a faint square drawn
    round each strip's place."""
    hx, hy = TILE_HALF
    hz = TILE_H / 2
    parts = [P.box((0, 0, 0), (hx, hy, hz), "ceramic:white")]
    for i in range(STRIPS):
        dy = (i - (STRIPS - 1) / 2) * STRIP_PITCH
        parts.append(P.box((0, dy, hz), (STRIP_HALF[0] + 0.003, STRIP_HALF[1] + 0.003, 0.0004),
                           (0.85, 0.86, 0.88), collides=False))
    x, y = ahead(*TILE_AT)
    return BodySpec(name, "compound", (hx, hy, hz), Pose((x, y, TABLE + hz)),
                    category="tile", color="ceramic:white", fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "rect", "graspable": False})


def rod(name: str = "glass_rod") -> BodySpec:
    """A glass stirring rod: a hidden square core the fingers close on, the
    glass drawn over it, standing in its block."""
    parts = [P.grip((0, 0, 0), (ROD_CORE, ROD_CORE, ROD_HZ)),
             P.lathe((0, 0, 0), "capsule", ROD_R, ROD_HZ - ROD_R, "glass:solid")]
    x, y = ahead(*BLOCK_AT)
    return BodySpec(name, "compound", (ROD_R, ROD_R, ROD_HZ),
                    Pose((x, y, TABLE + ROD_FOOT + ROD_HZ)),
                    category="glass rod", color="glass:solid", mass=0.02,
                    attrs={"parts": tuple(parts), "footprint": "rect",
                           "grasp_height": ROD_HZ - ROD_GRASP_BELOW_TOP})


def rod_block(name: str = "rod_block") -> BodySpec:
    """The wooden block the rod stands in: a base and a collar of four walls."""
    hx, hy = BLOCK_HALF
    hz = BLOCK_H / 2
    wood = "wood:ash"
    parts = [P.box((0, 0, -hz + 0.004), (hx, hy, 0.004), wood)]
    rail = (hx - BLOCK_HOLE) / 2
    for s in (-1, 1):
        parts.append(P.box((s * (BLOCK_HOLE + rail), 0, 0.004), (rail, hy, hz - 0.004), wood))
        parts.append(P.box((0, s * (BLOCK_HOLE + rail), 0.004), (BLOCK_HOLE, rail, hz - 0.004), wood))
    x, y = ahead(*BLOCK_AT)
    return BodySpec(name, "compound", (hx, hy, hz), Pose((x, y, TABLE + hz)),
                    category="block", color="wood", fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "rect", "graspable": False})


def rinse_pot(name: str = "water_pot") -> BodySpec:
    """A squat glass pot of water, labelled, for rinsing the rod between beakers."""
    r, hz = POT_R, POT_HZ
    parts = [P.vessel((0, 0, 0), r, hz, GLASS)]
    water = POT_WATER - 0.004
    parts.append(P.cylinder((0, 0, -hz + 0.003 + water / 2), r - 0.0022, water / 2, "glass:water",
                            collides=False))
    parts.append(P.grip((0, 0, -hz + 0.002), (r - 0.004, r - 0.004, 0.002)))
    parts += stuck_label((0.0, 0.0, 0.002), r, 0.036, "label:H\u2082O")
    x, y = ahead(*POT_AT)
    return BodySpec(name, "compound", (r, r, hz), Pose((x, y, TABLE + hz)),
                    category="water pot", color=GLASS, fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "round", "graspable": False})


def build_scene(level: str):
    bodies: List[BodySpec] = [beaker(l, beaker_xy(level, l)) for l in letters(level)]
    bodies.append(tile())
    for i in range(STRIPS):
        bodies += list(strip(i))
    bodies += [rod_block(), rod(), rinse_pot()]
    mats = [RED_MAT] + ([BLUE_MAT] if level == "hard" else [])
    bodies += [T.mat(m, ahead(*MAT_AT[m]), MAT_HALF, "fabric:red" if m == RED_MAT else "fabric:blue")
               for m in mats]
    return station_scene(f"ds5_litmus_{level}", bodies, frame=FRAME)


def on_mat(w: WorldState, beaker: str, mat: str) -> bool:
    return On(beaker, mat).holds(w) and not w.obj(beaker).is_held


def found_by_readings(held: Dict[str, str], kind: str):
    """What the readings so far leave open for the beaker holding ``kind``: a
    reading names the beaker its drop came from (``<strip>_read_<letter>``), and
    its colour says what that beaker holds -- the one sought, or one it is not.
    Every other beaker read rules them all out as well (``KnownBefore``)."""
    def narrow(w: WorldState, left: frozenset, memo: dict) -> frozenset:
        events = w.events_fired
        for e in events[memo.get("seen", 0):]:
            if "_read_" in e:
                letter = e.split("_read_")[1]
                left = frozenset((letter,)) if held.get(letter) == kind else left - {letter}
        memo["seen"] = len(events)
        return left
    return narrow


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DS5")
    scene = build_scene(level)
    held = contents(level, seed)
    wanted = {kind: beaker_name(l) for l, kind in held.items() if kind in MAT_FOR}
    others = [beaker_name(l) for l in letters(level)]

    def placed(w: WorldState, kind: str) -> bool:
        mat = MAT_FOR[kind]
        b = wanted[kind]
        alone = not any(on_mat(w, o, mat) for o in others if o != b)
        return on_mat(w, b, mat) and alone and AtRest(b).holds(w)

    done = Custom(lambda w: all(placed(w, k) for k in wanted), "each on its mat")
    # a beaker set on its mat only once the readings had found it: a pick at
    # random was right one time in three (easy), and nothing else said so
    found = tuple((f"{kind}_found_first",
                   KnownBefore(next(l for l, k in held.items() if k == kind), tuple(held),
                               found_by_readings(held, kind),
                               Custom(lambda w, b=wanted[kind], m=MAT_FOR[kind]: on_mat(w, b, m),
                                      f"the {kind} set on its mat"),
                               f"which beaker holds the {kind}"))
                  for kind in wanted)
    knocked = Custom(lambda w: any(not Upright(b, tol=KNOCKED).holds(w) for b in others),
                     "a beaker knocked over")
    dipped_strip = Custom(lambda w, l=letters(level): any(pad_in_beaker(w, i, l) for i in range(STRIPS)),
                          "a strip put into a beaker")
    milestones = (
        Milestone("first_reading", Custom(lambda w: any("_read_" in e for e in w.events_fired),
                                          "a drop read on a strip")),
        Milestone("acid_picked", Custom(lambda w: w.obj(wanted[ACID]).is_held, "the acid lifted")),
        Milestone("acid_on_mat", Custom(lambda w: placed(w, ACID), "the acid on the red mat")),
    ) + ((Milestone("base_on_mat", Custom(lambda w: placed(w, BASE), "the base on the blue mat")),)
         if level == "hard" else ())
    instruction = ("One beaker holds an acid. Put it on the red mat." if level == "easy"
                   else "Put the acid on the red mat and the base on the blue mat.")
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=instruction,
        scorer=Scorer(task_id="DS5", level=level, success=Finally(done), milestones=milestones,
                      violations=(("nothing_knocked_over", Never(knocked)),
                                  ("no_strip_in_a_beaker", Never(dipped_strip))) + found),
        milestones=milestones,
        reactor_factory=lambda h=dict(held): Litmus(h),
        reference_factory=lambda inst, l=level: LitmusPolicy(letters(l), find_base=(l == "hard")),
        metadata={"knob": spec.knob.field, "knob_value": BEAKERS[level], "strips": STRIPS,
                  # hidden from the agent: what each beaker holds (the gap-filled control)
                  "contents": dict(held), "max_model_calls": 40 + 12 * BEAKERS[level]},
        extra_metrics={"strips_used": lambda traj: len({e.split("_read_")[0]
                                                        for e in traj[-1].events_fired
                                                        if "_read_" in e}) if traj else 0,
                       "rod_carried_over": lambda traj: sum(1 for e in traj[-1].events_fired
                                                            if e.startswith("rod_still_wet"))
                       if traj else 0},
    )


def in_beaker(w: WorldState, tip, letter: str) -> bool:
    """Is the rod's tip in beaker ``letter``'s liquid -- over its floor and
    under the liquid's top?  Without the floor, a beaker carried over the rod
    standing in its rack wet it (the RUIKit collaborator's audit, v15)."""
    b = w.obj(beaker_name(letter)).pose.pos
    floor = b[2] - BEAKER_HZ
    return math.dist(tip[:2], b[:2]) < BEAKER_R - 0.006 and floor < tip[2] < floor + LIQUID_H


def in_pot(w: WorldState, tip) -> bool:
    pot = w.obj("water_pot").pose.pos
    floor = pot[2] - POT_HZ
    return math.dist(tip[:2], pot[:2]) < POT_R - 0.006 and floor < tip[2] < floor + POT_WATER


def on_pad(w: WorldState, tip, i: int) -> bool:
    """Is the rod's tip touching strip ``i``'s pad?"""
    pad = w.obj(pad_name(i)).pose.pos
    return (abs(tip[0] - pad[0]) < PAD_HALF[0] + 0.004
            and abs(tip[1] - pad[1]) < PAD_HALF[1] + 0.004
            and tip[2] < pad[2] + 0.008)


def pad_in_beaker(w: WorldState, i: int, letters_: Sequence[str]) -> bool:
    """A strip put into a beaker -- the one thing the lab does not allow: its
    pad inside the beaker, over the floor and under the rim.  Under the rim
    alone, a beaker carried 23 cm over the tile of strips counted as a strip
    put into it (the RUIKit collaborator's audit, v15: any path but a
    straight one failed DS5)."""
    pad = w.obj(pad_name(i)).pose.pos
    for letter in letters_:
        b = w.obj(beaker_name(letter)).pose.pos
        if (math.dist(pad[:2], b[:2]) < BEAKER_R
                and b[2] - BEAKER_HZ < pad[2] < b[2] + BEAKER_HZ):
            return True
    return False


class Litmus(Reactor):
    """The bench's chemistry.

    The rod takes up a drop from the beaker its tip goes into, and keeps that
    drop until it is rinsed in the water pot: carried into a second beaker
    unrinsed, it still holds the first one's.  Touched to a strip's pad, the
    drop on the rod colours it -- red, green or blue -- and the strip keeps
    that colour.  A dry rod reads nothing."""

    def __init__(self, held: Dict[str, str]) -> None:
        super().__init__()
        self.held = held
        self._used: Dict[int, str] = {}
        self._drop: Optional[str] = None       # which beaker's liquid is on the rod
        self._where: Optional[str] = None      # where the tip was last tick

    def tick(self, engine: Engine, world: WorldState) -> None:
        tip = rod_tip(world)
        here: Optional[str] = None
        for letter in self.held:
            if in_beaker(world, tip, letter):
                here = f"beaker:{letter}"
                break
        if here is None and in_pot(world, tip):
            here = "pot"
        if here is None:
            for i in range(STRIPS):
                if on_pad(world, tip, i):
                    here = f"pad:{i}"
                    break
        if here == self._where:
            return
        self._where = here
        if here is None:
            return
        what, _, which = here.partition(":")
        if what == "beaker":
            if self._drop is None:
                self._drop = which
                self.note(engine, world, f"rod_took_a_drop_from_{which}")
            elif self._drop != which:
                # not rinsed: the first drop is still on it and wins
                self.note(engine, world, f"rod_still_wet_from_{self._drop}_in_{which}")
        elif what == "pot":
            if self._drop is not None:
                self._drop = None
                self.note(engine, world, "rod_rinsed")
        else:
            i = int(which)
            if i in self._used:
                return
            if self._drop is None:
                self.note(engine, world, f"{strip_name(i)}_touched_by_a_dry_rod")
                return
            self._used[i] = self._drop
            self.act(engine, world, f"{strip_name(i)}_read_{self._drop}",
                     SetColor(pad_name(i), PAD_COLOR[self.held[self._drop]]))


def read_pad(color) -> Optional[str]:
    """What a pad's colour says -- acid, water, base -- or None, unused."""
    if not isinstance(color, (tuple, list)):
        return None
    best = min(PAD_COLOR, key=lambda k: math.dist(PAD_COLOR[k], color))
    return best if math.dist(PAD_COLOR[best], color) < math.dist(UNUSED, color) else None


class LitmusPolicy(ScriptedPolicy):
    """Test the beakers in turn the way the lab does it: take the glass rod,
    lift a drop out of a beaker, touch it to a fresh strip on the tile, rinse
    the rod in the water pot, and go on.  Stop as soon as the acid (and the
    base) are known -- by a reading, or because every other beaker has been
    tested -- stand the rod back in its block, and carry the beakers to their
    mats.  It reads the pads' colours and remembers which strip read which
    beaker."""

    name = "litmus"

    def __init__(self, lets: Sequence[str], find_base: bool, arm: str = RIGHT) -> None:
        self.lets, self.find_base, self.arm = list(lets), find_base, arm
        self.reset()

    def reset(self) -> None:
        self._tested: Dict[int, str] = {}      # strip -> the beaker its drop came from
        self._phase = "choose"
        self._strip: Optional[int] = None
        self._beaker: Optional[str] = None
        self._sub = None
        self._moving: Optional[str] = None
        self._tries = 0
        self._after = "choose"
        self._lift_at = (0.0, 0.0)

    # -- what is known ---------------------------------------------------------
    def _results(self, world: WorldState) -> Dict[str, str]:
        out = {}
        for i, letter in self._tested.items():
            kind = read_pad(world.obj(pad_name(i)).color)
            if kind:
                out[letter] = kind
        return out

    def _known(self, world: WorldState) -> Dict[str, Optional[str]]:
        """kind -> beaker letter, for the kinds to find; None while unknown."""
        results = self._results(world)
        kinds = [ACID] + ([BASE] if self.find_base else [])
        found = {k: next((l for l, r in results.items() if r == k), None) for k in kinds}
        untested = [l for l in self.lets if l not in results]
        missing = [k for k in kinds if found[k] is None]
        if len(missing) == len(untested) == 1:
            found[missing[0]] = untested[0]         # by elimination
        return found

    # -- moves ---------------------------------------------------------------------
    def _down(self, world: WorldState):
        from ...policies.scripted import grasp_pose

        return grasp_pose(world.obj("glass_rod")).quat

    def _hand_for_tip(self, world: WorldState, xy, foot: float) -> Pose:
        """The hand where the held rod's tip is ``foot`` above the table."""
        arm = world.arm(self.arm)
        rod_obj = world.obj("glass_rod")
        offset = arm.eef_pose.pos[2] - rod_obj.pose.pos[2]
        return Pose((xy[0], xy[1], TABLE + foot + ROD_HZ + offset), self._down(world))

    def _go(self, world: WorldState, xy, foot: float, nxt: str,
            wait: float = 0.6) -> List[Command]:
        """Over the spot and down until the tip is ``foot`` above the table,
        and stay: a Wait does not gate an arm's own queue, so lifting again is
        the next turn's work (``lift``, which then goes on to ``nxt``)."""
        hand = world.arm(self.arm).eef_pose.pos
        down = self._down(world)
        self._phase, self._after, self._lift_at = "lift", nxt, (xy[0], xy[1])
        return [MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), down)),
                MoveTo(self.arm, Pose((xy[0], xy[1], CARRY_Z), down)),
                MoveTo(self.arm, self._hand_for_tip(world, xy, foot)), Wait(wait)]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PlaceAtPointPolicy, grasp_pose

        if ctx.extras.get("executor_busy"):
            return []
        if self._phase == "choose":
            known = self._known(world)
            if all(known.values()):
                self._phase = "stow"
                self._todo = [(k, l) for k, l in known.items()]
            else:
                results = self._results(world)
                untested = [l for l in self.lets if l not in results and l not in self._tested.values()]
                free = [i for i in range(STRIPS) if i not in self._tested]
                if not untested or not free:
                    self._phase = "stow"
                    self._todo = [(k, l) for k, l in known.items() if l]
                else:
                    self._strip, self._beaker = free[0], untested[0]
                    self._phase = "take"
                    self._tries = 0
        if self._phase == "take":
            if world.obj("glass_rod").is_held:
                self._phase = "dip"
            else:
                self._tries += 1
                if self._tries > MAX_GRASP_ATTEMPTS:
                    return [Done("could not take the glass rod")]
                g = grasp_pose(world.obj("glass_rod"))
                above = Pose((g.pos[0], g.pos[1], CARRY_Z), g.quat)
                return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above), MoveTo(self.arm, g),
                        SetGripper(self.arm, 1.0), Wait(0.5)]
        if self._phase == "lift":
            self._phase = self._after
            return [MoveTo(self.arm, Pose((*self._lift_at, CARRY_Z), self._down(world))),
                    Wait(0.2)]
        if self._phase == "dip":
            bx, by = world.obj(beaker_name(self._beaker)).pose.pos[:2]
            return self._go(world, (bx, by), DIP_FOOT, "touch", wait=0.8)
        if self._phase == "touch":
            self._tested[self._strip] = self._beaker
            return self._go(world, pad_xy(world, self._strip), PAD_TOUCH, "rinse", wait=0.8)
        if self._phase == "rinse":
            pot = world.obj("water_pot").pose.pos
            return self._go(world, (pot[0], pot[1]), POT_FOOT, "choose", wait=0.8)
        if self._phase == "stow":
            if not world.obj("glass_rod").is_held:
                self._phase = "carry"
            else:
                x, y = ahead(*BLOCK_AT)
                out = self._go(world, (x, y), BLOCK_FOOT, "carry", wait=0.3)
                self._phase, self._after = "release", "carry"
                return out
        if self._phase == "release":
            x, y = self._lift_at
            self._phase = "carry"
            return [SetGripper(self.arm, 0.0), Wait(0.5),
                    MoveTo(self.arm, Pose((x, y, CARRY_Z), self._down(world)))]
        if self._phase == "carry":
            while self._todo:
                kind, letter = self._todo[0]
                name = beaker_name(letter)
                mat = world.obj(MAT_FOR[kind])
                if on_mat(world, name, MAT_FOR[kind]) and self._moving == name:
                    self._todo.pop(0)
                    self._sub = None
                    continue
                if self._moving != name:
                    self._moving = name
                    self._sub = PlaceAtPointPolicy(name, (mat.pose.pos[0], mat.pose.pos[1],
                                                          mat.pose.pos[2] + mat.size[2]),
                                                   arm=self.arm, tolerance=0.02,
                                                   release_clearance=0.004, carry_z=CARRY_Z)
                cmds = self._sub.commands(world, ctx)
                if self._sub.attempts > MAX_GRASP_ATTEMPTS:
                    return [Done(f"could not move beaker {letter}")]
                return [c for c in cmds if not isinstance(c, Done)] or [Wait(0.3)]
            return [Wait(1.0), Done("found and placed")]
        return [Wait(0.5)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind (the first picture only): no test to see -- the first beaker to
        the red mat, the second to the blue."""
        from ...policies.scripted import _pick_and_place_sequence

        out: List[Command] = []
        kinds = [ACID] + ([BASE] if self.find_base else [])
        for kind, letter in zip(kinds, self.lets):
            mat = world.obj(MAT_FOR[kind])
            out += _pick_and_place_sequence(self.arm, world.obj(beaker_name(letter)),
                                            mat.pose.pos, mat.size[2], 0.20, 0.004)
        return out + [Done("guessed")]


def secrets(inst) -> Tuple[str, ...]:
    """The beakers that hold something other than water (``TaskInstance.secrets``)."""
    return tuple(beaker_name(letter) for letter, held in dict(inst.metadata["contents"]).items()
                 if held != "water")
