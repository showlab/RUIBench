"""WK2 "river crossing" -- World Knowledge.

Knob: how many rules there are (1 -> 2).

A tabletop diorama of a river 20 cm wide between grassy banks.  On the near bank stand a
toy wolf, a toy goat and a third thing -- a little wooden bucket at the easy
level, a cabbage at the hard one.  A wooden raft lies at the near bank's edge
with the farmer standing on it.  "Ferry the wolf, the goat and the bucket
(the cabbage) across.  The raft carries the farmer and one more.  Never
leave the wolf alone with the goat (or the goat with the cabbage)."  The
robot lifts a passenger aboard and takes its hand away, and the raft sets off
by itself; it lifts the passenger off on the far side.  To cross with nobody
aboard, it taps the raft's deck.  A bank is watched while the raft lies at
it; while it crosses, neither is.  Until all three are across, no unwatched bank may hold a
pair the rules forbid; nothing may be put down in the water, and nothing may
cross except on the raft.  With two aboard the raft stays put ("Only one
passenger, please!").

Easy: the goat goes first or last (five crossings).  Hard: the classic
puzzle, where the goat has to be brought back once (seven crossings).  The
rules are in the instruction; the third thing shows which set applies.

(v3, 2026-09-19: the raft replaces "the hand is the farmer and the boat" --
now who watches which bank can be seen.  v9, 9/24, the advisor: no bells --
put a passenger aboard and the raft goes; an empty raft is sent by a tap on
its deck, the farmer rowing back alone.)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import AtRest, Custom, Finally, HoldsFor, Milestone, Scorer
from ...scoring.predicates import Trajectory, TrajectoryPredicate
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

WOLF, GOAT = "wolf", "goat"
THIRD = {"easy": "bucket", "hard": "cabbage"}
RULES = {"easy": ((WOLF, GOAT),), "hard": ((WOLF, GOAT), (GOAT, "cabbage"))}
#: the river runs along y; its edges, and the banks' lanes, by distance from the
#: robot (x, ahead).  The near lanes 29 cm in front of the Franka's base: at 26
#: the middle one was at the edge of its reach and the hand missed the bucket
RIVER = (0.21, 0.41)
#: the lanes stand back from the water: a hand taking the bucket (its 20 cm
#: along the bank) clears the raft at the river's edge
NEAR_X, FAR_X = 0.11, 0.51
LANES = (-0.20, 0.0, 0.20)
RIVER_AT, RIVER_HALF = (0.31, 0.0), (0.25, 0.28)
#: the river as a tabletop diorama (v10, the advisor: the felt mat -- flat green
#: banks, a flat blue river -- was fake): a light wooden tray, turf on both
#: banks, a strip of wet sand along each edge of the water, a pebbled bed under
#: clear water, and a few stones on the banks, clear of the lanes.  All drawn
#: only, and thin -- the turf's top 3.5 mm up: what stands on it stands on the
#: table, and the raft's hull, 4 mm up, is just clear of the water
TRAY_RIM = (0.005, 0.006)                 # half the rim's thickness, half its height
TRAY_FLOOR = 0.0012
TURF_TOP, SAND_TOP, WATER_TOP, BED_TOP = 0.0035, 0.0032, 0.0031, 0.0018
SAND_WIDTH = 0.018                        # 6 mm of it under the water
#: the bank stones: (catalogue entry, scale, place from the river's middle, turn)
STONES = (("river_rock", 0.55, (-0.228, 0.252), 0.4), ("sea_pebble", 0.6, (-0.236, -0.258), 1.9),
          ("sea_pebble", 0.5, (0.232, 0.256), 2.6), ("river_rock", 0.5, (0.226, -0.250), 0.9),
          ("sea_pebble", 0.45, (-0.215, 0.268), 0.2))
#: things are carried this high: a goat hanging 8 cm below the hand clears
#: everything on the mat, the farmer too (at 18 cm one clipped a 12 cm bell
#: post and fell in the river, Isaac, when there were bells)
HOVER = 0.24
#: the raft: its half size (across the river, along it, its thickness), where
#: it docks at the near bank, how far it goes, and its speed
RAFT, FERRY, FARMER = "raft", "raft_slide", "farmer"
RAFT_HALF = (0.045, 0.085, 0.008)
RAFT_Y = 0.0
DECK = TABLE + 0.004 + 2 * RAFT_HALF[2]                   # the deck's top
NEAR_DOCK = RIVER[0] + RAFT_HALF[0] + 0.003
TRAVEL = (RIVER[1] - RAFT_HALF[0] - 0.003) - NEAR_DOCK
SPEED = 0.03                                             # m/s: a crossing takes 3.5 s
#: the passenger's place on the deck (y, from the raft's middle); the farmer
#: stands at the other end
SEAT_Y, FARMER_Y = -0.025, 0.055
#: the raft sets off by itself this long after a passenger is put aboard and
#: the hand is away from it -- over it, lower than this above the deck, or
#: within this of its sides, the hand is not away
DEPART_DELAY = 1.0
CLEAR_ABOVE, CLEAR_AROUND = 0.12, 0.03
#: an empty raft is sent across by a tap on its deck: an empty hand's
#: fingertips within TAP_REACH of the deck, over it -- here, the end away
#: from the farmer, the hand turned across the river
TAP_REACH = 0.012
FINGERTIP = 0.008
TAP_Y, TAP_YAW = -0.062, math.pi / 2
#: each thing's catalogue entry, measured half extents (for the mock), its turn
#: on the bank, and the hand's turn taking it (v2 revision, 2026-09-18)
LOOKS = {"wolf": ("wolf_grey", (0.0275, 0.055, 0.0364), 0.0, math.pi / 2),
         "goat": ("goat_white", (0.0147, 0.05, 0.0482), 0.0, math.pi / 2),
         "cabbage": ("cabbage", (0.0314, 0.0325, 0.0227), 0.0, 0.0),
         "bucket": ("bucket_small", (0.0217, 0.0202, 0.0325), 0.0, 0.0)}
#: where the fingers hold each thing, above its middle, when not where its
#: height puts it (``grasp_offset``).  The goat by its body: held 2.3 cm up,
#: the fingers found only its neck and the ridge of its back -- the body's
#: top is 6.4 cm up, the horns 9.6 -- and squeezed it 1.4 cm along itself,
#: one grasp in three; it then hung crooked, was let go 2.5 cm over the deck
#: and fell off the raft (Isaac, v13).  1.2 cm up the palm still clears the
#: horns by 5 mm
GRASP_HEIGHT = {"goat": 0.012}
NEAR, FAR, WATER, BOAT, HAND = "near", "far", "water", "raft", "hand"
INSTRUCTION = {
    "easy": ("Ferry the wolf, the goat and the bucket across. The raft carries the farmer and "
             "one more. Never leave the wolf alone with the goat."),
    "hard": ("Ferry the wolf, the goat and the cabbage across. The raft carries the farmer and "
             "one more. Never leave the goat alone with the wolf or the cabbage."),
}
FRAME = Frame((0.060, 0.560), (-0.300, 0.320), 0.13)      # the goat aboard 11.6 cm


def diorama(name: str, xy) -> BodySpec:
    """The river crossing's diorama (``RIVER_HALF``, ``RIVER``), drawn only."""
    hx, hy = RIVER_HALF
    edge = RIVER[1] - RIVER_AT[0]                       # the water's edges at -edge, +edge
    t, h = TRAY_RIM
    parts = [P.slab((0.0, 0.0, TRAY_FLOOR / 2), (hx + 2 * t, hy + 2 * t, TRAY_FLOOR / 2), "wood:ash")]
    for sx in (-1, 1):                                  # the rim round it
        parts.append(P.slab((sx * (hx + t), 0.0, h), (t, hy + 2 * t, h), "wood:ash"))
    for sy in (-1, 1):
        parts.append(P.slab((0.0, sy * (hy + t), h), (hx, t, h), "wood:ash"))
    bank = (hx - edge - SAND_WIDTH + 0.006) / 2          # each bank's half width
    for sx in (-1, 1):
        mid = sx * (hx - bank)
        parts.append(P.slab((mid, 0.0, TURF_TOP / 2), (bank, hy, TURF_TOP / 2), "ground:grass"))
        parts.append(P.box((sx * (edge + SAND_WIDTH / 2 - 0.006), 0.0, SAND_TOP / 2),
                           (SAND_WIDTH / 2, hy, SAND_TOP / 2), "sand:wet", collides=False))
    parts.append(P.box((0.0, 0.0, (TRAY_FLOOR + BED_TOP) / 2), (edge, hy, (BED_TOP - TRAY_FLOOR) / 2),
                       "ground:pebbles", collides=False))
    parts.append(P.box((0.0, 0.0, (BED_TOP + WATER_TOP) / 2), (edge, hy, (WATER_TOP - BED_TOP) / 2),
                       "glass:stream", collides=False))
    for asset, scale, (x, y), turn in STONES:
        half = tuple(scale * v for v in STONE_HALF[asset])
        parts.append(P.visual(asset, (x, y, TURF_TOP + half[2] * 0.6), half, P.turn_z(turn), scale))
    return BodySpec(name, "compound", (hx + 2 * t, hy + 2 * t, 2 * h), Pose((xy[0], xy[1], TABLE)),
                    category="river", color="blue", fixed=True,
                    attrs={"parts": tuple(parts), "collides": False, "graspable": False})


#: the bank stones' half extents as catalogued (for the mock)
STONE_HALF = {"river_rock": (0.031, 0.0171, 0.0147), "sea_pebble": (0.028, 0.0211, 0.0143)}


def zone(x: float) -> str:
    """Which side of the river a point is on, from its world x."""
    if x < ahead(RIVER[0])[0]:
        return NEAR
    if x > ahead(RIVER[1])[0]:
        return FAR
    return WATER


def things(level: str) -> Tuple[str, str, str]:
    return (WOLF, GOAT, THIRD[level])


def raft_bodies():
    """The raft (on its slide, at the near dock at 0), the farmer on it."""
    x, y = ahead(NEAR_DOCK, RAFT_Y)
    hx, hy, hz = RAFT_HALF
    logs = [P.cylinder((0, dy, 0), hz, hx, "wood:oak", rot=P.turn_y(math.pi / 2))
            for dy in [-hy + hz + 2 * hz * k for k in range(int(hy / hz))]]
    logs += [P.box((dx, 0, hz * 0.6), (0.004, hy, 0.002), "wood:ash", collides=False)
             for dx in (-hx + 0.008, hx - 0.008)]
    deck = P.box((0, 0, 0), (hx, hy, hz), "wood:oak", collides="hidden")
    raft = BodySpec(RAFT, "compound", RAFT_HALF, Pose((x, y, TABLE + 0.004 + hz)), category="raft",
                    color="wood", mass=0.2,
                    attrs={"parts": (deck, *logs), "graspable": False, "carried_by": FERRY,
                           "footprint": "rect"})
    slide = ArticulationSpec(FERRY, "slide", axis=(1.0, 0.0, 0.0), limits=(0.0, TRAVEL),
                             initial=0.0, actuated=True, parent=RAFT, carries=True)
    doll_half = (0.0136, 0.0307, 0.045)                # grandad, standing
    farmer = BodySpec(FARMER, "compound", doll_half,
                      Pose((x, y + FARMER_Y, DECK + doll_half[2])), category="farmer",
                      color="tan", mass=0.01,
                      attrs={"parts": (P.visual("doll_modern_1", (0, 0, 0), doll_half,
                                                P.turn_x(math.pi / 2)),),
                             "collides": False, "graspable": False, "attached_to": RAFT})
    return raft, slide, farmer


def build_scene(level: str, seed: int):
    lanes = list(LANES)
    SeedBundle(seed).stream("lanes").shuffle(lanes)
    names = things(level)
    bodies = [diorama("river", ahead(*RIVER_AT))]
    homes = {}
    for name, y in zip(names, lanes):
        asset, size, yaw, grasp_yaw = LOOKS[name]
        held = {"grasp_height": GRASP_HEIGHT[name]} if name in GRASP_HEIGHT else {}
        bodies.append(asset_body(name, asset, ahead(NEAR_X, y), fallback_size=size, mass=0.04,
                                 category=name, yaw=yaw, kind=name, grasp_yaw=grasp_yaw, **held))
        homes[name] = {NEAR: ahead(NEAR_X, y), FAR: ahead(FAR_X, y)}
    raft, slide, farmer = raft_bodies()
    bodies += [raft, farmer]
    joints = [slide]
    return station_scene(f"wk2_river_{level}", bodies, frame=FRAME, articulations=joints), homes


def tapping(world: WorldState) -> bool:
    """An empty hand come down on the raft's deck: fingertips at it, over it."""
    raft = world.obj(RAFT).pose.pos
    for r in world.robots.values():
        for arm in r.arms.values():
            if arm.holding:
                continue
            hx, hy, hz = arm.eef_pose.pos
            if abs(hx - raft[0]) < RAFT_HALF[0] and abs(hy - raft[1]) < RAFT_HALF[1] \
                    and hz - FINGERTIP < DECK + TAP_REACH:
                return True
    return False


def hand_away(world: WorldState) -> bool:
    """No hand over the raft low enough to be in the way of its setting off."""
    raft = world.obj(RAFT).pose.pos
    for r in world.robots.values():
        for arm in r.arms.values():
            hx, hy, hz = arm.eef_pose.pos
            if abs(hx - raft[0]) < RAFT_HALF[0] + CLEAR_AROUND \
                    and abs(hy - raft[1]) < RAFT_HALF[1] + CLEAR_AROUND and hz < DECK + CLEAR_ABOVE:
                return False
    return True


def raft_side(world: WorldState) -> str:
    """Where the raft (and the farmer) is: a bank, or the water."""
    v = world.articulations.get(FERRY, 0.0)
    if v < 0.005:
        return NEAR
    if v > TRAVEL - 0.005:
        return FAR
    return WATER


def aboard(world: WorldState, name: str) -> bool:
    obj, raft = world.obj(name), world.obj(RAFT)
    dx = obj.pose.pos[0] - raft.pose.pos[0]
    dy = obj.pose.pos[1] - raft.pose.pos[1]
    return (not obj.is_held and abs(dx) < RAFT_HALF[0] and abs(dy) < RAFT_HALF[1]
            and obj.pose.pos[2] > DECK - 0.005)


def banks(world: WorldState, names: Sequence[str]) -> Dict[str, str]:
    """Where each thing is: a bank, the water, the raft, or the hand."""
    out = {}
    for n in names:
        obj = world.obj(n)
        out[n] = HAND if obj.is_held else BOAT if aboard(world, n) else zone(obj.pose.pos[0])
    return out


def all_across(world: WorldState, names: Sequence[str]) -> bool:
    return all(v == FAR for v in banks(world, names).values())


def eaten(world: WorldState, names: Sequence[str], rules) -> Optional[str]:
    """A pair the rules forbid, left on a bank the raft is not at -- or None."""
    farmer = raft_side(world)
    where = banks(world, names)
    for bank in (NEAR, FAR):
        if bank == farmer:
            continue
        here = {n for n, b in where.items() if b == bank}
        for a, b in rules:
            if a in here and b in here:
                return f"{a} and {b} alone on the {bank} bank"
    return None


def passengers(world: WorldState, names: Sequence[str]) -> List[str]:
    return [n for n in names if aboard(world, n)]


@dataclass
class RiverRules(TrajectoryPredicate):
    """Nothing forbidden happened before everything was across: no pair left
    unwatched, nothing in the water, nothing across but on the raft."""

    names: Tuple[str, ...]
    rules: Tuple[Tuple[str, str], ...]

    def first_breach(self, traj: Trajectory) -> Optional[Tuple[float, str]]:
        """A thing changes bank only aboard: put on the raft where it lies docked,
        at the thing's own bank, and taken off at the bank the raft brings it to
        (v2.3: a moment aboard used to wipe out where a thing had been, and a hand
        could then carry it over the water; and a thing dropped on the raft
        mid-river rode with another)."""
        last_bank: Dict[str, str] = {}
        riding: set = set()
        for w in traj:
            if all_across(w, self.names):
                return None
            why = eaten(w, self.names, self.rules)
            if why:
                return w.sim_time, why
            for n, b in banks(w, self.names).items():
                if b != BOAT:
                    riding.discard(n)
                if b == WATER:
                    return w.sim_time, f"the {n} put down in the river"
                if b in (NEAR, FAR):
                    if last_bank.get(n) not in (None, b):
                        return w.sim_time, f"the {n} taken across without the raft"
                    last_bank[n] = b
                elif b == BOAT:
                    side = raft_side(w)
                    if n not in riding:                       # just put aboard
                        if side not in (NEAR, FAR) or last_bank.get(n) not in (None, side):
                            return w.sim_time, f"the {n} taken across without the raft"
                        riding.add(n)
                    if side in (NEAR, FAR):
                        last_bank[n] = side
        return None

    def evaluate(self, traj: Trajectory) -> bool:
        return self.first_breach(traj) is None

    def describe(self) -> str:
        return "no forbidden pair left unwatched, nothing in the water, nothing across but by raft"


class Ferry(Reactor):
    """The raft crosses by itself -- with the farmer and at most one
    passenger.  A passenger put aboard (not the one it has just brought) sends
    it off ``DEPART_DELAY`` after the hand is away; a tap on its deck sends it
    off at once, with nobody or whoever is aboard; with two aboard it stays
    and says so."""

    def __init__(self, names: Sequence[str]) -> None:
        super().__init__()
        self.names = tuple(names)
        self._arrived: FrozenSet[str] = frozenset()     # who came across on the last crossing
        self._ready_since: Optional[float] = None
        self._tapped = False
        self._refused: FrozenSet[str] = frozenset()
        self._target: Optional[float] = None
        self._command = 0.0
        self._t = 0.0

    def _leave(self, engine: Engine, world: WorldState) -> None:
        here = raft_side(world)
        self._target = TRAVEL if here == NEAR else 0.0 if here == FAR else None
        if self._target is not None:
            self._ready_since = None
            self.note(engine, world, f"raft_leaves_{here}")

    def tick(self, engine: Engine, world: WorldState) -> None:
        dt, self._t = max(world.sim_time - self._t, 0.0), world.sim_time
        tap = tapping(world)
        new_tap, self._tapped = tap and not self._tapped, tap
        if self._target is not None:
            step = SPEED * dt
            self._command += max(-step, min(step, self._target - self._command))
            engine.set_articulation_target(FERRY, self._command)
            if abs(self._command - self._target) < 1e-9 and \
                    abs(world.articulations.get(FERRY, 0.0) - self._target) < 0.004:
                side = FAR if self._target > 0 else NEAR
                self._target = None
                self._arrived = frozenset(passengers(world, self.names))
                self.note(engine, world, f"raft_at_{side}")
            return
        aboard_now = frozenset(passengers(world, self.names))
        if len(aboard_now) > 1:
            self._ready_since = None
            if aboard_now != self._refused:
                self._refused = aboard_now
                self.note(engine, world, "raft_refused")
                self.say(world, RAFT, "refused", "Only one passenger, please!")
            return
        self._refused = frozenset()
        if new_tap:
            self._leave(engine, world)
            return
        if aboard_now and aboard_now != self._arrived and hand_away(world):
            if self._ready_since is None:
                self._ready_since = world.sim_time
            elif world.sim_time - self._ready_since >= DEPART_DELAY:
                self._leave(engine, world)
        else:
            self._ready_since = None


# -- the puzzle, as a search ---------------------------------------------------------
#: a state: the farmer's bank and the set of things on the far bank
State = Tuple[str, FrozenSet[str]]


def safe(state: State, names: Sequence[str], rules) -> bool:
    farmer, far = state
    near = frozenset(names) - far
    for bank, here in ((NEAR, near), (FAR, far)):
        if bank != farmer and any(a in here and b in here for a, b in rules):
            return False
    return True


def solve(names: Sequence[str], rules) -> List[Optional[str]]:
    """The shortest crossing plan: what the farmer carries each time (None:
    crossing empty-handed)."""
    start: State = (NEAR, frozenset())
    goal: State = (FAR, frozenset(names))
    prev: Dict[State, Optional[Tuple[State, Optional[str]]]] = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == goal:
            break
        farmer, far = state
        here = far if farmer == FAR else frozenset(names) - far
        other = NEAR if farmer == FAR else FAR
        for load in [None] + sorted(here):
            moved = far
            if load is not None:
                moved = far - {load} if farmer == FAR else far | {load}
            nxt = (other, frozenset(moved))
            if nxt not in prev and safe(nxt, names, rules):
                prev[nxt] = (state, load)
                queue.append(nxt)
    plan, state = [], goal
    while prev.get(state):
        state, load = prev[state]
        plan.append(load)
    return plan[::-1]


def rule_blind_plan(names: Sequence[str]) -> List[Optional[str]]:
    """Across in the order they stand, coming back empty-handed in between."""
    plan: List[Optional[str]] = []
    for i, n in enumerate(names):
        plan.append(n)
        if i < len(names) - 1:
            plan.append(None)
    return plan


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("WK2")
    scene, homes = build_scene(level, seed)
    names = things(level)
    rules = RULES[level]
    across = Custom(lambda w: all_across(w, names), "all three across")
    resting = Custom(lambda w: across.holds(w) and all(AtRest(n).holds(w) for n in names),
                     "all three across and still")

    def count_far(w: WorldState) -> int:
        return sum(1 for v in banks(w, names).values() if v == FAR)

    milestones = (
        Milestone("first_across", Custom(lambda w: count_far(w) >= 1, "one across")),
        Milestone("two_across", Custom(lambda w: count_far(w) >= 2, "two across")),
        Milestone("all_across", across),
        Milestone("finished", resting),
    )
    plan = solve(names, rules)
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=INSTRUCTION[level],
        scorer=Scorer(task_id="WK2", level=level,
                      success=Finally(across) & HoldsFor(resting, 1.0),
                      milestones=milestones,
                      violations=(("nothing_eaten", RiverRules(names, rules)),)),
        milestones=milestones,
        reactor_factory=lambda n=names: Ferry(n),
        reference_factory=lambda inst, h=homes: FerrymanPolicy(names, rules, h),
        metadata={"knob": spec.knob.field, "knob_value": len(rules),
                  "crossings": len(plan), "plan": [p or "-" for p in plan],
                  "max_model_calls": 120},
    )


class FerrymanPolicy(ScriptedPolicy):
    """Work the plan out, then for each crossing: lift the passenger aboard
    and take the hand away (or, crossing alone, tap the raft's deck), wait for
    the raft to land, lift the passenger off into its lane on the other bank.
    ``ignore_rules`` (the stress agent) ferries them in the order they stand."""

    name = "ferryman"

    def __init__(self, names: Sequence[str], rules, homes, arm: str = RIGHT) -> None:
        self.names = tuple(names)
        self.rules = tuple(rules)
        self.homes = homes
        self.arm = arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._plan: Optional[List[Optional[str]]] = None
        self._jobs: List[tuple] = []
        self._sub = None

    def _carry(self, world: WorldState, load: str, target) -> Optional[List[Command]]:
        """Moves toward putting ``load`` at ``target`` (x, y, surface z);
        None once it is there."""
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PlaceAtPointPolicy

        if self._sub is None:
            self._sub = PlaceAtPointPolicy(load, target, arm=self.arm, tolerance=0.02,
                                           release_clearance=0.004, lift_height=0.10,
                                           carry_z=TABLE + HOVER)
        obj = world.obj(load)
        if not obj.is_held and math.dist(obj.pose.pos[:2], target[:2]) < 0.03 and \
                abs(obj.pose.pos[2] - obj.size[2] - target[2]) < 0.02:
            self._sub = None
            return None
        cmds = self._sub.commands(world, ctx=self._ctx)
        if self._sub.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"could not carry the {load}")]
        return [c for c in cmds if not isinstance(c, Done)] or [Wait(0.3)]

    def _tap(self, world: WorldState) -> List[Command]:
        """A closed hand down on the deck's free end, and up again."""
        raft = world.obj(RAFT).pose.pos
        x, y = raft[0], raft[1] + TAP_Y
        q = turned_down(TAP_YAW)
        up = Pose((x, y, TABLE + HOVER), q)
        return [SetGripper(self.arm, 1.0), MoveTo(self.arm, up),
                MoveTo(self.arm, Pose((x, y, DECK + FINGERTIP + 0.004), q)), Wait(0.3),
                MoveTo(self.arm, up)]

    def _away(self, world: WorldState) -> List[Command]:
        """The hand straight up from the raft, out of its way."""
        x, y, _ = world.arm(self.arm).eef_pose.pos
        return [MoveTo(self.arm, Pose((x, y, TABLE + HOVER), turned_down(0.0)))]

    def _seat(self, world: WorldState):
        raft = world.obj(RAFT)
        return (raft.pose.pos[0], raft.pose.pos[1] + SEAT_Y, DECK)

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        self._ctx = ctx
        if self._plan is None:
            self._plan = list(rule_blind_plan(self.names) if self.ignore_rules
                              else solve(self.names, self.rules))
        while True:
            if not self._jobs:
                if all_across(world, self.names):
                    return [Done("all across")]
                if not self._plan:
                    return [Done("the plan is used up")]
                load = self._plan.pop(0)
                here = raft_side(world)
                there = FAR if here == NEAR else NEAR
                self._jobs = ([("aboard", load), ("away",)] if load else [("tap",)]) \
                    + [("land", there)] + ([("ashore", load, there)] if load else [])
            kind, *args = self._jobs[0]
            if kind == "aboard":
                # anywhere on the deck will do: the raft sets off with it, and
                # a hand going back to straighten a goat that landed 4 cm off
                # its seat chased it across the river (Isaac, v9)
                if aboard(world, args[0]):
                    self._sub = None
                    cmds = None
                else:
                    cmds = self._carry(world, args[0], self._seat(world))
            elif kind in ("away", "tap"):
                self._jobs.pop(0)
                return self._away(world) if kind == "away" else self._tap(world)
            elif kind == "land":
                cmds = None if raft_side(world) == args[0] else [Wait(1.0)]
            else:
                x, y = self.homes[args[0]][args[1]]
                cmds = self._carry(world, args[0], (x, y, TABLE))
            if cmds is not None:
                return cmds
            self._jobs.pop(0)
