"""WK1 "Tower of Hanoi" -- World Knowledge.

Knob: the game (three rings, the largest already home: 3 moves -> four rings
from a game in progress: 7 to 11 moves).

A wooden stand with three pegs and turned wooden rings.  The tower goes to the
right peg -- right as the fixed camera shows it.  The instruction only names
the game; its rules are the game's own and they are the scorer's: one ring at
a time, only a top ring, rings put down only on pegs, never a ring on a
smaller one -- checked from where the rings are, not only from what a gripper
is detected to hold: a ring that rides on top of a held one is carried too
(v2.3: an agent lifted two rings as one, and only one counted as held).
Easy: three rings, the largest already on the right peg, so two
move (3 moves).  Hard (v3): four rings, and the game starts part way through
-- the rings spread over the pegs as a legal game can leave them, drawn per
seed from every position 7 to 11 moves from the end.  A solution from the
standard start can be recited; from the middle of a game it has to be worked
out.

The rings are held from outside, across their width -- the largest is 6.2
cm across, the Franka opens 8 -- and threaded with 6 mm to spare around the
peg.  The pegs end in a ball, which guides a ring that is a little off.  The
Franka's fingers reach only 3.8 cm below its palm, so the pegs stand 3.8 cm
tall and the rings are 9 mm thick, four to a peg: a ring taken off the bottom
of a peg leaves the palm 6 mm above the ball (v3).

v9 (9/24), the advisor: WK1 is to be a showcase, and it looked a cartoon --
painted rings of sixteen flat blocks on a knife-edged board.  The rings are
turned discs, on an ebonised stand with a rounded-over edge and round-topped
dowels.  In four natural woods (v9) they were too plain; they are stained
(v10), the grain showing through: red, yellow, green and blue from the
smallest, a rainbow from the bottom of the full tower.
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import (
    AtRest, Custom, Finally, HoldsFor, LegalMoveMachine, Location, Milestone, NotT, Scorer,
)
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

#: smallest first; the easy game has the first three
ALL_RINGS = ("ring_small", "ring_medium", "ring_large", "ring_largest")
RING_COUNT = {"easy": 3, "hard": 4}
OUTER = {"ring_small": 0.019, "ring_medium": 0.023, "ring_large": 0.027, "ring_largest": 0.031}
#: a wooden Tower of Hanoi (v9: the advisor, for a showcase task, found red,
#: yellow, blue and green painted rings cartoonish): turned discs, rounded at
#: the edge, dark to light as they get smaller -- walnut, cherry, oak, ash --
#: on an ash stand with ash pegs
COLORS = {"ring_small": "wood:stain_red", "ring_medium": "wood:stain_yellow",
          "ring_large": "wood:stain_green", "ring_largest": "wood:stain_blue"}
INNER = 0.011
RING_HALF = 0.0045
#: the grasp's tool centre point above a ring's centre: the fingertips close
#: on the ring's upper 6 mm and stay 3 mm clear of what it stands on
GRASP_HEIGHT = 0.0065
#: the hard game's start: this far from the end, in moves
HARD_MOVES = (7, 11)

STAND = "stand"
STAND_FINISH = "wood:ebony"
BASE_HALF = (0.045, 0.165, 0.006)
PEG_R, PEG_HALF = 0.005, 0.0165
PEG_SPACING = 0.12
PEGS = ("left", "middle", "right")          # along +y: left to right in the fixed camera
GOAL = 2
#: the stand's height, table to ball top; its origin is half way up
STAND_HEIGHT = 2 * BASE_HALF[2] + 2 * PEG_HALF + PEG_R
BASE_TOP = TABLE + 2 * BASE_HALF[2]
PEG_TOP = BASE_TOP + 2 * PEG_HALF + PEG_R
#: a ring this close to a peg's axis is on that peg
ON_PEG = 0.012
STAND_AT = (0.26, 0.0)
STAND_JITTER = 0.01
STABLE_SECONDS = 2.0

State = Tuple[int, ...]            # the peg of each ring, smallest first


#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.18, 0.345), (-0.19, 0.19), 0.06)


def ring_body(name: str, xy, z: float) -> BodySpec:
    """A disc: drawn round (``parts``: "washer"), what collides a hidden ring
    of sixteen blocks, as until v9 it was drawn."""
    parts = [q[:5] + ("hidden",) for q in
             P.ring((0.0, 0.0, 0.0), OUTER[name], INNER, RING_HALF, "grey", segments=16)]
    parts.append(P.lathe((0.0, 0.0, 0.0), "washer", OUTER[name], RING_HALF, COLORS[name],
                         inner=INNER))
    return BodySpec(name, "compound", P.half_extents(parts), Pose((xy[0], xy[1], z)),
                    category="ring", color=COLORS[name], mass=0.03,
                    attrs={"parts": tuple(parts), "footprint": "round",
                           "grasp_height": GRASP_HEIGHT})


def stand_body(xy) -> BodySpec:
    lift = -STAND_HEIGHT / 2
    base = (0.0, 0.0, lift + BASE_HALF[2])
    parts = [P.box(base, BASE_HALF, STAND_FINISH, collides="hidden"),
             P.slab(base, BASE_HALF, STAND_FINISH)]
    peg_half = (2 * PEG_HALF + PEG_R) / 2
    for k in range(3):
        y = (k - 1) * PEG_SPACING
        # a round-topped dowel drawn over the peg's colliders
        parts.append(P.cylinder((0.0, y, lift + 2 * BASE_HALF[2] + PEG_HALF), PEG_R, PEG_HALF,
                                STAND_FINISH, collides="hidden"))
        parts.append(P.sphere((0.0, y, lift + 2 * BASE_HALF[2] + 2 * PEG_HALF), PEG_R,
                              STAND_FINISH, collides="hidden"))
        parts.append(P.lathe((0.0, y, lift + 2 * BASE_HALF[2] + peg_half), "dowel", PEG_R, peg_half,
                             STAND_FINISH))
    return BodySpec(STAND, "compound", P.half_extents(parts),
                    Pose((xy[0], xy[1], TABLE + STAND_HEIGHT / 2)), category="stand",
                    color=STAND_FINISH, fixed=True, attrs={"parts": tuple(parts), "graspable": False})


def peg_xy(stand, k: int) -> Tuple[float, float]:
    return stand[0], stand[1] + (k - 1) * PEG_SPACING


def rings_for(level: str) -> Tuple[str, ...]:
    return ALL_RINGS[:RING_COUNT[level]]


def mid_games() -> Tuple[State, ...]:
    """Every four-ring position ``HARD_MOVES`` from the end (any position
    with each peg's rings largest-down is one a legal game can reach)."""
    from itertools import product

    goal = (GOAL,) * 4
    lo, hi = HARD_MOVES
    return tuple(s for s in product(range(3), repeat=4) if lo <= len(shortest_plan(s, goal)) <= hi)


def layout(level: str, seed: int) -> State:
    """Where the rings start: easy, the top two on the left or the middle peg
    and the largest home; hard, a game in progress, drawn per seed."""
    if level == "easy":
        start = SeedBundle(seed).stream("start").choice((0, 1))
        return (start, start, GOAL)
    games = list(mid_games())
    return SeedBundle(seed).stream("start").choice(games)


def build_scene(level: str, seed: int):
    stand = SeedBundle(seed).jitter("stand", ahead(*STAND_AT), STAND_JITTER)
    start = layout(level, seed)
    rings = rings_for(level)
    bodies = [stand_body(stand)]
    height = {k: BASE_TOP for k in range(3)}
    for i in sorted(range(len(rings)), reverse=True):             # largest first
        k = start[i]
        bodies.append(ring_body(rings[i], peg_xy(stand, k), height[k] + RING_HALF))
        height[k] += 2 * RING_HALF
    return station_scene(f"wk1_hanoi_{level}", bodies, frame=FRAME), stand, start


def peg_of(world: WorldState, name: str, stand) -> Optional[int]:
    obj = world.obj(name)
    for k in range(3):
        if math.dist(obj.pose.pos[:2], peg_xy(stand, k)) <= ON_PEG:
            return k
    return None


def state_of(world: WorldState, stand, rings) -> Optional[State]:
    """The peg each ring is on -- None while one is held or off the pegs."""
    out = []
    for name in rings:
        k = None if world.obj(name).is_held else peg_of(world, name, stand)
        if k is None:
            return None
        out.append(k)
    return tuple(out)


def legal_moves(state: State):
    for p in range(3):
        mine = [i for i, q in enumerate(state) if q == p]
        if not mine:
            continue
        top = min(mine)                         # the smallest ring there is on top
        for q in range(3):
            if q != p and all(i > top for i, r in enumerate(state) if r == q):
                yield top, q


def shortest_plan(start: State, goal: State) -> List[Tuple[int, int]]:
    """Moves (ring index, to peg) by breadth-first search."""
    prev: Dict[State, Optional[tuple]] = {start: None}
    queue = deque([start])
    while queue:
        s = queue.popleft()
        if s == goal:
            break
        for ring, q in legal_moves(s):
            n = tuple(q if i == ring else r for i, r in enumerate(s))
            if n not in prev:
                prev[n] = (s, (ring, q))
                queue.append(n)
    plan, s = [], goal
    while prev.get(s):
        s, move = prev[s]
        plan.append(move)
    return plan[::-1]


def rule_blind_plan(state: State, goal: int) -> List[Tuple[int, int]]:
    """The tower moved as if the game had no rules: the top ring of the first
    pile that is not the goal, straight onto the goal peg, until none is left."""
    state, out = list(state), []
    while any(k != goal for k in state):
        pile = min(k for k in state if k != goal)
        ring = min(i for i, k in enumerate(state) if k == pile)
        out.append((ring, goal))
        state[ring] = goal
    return out


def threaded_on(world: WorldState, name: str, stand) -> Optional[int]:
    """The peg a ring is threaded on -- around its stem, not lifted clear of its
    top -- held or not; None while it is off every peg."""
    k = peg_of(world, name, stand)
    return k if k is not None and world.obj(name).pose.pos[2] - RING_HALF < PEG_TOP else None


def rules(stand, rings) -> Tuple[Tuple[str, NotT], ...]:
    """The game's rules from where the rings are, whatever is held: one ring off
    the pegs at a time, a ring left lying is on a peg, and none rests on a
    smaller one.  Each must last a moment to count, so a ring passing a peg's
    top or a wobble at the moment of a grasp does not."""
    size = {name: i for i, name in enumerate(rings)}            # smallest first

    def off(w: WorldState) -> List[str]:
        return [n for n in rings if threaded_on(w, n, stand) is None]

    def on_smaller(w: WorldState) -> bool:
        for k in range(3):
            pile = sorted((w.obj(n).pose.pos[2], size[n]) for n in rings
                          if not w.obj(n).is_held and threaded_on(w, n, stand) == k)
            if any(upper > lower for (_, lower), (_, upper) in zip(pile, pile[1:])):
                return True
        return False

    return (
        ("one_ring_at_a_time", NotT(HoldsFor(Custom(lambda w: len(off(w)) > 1,
                                                    "two rings off the pegs"), 0.5))),
        ("rings_put_down_on_pegs", NotT(HoldsFor(Custom(
            lambda w: any(AtRest(n).holds(w) for n in off(w)), "a ring lying off the pegs"),
            1.0))),
        ("never_on_a_smaller_ring", NotT(HoldsFor(Custom(on_smaller, "a ring on a smaller one"),
                                                  0.5))),
    )


def machine(stand, rings) -> LegalMoveMachine:
    return LegalMoveMachine(
        locations=tuple(Location(f"peg_{PEGS[k]}", (*peg_xy(stand, k), BASE_TOP), ON_PEG,
                                 capacity=len(rings)) for k in range(3)),
        size_order={name: i for i, name in enumerate(rings)},
        allow_offboard=False, pieces=rings)


def tower_on(stand, k: int, rings) -> Custom:
    """Every ring on peg ``k``, largest at the bottom, none held."""
    def holds(w: WorldState) -> bool:
        if any(w.obj(n).is_held or peg_of(w, n, stand) != k for n in rings):
            return False
        heights = [w.obj(n).pose.pos[2] for n in rings]            # smallest first
        return all(a > b for a, b in zip(heights, heights[1:]))
    return Custom(holds, f"the tower on the {PEGS[k]} peg")


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("WK1")
    scene, stand, start = build_scene(level, seed)
    rings = rings_for(level)
    goal: State = (GOAL,) * len(rings)
    plan = shortest_plan(start, goal)
    movers = [n for n, k in zip(rings, start) if k != GOAL]
    tower = tower_on(stand, GOAL, rings)
    still = Custom(lambda w: tower.holds(w) and all(AtRest(n).holds(w) for n in rings),
                   "the tower standing still")

    def state_is(test, label: str) -> Custom:
        def holds(w: WorldState) -> bool:
            state = state_of(w, stand, rings)
            return state is not None and test(state)
        return Custom(holds, label)

    first, last = rings.index(movers[0]), rings.index(movers[-1])
    milestones = (
        Milestone("first_move", state_is(lambda st: st != start, "a ring moved")),
        Milestone("smallest_moved", state_is(lambda st: st[first] != start[first],
                                             f"{movers[0]} moved")),
        Milestone("largest_placed", state_is(lambda st: st[last] == GOAL,
                                             f"{movers[-1]} on the right peg")),
        Milestone("tower_built", still),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Solve the Tower of Hanoi: move the tower to the peg on the right, "
                    "as the front camera shows it.",
        scorer=Scorer(task_id="WK1", level=level,
                      success=Finally(tower) & HoldsFor(still, STABLE_SECONDS),
                      milestones=milestones, violations=rules(stand, rings),
                      machine=machine(stand, rings)),
        milestones=milestones,
        reference_factory=lambda inst, s=stand, r=rings: HanoiPolicy(s, r),
        metadata={"knob": spec.knob.field, "knob_value": len(plan),
                  "rings": len(rings), "start": {n: PEGS[k] for n, k in zip(rings, start)},
                  "goal": PEGS[GOAL], "minimum_moves": len(plan),
                  "right_means": "as the fixed camera shows it (+y)",
                  "stand": tuple(stand), "max_model_calls": 120},
    )


class HanoiPolicy(ScriptedPolicy):
    """Read which peg each ring is on, take the first move of a shortest legal
    plan from there, repeat -- a ring that landed wrong is recovered from, not
    compounded.  A move: up off the peg, across above the peg tops, down the
    target peg until the ring's underside is just above the pile, let go.

    ``ignore_rules`` plays the rule-blind game instead (the stress agent)."""

    name = "hanoi"
    #: tool centre point above the table while a ring travels: its underside
    #: then clears the ball tops by 3 cm
    CARRY = TABLE + 0.13
    RELEASE = 0.004
    SETTLE = 0.5

    def __init__(self, stand, rings=ALL_RINGS[:3], arm: str = RIGHT) -> None:
        self.stand = tuple(stand)
        self.rings = tuple(rings)
        self.arm = arm
        self.ignore_rules = False
        self._moves: List[Tuple[int, int]] = []
        self._current: Optional[Tuple[int, int]] = None
        self._last_state: Optional[State] = None
        self.attempts = 0

    def reset(self) -> None:
        self._moves, self._current, self._last_state, self.attempts = [], None, None, 0

    def _next_move(self, state: State) -> Optional[Tuple[int, int]]:
        goal = (GOAL,) * len(self.rings)
        if self.ignore_rules:
            if not self._moves and self._current is None:
                self._moves = rule_blind_plan(state, GOAL)
            return self._moves.pop(0) if self._moves else None
        plan = shortest_plan(state, goal)
        return plan[0] if plan else None

    def _pile_top(self, world: WorldState, k: int, moving: str) -> float:
        on = [n for n in self.rings if n != moving and not world.obj(n).is_held
              and peg_of(world, n, self.stand) == k]
        return max([world.obj(n).pose.pos[2] + RING_HALF for n in on], default=BASE_TOP)

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, grasp_pose, retreat_pose

        if ctx.extras.get("executor_busy"):
            return []
        held = next((n for n in self.rings if world.obj(n).is_held), None)
        state = state_of(world, self.stand, self.rings)
        if state is not None:
            self._last_state = state
        if held is None:
            if state is None:
                return [Done("a ring is off the pegs")]
            if state == (GOAL,) * len(self.rings):
                return [Done("tower built")]
            if self._current is None or state[self._current[0]] == self._current[1]:
                self._current = self._next_move(state)
                self.attempts = 0
                if self._current is None:
                    return [Done("no moves left")]
            ring = self.rings[self._current[0]]
            self.attempts += 1
            if self.attempts > MAX_GRASP_ATTEMPTS:
                return [Done(f"could not move {ring}")]
            grasp = grasp_pose(world.obj(ring))
            above = Pose((grasp.pos[0], grasp.pos[1], self.CARRY), GRASP_DOWN)
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above), MoveTo(self.arm, grasp),
                    SetGripper(self.arm, 1.0), MoveTo(self.arm, above)]

        # carrying a ring: over the target peg, then down it
        if not self._current or self.rings[self._current[0]] != held:
            # the wrong ring came up: it goes back where it was
            if self._last_state is None:
                return [Done(f"holding {held} without knowing where it came from")]
            i = self.rings.index(held)
            self._current = (i, self._last_state[i])
        k = self._current[1]
        hand, obj = world.arm(self.arm).eef_pose, world.obj(held)
        px, py = peg_xy(self.stand, k)
        # where the ring hangs in the fingers is taken off at the peg
        x, y = px - (obj.pose.pos[0] - hand.pos[0]), py - (obj.pose.pos[1] - hand.pos[1])
        z = (self._pile_top(world, k, held) + RING_HALF + self.RELEASE
             + hand.pos[2] - obj.pose.pos[2])
        low = Pose((x, y, z), GRASP_DOWN)
        if hand.distance_to(low) < 0.003:
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, retreat_pose(world.arm(self.arm))),
                    Wait(1.0)]
        over = Pose((x, y, max(self.CARRY, hand.pos[2])), GRASP_DOWN)
        return [MoveTo(self.arm, Pose((hand.pos[0], hand.pos[1], over.pos[2]), GRASP_DOWN)),
                MoveTo(self.arm, over), MoveTo(self.arm, low), Wait(self.SETTLE)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """The whole game from the first frame: a static puzzle can be."""
        state = state_of(world, self.stand, self.rings)
        moves = (rule_blind_plan(state, GOAL) if self.ignore_rules
                 else shortest_plan(state, (GOAL,) * len(self.rings)))
        piles = {k: [i for i in sorted(range(len(self.rings)), reverse=True) if state[i] == k]
                 for k in range(3)}
        out: List[Command] = []
        for ring, k in moves:
            src = next(p for p, rings in piles.items() if ring in rings)
            sx, sy = peg_xy(self.stand, src)
            z_from = BASE_TOP + (piles[src].index(ring) * 2 + 1) * RING_HALF + GRASP_HEIGHT
            piles[src].remove(ring)
            tx, ty = peg_xy(self.stand, k)
            z_to = BASE_TOP + (len(piles[k]) * 2 + 1) * RING_HALF + self.RELEASE + GRASP_HEIGHT
            piles[k].append(ring)
            out += [SetGripper(self.arm, 0.0),
                    MoveTo(self.arm, Pose((sx, sy, self.CARRY), GRASP_DOWN)),
                    MoveTo(self.arm, Pose((sx, sy, z_from), GRASP_DOWN)),
                    SetGripper(self.arm, 1.0),
                    MoveTo(self.arm, Pose((sx, sy, self.CARRY), GRASP_DOWN)),
                    MoveTo(self.arm, Pose((tx, ty, self.CARRY), GRASP_DOWN)),
                    MoveTo(self.arm, Pose((tx, ty, z_to), GRASP_DOWN)),
                    SetGripper(self.arm, 0.0),
                    MoveTo(self.arm, Pose((tx, ty, self.CARRY), GRASP_DOWN))]
        return out + [Wait(2.5), Done("planned")]
