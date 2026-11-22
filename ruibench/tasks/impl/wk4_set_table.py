"""WK4 "set the table" -- World Knowledge, two arms.

Knob: how many places to set (1 -> 2), three pieces each.

The two-arm station at a dining table.  Easy: one linen placemat straight
ahead between the arms, its plate already on it, for a guest sitting across
the table from the robot.  Hard: two placemats side by side, for two guests.
Beside each arm, outboard, a white tray with a fork, a knife and a spoon
side by side: 12.5 to 14.5 cm cutlery with steel heads and walnut handles (v9;
a scanned wooden toy set until then, which the advisor found crude), and 13
cm porcelain plates.
"Set the table for the guest across from you." or "... for the two guests
across from you."  Nothing on the mats
says where anything goes (v3: the outlines used to be printed on them --
that is now the gap-filled control): the custom does -- from the guest's
side, the fork on the left of the plate, the knife on its right and the
spoon beyond the knife.  From the robot's side,
mirrored.  Which of two alike pieces goes where does not matter.  What is
judged is what the custom says -- which side of the plate each piece lies
on, and in what order -- not the centimetre: this is a long task and a
knowledge task, not a precise one.

Each arm reaches only its own half of the table.  So at the easy level the
custom decides the division of labour as well: the fork, on the guest's
left, is on the robot's right and only the right arm reaches it; the knife
and the spoon are the left arm's.  At the hard level each arm sets
the place on its own side.  Both trays hold a whole set, so where a piece
starts says nothing about where it goes.

v6 (2026-09-22): two arms, the advisor's fourth round (the flame test went
the other way, to one).  Until then the FR3 set one or two places from one
tray.  v10 (9/24), the advisor: no cups; and the arms go home as DI1's do,
level back in front of their bases first (``_common.to_rest``).  v12 (9/24),
the advisor: the left arm folded forward putting the knife down (it had been
sent past its wrist's reach: ``LIFT``), and the knife and fork were hard to
see from the front (wider heads; the trays' cutlery out from under the
folded grippers: ``CUTLERY_AT``).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import yaw_of
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines.base import BodySpec
from ...engines.robots import arm_base
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, HoldsFor, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import LEFT, RIGHT, TABLE, TWO_ARMS, Frame, ahead, station_scene, to_rest

#: the places at each level, as the head camera shows them, and where across
SEATS = {"easy": {"middle": 0.0}, "hard": {"left": 0.17, "right": -0.17}}
MAT_X, MAT_HALF = 0.37, (0.11, 0.155)
MAT_COLORS = {"middle": "fabric:red", "left": "fabric:red", "right": "fabric:blue"}
ON_MAT = TABLE + 2 * T.MAT_HALF_Z
#: where each piece goes, from the mat's middle, for a guest across the table
#: (facing -x, their left at -y): the fork left of the 13 cm plate, the knife
#: right of it and the spoon beyond the knife.  The spoon 6 cm beyond the
#: knife: 3.5 cm off, the open
#: fingers putting it down knocked the knife out of its place (Isaac).  The
#: plate 2 cm off the mat's middle, toward the fork, so that the spoon lies
#: on the mat: at 15.5 cm out it lay on the mat's edge and slid off (Isaac, v9)
PLACES = {"fork": (0.02, -0.115), "knife": (0.02, 0.075), "spoon": (0.02, 0.135)}
PLATE_AT = (0.02, -0.02)
#: where the reference puts a piece: this close to the place's middle
SLACK = {"fork": (0.022, 0.012), "knife": (0.022, 0.012), "spoon": (0.022, 0.012)}
ALIGNED = math.radians(25)
#: how a place is judged: by side and order (``side_of``).  A piece lies at a
#: place lying straight on its mat, level with its plate and clear of it -- its
#: handle's middle at most this far inside the rim -- and the spoon at least
#: this far beyond the knife.  Until v2.3.4 a piece had to lie within the
#: reference's ``SLACK`` of a point: agents that see only pixels laid the right
#: pieces on the right sides, 2-4 cm along and the spoon 4 cm beyond the knife,
#: not 6, and no place counted as set
RIM_IN, BEYOND = 0.01, 0.015
KINDS = ("fork", "knife", "spoon")
SIDES = {LEFT: "left", RIGHT: "right"}
#: every piece on the table: a whole set on each arm's tray
PIECES = {f"{kind}_{SIDES[arm]}": kind for arm in (LEFT, RIGHT) for kind in KINDS}
#: a piece of cutlery, 12.5 to 14.5 cm, lying flat with its head toward the
#: robot: the body's origin in the middle of its walnut handle (6 cm long,
#: 1.1 cm across), which the fingers close on -- a hidden square core under
#: it, 9 mm across (v9: the toy set's scans collided as their convex hulls,
#: and the fingers stopped a centimetre short of the fork's neck); the steel
#: 6.5 to 8.5 cm beyond it, resting on the table.  v12, the advisor: "the
#: knife and fork are hard to see from the front" -- the head camera, 65 cm
#: off, draws a centimetre in 4.5 pixels, and the heads were 1.4 to 2.4 cm
#: across.  Now the fork is 2.5 cm across its four tines, 3 mm apart; the
#: knife's blade 2 cm, 6.3 cm long; the spoon's bowl 2.8 by 4.4 cm
HANDLE_HALF, HANDLE_R = 0.03, 0.0055
CORE_HALF = 0.0045
STEEL, WOOD = "metal:aluminium", "wood:walnut"
STEEL_T = 0.0006                      # half the steel's thickness
#: the fork's tines: how many, how wide, the gap between two, how long
TINES, TINE_HALF, TINE_GAP, TINE_LENGTH = 4, 0.002, 0.003, 0.038
#: the knife's blade (straight, then its round tip) and the spoon's bowl
BLADE_HALF, BLADE_LENGTH, BLADE_TIP = 0.010, 0.050, 0.013
BOWL_HALF = (0.022, 0.014, 0.0035)
#: the plates: white porcelain, 13 cm across, a hidden floor where the well is
PLATE_R, PLATE_HZ = 0.064, 0.0075
#: Each arm's tray stands outboard of the places, along the arm's own side.
#: From the arm's base: ahead, and out (away from the middle).  The cutlery
#: in one row, side by side, pointing away, 7 cm apart -- an open ARX X5
#: finger reaches 5.2 cm from the grasp's middle, a handle 0.55 cm from its
#: neighbour's.  Which piece lies where is the seed's.  v12: the row lies
#: where the head camera sees past the folded grippers -- they hide the table
#: up to 24 cm ahead of the base, and their fingers, 6 and 19 cm out, to 34
#: cm (``research/v12_wk4_visible.py``); until then two pieces of three lay
#: under them.  So two heads lie between the fingers and one beyond the
#: outer, and the row as far ahead as the wrist lets the hand lift the outer
#: piece ``LIFT`` with its fingers down: its handle 47 cm from the base
#: (joint4 at -1.50); the middle piece 2 cm further, clear of the outer
#: finger, which in line with the others hid its bowl or its tines.  All of
#: it inside the head camera's picture, and clear of the hard level's mats
CUTLERY_AT = ((0.41, 0.09), (0.43, 0.16), (0.41, 0.23))
TRAY_X, TRAY_OUT = (0.28, 0.48), (0.065, 0.255)
STABLE_SECONDS = 1.5
#: How high the reference carries a piece, over its grasp and over its place,
#: and lifts its hand after letting go.  An ARX X5 with its fingers straight
#: down reaches only so high that far out: 45 cm from its base (the easy
#: knife) its wrist (joint4, -1.57 rad) runs out at 9 cm, and 10 cm up -- the
#: shared ``PlacePosePolicy.HOVER`` -- the left arm stood 2.5 s at the limit
#: 4 cm short of the knife's place, then lurched forward and down to it (v12,
#: the advisor: "folds forward as it puts the knife down"; the lift after
#: letting go stood 2.9 s the same way).  At 5 cm every place of both levels
#: is within the wrist's reach, the furthest (the hard level's right spoon, 47
#: cm out) at -1.53 (``research/v12_wk4_joints.py``)
LIFT = 0.05


#: what the head camera must show: the mats and both trays -- and a piece's
#: box, which reaches as far past its handle as its head does the other way
#: (the head camera sees the table to 55 cm ahead and more)
FRAME = Frame((0.15, 0.54), (-0.55, 0.55), 0.07)


def place_xy(seat_y: float, kind: str, mirrored: bool = False) -> Tuple[float, float]:
    """Where a piece goes at the place ``seat_y`` across; ``mirrored``: set
    from the robot's own side, as if the guest sat where the robot is -- the
    custom's usual slip."""
    dx, dy = PLACES[kind]
    if mirrored:
        dx, dy = 2 * PLATE_AT[0] - dx, 2 * PLATE_AT[1] - dy
    return ahead(MAT_X + dx, seat_y + dy)


def arm_for(xy: Tuple[float, float]) -> str:
    """The arm that reaches a point: the one on its side of the middle."""
    return LEFT if xy[1] > 0 else RIGHT


def tray_spots(arm: str, seed: int) -> Dict[str, Tuple[float, float]]:
    """Where each piece lies on an arm's tray: shuffled over its spots."""
    out = 1.0 if arm == LEFT else -1.0
    side = SIDES[arm]
    kinds = list(KINDS)
    SeedBundle(seed).stream(f"tray/{side}").shuffle(kinds)
    return {k: ahead(x, out * y, arm=side) for k, (x, y) in zip(kinds, CUTLERY_AT)}


def cutlery(name: str, kind: str, xy, on: float) -> BodySpec:
    """A fork, a knife or a spoon: a walnut handle with two brass rivets, a
    steel ferrule, a neck bending down to the head, and the head -- four
    tines, a blade, a bowl -- flat on the table."""
    from ...engines import parts as P

    h, r, t = HANDLE_HALF, HANDLE_R, STEEL_T
    low = -r + t                                        # the steel lying on the table
    parts = [P.box((0, 0, 0), (h, CORE_HALF, r), "grey", collides="hidden"),
             P.box((-h - 0.04, 0, low), (0.04, 0.008, t), "grey", collides="hidden"),
             P.lathe((0, 0, 0), "capsule", r, h - r, WOOD, P.turn_y(math.pi / 2))]
    parts += [P.cylinder((dx, 0, r - 0.0002), 0.0014, 0.0003, "metal:brass", collides=False)
              for dx in (-0.012, 0.012)]
    ferrule = -h - 0.002
    parts.append(P.cylinder((ferrule, 0, 0), r * 0.85, 0.003, STEEL, P.turn_y(math.pi / 2),
                            collides=False))
    # the neck: from the ferrule down to the table, 16 mm long
    slope = math.atan2(-low, 0.016)
    parts.append(P.box((ferrule - 0.011, 0, low / 2), (0.0085, 0.0028, t), STEEL,
                       P.turn_y(slope), collides=False))
    head = ferrule - 0.019                                # where the head starts
    if kind == "fork":
        pitch = 2 * TINE_HALF + TINE_GAP
        across = TINES * pitch - TINE_GAP
        parts.append(P.box((head - 0.005, 0, low), (0.005, across / 2, t), STEEL, collides=False))
        parts += [P.box((head - 0.01 - TINE_LENGTH / 2, (k - (TINES - 1) / 2) * pitch, low),
                        (TINE_LENGTH / 2, TINE_HALF, t), STEEL, collides=False)
                  for k in range(TINES)]
    elif kind == "knife":
        spine = 0.002                                     # the blade stands off the handle's line
        parts.append(P.box((head - BLADE_LENGTH / 2, spine, low), (BLADE_LENGTH / 2, BLADE_HALF, t),
                           STEEL, collides=False))
        parts.append(P.ellipsoid((head - BLADE_LENGTH, spine, low), (BLADE_TIP, BLADE_HALF, t),
                                 STEEL))
    else:
        bx, by, bz = BOWL_HALF
        parts.append(P.ellipsoid((head - bx, 0, low + bz - 0.001), BOWL_HALF, STEEL))
    size = P.half_extents(parts)
    return BodySpec(name, "compound", size, Pose((xy[0], xy[1], on + r)), category=kind,
                    color=STEEL, mass=0.03, attrs={"parts": tuple(parts), "footprint": "rect"})


def plate(name: str, xy) -> BodySpec:
    """A white porcelain plate: drawn round, a hidden floor where its well is."""
    from ...engines import parts as P

    r, hz = PLATE_R, PLATE_HZ
    well = -hz + 0.45 * 2 * hz
    parts = (P.cylinder((0, 0, (-hz + well) / 2), 0.6 * r, (well + hz) / 2, "grey", collides="hidden"),
             P.lathe((0, 0, 0), "plate", r, hz, "ceramic:white"))
    return BodySpec(name, "compound", (r, r, hz), Pose((xy[0], xy[1], ON_MAT + hz)),
                    category="plate", color="ceramic:white", fixed=True, mass=0.08,
                    attrs={"parts": parts, "graspable": False, "footprint": "round"})


def build_scene(level: str, seed: int):
    bodies = []
    top = TABLE + 2 * T.MAT_HALF_Z
    (x0, x1), (o0, o1) = TRAY_X, TRAY_OUT
    for arm in (LEFT, RIGHT):
        side = SIDES[arm]
        out = 1.0 if arm == LEFT else -1.0
        bodies.append(T.mat(f"tray_{side}", ahead((x0 + x1) / 2, out * (o0 + o1) / 2, arm=side),
                            ((x1 - x0) / 2, (o1 - o0) / 2), "plastic:white"))
        for kind, (x, y) in tray_spots(arm, seed).items():
            bodies.append(cutlery(f"{kind}_{side}", kind, (x, y), top))
    # the pieces in a fixed order: which spot each is on changes, not the set
    order = {n: i for i, n in enumerate(PIECES)}
    bodies.sort(key=lambda b: (b.name not in order, order.get(b.name, 0)))
    for seat, y in SEATS[level].items():
        bodies.append(T.mat(f"mat_{seat}", ahead(MAT_X, y), MAT_HALF, MAT_COLORS[seat]))
        bodies.append(plate(f"plate_{seat}", ahead(MAT_X + PLATE_AT[0], y + PLATE_AT[1])))
    return station_scene(f"wk4_set_table_{level}", bodies, two_arms=True, frame=FRAME)


def in_place(world: WorldState, name: str, seat_y: float, mirrored: bool = False) -> bool:
    obj = world.obj(name)
    kind = PIECES[name]
    if obj.is_held:
        return False
    x, y = place_xy(seat_y, kind, mirrored)
    sx, sy = SLACK[kind]
    if abs(obj.pose.pos[0] - x) > sx or abs(obj.pose.pos[1] - y) > sy:
        return False
    if abs(obj.pose.pos[2] - obj.size[2] - ON_MAT) > 0.01:
        return False
    off = yaw_of(obj.pose.quat) % math.pi
    return min(off, math.pi - off) <= ALIGNED


def plate_xy(seat_y: float) -> Tuple[float, float]:
    return ahead(MAT_X + PLATE_AT[0], seat_y + PLATE_AT[1])


def side_of(world: WorldState, name: str, seat_y: float) -> int:
    """The side of the place's plate a piece lies on, +1 or -1 across the
    table: lying straight on the place's mat, level with the plate and clear
    of it -- or 0 (held, on its edge, on the plate, off the mat)."""
    obj = world.obj(name)
    if obj.is_held or abs(obj.pose.pos[2] - obj.size[2] - ON_MAT) > 0.01:
        return 0
    off = yaw_of(obj.pose.quat) % math.pi
    if min(off, math.pi - off) > ALIGNED:
        return 0
    x, y = obj.pose.pos[0], obj.pose.pos[1]
    (mx, my), (px, py) = ahead(MAT_X, seat_y), plate_xy(seat_y)
    if abs(x - mx) > MAT_HALF[0] or abs(y - my) > MAT_HALF[1] or abs(x - px) > PLATE_R:
        return 0
    return 0 if abs(y - py) < PLATE_R - RIM_IN else (1 if y > py else -1)


def kind_side(seat_y: float, kind: str) -> int:
    """The side of the plate the custom gives a kind, for the guest across."""
    return 1 if place_xy(seat_y, kind)[1] > plate_xy(seat_y)[1] else -1


def set_for(world: WorldState, seat_y: float) -> bool:
    """The place set as the custom has it, for the guest across: a fork beside
    the plate on their left, a knife on their right and a spoon beyond it."""
    at = {n: side_of(world, n, seat_y) for n in PIECES}
    right = kind_side(seat_y, "knife")
    knives = [n for n, k in PIECES.items() if k == "knife" and at[n] == right]
    spoons = [n for n, k in PIECES.items() if k == "spoon" and at[n] == right]
    return (any(k == "fork" and at[n] == kind_side(seat_y, "fork") for n, k in PIECES.items())
            and any(right * (world.obj(s).pose.pos[1] - world.obj(k).pose.pos[1]) >= BEYOND
                    for k in knives for s in spoons))


def hedged(world: WorldState, seats: Sequence[float]) -> bool:
    """A piece beside a plate on the side the custom does not give its kind --
    a fork on the knife's side, a knife or a spoon on the fork's: a spare laid
    as the custom's usual slip has it hedged the bet, and the table still
    counted as set (v2.3)."""
    return any(side_of(world, n, y) == -kind_side(y, k) for n, k in PIECES.items() for y in seats)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("WK4")
    seats = SEATS[level]
    scene = build_scene(level, seed)
    done = Custom(lambda w: all(set_for(w, y) for y in seats.values())
                  and not hedged(w, tuple(seats.values())),
                  "the place set" if len(seats) == 1 else "both places set")
    resting = Custom(lambda w: all(AtRest(n).holds(w) for n in PIECES), "everything still")

    def laid(w: WorldState) -> List[str]:
        """The pieces beside a plate on the side the custom gives their kind."""
        return [n for n, k in PIECES.items()
                if any(side_of(w, n, y) == kind_side(y, k) for y in seats.values())]

    def placed(w: WorldState) -> int:
        """Of the places' pieces, how many are laid: a kind at a place once."""
        return sum(1 for y in seats.values() for kind in PLACES
                   if any(k == kind and side_of(w, n, y) == kind_side(y, kind)
                          for n, k in PIECES.items()))

    total = len(PLACES) * len(seats)
    milestones = (
        Milestone("first_piece", Custom(lambda w: placed(w) >= 1, "a piece in its place")),
        Milestone("both_hands", Custom(
            lambda w: {arm_for(w.obj(n).pose.pos[:2]) for n in laid(w)} == {LEFT, RIGHT},
            "pieces in their places on both sides")),
        Milestone("all_placed", Custom(lambda w: placed(w) >= total, f"{total} pieces placed")),
        Milestone("table_set", Custom(lambda w: done.holds(w) and resting.holds(w),
                                      "the table set and still")),
    )
    who = "the guest" if level == "easy" else "the two guests"
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction=f"Set the table for {who} across from you.",
        scorer=Scorer(task_id="WK4", level=level,
                      success=Finally(done) & HoldsFor(Custom(
                          lambda w: done.holds(w) and resting.holds(w), "set and still"),
                          STABLE_SECONDS),
                      milestones=milestones),
        milestones=milestones,
        reference_factory=lambda inst, s=tuple(seats.values()): SetTablePolicy(s),
        metadata={"knob": spec.knob.field, "knob_value": len(seats), "pieces_to_place": total,
                  "report_completion_ratio": True, "max_model_calls": 90},
        extra_metrics={"pieces_placed": lambda traj: placed(traj[-1]) if traj else 0},
    )


class SetTablePolicy(ScriptedPolicy):
    """Both hands at once, piece by piece: each takes the pieces whose places
    by the custom are on its side, one after another, from its own tray, and
    puts each down turned along its place; then both arms fold away.
    ``ignore_rules`` (the stress agent, and the blind control): the custom
    seen from the robot's own side, mirrored."""

    name = "set_table"

    def __init__(self, seats: Sequence[float]) -> None:
        self.seats = list(seats)
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._subs: Dict[Tuple[float, str], object] = {}
        self._given_up = set()
        self._rested = set()

    def _targets(self, arm: str) -> List[Tuple[float, str]]:
        return [(seat, kind) for seat in self.seats for kind in PLACES
                if arm_for(place_xy(seat, kind, self.ignore_rules)) == arm]

    @staticmethod
    def _policy(name: str, kind: str, point, arm: str):
        from ...policies.scripted import PlacePosePolicy

        base = arm_base(TWO_ARMS, SIDES[arm], TABLE)
        return PlacePosePolicy(name, point, 0.0, arm=arm, symmetry=math.pi, base=base,
                               tolerance=0.01, yaw_tolerance=math.radians(10), lift=LIFT)

    def _sub(self, world: WorldState, arm: str, seat: float, kind: str):
        key = (seat, kind)
        if key not in self._subs:
            mine = f"_{SIDES[arm]}"
            free = [n for n, k in PIECES.items() if k == kind and n not in self._given_up
                    and n.endswith(mine)
                    and not any(in_place(world, n, s, self.ignore_rules) for s in self.seats)]
            if not free:
                return None
            x, y = place_xy(seat, kind, self.ignore_rules)
            name = min(free, key=lambda n: math.dist(world.obj(n).pose.pos[:2], (x, y)))
            self._subs[key] = (name, self._policy(name, kind, (x, y, ON_MAT), arm))
        return self._subs[key]

    def _moves(self, world: WorldState, ctx: TurnContext, arm: str) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS

        for seat, kind in self._targets(arm):
            if any(in_place(world, n, seat, self.ignore_rules)
                   for n, k in PIECES.items() if k == kind):
                self._subs.pop((seat, kind), None)      # done with
                continue
            sub = self._sub(world, arm, seat, kind)
            if sub is None:
                continue
            name, policy = sub
            cmds = policy.commands(world, ctx)
            if policy.attempts > MAX_GRASP_ATTEMPTS:
                self._given_up.add(name)
                del self._subs[(seat, kind)]
                continue
            moves = [c for c in cmds if not isinstance(c, Done)]
            if moves:
                return moves
            del self._subs[(seat, kind)]            # it thinks it is done: look again
            return []
        return []

    def _finished(self, world: WorldState, arm: str) -> bool:
        """Every piece this arm places is in its place (or given up on)."""
        return all(any(in_place(world, n, seat, self.ignore_rules)
                       for n, k in PIECES.items() if k == kind)
                   or self._sub(world, arm, seat, kind) is None
                   for seat, kind in self._targets(arm))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        out: List[Command] = []
        for arm in (LEFT, RIGHT):
            moves = self._moves(world, ctx, arm)
            if moves:
                out += moves
            elif arm not in self._rested and self._finished(world, arm):
                # this hand is done: back to rest now, not after the other
                # hand (v7: the right arm, done after the fork, hung over the
                # table while the left one set three pieces)
                self._rested.add(arm)
                out += to_rest(world, [arm])
        if out:
            return out
        if len(self._rested) < 2:
            return []                                 # a hand is only now done: look again
        return [Done("the table is set")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        out: List[Command] = []
        taken = set()
        for arm in (LEFT, RIGHT):
            for seat, kind in self._targets(arm):
                x, y = place_xy(seat, kind, self.ignore_rules)
                mine = [n for n, k in PIECES.items() if k == kind and n.endswith(f"_{SIDES[arm]}")
                        and n not in taken]
                name = min(mine, key=lambda n: math.dist(world.obj(n).pose.pos[:2], (x, y)))
                taken.add(name)
                policy = self._policy(name, kind, (x, y, ON_MAT), arm)
                out += [c for c in policy.plan(world, ctx) if not isinstance(c, Done)]
        return out + [Done("planned")]
