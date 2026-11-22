"""WK3 "domino run" -- World Knowledge.

Knob: the route (a straight run to the bell -> around a corner, the bell
behind a toy house).

Five dominoes stand in a rack near the robot.  Across the table a pendulum
is held up on its frame; further on stands a little bell.  "Set up the
dominoes so the pendulum rings the bell.  It is let go when you say you are
done."  Nothing is painted (v3): where each domino goes is the agent's
design -- in front of the pendulum's ball, each close enough to the one
before to hit it (nearer than its height), square to the way the chain runs,
the last falling on the bell.  Easy: the bell straight ahead of the swing.
Hard: a toy house stands in the way and the bell is round the corner, so the
chain has to turn, a little at each domino.

The chain runs on the physics (Isaac): the bell rings when a falling domino
comes down on it.  On a backend without falling (the mock) the chain is played
out from the same geometry: a domino goes down if it stands where the one
before would hit it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import quat_mul, rotate, yaw_of
from ...core.types import Pose, WorldState
from ...engines.base import Engine
from ...events.actions import DriveJoint, SetColor, SetJointDrive
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import MAX_GRASP_ATTEMPTS, grasp_offset, turned_down
from ...scoring import Custom, Milestone, Scorer
from ...scoring.predicates import TrajectoryPredicate
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

DOMINOES = tuple(f"domino{i}" for i in range(1, 6))
RACK, BELL, HINGE, HOUSE = "rack", "bell", "pendulum_hinge", "house"
HEIGHT = 2 * T.DOMINO_HALF[2]
#: the reference's spacing, as a fraction of a domino's height: at 0.7 the
#: hand, turned for a domino in the corner, knocked over the one before it
#: (Isaac, hard seed 0)
GAP = 0.75
STEP = GAP * HEIGHT
#: where the pendulum's ball meets a domino standing in front of it; the chain
#: starts falling toward -y
STRIKE = ahead(0.30, 0.004)
#: the corner (hard): two straight, then a quarter turn of this radius in
#: three steps, toward +x
TURN_R = 3 * STEP / (math.pi / 2)
ROUTES = ("straight", "corner")
ROUTE = {"easy": "straight", "hard": "corner"}
RACK_AT, RACK_PITCH = (0.15, 0.0), 0.09
PIVOT_BEYOND, PIVOT_Z = 0.06, TABLE + 0.135
HELD_UP, LET_DOWN = 1.3, -0.75
SWING_DAMPING = 0.0
BELL_BEYOND = 0.045
TAKE = turned_down(0.0)
#: the hand over a domino on the straight: a quarter turn from how it took it,
#: this way round because in the corner the wrist then turns back toward
#: straight (turned the other way, the corner's last place wants the hand at
#: 180 degrees, where the wrist stops a few millimetres short -- 8 s a seed,
#: Isaac v16).  Except at the first place (``hand_yaw``)
HAND = turned_down(-math.pi / 2)
CARRY_Z = TABLE + 0.17
RELEASE = 0.002
FALLEN = math.radians(45)
UPRIGHT = math.radians(8)
LET_GO = "pendulum_let_go"
RANG = "bell_rang"
SHOW_SECONDS = 6.0
#: a domino hits the next if that one stands this far on (as a fraction of
#: its height), this far to the side at most, and turned at most this much
REACH = (0.2, 0.95)
SIDE = 0.015
TURN = math.radians(40)
#: how square to its place the reference wants a domino before it moves on.
#: Letting go turns one by about 6 degrees either way (Isaac), and at 6 the
#: reference took a domino up again and again and gave up (the RUIKit
#: collaborator, hard seed 1); the corner's chain still rang the bell with
#: dominoes 14-18 degrees off (v15, research/v15_wk3_yaw.py)
PLACED_YAW = math.radians(10)

FRAME = Frame((0.100, 0.470), (-0.325, 0.225), 0.16)


def route(level: str) -> List[Tuple[float, float, float]]:
    """The reference's five places (x, y, yaw), the first by the pendulum."""
    x0, y0 = STRIKE
    if ROUTE[level] == "straight":
        return [(x0, y0 - i * STEP, 0.0) for i in range(5)]
    cx, cy = x0 + TURN_R, y0 - STEP
    out = [(x0, y0, 0.0), (x0, y0 - STEP, 0.0)]
    for k in (1, 2, 3):
        phi = k * math.pi / 6
        out.append((cx - TURN_R * math.cos(phi), cy - TURN_R * math.sin(phi), phi))
    return out


def hand_yaw(places: Sequence[Tuple[float, float, float]], i: int) -> float:
    """How the hand is turned over place ``i``: as ``HAND`` says, but at the
    first place, by the pendulum, the other way round.  The wrist camera rides
    on the hand's robot side (v16), and turned as ``HAND`` it came down on the
    pendulum's frame there: the hand stopped 4.4 cm high and let the domino go,
    every seed (Isaac, v16).  Turned the other way it looks on down the chain,
    over the table."""
    return yaw_of(HAND) + places[i][2] + (math.pi if i == 0 else 0.0)


def bell_xy(level: str) -> Tuple[float, float]:
    x, y, yaw = route(level)[-1]
    heading = (math.sin(yaw), -math.cos(yaw))
    return (x + heading[0] * BELL_BEYOND, y + heading[1] * BELL_BEYOND)


#: the house: small (6.3 cm across, 6.5 cm tall) and on the straight line
#: 20 cm on from the pendulum, where a chain that did not turn would run.
#: The hand, turned across the chain in the corner, sweeps 10 cm either side
#: of it: where the house stood first (4.5 cm to the robot's side of the line,
#: 12 cm on), a finger came down on its roof, the hand stopped 3 cm high and
#: the domino slipped out (Isaac, hard seed 1); here the palm passes 2.5 cm
#: over the roof and the fingers stay 1.8 cm clear of it
HOUSE_SCALE = 0.75
HOUSE_ON = 0.205


def house_xy() -> Tuple[float, float]:
    """On the straight line, where a chain that did not turn would run."""
    x0, y0 = STRIKE
    return (x0, y0 - HOUSE_ON)


def build_scene(level: str):
    rack = T.domino_rack(RACK, ahead(*RACK_AT), len(DOMINOES), RACK_PITCH)
    bodies = [rack]
    for k, (name, color) in enumerate(zip(DOMINOES, T.DOMINO_COLORS)):
        xy, floor = T.rack_slot(rack, k, len(DOMINOES), RACK_PITCH)
        bodies.append(T.domino(name, xy, color, on=floor, yaw=math.pi / 2, grasp_yaw=0.0))
    hammer, hinge, frame = T.pendulum("", (STRIKE[0], STRIKE[1] + PIVOT_BEYOND), PIVOT_Z, HINGE,
                                      HELD_UP, LET_DOWN)
    bodies += [hammer] + frame + list(T.bell(BELL, bell_xy(level)))
    if ROUTE[level] == "corner":
        bodies.append(T.toy_house(HOUSE, house_xy(), "paint:red", scale=HOUSE_SCALE))
    return station_scene(f"wk3_dominoes_{level}", bodies, articulations=[hinge], frame=FRAME)


def tilt(world: WorldState, name: str) -> float:
    up = rotate(world.obj(name).pose.quat, (0.0, 0.0, 1.0))
    return math.acos(max(-1.0, min(1.0, up[2])))


def standing(world: WorldState, name: str) -> bool:
    obj = world.obj(name)
    return not obj.is_held and tilt(world, name) <= UPRIGHT


def _normal(yaw: float) -> Tuple[float, float]:
    """The way a domino turned ``yaw`` falls (either way along this)."""
    return (math.sin(yaw), -math.cos(yaw))


def chain(world: WorldState) -> Tuple[List[str], bool]:
    """Which dominoes the pendulum brings down, in order, and whether the last
    comes down on the bell -- from where they stand."""
    x0, y0 = STRIKE
    up = [n for n in DOMINOES if standing(world, n)]
    first = [n for n in up
             if math.dist(world.obj(n).pose.pos[:2], (x0, y0)) < 0.012
             and abs(_normal(yaw_of(world.obj(n).pose.quat))[1]) > math.cos(math.radians(30))]
    if not first:
        return [], False
    order, d = [first[0]], (0.0, -1.0)
    while True:
        cur = world.obj(order[-1]).pose.pos
        best = None
        for n in up:
            if n in order:
                continue
            p = world.obj(n).pose.pos
            v = (p[0] - cur[0], p[1] - cur[1])
            along, side = v[0] * d[0] + v[1] * d[1], abs(v[0] * d[1] - v[1] * d[0])
            nrm = _normal(yaw_of(world.obj(n).pose.quat))
            if (REACH[0] * HEIGHT <= along <= REACH[1] * HEIGHT and side <= SIDE
                    and abs(nrm[0] * d[0] + nrm[1] * d[1]) >= math.cos(TURN)):
                if best is None or along < best[0]:
                    best = (along, n, nrm)
        if best is None:
            break
        _, n, nrm = best
        sign = 1.0 if nrm[0] * d[0] + nrm[1] * d[1] >= 0 else -1.0
        d = (sign * nrm[0], sign * nrm[1])
        order.append(n)
    cur = world.obj(order[-1]).pose.pos
    b = world.obj(BELL).pose.pos
    v = (b[0] - cur[0], b[1] - cur[1])
    along, side = v[0] * d[0] + v[1] * d[1], abs(v[0] * d[1] - v[1] * d[0])
    return order, (0.3 * HEIGHT <= along <= 1.05 * HEIGHT and side <= 0.02)


def on_bell(world: WorldState, name: str) -> bool:
    """A falling domino come down on the bell: tipped, its upper half over it."""
    if tilt(world, name) < math.radians(25):
        return False
    obj = world.obj(name)
    base = obj.pose.pos
    top = rotate(obj.pose.quat, (0.0, 0.0, T.DOMINO_HALF[2]))
    b = world.obj(BELL).pose.pos
    return any(math.dist((base[0] + t * top[0], base[1] + t * top[1]), b[:2]) < 0.022
               for t in (0.4, 0.6, 0.8, 1.0))


def released_at(traj) -> Optional[int]:
    return next((i for i, w in enumerate(traj) if LET_GO in w.events_fired), None)


class RangAfterLetGo(TrajectoryPredicate):
    """The pendulum let go with no domino in the hand and none already lying on
    the bell, then the bell rung (v2.3: a domino leant on the bell rang it at
    the let-go, with no chain)."""

    def evaluate(self, traj) -> bool:
        r = released_at(traj)
        if r is None:
            return False
        if any(traj[r].obj(n).is_held or on_bell(traj[r], n) for n in DOMINOES):
            return False
        return RANG in traj[-1].events_fired

    def describe(self) -> str:
        return "the pendulum let go, the bell rung by the dominoes"


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("WK3")
    scene = build_scene(level)
    places = route(level)

    def out_of_rack(w: WorldState) -> int:
        rack_x = ahead(*RACK_AT)[0]
        return sum(standing(w, n) and abs(w.obj(n).pose.pos[0] - rack_x) > 0.04 for n in DOMINOES)

    def down(w: WorldState) -> int:
        if LET_GO not in w.events_fired:
            return 0
        return sum(tilt(w, n) > FALLEN for n in DOMINOES)

    milestones = (
        Milestone("first_in_front", Custom(lambda w: bool(chain(w)[0]),
                                           "a domino standing in front of the pendulum")),
        Milestone("all_set_up", Custom(lambda w: out_of_rack(w) == len(DOMINOES),
                                       "all five standing out of the rack")),
        Milestone("chain_half_way", Custom(lambda w: down(w) >= 3,
                                           "the pendulum gone, three dominoes down")),
        Milestone("bell_rang", Custom(lambda w: RANG in w.events_fired, "the bell rung")),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction=("Set up the dominoes so the pendulum rings the bell. "
                     "It is let go when you say you are done."),
        scorer=Scorer(task_id="WK3", level=level, success=RangAfterLetGo(), milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda: DominoShow(),
        reference_factory=lambda inst, p=tuple(places): DominoPolicy(p),
        metadata={"knob": spec.knob.field, "knob_value": ROUTE[level],
                  # hidden from the agent: a layout that works (the gap-filled control)
                  "reference_places": [tuple(round(v, 4) for v in p) for p in places],
                  "gap_to_height": GAP, "max_model_calls": 60},
        extra_metrics={"fallen": lambda traj: _fallen(traj)},
    )


def _fallen(traj) -> int:
    r = released_at(traj)
    if r is None:
        return 0
    return sum(any(tilt(w, n) > FALLEN for w in traj[r:]) for n in DOMINOES)


class DominoShow(Reactor):
    """Lets the pendulum go when the agent is done, and rings the bell when
    a domino comes down on it.  Without falling (the mock) the chain is
    played out here from where the dominoes stand (``chain``)."""

    finale_seconds = SHOW_SECONDS
    STEP = 0.3

    def __init__(self) -> None:
        super().__init__()
        self._t0: Optional[float] = None
        self._chain: List[str] = []
        self._reaches = False
        self._dirs: Dict[str, Tuple[float, float]] = {}
        self._rung = False
        #: the dominoes standing at the let-go: only one of them falling rings the bell
        self._standing: Tuple[str, ...] = ()

    def finale(self, engine: Engine, world: WorldState) -> None:
        # a slow push only leans the first domino on the second; it is the
        # swing that knocks it over
        self.act(engine, world, LET_GO, SetJointDrive(HINGE, 0.0, SWING_DAMPING))
        self.act(engine, world, f"{LET_GO}_target", DriveJoint(HINGE, LET_DOWN))
        self._t0 = world.sim_time
        self._standing = tuple(n for n in DOMINOES if standing(world, n))
        if not engine.capabilities.dynamics:
            self._chain, self._reaches = chain(world)

    def tick(self, engine: Engine, world: WorldState) -> None:
        if self._t0 is None:
            return
        if not engine.capabilities.dynamics:
            due = int((world.sim_time - self._t0) / self.STEP)
            for name in self._chain[:due]:
                if tilt(world, name) < FALLEN:
                    engine.teleport(name, _leaning(world.obj(name).pose))
            hit = due >= len(self._chain) + 1 and self._reaches and bool(self._chain)
        else:
            hit = any(on_bell(world, n) for n in self._standing)
        if not self._rung and hit:
            self._rung = True
            self.act(engine, world, RANG, SetColor(f"{BELL}_light", T.BELL_RINGING))
            self.say(world, BELL, "rang", "Ding!")


def _leaning(pose: Pose) -> Pose:
    """A domino gone over 60 degrees about its own x axis, onto the next."""
    angle = math.radians(60)
    hx, hy, hz = T.DOMINO_HALF
    turn = (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)
    q = quat_mul(pose.quat, turn)
    centre = rotate(q, (0.0, 0.0, hz))
    foot = rotate(pose.quat, (0.0, -hy, 0.0))
    return Pose((pose.pos[0] + foot[0] + centre[0], pose.pos[1] + foot[1] + centre[1],
                 TABLE + hy * math.sin(angle) + hz * math.cos(angle)), q)


class DominoPolicy(ScriptedPolicy):
    """From the rack to its places, the last (by the bell) first; then
    done.  ``ignore_rules`` (the blind control): no feel for how a chain
    falls -- the dominoes spread evenly from the pendulum to the bell's side,
    a domino and a half apart, straight."""

    name = "dominoes"

    def __init__(self, places: Sequence[Tuple[float, float, float]], arm: str = RIGHT) -> None:
        self.places = list(places)
        self.arm = arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._tries: Dict[str, int] = {}

    def _targets(self) -> List[Tuple[float, float, float]]:
        if not self.ignore_rules:
            return self.places
        x0, y0 = STRIKE
        return [(x0, y0 - i * 1.5 * HEIGHT, 0.0) for i in range(len(DOMINOES))]

    def _at_place(self, world: WorldState, name: str, place) -> bool:
        obj = world.obj(name)
        if not standing(world, name):
            return False
        off = (yaw_of(obj.pose.quat) - place[2]) % math.pi
        return (math.dist(obj.pose.pos[:2], place[:2]) < 0.004
                and min(off, math.pi - off) < PLACED_YAW)

    def _take(self, world: WorldState, name: str) -> List[Command]:
        obj = world.obj(name)
        x, y, z = obj.pose.pos
        take = TAKE if abs(math.sin(yaw_of(obj.pose.quat))) > 0.5 else HAND
        at = lambda pz: MoveTo(self.arm, Pose((x, y, pz), take))
        return [SetGripper(self.arm, 0.0), at(CARRY_Z), at(z + grasp_offset(obj)),
                SetGripper(self.arm, 1.0), at(CARRY_Z)]

    def _drop(self, world: WorldState, name: str, places, i: int) -> List[Command]:
        """Over its place (``places[i]``), turned to its yaw, with where the
        domino hangs in the fingers taken off: the Franka's fingers held it
        2 mm off and 3 degrees turned, and a domino put down by the hand alone
        missed."""
        place = places[i]
        hand, obj = world.arm(self.arm).eef_pose, world.obj(name)
        wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi
        err = (yaw_of(obj.pose.quat) - place[2]) % math.pi
        now, want = yaw_of(hand.quat), hand_yaw(places, i)
        # of the ways to turn it square, the one that ends nearest the hand's
        # turn for this place (``hand_yaw``): turned the other way, the wrist
        # camera comes down on the pendulum's frame (Isaac, 2026-09-18 and v16)
        turn = min((-err + k * math.pi for k in (-1, 0, 1, 2)),
                   key=lambda t: abs(wrap(now + t - want)))
        dx, dy = obj.pose.pos[0] - hand.pos[0], obj.pose.pos[1] - hand.pos[1]
        c, s = math.cos(turn), math.sin(turn)
        x, y = place[0] - (c * dx - s * dy), place[1] - (s * dx + c * dy)
        z = TABLE + T.DOMINO_HALF[2] + RELEASE + hand.pos[2] - obj.pose.pos[2]
        q = turned_down(yaw_of(hand.quat) + turn)
        return [MoveTo(self.arm, Pose((x, y, CARRY_Z), q)), MoveTo(self.arm, Pose((x, y, z), q)),
                Wait(0.3), SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose((x, y, CARRY_Z), q)),
                Wait(0.5)]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        targets = self._targets()
        held = next((n for n in DOMINOES if world.obj(n).is_held), None)
        if held is not None:
            return self._drop(world, held, targets, DOMINOES.index(held))
        # from the bell's end back: a domino that tips back as it is put down
        # (it did, turned in the corner) then has nothing behind it to knock
        # over -- set up from the pendulum's end, it took the chain down (Isaac)
        for name, place in reversed(list(zip(DOMINOES, targets))):
            if self._at_place(world, name, place):
                continue
            self._tries[name] = self._tries.get(name, 0) + 1
            if self._tries[name] > MAX_GRASP_ATTEMPTS:
                return [Done(f"could not stand {name} in its place")]
            return self._take(world, name)
        return [Done("ready")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """The whole setup from the first picture: nothing moves until the
        pendulum goes.  A domino in the rack faces the front and is taken with
        the hand straight; turning the hand by the place's yaw less a quarter
        (the other way round at the first place: ``hand_yaw``) stands it
        square to the route there."""
        out: List[Command] = []
        places = self._targets()
        for i, (name, place) in reversed(list(enumerate(zip(DOMINOES, places)))):
            x, y, z = world.obj(name).pose.pos
            grip = z + grasp_offset(world.obj(name))
            down = TABLE + T.DOMINO_HALF[2] + RELEASE + grasp_offset(world.obj(name))
            hand = turned_down(hand_yaw(places, i))
            at = lambda px, py, pz, q: MoveTo(self.arm, Pose((px, py, pz), q))
            out += [SetGripper(self.arm, 0.0), at(x, y, CARRY_Z, TAKE), at(x, y, grip, TAKE),
                    SetGripper(self.arm, 1.0), at(x, y, CARRY_Z, TAKE),
                    at(place[0], place[1], CARRY_Z, hand), at(place[0], place[1], down, hand), Wait(0.3),
                    SetGripper(self.arm, 0.0), at(place[0], place[1], CARRY_Z, hand), Wait(0.5)]
        return out + [Done("planned")]
