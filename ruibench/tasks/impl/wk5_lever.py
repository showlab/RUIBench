"""WK5 "the balance beam" -- World Knowledge.

Knob: the problem on the beam (one weight to place against one -> two
weights to place against one, in two pans).

A balance beam on a stand, with four little pans along each arm, one, two,
three and four steps from the pivot.  On the left something is already in a
pan; on a wooden tray in front lies what may be put on the right.  "Make the
balance level, using everything on the tray.  Leave what is on it already."
(v2.3.4: "using what is on the tray" did not say all of it, and the hard
level balances with the double weight alone at step three: eight of Claude
Code's thirteen hard episodes balanced it so, and failed.)
Each weight from the tray may be set down once: one let go in a pan and
picked up again is a foul (10/4 -- the pan has to be worked out, not found by
trying; the hint says so, as a person at the beam would be told).

Nothing says how a lever works.  Easy: a single weight at step two on the
left, a double one on the tray -- it has to go at step one, where a weight
twice as heavy at half the distance pulls the same.  Put where it looks
right, opposite at step two, it takes the beam down on the right.  Hard: a
double weight at step three on the left, a double and a single on the tray,
both to be used, one to a pan: the double at step one and the single at step
four (2 x 1 + 1 x 4 = 2 x 3) is the only way with both.

Siegler (1976) used this apparatus to chart children's rules: the youngest
judge by weight alone, older ones add distance only when the weights are
equal, and it is years later that they trade weight against distance --
which is what the hard level asks for.

The beam shows which way it leans; each pan rides a slide straight up and
down, lifted by as much as its step's end of the beam would rise, carrying
what lies in it.  What the beam does is worked out from what is in the pans.
It starts where the weight already on it puts it, leaning to the left: level
at the start, it showed a balance that needed nothing in the first picture
and only leaned once the clock ran (four agents on Isaac took it as done;
the RUIKit collaborator, bench-issue #18).

v4 (2026-09-21), replacing "park the bus", which had nothing behind it.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...core.types import Pose, WorldState
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, HoldsFor, Milestone, Never, Scorer, TrajectoryPredicate
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

BEAM, HINGE, TRAY = "beam", "beam_hinge", "tray"
#: the event when a weight from the tray, let go in a pan, is in the hand again
TAKEN_BACK = "weight_taken_back"
STEPS = (1, 2, 3, 4)
STEP = 0.058                         # from one pan to the next, along the beam
PIVOT_AT = (0.42, 0.0)
PIVOT_Z = TABLE + 0.10
PAN_R, PAN_UP = 0.022, 0.022         # a pan, and its floor above the pivot, level
TILT = math.radians(7.0)
TURN_RATE = 0.25                     # rad/s
UNIT = 0.02                          # kg, one weight
#: per level: what is on the left already (its step, how many units), and the
#: weights on the tray (how many units each)
PROBLEM = {"easy": ((2, 1), (2,)), "hard": ((3, 2), (2, 1))}
TRAY_AT, TRAY_HALF = (0.18, 0.0), (0.09, 0.05)
FRAME = Frame((0.08, 0.50), (-0.30, 0.30), 0.18)
LEVEL_SECONDS = 2.0


def pan_name(side: int, step: int) -> str:
    return f"pan_{'L' if side < 0 else 'R'}{step}"


def slide_name(side: int, step: int) -> str:
    return f"{pan_name(side, step)}_slide"


def pivot() -> Tuple[float, float, float]:
    x, y = ahead(*PIVOT_AT)
    return (x, y, PIVOT_Z)


def pan_floor(side: int, step: int, lift: float = 0.0) -> Tuple[float, float, float]:
    x, y, z = pivot()
    return (x, y + side * step * STEP, z + PAN_UP + 0.002 + lift)


def weight_body(name: str, xy, on: float, units: int, graspable: bool = True) -> BodySpec:
    """One or two brass disc weights, stacked, as one body: round, their
    edges chamfered, so that two show a groove between them (v8, 9/24: the
    advisor could not see that the double weight was two -- eight-sided
    ones, stacked, read as one block).  Each over a hidden eight-sided core
    whose flats the fingers close on (``_toys.octagon_core``)."""
    w, h = T.COIN_HALF
    parts = []
    for k in range(units):
        z = (2 * k + 1 - units) * h
        parts += list(T.octagon_core((w, h * 0.98), z))
        parts.append(P.lathe((0, 0, z), "chamfer", T.ROUND_OVER * w, h * 0.98, "metal:brass"))
    size = P.half_extents(parts, solid_only=False)
    return BodySpec(name, "compound", size, Pose((xy[0], xy[1], on + size[2])),
                    category="weight" if units == 1 else "double weight", color="metal:brass",
                    mass=UNIT * units, attrs={"parts": tuple(parts), "footprint": "round",
                                              "units": units, "graspable": graspable})


def start_tilt(level: str) -> float:
    """The beam's angle at the start: leaning to the side the weight already
    on it pulls (positive: the left down), as it would have come to rest."""
    (step, units), _ = PROBLEM[level]
    pull = step * units                       # left minus right, in units x steps
    return TILT if pull > 0 else (-TILT if pull < 0 else 0.0)


def pan_lift(side: int, step: int, tilt: float) -> float:
    """How far a pan rides up (down, if negative) with the beam at ``tilt``."""
    return side * step * STEP * math.sin(tilt)


def balance(tilt: float = 0.0) -> Tuple[List[BodySpec], List[ArticulationSpec]]:
    x, y, z = pivot()
    arm = STEPS[-1] * STEP + 0.02
    # v8 (9/24): a wooden beam with its steps marked, brass pans on brass
    # stems, a brass post -- until then a red beam and blue plastic pans
    beam_parts = [P.box((0, 0, 0), (0.008, arm, 0.005), "wood:ash"),
                  P.box((0, 0, 0.022), (0.0012, 0.0012, 0.022), "paint:black", collides=False),
                  P.sphere((0, 0, 0.045), 0.0026, "paint:black", collides=False)]
    beam_parts += [P.box((0, s * k * STEP, 0.0055), (0.009, 0.0012, 0.0006), "paint:black",
                         collides=False) for s in (-1, 1) for k in STEPS]
    beam = BodySpec(BEAM, "compound", P.half_extents(beam_parts, solid_only=False),
                    Pose((x, y, z)), category="balance beam", color="wood:ash", mass=0.1,
                    attrs={"parts": tuple(beam_parts), "graspable": False, "carried_by": HINGE})
    joints = [ArticulationSpec(HINGE, "hinge", axis=(1.0, 0.0, 0.0),
                               limits=(-TILT - 0.02, TILT + 0.02), initial=tilt, actuated=True,
                               parent=BEAM, carries=False)]
    bodies = [beam]
    travel = STEPS[-1] * STEP * math.sin(TILT) + 0.005
    for side in (-1, 1):
        for k in STEPS:
            rim = [q[:5] + ("hidden",) for q in
                   P.ring((0, 0, 0.003), PAN_R, PAN_R - 0.003, 0.0015, "metal:brass", segments=16)]
            parts = [P.cylinder((0, 0, 0), PAN_R, 0.002, "metal:brass", collides="hidden"), *rim,
                     P.lathe((0, 0, 0.0017), "pan", PAN_R, 0.0037, "metal:brass"),
                     P.cylinder((0, 0, -(PAN_UP - 0.006) / 2 - 0.002), 0.003,
                                (PAN_UP - 0.006) / 2, "metal:brass", collides=False)]
            fx, fy, fz = pan_floor(side, k)
            bodies.append(BodySpec(pan_name(side, k), "compound", P.half_extents(parts),
                                   Pose((fx, fy, fz - 0.002)), category="balance pan",
                                   color="metal:brass", mass=0.02,
                                   attrs={"parts": tuple(parts), "graspable": False,
                                          "carried_by": slide_name(side, k), "container": "open",
                                          "wall": 0.003}))
            joints.append(ArticulationSpec(slide_name(side, k), "slide", axis=(0.0, 0.0, 1.0),
                                           limits=(-travel, travel),
                                           initial=pan_lift(side, k, tilt), actuated=True,
                                           parent=pan_name(side, k), carries=True))
    top = PIVOT_Z - 0.008 - TABLE
    stand = [P.box((0, 0, -top / 2 + 0.006), (0.05, 0.07, 0.006), "wood:oak"),
             P.cylinder((0, 0, -top / 2 + 0.013), 0.016, 0.001, "metal:brass", collides=False),
             P.cylinder((0, 0, 0.006), 0.007, top / 2 - 0.006, "metal:brass")]
    bodies.append(BodySpec("stand", "compound", (0.05, 0.07, top / 2), Pose((x, y, TABLE + top / 2)),
                           category="balance", color="metal:brass", fixed=True,
                           attrs={"parts": tuple(stand), "graspable": False}))
    return bodies, joints


def build_scene(level: str):
    (step, units), loose = PROBLEM[level]
    tilt = start_tilt(level)
    bodies, joints = balance(tilt)
    fx, fy, fz = pan_floor(-1, step, pan_lift(-1, step, tilt))
    # not one to pick up: it is to be left where it is
    bodies.append(weight_body("left_weight", (fx, fy), fz, units, graspable=False))
    bodies.append(T.mat(TRAY, ahead(*TRAY_AT), TRAY_HALF, "wood:ash"))
    tx, ty = ahead(*TRAY_AT)
    for k, u in enumerate(loose):
        # one behind the other: side by side across, the open fingers reach the next
        bodies.append(weight_body(f"weight{k + 1}", (tx + (k - (len(loose) - 1) / 2) * 0.09, ty),
                                  TABLE + 2 * T.MAT_HALF_Z, u))
    return station_scene(f"wk5_lever_{level}", bodies, frame=FRAME, articulations=joints)


def weights(level: str) -> List[str]:
    return ["left_weight"] + [f"weight{k + 1}" for k in range(len(PROBLEM[level][1]))]


def solution(level: str) -> Dict[str, int]:
    """The step on the right for each weight on the tray, one to a pan, that
    balances the left: the only one (the levels are chosen so)."""
    from itertools import permutations

    (step, units), loose = PROBLEM[level]
    names = weights(level)[1:]
    for steps in permutations(STEPS, len(loose)):
        if sum(u * k for u, k in zip(loose, steps)) == step * units:
            return dict(zip(names, steps))
    raise ValueError(f"no way to balance {level}")                # pragma: no cover


def lift_of(w: WorldState, side: int, step: int) -> float:
    return w.articulations.get(slide_name(side, step), 0.0)


def placed(w: WorldState, names: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """Which pan each weight lies in: name -> (side, step)."""
    held = {a.holding for r in w.robots.values() for a in r.arms.values()}
    out = {}
    for n in names:
        if n in held:
            continue
        p = w.obj(n).pose.pos
        for side in (-1, 1):
            for k in STEPS:
                cx, cy, cz = pan_floor(side, k, lift_of(w, side, k))
                if math.hypot(p[0] - cx, p[1] - cy) < PAN_R and -0.01 < p[2] - cz < 0.05:
                    out[n] = (side, k)
    return out


class OnePanEach(TrajectoryPredicate):
    """A weight from the tray, once lying in a pan, is never found in another:
    one let go in the wrong pan could be slid over the low rim into the next
    without a hand taking it up, and the beam answered each try (v2.3)."""

    def __init__(self, names: Sequence[str]) -> None:
        self.names = list(names)

    def evaluate(self, traj) -> bool:
        first: Dict[str, Tuple[int, int]] = {}
        for w in traj:
            for n, pan in placed(w, self.names).items():
                if n != self.names[0] and first.setdefault(n, pan) != pan:
                    return False
        return True

    def describe(self) -> str:
        return "each weight from the tray in one pan only"


def turning(w: WorldState, names: Sequence[str]) -> float:
    """Units x steps: positive when the left (-y) side pulls harder."""
    return sum(-side * k * w.obj(n).attrs.get("units", 1)
               for n, (side, k) in placed(w, names).items())


class Beam(Reactor):
    """Leans the beam toward the side that pulls harder, turning at
    ``TURN_RATE``; every pan goes up or down with its end of the beam."""

    def __init__(self, names: Sequence[str], angle: float = 0.0) -> None:
        super().__init__()
        self.names = list(names)
        self._angle, self._t = angle, 0.0
        #: the tray's weights let go in a pan so far, and whether one has been
        #: picked up again
        self._down: set = set()
        self._taken_back = False

    def tick(self, engine: Engine, world: WorldState) -> None:
        self._down.update(n for n in placed(world, self.names) if n != self.names[0])
        if not self._taken_back:
            held = {a.holding for r in world.robots.values() for a in r.arms.values()}
            back = sorted(self._down & held)
            if back:
                self._taken_back = True
                self.note(engine, world, TAKEN_BACK, back[0])
        pull = turning(world, self.names)
        target = TILT if pull > 1e-6 else (-TILT if pull < -1e-6 else 0.0)
        step = TURN_RATE * max(world.sim_time - self._t, 0.0)
        self._t = world.sim_time
        self._angle += max(-step, min(step, target - self._angle))
        engine.set_articulation_target(HINGE, self._angle)
        # a positive angle takes the +y end up
        for side in (-1, 1):
            for k in STEPS:
                engine.set_articulation_target(slide_name(side, k),
                                               pan_lift(side, k, self._angle))


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("WK5")
    scene = build_scene(level)
    names = weights(level)
    loose = names[1:]
    (step, units), _ = PROBLEM[level]
    home = (-1, step)

    def level_now(w: WorldState) -> bool:
        where = placed(w, names)
        return (all(n in where and where[n][0] == 1 for n in loose)
                and where.get("left_weight") == home and turning(w, names) == 0
                and abs(w.articulations.get(HINGE, 0.0)) < 0.01)

    still = Custom(lambda w: all(AtRest(n).holds(w) for n in names), "nothing moving")
    level_c = Custom(level_now, "the beam level, with the tray's weights on the right")
    milestones = (
        Milestone("a_weight_on_the_beam", Custom(lambda w: any(n in placed(w, names) for n in loose),
                                                 "a weight from the tray in a pan")),
        Milestone("on_the_right", Custom(lambda w: all(placed(w, names).get(n, (0,))[0] == 1
                                                       for n in loose), "all of them on the right")),
        Milestone("level", level_c),
    )
    moved = Custom(lambda w: w.obj("left_weight").is_held
                   or placed(w, names).get("left_weight") != home,
                   "the weight already on the beam taken off or moved")
    again = Custom(lambda w: TAKEN_BACK in w.events_fired,
                   "a weight let go in a pan picked up again")
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Make the balance level, using everything on the tray. "
                    "Leave what is on it already.",
        scorer=Scorer(task_id="WK5", level=level,
                      success=Finally(level_c) & HoldsFor(level_c & still, LEVEL_SECONDS),
                      milestones=milestones,
                      violations=(("leave_what_is_on_it", Never(moved)),
                                  ("each_weight_set_down_once", Never(again)),
                                  ("each_weight_in_one_pan", OnePanEach(names)))),
        milestones=milestones,
        reactor_factory=lambda n=names, a=start_tilt(level): Beam(n, a),
        reference_factory=lambda inst, l=level: LeverPolicy(l),
        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level],
                  # hidden from the agent: where it goes (the gap-filled control)
                  "answer": {n: f"step {k} on the right" for n, k in solution(level).items()},
                  "max_model_calls": 60},
    )


class LeverPolicy(ScriptedPolicy):
    """Works out where the weights on the tray balance the left side -- steps
    times weight the same on both, one to a pan -- and puts them there, one
    at a time, letting the beam come to rest before aiming at the next pan.
    ``ignore_rules`` (the blind control): goes by how it looks -- every weight
    opposite the one on the left, at its step, and the next pan out when that
    one is taken."""

    name = "lever"

    def __init__(self, level: str, arm: str = RIGHT) -> None:
        self.level, self.arm = level, arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._sub = None
        self._current: Optional[str] = None
        self._since = 0.0

    def _plan(self) -> Dict[str, int]:
        if not self.ignore_rules:
            return solution(self.level)
        (step, _), _ = PROBLEM[self.level]
        names = weights(self.level)[1:]
        return {n: min(STEPS[-1], step + k) for k, n in enumerate(names)}

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PlaceAtPointPolicy

        if ctx.extras.get("executor_busy"):
            return []
        plan = self._plan()
        where = placed(world, weights(self.level))
        todo = [n for n, k in plan.items() if where.get(n) != (1, k)]
        if not todo:
            return [Wait(2.5), Done("level")]
        if self._current != todo[0]:
            self._current, self._sub, self._since = todo[0], None, world.sim_time
        if self._sub is None:
            if world.sim_time - self._since < 1.5:
                return [Wait(1.5)]          # let the beam come to rest before aiming at a pan
            k = plan[self._current]
            fx, fy, fz = pan_floor(1, k, lift_of(world, 1, k))
            self._sub = PlaceAtPointPolicy(self._current, (fx, fy, fz), arm=self.arm,
                                           tolerance=0.012, release_clearance=0.006,
                                           carry_z=TABLE + 0.24)
        if self._sub.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"could not place {self._current}")]
        return [c for c in self._sub.commands(world, ctx) if not isinstance(c, Done)] \
            or [Wait(0.3)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """The whole job from the first picture: the beam's lean before each
        placement is known from what is on it, so every pan's height is too."""
        from ...policies.scripted import _pick_and_place_sequence

        (step, units), loose = PROBLEM[self.level]
        pull = step * units                     # left minus right, in units x steps
        out: List[Command] = [Wait(1.5)]
        for name, k in self._plan().items():
            lean = TILT if pull > 0 else (-TILT if pull < 0 else 0.0)
            fx, fy, fz = pan_floor(1, k, k * STEP * math.sin(lean))
            out += _pick_and_place_sequence(self.arm, world.obj(name), (fx, fy, fz), 0.0,
                                            0.24, 0.006)
            out.append(Wait(2.0))                # the beam turns to its new lean
            pull -= world.obj(name).attrs.get("units", 1) * k
        return out + [Wait(1.0), Done("level")]
