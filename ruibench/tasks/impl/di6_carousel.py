"""DI6 "the dessert covers" -- Dynamic Interaction, the shell game.

Knob: how the turntable turns (once, slowly -> three times, quickly, back
and forth).

Three identical white porcelain dessert covers stand on a walnut turntable,
and a little raspberry tart is under the one nearest the robot.  "The cake
is under the cover nearest you.  Watch the turntable, and when it stops,
find the cake and put it on the plate."  The table turns a third or two
thirds of the way round, one way or the other; where the cake ends up does
not exist until it stops, and only following the cover gives it.

Call (2001) studied object permanence in orangutans, chimpanzees and
children of 19 and 26 months by moving a reward among three containers:
all three species followed a visible move, and every kind of move but the
hardest (invisible and to two containers apart).  The turntable is the
visible kind, made harder by speed and by turning back.

Lifting a cover before the table has stopped is a violation, and so is
lifting a second cover: the one lift has to be the right one.

v4 (2026-09-21), replacing "find the ball" (a ball rolling about under
still cups): the advisor found it not intuitive.  Cups that swap places need
two degrees of freedom each, which a joint here does not have; a turntable
moves them all with one, and the shell game stays a shell game.  v8 (9/24),
the advisor: the brass star (the coin) had been seen too often, and upturned
cups with handles are not a common sight -- make it little cakes, and make
it pretty.  So a cake under dessert covers (cloches), and a dessert plate.
Each cover is drawn round (``parts``: "dome") over a hidden ring it stands
on and a hidden block in its knob, which the fingers close on; the tart, a
scan, over a hidden eight-sided core the height of its pastry case.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...events.actions import SetJointDrive
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import Custom, Finally, In, Milestone, Never, Scorer, TrajectoryPredicate, Upright
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ...engines.isaac.assets import get as catalogue
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

CUPS = 3
TURNTABLE, HINGE, CAKE, DISH = "turntable", "turntable_hinge", "cake", "plate"
#: per level: how many turns, and how long a third of the way round takes, in
#: T_ref.  v13, provisional (T_ref 8 s): 14 s at the easy level (0.15 rad/s) --
#: a model taking T_ref a turn sees a picture every T_ref, and at 0.6 rad/s a
#: third of a turn was over between two of them, the covers all alike, so no
#: look could tell a third from two thirds -- and 1.75 s at the hard level
#: (1.2 rad/s), three times, which only a faster loop follows
SPIN = {"easy": (1, 1.75), "hard": (3, 0.218)}
#: the table starts turning this many T_ref in, and never sooner than
#: STARTS_AT seconds (v2.3.4): an agent that reads the task, looks and gets
#: ready -- three replies -- sees the cake's cover where it stands before
#: the table moves.  At 3 s the hard level's turns were over before any first
#: look (16 s), and the easy turn half done: the cover was guessed, not followed
STARTS_IN, STARTS_AT, PAUSE = 3.0, 3.0, 0.8
#: the turntable's drive: (stiffness, damping) -- about a third of a second to
#: follow a new setting, smoothly
SOFT = (40.0, 12.0)
SETTLE_END = 1.6
STOPPED = "table_stopped"
#: a cover: a white porcelain dome 9 cm across and 7.2 cm tall, a gold band
#: round its foot, a gold knob on a stem -- room under it for the tart, 6.4
#: cm across and 5 cm tall to the top of its raspberries (under a 6.2 cm
#: dome the raspberries came through its top, Isaac 9/24)
COVER_R, COVER_H = 0.045, 0.072
KNOB_R, STEM_H = 0.011, 0.005
COVER_HALF_Z = (COVER_H + STEM_H + 2 * KNOB_R) / 2
#: the hidden ring it stands on: 12 blocks just inside its foot, 2 cm tall
RING_IN, RING_H = 0.039, 0.02
COVER_MASS = 0.08
#: the tart: the scan's size, and the hidden core of its pastry case -- the
#: fingers close on its flats, 2.8 cm out, the pastry 3 cm out -- taken at
#: the case's rim: lower, the hand hid 62% of the tart from the fixed camera
TART = "raspberry_tart"
TART_FALLBACK = (0.0316, 0.032, 0.025)
TART_CORE = (0.028, 0.0125)
TART_GRASP = -0.004

TABLE_AT = (0.34, 0.0)
DISC_R, DISC_HZ = 0.14, 0.006
DISC_UP = 0.012                        # the disc's underside above the table, on its hub
RING_R = 0.085                         # where the cups stand, from the middle
DISH_AT = (0.12, -0.28)
#: the dessert plate: white porcelain, 15 cm across, 2 cm tall, a hidden
#: floor where its well is (``parts._plate_profile``)
PLATE_R, PLATE_HZ = 0.075, 0.010
MOVED = 0.03
CARRY_Z = TABLE + 0.26
FRAME = Frame((0.05, 0.50), (-0.36, 0.26), 0.13)


#: a cover tipped further than this from upright counts as lifted
TIPPED = 0.35


class OneCoverEver(TrajectoryPredicate):
    """At most one cover lifted in the whole episode, whatever was put back: a
    cover lifted, looked under and put down again, then the next, is a search,
    not a choice (v2.3; before, only covers off at the same moment counted)."""

    def __init__(self, lifted) -> None:
        self.lifted = lifted

    def evaluate(self, traj) -> bool:
        seen = set()
        for w in traj:
            seen.update(self.lifted(w))
            if len(seen) > 1:
                return False
        return True

    def describe(self) -> str:
        return "one cover lifted in the whole episode"


def cup_name(i: int) -> str:
    return f"cover{i + 1}"


def disc_top() -> float:
    return TABLE + DISC_UP + 2 * DISC_HZ


def home_angle(i: int) -> float:
    """Cover i's place round the ring at the start: cover 1 nearest the robot."""
    return math.pi + i * 2 * math.pi / CUPS


def ring_xy(angle: float) -> Tuple[float, float]:
    cx, cy = ahead(*TABLE_AT)
    return (cx + RING_R * math.cos(angle), cy + RING_R * math.sin(angle))


def turn_start(t_ref: float) -> float:
    """When the table starts turning (``STARTS_IN``)."""
    return max(STARTS_AT, STARTS_IN * t_ref)


def spin_rate(level: str, t_ref: float) -> float:
    """How fast the table turns (rad/s): a third of the way round in its level's
    share of T_ref."""
    return (2 * math.pi / CUPS) / (SPIN[level][1] * t_ref)


def turns(level: str, seed: int) -> Tuple[float, ...]:
    """Each turn: plus or minus a third or two thirds of the way round,
    alternating in direction at the hard level, never ending where it began."""
    n, _ = SPIN[level]
    rng = SeedBundle(seed).stream("spin")
    third = 2 * math.pi / CUPS
    for _ in range(50):
        sign = rng.choice((1, -1))
        out = []
        for k in range(n):
            out.append((sign if k % 2 == 0 else -sign) * third * rng.choice((1, 2)))
        if round(sum(out) / third) % CUPS:
            return tuple(out)
    return tuple(out)                         # pragma: no cover


def turntable() -> Tuple[List[BodySpec], ArticulationSpec]:
    cx, cy = ahead(*TABLE_AT)
    # a walnut lazy susan with a raised oak rim (until v5, 9/22: a bare disc
    # with three white spokes painted across it, which looked like cardboard)
    parts = [P.cylinder((0, 0, 0), DISC_R, DISC_HZ, "wood:walnut")]
    parts += [p[:5] + (False,) for p in P.ring((0, 0, DISC_HZ + 0.0015), DISC_R, DISC_R - 0.008,
                                                 0.0015, "wood:oak", segments=40)]
    disc = BodySpec(TURNTABLE, "compound", (DISC_R, DISC_R, DISC_HZ),
                    Pose((cx, cy, TABLE + DISC_UP + DISC_HZ)), category="turntable",
                    color="wood:walnut", mass=0.3,
                    attrs={"parts": tuple(parts), "graspable": False, "carried_by": HINGE,
                           "driven": True})
    # 3 mm short of the disc: touching it, the fixed hub dragged on the turning
    # disc, which went round in fits and starts and threw the cups (Isaac, 9/21)
    hub_h = DISC_UP - 0.003
    hub = BodySpec("hub", "cylinder", (0.03, 0.03, hub_h / 2), Pose((cx, cy, TABLE + hub_h / 2)),
                   category="stand", color="metal:steel", fixed=True, attrs={"graspable": False})
    # within a turn either way: PhysX takes a revolute joint's limits in (-2 pi, 2 pi),
    # and the most the table ever turns from where it started is two thirds of a turn
    hinge = ArticulationSpec(HINGE, "hinge", axis=(0.0, 0.0, 1.0),
                             limits=(-2 * math.pi + 0.05, 2 * math.pi - 0.05),
                             initial=0.0, actuated=True, parent=TURNTABLE, carries=True)
    return [disc, hub], hinge


def cover(name: str, xy, on: float) -> BodySpec:
    """A dessert cover (a cloche): drawn round over a hidden ring of twelve
    blocks it stands on, lifted by the knob -- a hidden block inside it, as
    the knob of v5's cups had (on a round knob a cover carried by it hung
    tilted about the fingers' line, Isaac, DS3, 9/22)."""
    hz = COVER_HALF_Z
    foot = -hz
    apex = foot + COVER_H
    knob = (0, 0, apex + STEM_H + KNOB_R)
    parts = [q[:5] + ("hidden",) for q in P.ring((0, 0, foot + RING_H / 2), COVER_R - 0.001, RING_IN,
                                                 RING_H / 2, "grey", segments=12)]
    parts += [P.lathe((0, 0, foot + COVER_H / 2), "dome", COVER_R, COVER_H / 2, "ceramic:white"),
              P.lathe((0, 0, foot + 0.006), "cylinder", COVER_R + 0.0006, 0.0014, "metal:gold"),
              P.cylinder((0, 0, apex + STEM_H / 2 - 0.001), 0.0035, STEM_H / 2 + 0.001, "metal:gold",
                         collides=False),
              P.grip(knob, (KNOB_R * 0.8, KNOB_R * 0.8, KNOB_R)),
              P.sphere(knob, KNOB_R, "metal:gold", collides=False)]
    return BodySpec(name, "compound", (COVER_R, COVER_R, hz), Pose((xy[0], xy[1], on + hz)),
                    category="dessert cover", color="ceramic:white", mass=COVER_MASS,
                    attrs={"parts": tuple(parts), "footprint": "round", "container": "inverted",
                           # the drawn dome hides what is under it (``sight_boxes``)
                           "opaque": True,
                           # by the knob, not round the dome
                           "grasp_height": hz - KNOB_R})


def tart(name: str, xy, on: float) -> BodySpec:
    """The little cake: the scanned raspberry tart over a hidden core the
    height of its pastry case, taken by the case."""
    half = tuple(catalogue(TART).half_extents or TART_FALLBACK)
    core = T.octagon_core(TART_CORE, z=-half[2] + TART_CORE[1])
    parts = core + (P.visual(TART, (0, 0, 0), half),)
    return BodySpec(name, "compound", half, Pose((xy[0], xy[1], on + half[2])),
                    category="cake", color="cream", mass=0.05,
                    attrs={"parts": parts, "footprint": "round", "grasp_height": TART_GRASP,
                           "hidden_at_start": True})


def plate(name: str, xy) -> BodySpec:
    """A white porcelain dessert plate: drawn round, a hidden floor where its
    well is."""
    r, hz = PLATE_R, PLATE_HZ
    well = -hz + 0.45 * 2 * hz
    parts = (P.cylinder((0, 0, (-hz + well) / 2), 0.6 * r, (well + hz) / 2, "grey", collides="hidden"),
             P.lathe((0, 0, 0), "plate", r, hz, "ceramic:white"))
    return BodySpec(name, "compound", (r, r, hz), Pose((xy[0], xy[1], TABLE + hz)),
                    category="plate", color="ceramic:white", fixed=True,
                    attrs={"parts": parts, "container": "open", "wall": 0.004, "graspable": False,
                           "footprint": "round", "floor": well + hz})


def build_scene(level: str):
    bodies, hinge = turntable()
    for i in range(CUPS):
        bodies.append(cover(cup_name(i), ring_xy(home_angle(i)), disc_top()))
    bodies.append(tart(CAKE, ring_xy(home_angle(0)), disc_top()))
    bodies.append(plate(DISH, ahead(*DISH_AT)))
    return station_scene(f"di6_carousel_{level}", bodies, articulations=(hinge,), frame=FRAME)


class Turntable(Reactor):
    """Turns the table through the seed's turns on its own clock, easing in
    and out (things standing on it ride by friction on Isaac), then says it
    has stopped."""

    def __init__(self, spins: Sequence[float], rate: float, start: float = STARTS_AT) -> None:
        super().__init__()
        self.spins, self.rate = list(spins), rate
        self._angle, self._t, self._k = 0.0, None, 0
        self._goal = 0.0
        self._rest_until = start
        self._done = False

    def tick(self, engine: Engine, world: WorldState) -> None:
        t = world.sim_time
        if self._t is None:
            # a soft drive: the angle is set ten times a second, and a hard one
            # jerked to each new setting at once -- 100 rad/s^2, enough to fling
            # the cups off the table (Isaac, 9/21).  Heavily damped, it follows
            # the steps as one smooth turn.
            self.act(engine, world, "soft_drive", SetJointDrive(HINGE, SOFT[0], SOFT[1]))
        dt = 0.0 if self._t is None else max(t - self._t, 0.0)
        self._t = t
        if self._done or t < self._rest_until:
            return
        if abs(self._goal - self._angle) < 1e-6:
            if self._k >= len(self.spins):
                self._done = True
                self.note(engine, world, STOPPED)
                self.say(world, TURNTABLE, "stopped", "It stopped. Where is the cake?")
                return
            self._goal += self.spins[self._k]
            self._k += 1
            self.note(engine, world, f"turn_{self._k}")
        left = self._goal - self._angle
        # ease out over the last quarter turn's worth of angle, never below a tenth of the rate
        speed = self.rate * max(0.1, min(1.0, abs(left) / 0.6))
        step = max(-speed * dt, min(speed * dt, left))
        self._angle += step
        engine.set_articulation_target(HINGE, self._angle)
        if abs(self._goal - self._angle) < 1e-6:
            # the soft drive trails its setting: after the last turn, give the
            # table time to come to rest before saying it has stopped
            self._rest_until = t + (PAUSE if self._k < len(self.spins) else SETTLE_END)


def covered(w: WorldState) -> bool:
    c = w.obj(CAKE).pose.pos
    return any(math.dist(w.obj(cup_name(i)).pose.pos[:2], c[:2]) < 0.03 for i in range(CUPS))


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DI6")
    scene = build_scene(level)
    spins = turns(level, seed)
    t_ref = (config or RunConfig()).t_ref_seconds
    rate = spin_rate(level, t_ref)

    def lifted(w: WorldState) -> List[str]:
        """Covers off the turntable, in the hand, or tipped -- a cover tipped up is
        as good as lifted for a look under it."""
        return [cup_name(i) for i in range(CUPS)
                if w.obj(cup_name(i)).is_held
                or w.obj(cup_name(i)).pose.pos[2] - w.obj(cup_name(i)).size[2] > disc_top() + 0.02
                or math.dist(w.obj(cup_name(i)).pose.pos[:2], ahead(*TABLE_AT)) > DISC_R + 0.01
                or not Upright(cup_name(i), TIPPED).holds(w)]

    found = Custom(lambda w: In(CAKE, DISH).holds(w) and not w.obj(CAKE).is_held, "the cake on the plate")
    early = Custom(lambda w: STOPPED not in w.events_fired and bool(lifted(w)),
                   "a cover lifted before the table stopped")
    milestones = (
        Milestone("turning", Custom(lambda w: "turn_1" in w.events_fired, "the table turning")),
        Milestone("stopped", Custom(lambda w: STOPPED in w.events_fired, "the table stopped")),
        Milestone("the_right_cover", Custom(lambda w: STOPPED in w.events_fired and not covered(w),
                                            "the cake's cover lifted")),
        Milestone("cake_on_the_plate", found),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="The cake is under the cover nearest you. Watch the turntable, and when it "
                    "stops, find the cake and put it on the plate.",
        scorer=Scorer(task_id="DI6", level=level, success=Finally(found), milestones=milestones,
                      violations=(("wait_for_it_to_stop", Never(early)),
                                  ("only_one_cover_lifted", OneCoverEver(lifted)))),
        milestones=milestones,
        reactor_factory=lambda s=spins, r=rate, g=turn_start(t_ref): Turntable(s, r, g),
        reference_factory=lambda inst, g=turn_start(t_ref): FollowPolicy(start=g),
        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level],
                  # hidden from the agent: how it turns (the gap-filled control)
                  "turns_degrees": [round(math.degrees(a)) for a in spins],
                  "rad_per_s": round(rate, 3),
                  # a look every 0.3 s while it turns, and more
                  "max_model_calls": 60 + int(4 * sum(abs(a) for a in spins) / rate)},
    )


class FollowPolicy(ScriptedPolicy):
    """Keeps its eye on the cover that was nearest at the start, and once the
    table has stopped lifts that cover off and takes the cake to the plate.
    Blind (the first picture only): waits, then lifts whatever cover stands
    where the cake's cover stood at the start."""

    name = "follow"

    def __init__(self, arm: str = RIGHT, start: float = STARTS_AT) -> None:
        self.arm, self.start = arm, start
        self.reset()

    def reset(self) -> None:
        self._sub = None
        self._phase = "cover"
        self._cover: Optional[str] = None

    def _carry(self, world, ctx, name, point) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PlaceAtPointPolicy

        if self._sub is None:
            self._sub = PlaceAtPointPolicy(name, point, arm=self.arm, tolerance=0.025,
                                           release_clearance=0.004, carry_z=CARRY_Z)
        if self._sub.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"could not move {name}")]
        return [c for c in self._sub.commands(world, ctx) if not isinstance(c, Done)] \
            or [Wait(0.3)]

    def _nearest(self, world: WorldState) -> str:
        home = ring_xy(home_angle(0))
        return min((cup_name(i) for i in range(CUPS)),
                   key=lambda c: math.dist(world.obj(c).pose.pos[:2], home))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._cover is None:
            self._cover = self._nearest(world)       # the first look: the cake's cover
        if STOPPED not in world.events_fired:
            return [Wait(0.5)]
        return self._act(world, ctx)

    def _act(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if self._phase == "cover":
            c = world.obj(self._cover)
            if not c.is_held and math.dist(c.pose.pos[:2], ahead(*TABLE_AT)) > DISC_R + 0.02:
                self._phase, self._sub = "cake", None
                return [Wait(0.3)]
            aside = ahead(TABLE_AT[0] - 0.02, TABLE_AT[1] + DISC_R + 0.07)
            return self._carry(world, ctx, self._cover, (aside[0], aside[1], TABLE))
        if In(CAKE, DISH).holds(world) and not world.obj(CAKE).is_held:
            return [Wait(0.5), Done("found it")]
        if covered(world) and not world.obj(CAKE).is_held:
            return [Done("not under that one")]
        d = world.obj(DISH)
        return self._carry(world, ctx, CAKE, (d.pose.pos[0], d.pose.pos[1],
                                              d.pose.pos[2] - d.size[2] + d.attrs["floor"]))

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: all it has is the first picture -- wait out a turning it never
        watches, then lift what stands where the cake's cover stood, and reach
        for the cake where it was."""
        from ...policies.scripted import _pick_and_place_sequence

        cover_ = world.obj(self._nearest(world))
        aside = ahead(TABLE_AT[0] - 0.02, TABLE_AT[1] + DISC_R + 0.07)
        d = world.obj(DISH)
        return ([Wait(self.start + 12.0)]
                + _pick_and_place_sequence(self.arm, cover_, (aside[0], aside[1], TABLE), 0.0,
                                           0.26, 0.004)
                + _pick_and_place_sequence(self.arm, world.obj(CAKE), d.pose.pos, d.size[2],
                                           0.26, 0.006)
                + [Done("guessed")])


def secrets(inst) -> Tuple[str, ...]:
    """The cover the tart is under (``TaskInstance.secrets``): the covers ride
    the turntable with it, so the one over it at the start is the answer."""
    bodies = {b.name: b for b in inst.scene.bodies}
    tart = bodies[CAKE].pose.pos
    covers = [b for b in inst.scene.bodies if b.category == "dessert cover"]
    return (min(covers, key=lambda b: math.dist(b.pose.pos[:2], tart[:2])).name,)
