"""DI2 "load the train" -- Dynamic Interaction.

Knob: how far the train goes in one reference response time, measured in the
wagon's spare room (0.5 -> 3).

A wooden toy train -- an engine, a red wagon and a blue one -- runs back and
forth along a straight track, each wagon on a driven slide with the same
program (never a teleported body: only a joint carries what is in a wagon
the way a train would).  A small plush teddy bear and a rubber duck (scans)
lie beside the track.  "Put the teddy in the red wagon and the duck in the
blue one."  Both have to ride along, still relative to their wagons, for 2 s.
One pass is seldom enough for both: the train comes round again (v3: one
wagon and the bear, before).  A plan made from the first picture puts each
where its wagon was.

The knob is a ratio in units of T_ref (``RunConfig.t_ref_seconds``), a
reference interface's response time, which Gate C measures with the real API;
until then it is the provisional 8 s (the card says so and ``ruibench
validate`` warns).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines.base import ArticulationSpec
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import Custom, Finally, Held, HoldsFor, In, Milestone, RelativeStatic, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, Frame, ahead, asset_body, station_scene

BEAR, DUCK, ENGINE, TRACK = "teddy_bear", "duck", "engine", "track"
RED, BLUE = "red_wagon", "blue_wagon"
WAGON = RED
SLIDES = {RED: "red_wagon_slide", BLUE: "blue_wagon_slide"}
#: what goes where, in the order the reference loads them
CARGO = ((BEAR, RED), (DUCK, BLUE))
#: the blue wagon runs this far behind the red one (-y)
COUPLING = 0.12
DUCK_HALF = (0.0249, 0.035, 0.0333)
DUCK_AT = (0.15, -0.14)
#: the wagon's run either way: a blind drop where it was lands in it about
#: one time in five (its room over the run)
AMPLITUDE = 0.12
#: the wagon's middle runs along y through this point
TRACK_AT = (0.30, 0.03)
#: the bear lies across the track, taken across its body: 7.5 cm long, 5.4
#: wide, 2.5 thick
BEAR_HALF, BEAR_YAW = (0.0272, 0.0375, 0.0125), math.pi / 2
#: room for the bear's middle inside the wagon, along the track
MARGIN = T.WAGON_HALF[1] - T.WAGON_WALL - BEAR_HALF[0]
BEAR_AT, BEAR_JITTER = (0.15, 0.14), 0.015
HOVER = 0.08


#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.050, 0.400), (-0.300, 0.350), 0.10)


def speed_for(ratio: float, t_ref: float) -> float:
    """The train's speed: ``ratio`` wagon's spare rooms in one T_ref."""
    return ratio * MARGIN / t_ref


def wagon_at(car, wagon: str, dt: float) -> Tuple[float, float]:
    """Where ``wagon`` (``car``, its state) will be ``dt`` from now: it keeps
    its speed and turns back at the ends of its run."""
    mid = ahead(*TRACK_AT)[1] - (COUPLING if wagon == BLUE else 0.0)
    s = car.pose.pos[1] - mid + car.lin_vel[1] * dt
    while abs(s) > AMPLITUDE:
        s = math.copysign(2 * AMPLITUDE, s) - s
    return (car.pose.pos[0], mid + s)


def build_scene(seed: int, speed: float):
    s = SeedBundle(seed)
    wx, wy = ahead(*TRACK_AT)
    red = T.wagon(RED, (wx, wy), SLIDES[RED], "paint:red")
    blue = T.wagon(BLUE, (wx, wy - COUPLING), SLIDES[BLUE], "paint:blue")
    bear = asset_body(BEAR, "teddy_small", s.jitter("bear", ahead(*BEAR_AT), BEAR_JITTER),
                      fallback_size=BEAR_HALF, mass=0.03, category="teddy bear", yaw=BEAR_YAW,
                      grasp_yaw=0.0)
    duck = asset_body(DUCK, "duck_small", s.jitter("duck", ahead(*DUCK_AT), BEAR_JITTER),
                      fallback_size=DUCK_HALF, mass=0.03, category="duck", yaw=math.pi / 2)
    bodies = [bear, duck, red, blue, T.locomotive(ENGINE, red),
              T.track(TRACK, (wx, wy + 0.055 - COUPLING / 2), AMPLITUDE + 0.13 + COUPLING / 2)]
    phase = s.uniform("phase", 0.0, 1.0)
    joints = [ArticulationSpec(SLIDES[w], "slide", axis=(0.0, 1.0, 0.0),
                               limits=(-AMPLITUDE - 0.01, AMPLITUDE + 0.01), initial=0.0,
                               actuated=True, parent=w, oscillate=(AMPLITUDE, speed, phase))
              for w in (RED, BLUE)]
    return station_scene("di2_train", bodies, articulations=joints, frame=FRAME)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DI2")
    ratio = float(spec.knob.levels[level])
    t_ref = (config or RunConfig()).t_ref_seconds
    speed = speed_for(ratio, t_ref)
    scene = build_scene(seed, speed)
    inside = {item: In(item, wagon) for item, wagon in CARGO}
    both_in = Custom(lambda w: all(p.holds(w) for p in inside.values()),
                     "the teddy in the red wagon and the duck in the blue one")
    riding = Custom(lambda w: both_in.holds(w)
                    and all(RelativeStatic(i, wg).holds(w) for i, wg in CARGO),
                    "both in their wagons, riding with them")
    milestones = (
        Milestone("bear_grasped", Held(BEAR)),
        Milestone("bear_in", inside[BEAR]),
        Milestone("duck_in", inside[DUCK]),
        Milestone("riding", riding),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Put the teddy in the red wagon and the duck in the blue one.",
        scorer=Scorer(task_id="DI2", level=level,
                      success=Finally(both_in) & HoldsFor(riding, 2.0), milestones=milestones),
        milestones=milestones,
        reference_factory=lambda inst: LoadPolicy(),
        metadata={"knob": spec.knob.field, "knob_value": ratio,
                  "train_speed_m_s": round(speed, 6), "t_ref_s": t_ref,
                  "max_model_calls": 80},
    )


class LoadPolicy(ScriptedPolicy):
    """The teddy into the red wagon, then the duck into the blue one, each by
    chasing its wagon (``ChasePolicy``)."""

    name = "load"

    def __init__(self, arm: str = RIGHT) -> None:
        self.arm = arm
        self.reset()

    def reset(self) -> None:
        self._chasers = [ChasePolicy(item, wagon, self.arm) for item, wagon in CARGO]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        for chaser in self._chasers:
            if In(chaser.item, chaser.wagon).holds(world) and not world.obj(chaser.item).is_held:
                continue
            cmds = chaser.commands(world, ctx)
            if any(isinstance(c, Done) for c in cmds) and not In(chaser.item, chaser.wagon).holds(world):
                return cmds                                  # it gave up
            return [c for c in cmds if not isinstance(c, Done)] or [Wait(0.2)]
        return [Done("both loaded")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: each where its wagon was in the first picture."""
        out: List[Command] = []
        for chaser in self._chasers:
            out += chaser.plan(world, ctx)[:-1]
        return out + [Done("planned")]


class ChasePolicy(ScriptedPolicy):
    """Take the thing, keep it over where its wagon will be, and let it down
    once the hand is over the wagon."""

    name = "chase"
    LEAD = 0.5          # s: aim where the wagon will be this soon
    #: ...and, going down to let go, where it will be once the thing is down
    #: and let go -- clear of the walls, this far over the rim
    LEAD_DOWN, OVER_RIM = 1.0, 0.003

    def __init__(self, item: str = BEAR, wagon: str = RED, arm: str = RIGHT) -> None:
        self.item, self.wagon, self.arm = item, wagon, arm
        self.attempts = 0

    def reset(self) -> None:
        self.attempts = 0

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, grasp_pose, retreat_pose

        if ctx.extras.get("executor_busy"):
            return []
        bear, car, arm = world.obj(self.item), world.obj(self.wagon), world.arm(self.arm)
        if In(self.item, self.wagon).holds(world):
            return [Done("in the wagon")]
        if not bear.is_held:
            self.attempts += 1
            if self.attempts > MAX_GRASP_ATTEMPTS:
                return [Done(f"could not keep hold of the {self.item}")]
            grasp = grasp_pose(bear)
            above = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + 0.10), GRASP_DOWN)
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above), MoveTo(self.arm, grasp),
                    SetGripper(self.arm, 1.0), MoveTo(self.arm, above)]
        offset = arm.eef_pose.pos[2] - bear.pose.pos[2]
        rim = car.pose.pos[2] + car.size[2]
        lead = wagon_at(car, self.wagon, self.LEAD)
        hover_z = rim + HOVER + bear.size[2] + offset
        if math.dist(arm.eef_pose.pos[:2], car.pose.pos[:2]) < 0.012:
            # over it now: down to just over the rim, where the wagon will be
            # by then, and let go -- nothing touches the thing on its way.
            # Let go just above the floor, it was held 2-3 mm short of there,
            # the move never finished, and the fast train's wall pushed it,
            # hand and all, up to 18 cm along the track (Isaac, v13)
            x, y = wagon_at(car, self.wagon, self.LEAD_DOWN)
            low = Pose((x, y, rim + self.OVER_RIM + bear.size[2] + offset), GRASP_DOWN)
            return [MoveTo(self.arm, low), SetGripper(self.arm, 0.0),
                    MoveTo(self.arm, retreat_pose(arm)), Wait(0.5)]
        return [MoveTo(self.arm, Pose((lead[0], lead[1], hover_z), GRASP_DOWN))]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: the thing goes where its wagon was in the first picture."""
        from ...policies.scripted import _pick_and_place_sequence

        car = world.obj(self.wagon)
        floor = car.pose.pos[2] - car.size[2] + T.WAGON_WALL
        return _pick_and_place_sequence(self.arm, world.obj(self.item),
                                        (car.pose.pos[0], car.pose.pos[1], floor), 0.0,
                                        0.12, 0.008) + [Done("planned")]
