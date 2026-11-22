"""CL4 "the crow and the pitcher" -- In-Context Learning.

Knob: how many pebbles it takes (four -> eight, with the water starting as low
and the pebbles half the size).

A tall glass jar with a little water in it and a ball floating on the water,
too far down for the hand to reach: the jar is narrower than the hand is
long, so there is no reaching in, at any depth.  A wider well stands beside
it, joined to its foot by a glass elbow, so the water in the two stands at
the same height.  A handful of stones lies on the bench, and an empty dish.
"Get the ball out of the jar and put it in the dish."

Nothing says how.  Two short films show what the answer is made of (v9;
until then one film showed the whole of it): a stone let go into the well --
the water comes up a step in both, the ball with it -- and a cork let go
into the well, which floats and brings nothing up (v10: a cork; a block of
white foam did not read as foam).  Putting the two
together is the agent's: stones into the well, one at a time, until the ball
stands proud of the jar's rim.  Without the films there is nothing to try --
reaching for the ball fails wherever the hand goes.  The cork lies on the
bench in the task too: it is no use.

The stones go into the well and not into the jar for a plain reason: a stone
big enough for the hand to take and for the fixed camera to see is wider than
the gap beside the ball, and stones dropped down the jar would end up on top
of it.  Two chambers is also how the demonstration is really done.

The water is a disc on a driven slide with its column drawn below it, and the
reactor raises it by a fixed step per pebble that goes in; the ball rides on
the disc, so it is an ordinary body the hand can take once it is high enough.
The step is the fable's, not Archimedes': a pebble of this size would not
really raise a jar this wide by that much, and the point being made -- put
something in, the level comes up -- is what the picture shows.

v3.1 (2026-09-20), replacing "use the tool": a ball under a cabinet and a
hooked stick read as nothing much in the picture, and the method was one
anybody would try without being shown.

A stone is a ``compound`` of one ellipsoid part, not an ``ellipsoid`` body:
the backends take box, cylinder, sphere and compound bodies and nothing
else, and the mock does not mind either way -- Isaac was the first to say so
(2026-09-20).
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...events.actions import DriveJoint
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, In, Milestone, Scorer
from ..instance import Demonstration, TaskInstance
from ..loader import get
from ...engines.isaac.assets import get as catalogue
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

#: how many pebbles it takes.  They are the same size at both levels: a
#: pebble small enough to make the hard level "more of them" was flat enough
#: for the hand to hide it from the fixed camera
GAME = {"easy": 4, "hard": 8}
#: the stones: two photoscanned river stones, in turn (``assets.yaml``), 5.6
#: and 5.4 cm long -- not smaller: the hand hid a pebble half the size of the
#: v3 ones from the fixed camera while it held it -- and their half extents
#: until the catalogue has measured them
PEBBLE_LOOKS = ("river_rock", "sea_pebble")
PEBBLE_FALLBACK = {"river_rock": (0.028, 0.017, 0.012), "sea_pebble": (0.027, 0.017, 0.012)}
#: how many spare pebbles the tray holds beyond what is needed
SPARE = 1                        # one more stone than is needed

# -- the jar: a tall glass tube, wider than the ball and far narrower than the
#    hand is long, so there is no reaching in ------------------------------------
JAR_R, JAR_WALL, JAR_H, JAR_FLOOR = 0.032, 0.003, 0.170, 0.005
GLASS = "glass:clear"
#: the jar in pale green bottle glass, its wall and rim plain to see: clear,
#: it showed two faint lines and no rim, and whether the ball stood proud of
#: it could not be told (v11, the advisor)
JAR_GLASS = "glass:bottle"
WATER = "glass:water"
#: the water: a disc the ball rests on, its column drawn below it
DISC_R, DISC_HZ = JAR_R - 0.004, 0.002
COLUMN = 0.36                    # down past the table's top, out of sight
SURFACE = 0.012                  # the drawn water's top, above the disc's
START_TOP = 0.030                # the disc's top, above the table
END_TOP = JAR_H - 0.010
RISE = END_TOP - START_TOP
SLIDE = "water_slide"

# -- the well the stones go in: joined to the jar's foot.  Wide enough for
#    four stones side by side and deep enough for two such layers -- at 9 cm
#    across only one stone fitted a layer, and the third balanced on the rim
#    instead of going in (Isaac, 2026-09-20) --------------------------------
WELL_R, WELL_WALL, WELL_H, WELL_FLOOR = 0.065, 0.003, 0.170, 0.005
WELL_DY = -0.125               # from the jar, across the bench -- away from the
                               # stones, which it would otherwise stand in front of

# -- what floats in the well floats on a hidden disc there, rising with the
#    water; the stones pass through it (``engines.base.passes``) -------------------
WELL_WATER, WELL_SLIDE = "well_water", "well_water_slide"
#: how far under the drawn water's top a floating thing's bottom sits
FLOAT_SINK = 0.006

# -- the cork: a bottle's cork stopper lying on the bench, the second film's -------
CORK = "cork"
CORK_ASSET, CORK_HALF = "cork_stopper", (0.0182, 0.022, 0.0182)   # 3.6 cm across, 4.4 cm long
CORK_AT = (0.33, 0.15)

# -- the ball: it floats, and once it is proud of the rim the hand can take it ----
BALL_R = 0.025
BALL_COLOR = "plastic:red"

# -- where -------------------------------------------------------------------------
JAR_AT = (0.33, 0.0)
STONES_AT = (0.154, 0.200)
DISH_AT = (0.15, -0.225)
#: the dish: a glazed bowl 15 cm across (v8; a white open box until then),
#: over a hidden square shell inscribed in it, 10.4 cm inside
DISH_R, DISH_HZ = 0.075, 0.026
DISH_HALF, DISH_WALL = (0.052, 0.052, DISH_HZ), 0.006
#: the pebbles lie in two rows along the tray, which runs away from the robot
#: beside the jar: five to a row, 5.4 cm apart, the rows 7.1 cm apart, so the
#: open fingers (6.6 cm out along y) and the hand (3.2 cm across) clear both
PEBBLE_ROWS = 3
PEBBLE_STEP, ROW_GAP = 0.066, 0.106
CARRY_Z = TABLE + 0.30
#: a stone counts as in the well within this of its axis, and it counts as let
#: go into it when the fingers open with it this far above the rim at most
IN_WELL_R = WELL_R - 0.008
DROP_HEIGHT = 0.120
FRAME = Frame((0.04, 0.42), (-0.30, 0.34), 0.24)


def pebbles(level: str) -> int:
    return GAME[level] + SPARE


def pebble_name(k: int) -> str:
    return f"pebble{k + 1}"


def jar(name: str = "jar") -> BodySpec:
    """A tall glass jar: a hidden square shell for the fingers to meet if they
    ever touch it, the glass drawn over it, open at the top."""
    hz = JAR_H / 2
    parts = [P.grip((0, 0, -hz + JAR_FLOOR / 2), (JAR_R, JAR_R, JAR_FLOOR / 2))]
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = ((JAR_WALL / 2, JAR_R, hz) if sx else (JAR_R - JAR_WALL, JAR_WALL / 2, hz))
        parts.append(P.grip((sx * (JAR_R - JAR_WALL / 2), sy * (JAR_R - JAR_WALL / 2), 0), half))
    # the glass all round: the fixed camera sees the water and the ball
    # through it (v8); the hidden shell above is what a hand meets
    parts.append(P.vessel((0, 0, 0), JAR_R, hz, JAR_GLASS))
    x, y = ahead(*JAR_AT)
    return BodySpec(name, "compound", (JAR_R, JAR_R, hz), Pose((x, y, TABLE + hz)),
                    category="jar", color=JAR_GLASS, fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "round", "graspable": False})


def well(name: str = "well") -> BodySpec:
    """The wider chamber the stones go in, joined to the jar's foot by a glass
    elbow: the water in the two stands at the same height."""
    hz = WELL_H / 2
    parts = [P.grip((0, 0, -hz + WELL_FLOOR / 2), (WELL_R, WELL_R, WELL_FLOOR / 2))]
    for sx, sy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        half = ((WELL_WALL / 2, WELL_R, hz) if sx else (WELL_R - WELL_WALL, WELL_WALL / 2, hz))
        parts.append(P.grip((sx * (WELL_R - WELL_WALL / 2), sy * (WELL_R - WELL_WALL / 2), 0),
                            half))
    parts.append(P.vessel((0, 0, 0), WELL_R, hz, GLASS))
    # the elbow to the jar, drawn only
    parts.append(P.cylinder((0, -WELL_DY / 2, -hz + 0.010), 0.008, abs(WELL_DY) / 2,
                            GLASS, rot=P.turn_x(math.pi / 2), collides=False))
    x, y = well_xy()
    return BodySpec(name, "compound", (WELL_R, WELL_R, hz), Pose((x, y, TABLE + hz)),
                    category="well", color=GLASS, fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "round", "graspable": False})


def well_xy() -> Tuple[float, float]:
    x, y = ahead(*JAR_AT)
    return (x, y + WELL_DY)


def water() -> Tuple[BodySpec, ArticulationSpec]:
    """The water: a disc things rest on, the column drawn below it and down
    past the table's top, where nothing sees it."""
    x, y = ahead(*JAR_AT)
    parts = [P.cylinder((0, 0, 0), DISC_R, DISC_HZ, WATER),
             P.cylinder((0, 0, DISC_HZ + SURFACE / 2 - 0.0005), DISC_R, SURFACE / 2,
                        WATER, collides=False),
             P.cylinder((0, 0, -DISC_HZ - COLUMN / 2), DISC_R, COLUMN / 2, WATER, collides=False),
             # the same water, drawn in the well: it stands at the same height,
             # and nothing rests on it there -- the stones drop past it
             P.cylinder((0, WELL_DY, DISC_HZ + SURFACE / 2 - 0.0005), WELL_R - 0.005,
                        SURFACE / 2, WATER, collides=False),
             P.cylinder((0, WELL_DY, -DISC_HZ - COLUMN / 2), WELL_R - 0.005, COLUMN / 2,
                        WATER, collides=False)]
    body = BodySpec("water", "compound", (DISC_R, DISC_R, DISC_HZ),
                    Pose((x, y, TABLE + START_TOP - DISC_HZ)), category="water", color=WATER,
                    attrs={"parts": tuple(parts), "footprint": "round", "graspable": False,
                           "carried_by": SLIDE, "driven": True})
    slide = ArticulationSpec(SLIDE, "slide", axis=(0.0, 0.0, 1.0), limits=(0.0, RISE),
                             initial=0.0, actuated=True, parent="water", carries=True)
    return body, slide


def well_water(level: str) -> Tuple[BodySpec, ArticulationSpec]:
    """A hidden disc at the water's surface in the well, on its own slide,
    risen with the jar's: what floats there rests on it; the stones drop
    through it (``passes``) to the well's floor."""
    x, y = well_xy()
    r, hz = WELL_R - 0.006, 0.002
    top = TABLE + START_TOP + SURFACE - 0.0005 - FLOAT_SINK
    body = BodySpec(WELL_WATER, "compound", (r, r, hz), Pose((x, y, top - hz)),
                    category="water", color=WATER,
                    attrs={"parts": (P.cylinder((0, 0, 0), r, hz, "grey", collides="hidden"),),
                           "footprint": "round", "graspable": False, "carried_by": WELL_SLIDE,
                           "driven": True,
                           "passes": tuple(pebble_name(k) for k in range(pebbles(level)))})
    slide = ArticulationSpec(WELL_SLIDE, "slide", axis=(0.0, 0.0, 1.0), limits=(0.0, RISE),
                             initial=0.0, actuated=True, parent=WELL_WATER, carries=True)
    return body, slide


def cork() -> BodySpec:
    """A cork stopper lying on its side: light, and it floats."""
    return asset_body(CORK, CORK_ASSET, ahead(*CORK_AT), fallback_size=CORK_HALF, mass=0.004,
                      category="cork")


def ball() -> BodySpec:
    x, y = ahead(*JAR_AT)
    return BodySpec("ball", "sphere", (BALL_R, BALL_R, BALL_R),
                    Pose((x, y, TABLE + START_TOP + BALL_R)), category="ball",
                    color=BALL_COLOR, mass=0.02,
                    # down in the jar no descent reaches it; the water brings
                    # it up to where the hand can (``graspable_later``)
                    attrs={"footprint": "round", "graspable_later": True})


def pebble(level: str, k: int, xy) -> BodySpec:
    """A river stone: a hidden box the fingers meet, a photoscanned stone
    drawn over it (v8; a grey ellipsoid until then) -- two stones in turn,
    every other pair turned end for end.

    A ``compound``: an ellipsoid or a scan never collides by itself, and a
    stone made of one had no collider at all and fell through the world
    (Isaac, 2026-09-20).  The box lies inside the stone: its corners would
    otherwise reach into the next stone along the row."""
    look = PEBBLE_LOOKS[k % len(PEBBLE_LOOKS)]
    entry = catalogue(look)
    half = tuple(entry.half_extents) if entry.half_extents else PEBBLE_FALLBACK[look]
    core = (half[0] * 0.78, half[1] * 0.78, half[2])
    turn = P.turn_z(math.pi) if (k // 2) % 2 else (1.0, 0.0, 0.0, 0.0)
    return BodySpec(pebble_name(k), "compound", half,
                    Pose((xy[0], xy[1], TABLE + half[2])),
                    category="pebble", color="grey", mass=0.03,
                    attrs={"parts": (P.grip((0, 0, 0), core), P.visual(look, (0, 0, 0), half, turn)),
                           "footprint": "round", "grasp_height": 0.0, "look": look})


def dish(name: str = "dish") -> BodySpec:
    """A glazed bowl, the hidden square shell inside it what the ball meets."""
    hx, hy, hz = DISH_HALF
    wall = DISH_WALL
    parts = [P.grip((0, 0, -hz + wall / 2), (hx, hy, wall / 2)),
             P.grip((hx - wall / 2, 0, 0), (wall / 2, hy, hz)),
             P.grip((-hx + wall / 2, 0, 0), (wall / 2, hy, hz)),
             P.grip((0, hy - wall / 2, 0), (hx - wall, wall / 2, hz)),
             P.grip((0, -hy + wall / 2, 0), (hx - wall, wall / 2, hz)),
             P.lathe((0, 0, 0), "bowl", DISH_R, DISH_HZ, "ceramic:celadon")]
    x, y = ahead(*DISH_AT)
    return BodySpec(name, "compound", (DISH_R, DISH_R, DISH_HZ), Pose((x, y, TABLE + DISH_HZ)),
                    category="dish", color="ceramic:celadon", fixed=True,
                    attrs={"parts": tuple(parts), "container": "open", "wall": wall,
                           "graspable": False, "footprint": "round"})


def pebble_spots(level: str, seed: int) -> List[Tuple[float, float]]:
    n = pebbles(level)
    per = math.ceil(n / PEBBLE_ROWS)
    x0, y0 = ahead(*STONES_AT)
    spots = [(x0 + (k - (per - 1) / 2) * PEBBLE_STEP,
              y0 + (r - (PEBBLE_ROWS - 1) / 2) * ROW_GAP)
             for r in range(PEBBLE_ROWS) for k in range(per)][:n]
    SeedBundle(seed).stream("pebbles").shuffle(spots)
    return spots


def scene_for(level: str, seed: int, tag: str):
    w, slide = water()
    ww, well_slide = well_water(level)
    bodies: List[BodySpec] = [jar(), well(), w, ww, ball(), dish(), cork()]
    bodies += [pebble(level, k, xy) for k, xy in enumerate(pebble_spots(level, seed))]
    return station_scene(f"cl4_pitcher_{level}", bodies, articulations=(slide, well_slide),
                         frame=FRAME)


def in_jar(w: WorldState, name: str) -> bool:
    """Down inside the jar -- where the ball is until the water carries it up."""
    jar_xy = w.obj("jar").pose.pos[:2]
    x, y, z = w.obj(name).pose.pos
    return math.dist((x, y), jar_xy) < JAR_R and z < TABLE + JAR_H


def in_well(w: WorldState, name: str) -> bool:
    x, y, z = w.obj(name).pose.pos
    return math.dist((x, y), well_xy()) < IN_WELL_R and z < TABLE + WELL_H


def over_mouth(w: WorldState, name: str) -> bool:
    """Held over the well's mouth, low enough to be let go into it."""
    x, y, z = w.obj(name).pose.pos
    return math.dist((x, y), well_xy()) < IN_WELL_R and z < TABLE + WELL_H + DROP_HEIGHT


class Rising(Reactor):
    """Every pebble that goes into the jar brings the water up one step."""

    def __init__(self, level: str) -> None:
        super().__init__()
        self.step = RISE / GAME[level]
        self.count = pebbles(level)
        self._in: set = set()
        self._over: set = set()

    def tick(self, engine: Engine, world: WorldState) -> None:
        for k in range(self.count):
            name = pebble_name(k)
            if name in self._in:
                continue
            if world.obj(name).is_held:
                # a stone counts when the fingers open over the mouth: on
                # Isaac it falls the rest of the way, and the mock, which has
                # no gravity, would leave it hanging there
                self._over.add(name) if over_mouth(world, name) else self._over.discard(name)
                continue
            if name not in self._over and not in_well(world, name):
                continue
            self._over.discard(name)
            self._in.add(name)
            level = min(RISE, len(self._in) * self.step)
            self.act(engine, world, f"{name}_in_the_well", DriveJoint(SLIDE, level))
            self.act(engine, world, f"{name}_raises_the_well", DriveJoint(WELL_SLIDE, level))
            if level >= RISE - 1e-6:
                self.note(engine, world, "ball_at_the_rim")


def proud(w: WorldState) -> bool:
    """Is the ball high enough to be taken?"""
    return w.obj("ball").pose.pos[2] > TABLE + JAR_H + 0.008


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CL4")
    scene = scene_for(level, seed, "episode")
    stone = Demonstration(scene=scene_for(level, seed + 1000, "film"),
                          demonstrator=lambda: DropIntoWell(pebble_name(0)),
                          reactor_factory=lambda l=level: Rising(l),
                          caption="A stone dropped into the well.")
    floats = Demonstration(scene=scene_for(level, seed + 2000, "film"),
                           demonstrator=lambda: DropIntoWell(CORK),
                           reactor_factory=lambda l=level: Rising(l),
                           caption="A cork dropped into the well.")
    done = Custom(lambda w: In("ball", "dish").holds(w) and AtRest("ball").holds(w)
                  and not w.obj("ball").is_held, "the ball in the dish")
    milestones = (
        Milestone("first_stone", Custom(lambda w: any(e.endswith("_in_the_well")
                                                      for e in w.events_fired),
                                        "a stone in the well")),
        Milestone("ball_at_the_rim", Custom(lambda w: "ball_at_the_rim" in w.events_fired,
                                            "the water up, the ball proud of the rim")),
        Milestone("ball_out", Custom(lambda w: not in_jar(w, "ball"), "the ball out of the jar")),
        Milestone("ball_in_dish", done),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Get the ball out of the jar and put it in the dish.",
        scorer=Scorer(task_id="CL4", level=level, success=Finally(done), milestones=milestones),
        milestones=milestones,
        demonstrations=(stone, floats),
        reactor_factory=lambda l=level: Rising(l),
        reference_factory=lambda inst, l=level: PitcherPolicy(l, shown=False),
        metadata={"knob": spec.knob.field, "knob_value": GAME[level],
                  # hidden from the agent: the method (the gap-filled control)
                  "method": "drop stones into the well until the water carries the ball up",
                  "stones_needed": GAME[level],
                  "max_model_calls": 40 + 14 * pebbles(level)},
        extra_metrics={"stones_used": lambda traj: sum(1 for e in traj[-1].events_fired
                                                        if e.endswith("_in_the_well"))
                       if traj else 0},
    )


class DropIntoWell(ScriptedPolicy):
    """A film: one thing -- a stone, or the cork -- taken up and let go over
    the middle of the well, and a moment to see what it does."""

    name = "drop_into_well"

    def __init__(self, thing: str, arm: str = RIGHT) -> None:
        self.thing, self.arm = thing, arm
        self.reset()

    def reset(self) -> None:
        self._stage = "take"

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import grasp_pose

        if ctx.extras.get("executor_busy"):
            return []
        obj = world.obj(self.thing)
        if self._stage == "take":
            self._stage = "carry"
            g = grasp_pose(obj)
            return [SetGripper(self.arm, 0.0),
                    MoveTo(self.arm, Pose((g.pos[0], g.pos[1], CARRY_Z), g.quat)),
                    MoveTo(self.arm, g), SetGripper(self.arm, 1.0), Wait(0.4)]
        if self._stage == "carry":
            self._stage = "watch"
            wx, wy = well_xy()
            hand = world.arm(self.arm).eef_pose.pos
            over = (wx, wy + 0.02, TABLE + WELL_H + 0.048)
            return [MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), GRASP_DOWN)),
                    MoveTo(self.arm, Pose(over, GRASP_DOWN)), SetGripper(self.arm, 0.0), Wait(0.6),
                    MoveTo(self.arm, Pose((over[0], over[1], CARRY_Z), GRASP_DOWN)), Wait(2.5)]
        return [Done(f"the {obj.category} is in the well")]


class PitcherPolicy(ScriptedPolicy):
    """Stones into the well, one at a time, until the ball stands proud of the
    jar's rim; then the ball into the dish -- what the two films, put together,
    say.  Without the films watched it tries the ball straight away, which no
    descent can reach."""

    name = "pitcher"

    def __init__(self, level: str, shown: bool, arm: str = RIGHT) -> None:
        self.level, self.shown, self.arm = level, shown, arm
        self.reset()

    def reset(self) -> None:
        self._k = 0
        self._phase = "pebble"
        self._sub = None

    def _watched(self, ctx: TurnContext) -> bool:
        return self.shown or any(getattr(f, "states", ()) for f in ctx.extras.get("demonstrations", ()))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PlaceAtPointPolicy, grasp_pose

        if ctx.extras.get("executor_busy"):
            return []
        if not self._watched(ctx):
            return self.plan(world, ctx)
        if proud(world) and self._phase != "ball":
            self._phase, self._sub = "ball", None
        if self._phase == "ball":
            dish = world.obj("dish")
            if self._sub is None:
                floor = dish.pose.pos[2] - DISH_HALF[2] + DISH_WALL
                self._sub = PlaceAtPointPolicy("ball", (dish.pose.pos[0], dish.pose.pos[1], floor),
                                               arm=self.arm, tolerance=0.03,
                                               release_clearance=0.010, carry_z=CARRY_Z)
            if In("ball", "dish").holds(world) and not world.obj("ball").is_held:
                return [Wait(0.5), Done("the ball is in the dish")]
            if self._sub.attempts > MAX_GRASP_ATTEMPTS:
                return [Done("could not take the ball")]
            return [c for c in self._sub.commands(world, ctx) if not isinstance(c, Done)] \
                or [Wait(0.3)]
        if self._k >= pebbles(self.level):
            return [Done("out of pebbles")]
        name = pebble_name(self._k)
        if f"{name}_in_the_well" in world.events_fired:
            self._k += 1
            return [Wait(0.3)]
        obj = world.obj(name)
        if not obj.is_held:
            g = grasp_pose(obj)
            return [SetGripper(self.arm, 0.0),
                    MoveTo(self.arm, Pose((g.pos[0], g.pos[1], CARRY_Z), g.quat)),
                    MoveTo(self.arm, g), SetGripper(self.arm, 1.0), Wait(0.4)]
        # over the well, a different place each time so they do not stack up
        wx, wy = well_xy()
        angle = 2 * math.pi * self._k / max(1, pebbles(self.level))
        off = WELL_R - 0.024
        over = (wx + off * math.cos(angle), wy + off * math.sin(angle),
                TABLE + WELL_H + 0.048)
        hand = world.arm(self.arm).eef_pose.pos
        return [MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), GRASP_DOWN)),
                MoveTo(self.arm, Pose(over, GRASP_DOWN)), SetGripper(self.arm, 0.0), Wait(1.0)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: no film watched -- reach straight in for the ball, which no
        descent can do (the jar is narrower than the hand is long)."""
        from ...policies.scripted import grasp_pose

        g = grasp_pose(world.obj("ball"))
        return [SetGripper(self.arm, 0.0),
                MoveTo(self.arm, Pose((g.pos[0], g.pos[1], CARRY_Z), g.quat)),
                MoveTo(self.arm, g), SetGripper(self.arm, 1.0), Wait(0.6),
                Done("reached for the ball")]
