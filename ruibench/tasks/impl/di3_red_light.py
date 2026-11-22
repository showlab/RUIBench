"""DI3 "red light, green light" -- Dynamic Interaction.

Knob: how the light changes (long greens and a second to clear the road ->
short greens and half a second).

Three ducklings on the pavement of a model street, a lawn across the road,
and a pedestrian light beside the zebra crossing.  "Take the ducklings across the
road to the grass, one at a time.  Cross only while the light is green: no
duckling on the road when it is red."  The light goes red and green on its
own clock.  On the pavement a duckling may be taken up and walked to the
kerb at any time; over the road it may be only while the light is green --
so a duckling waits at the kerb for the green and goes over the road the
moment it comes, and one caught on the road when the light turns red has a
moment to get off it.

The game is one of the circle-time games Tominey & McClelland (2011) used,
in a randomised trial, to train and measure preschoolers' behavioural self-
regulation ("Red Light, Purple Light").  What it asks for is the same here:
act while allowed, and stop -- quickly -- when not.

A duckling is on the road when any of it is over the road's strip, carried
or standing -- a foul once the light has been red for longer than the
moment allowed to get off.

v14 (2026-09-26): the street is a diorama, as WK2's river is -- a pavement
of concrete slabs, kerbstones, an asphalt road with a zebra crossing, a lawn
with a concrete edging, and a pedestrian light with a visor over each lamp;
until then a grey strip, a felt mat and a black box on a stick.

v8 (2026-09-24): the rule was the playground game's, "move only on green",
and a duckling had to stop dead wherever it was when the light turned red;
the advisor: "red means you may not cross the road -- not that you may not
move".  It is a pedestrian crossing's rule now.

v4 (2026-09-21), replacing "what does the little robot want?" (a guessing
game with a toy robot that answered yes or no): the advisor found it neither
clear nor interesting.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ...core.config import DEFAULT_T_REF_SECONDS, RunConfig
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...events.actions import SetColor
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, Milestone, Never, On, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

DUCKS = ("duck1", "duck2", "duck3")
DUCK_HALF = (0.0249, 0.035, 0.0333)
GRASS, ROAD, PAVEMENT, STREET = "grass", "road", "pavement", "street"
#: per level: (how long a green lasts, lowest and highest, and a red, in
#: T_ref; seconds allowed to stop once it goes red).  v13, provisional (T_ref
#: 8 s): 16-20 s greens at the easy level -- a model taking T_ref a turn sees
#: a green up to T_ref late and acts T_ref after that, and at 6-8 s the
#: reference stepping at 8 s crossed 7 seeds in 10 -- with 16 s reds, so the
#: light is green only about half the time and crossing without looking is
#: still caught; 2.5-3.5 s greens and 3 s reds at the hard level, a green only
#: anticipation or a faster loop catches
LIGHT = {"easy": ((2.0, 2.5), 2.0, 1.0), "hard": ((0.3125, 0.4375), 0.375, 0.5)}
#: the first green comes no sooner than this (and not before a whole red)
FIRST_GREEN = 2.0
#: a green never shorter than this, in seconds: the arm's own time to carry a
#: duckling across the road does not shrink with T_ref.  At T_ref = 8 s the
#: hard greens start at 2.5 s and the reference crossed every one on Isaac
#: (25/25); at 5.5 s they would start at 1.7 s, and at zero latency the
#: reference missed a seed in 10 (10/1, before Gate C set T_ref)
GREEN_FLOOR = {"easy": 0.0, "hard": 2.5}
RED_ON, GREEN_ON = (1.0, 0.1, 0.08), (0.1, 1.0, 0.3)
LAMP_OFF = (0.1, 0.1, 0.1)
ON_ROAD_ON_RED = "on_the_road_on_red"

# -- the layout: a model street (v14) -- the ducklings in a row on the
#    pavement at the near side, the road across the bench, the lawn beyond
#    it, the light on the far pavement ------------------------------------------
DUCK_X, DUCK_PITCH = 0.14, 0.13
#: the street's tray: its middle and half size inside the rim, the rim's half
#: thickness and half height, the floor's thickness.  Rim and all, it is the
#: fixed camera's frame (``FRAME``): off-centre, room for the light at +y
STREET_AT, STREET_HALF = (0.30, 0.04), (0.225, 0.33)
TRAY_RIM, TRAY_FLOOR = (0.005, 0.006), 0.0012
#: the pavements, the kerbstones and the lawn stand this high; the road's
#: asphalt this; a kerbstone is this wide
KERB_TOP, ROAD_TOP, KERB_W = 0.006, 0.0015, 0.01
#: the road runs the tray's length
ROAD_X, ROAD_Y, ROAD_HALF = 0.28, STREET_AT[1], (0.06, STREET_HALF[1], ROAD_TOP / 2)
#: the zebra crossing: its stripes run along the road, side by side across
#: it -- a stripe's width, the gap, and half their length: the crossing is as
#: wide as the three ducklings' ways over it
ZEBRA = (0.013, 0.011, 0.18)
GRASS_AT, GRASS_HALF = (0.44, 0.0), (0.07, 0.20)
#: the light stands on the far pavement at the crossing's end, clear of the
#: hand: carrying a duckling in an outer lane, its side is 23 cm out
LIGHT_AT = (0.36, 0.30)
POLE_H = 0.20
#: the light's head: half its depth, width and height; its lamps, this far
#: above and below its middle, and their lenses' radius
HEAD = (0.012, 0.018, 0.042)
LAMP_DZ, LENS_R = 0.021, 0.011
#: how high the hand carries a duckling; a duckling waits at the kerb this
#: far short of the road, and is over the grass this far past it
CARRY_Z = TABLE + 0.16
KERB = 0.012
FRAME = Frame((0.06, 0.54), (-0.30, 0.38), 0.24)


def schedule(level: str, seed: int, t_ref: float = DEFAULT_T_REF_SECONDS,
             horizon: float = 600.0) -> List[Tuple[float, str]]:
    """(when, "green" or "red"): the light's changes, from red at the start."""
    (lo, hi), red, _ = LIGHT[level]
    lo, hi, red = lo * t_ref, hi * t_ref, red * t_ref
    rng = SeedBundle(seed).stream("light")
    # it starts red, for a whole red: with greens long enough for a slow model,
    # the first one otherwise saw all three over before a crossing that never
    # looked at the light had met a red (v13)
    first = max(FIRST_GREEN, red)
    out, t = [(first, "green")], first
    while t < horizon:
        t += max(rng.uniform(lo, hi), GREEN_FLOOR[level])
        out.append((t, "red"))
        t += red
        out.append((t, "green"))
    return out


def street() -> BodySpec:
    """The street's tray and what is only drawn on it: the kerbstones either
    side of the road, the far pavement round the lawn, the lawn's concrete
    edging (the road, the near pavement and the lawn are bodies of their
    own)."""
    cx, cy = STREET_AT
    hx, hy = STREET_HALF
    t, h = TRAY_RIM
    parts = [P.slab((0.0, 0.0, TRAY_FLOOR / 2), (hx + 2 * t, hy + 2 * t, TRAY_FLOOR / 2), "wood:ash")]
    for sx in (-1, 1):                                  # the rim round it
        parts.append(P.slab((sx * (hx + t), 0.0, h), (t, hy + 2 * t, h), "wood:ash"))
    for sy in (-1, 1):
        parts.append(P.slab((0.0, sy * (hy + t), h), (hx, t, h), "wood:ash"))
    near, far = ROAD_X - ROAD_HALF[0] - cx, ROAD_X + ROAD_HALF[0] - cx
    for x0 in (near - KERB_W, far):                     # the kerbstones
        parts.append(P.slab((x0 + KERB_W / 2, 0.0, KERB_TOP / 2 + 0.0002),
                            (KERB_W / 2, hy, KERB_TOP / 2 + 0.0002), "ground:concrete"))
    # the lawn, from the tray's middle: its near and far edges, its sides
    gx0, gx1 = GRASS_AT[0] - GRASS_HALF[0] - cx, GRASS_AT[0] + GRASS_HALF[0] - cx
    gy0, gy1 = GRASS_AT[1] - GRASS_HALF[1] - cy, GRASS_AT[1] + GRASS_HALF[1] - cy
    x0 = far + KERB_W
    pave = [(x0, gx0, -hy, hy),                         # (x from, to, y from, to): along the kerb
            (gx1, hx, gy0, gy1),                        # beyond the lawn
            (gx0, hx, -hy, gy0), (gx0, hx, gy1, hy)]    # either side of it
    for xa, xb, ya, yb in pave:
        parts.append(P.box(((xa + xb) / 2, (ya + yb) / 2, KERB_TOP / 2),
                           ((xb - xa) / 2, (yb - ya) / 2, KERB_TOP / 2), "ground:paving", collides=False))
    edge, top = 0.004, KERB_TOP / 2 + 0.0003           # the lawn's edging
    mx, my = GRASS_AT[0] - cx, GRASS_AT[1] - cy
    for sx in (-1, 1):
        parts.append(P.slab((mx + sx * (GRASS_HALF[0] + edge / 2), my, top),
                            (edge / 2, GRASS_HALF[1] + edge, top), "ground:concrete"))
    for sy in (-1, 1):
        parts.append(P.slab((mx, my + sy * (GRASS_HALF[1] + edge / 2), top),
                            (GRASS_HALF[0], edge / 2, top), "ground:concrete"))
    x, y = ahead(cx, cy)
    return BodySpec(STREET, "compound", (hx + 2 * t, hy + 2 * t, h), Pose((x, y, TABLE)),
                    category="street", color="wood:ash", fixed=True,
                    attrs={"parts": tuple(parts), "collides": False, "graspable": False})


def pavement() -> BodySpec:
    """The near pavement the ducklings stand on: concrete slabs, as high as
    the kerb -- a body of its own, what stands on it stands on it."""
    x0, x1 = STREET_AT[0] - STREET_HALF[0], ROAD_X - ROAD_HALF[0] - KERB_W
    hx, hy, hz = (x1 - x0) / 2, STREET_HALF[1], KERB_TOP / 2
    x, y = ahead((x0 + x1) / 2, STREET_AT[1])
    parts = (P.box((0, 0, 0), (hx, hy, hz), "ground:paving", collides="hidden"),
             P.box((0, 0, 0), (hx, hy, hz), "ground:paving", collides=False))
    return BodySpec(PAVEMENT, "compound", (hx, hy, hz), Pose((x, y, TABLE + hz)), category="pavement",
                    color="ground:paving", fixed=True,
                    attrs={"parts": parts, "graspable": False, "footprint": "rect"})


def lawn() -> BodySpec:
    """The lawn across the road: turf, as high as the pavements."""
    x, y = ahead(*GRASS_AT)
    hx, hy = GRASS_HALF
    hz = KERB_TOP / 2
    parts = (P.box((0, 0, 0), (hx, hy, hz), "ground:grass", collides="hidden"),
             P.box((0, 0, 0), (hx, hy, hz), "ground:grass", collides=False))
    return BodySpec(GRASS, "compound", (hx, hy, hz), Pose((x, y, TABLE + hz)), category="grass",
                    color="ground:grass", fixed=True,
                    attrs={"parts": parts, "graspable": False, "footprint": "rect"})


def road() -> BodySpec:
    """The road, drawn only: asphalt, a zebra crossing, a stop line across
    each lane before it (traffic keeps right) and a dashed line down the
    middle beyond."""
    x, y = ahead(ROAD_X, ROAD_Y)
    hx, hy, hz = ROAD_HALF
    mid = -ROAD_Y                                       # the ducklings' middle, from the road's
    paint, top, coat = "paint:white", hz + 0.0003, 0.0003
    parts = [P.box((0, 0, 0), (hx, hy, hz), "ground:asphalt", collides=False)]
    width, gap, half = ZEBRA
    n = int((2 * hx + gap) // (width + gap))
    span = n * width + (n - 1) * gap
    parts += [P.box((-span / 2 + width / 2 + k * (width + gap), mid, top), (width / 2, half, coat), paint,
                    collides=False) for k in range(n)]
    for sy in (-1, 1):
        parts.append(P.box((-sy * hx / 2, mid + sy * (half + 0.02), top), (hx / 2 - 0.004, 0.003, coat),
                           paint, collides=False))
        parts += [P.box((0, mid + sy * (half + 0.045 + k * 0.05), top), (0.0015, 0.015, coat), paint,
                        collides=False)
                  for k in range(3) if abs(mid + sy * (half + 0.06 + k * 0.05)) < hy]
    return BodySpec(ROAD, "compound", (hx, hy, hz), Pose((x, y, TABLE + hz)), category="road",
                    color="ground:asphalt", fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect",
                           "collides": False})


def _visor(centre, facing: int, radius: float, depth: float, color, segments: int = 9) -> list:
    """A lamp's visor: a half tube over the top of its lens, standing out
    ``depth`` from the face (``facing`` +1 or -1 along x)."""
    cx, cy, cz = centre
    seg = radius * math.pi / segments / 2 * 1.1
    out = []
    for k in range(segments):
        a = math.pi * (k + 0.5) / segments                # over the top, from +y to -y
        out.append(P.box((cx + facing * depth / 2, cy + radius * math.cos(a), cz + radius * math.sin(a)),
                         (depth / 2, seg, 0.0006), color, P.turn_x(a - math.pi / 2), collides=False))
    return out


def traffic_light() -> List[BodySpec]:
    """A pedestrian light: a steel pole on a round foot, a black two-lamp
    head, a visor over each lamp on either face -- the robot looks at one and
    the fixed camera, across the table, at the other (with one face only, the
    camera saw the back of a dark head, 9/21) -- and the two lamps, bodies of
    their own so they can light: red over green."""
    x, y = ahead(*LIGHT_AT)
    hd, hw, hh = HEAD
    head = POLE_H / 2 - hh                              # the head's middle, from the body's
    pole = head - hh + POLE_H / 2                       # the pole's length under the head
    parts = [P.cylinder((0, 0, -POLE_H / 2 + pole / 2), 0.006, pole / 2, "metal:steel", collides="hidden"),
             P.box((0, 0, head), (hd, hw, hh), "metal:black", collides="hidden"),
             P.lathe((0, 0, -POLE_H / 2 + 0.003), "plinth", 0.016, 0.003, "metal:steel"),
             P.lathe((0, 0, -POLE_H / 2 + pole / 2 + 0.003), "cylinder", 0.0045, pole / 2 - 0.003,
                     "metal:steel"),
             P.slab((0, 0, head), (hd, hw, hh), "metal:black")]
    for dz in (LAMP_DZ, -LAMP_DZ):
        for facing in (-1, 1):
            parts += _visor((facing * hd, 0, head + dz), facing, LENS_R + 0.0025, 0.011, "metal:black")
    out = [BodySpec("traffic_light", "compound", (hd, hw, POLE_H / 2), Pose((x, y, TABLE + POLE_H / 2)),
                    category="traffic light", color="metal:black", fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})]
    for name, dz in (("red_lamp", LAMP_DZ), ("green_lamp", -LAMP_DZ)):
        lenses = tuple(P.lathe((facing * (hd + 0.0006), 0, 0), "cylinder", LENS_R, 0.0006, "flame:ink",
                               rot=P.turn_y(math.pi / 2)) for facing in (-1, 1))
        out.append(BodySpec(name, "compound", (hd + 0.0012, LENS_R, LENS_R),
                            Pose((x, y, TABLE + POLE_H / 2 + head + dz)), category="lamp",
                            color="flame:ink",
                            # fixed: a body with nothing that collides, left free, falls
                            # through the table on Isaac (the mock has no gravity)
                            fixed=True,
                            attrs={"parts": lenses, "collides": False, "graspable": False,
                                   "recolorable": True}))
    return out


def build_scene(level: str, seed: int):
    places = [ahead(DUCK_X, (1 - i) * DUCK_PITCH) for i in range(len(DUCKS))]
    bodies = [asset_body(d, "duck_small", xy, fallback_size=DUCK_HALF, mass=0.04,
                         category="duckling", on=TABLE + KERB_TOP) for d, xy in zip(DUCKS, places)]
    bodies += [street(), pavement(), road(), lawn()]
    bodies += traffic_light()
    return station_scene(f"di3_red_light_{level}", bodies, frame=FRAME)


def on_road(pos) -> bool:
    """Is any of a duckling at ``pos`` between the kerbs?  Along the road the
    strip has no end: a duckling taken round the end of the street is crossing
    all the same (v2.3: round the end, all three crossed on red unflagged)."""
    x, _ = ahead(ROAD_X, ROAD_Y)
    return abs(pos[0] - x) < ROAD_HALF[0] + DUCK_HALF[0]


def light_at(changes: Sequence[Tuple[float, str]], t: float) -> Tuple[str, float]:
    """The light at ``t`` and when it last changed (red from the start)."""
    state, since = "red", 0.0
    for when, s in changes:
        if t >= when:
            state, since = s, when
    return state, since


class TrafficLight(Reactor):
    """Changes the lamps on the schedule, and catches a duckling on the road
    once the light has been red for longer than the moment allowed to get
    off it."""

    def __init__(self, changes: Sequence[Tuple[float, str]], grace: float) -> None:
        super().__init__()
        self.changes, self.grace = list(changes), grace
        self._shown = ""                     # nothing lit yet: the first look lights red
        self._caught: set = set()            # (duckling, which red)
        self._flips = 0

    def tick(self, engine: Engine, world: WorldState) -> None:
        state, since = light_at(self.changes, world.sim_time)
        if state != self._shown:
            self._shown, self._flips = state, self._flips + (state == "green")
            red = state == "red"
            self.act(engine, world, f"light_{state}_{self._flips}",  # the k-th green, its red after
                     SetColor("red_lamp", RED_ON if red else LAMP_OFF))
            self.act(engine, world, f"lamp_{self._flips}",
                     SetColor("green_lamp", LAMP_OFF if red else GREEN_ON))
        if state != "red" or world.sim_time - since <= self.grace:
            return
        for d in DUCKS:
            if on_road(world.obj(d).pose.pos) and (d, self._flips) not in self._caught:
                self._caught.add((d, self._flips))
                self.note(engine, world, ON_ROAD_ON_RED, d)
                self.say(world, d, "foul", "Red light -- off the road!")


def across(w: WorldState, d: str) -> bool:
    return On(d, GRASS).holds(w) and AtRest(d).holds(w)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DI3")
    t_ref = (config or RunConfig()).t_ref_seconds
    changes = schedule(level, seed, t_ref)
    (_, longest), red, grace = LIGHT[level]
    scene = build_scene(level, seed)
    done = Custom(lambda w: all(across(w, d) for d in DUCKS), "every duckling on the grass")
    milestones = (
        Milestone("first_green", Custom(lambda w: any(e.startswith("light_green")
                                                      for e in w.events_fired), "the first green")),
        Milestone("a_duckling_picked_up", Custom(lambda w: any(w.obj(d).is_held for d in DUCKS),
                                                 "a duckling in the hand")),
        Milestone("one_across", Custom(lambda w: any(across(w, d) for d in DUCKS),
                                       "a duckling on the grass")),
        Milestone("all_across", done),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Take the ducklings across the road to the grass, one at a time. "
                    "Cross only while the light is green: no duckling on the road when it is red.",
        scorer=Scorer(task_id="DI3", level=level, success=Finally(done), milestones=milestones,
                      violations=(("on_the_road_on_red", Never(Custom(
                          lambda w: ON_ROAD_ON_RED in w.events_fired,
                          "a duckling on the road once the light had turned red"))),
                                  ("one_at_a_time", Never(Custom(
                                      lambda w: sum(on_road(w.obj(d).pose.pos) for d in DUCKS) > 1,
                                      "two ducklings on the road at once"))))),
        milestones=milestones,
        reactor_factory=lambda c=changes, g=grace: TrafficLight(c, g),
        reference_factory=lambda inst: CrossingPolicy(),
        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level],
                  "seconds_to_clear": grace,
                  # hidden from the agent: when the light will change (the gap-filled control)
                  "changes": [(round(t, 2), s) for t, s in changes[:12]],
                  "green_seconds": [round(LIGHT[level][0][0] * t_ref, 1), round(longest * t_ref, 1)],
                  # a look every 0.3 s through three whole cycles, and more
                  "max_model_calls": max(400, int(12 * len(DUCKS) * (longest + red) * t_ref))},
    )


class CrossingPolicy(ScriptedPolicy):
    """One duckling at a time: up and to the kerb whatever the light, then
    wait there for the green, and the moment it comes over the road in one
    go and down on the grass.  A green only just begun is good to go; one
    already under way when the duckling gets to the kerb is let pass.
    ``ignore_rules`` (the blind control): crosses without looking at the
    light."""

    name = "crossing"
    #: a green this young is still long enough to cross on
    FRESH = 1.0

    def __init__(self, arm: str = RIGHT) -> None:
        self.arm = arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._duck: Optional[str] = None
        self._stage = "take"
        self._greens = 0
        self._green_at = 0.0

    def _watch(self, world: WorldState, now: float) -> bool:
        """Is the light green, and green only just now?"""
        greens = sum(1 for e in world.events_fired if e.startswith("light_green"))
        if greens > self._greens:
            self._greens, self._green_at = greens, now
        seen = [e for e in world.events_fired if e.startswith("light_")]
        green = bool(seen) and seen[-1].startswith("light_green")
        return green and now - self._green_at < self.FRESH

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import grasp_offset, grasp_pose

        if ctx.extras.get("executor_busy"):
            return []
        fresh = self._watch(world, ctx.sim_time)
        left = [d for d in DUCKS if not across(world, d)]
        if not left and self._duck is None:
            return [Wait(0.5), Done("all across")]
        rx, _ = ahead(ROAD_X)
        if self._duck is None:
            self._duck, self._stage = left[0], "kerb"
            g = grasp_pose(world.obj(self._duck))
            self._quat = g.quat
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose((g.pos[0], g.pos[1], CARRY_Z), g.quat)),
                    MoveTo(self.arm, g), SetGripper(self.arm, 1.0)]
        o = world.obj(self._duck)
        hand = world.arm(self.arm).eef_pose.pos
        kerb_x = rx - ROAD_HALF[0] - DUCK_HALF[0] - KERB
        if self._stage == "kerb":
            self._stage = "wait"
            return [MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), self._quat)),
                    MoveTo(self.arm, Pose((kerb_x, hand[1], CARRY_Z), self._quat))]
        if self._stage == "wait":
            if not (self.ignore_rules or fresh):
                return [Wait(0.1)]
            self._stage = "land"
            far_x = rx + ROAD_HALF[0] + DUCK_HALF[0] + KERB
            return [MoveTo(self.arm, Pose((far_x, hand[1], CARRY_Z), self._quat))]
        # over the grass: down, let go, up
        k = DUCKS.index(self._duck)
        g = world.obj(GRASS)
        gx, gy = g.pose.pos[0], g.pose.pos[1] + (1 - k) * 0.12
        land = g.pose.pos[2] + g.size[2] + o.size[2] + grasp_offset(o) + 0.004
        self._duck, self._stage = None, "take"
        return [MoveTo(self.arm, Pose((gx, gy, CARRY_Z), self._quat)),
                MoveTo(self.arm, Pose((gx, gy, land), self._quat)), SetGripper(self.arm, 0.0), Wait(0.3),
                MoveTo(self.arm, Pose((gx, gy, CARRY_Z), self._quat))]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return [Done("the light is red")]
