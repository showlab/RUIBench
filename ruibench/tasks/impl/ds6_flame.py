"""DS6 "flame test" -- Discovery, one arm (World Knowledge until 10/4).

Knob: the colours asked for (the flame green -> red, then green).

A lab bench before the FR3: a test-tube rack with four tubes of salt
solution in it, each as clear as water, in a seed-drawn order, and a wooden
splint standing in each tube, soaked in what the tube holds; a Bunsen burner
beside the rack, burning pale blue; and a beaker of water for used splints.
Easy: "Make the flame green."  Hard: "Make the flame red, then green."  Each
colour for a second.

Nothing on the bench says which tube holds what: which splint burns which
colour is found by holding its soaked end in the flame.  A wrong colour costs
nothing.  At the hard level a splint that burns green before the red has
shown is wanted again later: it has to be kept -- put back in its tube -- and
remembered.

10/4, from World Knowledge to Discovery: until then a card lay in front of
each tube with its salt's formula (NaCl, CuCl2, LiCl, KCl), and the task was
to know the chemistry -- copper green, lithium red, sodium yellow, potassium
lilac.  With the paper's suites redrawn by what has to be thought out, the
cards went: what is asked now is to find out by trying.

v7 (2026-09-23), the splint method: the advisor found the wire loop "too
many operations, too many things on the bench" -- the loop came with sodium
on it, was cleaned in the acid and burned off, wetted, dipped in a salt and
held in the outer cone, from six watch glasses on a tray.  Schools also do
the test with wooden splints soaked in the solutions, one splint to a salt,
and that is what this is now: nothing to clean.  The rack is the Poly Haven
chemistry set's (CC0); the tubes, the splints and the beaker are built from
parts, glass being see-through.

v6 (2026-09-22), still true: one arm, the chemistry set's burner at its own
size, and the hard level's two colours one after the other.

The splint is held near its top, hanging straight down from the hand, and
its soaked end goes into the flame from above.  Nothing burns: the splint's
end is tracked, and a reactor recolours the flame, a glowing body.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import rotate
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...events.actions import SetColor
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import MAX_GRASP_ATTEMPTS, grasp_pose
from ...scoring import Custom, Finally, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

ARM = RIGHT

# -- the chemistry: the salts and their flame colours (hidden: found by trying) ------
SALTS = {"NaCl": "yellow", "CuCl₂": "green", "LiCl": "red", "KCl": "lilac"}
FLAME_RGB = {"blue": (0.35, 0.55, 1.0), "green": (0.25, 0.95, 0.35), "red": (0.95, 0.12, 0.22),
             "yellow": (1.0, 0.72, 0.12), "lilac": (0.78, 0.5, 1.0)}
#: the colours asked for, in turn, each held for HOLD_SECONDS
ASKED = {"easy": ("green",), "hard": ("red", "green")}
HOLD_SECONDS = 1.0
#: every colour asked for shown, in order
DONE = "flame_test_done"
#: body names for the salts
KEY = {"NaCl": "nacl", "CuCl₂": "cucl2", "LiCl": "licl", "KCl": "kcl"}


def held(colour: str) -> str:
    return f"flame_{colour}_held"


def splint_name(salt: str) -> str:
    return f"splint_{KEY[salt]}"


# -- the layout: x ahead of the station line, y to the robot's left ----------------
#: the rack, along y: a beech rack of our own (v9; the chemistry set's until
#: then, whose holes were 2.74 cm apart and left the labels in two rows) --
#: 33 cm long, 12 cm tall, four holes 8.8 cm apart in its top board; the
#: tubes stand 1.6 cm up
RACK_AT = (0.36, 0.06)
RACK_HALF = (0.030, 0.165, 0.060)
RACK_WOOD = "wood:ash"
HOLE_PITCH, HOLE_R = 0.088, 0.011
HOLES = tuple((1.5 - k) * HOLE_PITCH for k in range(4))  # the four, from the robot's left
FOOT = 0.016                                      # the tubes' foot, over the table
#: the tubes: 15 mm glass, 15 cm tall; the solution in them 6 cm deep
TUBE_R, TUBE_H = 0.0075, 0.15
TUBE_GLASS = 0.0009                                # the solution stands this far inside the glass
SOLUTION_H = 0.06
#: a hidden square sleeve round each tube's inside holds its splint up: its
#: walls this far out from the middle, this thick
SLEEVE_IN, SLEEVE_T = 0.0068, 0.001
#: a splint: 20 cm of wood, 7 x 2.4 mm, over a square core the fingers close
#: on; its bottom 5 cm soaked (darker); taken 2.5 cm below its top, the
#: fingers across the rack
SPLINT_LEN, SPLINT_LOOK, SPLINT_CORE = 0.20, (0.0012, 0.0035), 0.002
SOAKED = 0.05
GRASP_BELOW_TOP = 0.025
#: a splint put back in its tube is let go with its end this far over the
#: tube's floor
BACK_IN_TUBE = 0.004
#: the burner on the rack's right, the beaker for used splints nearer the
#: robot beside it
BURNER_AT = (0.33, -0.19)
BEAKER_AT, BEAKER_R, BEAKER_H = (0.20, -0.18), 0.032, 0.08
#: the chemistry set's burner, 11 cm, at its own size
BURNER_H = 0.11
#: the scanned burner's half extents (assets.lock.json)
BURNER_LOOK = (0.0281, 0.0317, 0.055)
#: tall and slim, as a Bunsen flame burns
FLAME_HALF = (0.010, 0.010, 0.045)
FLAME_Z = TABLE + BURNER_H + 0.004 + FLAME_HALF[2]
#: where the splint's end counts as in the flame: near its axis, above its
#: lowest part (the gas coming out of the burner is not yet burning)
IN_FLAME_R = 0.015
IN_FLAME_Z = (FLAME_Z - 0.5 * FLAME_HALF[2], FLAME_Z + FLAME_HALF[2])
#: the splint's end held here in the flame, and carried at this height --
#: over the tubes and the flame's top
TIP_IN_FLAME = FLAME_Z + 0.01
CARRY_TIP = TABLE + 0.23
#: let go into the beaker with its end this far over the beaker's floor
DROP_TIP = TABLE + 0.03

#: the part of the table the fixed camera shows, with the gripper where it
#: starts: the rack, the burner and its flame, the beaker
FRAME = Frame((0.10, 0.47), (-0.25, 0.24), 0.23)

RACK, FLAME, BURNER, BEAKER = "rack", "flame", "burner", "beaker"


def slot_xy(k: int) -> Tuple[float, float]:
    return ahead(RACK_AT[0], RACK_AT[1] + HOLES[k])


def burner_xy() -> Tuple[float, float]:
    return ahead(*BURNER_AT)


def layout(seed: int) -> Tuple[str, ...]:
    """Which salt's tube stands in each slot, from the robot's left: a
    seed-drawn order, turned a place a seed -- in each block of four seeds
    every salt stands in every slot once (so the blind control, which takes
    the first splint, is right one time in four at the easy level)."""
    group, place = divmod(seed, len(SALTS))
    order = list(SALTS)
    SeedBundle(group).stream("flame").shuffle(order)
    return tuple(order[place:] + order[:place])


def rack_parts() -> List[tuple]:
    """The rack itself, drawn only (the sleeves hold the splints): a base, an
    upright at each end, and a top board with four holes, each with a rim."""
    hx, hy, hz = RACK_HALF
    t, post = 0.006, 0.0075
    top = hz - t
    parts = [P.box((0, 0, -hz + t), (hx, hy, t), RACK_WOOD, collides=False)]
    parts += [P.box((0, s * (hy - post), 0), (hx, post, hz), RACK_WOOD, collides=False)
              for s in (1, -1)]
    inner = hy - 2 * post
    # the top board: a strip along each side of the holes, and blocks between them
    side = (hx - HOLE_R) / 2
    parts += [P.box((s * (HOLE_R + side), 0, top), (side, inner, t), RACK_WOOD, collides=False)
              for s in (1, -1)]
    edges = [-inner] + [e for y in sorted(HOLES) for e in (y - HOLE_R, y + HOLE_R)] + [inner]
    for a, b in zip(edges[0::2], edges[1::2]):
        parts.append(P.box((0, (a + b) / 2, top), (HOLE_R, (b - a) / 2, t), RACK_WOOD,
                           collides=False))
    # a rim round each hole, over the board's square cut
    parts += [P.lathe((0, y, top + t + 0.001), "washer", HOLE_R + 0.0055, 0.001, RACK_WOOD,
                      inner=HOLE_R) for y in HOLES]
    return parts


def rack(contents: Tuple[str, ...]) -> BodySpec:
    """The rack with its four tubes of solution and the hidden sleeves that
    hold the splints up (what is where: its ``contents``)."""
    x, y = ahead(*RACK_AT)
    cz = RACK_HALF[2]                                 # the body's origin, over the table
    parts = rack_parts()
    foot = FOOT - cz
    for dy in HOLES:
        # the glass: a round bottom and the tube up from it; the solution in it
        parts += [P.lathe((0, dy, foot + TUBE_H / 2), "test_tube", TUBE_R, TUBE_H / 2, "glass:clear"),
                  P.lathe((0, dy, foot + TUBE_GLASS + SOLUTION_H / 2), "round_bottom",
                          TUBE_R - TUBE_GLASS, SOLUTION_H / 2, "glass:water")]
        # the sleeve, and a floor at the tube's bottom
        w, t, hz = SLEEVE_IN + SLEEVE_T / 2, SLEEVE_T / 2, TUBE_H / 2
        mid = foot + hz
        parts += [P.grip((w, dy, mid), (t, SLEEVE_IN + SLEEVE_T, hz)),
                  P.grip((-w, dy, mid), (t, SLEEVE_IN + SLEEVE_T, hz)),
                  P.grip((0, dy + w, mid), (SLEEVE_IN, t, hz)),
                  P.grip((0, dy - w, mid), (SLEEVE_IN, t, hz)),
                  P.grip((0, dy, foot - 0.002), (SLEEVE_IN, SLEEVE_IN, 0.002))]
    return BodySpec(RACK, "compound", RACK_HALF, Pose((x, y, TABLE + cz)),
                    category="test tube rack", color=RACK_WOOD, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect",
                           "contents": tuple(contents)})


def splint(salt: str, k: int) -> BodySpec:
    """A wooden splint standing in slot ``k``'s tube, soaked at its bottom."""
    x, y = slot_xy(k)
    h = SPLINT_LEN / 2
    lx, ly = SPLINT_LOOK
    parts = (P.grip((0, 0, 0), (SPLINT_CORE, SPLINT_CORE, h)),
             P.box((0, 0, SOAKED / 2), (lx, ly, h - SOAKED / 2), "wood:ash", collides=False),
             P.box((0, 0, -h + SOAKED / 2), (lx + 0.0002, ly + 0.0002, SOAKED / 2), "wood:oak",
                   collides=False))
    return BodySpec(splint_name(salt), "compound", (SPLINT_CORE, SPLINT_CORE, h),
                    Pose((x, y, TABLE + FOOT + h)), category="wooden splint", color="wood:ash", mass=0.004,
                    attrs={"parts": parts, "footprint": "rect", "salt": salt,
                           # the fingers close across the rack, between the tubes
                           "grasp_height": h - GRASP_BELOW_TOP, "grasp_yaw": math.pi / 2,
                           # standing in its tube: only its top 5 cm shows, as a
                           # real one's does, and that is what the hand takes
                           "hidden_at_start": True})


def beaker() -> BodySpec:
    """A glass beaker of water, for used splints: a hidden cup they fall
    into, drawn as glass."""
    x, y = ahead(*BEAKER_AT)
    r, hz, t = BEAKER_R, BEAKER_H / 2, 0.0015
    w = r - t
    parts = [P.grip((w, 0, 0), (t, r, hz)), P.grip((-w, 0, 0), (t, r, hz)),
             P.grip((0, w, 0), (r, t, hz)), P.grip((0, -w, 0), (r, t, hz)),
             P.grip((0, 0, -hz + t), (r, r, t)),
             P.vessel((0, 0, 0), r, hz, "glass:clear"),
             P.cylinder((0, 0, -hz + 0.02), r - 0.002, 0.018, "glass:water", collides=False)]
    return BodySpec(BEAKER, "compound", (r, r, hz), Pose((x, y, TABLE + hz)), category="beaker",
                    color="glass:clear", fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "round",
                           "container": "open", "wall": t})


def burner() -> BodySpec:
    """A Bunsen burner: the chemistry set's own (Poly Haven, CC0, 11 cm tall)
    over hidden colliders -- its heavy base and its barrel."""
    x, y = burner_xy()
    hz = BURNER_H / 2
    parts = [P.grip((0, 0, -hz + 0.004), (0.026, 0.026, 0.004)),
             P.grip((0, 0, 0.004), (0.008, 0.008, hz - 0.004)),
             P.visual("bunsen_burner_cc0", (0, 0, 0), BURNER_LOOK)]
    return BodySpec(BURNER, "compound", BURNER_LOOK, Pose((x, y, TABLE + hz)),
                    category="Bunsen burner", color="metal:steel", fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "round", "graspable": False})


def flame() -> BodySpec:
    """The flame: an outer and an inner cone, glowing and seen through,
    recoloured by the reactor.  v8 (9/24): shaped as a flame is (``parts``:
    "flame") and translucent (``fire:``) -- a solid glowing ellipsoid until
    then, which looked like a blue leaf standing on the burner."""
    x, y = burner_xy()
    rx, ry, hz = FLAME_HALF
    color = "fire:sky"
    # the outer cone, and the brighter inner one low in it
    parts = [P.lathe((0, 0, 0), "flame", rx, hz, color),
             P.lathe((0, 0, -hz * 0.5), "flame", rx * 0.55, hz * 0.45, color)]
    return BodySpec(FLAME, "compound", FLAME_HALF, Pose((x, y, FLAME_Z)),
                    category="flame", color=color, fixed=True,
                    attrs={"parts": tuple(parts), "collides": False, "graspable": False,
                           "recolorable": True})


def build_scene(level: str, seed: int):
    where = layout(seed)
    bodies: List[BodySpec] = [rack(where), burner(), flame(), beaker()]
    # the splints in a fixed order; the seed only says which slot each is in
    bodies += [splint(salt, where.index(salt)) for salt in SALTS]
    return station_scene(f"ds6_flame_{level}", bodies, frame=FRAME)


def tip(w: WorldState, name: str):
    """Where a splint's soaked end is."""
    o = w.obj(name)
    dx, dy, dz = rotate(o.pose.quat, (0.0, 0.0, -SPLINT_LEN / 2))
    x, y, z = o.pose.pos
    return (x + dx, y + dy, z + dz)


def in_flame(point) -> bool:
    bx, by = burner_xy()
    return math.dist(point[:2], (bx, by)) < IN_FLAME_R and IN_FLAME_Z[0] <= point[2] <= IN_FLAME_Z[1]


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DS6")
    scene = build_scene(level, seed)
    asked = ASKED[level]
    splints = tuple(splint_name(s) for s in SALTS)
    milestones = (
        Milestone("splint_taken", Custom(lambda w: any(w.obj(n).is_held for n in splints),
                                         "a splint in hand")),
        Milestone("flame_coloured", Custom(lambda w: any(e.startswith(f"{FLAME}_turned_")
                                                         and not e.endswith("_blue")
                                                         for e in w.events_fired),
                                           "a splint's colour in the flame")),
        Milestone("first_colour", Custom(lambda w, c=asked[0]: held(c) in w.events_fired,
                                         f"the flame {asked[0]} for a second")),
        Milestone("all_colours", Custom(lambda w: DONE in w.events_fired,
                                        "every colour asked for, in turn")),
    )
    if level == "easy":
        instruction = "Make the flame green."
    else:
        instruction = "Make the flame red, then green."
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=instruction,
        scorer=Scorer(task_id="DS6", level=level,
                      success=Finally(Custom(lambda w: DONE in w.events_fired,
                                             "the flame the colours asked for")),
                      milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda a=asked: FlameTest(a),
        reference_factory=lambda inst, a=asked: FlameTestPolicy(a),
        metadata={"knob": spec.knob.field, "knob_value": ", then ".join(asked),
                  # hidden from the agent: which splint burns which colour (the
                  # gap-filled control names the splint to take, by its place)
                  "flame_colours": dict(SALTS), "hold_seconds": HOLD_SECONDS,
                  "answer": {c: f"the splint {place(scene_order(scene), c)} in the rack"
                             for c in asked},
                  # as the other Discovery tasks: a dozen calls for each thing tried
                  "max_model_calls": 40 + 12 * (len(SALTS) + len(asked) - 1)},
    )


class FlameTest(Reactor):
    """The flame takes the colour of the salt on the splint whose soaked end
    is in it, and is blue again when none is.  Each colour asked for, in
    turn, for ``HOLD_SECONDS``: done."""

    def __init__(self, asked: Tuple[str, ...]) -> None:
        super().__init__()
        self.asked = tuple(asked)
        self._colour = "blue"
        self._since: Optional[float] = None
        self._shown = 0                         # colours asked for, shown so far

    def tick(self, engine: Engine, world: WorldState) -> None:
        colour = next((SALTS[s] for s in SALTS if in_flame(tip(world, splint_name(s)))), "blue")
        if colour != self._colour:
            self._colour = colour
            self.act(engine, world, f"{FLAME}_turned_{colour}", SetColor(FLAME, FLAME_RGB[colour]))
        if self._shown >= len(self.asked):
            return
        want = self.asked[self._shown]
        if self._colour != want:
            self._since = None
        elif self._since is None:
            self._since = world.sim_time
        elif world.sim_time - self._since >= HOLD_SECONDS:
            self._since = None
            self._shown += 1
            self.note(engine, world, held(want))
            if self._shown == len(self.asked):
                self.note(engine, world, DONE)


def scene_order(scene) -> Tuple[str, ...]:
    """The salts in the rack's slots, from the robot's left."""
    return tuple(next(b for b in scene.bodies if b.name == RACK).attrs["contents"])


def place(order: Sequence[str], colour: str) -> str:
    """Where the splint that burns ``colour`` stands: "first from the left"..."""
    k = order.index(next(s for s, c in SALTS.items() if c == colour))
    return ("first", "second", "third", "fourth")[k] + " from the left"


def flame_now(world: WorldState) -> str:
    """The flame's colour, as it shows."""
    turned = [e for e in world.events_fired if e.startswith(f"{FLAME}_turned_")]
    return turned[-1].rsplit("_", 1)[1] if turned else "blue"


class FlameTestPolicy(ScriptedPolicy):
    """By trying, as nothing says which splint is which: take the splints one
    by one, from the robot's left, hold each soaked end in the flame and see
    the colour it gives.  The colour wanted now is held for its second and
    the splint put in the water; one that burns a colour asked for later goes
    back into its tube, to be taken again when its turn comes; any other goes
    in the water.  Only the flame is read -- what a camera shows.
    ``ignore_rules`` (the blind control): no trying -- the first splint in the
    rack, whatever it burns, and done."""

    name = "flame_test"
    #: turns spent holding a splint in the flame before giving up on it
    PATIENCE = 8

    def __init__(self, asked: Tuple[str, ...]) -> None:
        self.asked = tuple(asked)
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._i = 0                               # which colour asked for is next
        self._phase = "take"
        self._tries = 0
        self._waited = 0
        self._name: Optional[str] = None          # the splint being taken or in hand
        self._burns: dict = {}                    # splint -> the colour it gave
        self._home: dict = {}                     # splint -> where its tube stands
        self._used: set = set()                   # splints in the water

    def _colours(self) -> Tuple[str, ...]:
        return self.asked[:1] if self.ignore_rules else self.asked

    def _next(self, world: WorldState, colour: str) -> Optional[str]:
        """The splint to take for ``colour``: one seen to burn it, else the
        next not tried yet, from the robot's left."""
        names = sorted((splint_name(s) for s in SALTS),
                       key=lambda n: -world.obj(n).pose.pos[1])
        left = [n for n in names if n not in self._used]
        known = [n for n in left if self._burns.get(n) == colour]
        if known:
            return known[0]
        fresh = [n for n in left if n not in self._burns]
        return fresh[0] if fresh else None

    @staticmethod
    def _hand_for_tip(world: WorldState, name: str, target) -> Pose:
        """The hand where the held splint's end is at ``target``, turned as
        it is."""
        hand = world.arm(ARM).eef_pose
        t = tip(world, name)
        return Pose(tuple(h + g - c for h, g, c in zip(hand.pos, target, t)), hand.quat)

    def _into_the_water(self, world: WorldState, name: str) -> List[Command]:
        bx, by = burner_xy()
        wx, wy = ahead(*BEAKER_AT)
        self._used.add(name)
        return [MoveTo(ARM, self._hand_for_tip(world, name, (bx, by, CARRY_TIP))),
                MoveTo(ARM, self._hand_for_tip(world, name, (wx, wy, CARRY_TIP))),
                MoveTo(ARM, self._hand_for_tip(world, name, (wx, wy, DROP_TIP))),
                SetGripper(ARM, 0.0), Wait(0.5),
                MoveTo(ARM, self._hand_for_tip(world, name, (wx, wy, CARRY_TIP)))]

    def _back_in_its_tube(self, world: WorldState, name: str) -> List[Command]:
        bx, by = burner_xy()
        hx, hy = self._home[name]
        down = self._hand_for_tip(world, name, (hx, hy, TABLE + FOOT + BACK_IN_TUBE))
        return [MoveTo(ARM, self._hand_for_tip(world, name, (bx, by, CARRY_TIP))),
                MoveTo(ARM, self._hand_for_tip(world, name, (hx, hy, CARRY_TIP))),
                MoveTo(ARM, down), SetGripper(ARM, 0.0), Wait(0.5),
                MoveTo(ARM, Pose((down.pos[0], down.pos[1], down.pos[2] + 0.08), down.quat))]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        colours = self._colours()
        if self._i >= len(colours):
            return [Wait(0.5), Done("the flame test is done")]
        colour = colours[self._i]
        if self._phase == "take":
            if self._name is None:
                self._name = (splint_name(world.obj(RACK).attrs["contents"][0])
                              if self.ignore_rules else self._next(world, colour))
                self._tries = 0
                if self._name is None:
                    return [Done(f"no splint left to try for {colour}")]
            name = self._name
            if world.obj(name).is_held:
                self._phase = "flame"
            else:
                self._tries += 1
                if self._tries > MAX_GRASP_ATTEMPTS:
                    return [Done(f"could not take {name}")]
                self._home.setdefault(name, tuple(world.obj(name).pose.pos[:2]))
                g = grasp_pose(world.obj(name))
                lift = g.pos[2] + (CARRY_TIP - (world.obj(name).pose.pos[2] - SPLINT_LEN / 2))
                return [SetGripper(ARM, 0.0), MoveTo(ARM, Pose((g.pos[0], g.pos[1], lift), g.quat)),
                        MoveTo(ARM, g), SetGripper(ARM, 1.0), Wait(0.5),
                        MoveTo(ARM, Pose((g.pos[0], g.pos[1], lift), g.quat))]
        name = self._name
        if self._phase == "flame":
            self._phase = "hold"
            self._waited = 0
            bx, by = burner_xy()
            return [MoveTo(ARM, self._hand_for_tip(world, name, (bx, by, CARRY_TIP))),
                    MoveTo(ARM, self._hand_for_tip(world, name, (bx, by, TIP_IN_FLAME))), Wait(0.6)]
        if self._phase == "hold":
            shown = held(colour) in world.events_fired
            seen = flame_now(world)
            if seen != "blue":
                self._burns[name] = seen
            self._waited += 1
            mine = self.ignore_rules or seen in ("blue", colour)
            if mine and not shown and self._waited <= self.PATIENCE:
                # the splint hangs from the fingers and swings: its end drifted
                # to the flame's edge, the colour came and went, and it never
                # held for a second (v13, a seed in eight) -- so put it back
                bx, by = burner_xy()
                t = tip(world, name)
                if math.dist(t[:2], (bx, by)) > 0.4 * IN_FLAME_R or not in_flame(t):
                    return [MoveTo(ARM, self._hand_for_tip(world, name, (bx, by, TIP_IN_FLAME))),
                            Wait(0.6)]
                return [Wait(0.6)]
            self._phase, self._name = "take", None
            if shown:                                 # its second is up: on to the next colour
                self._i += 1
                return self._into_the_water(world, name)
            if self.ignore_rules or seen in ("blue", colour):
                return self._into_the_water(world, name) + [
                    Done("the flame did not turn the colour asked for")]
            if seen in colours[self._i + 1:]:         # wanted later: keep it, and remember
                return self._back_in_its_tube(world, name)
            return self._into_the_water(world, name)  # not wanted at all
        return [Done("done at the bench")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return [Done("no plan without looking")]


def secrets(inst) -> Tuple[str, ...]:
    """The splints' own names, which name their salts (``TaskInstance.secrets``)."""
    return tuple(b.name for b in inst.scene.bodies if b.name.startswith("splint_"))
