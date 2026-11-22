"""DS3 "treasure hunt" -- Discovery: a toy metal detector, hot and cold.

Knob: how many shells (four -> nine).

A wooden sandbox before the arm with scallop shells lying on the sand; a
gold coin is under one of them.  Beside the sandbox lies a toy metal detector
-- a handle with a signal bar on it, a shaft and a round search coil -- and
on the other side a little wooden bucket.  "Find the gold coin and put it in
the bucket.  Lift only one shell."  The shells look alike and nothing shows
which hides the coin; the detector does: held with its coil low over the
sand, the nearer the coil is to the coin, the more its face glows (dark,
amber, red) and the more of the five lights on its handle are lit -- hot and
cold.  Lifting a second shell is a foul.  Easy: four shells.  Hard: nine,
where sweeping every one takes long and the lights over the neighbours tell
which way to go.

Which shell hides the coin is drawn so that every shell does once in any n
consecutive seeds (n shells): a blind guess is right one time in n.

The shells are scans (two kinds of scallop, CC-BY) drawn over hidden
colliders: a low cage the coin fits under, whose sides the fingers close on.
The detector is built from parts; its lights are glowing bodies riding on
it, which the reactor recolours from the coil's distance to the coin.

v6 (2026-09-22), replacing "find the pairs" (the memory game): the advisor
found it the egg hunt (DS2) again, only harder -- lifting things to look
under them.  Here the looking is done with an instrument, and the lifting is
a single commitment.  The children's game of hot and cold, with a metal
detector.

v7 (2026-09-23): the advisor could not see the detector change.  Its
answer had been a lamp 1.8 cm across on the coil: a dot in the fixed
camera's picture, and out of the wrist camera's, which sees only the
handle.  Now the coil's whole face lights up, and a signal bar on the
handle -- in the wrist camera's picture -- fills as the coil nears the coin.

v2.3.9: the coil and the shaft collide.  Drawn only until then, a coil swept
into the shells passed through them -- a physics fault to whoever watched the
videos -- and read the coin from inside a shell at no cost.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import rotate
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...events.actions import SetColor
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import grasp_pose
from ...scoring import AtRest, Custom, Finally, In, KnownBefore, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

ARM = RIGHT
COIN, BUCKET, DETECTOR = "coin", "bucket", "detector"

# -- the sandbox ---------------------------------------------------------------------
#: its middle ahead of the station line, the sand's half size, the frame.
#: Out this far because the detector's coil is 15 cm ahead of the hand, and
#: the Franka's hand pointing down comes no nearer its base than 30 cm: over
#: the near row the hand is 14 cm ahead of the station line (32 from the base)
#: The sand reaches 8 cm past the outer shells: an open Franka finger comes
#: 6.6 cm out from the middle of what it takes, and down to the sand
BOX_AT, SAND_HALF = (0.40, 0.0), 0.19
FRAME_T, FRAME_H = 0.015, 0.042          # the boards: thickness, height over the table
SAND_TOP = TABLE + 0.03
#: the shells: a grid of this pitch, centred on the sandbox.  The Franka
#: hand over one (21 cm along its fingers' travel) passes over the next with
#: its underside 2.5 cm clear
PITCH = 0.11
GRID = {"easy": 2, "hard": 3}
KINDS = ("scallop_brown", "scallop_sea")
#: a shell's half extents as drawn (assets.lock.json), for the mock
SHELL_HALF = {"scallop_brown": (0.0325, 0.0301, 0.0131), "scallop_sea": (0.0325, 0.0322, 0.0103)}
#: the hidden cage under each shell: its walls this far out from the middle
#: (the fingers close on the two across the hand), this thick, and as tall as
#: the shell -- the coin lies inside
CAGE_OUT, CAGE_T = 0.030, 0.003
#: the coin: 2.7 cm across its flats, 1 cm thick (2.4 until v8: drawn round, the
#: hand hid 67% of it from the fixed camera) -- smaller than DS4's, so
#: that under its shell it has room to be a few millimetres off the middle
COIN_HALF = (0.0135, 0.005)
#: a shell is taken this far over the sand at least (the tool centre point;
#: the fingertips are 8 mm lower).  Taken at the middle of the thinner shell
#: the fingertips were 2 mm over the sand, dragged on it as they closed, and
#: the shell tilted in the hand and would not go down (Isaac, 9/23)
FINGERTIPS_UP = 0.013
#: a shell has been lifted once it is this far up or this far off its place
LIFTED_UP, LIFTED_OFF = 0.02, 0.04
#: where a lifted shell is set down, on the table beside the sandbox, and
#: the bucket nearer the robot
SHELL_ASIDE = (0.33, -0.31)
BUCKET_AT = (0.19, -0.31)
BUCKET_HALF = (0.0435, 0.0405, 0.065)

# -- the detector --------------------------------------------------------------------
#: body frame: the grip's middle at the origin, the coil ahead along +x
GRIP_HALF = (0.04, 0.008, 0.008)
SHAFT_LEN, SHAFT_R = 0.08, 0.005
COIL_X, COIL_R, COIL_HZ = 0.04 + 0.08 + 0.03, 0.032, 0.006
#: it lies beside the sandbox, on the robot's left, pointing away
DETECTOR_AT = (0.19, 0.33)
#: the control box between the grip and the shaft, clear of the fingers (they
#: take the grip's middle and are 2 cm wide) and right under the wrist
#: camera; on it a dark window with the signal bar: five lights in a row
#: along the shaft.  It sits on the shaft's line, not on top of the grip: its
#: top is then 6 cm from the camera, past the 5 cm the camera starts seeing
#: at -- on the grip, the nearer half of it was cut away and drew black
#: (Isaac, 9/23)
BOX_X, BOX_HALF = 0.036, (0.018, 0.013, 0.007)
WINDOW_HALF = (0.0165, 0.0105, 0.0005)
LED_PITCH, LED_HALF = 0.0064, (0.0022, 0.0085, 0.0006)
#: the lights: the coil's face, and the bar's five, lit one by one from the
#: grip end
GLOW, LEDS = "coil_glow", tuple(f"signal_light{k}" for k in range(5))
#: the coil's face: dark when cold, amber when warm, red when hot
GLOW_OFF, GLOW_WARM, GLOW_HOT = (0.06, 0.05, 0.04), (1.0, 0.55, 0.05), (1.0, 0.08, 0.03)
#: the bar's lights, unlit and lit: green, green, yellow, orange, red
LED_OFF = (0.07, 0.08, 0.07)
LED_LIT = ((0.15, 0.95, 0.20), (0.15, 0.95, 0.20), (1.0, 0.85, 0.05), (1.0, 0.45, 0.03),
           (1.0, 0.07, 0.03))
#: the detector answers within REACH of the coin -- over the next shell two
#: lights, over the diagonal one, beyond it none -- the coil no higher than
#: RANGE over the sand; STEPS: one per light
REACH, RANGE = 0.20, 0.08
STEPS = len(LEDS)
#: a sweep carries the coil this far over the shells' tops
SWEEP_ABOVE = 0.025
CARRY = TABLE + 0.16

#: the part of the table the fixed camera shows, with the gripper where it
#: starts: the sandbox, the detector and the bucket
FRAME = Frame((0.08, 0.62), (-0.40, 0.42), 0.14)


def shell_names(level: str) -> List[str]:
    return [f"shell{k}" for k in range(GRID[level] ** 2)]


def spots(level: str) -> List[Tuple[float, float]]:
    n = GRID[level]
    bx, by = BOX_AT
    return [ahead(bx + (i - (n - 1) / 2) * PITCH, by + (j - (n - 1) / 2) * PITCH)
            for i in range(n) for j in range(n)]


def hidden_under(level: str, seed: int) -> int:
    """Which shell hides the coin: every one exactly once in any n
    consecutive seeds."""
    n = GRID[level] ** 2
    group, place = divmod(seed, n)
    order = list(range(n))
    SeedBundle(group).stream(f"treasure/{n}").shuffle(order)
    return order[place]


def sandbox() -> BodySpec:
    """The sand, and the four ash boards round it: one body, so that what
    stands on the sand stands on the sand's own top (the mock's settling goes
    by parts for a body built of them; a long board alone reached, as a box,
    over half the sandbox)."""
    bx, by = ahead(*BOX_AT)
    sz = (SAND_TOP - TABLE) / 2
    hz = FRAME_H / 2
    outer = SAND_HALF + FRAME_T
    # in the body's frame, its origin at the boards' middle height
    parts = [P.box((0, 0, sz - hz), (SAND_HALF, SAND_HALF, sz), "sand:beach")]
    for dx, dy, hx, hy in ((outer - FRAME_T / 2, 0, FRAME_T / 2, outer),
                           (-outer + FRAME_T / 2, 0, FRAME_T / 2, outer),
                           (0, outer - FRAME_T / 2, SAND_HALF, FRAME_T / 2),
                           (0, -outer + FRAME_T / 2, SAND_HALF, FRAME_T / 2)):
        parts.append(P.box((dx, dy, 0), (hx, hy, hz), "wood:ash"))
    return BodySpec("sandbox", "compound", (outer, outer, hz), Pose((bx, by, TABLE + hz)),
                    category="sandbox", color="sand:beach", fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def shell(name: str, kind: str, xy) -> BodySpec:
    """A scallop lying on the sand, drawn over its hidden cage: four low walls
    and a lid, the coin's room under it."""
    hx, hy, hz = SHELL_HALF[kind]
    w, t = CAGE_OUT, CAGE_T
    parts = [P.grip((w - t / 2, 0, 0), (t / 2, w, hz)), P.grip((-w + t / 2, 0, 0), (t / 2, w, hz)),
             P.grip((0, w - t / 2, 0), (w, t / 2, hz)), P.grip((0, -w + t / 2, 0), (w, t / 2, hz)),
             P.grip((0, 0, hz - t / 2), (w, w, t / 2)),
             P.visual(kind, (0, 0, 0), (hx, hy, hz))]
    return BodySpec(name, "compound", (w, w, hz), Pose((xy[0], xy[1], SAND_TOP + hz)),
                    category="seashell", color="paint:cream", mass=0.03,
                    attrs={"parts": tuple(parts), "footprint": "rect", "container": "inverted",
                           "wall": t, "grasp_height": max(0.0, FINGERTIPS_UP - hz), "kind": kind, "angular_damping": 5.0})


def detector() -> BodySpec:
    """A toy metal detector lying on the table: a black rubber grip (a square
    core under it), the yellow control box with its dark window, a steel
    shaft and the round search coil (its lights are bodies of their own).
    The grip, the shaft and the coil collide (the last two from v2.3.9); the
    box, right behind the fingers, and the rubber sleeve are drawn only."""
    along = P.turn_y(math.pi / 2)
    gx = GRIP_HALF[0]
    box0, box1 = BOX_X - BOX_HALF[0], BOX_X + BOX_HALF[0]
    shaft1 = gx + SHAFT_LEN
    parts = [P.grip((0, 0, 0), GRIP_HALF),
             # the rubber sleeve, up to the box (the core it hides runs on into the box)
             P.cylinder(((box0 - gx) / 2, 0, 0), GRIP_HALF[1] + 0.002, (box0 + gx) / 2, "plastic:dark",
                        rot=along, collides=False),
             P.box((BOX_X, 0, 0), BOX_HALF, "plastic:yellow", collides=False),
             P.box((BOX_X, 0, BOX_HALF[2] + WINDOW_HALF[2]), WINDOW_HALF, "plastic:dark", collides=False),
             P.cylinder(((box1 + shaft1) / 2, 0, 0), SHAFT_R, (shaft1 - box1) / 2, "metal:steel", rot=along),
             P.cylinder((COIL_X, 0, -0.002), COIL_R, COIL_HZ, "plastic:dark")]
    x, y = ahead(*DETECTOR_AT)
    return BodySpec(DETECTOR, "compound", GRIP_HALF, Pose((x, y, TABLE + GRIP_HALF[2])),
                    category="metal detector", color="plastic:dark", mass=0.08,
                    # taken 4 mm over the grip's middle: at the middle itself
                    # the fingertips dragged on the table and stalled 2.9 cm
                    # apart, closed on nothing (Isaac, 9/22)
                    attrs={"parts": tuple(parts), "footprint": "rect", "grasp_height": 0.004,
                           "grasp_yaw": 0.0})


def _light(name: str, at, half, shape: str, category: str) -> BodySpec:
    """A light riding on the detector: glowing, recoloured by the reactor."""
    part = (P.cylinder((0, 0, 0), half[0], half[2], "flame:dark") if shape == "disc"
            else P.box((0, 0, 0), half, "flame:dark"))
    return BodySpec(name, "compound", half, Pose(at), category=category, color="flame:dark",
                    attrs={"parts": (part,), "collides": False, "graspable": False,
                           "attached_to": DETECTOR, "recolorable": True})


def lights() -> List[BodySpec]:
    """The coil's face, and the signal bar's five in the control box's
    window (the first nearest the grip)."""
    x, y = ahead(*DETECTOR_AT)
    z = TABLE + GRIP_HALF[2]
    glow_hz = 0.0008
    out = [_light(GLOW, (x + COIL_X, y, z + COIL_HZ - 0.002 + glow_hz),
                  (COIL_R * 0.8, COIL_R * 0.8, glow_hz), "disc", "coil light")]
    top = z + BOX_HALF[2] + 2 * WINDOW_HALF[2] + LED_HALF[2]
    first = BOX_X - (len(LEDS) - 1) / 2 * LED_PITCH
    out += [_light(name, (x + first + k * LED_PITCH, y, top), LED_HALF, "box", "signal light")
            for k, name in enumerate(LEDS)]
    return out


def build_scene(level: str, seed: int):
    names = shell_names(level)
    places = spots(level)
    under = hidden_under(level, seed)
    # which kind each shell is stays put: only the coin moves with the seed
    kinds = [KINDS[k % len(KINDS)] for k in range(len(names))]
    shells = [shell(n, kind, xy) for n, kind, xy in zip(names, kinds, places)]
    coin = T.gold_coin(COIN, places[under], SAND_TOP, half=COIN_HALF, hidden_at_start=True)
    bucket = asset_body(BUCKET, "sand_pail", ahead(*BUCKET_AT), fallback_size=BUCKET_HALF,
                        mass=0.2, fixed=True, category="bucket", container="open", wall=0.006,
                        graspable=False)
    bodies = [sandbox()] + shells + [coin, detector()] + lights() + [bucket]
    return station_scene(f"ds3_treasure_{level}", bodies, frame=FRAME), names, places, under


def coil(world: WorldState) -> Tuple[float, float, float]:
    o = world.obj(DETECTOR)
    d = rotate(o.pose.quat, (COIL_X, 0.0, 0.0))
    return tuple(p + v for p, v in zip(o.pose.pos, d))


def warmth(world: WorldState) -> float:
    """How hot the detector is, 0 to 1: the coil within ``RANGE`` over the
    sand, and near the coin."""
    x, y, z = coil(world)
    if z - SAND_TOP > RANGE:
        return 0.0
    cx, cy, _ = world.obj(COIN).pose.pos
    d = math.hypot(x - cx, y - cy)
    return max(0.0, 1.0 - d / REACH)


def found_by_the_detector(names: Sequence[str], places, under: int):
    """What the detector's lights have shown so far leave open for the shell over
    the coin: at each snapshot with the coil in range over the sand, the shells
    kept are those that, with the coin under them, would light the bar as it is
    lit -- a hot reading over the coin, or cold ones over the others, or anything
    between that tells them apart (``KnownBefore``)."""
    def narrow(w: WorldState, left: frozenset, memo: dict) -> frozenset:
        x, y, z = coil(w)
        if z - SAND_TOP > RANGE:
            return left
        shown = round(warmth(w) * STEPS)
        cx, cy, _ = w.obj(COIN).pose.pos

        def would_show(name: str) -> int:
            px, py = (cx, cy) if name == names[under] else places[list(names).index(name)][:2]
            return round(max(0.0, 1.0 - math.hypot(x - px, y - py) / REACH) * STEPS)
        return frozenset(n for n in left if would_show(n) == shown)
    return narrow


def lifted(world: WorldState, name: str, home) -> bool:
    o = world.obj(name)
    return (o.is_held or o.pose.pos[2] - o.size[2] - SAND_TOP > LIFTED_UP
            or math.dist(o.pose.pos[:2], home) > LIFTED_OFF)


FOUL = "a_second_shell_lifted"


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DS3")
    scene, names, places, under = build_scene(level, seed)
    homes = dict(zip(names, places))
    in_bucket = In(COIN, BUCKET)
    second = Custom(lambda w: FOUL in w.events_fired, "a second shell lifted")
    milestones = (
        Milestone("detector_up", Custom(lambda w: w.obj(DETECTOR).is_held, "the detector in hand")),
        Milestone("hot", Custom(lambda w: "signal_hot" in w.events_fired, "the detector reading hot")),
        Milestone("coin_uncovered", Custom(lambda w: lifted(w, names[under], places[under]),
                                           "the coin's shell lifted")),
        Milestone("coin_in_bucket", in_bucket),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Find the gold coin and put it in the bucket. Lift only one shell.",
        scorer=Scorer(task_id="DS3", level=level,
                      success=Finally(Custom(lambda w: in_bucket.holds(w) and AtRest(COIN).holds(w),
                                             "the coin at rest in the bucket")),
                      milestones=milestones,
                      violations=(("one_shell_only", Never(second)),
                                  # the coin's shell lifted only once the detector had found it:
                                  # one shell lifted blind was right one time in four (easy)
                                  ("coin_found_first",
                                   KnownBefore(names[under], tuple(names),
                                               found_by_the_detector(names, places, under),
                                               Custom(lambda w: lifted(w, names[under], places[under]),
                                                      "the coin's shell lifted"),
                                               "which shell the coin is under")))),
        milestones=milestones,
        reactor_factory=lambda h=dict(homes): Detector(h),
        reference_factory=lambda inst, n=tuple(names), h=dict(homes): TreasurePolicy(n, h),
        metadata={"knob": spec.knob.field, "knob_value": len(names),
                  # hidden from the agent; here so the gap-filled control can name it
                  "under": names[under], "max_model_calls": 40 + 8 * len(names),
                  "needs_visibility": True},
    )


def display(step: int):
    """What the detector shows at a step (0 to ``STEPS``): the coil's colour,
    and each light of the bar's."""
    f = step / STEPS
    if f <= 0.5:
        glow = tuple(a + (b - a) * 2 * f for a, b in zip(GLOW_OFF, GLOW_WARM))
    else:
        glow = tuple(a + (b - a) * (2 * f - 1) for a, b in zip(GLOW_WARM, GLOW_HOT))
    return glow, tuple(LED_LIT[k] if k < step else LED_OFF for k in range(len(LEDS)))


class Detector(Reactor):
    """The detector's lights, and the rule: they follow the coil's distance
    to the coin (in ``STEPS`` steps), and the moment a second shell is
    lifted, that is a foul."""

    def __init__(self, homes: Dict[str, Tuple[float, float]]) -> None:
        super().__init__()
        self.homes = dict(homes)
        self._step = 0
        self._lifted: List[str] = []

    def tick(self, engine: Engine, world: WorldState) -> None:
        step = round(warmth(world) * STEPS)
        if step != self._step:
            self._step = step
            glow, bar = display(step)
            for name, rgb in zip(LEDS, bar):
                SetColor(name, rgb).apply(engine)
            self.act(engine, world, f"signal_{step}", SetColor(GLOW, glow))
            if step >= STEPS - 1:
                self.note(engine, world, "signal_hot")
        for name, home in self.homes.items():
            if name not in self._lifted and lifted(world, name, home):
                self._lifted.append(name)
                if len(self._lifted) == 2:
                    self.note(engine, world, FOUL)
                    self.say(world, name, "foul", "Only one shell may be lifted.")


class TreasurePolicy(ScriptedPolicy):
    """Take the detector and sweep its coil over the shells one by one,
    watching its lights; put it back where it lay, lift the shell it read
    hottest over and set it aside, and put the coin in the bucket.  The
    blind plan (``first_frame_only``) lifts the first shell."""

    name = "treasure"

    def __init__(self, names: Sequence[str], homes: Dict[str, Tuple[float, float]]) -> None:
        self.names, self.homes = list(names), dict(homes)
        self.reset()

    def reset(self) -> None:
        self._phase = "take"
        self._next = 0
        self._readings: Dict[str, int] = {}
        self._grip: Optional[Pose] = None
        self._pick: Optional[str] = None
        self._sub = None

    def _hand_for_coil(self, world: WorldState, target) -> Pose:
        hand = world.arm(ARM).eef_pose.pos
        c = coil(world)
        return Pose(tuple(h + g - k for h, g, k in zip(hand, target, c)), GRASP_DOWN)

    def _over(self, world: WorldState, name: str) -> Pose:
        x, y = self.homes[name]
        top = SAND_TOP + 2 * max(h[2] for h in SHELL_HALF.values())
        return self._hand_for_coil(world, (x, y, top + SWEEP_ABOVE + COIL_HZ))

    @staticmethod
    def _reading(world: WorldState) -> int:
        """What the detector shows now: its step, read off the events the
        reactor wrote as it changed (what a camera sees of it)."""
        steps = [int(e.split("_")[1]) for e in world.events_fired
                 if e.startswith("signal_") and e.split("_")[1].isdigit()]
        return steps[-1] if steps else 0

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PickAndPlacePolicy, PlaceAtPointPolicy

        if ctx.extras.get("executor_busy"):
            return []
        if self._phase == "take":
            if world.obj(DETECTOR).is_held:
                self._phase = "sweep"
                return [MoveTo(ARM, Pose((*world.arm(ARM).eef_pose.pos[:2], CARRY), GRASP_DOWN))]
            self._grip = g = grasp_pose(world.obj(DETECTOR))
            return [SetGripper(ARM, 0.0), MoveTo(ARM, Pose((g.pos[0], g.pos[1], CARRY), g.quat)),
                    MoveTo(ARM, g), SetGripper(ARM, 1.0), Wait(0.5)]
        if self._phase == "sweep":
            if self._next > 0:
                self._readings[self.names[self._next - 1]] = self._reading(world)
                if self._readings[self.names[self._next - 1]] >= STEPS - 1:
                    self._next = len(self.names)          # hot: no need to sweep on
            if self._next < len(self.names):
                name = self.names[self._next]
                self._next += 1
                return [MoveTo(ARM, self._over(world, name)), Wait(0.6)]
            self._pick = max(self._readings, key=self._readings.get)
            self._phase = "put_back"
            g = self._grip
            hand = world.arm(ARM).eef_pose.pos
            return [MoveTo(ARM, Pose((hand[0], hand[1], CARRY), GRASP_DOWN)),
                    MoveTo(ARM, Pose((g.pos[0], g.pos[1], CARRY), g.quat)),
                    MoveTo(ARM, Pose((g.pos[0], g.pos[1], g.pos[2] + 0.003), g.quat)),
                    SetGripper(ARM, 0.0), Wait(0.4),
                    MoveTo(ARM, Pose((g.pos[0], g.pos[1], CARRY), g.quat))]
        if self._phase == "put_back":
            self._phase = "lift"
            # up, across and straight down: a slanting carry took the shell
            # through the sandbox's board, which knocked it out of the hand
            self._sub = PlaceAtPointPolicy(self._pick, (*ahead(*SHELL_ASIDE), TABLE), arm=ARM,
                                           lift_height=0.10, tolerance=0.02, carry_z=CARRY)
        if self._phase == "lift":
            cmds = self._sub.commands(world, ctx)
            if self._sub.attempts > MAX_GRASP_ATTEMPTS:
                return [Done(f"could not lift {self._pick}")]
            moves = [c for c in cmds if not isinstance(c, Done)]
            if moves:
                return moves
            # set aside: now the coin
            self._phase = "coin"
            self._sub = PickAndPlacePolicy(COIN, lambda w: BUCKET, arm=ARM, lift_height=0.10,
                                           place_clearance=0.004)
        if self._phase == "coin":
            cmds = self._sub.commands(world, ctx)
            if self._sub.attempts > MAX_GRASP_ATTEMPTS:
                return [Done("could not get the coin")]
            return cmds
        return [Done("done")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: no detector, the first shell lifted and set aside, and the
        hand then takes whatever lies where it was, into the bucket -- right
        one time in n."""
        from ...policies.scripted import PlaceAtPointPolicy

        first = self.names[0]
        out = [c for c in PlaceAtPointPolicy(first, (*ahead(*SHELL_ASIDE), TABLE), arm=ARM,
                                              lift_height=0.10).plan(world, ctx)
               if not isinstance(c, Done)]
        x, y = self.homes[first]
        bx, by, _ = world.obj(BUCKET).pose.pos
        grasp = SAND_TOP + COIN_HALF[1]
        out += [SetGripper(ARM, 0.0), MoveTo(ARM, Pose((x, y, CARRY), GRASP_DOWN)),
                MoveTo(ARM, Pose((x, y, grasp), GRASP_DOWN)), SetGripper(ARM, 1.0), Wait(0.4),
                MoveTo(ARM, Pose((x, y, CARRY), GRASP_DOWN)), MoveTo(ARM, Pose((bx, by, CARRY), GRASP_DOWN)),
                SetGripper(ARM, 0.0), Wait(0.5)]
        return out + [Done("guessed")]


def secrets(inst) -> Tuple[str, ...]:
    """The shell the coin is under (``TaskInstance.secrets``)."""
    return (inst.metadata["under"],)
