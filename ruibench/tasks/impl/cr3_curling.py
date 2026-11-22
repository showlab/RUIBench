"""CR3 "curling" -- Contact-Rich Manipulation.

Knob: how near the middle of the house the stone has to stop (within 9 cm:
in the house -> within 1.5 cm: covering the button).

A tabletop curling sheet with the house painted at one end, and a stone at
the other.  "Push the stone into the house."  The stone may not be gripped or
lifted (the hint says so, as a person at the controls would be told), so it
has to be pushed -- and how is left out: the hand has to go round to the far
side of the stone from the house, come down behind it, and push along the line
through its middle; a push off that line sends it sideways, and it then has
to be pushed again from another side.  At the hard level the stone has to
cover the button, which one push seldom does.

The stone is pushed along, not sent sliding: the controller moves the hand at
one speed (0.25 m/s) whatever the interface, and a stone let go at that speed
on this sheet stops within a centimetre.  How far a thrown stone would run is
something no interface could set, so it would be luck (10/4).

On Isaac the hand pushes the stone by contact.  The mock has no pushing, so
there the task's own reactor plays it out (as WK3's dominoes): a finger that
comes inside the stone's round shoves it out along the line of their middles.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ...core.config import RunConfig
from ...core.geometry import rotate
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import GRIP_CLOSING, BodySpec, Engine, in_grasp
from ...engines.robots import arm_base, model_of, natural_yaw
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import AtRest, Custom, Finally, HoldsFor, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import ONE_ARM, RIGHT, TABLE, Frame, ahead, station_scene

STONE, SHEET, HOUSE = "stone", "sheet", "house"

#: the sheet: a board 28 cm by 54 cm, 6 mm thick, lying across the table
SHEET_AT = (0.26, 0.0)
SHEET_HALF = (0.14, 0.27, 0.003)
SHEET_TOP = TABLE + 2 * SHEET_HALF[2]
SHEET_FINISH = "plastic:white"
#: the house, painted on it: a blue ring, a white one, a red one and the
#: button, the white round in the middle
HOUSE_R, RED_R, BUTTON_R = 0.09, 0.045, 0.015
RING_BLUE = (HOUSE_R, 0.06)
#: its middle (the tee) is this far along the sheet from the sheet's middle,
#: toward one end or the other by seed
TEE_ALONG = 0.14
COAT = 0.0003                                  # how thick the paint is drawn

#: the stone: a curling stone at a fifth of its size -- a body 6 cm across
#: and 2.6 cm tall, its rim rounded, polished grey with a rough band round its
#: middle, a red plate and handle on top -- 5 cm in all, 200 g.  Turned from
#: parts: the one model of a curling stone with a usable licence (Objaverse,
#: "Curling Stone" by Lauren Herda, CC-BY) lost its handle when converted --
#: its granite showed well, but a stone without a handle is a grey puck (10/4)
STONE_R, BODY_HH = 0.03, 0.013
STONE_HALF = (STONE_R, STONE_R, 0.025)
STONE_Z = SHEET_TOP + STONE_HALF[2]
STONE_FINISH, BAND_FINISH, HANDLE_FINISH = "ceramic:grey", "ground:concrete", "plastic:red"
#: where it starts: toward the other end of the sheet from the house, and off
#: the sheet's middle line either way
START_ALONG, START_ACROSS = (0.06, 0.12), 0.06

#: the knob: how far from the tee the stone's middle may come to rest
ZONE = {"easy": HOUSE_R, "hard": BUTTON_R}
STABLE_SECONDS = 2.0
#: lifted: its bottom this far off the sheet
LIFTED = 0.005

#: the pushing hand: fingers closed, the tool centre point this far above the
#: sheet (the fingertips 5 mm above it), the fingers' flat side leading -- it
#: reaches ``HAND_HALF`` ahead of the tool centre point
PUSH_Z, HOVER, HAND_HALF = 0.013, 0.10, 0.010
#: the hand comes down this far behind the stone, and stops this far short of
#: where it would put the stone's middle on the tee: the stone runs on a few
#: millimetres after the hand stops
GAP, STOP_SHORT = 0.02, 0.003
MAX_PUSHES = 10

#: the mock's fingers, for the reactor: closed, one round this big at the tool
#: centre point; open, one at each finger
FINGER_R, FINGER_OUT = 0.012, 0.048

FRAME = Frame((0.10, 0.42), (-0.29, 0.29), 0.08)


def sheet_body() -> BodySpec:
    x, y = ahead(*SHEET_AT)
    hx, hy, hz = SHEET_HALF
    parts = [P.box((0, 0, 0), SHEET_HALF, SHEET_FINISH, collides="hidden"),
             P.slab((0, 0, 0), SHEET_HALF, SHEET_FINISH),
             # the centre line, and a hog line across each half
             P.box((0, 0, hz + COAT), (0.001, hy - 0.01, COAT), "paint:blue", collides=False)]
    parts += [P.box((0, s * 0.03, hz + COAT), (hx - 0.01, 0.0015, COAT), "paint:red", collides=False)
              for s in (-1, 1)]
    return BodySpec(SHEET, "compound", SHEET_HALF, Pose((x, y, TABLE + hz)),
                    category="curling sheet", color=SHEET_FINISH, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def house_body(tee) -> BodySpec:
    """The house, drawn only: where it is painted changes with the seed, the
    sheet does not."""
    parts = (P.lathe((0, 0, 0), "washer", RING_BLUE[0], COAT, "paint:blue", inner=RING_BLUE[1]),
             P.lathe((0, 0, 0), "washer", RED_R, COAT, "paint:red", inner=BUTTON_R),
             P.lathe((0, 0, 0), "cylinder", 0.002, COAT, "paint:black"))
    return BodySpec(HOUSE, "compound", (HOUSE_R, HOUSE_R, COAT),
                    Pose((tee[0], tee[1], SHEET_TOP + 2 * COAT)), category="house",
                    color="paint:blue", fixed=True,
                    attrs={"parts": parts, "graspable": False, "footprint": "round",
                           "collides": False})


def stone_body(xy) -> BodySpec:
    """What the hand meets is a cylinder (the stone is pushed, never held, so
    it need not be eight-sided) and the handle's bar; what is drawn is the
    turned body with its rim rounded over, a band round its middle, a plate
    and the handle."""
    h = STONE_HALF[2]
    body_z = -h + BODY_HH
    plate_z, post_z, bar_z = -h + 2 * BODY_HH + 0.003, -h + 2 * BODY_HH + 0.010, h - 0.005
    parts = (P.cylinder((0, 0, body_z), STONE_R, BODY_HH, "grey", collides="hidden"),
             P.lathe((0, 0, body_z), "washer", STONE_R, BODY_HH, STONE_FINISH, inner=0.004),
             P.lathe((0, 0, body_z), "cylinder", STONE_R + 0.0003, 0.004, BAND_FINISH),
             P.lathe((0, 0, plate_z), "plinth", 0.02, 0.003, HANDLE_FINISH),
             P.lathe((0, 0, post_z), "cylinder", 0.005, 0.004, "metal:steel"),
             P.capsule((0, 0, bar_z), 0.005, 0.014, HANDLE_FINISH, rot=P.turn_y(math.pi / 2)))
    return BodySpec(STONE, "compound", STONE_HALF, Pose((xy[0], xy[1], STONE_Z)),
                    category="curling stone", color=STONE_FINISH, mass=0.2,
                    attrs={"parts": parts, "footprint": "round", "graspable": False})


def layout(seed: int) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """(the tee, where the stone starts): the house toward one end of the
    sheet, the stone toward the other and off the middle line."""
    rng = SeedBundle(seed).stream("sheet")
    side = rng.choice((-1, 1))
    tee = ahead(SHEET_AT[0], side * TEE_ALONG)
    stone = ahead(SHEET_AT[0] + rng.uniform(-START_ACROSS, START_ACROSS),
                  -side * rng.uniform(*START_ALONG))
    return tee, stone


def build_scene(level: str, seed: int):
    tee, stone = layout(seed)
    bodies = [sheet_body(), house_body(tee), stone_body(stone)]
    return station_scene("cr3_curling", bodies, frame=FRAME), tee, stone


# -- what counts ----------------------------------------------------------------
def off_tee(w: WorldState, tee) -> float:
    return math.dist(w.obj(STONE).pose.pos[:2], tee)


def lifted(w: WorldState) -> bool:
    return w.obj(STONE).pose.pos[2] > STONE_Z + LIFTED


#: fingers stopped this much wider than they were sent are on something
SQUEEZE_STALL = 0.003


#: a grip round the stone holds the fingers this far apart at least: the
#: stone is 6 cm across
GRIPPED_OPEN = 0.01


def gripped(w: WorldState) -> bool:
    """The stone held, the fingers round it (v2.3.4): closed fingers pushing
    it were spread 2-3 mm by its rim, and the engine called that held -- a
    push on the button failed as a grip."""
    holder = w.obj(STONE).held_by
    if not holder:
        return False
    opening = w.arm(holder).opening if holder in w.arm_ids else None
    return opening is None or opening >= GRIPPED_OPEN


def squeezed(w: WorldState, stone: BodySpec) -> bool:
    """The stone gripped though not held: the fingers sent part way closed
    (``robot_gripper(width_m=...)``, below ``GRIP_CLOSING``, where no hold is
    reported) and stopped wider than sent, round the stone (v2.3: such a grip
    slid the stone with up to 19 N while ``Held`` stayed false)."""
    h = w.arm(RIGHT)
    if h.opening is None or h.gripper >= GRIP_CLOSING:
        return False
    sent = (1.0 - h.gripper) * model_of(ONE_ARM).max_opening
    return (h.opening > sent + SQUEEZE_STALL
            and in_grasp(stone, w.obj(STONE).pose, h.eef_pose.pos))


def on_sheet(w: WorldState) -> bool:
    x, y, z = w.obj(STONE).pose.pos
    sx, sy = ahead(*SHEET_AT)
    return (abs(x - sx) <= SHEET_HALF[0] and abs(y - sy) <= SHEET_HALF[1]
            and STONE_Z - 0.004 <= z <= STONE_Z + LIFTED)


def behind(w: WorldState, tee, arm: str = RIGHT) -> bool:
    """The hand down beside the stone, on its far side from the house."""
    s, tcp = w.obj(STONE).pose.pos, w.arm(arm).eef_pose.pos
    hx, hy = tcp[0] - s[0], tcp[1] - s[1]
    return (tcp[2] < SHEET_TOP + 2 * BODY_HH + 0.008 and math.hypot(hx, hy) < STONE_R + 0.05
            and hx * (s[0] - tee[0]) + hy * (s[1] - tee[1]) > 0)


class Pushing(Reactor):
    """The mock's contact: a finger inside the stone's round shoves it out
    along the line of their middles.  On an engine with dynamics, nothing."""

    def __init__(self, arm: str = RIGHT) -> None:
        super().__init__()
        self.arm = arm

    def fingers(self, world: WorldState) -> List[Tuple[float, float]]:
        hand = world.arm(self.arm)
        x, y, _ = hand.eef_pose.pos
        if hand.gripper >= 0.5:
            return [(x, y)]
        ax, ay, _ = rotate(hand.eef_pose.quat, (0.0, 1.0, 0.0))
        return [(x + s * FINGER_OUT * ax, y + s * FINGER_OUT * ay) for s in (-1, 1)]

    def tick(self, engine: Engine, world: WorldState) -> None:
        if engine.capabilities.dynamics:
            return
        stone = world.obj(STONE)
        tips = world.arm(self.arm).eef_pose.pos[2] - 0.008
        if stone.is_held or not SHEET_TOP - 0.004 < tips < stone.pose.pos[2] + STONE_HALF[2]:
            return
        x, y, z = stone.pose.pos
        for fx, fy in self.fingers(world):
            d = math.hypot(x - fx, y - fy)
            reach = STONE_R + FINGER_R
            if 1e-6 < d < reach:
                x, y = fx + (x - fx) * reach / d, fy + (y - fy) * reach / d
        if (x, y) != tuple(stone.pose.pos[:2]):
            engine.teleport(STONE, Pose((x, y, z), stone.pose.quat))


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CR3")
    scene, tee, start = build_scene(level, seed)
    zone = ZONE[level]
    far = math.dist(start, tee)

    def home(w: WorldState) -> bool:
        return off_tee(w, tee) <= zone and on_sheet(w) and AtRest(STONE).holds(w)

    there = Custom(home, "the stone at rest in the zone")
    milestones = (
        Milestone("hand_behind_the_stone", Custom(lambda w: behind(w, tee),
                                                  "the hand down behind the stone")),
        Milestone("stone_nearer", Custom(lambda w: off_tee(w, tee) <= far - 0.05 and on_sheet(w),
                                         "the stone 5 cm nearer the house")),
        Milestone("in_the_house", Custom(lambda w: off_tee(w, tee) <= HOUSE_R and on_sheet(w),
                                         "the stone's middle in the house")),
        Milestone("at_rest_in_the_zone", there),
    )
    instruction = ("Push the stone into the house." if level == "easy" else
                   "Push the stone onto the button -- the white round in the middle of the "
                   "house -- so that it covers it.")
    base = arm_base(ONE_ARM, "right", TABLE)
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=instruction,
        scorer=Scorer(task_id="CR3", level=level,
                      success=Finally(there) & HoldsFor(there, STABLE_SECONDS),
                      milestones=milestones,
                      violations=(("never_gripped", Never(Custom(gripped, "the stone gripped"))),
                                  ("never_squeezed", Never(Custom(
                                      lambda w, s=next(b for b in scene.bodies if b.name == STONE):
                                      squeezed(w, s), "the stone squeezed between the fingers"))),
                                  ("never_lifted", Never(Custom(lifted, "the stone off the sheet"))))),
        milestones=milestones,
        reactor_factory=Pushing,
        reference_factory=lambda inst, z=zone, t=tee: CurlingPolicy(t, z, base),
        metadata={"knob": spec.knob.field, "knob_value": zone, "tee": [round(v, 4) for v in tee],
                  "start_distance_m": round(far, 3), "max_model_calls": 60},
        extra_metrics={"pushes": lambda traj, t=tee: _pushes(traj, t),
                       "final_distance_m": lambda traj, t=tee: round(off_tee(traj[-1], t), 4)},
    )


def _pushes(traj, tee) -> int:
    """How many times the hand came down behind the stone."""
    down = [behind(w, tee) for w in traj]
    return sum(1 for a, b in zip(down, down[1:]) if b and not a)


# -- the reference --------------------------------------------------------------
class CurlingPolicy(ScriptedPolicy):
    """Push after push until the stone is where it should be: the hand closed,
    down behind the stone on the line from the tee through its middle, forward
    until the stone's middle would be on the tee, back off and up, look again.
    ``ignore_rules`` (the blind control): takes the stone up and carries it."""

    name = "curling"

    def __init__(self, tee, zone: float, base, arm: str = RIGHT) -> None:
        self.tee, self.zone, self.base, self.arm = tuple(tee), zone, base, arm
        #: how near the tee it leaves the stone: well inside the zone
        self.near = max(0.005, min(zone / 2, zone - 0.008))
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._pushes = 0
        self._carried = False

    def _push(self, world: WorldState) -> List[Command]:
        stone, hand = world.obj(STONE).pose.pos, world.arm(self.arm).eef_pose
        dx, dy = self.tee[0] - stone[0], self.tee[1] - stone[1]
        dist = math.hypot(dx, dy)
        ux, uy = dx / dist, dy / dist
        # the fingers' flat side leads: the tool's x along the push, either way round
        quat = turned_down(natural_yaw(self.base, stone, math.atan2(uy, ux)))
        low, high = SHEET_TOP + PUSH_Z, SHEET_TOP + HOVER
        back = STONE_R + GAP + HAND_HALF
        ahead_of = STONE_R + HAND_HALF + STOP_SHORT
        start = (stone[0] - ux * back, stone[1] - uy * back)
        end = (self.tee[0] - ux * ahead_of, self.tee[1] - uy * ahead_of)
        off = (end[0] - ux * 0.02, end[1] - uy * 0.02)
        cmds: List[Command] = [SetGripper(self.arm, 1.0)]
        if hand.pos[2] < high - 0.01:
            cmds.append(MoveTo(self.arm, Pose((hand.pos[0], hand.pos[1], high), hand.quat)))
        return cmds + [MoveTo(self.arm, Pose((start[0], start[1], high), quat)),
                       MoveTo(self.arm, Pose((start[0], start[1], low), quat)),
                       MoveTo(self.arm, Pose((end[0], end[1], low), quat)),
                       MoveTo(self.arm, Pose((off[0], off[1], low), quat)),
                       MoveTo(self.arm, Pose((off[0], off[1], high), quat)),
                       Wait(0.5, self.arm)]

    def _carry(self, world: WorldState) -> List[Command]:
        stone = world.obj(STONE).pose.pos
        down = turned_down(natural_yaw(self.base, stone, 0.0))
        grip, up = SHEET_TOP + 0.016, SHEET_TOP + HOVER
        return [SetGripper(self.arm, 0.0),
                MoveTo(self.arm, Pose((stone[0], stone[1], up), down)),
                MoveTo(self.arm, Pose((stone[0], stone[1], grip), down)),
                SetGripper(self.arm, 1.0),
                MoveTo(self.arm, Pose((stone[0], stone[1], up), down)),
                MoveTo(self.arm, Pose((self.tee[0], self.tee[1], up), down)),
                MoveTo(self.arm, Pose((self.tee[0], self.tee[1], grip + 0.002), down)),
                SetGripper(self.arm, 0.0),
                MoveTo(self.arm, Pose((self.tee[0], self.tee[1], up), down)),
                Wait(1.0, self.arm), Done("carried it there")]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self.ignore_rules:
            if self._carried:
                return [Done("carried it there")]
            self._carried = True
            return self._carry(world)
        if off_tee(world, self.tee) <= self.near:
            return [Done("the stone is home")]
        if self._pushes >= MAX_PUSHES or not on_sheet(world):
            return [Done(f"gave up after {self._pushes} pushes")]
        self._pushes += 1
        return self._push(world)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self._push(world) + [Done("one push, planned")]
