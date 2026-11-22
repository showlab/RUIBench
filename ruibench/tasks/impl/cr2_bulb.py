"""CR2 "screw in the bulb" -- Contact-Rich Manipulation.

Knob: how far the bulb has to be turned in its socket before the lamp lights
(a quarter turn -> two turns).

A table lamp without its shade -- a walnut foot, a brass stem, a black
socket facing up -- and a clear glass bulb standing cap down in the cardboard
collar it came in.  "Screw the light bulb into the lamp."

What to do is all in the instruction.  How is not: the bulb is taken by its
glass, its cap goes down into the socket, and it is turned clockwise seen
from above -- and at the hard level two full turns, which no wrist makes in
one go: turn, let go, turn the hand back, take hold again.  A hand that is
turned back as far as it goes before it takes hold makes the most of each
stroke.  The filament lights once the bulb is all the way in: that, and
nothing else, says it is done.

How it is built.  No thread is simulated (the colliders are convex): the
socket is a cup with a flared mouth, the cap stands in it with half a
millimetre of room, and what counts is how far the bulb has turned about its
own axis while its cap is in the socket -- judged where the cap is, so that a
bulb leaning in the bore is still in it -- clockwise adds, back takes away,
out of the socket starts again (the reactor).  The glass is round; what the
fingers close on is an eight-sided band hidden inside it, as for every round
thing here: flat pads on a round body hold it along a line and it slips.

FurnitureBench's lamp (Heo et al., 2023) has the bulb screwed in before the
shade goes on; EmbodiedSWE's assembly.bulb (Shen et al., 2026) simulates the
thread itself.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import quat_mul, rotate, yaw_of
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...events.actions import SetColor
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import Custom, Finally, HoldsFor, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

LAMP, BULB, GLOW, BOX = "lamp", "bulb", "filament", "bulb_box"

#: the lamp: a walnut foot 12 cm across, a brass stem, and a black socket whose
#: rim is 10 cm over the table: a cup 2.2 cm deep with a bore of 2.9 cm, and a
#: mouth flaring 1 cm up and out round it
FOOT_R, FOOT_H, STEM_R = 0.060, 0.018, 0.007
RIM, CUP, CUP_FLOOR = 0.100, 0.022, 0.004
BORE, SOCKET_R = 0.0145, 0.020
MOUTH_R, MOUTH_H, MOUTH_T = 0.0245, 0.010, 0.003
SEGMENTS = 16
FOOT_FINISH, STEM_FINISH, SOCKET_FINISH = "wood:walnut", "metal:brass", "plastic:black"

#: the bulb, an A60: 10.8 cm tall, its globe 6 cm across over a neck, its cap
#: 2.5 cm long and 2.6 cm across the flats of the eight-sided collider that
#: stands for its thread (2.8 cm across the corners, in a 2.9 cm bore)
BULB_H, GLOBE_R = 0.108, 0.030
CAP_H, CAP_FLAT = 0.025, 0.013
GLOBE_Z = BULB_H / 2 - GLOBE_R                  # the globe's middle, from the bulb's
#: what the fingers close on: an eight-sided band round the globe's middle,
#: its flats half a millimetre proud of the glass -- inside it, the pads met
#: the round globe first, at two points, and the bulb swung between them like
#: a bell (Isaac, 10/5)
BAND = (GLOBE_R + 0.0005, 0.016)
BULB_MASS = 0.04
#: pearl glass, milky and half seen through: clear, the bulb was two faint
#: lines to the camera across the table (Isaac, 10/5)
GLASS, CAP_FINISH, TIP_FINISH = "glass:white", "metal:aluminium", "metal:brass"
#: the filament on its stem: dark, and a warm glow once the lamp is lit
OFF, ON = "flame:dark", "flame:sunflower"

#: the cardboard collar the bulb stands in: 7 cm square, 3.5 cm tall
BOX_HALF, BOX_HOLE, BOX_H = 0.035, 0.0145, 0.035
BOX_FINISH = "paint:wood"

#: the knob: turns before the lamp lights
TURNS = {"easy": 0.25, "hard": 2.0}
#: the bulb is in the socket: its axis this near the socket's, its cap's end
#: no more than this far over the cup's floor -- 1 cm and more of the cap down
#: in the cup, where a thread would have hold of it -- and it stands this near
#: upright.  Both are measured at the cap's end, where the thread is: the
#: bulb's middle, 5.4 cm up its axis, is 3 mm off when the bulb leans 3 degrees
IN_XY, IN_Z, IN_TILT = 0.003, 0.012, math.radians(12)
#: let go in the mouth, this near the axis, it slides down into the cup
MOUTH_TAKES = MOUTH_R - MOUTH_T - CAP_FLAT
STABLE_SECONDS = 2.0
#: a hand is off the bulb: open, or this far from it
HAND_OFF = 0.08

#: where things stand: the lamp ahead of the robot and a little to its right
#: -- where the wrist turns furthest -- the bulb in its collar to the left
LAMP_AT, BOX_AT = (0.32, -0.06), (0.24, 0.20)
JITTER, STRAIGHT = 0.015, math.radians(10)

#: the reference: the hand's turn runs from ``TURNED_BACK`` (counter-clockwise)
#: down to ``TURNED_ON`` -- inside what an FR3's wrist reaches wherever it
#: stands, measured on Isaac: 180 and more one way, 100 the other.  One stroke
#: is the way between; a move is taken in steps short of half a turn, so that
#: it goes the way it is meant to -- and short of a quarter turn: the mock
#: turns a hand at once, two moves can end between two looks at the bulb, and
#: more than half a turn between two looks reads as the other way round
TURNED_BACK, TURNED_ON = math.radians(170), math.radians(-95)
STEP = math.radians(80)
#: a little past what the lamp asks -- a stroke loses a few degrees: the hand
#: stops short of where it is sent, the bulb settles on the flats as the
#: fingers close -- and, the lamp still dark after that, this much more at a
#: time, a few times: the lamp says when it is done, not the count
PAST, MORE, MORE_TIMES = math.radians(15), math.radians(30), 3
#: the bulb is carried this high over the mouth, and let go with its cap's end
#: this far down in it: it drops into the cup.  Brought down into the cup in
#: the hand, it caught on the rim, a degree off upright, and the hand pushed on
#: -- the bulb slid up between the fingers, or stood on the rim (Isaac, 10/5)
CARRY, SET_DOWN = 0.05, 0.004
FLATS = math.pi / 4

FRAME = Frame((0.10, 0.46), (-0.25, 0.30), 0.20)


# -- the lamp, the bulb and its box ---------------------------------------------
def _hidden(parts: Sequence[P.Part]) -> List[P.Part]:
    return [(kind, centre, dims, rot, color, "hidden") for kind, centre, dims, rot, color, _ in parts]


def lamp(xy) -> BodySpec:
    """The lamp, fixed to the table: what a hand meets is hidden -- the foot,
    the stem, the cup's floor and wall, the mouth's sloping wall in sixteen
    pieces -- and what is seen is turned."""
    top = RIM + MOUTH_H
    mid = top / 2

    def z(height: float) -> float:
        return height - mid

    cup_floor = RIM - CUP
    parts = [P.cylinder((0, 0, z(FOOT_H / 2)), FOOT_R, FOOT_H / 2, FOOT_FINISH, collides="hidden"),
             P.cylinder((0, 0, z((FOOT_H + cup_floor - CUP_FLOOR) / 2)), STEM_R,
                        (cup_floor - CUP_FLOOR - FOOT_H) / 2, STEM_FINISH, collides="hidden"),
             P.cylinder((0, 0, z(cup_floor - CUP_FLOOR / 2)), SOCKET_R, CUP_FLOOR / 2, SOCKET_FINISH,
                        collides="hidden")]
    parts += _hidden(P.ring((0, 0, z(RIM - CUP / 2)), SOCKET_R, BORE, CUP / 2, SOCKET_FINISH,
                            segments=SEGMENTS))
    # the mouth: a cone's wall from the bore up and out
    run, rise = MOUTH_R - MOUTH_T - BORE, MOUTH_H
    slope, lean = math.hypot(run, rise), math.atan2(run, rise)
    r_mid = BORE + run / 2 + MOUTH_T / 2 * math.cos(lean)
    z_mid = RIM + rise / 2 - MOUTH_T / 2 * math.sin(lean)
    for k in range(SEGMENTS):
        a = 2 * math.pi * k / SEGMENTS
        parts.append(P.box((r_mid * math.cos(a), r_mid * math.sin(a), z(z_mid)),
                           (MOUTH_T / 2, MOUTH_R * math.tan(math.pi / SEGMENTS), slope / 2),
                           SOCKET_FINISH, quat_mul(P.turn_z(a), P.turn_y(lean)), collides="hidden"))
    parts += [P.lathe((0, 0, z(FOOT_H / 2)), "plinth", FOOT_R, FOOT_H / 2, FOOT_FINISH),
              P.lathe((0, 0, z((FOOT_H + cup_floor - CUP_FLOOR) / 2)), "cylinder", STEM_R,
                      (cup_floor - CUP_FLOOR - FOOT_H) / 2, STEM_FINISH),
              P.lathe((0, 0, z(cup_floor - CUP_FLOOR / 2)), "cylinder", SOCKET_R, CUP_FLOOR / 2,
                      SOCKET_FINISH),
              P.lathe((0, 0, z(RIM - CUP / 2)), "washer", SOCKET_R, CUP / 2, SOCKET_FINISH, inner=BORE),
              P.lathe((0, 0, z(RIM + MOUTH_H / 2)), "funnel", MOUTH_R, MOUTH_H / 2, SOCKET_FINISH,
                      inner=BORE)]
    return BodySpec(LAMP, "compound", (FOOT_R, FOOT_R, mid), Pose((xy[0], xy[1], TABLE + mid)),
                    category="lamp", color=SOCKET_FINISH, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "round"})


def socket_floor() -> float:
    """The cup's floor, over the table."""
    return TABLE + RIM - CUP


def bulb(xy, yaw: float, z: Optional[float] = None) -> BodySpec:
    """The bulb, cap down.  The fingers close on the band hidden in its globe,
    two of whose eight flats are square to the bulb's own y."""
    h = BULB_H / 2
    cap_mid, cap_top = -h + CAP_H / 2, -h + CAP_H
    glass_h = BULB_H - CAP_H + 0.003
    # the cap and the neck above it are one eight-sided column, up into the globe
    column = (GLOBE_Z - GLOBE_R * 0.6 + h) / 2
    parts = [P.octagon((0, 0, -h + column), CAP_FLAT, column, "grey", collides="hidden"),
             P.sphere((0, 0, GLOBE_Z), GLOBE_R, "grey", collides="hidden"),
             P.octagon((0, 0, GLOBE_Z), BAND[0], BAND[1], "grey", collides="hidden"),
             P.lathe((0, 0, cap_mid + 0.001), "cylinder", CAP_FLAT + 0.0005, CAP_H / 2 - 0.001, CAP_FINISH),
             P.lathe((0, 0, -h + 0.001), "cylinder", 0.005, 0.001, TIP_FINISH),
             P.lathe((0, 0, h - glass_h / 2), "bulb", GLOBE_R, glass_h / 2, GLASS)]
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return BodySpec(BULB, "compound", (GLOBE_R, GLOBE_R, h),
                    Pose((xy[0], xy[1], TABLE + h if z is None else z), quat),
                    category="light bulb", color=GLASS, mass=BULB_MASS,
                    attrs={"parts": tuple(parts), "footprint": "round", "grasp_height": GLOBE_Z,
                           "grasp_yaw": yaw, "grasp_yaw_at": yaw, "starts_fitted": True})


def filament(xy, z: float) -> BodySpec:
    """The filament on its stem, inside the globe: it rides in the bulb, and
    the reactor lights it."""
    # drawn only, part by part: a body's "collides: False" does not reach the
    # parts of one built from them on Isaac, and a filament that collided,
    # carried along inside the globe, kicked the bulb at every step
    stem, bar = (0.004, 0.014), (0.011, 0.003, 0.003)
    parts = (P.cylinder((0, 0, 0), stem[0], stem[1], OFF, collides=False),
             P.box((0, 0, stem[1]), bar, OFF, collides=False))
    return BodySpec(GLOW, "compound", (bar[0], bar[0], stem[1] + bar[2]),
                    Pose((xy[0], xy[1], z + GLOBE_Z - 0.006)), category="filament", color=OFF,
                    attrs={"parts": parts, "collides": False, "graspable": False,
                           "attached_to": BULB, "recolorable": True})


def bulb_box(xy) -> BodySpec:
    """The cardboard collar: four walls round a square hole the cap stands in."""
    wall = (BOX_HALF - BOX_HOLE) / 2
    out = BOX_HOLE + wall
    parts = (P.box((out, 0, 0), (wall, BOX_HALF, BOX_H / 2), BOX_FINISH),
             P.box((-out, 0, 0), (wall, BOX_HALF, BOX_H / 2), BOX_FINISH),
             P.box((0, out, 0), (BOX_HOLE, wall, BOX_H / 2), BOX_FINISH),
             P.box((0, -out, 0), (BOX_HOLE, wall, BOX_H / 2), BOX_FINISH))
    return BodySpec(BOX, "compound", (BOX_HALF, BOX_HALF, BOX_H / 2),
                    Pose((xy[0], xy[1], TABLE + BOX_H / 2)), category="bulb box", color=BOX_FINISH,
                    fixed=True, attrs={"parts": parts, "graspable": False, "footprint": "rect"})


def layout(seed: int):
    """(the lamp's place, the box's, the bulb's turn in it)."""
    rng = SeedBundle(seed).stream("bulb")
    lamp_xy = ahead(LAMP_AT[0] + rng.uniform(-JITTER, JITTER), LAMP_AT[1] + rng.uniform(-JITTER, JITTER))
    box_xy = ahead(BOX_AT[0] + rng.uniform(-JITTER, JITTER), BOX_AT[1] + rng.uniform(-JITTER, JITTER))
    # the bulb looks the same whichever way it is turned; two of its hidden
    # flats start within this of across the table, where a hand taking it
    # leaves it in the fixed camera's sight between the fingers
    return lamp_xy, box_xy, rng.uniform(-STRAIGHT, STRAIGHT)


def build_scene(level: str, seed: int):
    lamp_xy, box_xy, yaw = layout(seed)
    z = TABLE + BULB_H / 2
    bodies = [lamp(lamp_xy), bulb_box(box_xy), bulb(box_xy, yaw), filament(box_xy, z)]
    return station_scene("cr2_bulb", bodies, frame=FRAME), lamp_xy


# -- what counts ----------------------------------------------------------------
def tilt(w: WorldState) -> float:
    return math.acos(max(-1.0, min(1.0, rotate(w.obj(BULB).pose.quat, (0.0, 0.0, 1.0))[2])))


def cap_end(w: WorldState) -> Tuple[float, float, float]:
    """Where the end of the bulb's cap is: half the bulb down its axis."""
    pose = w.obj(BULB).pose
    axis = rotate(pose.quat, (0.0, 0.0, 1.0))
    return tuple(p - BULB_H / 2 * a for p, a in zip(pose.pos, axis))


def in_socket(w: WorldState, lamp_xy) -> bool:
    """The cap down in the cup, the bulb standing."""
    x, y, z = cap_end(w)
    return (math.hypot(x - lamp_xy[0], y - lamp_xy[1]) <= IN_XY
            and -0.002 <= z - socket_floor() <= IN_Z and tilt(w) <= IN_TILT)


def lit(w: WorldState) -> bool:
    """Is the lamp lit now?  The reactor numbers each change: lit at the odd
    ones, out again at the even."""
    flips = [int(e.rsplit("_", 1)[1]) for e in w.events_fired if e.startswith(("lamp_lit_", "lamp_out_"))]
    return bool(flips) and max(flips) % 2 == 1


def hand_off(w: WorldState, arm: str = RIGHT) -> bool:
    hand = w.arm(arm)
    return hand.holding != BULB and (hand.gripper < 0.5 or math.dist(
        hand.eef_pose.pos, w.obj(BULB).pose.pos) > HAND_OFF)


class Thread(Reactor):
    """What stands for the thread: how far the bulb has turned in the socket.
    Clockwise seen from above screws it in, the other way out again, never
    past all the way in; taken out of the socket it starts again.  The lamp
    is lit while it is all the way in.  On an engine without contact, the
    mouth is played too: a bulb let go in it slides down into the cup."""

    def __init__(self, lamp_xy, turns: float) -> None:
        super().__init__()
        self.lamp_xy = tuple(lamp_xy)
        self.needed = 2 * math.pi * turns
        self.turned = 0.0
        self._yaw: Optional[float] = None
        self._flips = 0
        self._half = False

    def _into_the_cup(self, engine: Engine, w: WorldState) -> None:
        obj = w.obj(BULB)
        x, y, z = obj.pose.pos
        cx, cy, cz = cap_end(w)
        dx, dy = self.lamp_xy[0] - cx, self.lamp_xy[1] - cy
        low = cz <= TABLE + RIM + MOUTH_H + 0.002
        if not obj.is_held and low and 1e-4 < math.hypot(dx, dy) <= MOUTH_TAKES and tilt(w) <= IN_TILT:
            engine.teleport(BULB, Pose((x + dx, y + dy, z), obj.pose.quat))

    def tick(self, engine: Engine, world: WorldState) -> None:
        if not engine.capabilities.dynamics:
            self._into_the_cup(engine, world)
        if in_socket(world, self.lamp_xy):
            yaw = yaw_of(world.obj(BULB).pose.quat)
            if self._yaw is not None:
                step = (yaw - self._yaw + math.pi) % (2 * math.pi) - math.pi
                self.turned = min(max(self.turned - step, 0.0), self.needed)
            self._yaw = yaw
        else:
            self.turned, self._yaw = 0.0, None
        if not self._half and self.turned >= self.needed / 2:
            self._half = True
            self.note(engine, world, "half_turned")
        on = self.turned >= self.needed - math.radians(2)
        if on != (self._flips % 2 == 1):
            self._flips += 1
            self.act(engine, world, f"lamp_{'lit' if on else 'out'}_{self._flips}",
                     SetColor(GLOW, ON if on else OFF))


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CR2")
    scene, lamp_xy = build_scene(level, seed)
    turns = TURNS[level]

    def done(w: WorldState) -> bool:
        return lit(w) and in_socket(w, lamp_xy) and not w.obj(BULB).is_held and hand_off(w)

    screwed = Custom(done, "the lamp lit, the bulb standing in it and let go")
    milestones = (
        Milestone("bulb_taken", Custom(lambda w: w.obj(BULB).pose.pos[2] - BULB_H / 2 > TABLE + BOX_H,
                                       "the bulb out of its box")),
        Milestone("cap_in_socket", Custom(lambda w: in_socket(w, lamp_xy), "the cap in the socket")),
        Milestone("half_turned", Custom(lambda w: "half_turned" in w.events_fired,
                                        "half the turns made")),
        Milestone("lit_and_let_go", screwed),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Screw the light bulb into the lamp.",
        hint=spec.hint,
        scorer=Scorer(task_id="CR2", level=level,
                      success=Finally(screwed) & HoldsFor(screwed, STABLE_SECONDS),
                      milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda: Thread(lamp_xy, turns),
        reference_factory=lambda inst: ScrewPolicy(lamp_xy, turns),
        metadata={"knob": spec.knob.field, "knob_value": turns,
                  "max_model_calls": 60 if turns < 1 else 120},
        extra_metrics={"lit_at_the_end": lambda traj: bool(traj) and lit(traj[-1])},
    )


# -- the reference --------------------------------------------------------------
def _flat_at_or_below(yaw: float, top: float) -> float:
    """The hand's turn nearest below ``top`` at which its fingers close on two
    of the bulb's flats, the bulb turned ``yaw``."""
    return top - (top - yaw) % FLATS


class ScrewPolicy(ScriptedPolicy):
    """Take the bulb by its globe, carry it over the socket, let it down into
    the mouth and let go: it drops into the cup.  Take hold again, the hand
    turned back as far as the turns to come need; turn it clockwise as far as
    the wrist goes, let go, turn the hand back, take hold again, until the
    lamp is lit; let go.  ``ignore_rules`` (the stress agent): the same,
    turned the other way."""

    name = "screw_in"

    def __init__(self, lamp_xy, turns: float, arm: str = RIGHT) -> None:
        self.lamp_xy = tuple(lamp_xy)
        self.needed = 2 * math.pi * turns
        self.arm = arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._stage = "take"
        self._tries = 0
        self._made = 0.0                       # turned so far, by the hand's own count
        self._strokes = 0
        self._looks = 0
        self._more = 0

    # -- which way, and between where -------------------------------------------
    @property
    def _way(self) -> float:
        """The hand's yaw falls as it screws the bulb in; rises, the other way."""
        return 1.0 if self.ignore_rules else -1.0

    def _ends(self) -> Tuple[float, float]:
        """(where a stroke starts, where it must end by)."""
        return (TURNED_ON, TURNED_BACK) if self.ignore_rules else (TURNED_BACK, TURNED_ON)

    def _left(self) -> float:
        """How far there is still to turn, by the hand's own count; that made,
        a little more."""
        left = self.needed + PAST - self._made
        return left if left > 1e-6 else MORE

    def _start_yaw(self, world: WorldState) -> float:
        """The turn the hand takes hold at: on two flats, as far back as the
        turn still to make needs -- for a quarter turn, not far."""
        start, end = self._ends()
        left = self._left()
        far = start if left > abs(start - end) / 2 else end - self._way * (left + FLATS)
        yaw = yaw_of(world.obj(BULB).pose.quat)
        if self.ignore_rules:
            return far + (yaw - far) % FLATS
        return _flat_at_or_below(yaw, far)

    def _turn(self, pos, start: float, end: float) -> List[Command]:
        """The hand turned from ``start`` to ``end`` where it is, in steps short
        of half a turn."""
        out, yaw = [], start
        while abs(end - yaw) > 1e-6:
            yaw += math.copysign(min(STEP, abs(end - yaw)), end - yaw)
            out.append(MoveTo(self.arm, Pose(tuple(pos), turned_down(yaw))))
        return out

    def _grip(self, world: WorldState, yaw: Optional[float] = None) -> Pose:
        x, y, z = world.obj(BULB).pose.pos
        return Pose((x, y, z + GLOBE_Z),
                    turned_down(self._start_yaw(world) if yaw is None else yaw))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        obj, hand = world.obj(BULB), world.arm(self.arm)
        seated = in_socket(world, self.lamp_xy)
        if self._stage == "take":
            if obj.is_held:
                self._stage = "carry"
            else:
                self._tries += 1
                if self._tries > 3:
                    return [Done("could not take the bulb")]
                # out of the box, on the two flats nearest across the table
                flat = yaw_of(obj.pose.quat)
                grip = self._grip(world, (flat + FLATS / 2) % FLATS - FLATS / 2)
                above = (grip.pos[0], grip.pos[1], grip.pos[2] + 0.08)
                return ([SetGripper(self.arm, 0.0)]
                        + self._turn(above, yaw_of(hand.eef_pose.quat), yaw_of(grip.quat))
                        + [MoveTo(self.arm, Pose(above, grip.quat)), MoveTo(self.arm, grip),
                           SetGripper(self.arm, 1.0), Wait(0.3)])
        if self._stage == "carry":
            self._stage = "back"
            quat = hand.eef_pose.quat
            x, y, z = hand.eef_pose.pos
            held = z - (obj.pose.pos[2] - BULB_H / 2)        # the hand, over the cap's end
            high = TABLE + RIM + MOUTH_H + CARRY + held
            down = TABLE + RIM - SET_DOWN + held
            return [MoveTo(self.arm, Pose((x, y, high), quat)),
                    MoveTo(self.arm, Pose((self.lamp_xy[0], self.lamp_xy[1], high), quat)),
                    MoveTo(self.arm, Pose((self.lamp_xy[0], self.lamp_xy[1], down), quat)), Wait(0.3),
                    SetGripper(self.arm, 0.0), Wait(0.8)]
        if self._stage == "stroke":
            if not obj.is_held:
                return [Done("the bulb is not in the hand")]
            _, end = self._ends()
            yaw = yaw_of(hand.eef_pose.quat)
            to = yaw + self._way * min(self._left(), abs(end - yaw))
            self._made += abs(to - yaw)
            self._strokes += 1
            self._stage = "let_go"
            return self._turn(hand.eef_pose.pos, yaw, to) + [Wait(0.3)]
        if self._stage == "let_go":
            if lit(world):
                self._stage = "end"
            elif self._made < self.needed + PAST - 1e-6:
                self._stage = "back"
            elif self._more < MORE_TIMES:
                self._more += 1
                self._stage = "back"
            else:
                self._stage = "end"
            return [SetGripper(self.arm, 0.0), Wait(0.5)]
        if self._stage == "back":
            if not seated:
                self._looks += 1
                if self._looks > 3:
                    return [Done("the bulb is not standing in the socket")]
                return [Wait(0.5)]
            if self._strokes > 8:
                return [Done("the lamp would not light")]
            grip = self._grip(world)
            self._stage, self._looks = "stroke", 0
            return (self._turn(grip.pos, yaw_of(hand.eef_pose.quat), yaw_of(grip.quat))
                    + [MoveTo(self.arm, grip), SetGripper(self.arm, 1.0), Wait(0.3)])
        if self._stage == "end":
            self._stage = "said"
            x, y, z = hand.eef_pose.pos
            return [MoveTo(self.arm, Pose((x, y, z + 0.10), hand.eef_pose.quat)),
                    Wait(STABLE_SECONDS + 0.5)]
        return [Done("the lamp is lit" if lit(world) else "the lamp is not lit")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        grip = self._grip(world)
        return [SetGripper(self.arm, 0.0),
                MoveTo(self.arm, Pose((grip.pos[0], grip.pos[1], grip.pos[2] + 0.08), grip.quat)),
                MoveTo(self.arm, grip), SetGripper(self.arm, 1.0),
                Done("the bulb taken; the rest needs looking")]
