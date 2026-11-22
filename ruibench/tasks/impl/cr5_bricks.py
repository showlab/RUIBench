"""CR5 "snap the bricks" -- Contact-Rich Manipulation.

Knob: how many bricks there are to add (2 -> 4).

A toddler's building bricks, the big kind -- studs 3.2 cm apart, a two-by-two
6.4 cm square.  A red two-by-four stands pressed onto a grey baseplate, and
loose bricks lie on the table to either side.  "Press the yellow brick onto
the red one so that it covers it exactly, and the blue brick onto the middle
of the yellow one"; at the hard level the green brick goes onto the blue one
and the orange one onto the green.

What to do is all in the instruction.  How is not.  A brick set down over the
studs is not yet on: it rides half a centimetre up until it is pressed home
-- and it goes down over the studs at all only when it is put within a few
millimetres of its place and square to it; further off, it sits on top of
them and no pressing moves it.  The next brick holds only on one that
is itself pressed home.

How it is built.  The studs and the hollow undersides are solid as drawn: a
stud is a peg with a rounded top, and a brick's walls go down round the studs
of the one below with 2 mm of room each side.  A brick let go a few
millimetres off slides down the rounded tops into its place -- up to about 7
mm; further off it stays up on them.  What holds a real brick is the fit of
studs in tubes, which is not simulated; what stands for it is a sprung seat
under each brick's place -- a plate nothing of which is drawn, 5 mm over the
brick below, which gives under a push of a few newtons and stays down once it
has been all the way down with its brick on it (the reactor).  A seat is
there only once the brick below is home.  So "pressed fully down" is a height
anyone can see: the gap between two bricks is closed.

The bricks are our own -- no maker's mould is copied: round-topped studs,
plain walls, glossy plastic.  Verdine et al. (2014) had three-year-olds copy models
built of such interlocking bricks (the Test of Spatial Assembly).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import rotate, yaw_of
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...engines.robots import arm_base
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import PlacePosePolicy, fold_half_turn, turned_down
from ...scoring import Custom, Finally, HoldsFor, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import ONE_ARM, RIGHT, TABLE, Frame, ahead, station_scene

PLATE, RED = "baseplate", "red"
#: the loose bricks, in the order they go on, and how many studs long each is
#: (all are two wide)
BRICKS = ("yellow", "blue", "green", "orange")
LONG = {RED: 4, "yellow": 4, "blue": 2, "green": 2, "orange": 2}
COUNT = {"easy": 2, "hard": 4}
FINISH = {name: f"plastic:{name}" for name in (RED,) + BRICKS}
PLATE_FINISH = "plastic:grey"

#: the brick: studs 3.2 cm apart; a body 3.84 cm tall with walls 4.4 mm thick
#: under a 4 mm roof.  A stud is a peg 1.92 cm across, straight for 7 mm and
#: rounded over above -- a half ball -- 1.66 cm tall in all.  The walls stand
#: 2 mm clear of the studs they go round
PITCH = 0.032
BODY_H, WALL, ROOF = 0.0384, 0.0044, 0.004
STUD_R, STUD_STRAIGHT = 0.0096, 0.007
STUD_H = STUD_STRAIGHT + STUD_R
CLEAR = PITCH - WALL - PITCH / 2 - STUD_R
MASS = {4: 0.045, 2: 0.025}
#: how far off a brick let go over its place still slides home: a wall's edge
#: on a stud's rounded top slides off it where the top is steeper than about
#: 30 degrees (the bricks' friction is the engine's 0.5; a seat's is next to
#: nothing, so the side of the brick standing on it does not hold it back)
SLIDES_IN = 0.007
SEAT_FRICTION = 0.05
#: the baseplate: four studs by six, 8 mm thick
PLATE_STUDS, PLATE_T = (6, 4), 0.008

#: a seat: how far over the brick below it holds its brick, its spring -- 0.5
#: mm of sag under a two-by-four, 4 N at the bottom -- and how far down, for
#: how long, makes it stay
REST = 0.005
SEAT_T, SEAT_MASS, SEAT_STIFFNESS, SEAT_DAMPING = 0.002, 0.01, 800.0, 15.0
#: ...and it gives only under a brick that is over its place.  Under one that
#: is not -- a stud along, say, where its walls also clear the studs -- the
#: seat is as good as solid: the brick does not go down, and the hand that
#: presses it is stopped.  (Sprung there too, the brick went down under the
#: hand and came up again behind it: Isaac, 10/5)
SEAT_STIFF = (2.0e4, 2.0e3)
LATCH_AT, DWELL = 0.004, 0.3
#: latched, the seat is this far under the top it was over; not yet wanted --
#: the brick below is not home -- it waits this far down inside the red brick
UNDER, PARKED = 0.0005, 0.010
#: how fast a seat comes up to its place, and how long its brick may be gone
#: before it lets go
SEAT_SPEED, GONE = 0.10, 0.5

#: a brick is home: this near the middle of the one below, this near its top,
#: square to it within this
HOME_XY, HOME_Z, HOME_YAW = 0.0035, 0.0015, math.radians(6)
#: ...and over its place, as the seat sees it: within what slides home
OVER_XY, OVER_YAW = SLIDES_IN, math.radians(10)
#: the tower stands when every brick has been home this long with the hand
#: off it: the fingertips this far over the top brick's studs, or the hand
#: this far to the side.  (Not "at rest" by the engine's speeds: bricks pushed
#: to the end of their 2 mm of room stood still and were reported moving at
#: up to 14 mm/s for as long as they stood -- Isaac, 10/5)
STABLE_SECONDS = 2.0
HAND_OFF = (0.015, 0.10)

#: where things stand: the baseplate ahead of the robot, its long side and the
#: red brick's pointing away from it; loose bricks in two rows either side,
#: lying the same way, far enough out for the hand that takes one (20.8 cm
#: long along its fingers' travel) to stay clear of the plate
PLATE_AT = (0.30, 0.0)
SLOTS = ((0.21, -0.20), (0.38, -0.20), (0.21, 0.20), (0.38, 0.20))
JITTER, PLATE_JITTER = 0.010, (0.010, 0.012)
STRAIGHT, PLATE_TURN = math.radians(8), math.radians(5)

#: the reference: a brick is let go this far over the top it goes onto -- 2 mm
#: over its seat, its walls already round the studs -- carried this high over
#: what it is taken from and put on, and pressed with closed fingers over its
#: middle, on the four studs round it, sent this far past home (what stops
#: them is the brick below).  Pressed on two studs to one side, a brick that
#: was riding on the studs tipped and walked off (Isaac, 10/5)
LET_GO, LIFT = REST + 0.002, 0.10
FINGERTIP, PRESS_PAST, PRESS_ABOVE = 0.008, 0.002, 0.05
#: closed fingers: 2 cm along the hand's x, 5.2 cm along its y
FINGERS = (0.010, 0.026)
#: let go this near its place, a brick is put on: what is left is to press it
#: (whether it went down is read off the gap under it, after the press)
PUT_ON = 0.010
MAX_PRESSES = 3
#: the stress agent puts each brick down this far along from its place -- half
#: a stud: one end wall on top of a row of studs -- and lets it go just above
#: them.  (Turned 20 degrees instead, the brick stood tipped on the round tops
#: and three presses in six flicked it off the tower: Isaac, 10/5)
ASKEW = PITCH / 2

#: what the fixed camera must show: the plate, the bricks to either side and
#: the tower, five bricks tall
FRAME = Frame((0.10, 0.50), (-0.285, 0.285), 0.23)

SAID = {
    "easy": "Press the yellow brick onto the red one so that it covers it exactly, and the blue "
            "brick onto the middle of the yellow one.",
    "hard": "Press the yellow brick onto the red one so that it covers it exactly, the blue brick "
            "onto the middle of the yellow one, the green brick onto the blue one and the orange "
            "brick onto the green one.",
}


# -- the bricks -------------------------------------------------------------------
def half(name: str) -> Tuple[float, float, float]:
    """A brick's half extents, studs apart: long along its own x."""
    return (LONG[name] * PITCH / 2, PITCH, BODY_H / 2)


def stud_places(n_long: int, n_wide: int = 2) -> List[Tuple[float, float]]:
    return [((i - (n_long - 1) / 2) * PITCH, (j - (n_wide - 1) / 2) * PITCH)
            for i in range(n_long) for j in range(n_wide)]


def _studs(n_long: int, n_wide: int, top: float, color, skip=()) -> List[P.Part]:
    """Studs standing on ``top``: each a straight peg and a ball on it, half
    of which is its rounded end -- drawn as one turned peg."""
    out = []
    for x, y in stud_places(n_long, n_wide):
        if any(abs(x - sx) < 1e-6 and abs(y - sy) < 1e-6 for sx, sy in skip):
            continue
        out += [P.cylinder((x, y, top + STUD_STRAIGHT / 2), STUD_R, STUD_STRAIGHT / 2, color,
                           collides="hidden"),
                P.sphere((x, y, top + STUD_STRAIGHT), STUD_R, color, collides="hidden"),
                P.lathe((x, y, top + STUD_H / 2), "dowel", STUD_R, STUD_H / 2, color)]
    return out


def _wall(normal: float, out: float, length: float, color) -> List[P.Part]:
    """One wall of a hollow brick, its outer face ``out`` from the middle
    toward ``normal`` (an angle about z) and ``length`` half long."""
    turn = P.turn_z(normal - math.pi / 2)
    tall = (BODY_H - ROOF) / 2
    centre = rotate(turn, (0.0, out - WALL / 2, -BODY_H / 2 + tall))
    return [P.box(centre, (length, WALL / 2, tall), color, turn)]


def fingers_on(name: str, lx: float = 0.0, ly: float = 0.0, turn: float = 0.0) -> float:
    """How high over a brick's roof closed fingers come to rest on its studs:
    the fingers' flat end (``FINGERS``), its middle at (``lx``, ``ly``) on the
    brick -- in the brick's own frame -- and turned ``turn`` from it.  Over
    the middle of the brick they stand on the shoulders of the four studs
    round it, 2 mm below their tops; between studs, on the roof."""
    c, sn = math.cos(turn), math.sin(turn)
    best = 0.0
    for sx, sy in stud_places(LONG[name]):
        dx, dy = sx - lx, sy - ly
        fx, fy = c * dx + sn * dy, -sn * dx + c * dy
        d = math.hypot(max(abs(fx) - FINGERS[0], 0.0), max(abs(fy) - FINGERS[1], 0.0))
        if d < STUD_R:
            best = max(best, STUD_STRAIGHT + math.sqrt(STUD_R ** 2 - d ** 2))
    return best


def hollow_parts(name: str) -> Tuple[P.Part, ...]:
    """A loose brick: a roof, four walls round a hollow open underneath, and
    studs on top."""
    hx, hy, h = half(name)
    color = FINISH[name]
    parts = [P.box((0, 0, h - ROOF / 2), (hx, hy, ROOF / 2), color)]
    for normal, out, length in ((0.0, hx, hy), (math.pi, hx, hy),
                                (math.pi / 2, hy, hx), (-math.pi / 2, hy, hx)):
        parts += _wall(normal, out, length, color)
    return tuple(parts + _studs(LONG[name], 2, h, color))


def brick(name: str, xy, yaw: float) -> BodySpec:
    """A loose brick on the table, turned ``yaw``.  The hand takes it across
    its width -- the fingers close along its own y."""
    size = half(name)
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return BodySpec(name, "compound", size, Pose((xy[0], xy[1], TABLE + size[2]), quat),
                    category=f"{name} brick", color=FINISH[name], mass=MASS[LONG[name]],
                    attrs={"parts": hollow_parts(name), "footprint": "rect", "grasp_height": 0.0,
                           "grasp_yaw": yaw, "grasp_yaw_at": yaw, "kind": "brick"})


def base_bodies(xy, yaw: float) -> Tuple[BodySpec, BodySpec]:
    """(the baseplate, the red brick pressed onto its middle): both fixed."""
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    nx, ny = PLATE_STUDS
    plate_half = (nx * PITCH / 2, ny * PITCH / 2, PLATE_T / 2)
    under_red = stud_places(LONG[RED])
    plate_parts = ([P.box((0, 0, 0), plate_half, PLATE_FINISH)]
                   + _studs(nx, ny, PLATE_T / 2, PLATE_FINISH, skip=under_red))
    plate = BodySpec(PLATE, "compound", plate_half, Pose((xy[0], xy[1], TABLE + PLATE_T / 2), quat),
                     category="baseplate", color=PLATE_FINISH, fixed=True,
                     attrs={"parts": tuple(plate_parts), "graspable": False, "footprint": "rect"})
    size = half(RED)
    red_parts = [P.box((0, 0, 0), size, FINISH[RED])] + _studs(LONG[RED], 2, size[2], FINISH[RED])
    red = BodySpec(RED, "compound", size, Pose((xy[0], xy[1], TABLE + PLATE_T + size[2]), quat),
                   category="red brick", color=FINISH[RED], fixed=True,
                   attrs={"parts": tuple(red_parts), "graspable": False, "footprint": "rect"})
    return plate, red


def level_top(k: int) -> float:
    """The top a tower's ``k``-th loose brick goes onto: the red brick's for
    the first, each brick's own for the next."""
    return TABLE + PLATE_T + BODY_H * (k + 1)


def seat_name(name: str) -> str:
    return f"{name}_seat"


def seat(name: str, k: int, order: Sequence[str], xy, yaw: float) -> Tuple[BodySpec, ArticulationSpec]:
    """The sprung seat under a brick's place: a plate the size of the brick,
    nothing of it drawn, which only that brick meets.  It waits inside the red
    brick until the brick below is home."""
    hx, hy, _ = half(name)
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    size = (hx, hy, SEAT_T / 2)
    # ...and the other seats, which wait in the same place
    others = tuple(n for n in (PLATE, RED) + tuple(order) if n != name) \
        + tuple(seat_name(n) for n in order if n != name)
    body = BodySpec(seat_name(name), "compound", size,
                    Pose((xy[0], xy[1], level_top(k) + REST - SEAT_T / 2), quat), mass=SEAT_MASS,
                    attrs={"parts": (P.box((0, 0, 0), size, "grey", collides="hidden"),),
                           "graspable": False, "footprint": "rect", "carried_by": seat_name(name),
                           "passes": others, "friction": SEAT_FRICTION, "starts_fitted": True})
    down = k * BODY_H + REST + PARKED if k else REST + 2 * UNDER
    joint = ArticulationSpec(seat_name(name), "slide", axis=(0.0, 0.0, 1.0), limits=(-down, 0.0),
                             initial=-down if k else 0.0, actuated=False,
                             stiffness=SEAT_STIFFNESS, damping=SEAT_DAMPING,
                             parent=seat_name(name), carries=True)
    return body, joint


def layout(level: str, seed: int):
    """(the plate's place and turn, {brick: (place, turn)}): the bricks of this
    level dealt over the four places beside the plate."""
    rng = SeedBundle(seed).stream("bricks")
    plate = ahead(PLATE_AT[0] + rng.uniform(-PLATE_JITTER[0], PLATE_JITTER[0]),
                  PLATE_AT[1] + rng.uniform(-PLATE_JITTER[1], PLATE_JITTER[1]))
    turn = rng.uniform(-PLATE_TURN, PLATE_TURN)
    slots = list(SLOTS)
    rng.shuffle(slots)
    bricks = {}
    for name, (sx, sy) in zip(BRICKS, slots):
        place = ahead(sx + rng.uniform(-JITTER, JITTER), sy + rng.uniform(-JITTER, JITTER))
        bricks[name] = (place, rng.uniform(-STRAIGHT, STRAIGHT))
    return plate, turn, {n: bricks[n] for n in BRICKS[:COUNT[level]]}


def build_scene(level: str, seed: int):
    plate_xy, turn, bricks = layout(level, seed)
    order = tuple(bricks)
    bodies = list(base_bodies(plate_xy, turn))
    joints = []
    for k, name in enumerate(order):
        body, joint = seat(name, k, order, plate_xy, turn)
        bodies.append(body)
        joints.append(joint)
    bodies += [brick(name, xy, yaw) for name, (xy, yaw) in bricks.items()]
    scene = station_scene(f"cr5_bricks_{level}", bodies, articulations=joints, frame=FRAME)
    return scene, order


# -- what counts ----------------------------------------------------------------
def below(order: Sequence[str], name: str) -> str:
    k = order.index(name)
    return order[k - 1] if k else RED


def _turn_off(w: WorldState, order: Sequence[str], name: str) -> float:
    """How far a brick is from square to the one below: a two-by-two looks
    the same a quarter turn on, a two-by-four half a turn."""
    sym = math.pi / 2 if LONG[name] == 2 else math.pi
    d = (yaw_of(w.obj(name).pose.quat) - yaw_of(w.obj(below(order, name)).pose.quat)) % sym
    return min(d, sym - d)


def off_middle(w: WorldState, order: Sequence[str], name: str) -> Tuple[float, float]:
    """How far a brick's middle is from the middle of the one below, along
    that brick and across it.  Each brick is measured from the one it stands
    on: a brick that is home may sit 2 mm off, and the next one on it again."""
    under = w.obj(below(order, name)).pose
    dx, dy = (w.obj(name).pose.pos[i] - under.pos[i] for i in (0, 1))
    yaw = yaw_of(under.quat)
    return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)


def gap(w: WorldState, order: Sequence[str], name: str) -> float:
    """How far a brick's underside is above the top it goes onto."""
    return w.obj(name).pose.pos[2] - w.obj(below(order, name)).pose.pos[2] - BODY_H


def over_place(w: WorldState, order: Sequence[str], name: str) -> bool:
    """Near enough its place to slide home down the studs' rounded tops, and
    square enough."""
    return (max(abs(v) for v in off_middle(w, order, name)) <= OVER_XY
            and _turn_off(w, order, name) <= OVER_YAW)


def home(w: WorldState, order: Sequence[str], name: str) -> bool:
    """Over the middle of the brick below, square to it, and down on it."""
    return (not w.obj(name).is_held and math.hypot(*off_middle(w, order, name)) <= HOME_XY
            and abs(gap(w, order, name)) <= HOME_Z and _turn_off(w, order, name) <= HOME_YAW)


def homes(w: WorldState, order: Sequence[str]) -> int:
    """How many bricks are home, counted up from the red one: a brick on one
    that is not home is not on the tower."""
    n = 0
    for name in order:
        if not home(w, order, name):
            break
        n += 1
    return n


def hand_off(w: WorldState, order: Sequence[str], arms: Sequence[str] = (RIGHT,)) -> bool:
    """No hand on the tower: none over its top brick, down near the studs."""
    top = w.obj(order[-1]).pose.pos
    for arm in arms:
        x, y, z = w.arm(arm).eef_pose.pos
        over = math.hypot(x - top[0], y - top[1]) <= HAND_OFF[1]
        if over and z - FINGERTIP <= top[2] + BODY_H / 2 + STUD_H + HAND_OFF[0]:
            return False
    return True


class Clutch(Reactor):
    """What stands for the grip of studs in a brick: the seats.  A brick's
    seat comes up to its place once the brick below is home; it gives under
    its brick and, held all the way down with the brick on it, stays down --
    the brick is on.  A brick taken off again lets its seat go, and those
    above it.  On an engine without contact two things are played here too:
    the studs' rounded tops -- a brick let go over its place, within what
    slides home, goes to where its walls clear the studs -- and the pressing:
    the seat goes down with the brick a hand brings down on it, held or under
    closed fingers."""

    def __init__(self, order: Sequence[str], arms: Sequence[str] = (RIGHT,)) -> None:
        super().__init__()
        self.order = tuple(order)
        self.arms = tuple(arms)
        self.on = {name: False for name in order}
        self._down_for = {name: 0.0 for name in order}
        self._gone_for = {name: 0.0 for name in order}
        self._target: Dict[str, Optional[float]] = {name: None for name in order}
        self._gives: Dict[str, Optional[bool]] = {name: None for name in order}
        self._then: Optional[float] = None

    def _over(self, w: WorldState, name: str) -> bool:
        return over_place(w, self.order, name)

    def _slid_in(self, engine: Engine, w: WorldState, name: str, k: int) -> None:
        """Without contact: a brick let go on the studs of its place, no
        further off than slides home, slides down them."""
        obj, under = w.obj(name), w.obj(below(self.order, name)).pose
        lx, ly = off_middle(w, self.order, name)
        low = gap(w, self.order, name) <= STUD_H + 0.001
        if obj.is_held or not low or not self._over(w, name) or max(abs(lx), abs(ly)) <= CLEAR:
            return
        lx, ly = (math.copysign(min(abs(v), 0.9 * CLEAR), v) for v in (lx, ly))
        yaw = yaw_of(under.quat)
        x = under.pos[0] + math.cos(yaw) * lx - math.sin(yaw) * ly
        y = under.pos[1] + math.sin(yaw) * lx + math.cos(yaw) * ly
        engine.teleport(name, Pose((x, y, obj.pose.pos[2]), obj.pose.quat))

    def _played(self, w: WorldState, name: str, k: int) -> float:
        """Without contact: how far down a hand has its brick -- the one it
        holds, lowered past its seat, or the one its closed fingers are on."""
        obj = w.obj(name)
        if not self._over(w, name):
            return 0.0
        rest = level_top(k) + REST
        if obj.is_held:
            return min(max(rest - (obj.pose.pos[2] - BODY_H / 2), 0.0), REST)
        hx, hy, _ = half(name)
        down = 0.0
        for arm in (w.arm(a) for a in self.arms):
            if arm.gripper < 0.5 or arm.holding:
                continue
            dx, dy = (arm.eef_pose.pos[i] - obj.pose.pos[i] for i in (0, 1))
            yaw = yaw_of(obj.pose.quat)
            lx = math.cos(yaw) * dx + math.sin(yaw) * dy
            ly = -math.sin(yaw) * dx + math.cos(yaw) * dy
            if abs(lx) <= hx and abs(ly) <= hy:
                # where the fingers meet the studs, they have the brick this far down
                on = fingers_on(name, lx, ly, yaw_of(arm.eef_pose.quat) - yaw)
                tips = arm.eef_pose.pos[2] - FINGERTIP
                down = max(down, rest + BODY_H + on - tips)
        return min(max(down, 0.0), REST)

    def tick(self, engine: Engine, world: WorldState) -> None:
        dt = max(0.0, world.sim_time - self._then) if self._then is not None else 0.0
        self._then = world.sim_time
        played = not engine.capabilities.dynamics
        for k, name in enumerate(self.order):
            if played:
                self._slid_in(engine, world, name, k)
            joint = seat_name(name)
            value = world.articulations.get(joint, 0.0)
            parked = -(k * BODY_H + REST + PARKED)
            wanted = k == 0 or self.on[self.order[k - 1]]
            if not wanted:
                if self.on[name]:
                    self.on[name] = False
                    self.note(engine, world, f"{name}_off")
                goal = parked
            elif self.on[name]:
                goal = -(REST + UNDER)
                there = self._over(world, name) and gap(world, self.order, name) <= REST / 2
                self._gone_for[name] = 0.0 if there else self._gone_for[name] + dt
                if self._gone_for[name] >= GONE:
                    self.on[name] = False
                    self.note(engine, world, f"{name}_off")
            else:
                goal = -self._played(world, name, k) if played else 0.0
                gives = self._over(world, name)
                if gives != self._gives[name]:
                    self._gives[name] = gives
                    engine.set_joint_drive(joint, *((SEAT_STIFFNESS, SEAT_DAMPING) if gives
                                                    else SEAT_STIFF))
                down = (value <= -LATCH_AT and self._over(world, name)
                        and abs(gap(world, self.order, name)) <= HOME_Z)
                self._down_for[name] = self._down_for[name] + dt if down else 0.0
                if self._down_for[name] >= DWELL:
                    self.on[name] = True
                    self._gone_for[name] = 0.0
                    self.note(engine, world, f"{name}_on")
                    goal = -(REST + UNDER)
            # a seat travels to where it is wanted at its own pace; pressed, it
            # is where the brick has it
            now = self._target[name]
            if now is None:
                now = value
            step = SEAT_SPEED * dt
            near = abs(goal - now) <= REST + UNDER
            now = goal if near else now + math.copysign(min(step, abs(goal - now)), goal - now)
            self._target[name] = now
            engine.set_articulation_target(joint, now)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CR5")
    scene, order = build_scene(level, seed)
    n = len(order)

    def done(w: WorldState) -> bool:
        return homes(w, order) == n and hand_off(w, order)

    built = Custom(done, "every brick home, the hand off the tower")
    milestones = (
        Milestone("first_brick_home", Custom(lambda w: homes(w, order) >= 1, "the first brick home")),
        Milestone("half_home", Custom(lambda w: homes(w, order) >= (n + 1) // 2,
                                      "half of the bricks home")),
        Milestone("all_home", Custom(lambda w: homes(w, order) == n, "every brick home")),
        Milestone("hand_off", built),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction=SAID[level],
        hint=spec.hint,
        scorer=Scorer(task_id="CR5", level=level,
                      success=Finally(built) & HoldsFor(built, STABLE_SECONDS),
                      milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda o=order: Clutch(o),
        reference_factory=lambda inst: BuildPolicy(order),
        metadata={"knob": spec.knob.field, "knob_value": n, "max_model_calls": 40 + 15 * n},
        extra_metrics={"bricks_home": lambda traj: homes(traj[-1], order) if traj else 0},
    )


# -- the reference --------------------------------------------------------------
class BuildPolicy(ScriptedPolicy):
    """Brick by brick, from the red one up: take the brick, bring it square
    over its place and let it go just above its seat; then close the fingers
    and press down over its middle until it is home.  ``ignore_rules`` (the
    stress agent): put each brick down askew -- half a stud along from its
    place -- and press: it rides on the studs, a press shoves it on to the
    next stud at most, and no pressing takes it down."""

    name = "press_bricks"

    def __init__(self, order: Sequence[str], arm: str = RIGHT) -> None:
        self.order = tuple(order)
        self.arm = arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._k = 0
        self._stage = "place"
        self._place: Optional[PlacePosePolicy] = None
        self._presses = 0

    def _placing(self, world: WorldState, name: str) -> PlacePosePolicy:
        if self._place is None:
            # over the brick it goes on, where that one sits -- which may be 2 mm
            # off the one below it -- not over the red brick
            under = world.obj(below(self.order, name)).pose
            top = under.pos[2] + BODY_H / 2
            yaw = yaw_of(under.quat)
            off = ASKEW if self.ignore_rules else 0.0
            at = (under.pos[0] + off * math.cos(yaw), under.pos[1] + off * math.sin(yaw), top)
            self._place = PlacePosePolicy(
                name, at, yaw, arm=self.arm,
                symmetry=math.pi / 2 if LONG[name] == 2 else math.pi,
                release_clearance=STUD_H + 0.002 if self.ignore_rules else LET_GO,
                tolerance=PUT_ON, yaw_tolerance=math.radians(3),
                base=arm_base(ONE_ARM, "right", TABLE), refine=True, release_tolerance=0.0015,
                lift=LIFT)
        return self._place

    def _press(self, world: WorldState, name: str) -> List[Command]:
        """Closed fingers down over the brick's middle, their width across the
        brick: 2 cm along it and 5.2 cm across, they come down on the
        shoulders of the four studs round the middle."""
        obj = world.obj(name)
        yaw = yaw_of(obj.pose.quat)
        x, y = obj.pose.pos[0], obj.pose.pos[1]
        home_top = (world.obj(below(self.order, name)).pose.pos[2] + 1.5 * BODY_H
                    + fingers_on(name))
        # the hand is the same half a turn on: the turn nearest straight
        down = turned_down(fold_half_turn(yaw))
        self._presses += 1
        return [SetGripper(self.arm, 1.0),
                MoveTo(self.arm, Pose((x, y, home_top + FINGERTIP + PRESS_ABOVE), down)),
                MoveTo(self.arm, Pose((x, y, home_top + FINGERTIP - PRESS_PAST), down)),
                Wait(DWELL + 0.3),
                MoveTo(self.arm, Pose((x, y, home_top + FINGERTIP + PRESS_ABOVE), down)),
                Wait(0.3)]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._k >= len(self.order):
            if self._stage != "end":
                self._stage = "end"
                return [Wait(STABLE_SECONDS + 0.5)]
            n = homes(world, self.order)
            return [Done("the bricks are on" if n == len(self.order)
                         else f"{n} of {len(self.order)} bricks are home")]
        name = self.order[self._k]
        if self._stage == "place":
            cmds = self._placing(world, name).commands(world, ctx)
            said = [c for c in cmds if isinstance(c, Done)]
            if not said:
                return cmds
            if "gave up" in said[0].note:
                return [Done(f"could not put the {name} brick on")]
            self._stage, self._presses = "press", 0
        if self._stage == "press":
            # pressed until the gap under it is closed (where it is over the
            # brick below was settled by putting it there)
            if abs(gap(world, self.order, name)) <= HOME_Z:
                return self._next()
            if self._presses >= MAX_PRESSES:
                return [Done(f"the {name} brick would not go down")]
            return self._press(world, name)
        return [Wait(0.2)]

    def _next(self) -> List[Command]:
        self._k += 1
        self._stage, self._place = "place", None
        return [Wait(0.2)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        name = self.order[0]
        return [c for c in self._placing(world, name).plan(world, ctx) if not isinstance(c, Done)] \
            + [Done("one brick put on; the rest needs looking")]

