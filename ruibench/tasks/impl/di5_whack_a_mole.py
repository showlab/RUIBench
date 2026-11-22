"""DI5 "whack-a-mole" -- Dynamic Interaction, two arms.

Knob: how the moles come up (one at a time, staying long -> one on each side at
once, staying short).

A whack-a-mole cabinet stretches between the arms, three holes at each end:
the left arm's three and the right arm's.  Each hand starts holding a toy
mallet level, its head waiting over the outer edge of its own end.  "Whack
the moles with the mallets!"  Three T_ref in (24 s at 8 s), a mole comes up
from one of the holes, stays a while and goes back down unless a mallet comes
down on it first -- then it is whacked ("Bonk!") and drops.  Where the next one comes
up does not exist until it does.  Easy: one at a time.  Hard: one at each end
at once, up too briefly for one hand to do both: the two arms have to go
together.  Success: all but one whacked.

v6 (2026-09-22): the hands start with the mallets.  Until then the mallets
lay on stands on the cabinet and the game began once both were taken up;
the advisor found the taking-up slow to watch and beside the point, and the
task is the whacking.

v3.1 (2026-09-20): the mallets moved from outboard of each arm onto the
cabinet's top.  Outboard they stood 48 cm either side of the middle and the
head camera, fitted to show them, had to take the whole 1.06 m in; on the
cabinet the picture is the cabinet (0.88 m) and a mole is a fifth larger.

v3 (2026-09-19): the hands hold toy mallets -- a scanned wooden toy mallet
with a ball head, held level, the head 9 cm ahead of the fingers -- where they used to
tap with closed fingers.  The mallet is held by a hidden square core under the
round handle; a hit is the head coming down on a mole (a hand does not count).
The game starts two seconds after both mallets are lifted (at the latest 20 s
in).  Which hole and when come from the seed.  The stays are placeholders
until Gate C measures how fast an agent can answer (the card's knob is not
calibrated).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from ...core.config import DEFAULT_T_REF_SECONDS, RunConfig
from ...core.geometry import rotate
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArmStart, BodySpec, Engine
from ...events.actions import DriveJoint
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import Custom, Finally, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import Frame, LEFT, RIGHT, ahead, station_scene

#: each arm's three holes, from its own base: (ahead, inward toward the
#: middle -- ``OUT`` turns that into +-y per arm).  All of them inward of
#: the arm: straight ahead (y = +-0.30) two of the three once stood outside
#: the head camera's picture (9/21).  35 to 44 cm ahead since the ARX X5 (v4,
#: 9/22): RoboDojo's head camera sees the bench only from 27 cm ahead, and a
#: hand 9 cm behind the mallet's head is then 26 to 35 cm ahead, where the X5
#: can hold it 15 cm up -- which is also why a mole rises 3.5 cm and a head
#: waits 1.5 cm over it
#: rows 8 cm apart, because the cabinet's top is built as strips between the
#: hole rows and rows closer than 2 x MOLE_HOLE leave no board between them
HOLES = ((0.30, 0.16), (0.39, 0.16), (0.35, 0.08))
SIDES = {LEFT: "left", RIGHT: "right"}
#: which way is outward (+y for the left arm)
OUT = {LEFT: 1.0, RIGHT: -1.0}
#: (moles or pairs, how long one stays up in T_ref, the pause before the next
#: in seconds).  v13, provisional (T_ref 8 s): 20 s up at the easy level -- a
#: model taking T_ref a turn sees a mole up to T_ref late and swings T_ref
#: after that; at 10 s the reference stepping at 8 s hit none -- and 3 s at
#: the hard level
SCHEDULE = {"easy": (6, 2.5, 2.0), "hard": (4, 0.375, 2.5)}
#: the first mole: this many T_ref into the episode, and never sooner than
#: START_AFTER seconds (v2.3.4).  An agent that reads the task, looks and gets
#: ready -- three replies -- is watching when it comes up.  At 2 s the first
#: moles came and went before any reply could act on them: the easy level's
#: first could not be had, nor the hard level, all but one of eight to whack
START_IN, START_AFTER = 3.0, 2.0
#: a mallet's head is on a mole: within this of its middle, its underside
#: within this of the mole's top
HIT_REACH, HIT_ABOVE = 0.03, 0.012
#: a whack comes down: the head over the hole above a risen mole's top first, then
#: down at it (v2.3: a head swept to and fro at the height of the moles' tops, never
#: coming down, whacked every mole that rose under it) -- or, wherever over the hole
#: it came in, down by DROP to the mole's top (v2.3.4: a head brought in sideways a
#: centimetre over a mole and struck straight down through it had not counted).  One
#: that comes down on an empty hole is a swing at nothing, and more than this many
#: fail the game
DROP = 0.01
EMPTY_SWINGS = 3
EMPTY = "empty_swing"
READY_ABOVE = 0.015             # where a head waits, above a mole's top
WHACKED = "whacked"
STARTED = "moles_started"

# -- the mallets ----------------------------------------------------------------------
#: body frame: the grip's middle at the origin, the head ahead along +x.  A
#: wooden mallet drawn round (v8, 9/24): a straight ash handle 1.36 cm thick,
#: its middle on the hidden square grip the fingers close on (1.2 cm), and a
#: red lacquered head 4.8 cm tall across its end.  Until v8 a scanned toy
#: mallet, whose handle was bent: the advisor thought the fingers had missed it
GRIP_HALF = (0.035, 0.006, 0.006)
GRIP_X = -0.005
HANDLE_END = -0.045
HANDLE_R = 0.0068
HEAD_X, HEAD_R = HANDLE_END + 0.131, 0.0237
HEAD_DY, HEAD_DZ = 0.0, 0.0
#: where a head waits, from its arm's base: ahead, and inward -- over the
#: cabinet's outer edge on the arm's own side, outboard of its three holes,
#: so it hides none of them from the head camera and every hole is a short
#: sideways move away
WAIT_AT = (0.35, 0.025)
#: each mallet points in toward the middle, this far off straight ahead (v7,
#: 9/23: the advisor asked for the arms "turned in"): its hand then stands
#: 7 cm behind and 4-6 cm outboard of its head, and the arm, straight ahead
#: of its own base, stays off the cabinet instead of standing over its end
MALLET_YAW = {LEFT: -math.radians(35), RIGHT: math.radians(35)}


#: what the head camera must show: the cabinet, a mole up, and the hands
#: with their mallets where they start
FRAME = Frame((0.20, 0.48), (-0.37, 0.37), 0.15)


def mallet_name(arm: str) -> str:
    return f"{SIDES[arm]}_mallet"


def waiting_pose(arm: str) -> Pose:
    """The hand's pose with its mallet's head waiting over ``WAIT_AT``,
    ``READY_ABOVE`` over a risen mole's top: the mallet level and pointing
    in toward the middle (``MALLET_YAW``), held straight down across its grip."""
    x, y = ahead(WAIT_AT[0], OUT[arm] * -WAIT_AT[1], arm=SIDES[arm])
    z = T.MOLE_TOP + T.MOLE_RISE + 0.005 + HEAD_R + READY_ABOVE
    yaw = MALLET_YAW[arm]
    dx, dy, dz = rotate(P.turn_z(yaw), (HEAD_X, HEAD_DY, HEAD_DZ))
    return Pose((x - dx, y - dy, z - dz), turned_down(yaw))


def mallet(arm: str) -> BodySpec:
    """A toy mallet in the hand, where the hand starts (``waiting_pose``)."""
    shaft = (HEAD_X - HANDLE_END) / 2
    parts = [P.grip((GRIP_X, 0, 0), GRIP_HALF),
             P.lathe((HANDLE_END + shaft, 0, 0), "capsule", HANDLE_R, shaft - HANDLE_R, "wood:ash",
                     P.turn_y(math.pi / 2)),
             P.lathe((HEAD_X, 0, 0), "chamfer", HEAD_R * 0.92, HEAD_R, "paint:red"),
             # a brass ferrule where the handle goes into the head
             P.lathe((HEAD_X - HEAD_R * 0.92 - 0.004, 0, 0), "cylinder", HANDLE_R + 0.0012, 0.004,
                     "metal:brass", P.turn_y(math.pi / 2))]
    return BodySpec(mallet_name(arm), "compound", GRIP_HALF,
                    Pose(waiting_pose(arm).pos, P.turn_z(MALLET_YAW[arm])),
                    category="toy mallet", color="wood", mass=0.04,
                    attrs={"parts": tuple(parts), "footprint": "rect", "grasp_height": 0.0,
                           "grasp_yaw": 0.0})


def head(world: WorldState, arm: str) -> Tuple[float, float, float]:
    o = world.obj(mallet_name(arm))
    dx, dy, dz = rotate(o.pose.quat, (HEAD_X, HEAD_DY, HEAD_DZ))
    x, y, z = o.pose.pos
    return (x + dx, y + dy, z + dz)


def hole_names():
    return [(arm, k, f"mole_{SIDES[arm]}{k}") for arm in (LEFT, RIGHT) for k in range(len(HOLES))]


def build_scene(level: str):
    bodies, joints, holes = [], [], {}
    for arm, k, name in hole_names():
        xy = ahead(HOLES[k][0], OUT[arm] * -HOLES[k][1], arm=SIDES[arm])
        body, slide = T.mole(name, xy, f"{name}_slide")
        bodies.append(body)
        joints.append(slide)
        holes[name] = xy
    xs = [xy[0] for xy in holes.values()]
    ys = [xy[1] for xy in holes.values()]
    centre = ((min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2)
    half = ((max(xs) - min(xs)) / 2 + 0.06, (max(ys) - min(ys)) / 2 + 0.06)
    bodies.append(T.mole_cabinet("cabinet", centre, half, list(holes.values())))
    bodies += [mallet(arm) for arm in (LEFT, RIGHT)]
    starts = [ArmStart(arm, waiting_pose(arm), holding=mallet_name(arm)) for arm in (LEFT, RIGHT)]
    return station_scene(f"di5_whack_a_mole_{level}", bodies, two_arms=True,
                         articulations=joints, frame=FRAME, arm_starts=starts), holes


def schedule(level: str, seed: int, t_ref: float = DEFAULT_T_REF_SECONDS
             ) -> List[Tuple[float, Tuple[str, ...]]]:
    """(when, from the game's start; which moles) -- one mole at a time (easy)
    or one each side (hard)."""
    count, stay, pause = SCHEDULE[level]
    stay *= t_ref
    rng = SeedBundle(seed).stream("moles")
    names = {arm: [n for a, _, n in hole_names() if a == arm] for arm in (LEFT, RIGHT)}
    out, t = [], 0.0
    for i in range(count):
        if level == "easy":
            arm = (LEFT, RIGHT)[rng.randrange(2)]
            pick = (names[arm][rng.randrange(len(HOLES))],)
        else:
            pick = tuple(names[arm][rng.randrange(len(HOLES))] for arm in (LEFT, RIGHT))
        out.append((t, pick))
        t += stay + pause
    return out


def game_start(t_ref: float = DEFAULT_T_REF_SECONDS) -> float:
    """When the first mole comes up (``START_IN``)."""
    return max(START_AFTER, START_IN * t_ref)


def mole_up(world: WorldState, name: str) -> bool:
    return world.articulations.get(f"{name}_slide", 0.0) > 0.6 * T.MOLE_RISE


def mole_top(world: WorldState, name: str) -> float:
    return T.MOLE_TOP + world.articulations.get(f"{name}_slide", 0.0) + 0.005


class Moles(Reactor):
    """Starts the game ``start`` into the episode (``game_start``), puts the
    moles up on schedule and down when their time is up -- or at once, whacked,
    when a held mallet's head comes down on one that is up."""

    def __init__(self, plan, stay: float, holes: Dict[str, Tuple[float, float]],
                 start: float = START_AFTER) -> None:
        super().__init__()
        self.plan, self.stay, self.holes, self.start = list(plan), stay, dict(holes), start
        self._t0: Optional[float] = None
        self._up: Dict[str, float] = {}           # mole -> when it came up
        self._next = 0
        self.whacked: List[str] = []
        self.missed: List[str] = []
        self._above: Dict[str, bool] = {}         # a head over the hole, high
        self._peak: Dict[str, float] = {}         # the head's highest since it came over
        self._spent: Dict[str, bool] = {}         # come down, and not up out of reach since
        self._swings = 0

    def _down(self, engine: Engine, world: WorldState, name: str, why: str) -> None:
        self._up.pop(name, None)
        self.act(engine, world, f"{name}_{why}", DriveJoint(f"{name}_slide", 0.0))

    def tick(self, engine: Engine, world: WorldState) -> None:
        t = world.sim_time
        if self._t0 is None and t >= self.start:
            self._t0 = self.start
            self.note(engine, world, STARTED)
        while (self._t0 is not None and self._next < len(self.plan)
               and t >= self._t0 + self.plan[self._next][0]):
            for name in self.plan[self._next][1]:
                self._up[name] = t
                self.act(engine, world, f"{name}_up", DriveJoint(f"{name}_slide", T.MOLE_RISE))
            self._next += 1
        heads = [head(world, a) for a in (LEFT, RIGHT) if world.obj(mallet_name(a)).is_held]
        risen = T.MOLE_TOP + T.MOLE_RISE + 0.005
        came_down = set()
        for name, (x, y) in self.holes.items():
            over = [hz - HEAD_R for hx, hy, hz in heads if math.hypot(hx - x, hy - y) < HIT_REACH]
            if not over:
                self._above[name], self._spent[name] = False, False
                self._peak.pop(name, None)
                continue
            under = min(over)
            low = under < risen + HIT_ABOVE
            if not low:
                self._spent[name] = False
            peak = max(self._peak.get(name, under), under)
            through = peak - under >= DROP and under <= risen
            if low and not self._spent.get(name) and (self._above.get(name) or through):
                came_down.add(name)
                self._spent[name] = True
            self._peak[name] = under if self._spent.get(name) else peak
            self._above[name] = not low
        for name, since in list(self._up.items()):
            if not mole_up(world, name) and t - since < 1.0:
                came_down.discard(name)
                continue                          # still coming up
            if name in came_down:
                self.whacked.append(name)
                self._down(engine, world, name, WHACKED)
                self.say(world, name, "whacked", "Bonk!")
            elif t - since >= self.stay:
                self.missed.append(name)
                self._down(engine, world, name, "missed")
        for name in sorted(came_down - set(self.whacked) - set(self._up)):
            self._swings += 1
            self.note(engine, world, f"{name}_{EMPTY}_{self._swings}")


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DI5")
    scene, holes = build_scene(level)
    t_ref = (config or RunConfig()).t_ref_seconds
    plan = schedule(level, seed, t_ref)
    count, stay, _ = SCHEDULE[level]
    stay *= t_ref
    total = sum(len(p) for _, p in plan)
    need = total - 1

    def whacked(w: WorldState) -> int:
        return sum(1 for e in w.events_fired if e.endswith(f"_{WHACKED}"))

    def sides_whacked(w: WorldState) -> bool:
        return all(any(e.startswith(f"mole_{s}") and e.endswith(f"_{WHACKED}") for e in w.events_fired)
                   for s in SIDES.values())

    milestones = (
        Milestone("first_whack", Custom(lambda w: whacked(w) >= 1, "a mole whacked")),
        Milestone("both_sides", Custom(sides_whacked, "a mole whacked on each side")),
        Milestone("half", Custom(lambda w: whacked(w) >= (need + 1) // 2,
                                 f"{(need + 1) // 2} moles whacked")),
        Milestone("enough", Custom(lambda w: whacked(w) >= need, f"{need} moles whacked")),
    )
    length = plan[-1][0] + stay + 0.5
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Whack the moles with the mallets!",
        scorer=Scorer(task_id="DI5", level=level,
                      success=Finally(Custom(lambda w: whacked(w) >= need, f"{need} of {total} whacked")),
                      milestones=milestones,
                      violations=(("no_blind_swinging", Never(Custom(
                          lambda w: sum(f"_{EMPTY}_" in e for e in w.events_fired) > EMPTY_SWINGS,
                          f"more than {EMPTY_SWINGS} swings at empty holes"))),)),
        milestones=milestones,
        reactor_factory=lambda p=plan, s=stay, h=holes, g=game_start(t_ref): Moles(p, s, h, g),
        reference_factory=lambda inst, h=holes, n=length, k=need: WhackPolicy(h, n, k),
        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level], "moles": total,
                  "stay_seconds": stay, "both_hands_needed": level == "hard",
                  # a look every 0.3 s through the whole game, and more
                  "max_model_calls": 60 + int(4 * (plan[-1][0] + stay))},
    )


class WhackPolicy(ScriptedPolicy):
    """The hands wait with their mallets over their own ends; a mole comes up
    on a side and that hand brings the head down on it and back to waiting.
    Both hands in the same turn when both ends have one.  ``serial`` (the
    stress agent): one hand at a time."""

    name = "whack"

    def __init__(self, holes: Dict[str, Tuple[float, float]], length: float,
                 enough: Optional[int] = None) -> None:
        self.holes, self.length = dict(holes), length
        #: whacks that win the game: then the hands go back to waiting and
        #: stay, whatever moles are still to come
        self.enough = enough
        self.serial = False
        self.reset()

    def reset(self) -> None:
        self._started: Optional[float] = None
        self._rested = False

    @staticmethod
    def _home(arm: str) -> Tuple[float, float]:
        return ahead(WAIT_AT[0], OUT[arm] * -WAIT_AT[1], arm=SIDES[arm])

    @staticmethod
    def _hand_for_head(world: WorldState, arm: str, target) -> Pose:
        hand = world.arm(arm).eef_pose.pos
        h = head(world, arm)
        return Pose(tuple(p + g - c for p, g, c in zip(hand, target, h)), turned_down(MALLET_YAW[arm]))

    def _head_over(self, world: WorldState, arm: str, xy, above: float) -> Pose:
        z = T.MOLE_TOP + T.MOLE_RISE + 0.005 + HEAD_R + above
        return self._hand_for_head(world, arm, (xy[0], xy[1], z))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        arms = (LEFT, RIGHT)
        if STARTED in world.events_fired and self._started is None:
            self._started = world.sim_time
        won = self.enough is not None and sum(
            1 for e in world.events_fired if e.endswith(f"_{WHACKED}")) >= self.enough
        if won or (self._started is not None and world.sim_time > self._started + self.length):
            if not self._rested:
                # back to waiting, mallets in hand, for the last picture
                self._rested = True
                return [MoveTo(a, self._head_over(world, a, self._home(a), READY_ABOVE))
                        for a in arms] + [Wait(1.0)]
            return [Done("the moles are done")]
        strikes = []
        for arm in arms:
            up = [n for n, xy in self.holes.items()
                  if n.startswith(f"mole_{SIDES[arm]}") and mole_up(world, n)]
            if not up:
                continue
            xy = self.holes[up[0]]
            # over the mole at the waiting height, not higher: near its 15 cm
            # ceiling the X5's wrist is slow, and a hard mole is up for 3 s
            strikes.append((arm, [MoveTo(arm, self._head_over(world, arm, xy, READY_ABOVE)),
                                  MoveTo(arm, self._head_over(world, arm, xy, -0.005)),
                                  MoveTo(arm, self._head_over(world, arm, self._home(arm), READY_ABOVE))]))
            if self.serial:
                break
        if not strikes:
            return [Wait(0.3)]
        return [c for k in range(3) for _, cmds in strikes for c in cmds[k:k + 1]]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: the mallets held ready -- no mole to see yet."""
        return [Wait(30.0), Done("ready, blind")]
