"""CL1 "play it back" -- In-Context Learning, two arms.

Knob: the xylophone and the tune (the film's xylophone, five notes, one hand
after the other -> its bars' colours in another order, and two chords).

A toy xylophone lies across the table between the two arms: eight
coloured bars, four on either side of a wooden middle block -- the left four
the left hand's, the right four the right hand's (each reaches only its own
half).  Each hand starts holding a little xylophone mallet, its head waiting
outboard of its hand's bars.  Before the episode a film shows the two hands
playing a tune on it.
"Play the tune from the video."  The tune is only ever in the film, never in
words.  Easy: the same xylophone, five notes, the hands taking turns.  Hard:
the bars' colours in another order -- no bar keeps its colour, so the tune has
to be carried over by colour, and a note the left hand played in the film may
now be the right hand's -- and two of its steps are chords, a bar on each side
struck together (within 0.4 s, the same allowance as DI1's two buttons).
Every step must be played, in order, and nothing else.

v6 (2026-09-22): the hands hold mallets from the first frame, as a child
plays a xylophone (the advisor's wish); a bar sounds when a mallet's head
comes down on it -- the head does not collide, so ``Keyboard`` sees it come
down and sinks the bar on its spring.  Each mallet points in toward the
middle, 20 degrees off straight ahead, which keeps the two hands 19 cm apart
over the two middle bars.

v3 (2026-09-19): it was one arm and five bars.  The bars are sprung slides a
closed hand pressed down (v2), the hands turned a quarter outward so two of
them over the two middle bars cleared each other and their wrist cameras.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.geometry import rotate
from ...core.types import Pose, WorldState
from ...engines import parts as P
from ...engines.base import ArmStart, BodySpec
from ...events.actions import DriveJoint
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import Custom, Finally, Milestone, Scorer
from ..instance import Demonstration, TaskInstance
from ..loader import get
from . import _toys as T
from ._common import Frame, LEFT, RIGHT, TABLE, ahead, station_scene

BARS = 8
KEYS = tuple(f"key{i}" for i in range(BARS))
SPRINGS = tuple(f"{k}_spring" for k in KEYS)
#: the film's bars, left to right: a rainbow
COLOURS = ("red", "orange", "yellow", "green", "teal", "blue", "purple", "pink")
#: the bars, from the xylophone's middle: the first four on the left (+y)
BAR_Y = tuple(s * (0.055 + k * 0.052) for s, ks in ((1, (3, 2, 1, 0)), (-1, (0, 1, 2, 3))) for k in ks)
#: half lengths, longest (lowest) on the left
BAR_HALF = tuple(0.055 - 0.003 * i for i in range(BARS))
XYLO_X = 0.30
MIDDLE_HALF_Y = 0.029
STEPS = 5
CHORDS = {"easy": 0, "hard": 2}
#: two bars struck this close together are a chord
CHORD_WINDOW = 0.4
#: a bar has been played once it is down this far, and is free again once it
#: is back above the second mark
DOWN, UP = -0.005, -0.002
#: the bars' tops, at rest
BAR_TOP = T.XYLO_KEY_Z + T.XYLO_KEY_HALF_HEIGHT

# -- the mallets ----------------------------------------------------------------------
#: a toy xylophone's mallet, built from parts: a turned ash handle 1 cm thick
#: and a red lacquered ball 2.6 cm across.  Body frame: the grip's middle at
#: the origin, the head ahead along +x.  The fingers close on a hidden square
#: core as thick as the handle (``P.grip``: flat pads turn a round handle)
GRIP_HALF = (0.03, 0.005, 0.005)
HANDLE_R, HANDLE_BACK = 0.005, 0.035
HEAD_X, HEAD_R = 0.11, 0.013
#: each mallet points in toward the middle, this far off straight ahead: its
#: hand stands 4 cm outboard of its head, and the two hands 19 cm apart when
#: both heads are on the middle bars (a chord)
MALLET_YAW = {LEFT: -math.radians(20), RIGHT: math.radians(20)}
#: where a head waits: across from the xylophone's middle by this much,
#: outboard of its hand's outermost bar, and this far over the bars' tops
WAIT_Y, WAIT_ABOVE = 0.255, 0.04
#: a head has struck a bar when its underside comes this close to the bar's
#: top inside the bar; it has left when it is back up this far.  A struck bar
#: sinks this far on its spring (the mark for "played" is DOWN)
HIT_ABOVE, LEFT_ABOVE, STRUCK_TO = 0.003, 0.012, -0.006
#: a stroke: the head from WAIT_ABOVE over the bar down to this far into it
#: (the head passes through nothing: it does not collide)
STRIKE_TO = -0.002

Step = FrozenSet[str]


#: what the head camera must show: the bars, and both hands with their
#: mallets where they start
FRAME = Frame((0.17, 0.39), (-0.31, 0.31), 0.10)


def mallet_name(arm: str) -> str:
    return f"{'left' if arm == LEFT else 'right'}_mallet"


def waiting_pose(arm: str) -> Pose:
    """The hand's pose with its mallet's head waiting outboard of its bars."""
    x0, _ = ahead(XYLO_X, 0.0)
    yaw = MALLET_YAW[arm]
    head = (x0, WAIT_Y if arm == LEFT else -WAIT_Y, BAR_TOP + WAIT_ABOVE + HEAD_R)
    back = rotate(P.turn_z(yaw), (HEAD_X, 0.0, 0.0))
    return Pose(tuple(h - b for h, b in zip(head, back)), turned_down(yaw))


def mallet(arm: str) -> BodySpec:
    """A xylophone mallet in the hand, where the hand starts."""
    along = P.turn_y(math.pi / 2)                   # a cylinder along the body's x
    front = HEAD_X - 0.6 * HEAD_R
    parts = (P.grip((0.0, 0.0, 0.0), GRIP_HALF),
             P.cylinder(((front - HANDLE_BACK) / 2, 0.0, 0.0), HANDLE_R, (front + HANDLE_BACK) / 2,
                        "wood:ash", rot=along, collides=False),
             P.sphere((-HANDLE_BACK, 0.0, 0.0), HANDLE_R * 1.15, "wood:ash", collides=False),
             P.sphere((HEAD_X, 0.0, 0.0), HEAD_R, "paint:red", collides=False))
    start = waiting_pose(arm)
    return BodySpec(mallet_name(arm), "compound", GRIP_HALF, Pose(start.pos, P.turn_z(MALLET_YAW[arm])),
                    category="xylophone mallet", color="wood:ash", mass=0.03,
                    attrs={"parts": parts, "footprint": "rect", "grasp_height": 0.0,
                           "grasp_yaw": 0.0})


def head(world: WorldState, arm: str) -> Tuple[float, float, float]:
    o = world.obj(mallet_name(arm))
    d = rotate(o.pose.quat, (HEAD_X, 0.0, 0.0))
    return tuple(p + v for p, v in zip(o.pose.pos, d))


def side(k: int) -> str:
    return LEFT if k < BARS // 2 else RIGHT


def colours_for(level: str, seed: int, tune: Sequence[Step]) -> Tuple[str, ...]:
    """The bars' colours, left to right: the film's at the easy level; at the
    hard one another order -- no bar keeps its colour, and the two bars of
    every chord still on different sides."""
    if level == "easy":
        return COLOURS
    rng = SeedBundle(seed).stream("bars")
    while True:
        order = list(COLOURS)
        rng.shuffle(order)
        if any(a == b for a, b in zip(order, COLOURS)):
            continue
        where = {c: side(i) for i, c in enumerate(order)}
        if all(len({where[c] for c in step}) == len(step) for step in tune):
            return tuple(order)


def tune_for(level: str, seed: int) -> List[Step]:
    """Five steps on the film's xylophone: single notes with the hands taking
    turns; at the hard level two of the steps chords, a bar from each side.
    No bar twice in a row."""
    rng = SeedBundle(seed).stream("tune")
    left = [c for i, c in enumerate(COLOURS) if side(i) == LEFT]
    right = [c for i, c in enumerate(COLOURS) if side(i) == RIGHT]
    chords = set(rng.sample(range(STEPS), CHORDS[level])) if CHORDS[level] else set()
    hand = rng.choice((LEFT, RIGHT))
    out: List[Step] = []
    for i in range(STEPS):
        before = out[-1] if out else frozenset()
        if i in chords:
            step = frozenset({rng.choice([c for c in left if c not in before]),
                              rng.choice([c for c in right if c not in before])})
        else:
            pool = left if hand == LEFT else right
            step = frozenset({rng.choice([c for c in pool if c not in before])})
            hand = RIGHT if hand == LEFT else LEFT
        out.append(step)
    return out


def xylophone_frame(name: str, xy) -> BodySpec:
    """Two wooden rails under the bars' ends, a base, and a plain wooden block
    between the halves.  (Until v7 the block was painted red with a white
    note on it: in the head camera it read as a ninth, red bar with something
    lying on it, beside the real red one -- the advisor asked what it was.)"""
    half_y = max(abs(y) for y in BAR_Y) + T.XYLO_KEY_WIDTH + 0.012
    hz = (T.XYLO_FRAME_TOP - TABLE) / 2
    wood = "wood:ash"
    parts = [P.box((0.03, 0, 0), (0.008, half_y, hz), wood),
             P.box((-0.03, 0, 0), (0.008, half_y, hz), wood),
             P.box((0, 0, -hz + 0.003), (0.05, half_y, 0.003), wood),
             P.box((0, 0, hz + 0.003), (0.04, MIDDLE_HALF_Y - 0.006, 0.003), wood)]
    return BodySpec(name, "compound", (0.05, half_y, hz + 0.008), Pose((xy[0], xy[1], TABLE + hz)),
                    category="xylophone", color=wood, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def scene_for(colours: Sequence[str], label: str):
    x0, _ = ahead(XYLO_X, 0.0)
    bodies, joints = [], []
    for i, (name, spring, colour) in enumerate(zip(KEYS, SPRINGS, colours)):
        key, joint = T.xylophone_key(name, (x0, BAR_Y[i]), spring, f"paint:{colour}", BAR_HALF[i])
        key = type(key)(**{**key.__dict__, "attrs": {**key.attrs, "note": colour, "hand": side(i)}})
        bodies.append(key)
        joints.append(joint)
    bodies.append(xylophone_frame("xylophone", (x0, 0.0)))
    bodies += [mallet(arm) for arm in (LEFT, RIGHT)]
    starts = [ArmStart(arm, waiting_pose(arm), holding=mallet_name(arm)) for arm in (LEFT, RIGHT)]
    return station_scene("cl1_xylophone", bodies, two_arms=True, articulations=joints, frame=FRAME,
                         arm_starts=starts)


# -- hearing it -----------------------------------------------------------------------
def group(presses: Sequence[Tuple[float, str, str]]) -> List[Tuple[float, Step]]:
    """(time, colour, hand) presses -> steps: a press within ``CHORD_WINDOW``
    of a single note from the other hand joins it as a chord."""
    steps: List[Tuple[float, Step, str]] = []
    for t, colour, hand in presses:
        if steps and len(steps[-1][1]) == 1 and steps[-1][2] != hand and t - steps[-1][0] <= CHORD_WINDOW:
            t0, notes, _ = steps[-1]
            steps[-1] = (t0, notes | {colour}, "both")
        else:
            steps.append((t, frozenset({colour}), hand))
    return [(t, notes) for t, notes, _ in steps]


STEP = "step"


def played(w: WorldState) -> List[Step]:
    """The steps played so far (``Keyboard`` records them; a chord's second
    note rewrites its step)."""
    out: Dict[int, Step] = {}
    for e in w.events_fired:
        if e.startswith(STEP) and ":" in e:
            k, notes = e[len(STEP):].split(":", 1)
            out[int(k)] = frozenset(notes.split("+"))
    return [out[k] for k in sorted(out)]


def all_up(w: WorldState) -> bool:
    return all(w.articulations.get(s, 0.0) > UP for s in SPRINGS)


def on_bar(world: WorldState, key: str, point, above: float) -> bool:
    """Is a mallet's head at ``point`` down on ``key``: inside the bar, its
    underside within ``above`` of the bar's top at rest?"""
    o = world.obj(key)
    x, y, _ = o.pose.pos
    return (abs(point[0] - x) <= o.size[0] and abs(point[1] - y) <= o.size[1]
            and point[2] - HEAD_R <= BAR_TOP + above)


class Keyboard(Reactor):
    """The xylophone: a held mallet's head coming down on a bar sinks it on
    its spring until the head is lifted again, and a bar that goes down is a
    note, recorded with its colour -- or, struck with a bar of the other
    hand's, a chord."""

    def __init__(self, colour_of: Dict[str, str], hand_of: Dict[str, str]) -> None:
        super().__init__()
        self.colour_of, self.hand_of = dict(colour_of), dict(hand_of)
        self._down = {k: False for k in KEYS}
        self._struck = {k: False for k in KEYS}
        self._presses: List[Tuple[float, str, str]] = []

    def _strike(self, engine, world: WorldState) -> None:
        heads = [head(world, a) for a in (LEFT, RIGHT) if world.obj(mallet_name(a)).is_held]
        for key, spring in zip(KEYS, SPRINGS):
            if not self._struck[key] and any(on_bar(world, key, h, HIT_ABOVE) for h in heads):
                self._struck[key] = True
                self.act(engine, world, f"{key}_struck", DriveJoint(spring, STRUCK_TO))
            elif self._struck[key] and not any(on_bar(world, key, h, LEFT_ABOVE) for h in heads):
                self._struck[key] = False
                self.act(engine, world, f"{key}_let_up", DriveJoint(spring, 0.0))

    def tick(self, engine, world: WorldState) -> None:
        self._strike(engine, world)
        for key, spring in zip(KEYS, SPRINGS):
            value = world.articulations.get(spring, 0.0)
            if not self._down[key] and value <= DOWN:
                self._down[key] = True
                self._presses.append((world.sim_time, self.colour_of[key], self.hand_of[key]))
                steps = group(self._presses)
                k = len(steps) - 1
                self.note(engine, world, f"{STEP}{k}:{'+'.join(sorted(steps[-1][1]))}", key)
            elif self._down[key] and value >= UP:
                self._down[key] = False


def heard(states) -> List[Step]:
    """The steps a film played, from its states: its bars going down."""
    if not states:
        return []
    first = states[0]
    colour = {k: first.obj(k).attrs.get("note") for k in KEYS}
    hand = {k: first.obj(k).attrs.get("hand") for k in KEYS}
    down = {k: False for k in KEYS}
    presses = []
    for w in states:
        for key, spring in zip(KEYS, SPRINGS):
            value = w.articulations.get(spring, 0.0)
            if not down[key] and value <= DOWN:
                down[key] = True
                presses.append((w.sim_time, colour[key], hand[key]))
            elif down[key] and value >= UP:
                down[key] = False
    return [notes for _, notes in group(presses)]


# -- playing it -----------------------------------------------------------------------
def hand_for_head(world: WorldState, arm: str, target) -> Pose:
    """The hand's pose that puts its mallet's head at ``target``, the hand
    turned as it holds the mallet now (so a mallet a little off in the
    fingers still lands its head where it should)."""
    hand = world.arm(arm).eef_pose
    h = head(world, arm)
    return Pose(tuple(p + g - c for p, g, c in zip(hand.pos, target, h)), turned_down(MALLET_YAW[arm]))


def press_poses(world: WorldState, key: str) -> Tuple[str, Pose, Pose]:
    """(the bar's hand, its head over the bar's middle, the head struck down
    on it)."""
    obj = world.obj(key)
    arm = obj.attrs.get("hand")
    x, y, _ = obj.pose.pos
    return (arm, hand_for_head(world, arm, (x, y, BAR_TOP + WAIT_ABOVE + HEAD_R)),
            hand_for_head(world, arm, (x, y, BAR_TOP + STRIKE_TO + HEAD_R)))


class NotePlayer(ScriptedPolicy):
    """Plays the steps one at a time with the mallets, a turn for each phase:
    heads over the bars, down together, hold, up, rest.  The film shows a
    frame a second, so a step held ``hold`` seconds and followed by ``rest``
    is in the film."""

    name = "note_player"
    PHASES = 5

    def __init__(self, hold: float = 0.5, rest: float = 0.3) -> None:
        self.hold, self.rest = hold, rest
        #: the stress agent: one hand at a time -- a chord's bars one after
        #: the other
        self.serial = False
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._rested = False

    def keys(self, world: WorldState, ctx: TurnContext) -> List[List[str]]:
        raise NotImplementedError

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        steps = self.keys(world, ctx)
        if self.serial:
            steps = [[key] for step in steps for key in step]
        k, phase = divmod(self._step, self.PHASES)
        if k >= len(steps):
            if not self._rested:
                # the heads back to waiting, outboard of the bars
                self._rested = True
                return [MoveTo(a, waiting_pose(a)) for a in (LEFT, RIGHT)] + [Wait(0.5)]
            return [Wait(1.0), Done("played")]
        self._step += 1
        poses = [press_poses(world, key) for key in steps[k]]
        if phase == 0:
            return [MoveTo(arm, up) for arm, up, _ in poses]
        if phase == 1:                  # both hands down at once: a chord's notes together
            return [MoveTo(arm, down) for arm, _, down in poses]
        if phase == 2:
            return [Wait(self.hold)]
        if phase == 3:
            return [MoveTo(arm, up) for arm, up, _ in poses]
        return [Wait(self.rest)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        # a queued move takes at least one executor step (0.05 s): asking for
        # the same low pose again holds the bar down
        out: List[Command] = []
        for step in self.keys(world, ctx):
            for key in step:
                arm, up, down = press_poses(world, key)
                out += [MoveTo(arm, up)] + [MoveTo(arm, down)] * 8 + [MoveTo(arm, up)]
        return out + [Wait(1.0), Done("planned")]


class ShowTune(NotePlayer):
    """The film: slow, each step held long enough to be seen."""

    name = "show_tune"

    def __init__(self, keys: Sequence[Sequence[str]]) -> None:
        super().__init__(hold=1.5, rest=1.2)
        self._keys = [list(k) for k in keys]

    def keys(self, world: WorldState, ctx: TurnContext) -> List[List[str]]:
        return self._keys


class TunePolicy(NotePlayer):
    """Watch the film: which bars went down, in what order and together, and
    what colour each was.  Play the same colours on the episode's xylophone,
    each with the hand on its side.  Without the film (``ignore_demonstration``)
    it plays the first five bars, left to right."""

    name = "play_it_back"

    def reset(self) -> None:
        super().reset()
        self._keys: Optional[List[List[str]]] = None

    def keys(self, world: WorldState, ctx: TurnContext) -> List[List[str]]:
        if self._keys is None:
            steps: List[Step] = []
            for film in ctx.extras.get("demonstrations", ()):
                steps = heard(film.states)
                if steps:
                    break
            by_colour = {world.obj(k).attrs.get("note"): k for k in KEYS}
            if steps:
                self._keys = [sorted(by_colour[c] for c in step) for step in steps]
            else:
                self._keys = [[k] for k in KEYS[:STEPS]]
        return self._keys


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CL1")
    tune = tune_for(level, seed)
    film_scene = scene_for(COLOURS, "film")
    film_keys = [sorted(KEYS[COLOURS.index(c)] for c in step) for step in tune]
    hand_of = {k: side(i) for i, k in enumerate(KEYS)}
    films = (Demonstration(scene=film_scene, demonstrator=lambda k=tuple(map(tuple, film_keys)): ShowTune(k),
                           caption="The robot plays a tune.",
                           reactor_factory=lambda h=hand_of: Keyboard(dict(zip(KEYS, COLOURS)), h)),)
    colours = colours_for(level, seed, tune)
    scene = scene_for(colours, "episode")
    colour_of = dict(zip(KEYS, colours))

    def right(n: int) -> Custom:
        return Custom(lambda w, n=n: played(w)[:n] == tune[:n] and len(played(w)) >= n,
                      f"the first {n} steps right")

    written = " ".join("+".join(sorted(s)) for s in tune)
    whole = Custom(lambda w: played(w) == tune and all_up(w),
                   f"the tune ({written}) played, and nothing else")
    first_chord = next((i for i, s in enumerate(tune) if len(s) > 1), None)
    milestones = (
        Milestone("first_note", right(1)),
        Milestone("half_the_tune", right(3)),
    ) + ((Milestone("a_chord_right", right(first_chord + 1)),) if first_chord is not None else ()) + (
        Milestone("whole_tune", whole),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Play the tune from the video.",
        scorer=Scorer(task_id="CL1", level=level, success=Finally(whole), milestones=milestones),
        milestones=milestones,
        demonstrations=films,
        reactor_factory=lambda c=colour_of, h=hand_of: Keyboard(c, h),
        reference_factory=lambda inst: TunePolicy(),
        metadata={"knob": spec.knob.field,
                  "knob_value": "same xylophone" if level == "easy" else "bars in another order, chords",
                  # hidden from the agent: the tune, for the gap-filled control
                  "tune": [sorted(s) for s in tune], "film_bars": list(COLOURS),
                  "episode_bars": list(colours), "chord_window_s": CHORD_WINDOW,
                  "max_model_calls": 40},
        extra_metrics={"steps_played": lambda traj: len(played(traj[-1])) if traj else 0},
    )
