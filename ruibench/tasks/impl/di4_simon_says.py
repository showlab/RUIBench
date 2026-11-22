"""DI4 "Simon says" -- Dynamic Interaction.

Knob: how the calls come (four, one without "Simon says", slowly -> six, two
without, quickly, and each of the two right after a call on the same toy).

A wooden toy airplane, a tin robot and a wooden fire truck (and, at the hard
level, a spinning top) on the bench.  "Let's play Simon says."  Then the
calls come, one at a time, on their own clock: "Simon says: touch the
robot." -- "Touch the airplane." -- ...  A call
with "Simon says" in front is to be done while it stands; a call without is
to be left alone.  Neither the calls nor which of them carry the words exist
before they are said.

Strommen (1973) used this game to study children's self-regulation; the
error children make is the impulsive one, doing the call they should have
left, and it is most likely right after a call they were right to do.  The
hard level puts each bare call there, on the same toy.

A touch is the hand over a toy and its fingertips down to within 1.6 cm of
the toy's top, the moment it arrives (being there already does not count
again): the reactor keeps the clock, says the calls, and records which call
each touch fell in.

v4 (2026-09-21), replacing "change of mind": that was an instruction changed
mid-reach, and nothing outside RUIBench stood behind it.  v9 (9/24), the
advisor: the duck, the apple and the dinosaur were too plain.  The toys are
now ones a child names at a glance: a sitting plush teddy (scanned), a tin
wind-up robot with its key, a wooden fire engine (scanned), and a painted
spinning top turned from parts.  v10: a golden mohair teddy (the first one's
scan looked blotchy); v11: the teddy, plush and photographed, stood apart from
the tin and the painted wood -- a wooden airplane with its pilot, the fire
truck's kin, in its place.  Bigger and less regular than the three
were, they stand 13 cm apart, and a touch is anywhere over one, not within
3.5 cm of its middle: the truck is 16 cm long.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import DEFAULT_T_REF_SECONDS, RunConfig
from ...core.geometry import yaw_of
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec, Engine
from ...events.actions import SetInstruction
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import Custom, Ever, Finally, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

#: the hard level's spinning top, turned from parts (``engines/parts``' top
#: profiles): 7.2 cm from its steel tip to the end of its peg, 6 cm across,
#: painted red, yellow and blue, standing on its tip (it is a prop)
TOP_R, TOP_HZ = 0.030, 0.036
TOP_PAINT = {"top_tip": "metal:steel", "top_body": "paint:red", "top_band": "paint:yellow",
             "top_shoulder": "paint:blue", "top_peg": "wood:ash"}
#: name -> (catalogue entry (None: built here), what the calls call it, half
#: extents for the mock, its turn on the bench -- each facing the fixed
#: camera -- and its highest point, from its middle in its own frame, where
#: the reference touches it (``research/v9_di4_toy_tops.py``)
TOYS = {"plane": ("toy_airplane", "the airplane", (0.06, 0.0597, 0.0423), 0.0, (-0.006, -0.005)),
        "robot": ("tin_robot", "the robot", (0.0355, 0.021, 0.065), math.pi / 2, (0.003, 0.001)),
        "truck": ("fire_engine", "the fire truck", (0.0195, 0.08, 0.0564), math.radians(70),
                  (0.001, 0.053)),
        "top": (None, "the spinning top", (TOP_R, TOP_R, TOP_HZ), 0.0, (0.0, 0.0))}
LEVEL_TOYS = {"easy": ("plane", "robot", "truck"), "hard": ("plane", "robot", "truck", "top")}
#: per level: which calls carry "Simon says" (in order), and the time between
#: calls in T_ref.  v13, provisional (T_ref 8 s): 20 s at the easy level -- a
#: model taking T_ref a turn hears a call up to T_ref late and acts T_ref
#: after that; at 14 s the reference stepping at 8 s missed every game -- and
#: 8 s at the hard level
CALLS = {"easy": ((True, True, False, True), 2.5),
         "hard": ((True, False, True, True, False, True), 1.0)}
#: the first call, after this many T_ref (but never sooner than 4 s): time to
#: look at the toys first.  At 4 s every model in Gate C (GPT-5.6 Sol, Claude
#: Opus 5, Gemini 3.8 Flash, all four interfaces: 0/12) was still finding the
#: toys when the first call's window closed -- then did the later calls right,
#: within 7-13 s.  What this task tests is keeping up with the calls, not how
#: fast one looks round (10/1)
LEAD_IN = 2.0
FIRST_CALL = 4.0
OPENING = "Let's play Simon says: do what Simon says -- and only that."
SIMON = "Simon says: "

#: the toys in a row across the bench, 13 cm apart
ROW_X, PITCH = 0.30, 0.13
#: a touch: the hand's centre over the toy -- within its footprint, turned as
#: it stands, and this round it -- and within this of its top.  The FR3's
#: fingertips are 8 mm below its tool centre point
FINGERTIP = 0.008
TOUCH_MARGIN, TOUCH_ABOVE = 0.01, FINGERTIP + 0.016
#: where the reference brings its fingertips: just short of the toy's top
TAP = FINGERTIP + 0.004
HOVER = 0.10                            # where the hand waits, above a toy's top
GAME_OVER = "game_over"

FRAME = Frame((0.08, 0.40), (-0.27, 0.27), 0.20)


def schedule(level: str, seed: int, t_ref: float = DEFAULT_T_REF_SECONDS) -> List[Tuple[float, bool, str]]:
    """(when, with "Simon says", which toy) for every call.  A bare call at the
    hard level names the toy of the call before it -- the one the hand has
    just been sent to."""
    simons, gap = CALLS[level]
    gap *= t_ref
    first = max(FIRST_CALL, LEAD_IN * t_ref)
    toys = list(LEVEL_TOYS[level])
    rng = SeedBundle(seed).stream("simon")
    out: List[Tuple[float, bool, str]] = []
    for k, simon in enumerate(simons):
        if not simon and level == "hard":
            toy = out[-1][2]
        else:
            toy = rng.choice([t for t in toys if not out or t != out[-1][2]])
        out.append((first + k * gap, simon, toy))
    return out


def call_text(simon: bool, toy: str) -> str:
    body = f"touch {TOYS[toy][1]}."
    return SIMON + body if simon else body[0].upper() + body[1:]


def top_body(name: str, xy) -> BodySpec:
    """The spinning top: its painted bands drawn round (each a slice of one
    outline), what a hand meets a hidden body and peg."""
    peg_r, shoulder = TOP_R * P.TOP_PEG, -TOP_HZ + 2 * TOP_HZ * P.TOP_BANDS["top_peg"][0]
    body_hz, peg_hz = (shoulder + TOP_HZ) / 2, (TOP_HZ - shoulder) / 2
    parts = [P.cylinder((0.0, 0.0, -TOP_HZ + body_hz), 0.8 * TOP_R, body_hz, "grey", collides="hidden"),
             P.cylinder((0.0, 0.0, shoulder + peg_hz), peg_r, peg_hz, "grey", collides="hidden")]
    parts += [P.lathe((0.0, 0.0, 0.0), band, TOP_R, TOP_HZ, paint) for band, paint in TOP_PAINT.items()]
    return BodySpec(name, "compound", (TOP_R, TOP_R, TOP_HZ), Pose((xy[0], xy[1], TABLE + TOP_HZ)),
                    category="toy", color="paint:red", mass=0.05, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "round"})


def toy_body(toy: str, xy) -> BodySpec:
    asset, _, half, yaw, _ = TOYS[toy]
    if asset is None:
        return top_body(toy, xy)
    return asset_body(toy, asset, xy, fallback_size=half, mass=0.05, category="toy", fixed=True,
                      graspable=False, yaw=yaw)


def build_scene(level: str, seed: int):
    toys = list(LEVEL_TOYS[level])
    n = len(toys)
    places = [ahead(ROW_X, ((n - 1) / 2 - i) * PITCH) for i in range(n)]
    SeedBundle(seed).stream("places").shuffle(places)
    bodies = [toy_body(t, xy) for t, xy in zip(toys, places)]
    return station_scene(f"di4_simon_says_{level}", bodies, frame=FRAME)


def _in_toy(w: WorldState, toy: str, point) -> Tuple[float, float]:
    """A point's (along, across) from a toy's middle, in the toy's own frame."""
    o = w.obj(toy)
    yaw = yaw_of(o.pose.quat)
    dx, dy = point[0] - o.pose.pos[0], point[1] - o.pose.pos[1]
    return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)


def touching(w: WorldState, toy: str, arm: str = RIGHT) -> bool:
    o = w.obj(toy)
    tcp = w.arm(arm).eef_pose.pos
    along, across = _in_toy(w, toy, tcp)
    return (abs(along) <= o.size[0] + TOUCH_MARGIN and abs(across) <= o.size[1] + TOUCH_MARGIN
            and tcp[2] <= o.pose.pos[2] + o.size[2] + TOUCH_ABOVE)


def tap_point(w: WorldState, toy: str) -> Tuple[float, float, float]:
    """Where the reference touches a toy: over its highest point, just short
    of it -- a scanned toy's top is seldom over its middle (the fire truck's
    is its raised bucket, 5 cm from it)."""
    o = w.obj(toy)
    ax, ay = TOYS[toy][4]
    yaw = yaw_of(o.pose.quat)
    return (o.pose.pos[0] + math.cos(yaw) * ax - math.sin(yaw) * ay,
            o.pose.pos[1] + math.sin(yaw) * ax + math.cos(yaw) * ay,
            o.pose.pos[2] + o.size[2])


class Simon(Reactor):
    """Says the calls on the schedule and records every touch against the call
    it fell in: ``touched_k_<toy>``."""

    def __init__(self, calls: Sequence[Tuple[float, bool, str]], toys: Sequence[str],
                 gap: float) -> None:
        super().__init__()
        self.calls, self.toys, self.gap = list(calls), list(toys), gap
        self._said = -1
        self._in: Dict[str, bool] = {}
        self._over = False

    def current(self, t: float) -> int:
        k = -1
        for i, (when, _, _) in enumerate(self.calls):
            if t >= when:
                k = i
        return k

    def tick(self, engine: Engine, world: WorldState) -> None:
        t = world.sim_time
        k = self.current(t)
        if k > self._said:
            self._said = k
            _, simon, toy = self.calls[k]
            self.act(engine, world, f"call_{k}", SetInstruction(call_text(simon, toy)))
        for toy in self.toys:
            now = touching(world, toy)
            # once the game is over a touch is no answer to its last call (v2.3)
            if now and not self._in.get(toy) and k >= 0 and not self._over:
                self.note(engine, world, f"touched_{k}_{toy}")
            self._in[toy] = now
        if not self._over and t >= self.calls[-1][0] + self.gap:
            self._over = True
            self.note(engine, world, GAME_OVER)
            self.say(world, "simon", "done", "That's the end of the game.")


def obeyed(w: WorldState, calls) -> bool:
    """Every "Simon says" call done while it stood."""
    return all(f"touched_{k}_{toy}" in w.events_fired
               for k, (_, simon, toy) in enumerate(calls) if simon)


def hedged(w: WorldState, calls, toys) -> bool:
    """Two or more toys touched that a call did not name, while it stood: every
    toy tapped in case it was the one -- doing more than Simon says (v2.3).  One
    is let pass: the toys stand close, and a hand on its way can brush one."""
    return any(sum(f"touched_{k}_{t}" in w.events_fired for t in toys if t != toy) >= 2
               for k, (_, _, toy) in enumerate(calls))


def caught_out(w: WorldState, calls) -> bool:
    """A bare call done: the toy it named touched while it stood."""
    return any(f"touched_{k}_{toy}" in w.events_fired
               for k, (_, simon, toy) in enumerate(calls) if not simon)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DI4")
    t_ref = (config or RunConfig()).t_ref_seconds
    calls = schedule(level, seed, t_ref)
    gap = CALLS[level][1] * t_ref
    scene = build_scene(level, seed)
    over = Custom(lambda w: GAME_OVER in w.events_fired, "the game over")
    won = Custom(lambda w, c=calls: GAME_OVER in w.events_fired and obeyed(w, c)
                 and not caught_out(w, c), "every Simon call done, no other")
    milestones = (
        Milestone("first_call", Custom(lambda w: "call_0" in w.events_fired, "the first call")),
        Milestone("a_simon_call_done",
                  Custom(lambda w, c=calls: any(f"touched_{k}_{t}" in w.events_fired
                                                for k, (_, s, t) in enumerate(c) if s),
                         "a Simon call done")),
        Milestone("every_simon_call_done", Custom(lambda w, c=calls: obeyed(w, c),
                                                  "every Simon call done")),
        Milestone("game_over", over),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=OPENING,
        scorer=Scorer(task_id="DI4", level=level,
                      # every Simon call done while the game was on
                      success=Finally(won) & Ever(Custom(
                          lambda w, c=calls: obeyed(w, c) and GAME_OVER not in w.events_fired,
                          "every Simon call done before the end")),
                      milestones=milestones,
                      violations=(("only_when_simon_says",
                                   Never(Custom(lambda w, c=calls: caught_out(w, c),
                                                "a call without Simon done"))),
                                  ("only_what_simon_says",
                                   Never(Custom(lambda w, c=calls, ts=LEVEL_TOYS[level]:
                                                hedged(w, c, ts), "other toys touched too"))))),
        milestones=milestones,
        reactor_factory=lambda c=calls, t=LEVEL_TOYS[level], g=gap: Simon(c, t, g),
        reference_factory=lambda inst: SimonPolicy(LEVEL_TOYS[inst.level]),
        metadata={"knob": spec.knob.field, "knob_value": len(calls),
                  "seconds_between_calls": gap,
                  # hidden from the agent: the calls to come (the gap-filled control)
                  "calls": [call_text(s, t) for _, s, t in calls],
                  # one look every half second for the whole game, and some over
                  "max_model_calls": int(2 * (calls[-1][0] + gap)) + 20},
    )


class SimonPolicy(ScriptedPolicy):
    """Does each call that starts with "Simon says" as soon as it is heard --
    down to the toy, a touch, and back up -- and nothing for the others.
    ``ignore_rules`` (the blind control): does every call."""

    name = "simon"

    def __init__(self, toys: Sequence[str], arm: str = RIGHT) -> None:
        self.toys, self.arm = list(toys), arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._done_version = 0
        self._lift: Optional[Pose] = None

    def _toy(self, text: str) -> Optional[str]:
        for toy in self.toys:
            if TOYS[toy][1] in text:
                return toy
        return None

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._lift is not None:
            # back up in a turn of its own: the touch is read off the world as it
            # is sampled, and a hand that goes down and up in one queue can be
            # through the bottom between two samples
            up, self._lift = self._lift, None
            return [MoveTo(self.arm, up)]
        if GAME_OVER in world.events_fired:
            return [Done("game over")]
        v, text = world.instruction_version, world.instruction or ""
        toy = self._toy(text)
        if v == self._done_version or toy is None:
            return [Wait(0.5)]
        self._done_version = v
        if not (text.startswith(SIMON) or self.ignore_rules):
            return [Wait(0.5)]
        x, y, top = tap_point(world, toy)
        self._lift = Pose((x, y, top + HOVER), GRASP_DOWN)
        return [SetGripper(self.arm, 1.0),
                MoveTo(self.arm, Pose((x, y, top + HOVER), GRASP_DOWN)),
                MoveTo(self.arm, Pose((x, y, top + TAP), GRASP_DOWN))]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return [Done("heard no calls yet")]
