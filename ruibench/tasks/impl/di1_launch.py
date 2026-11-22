"""DI1 "two-button launch" -- Discovery.

Knob: whether a button stays down once pressed (it does -> it springs back,
and both must be held down together for half a second).

A toy rocket on its pad with a lamp on either side, and a console with a
big red button at each end, 40 cm apart -- one in reach of each arm; the
left lamp is the left button's, the right lamp the right's.  "Launch the
rocket."  How is not said: a pressed button lights its lamp, and the rocket
goes when both lamps are lit -- found by trying (v3: the instruction used to
say "press both buttons together"; that is now the gap-filled control).  At
the easy level a pressed button locks down and its lamp stays lit, so one
after the other will do; at the hard level they spring back up, a lamp is
lit only while its button is held, and only both held down at once, for
0.5 s, lights the rocket.

v7 (2026-09-23): "why four buttons?  A person would not know to press two
together either."  The horn and the lights buttons are gone, and the lamps
show what a press does: one lamp of the two lit is half of what the rocket
needs.  Nothing but the lamps says so -- the pressing is still the finding.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from ...core.config import RunConfig
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines.base import Engine
from ...events.actions import DriveJoint, SetColor
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import Custom, Finally, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import Frame, LEFT, RIGHT, TABLE, ahead, station_scene, to_rest

BUTTONS = {LEFT: "left_button", RIGHT: "right_button"}
SPRINGS = {LEFT: "left_spring", RIGHT: "right_spring"}
#: the lamps on the launch pad, each this far from the rocket on its button's
#: side; lit, green
LAMPS = {LEFT: "left_lamp", RIGHT: "right_lamp"}
LAMP_APART = 0.06
LAMP_LIT = (0.25, 1.0, 0.3)
ROCKET, LIFT = "rocket", "rocket_lift"
FLAME = f"{ROCKET}_flame"
#: the red buttons: this far ahead of each arm, and this far in toward the
#: middle.  Straight ahead of the Pipers (y = +-0.30) they stood outside the
#: head camera's picture, which saw +-0.29 at that range, so an arm pressed
#: its button off the side of the observation (9/21)
BUTTON_AT, BUTTON_IN = 0.28, 0.10
PAD_AT, PAD_TOP = (0.34, 0.30), TABLE + 0.012
#: down far enough to count: half the travel
PRESSED = -0.5 * T.BUTTON_TRAVEL
#: where the hand pushes a cap to: short of the stop, so the move can finish
PUSH_TO = -0.9 * T.BUTTON_TRAVEL
HOLD_SECONDS = 0.5
LATCHES = {"easy": True, "hard": False}
#: the rocket's climb: faster and faster, as a rocket's is, until it is gone
CLIMB_ACCEL = 0.5               # m/s^2
LAUNCHED = "rocket_launched"
FINGERTIP = 0.008
ABOVE = 0.06


#: what the head camera must show: the console, the rocket at the top of its
#: climb, and both hands where they start
FRAME = Frame((0.21, 0.41), (-0.27, 0.27), 0.40)


def pressed(w: WorldState, arm: str) -> bool:
    return w.articulations.get(SPRINGS[arm], 0.0) <= PRESSED


def build_scene(level: str):
    bodies, joints, spots = [], [], []
    for arm, name in BUTTONS.items():
        side = "left" if arm == LEFT else "right"
        xy = ahead(BUTTON_AT, -BUTTON_IN if arm == LEFT else BUTTON_IN, arm=side)
        cap, spring = T.push_button(name, xy, SPRINGS[arm])
        bodies.append(cap)
        joints.append(spring)
        spots.append(xy)
    (lx, ly), (rx, ry) = spots
    bodies.append(T.launch_console("console", ((lx + rx) / 2, (ly + ry) / 2),
                                   abs(ly - ry) / 2 + 0.06, spots))
    pad = ahead(*PAD_AT, arm="right")
    rocket, lift, flame, pad_body = T.rocket(ROCKET, pad, LIFT, PAD_TOP)
    bodies += [rocket, flame, pad_body]
    lamps = {LEFT: (pad[0], pad[1] + LAMP_APART), RIGHT: (pad[0], pad[1] - LAMP_APART)}
    bodies.append(T.lamp_posts("lamp_posts", tuple(lamps.values())))
    bodies += [T.indicator_lamp(LAMPS[arm], xy) for arm, xy in lamps.items()]
    joints.append(lift)
    return station_scene(f"bc3_launch_{level}", bodies, two_arms=True, articulations=joints,
                         frame=FRAME)


def launched(w: WorldState) -> bool:
    return w.articulations.get(LIFT, 0.0) > 0.5 * T.ROCKET_LIFT


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DI1")
    scene = build_scene(level)
    both = Custom(lambda w: pressed(w, LEFT) and pressed(w, RIGHT), "both buttons down")
    milestones = (
        Milestone("one_down", Custom(lambda w: pressed(w, LEFT) or pressed(w, RIGHT),
                                     "a button down")),
        Milestone("both_down", both),
        Milestone("ignited", Custom(lambda w: LAUNCHED in w.events_fired, "the rocket lit")),
        Milestone("lifted_off", Custom(launched, "the rocket up")),
    )
    latch = LATCHES[level]
    instruction = "Launch the rocket."
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=instruction,
        scorer=Scorer(task_id="DI1", level=level,
                      success=Finally(Custom(launched, "the rocket up")), milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda l=latch: Launcher(l),
        reference_factory=lambda inst: PressBothPolicy(),
        metadata={"knob": spec.knob.field, "knob_value": "latching" if latch else "spring back",
                  "hold_seconds": None if latch else HOLD_SECONDS,
                  "both_hands_needed": not latch, "max_model_calls": 30},
        extra_metrics={"both_down_seconds": lambda traj: round(
            sum(1 for w in traj if both.holds(w)) * 0.1, 1)},
    )


class Launcher(Reactor):
    """Locks a pressed button down (easy); lights a button's lamp while it is
    down; and lights the rocket once both lamps are lit -- together for
    ``HOLD_SECONDS`` at the hard level."""

    def __init__(self, latch: bool) -> None:
        super().__init__()
        self.latch = latch
        self._locked: Dict[str, bool] = {}
        self._lit: Dict[str, bool] = {}
        self._both_since: Optional[float] = None
        self._lit_at: Optional[float] = None

    def tick(self, engine: Engine, world: WorldState) -> None:
        if self._lit_at is not None:
            t = world.sim_time - self._lit_at
            climb = min(T.ROCKET_FLIGHT, 0.5 * CLIMB_ACCEL * t * t + 0.05 * t)
            engine.set_articulation_target(LIFT, climb)
            return
        if self.latch:
            for arm, spring in SPRINGS.items():
                if not self._locked.get(arm) and pressed(world, arm):
                    self._locked[arm] = True
                    self.act(engine, world, f"{BUTTONS[arm]}_locked",
                             DriveJoint(spring, -T.BUTTON_TRAVEL))
        for arm, lamp in LAMPS.items():
            lit = bool(self._locked.get(arm)) or pressed(world, arm)
            if lit != self._lit.get(arm, False):
                self._lit[arm] = lit
                self.act(engine, world, f"{lamp}_{'on' if lit else 'off'}",
                         SetColor(lamp, LAMP_LIT if lit else T.LAMP_DARK))
        if self._lit.get(LEFT) and self._lit.get(RIGHT):
            if self._both_since is None:
                self._both_since = world.sim_time
        else:
            self._both_since = None
        ready = self._both_since is not None and (
            self.latch or world.sim_time - self._both_since >= HOLD_SECONDS)
        if ready:
            self._lit_at = world.sim_time
            self.act(engine, world, LAUNCHED, SetColor(FLAME, (1.0, 0.55, 0.1)))
            self.say(world, ROCKET, "launched", "Liftoff!")


class PressBothPolicy(ScriptedPolicy):
    """Find out as a person would: try the left button alone -- down for a
    second and up; its lamp lights, the rocket does not go.  If it stayed
    down (its lamp still lit), press the right one; if it sprang back, both
    at once, held.  ``serial`` (the stress agent): one button, then the
    other, never together."""

    name = "press_both"

    def __init__(self) -> None:
        self.serial = False
        self.reset()

    def reset(self) -> None:
        self._step = 0
        self._arm = 0
        self._arms: List[str] = [LEFT, RIGHT]

    @staticmethod
    def _cap(world: WorldState, arm: str):
        x, y, _ = world.obj(BUTTONS[arm]).pose.pos
        top = T.BUTTON_REST_Z + T.BUTTON_HALF
        return x, y, top

    def _poses(self, world: WorldState, arm: str):
        x, y, top = self._cap(world, arm)
        up = Pose((x, y, top + FINGERTIP + ABOVE), GRASP_DOWN)
        down = Pose((x, y, top + FINGERTIP + PUSH_TO), GRASP_DOWN)
        return up, down

    def _press(self, world: WorldState, arms) -> List[Command]:
        out: List[Command] = [SetGripper(a, 1.0) for a in arms]
        out += [MoveTo(a, self._poses(world, a)[0]) for a in arms]
        out += [MoveTo(a, self._poses(world, a)[1]) for a in arms]
        return out

    def _finish(self, world: WorldState, lift) -> List[Command]:
        return lift + [Wait(1.0)] + to_rest(world, [LEFT, RIGHT]) + [Wait(1.5), Done("launched")]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self.serial:
            return self._one_then_other(world)
        step = self._step
        self._step += 1
        if step == 0:                                # the left one alone, to see what it does
            return self._press(world, [LEFT])
        if step in (1, 4):
            return [Wait(1.0)]                       # hold down
        if step == 2:
            return [MoveTo(LEFT, self._poses(world, LEFT)[0]), Wait(1.0)]
        if step == 3:
            if launched(world):
                return self._finish(world, [])
            # stayed down, its lamp lit: the other one; sprang back: both together
            self._arms = [RIGHT] if pressed(world, LEFT) else [LEFT, RIGHT]
            return self._press(world, self._arms)
        return self._finish(world, [MoveTo(a, self._poses(world, a)[0]) for a in self._arms])

    def _one_then_other(self, world: WorldState) -> List[Command]:
        arms = [LEFT, RIGHT]
        if self._arm >= len(arms):
            return to_rest(world, [LEFT, RIGHT]) + [Wait(1.5), Done("both pressed, one after the other")]
        arm = arms[self._arm]
        step = self._step
        self._step = (self._step + 1) % 3
        if step == 0:
            return self._press(world, [arm])
        if step == 1:
            return [Wait(1.0)]
        self._arm += 1
        return [MoveTo(arm, self._poses(world, arm)[0])]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self._press(world, [LEFT, RIGHT]) + [Done("pressed")]
