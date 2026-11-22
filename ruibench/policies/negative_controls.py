"""The negative controls.

Each one removes exactly one kind of information, and each is checked only
against the thing it removes (v0 §5.4).  **They are not required to fail
everywhere**: a static planning task solved from the first frame alone is a
correct outcome, and ignoring an event in a condition that has no event should
of course succeed.  Demanding universal failure would be a misreading of what a
negative control is for.

Every control is an ablation of the same scripted reference, so the difference
between control and reference is the information, not the competence.
"""

from __future__ import annotations

from typing import List, Optional

from ..core.types import WorldState
from ..interfaces.base import TurnContext
from ..interfaces.commands import Command, Done
from .base import ScriptedPolicy

#: control name -> what it is allowed to prove
CONTROL_CHECKS = {
    "do_nothing": "the task requires acting at all",
    "first_frame_only": "the task requires looking again after the first frame",
    "ignore_feedback": "the task requires using the correct/wrong signal",
    "ignore_change": "the task requires noticing the mid-episode change",
    "ignore_demonstration": "the task requires using the demonstration it shows first",
}


class DoNothing(ScriptedPolicy):
    """Never moves.  Establishes that the task is not passed by standing still."""

    name = "do_nothing"

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return [Done("did nothing")]


class FirstFrameOnly(ScriptedPolicy):
    """Commits to a whole plan from the first observation, then executes it open
    loop and never looks again.

    Checks that later observations carry necessary information.  **Static tasks
    should pass this** -- a pick-and-place whose geometry is fully visible at
    t=0 really is solvable from the first frame, and a control that failed there
    too would be removing something other than what it claims.

    Judge it against the right baseline: on a task where the deciding evidence
    is hidden at t=0 and there are two possible answers, an open-loop plan
    guesses, so the bar is chance, not zero.
    """

    name = "first_frame_only"

    def __init__(self, inner: ScriptedPolicy) -> None:
        self.inner = inner
        self._planned = False

    def reset(self) -> None:
        self.inner.reset()
        self._planned = False

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if self._planned:
            return [Done("plan already issued; not looking again")]
        self._planned = True
        return self.inner.plan(world, ctx)


class IgnoreChange(ScriptedPolicy):
    """Carries out the original plan as if no event ever happened: the
    instruction is the first one, and whatever a driven joint moves by itself
    (a body marked ``attrs["driven"]``) is where it was first seen.  What the
    agent moves itself -- a drawer it pulls -- is seen as it is.

    The control for the Responsiveness suite: it should fail the dynamic levels
    and pass the matched static control.
    """

    name = "ignore_change"

    def __init__(self, inner: ScriptedPolicy) -> None:
        self.inner = inner
        self._first: Optional[WorldState] = None

    def reset(self) -> None:
        self.inner.reset()
        self._first = None

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from dataclasses import replace
        if self._first is None:
            self._first = world
        objects = {name: (replace(obj, pose=self._first.obj(name).pose, lin_vel=(0.0, 0.0, 0.0))
                          if obj.attrs.get("driven") else obj)
                   for name, obj in world.objects.items()}
        frozen = replace(world, instruction_version=0, objects=objects)
        return self.inner.commands(frozen, ctx)


class IgnoreFeedback(ScriptedPolicy):
    """Never reads the one-bit correct/wrong signal.

    The control for the In-Context Learning suite.
    """

    name = "ignore_feedback"

    def __init__(self, inner: ScriptedPolicy) -> None:
        self.inner = inner

    def reset(self) -> None:
        self.inner.reset()

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        ctx.extras = dict(ctx.extras, feedback=())
        return self.inner.commands(world, ctx)


class IgnoreDemonstration(ScriptedPolicy):
    """Never looks at the demonstrations played before the episode."""

    name = "ignore_demonstration"

    def __init__(self, inner: ScriptedPolicy) -> None:
        self.inner = inner

    def reset(self) -> None:
        self.inner.reset()

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        ctx.extras = dict(ctx.extras, demonstrations=())
        return self.inner.commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        ctx.extras = dict(ctx.extras, demonstrations=())
        return self.inner.plan(world, ctx)


def wrap(control: str, inner: ScriptedPolicy) -> ScriptedPolicy:
    """Build a named negative control around a task's scripted reference."""
    if control == "do_nothing":
        return DoNothing()
    if control == "first_frame_only":
        return FirstFrameOnly(inner)
    if control == "ignore_change":
        return IgnoreChange(inner)
    if control == "ignore_feedback":
        return IgnoreFeedback(inner)
    if control == "ignore_demonstration":
        return IgnoreDemonstration(inner)
    raise KeyError(f"unknown negative control {control!r}; have {sorted(CONTROL_CHECKS)}")
