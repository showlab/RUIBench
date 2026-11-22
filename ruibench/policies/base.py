"""Agents: anything that answers a model turn.

The runner only knows this protocol, so a frontier model, a scripted reference
and a negative control are interchangeable.  That is what makes the three-layer
solvability check and the four negative controls run through exactly the same
pipeline as a real evaluation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from ..core.types import WorldState
from ..interfaces.base import ModelTurn, TurnContext
from ..interfaces.commands import Command


class Agent(ABC):
    name: str = "agent"
    #: wall-clock seconds this agent takes per call.  Scripted agents cost 0;
    #: a real model reports what the API actually took, which the measured
    #: clock then converts into simulated time.
    latency_seconds: float = 0.0
    usd_per_call: float = 0.0

    @abstractmethod
    def act(self, ctx: TurnContext, turn: ModelTurn) -> str:
        """Return a raw response in the interface's action language."""

    def reset(self) -> None:
        """Called once at the start of each episode."""


class ScriptedPolicy(Agent):
    """A privileged reference: it reads ground truth and emits commands.

    Used for

    * **layer 1 of solvability** -- does the scene work at all, is the success
      criterion reachable;
    * **knob calibration** -- run it 25 times and see what the physics actually
      does before fixing a level;
    * **the negative controls**, which are ablations of this policy;
    * **the demonstrations** shown in CL1 / CL2, recorded as frames.

    Passing through a *scripted* policy proves nothing about an interface.  For
    layer 2 ("the interface can express it") the same policy must go through the
    interface under test.
    """

    name = "scripted"

    @abstractmethod
    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """The next batch of commands, given privileged world state."""

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """The **whole** sequence, computed from this one world state.

        Only the `first_frame_only` negative control uses it, and it is what
        makes that control mean what it claims.  A control built by replaying a
        stale world through a closed-loop policy removes far more than later
        observations -- it removes the policy's knowledge of its own progress --
        and then fails everywhere, which would falsely "validate" tasks that a
        first-frame plan really can solve.

        Override it wherever the task is plannable open-loop.  The default falls
        back to one closed-loop step, which is the honest answer for a policy
        that cannot commit to a full plan.
        """
        return self.commands(world, ctx)

    def act(self, ctx: TurnContext, turn: ModelTurn) -> str:
        from ..interfaces.direct import encode
        world = ctx.extras.get("world")
        if world is None:
            raise RuntimeError(
                "a scripted policy needs privileged world state; run it on the "
                "I0_direct interface, or render its commands into an interface's "
                "action language for a layer-2 check"
            )
        return encode(self.commands(world, ctx))
