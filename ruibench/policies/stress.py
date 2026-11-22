"""Stress agents: the scripted reference with one competence degraded.

Not negative controls.  A negative control removes one kind of *information*
and is judged only where that information matters.  A stress agent keeps every
bit of information and degrades an *ability* -- here, knowing where things are.
It is how the benchmark shows that its tasks punish the errors real interfaces
make (a pointing interface lifting a pixel to 3-D is off by centimetres), and it
is where the viewer's failure examples come from.
"""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Dict, List, Tuple

from ..core.types import Pose, WorldState
from ..interfaces.base import TurnContext
from ..interfaces.commands import Command
from .base import ScriptedPolicy


class Misperceiving(ScriptedPolicy):
    """Sees every object ``sigma`` metres (per axis, in the table plane) from
    where it is.

    The error is fixed per object for the whole episode -- a mislocalisation,
    not jitter -- so the agent acts consistently on a wrong belief: it reaches
    for the same wrong spot every time, the way a grounding error behaves.
    Where the arm itself is stays exact.
    """

    def __init__(self, inner: ScriptedPolicy, sigma: float = 0.02, seed: int = 0) -> None:
        self.inner = inner
        self.sigma = float(sigma)
        self.seed = int(seed)
        self.name = f"{getattr(inner, 'name', 'agent')}+misperceive({self.sigma:g} m)"
        self._offsets: Dict[str, Tuple[float, float]] = {}

    def reset(self) -> None:
        self.inner.reset()
        self._offsets = {}

    def offset(self, name: str) -> Tuple[float, float]:
        if name not in self._offsets:
            rng = random.Random(f"misperceive/{self.seed}/{name}")
            self._offsets[name] = (rng.gauss(0.0, self.sigma), rng.gauss(0.0, self.sigma))
        return self._offsets[name]

    def seen(self, world: WorldState) -> WorldState:
        objects = {}
        for name, obj in world.objects.items():
            dx, dy = self.offset(name)
            x, y, z = obj.pose.pos
            objects[name] = replace(obj, pose=Pose((x + dx, y + dy, z), obj.pose.quat))
        return replace(world, objects=objects)

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.commands(self.seen(world), ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.plan(self.seen(world), ctx)


class Serial(ScriptedPolicy):
    """Uses one arm at a time.  Every bit of information is still there --
    what is lost is concurrency.  Only a two-arm reference that knows how to
    take turns (a ``serial`` attribute) can be run this way."""

    def __init__(self, inner: ScriptedPolicy) -> None:
        from ..core.errors import ConfigError

        if not hasattr(inner, "serial"):
            raise ConfigError(f"{getattr(inner, 'name', inner)!r} cannot take turns "
                              "between arms; the serial modifier needs a two-arm reference")
        inner.serial = True
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'agent')}+serial"

    def reset(self) -> None:
        self.inner.reset()

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.plan(world, ctx)


class IgnoreRules(ScriptedPolicy):
    """Knows where everything is and what the goal is, and plays as if the
    game had no rules -- the tower moved ring by ring straight to its peg, the
    river crossed in a fixed order.  What is lost is the rule, so the failure
    it shows is a broken rule, not a missed goal.  Only a reference that can
    play that way (an ``ignore_rules`` attribute) can be run like this."""

    def __init__(self, inner: ScriptedPolicy) -> None:
        from ..core.errors import ConfigError

        if not hasattr(inner, "ignore_rules"):
            raise ConfigError(f"{getattr(inner, 'name', inner)!r} has no rule-blind play; "
                              "the ignore_rules modifier needs a reference from a rules task")
        inner.ignore_rules = True
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'agent')}+ignore_rules"

    def reset(self) -> None:
        self.inner.reset()

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.plan(world, ctx)


class Greedy(ScriptedPolicy):
    """Plays for its own lines only: takes a win when it has one, and
    otherwise the move that makes the most of its own threats -- without ever
    looking at the opponent's.  What is lost is looking ahead, so the failure
    it shows is a game lost to a threat left open.  Only a reference from a
    game with an opponent (a ``greedy`` attribute) can be run like this."""

    def __init__(self, inner: ScriptedPolicy) -> None:
        from ..core.errors import ConfigError

        if not hasattr(inner, "greedy"):
            raise ConfigError(f"{getattr(inner, 'name', inner)!r} has no greedy play; "
                              "the greedy modifier needs a reference from a game task")
        inner.greedy = True
        self.inner = inner
        self.name = f"{getattr(inner, 'name', 'agent')}+greedy"

    def reset(self) -> None:
        self.inner.reset()

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        return self.inner.plan(world, ctx)
