"""One task at one level with one seed: everything an episode needs.

A task implementation module exposes a single function::

    def build(level: str, seed: int, config: RunConfig) -> TaskInstance

and nothing else.  Keeping the signature this narrow is what lets the runner,
the negative controls, the scorer and the report layer stay task-agnostic.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Callable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

from ..core.config import RunConfig
from ..core.errors import ConfigError
from ..core.types import WorldState
from ..engines.base import SceneSpec
from ..events import Event
from ..scoring import Milestone, Scorer
from .loader import get
from .spec import Status, TaskSpec


@dataclass(frozen=True)
class Demonstration:
    """A scripted demonstration the agent watches before its episode.

    Shown as frames from the episode's own cameras at the episode's own
    resolution -- never as code: source would hand the solution to the program
    interface and break the equal-information premise (CL1's card).  Its scene
    must have the episode's structure (poses and colours may differ), because
    an engine keeps one stage per process.
    """

    scene: SceneSpec
    #: builds the scripted demonstrator (a ``ScriptedPolicy``)
    demonstrator: Callable[[], object]
    #: what the agent is told the demonstration shows
    caption: str
    max_seconds: float = 90.0
    #: the toys' behaviour during the film (CL6's blender blending), if any
    reactor_factory: Optional[Callable[[], object]] = None
    #: where the film was shot, if not at the episode's table: another table
    #: and another light (CL2, v3), so what is learnt is the thing, not the
    #: picture.  Keys (``engines.isaac.environment.LOOK_KEYS``): ``table`` --
    #: a finish (wood:ash, wood:plywood, fabric:<colour>); ``light`` -- the
    #: studio light's strength, times the usual; ``light_colour`` -- its tint;
    #: ``light_turn`` -- degrees it is turned about the vertical.  Applied for
    #: the film only, and never part of the scene's structure.
    look: Optional[Mapping[str, object]] = None
    #: a still picture, not a film: the scene as it is, one frame, and the
    #: demonstrator never run (CL5, v6: a finished card)
    picture: bool = False


@dataclass
class TaskInstance:
    spec: TaskSpec
    level: str
    seed: int
    scene: SceneSpec
    instruction: str
    scorer: Scorer
    milestones: Tuple[Milestone, ...] = ()
    events: Tuple[Event, ...] = ()
    #: builds the privileged scripted reference -- layer 1 of the three-layer
    #: solvability check, and the thing the negative controls are ablations of
    reference_factory: Optional[Callable[["TaskInstance"], object]] = None
    #: the one-bit correct/wrong channel of the In-Context Learning suite.
    #: A factory, because the channel is stateful and each episode needs its own.
    feedback_factory: Optional[Callable[[], object]] = None
    #: the toys that answer (``events.reactor``): a monkey that eats, an
    #: opponent that moves.  A factory, for the same reason.
    reactor_factory: Optional[Callable[[], object]] = None
    #: values of the knob and anything else worth recording per episode
    metadata: Mapping[str, object] = field(default_factory=dict)
    #: told to the agent with the instruction (``TaskSpec.hint``); a task
    #: that words it per level sets it, the rest get their card's
    hint: str = ""
    #: numbers reported next to the success rate and never folded into it -- a
    #: continuous placement error, a lenient second scoring.  Each is computed
    #: from the finished trajectory and lands in the log's ``task_metrics``.
    extra_metrics: Mapping[str, Callable[[Sequence[WorldState]], object]] = field(
        default_factory=dict)
    #: played, in order, before the episode (the In-Context suite)
    demonstrations: Tuple[Demonstration, ...] = ()
    #: strings nothing shown to an agent may contain (v15): the canary, and
    #: what names this episode's answer -- the fake coin's body, the pot the egg
    #: is under.  Each is specific: at least four characters, and in no
    #: category, instruction, hint or caption the agent is shown
    #: (``tests/ruibench/test_release.py``).  A task module's ``secrets(inst)``
    #: gives its own
    secrets: Tuple[str, ...] = ()

    @property
    def task_id(self) -> str:
        return self.spec.id

    @property
    def knob_value(self):
        return self.spec.knob.levels[self.level]


def build(task_id: str, level: str = "easy", seed: int = 0,
          config: Optional[RunConfig] = None) -> TaskInstance:
    """Instantiate a task.  Raises a readable error if it is not built yet."""
    spec = get(task_id)
    if level not in spec.knob.levels:
        raise ConfigError(f"{task_id}: unknown level {level!r}; have {sorted(spec.knob.levels)}")
    if spec.status is Status.DEFERRED:
        raise ConfigError(
            f"{task_id} is the held-out fourth task of its suite and is deliberately not "
            "implemented.  Implementing it spends the benchmark's only clean "
            "extrapolation check (v0 §2.4)."
        )
    if not spec.impl:
        raise ConfigError(
            f"{task_id} ({spec.name}) has no implementation yet. "
            f"Add ruibench/tasks/impl/<module>.py exposing build(level, seed, config) "
            f"and set `impl:` in ruibench/tasks/cards/{spec.suite}.yaml."
        )
    module = importlib.import_module(spec.impl)
    inst = module.build(level=level, seed=seed, config=config or RunConfig())
    if not inst.hint and spec.hint:
        from dataclasses import replace

        inst = replace(inst, hint=spec.hint)
    # the physics audit's view of the scene (``scoring/audit.py``)
    inst.scorer.joint_limits = {j.name: tuple(j.limits) for j in inst.scene.articulations}
    inst.scorer.fixed = tuple(b.name for b in inst.scene.bodies if b.fixed)
    from ..core.canary import CANARY

    own = getattr(module, "secrets", None)
    inst.secrets = (CANARY,) + tuple(dict.fromkeys(str(s) for s in (own(inst) if own else ())))
    return inst


def implemented() -> Tuple[str, ...]:
    from .loader import load_all
    return tuple(t.id for t in load_all() if t.status is Status.IMPLEMENTED)


def assets_shown(task_ids: Optional[Iterable[str]] = None, seeds: Iterable[int] = range(10)) -> Set[str]:
    """Every catalogue asset these tasks (default: all built ones) put on the
    table, at any level and any of these seeds -- what the credits must name."""
    seeds = tuple(seeds)
    out = set()
    for t in (implemented() if task_ids is None else task_ids):
        for level in get(t).knob.levels:
            for seed in seeds:
                for b in build(t, level, seed).scene.bodies:
                    if b.asset:
                        out.add(b.asset)
                    # a scan drawn over hidden parts (``parts.visual``): WK2's
                    # farmer, its bank stones -- credited as any other
                    out.update(part[4].split("@", 1)[0] for part in b.attrs.get("parts") or ()
                               if part[0] == "asset")
    return out
