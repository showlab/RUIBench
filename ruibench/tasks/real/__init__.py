"""The tasks set up on the real station (``tasks.yaml``; the engine is ``engines.franka``).

A real task is a table an operator lays out, not a scene a simulator builds:
its card says what the agent is told, what is on the table, how the operator
lays it out before each episode and how the operator judges its end.  Built,
it is a ``TaskInstance`` like any other -- the same episode, interfaces and
tools run it -- with three differences:

* its scene is the real station's (``engines.franka.scene.station_scene``):
  the robot where the calibration puts it, its cameras, and the things the
  card names, declared but never placed;
* it is built as level ``easy``, and a seed is a table the operator lays out
  rather than one a simulator draws: the card says what each of its seeds has
  (``seeds``), and one that says nothing has the one table, seed 0;
* a picture it needs is no film played on the engine: the card names a file
  (``picture``, in ``cards/``), and ``task_demo`` shows it (``Drawn``);
* its success is the operator's verdict (``OperatorVerdict``): no state of the
  world is known to judge it from, so the episode's own score is never a
  success, and the runner records what the operator saw
  (``ruikit.experiments.real``).  A game the program plays (``simon.py``) keeps a
  score of its own, for the operator to be shown; the verdict is still the result.

Not in the frozen benchmark's fingerprint: a real task changes with the
table it is laid out on, and no simulation result depends on one.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import yaml

from ...core.canary import CANARY
from ...core.config import RunConfig
from ...core.errors import ConfigError
from ...core.types import CameraFrame
from ...scoring import Scorer
from ...scoring.predicates import TrajectoryPredicate
from ..instance import Demonstration, TaskInstance
from ..spec import Knob, Status, TaskSpec

CARDS = Path(__file__).with_name("tasks.yaml")
#: the pictures the cards name
PICTURES = Path(__file__).with_name("cards")
#: the level a real task is built at, and the seed of a card that names none
LEVEL, SEED = "easy", 0


@dataclass(frozen=True)
class RealTask:
    """One card of ``tasks.yaml``."""
    id: str
    name: str
    summary: str
    instruction: str
    setup: str
    success: str
    things: Tuple[Tuple[str, str], ...] = ()
    hint: str = ""
    mirrors: str = ""
    max_model_calls: int = 60
    #: what sets each seed's table apart, on top of ``setup``, by its number
    seeds: Dict[int, str] = field(default_factory=dict)
    #: a picture ``task_demo`` shows (a file in ``cards/``), and what the agent is told it shows
    picture: str = ""
    caption: str = ""
    #: a game the program plays with the agent (``simon.py``), and the toys it is played on
    game: str = ""
    toys: Tuple["Toy", ...] = ()

    def setup_of(self, seed: int) -> str:
        """How the operator lays out this seed's table; a seed the card has not is said."""
        if int(seed) not in (self.seeds or {SEED: ""}):
            raise ConfigError(f"{self.id} has no seed {seed}: its card has "
                              f"{sorted(self.seeds or {SEED})} (ruibench/tasks/real/tasks.yaml)")
        return " ".join(filter(None, (self.setup, self.seeds.get(int(seed)))))


@dataclass(frozen=True)
class Drawn(Demonstration):
    """A picture a real task shows its agent, from a file: nothing moves, and nothing of
    it is on the table (``runner.demonstration.play``)."""
    path: Path = PICTURES

    def drawn(self, cameras: Sequence[str]) -> Dict[str, CameraFrame]:
        import numpy as np
        from PIL import Image

        image = np.asarray(Image.open(self.path).convert("RGB"))
        return {name: CameraFrame(camera=name, width=image.shape[1], height=image.shape[0],
                                  captured_at=0.0, image=image) for name in cameras}


@dataclass
class OperatorVerdict(TrajectoryPredicate):
    """Success as the operator sees it at the end: nothing in the trajectory
    can tell it, so it never holds here (``ruikit.experiments.real`` records
    the verdict)."""
    criterion: str

    def evaluate(self, traj) -> bool:
        return False

    def describe(self) -> str:
        return f"judged by the operator: {self.criterion}"


_FIELDS = set(RealTask.__dataclass_fields__)


@functools.lru_cache(maxsize=1)
def tasks() -> Dict[str, RealTask]:
    """The real tasks, by id."""
    out: Dict[str, RealTask] = {}
    for d in yaml.safe_load(CARDS.read_text(encoding="utf-8")).get("tasks") or []:
        unknown = set(d) - _FIELDS
        if unknown:
            raise ConfigError(f"{d.get('id', '?')}: unknown real-task fields {sorted(unknown)}")
        things = tuple((str(n), str(c)) for n, c in d.get("things") or ())
        text = {k: " ".join(str(d.get(k) or "").split())
                for k in ("summary", "instruction", "setup", "success", "hint", "caption")}
        seeds = {int(k): " ".join(str(v).split()) for k, v in (d.get("seeds") or {}).items()}
        out[str(d["id"])] = RealTask(id=str(d["id"]), name=str(d["name"]), things=things,
                                     mirrors=str(d.get("mirrors") or ""), seeds=seeds,
                                     picture=str(d.get("picture") or ""),
                                     game=str(d.get("game") or ""), toys=_toys(d),
                                     max_model_calls=int(d.get("max_model_calls", 60)), **text)
    return out


def _toys(d: dict) -> Tuple[object, ...]:
    if not d.get("toys"):
        return ()
    from .simon import Toy

    return tuple(Toy(str(t["name"]), str(t["called"]), tuple(t["at"]), tuple(t["half"]),
                     float(t["top"])) for t in d.get("toys") or ())


def get(task_id: str) -> RealTask:
    try:
        return tasks()[task_id]
    except KeyError:
        raise ConfigError(f"no real task {task_id!r}: the real station has "
                          f"{sorted(tasks())} (ruibench/tasks/real/tasks.yaml)") from None


def build(task_id: str, level: str = LEVEL, seed: Optional[int] = None,
          config: Optional[RunConfig] = None) -> TaskInstance:
    """A real task as an episode runs it (the module docstring); no seed named, the card's first."""
    from ...engines.franka.scene import station_scene

    card = get(task_id)
    if level != LEVEL:
        raise ConfigError(f"{task_id} is built as level {LEVEL!r}: the operator lays the table out")
    seed = min(card.seeds or {SEED}) if seed is None else int(seed)
    setup = card.setup_of(seed)
    suite = _suite_of(card.mirrors)
    spec = TaskSpec(id=card.id, suite=suite, name=card.name, summary=card.summary,
                    knob=Knob("setup", "the table the operator lays out", {LEVEL: setup}),
                    success=card.success, status=Status.IMPLEMENTED, hint=card.hint,
                    notes=f"mirrors {card.mirrors}" if card.mirrors else "")
    scorer = Scorer(task_id=card.id, level=LEVEL, success=OperatorVerdict(card.success))
    reactor = None
    if card.game:
        from .simon import simon_game

        scorer, reactor = simon_game(card.id, card.toys, LEVEL, seed)
    scene = station_scene(card.id, card.things)
    shown = (Drawn(scene=scene, demonstrator=lambda: None, caption=card.caption, picture=True,
                   path=PICTURES / card.picture),) if card.picture else ()
    return TaskInstance(spec=spec, level=LEVEL, seed=seed, scene=scene, demonstrations=shown,
                        reactor_factory=reactor, instruction=card.instruction, scorer=scorer, hint=card.hint,
                        metadata={"max_model_calls": card.max_model_calls, "real": True,
                                  "mirrors": card.mirrors, "setup": setup},
                        secrets=(CANARY,))


def _suite_of(task_id: str) -> str:
    """The suite of the simulated task a real one stands for; '' for none."""
    if not task_id:
        return ""
    from ..loader import get as sim_card

    return sim_card(task_id).suite


__all__ = ["CARDS", "Drawn", "LEVEL", "OperatorVerdict", "PICTURES", "RealTask", "SEED", "build",
           "get", "tasks"]
