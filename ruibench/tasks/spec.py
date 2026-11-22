"""The task card.

A card is the single place a task is defined: what it asks for, what it
leaves out on purpose (the gap), which one field is the knob, what else changes
when that field changes, which basic abilities it draws on, and which split it
belongs to.  Cards are YAML so that a
reviewer can read the whole benchmark without reading Python; the executable
parts (predicates, events, scripted reference) live in ``ruibench/tasks/impl``
and are referenced by dotted path.

Task specs are **not** BDDL.  BDDL can state a static goal but cannot state an
event script, a knob or a process constraint such as "never touched the blue
block", and a second DSL would cost more than it returns (details §1.1).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Optional, Sequence, Tuple

#: The suites (10/4, "head and hand"): what has to be thought out before the
#: hand can do it.  What to do -- from the model's own knowledge, from what this
#: episode gives, from a fact that has to be found by acting -- when to do it,
#: as the scene changes -- or how to do it, where the difficulty is contact
#: (the fifth suite, v1.7).  Every task still leaves something out on purpose:
#: the instruction and the first picture are not enough.
#:
#: A task's id is its suite's prefix and a number.
SUITES = {
    "knowledge": ("WK", "World Knowledge",
                  "what to do comes from the model's own knowledge and reasoning; nothing on "
                  "the table gives the answer"),
    "learning": ("CL", "In-Context Learning",
                 "what to do comes from what this episode gives: a film, a picture, a rule "
                 "told on the spot"),
    "discovery": ("DS", "Discovery",
                  "what to do depends on a fact hidden in the scene, found only by acting"),
    "interaction": ("DI", "Dynamic Interaction",
                    "what to do is plain; when to do it is decided as the scene changes, and "
                    "time is short"),
    "contact": ("CR", "Contact-Rich Manipulation",
                "what to do is plain; how to do it has to be worked out, and the difficulty "
                "is in the contact: fitting, screwing, pushing, pressing, folding"),
}

#: paper order of the suites
SUITE_ORDER = tuple(SUITES)

#: The six basic abilities: tags on tasks, not suites.  Every task draws on at
#: least three; how much is ``primary`` or ``secondary``.
ABILITIES = {
    "LH": "Long-Horizon", "RS": "Reasoning", "ME": "Memory",
    "FG": "Fine-Grained", "SP": "Speed", "BI": "Bimanual",
}
ABILITY_WEIGHTS = ("primary", "secondary")


class Status(str, Enum):
    IMPLEMENTED = "implemented"   # built and runnable
    PLANNED = "planned"           # designed, being built this round
    DEFERRED = "deferred"         # a suite's 7th task: the future held-out template


class SplitRole(str, Enum):
    DEV = "dev"           # prompts, knobs, reference policy, scorer debugging
    MAIN = "main"         # the leaderboard
    HOLDOUT = "holdout"   # not released with v1


class AssetClass(str, Enum):
    A = "A"   # assembled from existing assets
    B = "B"   # needs physics tuning
    C = "C"   # needs a new scene


@dataclass(frozen=True)
class Knob:
    """The one declared configuration field this suite varies."""

    field: str
    description: str
    levels: Mapping[str, object]      # {"easy": ..., "hard": ...}
    unit: str = ""
    calibrated: bool = False          # True once Gate C has measured it

    @property
    def easy(self): return self.levels["easy"]

    @property
    def hard(self): return self.levels["hard"]


@dataclass(frozen=True)
class TaskSpec:
    id: str                      # WK1, CL2, ...
    suite: str
    name: str
    summary: str
    knob: Knob
    success: str                 # prose; the executable form is in impl
    milestones: Tuple[str, ...] = ()
    #: what else moves when the knob moves.  Required, and non-empty: a single
    #: config field does not by itself make a single-factor experiment (v0 §1.1).
    confounds: Tuple[str, ...] = ()
    #: what the task leaves out: the information the instruction and the first
    #: picture do not give, that the agent must get through the robot
    gap: str = ""
    #: the same task with the gap filled in (told in words, or the world held
    #: still): what is left is execution
    gap_filled: str = ""
    #: an agent spec that executes perfectly but never fills the gap; it must do
    #: no better than chance.  Empty until that agent exists.
    blind: str = ""
    #: basic abilities drawn on: {"LH": "primary", "FG": "secondary", ...}
    abilities: Mapping[str, str] = field(default_factory=dict)
    #: the v2 id, so older logs can still be matched
    formerly: str = ""
    asset_class: AssetClass = AssetClass.A
    base_env: str = ""
    #: free-text list of what the scene contains, for the reader
    assets: Tuple[str, ...] = ()
    #: logical names from the Isaac asset catalogue.  A card never names a USD
    #: path: one indirection means a moved or missing asset changes one file,
    #: not twenty-eight task cards.
    catalogue_assets: Tuple[str, ...] = ()
    arms: int = 1
    status: Status = Status.PLANNED
    split_role: SplitRole = SplitRole.MAIN
    impl: str = ""               # dotted path to the builder module
    negative_controls: Tuple[str, ...] = ()
    #: an agent spec (``policies.build_agent``) whose failure shows how this
    #: task goes wrong, for the showcase: a control that removes the one piece
    #: of information the task hinges on, or a misperceiving reference
    failure_example: str = ""
    risks: Tuple[str, ...] = ()
    notes: str = ""
    #: where the mission comes from (v4): ``grade`` A -- a named paradigm from
    #: developmental or comparative psychology with children's data; B -- an
    #: established children's game, puzzle or school practical; C -- our own.
    #: ``refs``: the citations; ``children``: what the source found in children,
    #: for the robot-and-child comparison.  A benchmark whose missions are games
    #: we made up is one whose choice of missions is arbitrary.
    source: Mapping[str, object] = field(default_factory=dict)
    #: told to the agent with the instruction: how the things on the bench
    #: work that a person working the arm would need told too -- what a bell
    #: does, what a weight weighs, what counts as a touch.  Never the task's
    #: ``gap`` (v5, 9/22).  A task may word it per level (``TaskInstance.hint``)
    hint: str = ""
    #: what of this task must never reach an agent, in words; each episode's
    #: own strings are ``TaskInstance.secrets`` (v15)
    secrets: str = ""

    @property
    def source_grade(self) -> str:
        return str(self.source.get("grade", "")) if self.source else ""

    @property
    def prefix(self) -> str:
        return SUITES[self.suite][0]

    @property
    def suite_name(self) -> str:
        return SUITES[self.suite][1]

    @property
    def levels(self) -> Tuple[str, ...]:
        return tuple(self.knob.levels)

    def as_dict(self) -> dict:
        return {
            "id": self.id, "suite": self.suite, "suite_name": self.suite_name,
            "name": self.name, "summary": self.summary,
            "knob": {"field": self.knob.field, "description": self.knob.description,
                     "levels": dict(self.knob.levels), "unit": self.knob.unit,
                     "calibrated": self.knob.calibrated},
            "success": self.success, "milestones": list(self.milestones),
            "confounds": list(self.confounds), "gap": self.gap, "gap_filled": self.gap_filled,
            "blind": self.blind, "abilities": dict(self.abilities), "formerly": self.formerly,
            "asset_class": self.asset_class.value, "base_env": self.base_env,
            "assets": list(self.assets), "catalogue_assets": list(self.catalogue_assets),
            "arms": self.arms,
            "status": self.status.value, "split_role": self.split_role.value,
            "impl": self.impl, "negative_controls": list(self.negative_controls),
            "failure_example": self.failure_example,
            "risks": list(self.risks), "notes": self.notes,
            "source": dict(self.source),
        }
