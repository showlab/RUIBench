"""The Robot-Use Interface (RUI) contract.

**This module is the seam of the whole benchmark.**  Four interfaces share one
robot, one controller, one set of perception tools and one clock; the only
things that differ are *what the model emits*, *how often it may speak* and
*whether it can take an action back* (design doc v0 §4).

Implementing an interface means filling in three methods:

===========================  ======================================================
:meth:`Interface.system_prompt`  the action language, in words, frozen per interface
:meth:`Interface.observe`        which part of the observation the model receives
:meth:`Interface.parse`          model text -> ``list[Command]``
===========================  ======================================================

Two rules that are easy to get wrong:

1. **Never reach for the engine.**  Emit commands; the executor runs them.
   This is what keeps the four interfaces comparable.
2. **"Cannot express" is not "failed".**  If a task needs something your action
   language genuinely cannot say (the classic: a pixel + an action type cannot
   carry a 6-DoF orientation, so WK3 and the Discovery look-arounds are
   at risk for I3), return a :class:`CoverageGap` from :meth:`coverage_gap`.
   Coverage gaps are reported in their own column and never counted as zeros
   (v0 §7.1, TODO §3).
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Mapping, Optional, Sequence, Tuple

from ..core.types import Observation
from .commands import Command


class Expressivity(str, Enum):
    """How a quantity can be named in an action language."""

    NONE = "none"              # cannot be said at all
    SYMBOLIC = "symbolic"      # "down", "left a bit", "grasp"
    RELATIVE = "relative"      # numeric offset from the current pose
    ABSOLUTE = "absolute"      # numeric pose in the world frame
    PIXEL = "pixel"            # a point in an image, lifted to 3D by a service


@dataclass(frozen=True)
class InterfaceContract:
    """The contract table of TODO §3, as data.

    Filled in once per interface and checked by
    ``tests/ruibench/test_interface_contract.py``.  It is also printed by
    ``ruibench interfaces`` and rendered into the paper's appendix table, so
    the claim "these interfaces differ only in X" is auditable instead of
    asserted.
    """

    position: Expressivity
    orientation: Expressivity
    gripper: Expressivity
    blocking: bool                 # does one model turn wait for the motion to finish?
    cancellable: bool              # can the model take back an action in flight?
    concurrent_arms: bool          # can it drive two arms at once?
    camera_control: str            # none | via_arm | explicit
    services: Tuple[str, ...]      # perception tools it may call
    feedback_delivery: str         # none | between_turns | inline
    reentry: str                   # per_step | per_target | per_program
    notes: str = ""

    def as_row(self) -> dict:
        return {
            "position": self.position.value, "orientation": self.orientation.value,
            "gripper": self.gripper.value, "blocking": self.blocking,
            "cancellable": self.cancellable, "concurrent_arms": self.concurrent_arms,
            "camera_control": self.camera_control, "services": list(self.services),
            "feedback_delivery": self.feedback_delivery, "reentry": self.reentry,
        }


@dataclass
class TurnContext:
    """Everything an interface may look at when building one model turn."""

    observation: Observation
    task_id: str
    instruction: str
    arm_ids: Tuple[str, ...]
    turn_index: int
    sim_time: float
    budget_left: Mapping[str, float]
    history: Sequence["ModelTurn"] = field(default_factory=list)
    #: always None: perception is RUIKit's tools now (its estimated world); the
    #: benchmark's own services were removed in v16.  Kept so callers that name it work
    services: object = None
    extras: Mapping[str, object] = field(default_factory=dict)
    #: told with the instruction: how the things on the bench work
    #: (``TaskInstance.hint``); "" for a task that needs none
    hint: str = ""


@dataclass
class ModelTurn:
    """One request/response pair, recorded verbatim in the episode log."""

    turn_index: int
    prompt: str
    images: Tuple[str, ...] = ()
    raw_response: str = ""
    requested_at: float = 0.0        # sim time the request was issued
    observed_at: float = 0.0         # sim time of the newest frame in the request
    returned_at: float = 0.0         # sim time the response landed
    latency_seconds: float = 0.0     # wall-clock latency of the API call
    usd: float = 0.0
    tokens: Mapping[str, int] = field(default_factory=dict)

    @property
    def observation_age(self) -> float:
        """How stale the picture was by the time the answer arrived.

        This is the quantity the mechanism figure is built on (v0 §5.3).
        """
        return max(0.0, self.returned_at - self.observed_at)


@dataclass
class ParsedAction:
    """The result of reading a model response.

    Exactly one of these is meaningful: ``commands`` (normal), ``parse_error``
    (the model said something the language does not accept -- a *model* failure)
    or ``coverage_gap`` (the language cannot express what the task needs -- an
    *interface* limitation, reported separately).
    """

    commands: List[Command] = field(default_factory=list)
    parse_error: Optional[str] = None
    coverage_gap: Optional[str] = None
    thought: str = ""

    @property
    def ok(self) -> bool:
        return self.parse_error is None and self.coverage_gap is None


class Interface(ABC):
    """Base class for the four Robot-Use Interfaces."""

    #: short id used in configs, logs and figures ("I1".."I4")
    id: str = ""
    #: human-readable name used in the paper
    name: str = ""
    #: where the action language comes from, cited in the appendix
    source: str = ""
    #: True for the four interfaces under study; False for harness test doubles
    is_benchmark_interface: bool = True

    @property
    @abstractmethod
    def contract(self) -> InterfaceContract:
        """The interface's row of the contract table."""

    # -- the three methods an implementation fills in -----------------------
    @abstractmethod
    def system_prompt(self, ctx: TurnContext) -> str:
        """The frozen description of this action language.

        Tuned on the dev split only, then frozen; its hash goes into every log
        (design doc details §3.2, §4.3).
        """

    @abstractmethod
    def observe(self, ctx: TurnContext) -> ModelTurn:
        """Assemble the request: which frames, which text, which feedback."""

    @abstractmethod
    def parse(self, raw: str, ctx: TurnContext) -> ParsedAction:
        """Compile one model response into shared commands."""

    # -- optional overrides -------------------------------------------------
    def coverage_gap(self, task) -> Optional[str]:
        """Return a reason string if this interface cannot express ``task``.

        Called before the episode runs.  A non-None result skips the episode
        and records a coverage gap instead of a failure.
        """
        return None

    def on_feedback(self, ctx: TurnContext, feedback: str) -> None:
        """Delivery hook for the 1-bit correct/wrong signal.

        Every interface must receive it, the program interface included
        (v0 §2, In-Context Learning control requirements).
        """

    def prompt_hash(self, ctx: TurnContext) -> str:
        return hashlib.sha256(self.system_prompt(ctx).encode("utf-8")).hexdigest()[:16]

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"<{type(self).__name__} {self.id} {self.name!r}>"


class NotImplementedInterface(Interface):
    """Base for the four stubs below: a clear, actionable failure.

    The interfaces are being written separately; until then every entry point
    says exactly which method is missing rather than failing deep inside a run.
    """

    def _todo(self, method: str):
        raise NotImplementedError(
            f"{self.id} ({self.name}) is not implemented yet: {method}(). "
            f"See docs/INTERFACES.md and {self.source or 'the design doc'}."
        )

    def system_prompt(self, ctx: TurnContext) -> str:
        self._todo("system_prompt")

    def observe(self, ctx: TurnContext) -> ModelTurn:
        self._todo("observe")

    def parse(self, raw: str, ctx: TurnContext) -> ParsedAction:
        self._todo("parse")
