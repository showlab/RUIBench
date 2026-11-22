"""What an event does to the world.

All three main-board event kinds live here: move an object, rewrite the
instruction, change a colour.  They go through the engine's event hooks, never
around them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

from ..core.types import Pose
from ..engines.base import Engine


class EventAction(ABC):
    @abstractmethod
    def apply(self, engine: Engine) -> str:
        """Perform the change and return a log line."""


@dataclass
class DriveJoint(EventAction):
    """Move a carried object by driving its joint.

    This is how DI2's sliding bowl and DI3's pushed plate move: a
    position-actuated joint, **not** a mocap body, because mocap bodies take no
    force and so carry things inconsistently (details §5.1).
    """

    joint: str
    value: float

    def apply(self, engine: Engine) -> str:
        engine.set_articulation_target(self.joint, self.value)
        return f"drive_joint({self.joint}, {self.value:g})"


@dataclass
class SetJointDrive(EventAction):
    """Change a joint's spring and damping -- let a pendulum go."""

    joint: str
    stiffness: float
    damping: float

    def apply(self, engine: Engine) -> str:
        engine.set_joint_drive(self.joint, self.stiffness, self.damping)
        return f"set_joint_drive({self.joint}, {self.stiffness:g}, {self.damping:g})"


@dataclass
class SetInstruction(EventAction):
    """Rewrite the instruction mid-episode -- the DI4 'restatement' event."""

    text: str

    def apply(self, engine: Engine) -> str:
        engine.set_instruction(self.text)          # type: ignore[attr-defined]
        return f"set_instruction({self.text!r})"


@dataclass
class SetColor(EventAction):
    colored_body: str
    color: str

    def apply(self, engine: Engine) -> str:
        engine.set_color(self.colored_body, self.color)   # type: ignore[attr-defined]
        return f"set_color({self.colored_body}, {self.color})"


@dataclass
class Launch(EventAction):
    """Throw a free body: a toy's spring letting go of what it holds (CL6's
    catapult).  The flight is the engine's -- real on Isaac, a parabola on
    the mock."""

    body: str
    velocity: Tuple[float, float, float]

    def apply(self, engine: Engine) -> str:
        engine.launch(self.body, tuple(self.velocity))
        return f"launch({self.body}, {tuple(round(v, 2) for v in self.velocity)})"


@dataclass
class Teleport(EventAction):
    """Hard-set a pose.  Smoke tests only; the engine refuses when the body is
    grasped, and results from runs containing a Teleport are marked unfit for
    the main board."""

    body: str
    pose: Pose
    main_board_allowed: bool = False

    def apply(self, engine: Engine) -> str:
        engine.teleport(self.body, self.pose)
        return f"teleport({self.body})"
