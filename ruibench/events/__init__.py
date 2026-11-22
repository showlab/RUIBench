"""Event scripts: move an object, rewrite the instruction, change a colour --
once, at a milestone (events), or every time a toy is played with (reactors)."""

from .actions import DriveJoint, EventAction, SetColor, SetInstruction, Teleport
from .feedback import CORRECT, WRONG, FeedbackChannel, FeedbackMessage
from .reactor import Reactor
from .scheduler import Event, EventRecord, EventScheduler
from .triggers import (
    AllOf, MilestoneTrigger, ModelCallTrigger, PredicateTrigger, SimTimeTrigger,
    Trigger, TriggerContext,
)

__all__ = [
    "Event", "EventRecord", "EventScheduler", "EventAction", "DriveJoint",
    "SetInstruction", "SetColor", "Teleport", "Trigger", "TriggerContext",
    "SimTimeTrigger", "MilestoneTrigger", "PredicateTrigger", "ModelCallTrigger", "AllOf",
    "FeedbackChannel", "FeedbackMessage", "CORRECT", "WRONG", "Reactor",
]
