"""The four Robot-Use Interfaces, plus the harness test double.

    >>> from ruibench.interfaces import make_interface, INTERFACES
    >>> INTERFACES.names()
    ['I0_direct', 'I1', 'I2', 'I3', 'I4']

I1-I4 are deliberately unimplemented here: their action languages are written
separately.  Their **contracts** are filled in, which is what the rest of the
harness, the validator and the appendix table read.
"""

from __future__ import annotations

from ..core.registry import Registry
from .base import (
    Expressivity, Interface, InterfaceContract, ModelTurn, NotImplementedInterface,
    ParsedAction, TurnContext,
)
from .commands import (
    Cancel, Command, Done, MoveDelta, MoveTo, SetGripper, Wait, describe,
)
from .direct import DirectInterface, encode
from .pointing import PointingInterface
from .pose import PoseInterface
from .program import ProgramInterface
from .semantic_step import SemanticStepInterface

INTERFACES: Registry = Registry("interface")
for _cls in (DirectInterface, SemanticStepInterface, PoseInterface,
             PointingInterface, ProgramInterface):
    INTERFACES.register(_cls.id, _cls)

#: The four interfaces under study, in paper order.
BENCHMARK_INTERFACES = ("I1", "I2", "I3", "I4")


def make_interface(name: str, **kwargs) -> Interface:
    return INTERFACES.get(name)(**kwargs)


def contract_table() -> dict:
    """Every interface's contract row -- the appendix table, as data."""
    return {name: INTERFACES.get(name)().contract.as_row() for name in INTERFACES.names()}


__all__ = [
    "INTERFACES", "BENCHMARK_INTERFACES", "make_interface", "contract_table",
    "Interface", "InterfaceContract", "Expressivity", "NotImplementedInterface",
    "ModelTurn", "ParsedAction", "TurnContext",
    "Command", "MoveTo", "MoveDelta", "SetGripper", "Wait", "Cancel",
    "Done", "describe", "encode",
    "DirectInterface", "SemanticStepInterface", "PoseInterface",
    "PointingInterface", "ProgramInterface",
]
