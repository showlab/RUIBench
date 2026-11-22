"""The program form: the model writes a program, and the program calls the
action tools (CodeAct, Code as Policies, CaP-X).

A program over an abstraction may call exactly what that abstraction's
per-action interface offers, so the two differ only in how the model speaks --
a program instead of a call -- and in when control comes back: when the
program ends, after one call or a hundred.  ``code_run`` runs the programs
(``ruikit.interfaces.sandbox.code_run``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Tuple

from ruikit.interfaces.base import TASK, Interface
from ruikit.interfaces.per_action import PerAction

if TYPE_CHECKING:
    from ruikit.actions.base import ActionAbstraction


class Program(Interface):
    """``code_run`` and the task's tools; the program calls the rest."""

    name = "program"
    channel = "code"

    def tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        return ("code_run",) + TASK

    def code_tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        return PerAction().tools(action)

    def hand_back(self, action: Optional[ActionAbstraction]) -> str:
        return "program_end"

    def dialect(self, action: Optional[ActionAbstraction]) -> str:
        return "program"
