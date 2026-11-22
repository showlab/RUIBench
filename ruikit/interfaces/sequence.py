"""The sequence form: the model writes a program that only acts -- a series of
actions run one after another, with no look and no result in between -- and
looks with the perception tools between replies, as in the per-action form.

It sits between the other two forms.  Like a program, one reply sends many
actions; like the per-action form, every look, and every decision on what was
seen, is a reply of the model's own.  The three side by side separate what a
program gains by deciding less often from what it gains by reading the world
inside the program and correcting itself.  ``code_run`` runs such a program
blind (its ``blind`` setting): each call starts its action and hands the
program nothing back, and the actions' reports come with the result once the
program ends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ruikit.interfaces.base import CONTROL, SENSING, TASK, Interface

if TYPE_CHECKING:
    from ruikit.actions.base import ActionAbstraction


def _acting(action: ActionAbstraction) -> Tuple[str, ...]:
    """The abstraction's tools that move an arm or its gripper."""
    from ruikit.core.tool import get_tool

    return tuple(t for t in action.tools if get_tool(t).spec.controls == "arm")


class Sequence(Interface):
    """The shared tools and ``code_run``, whose programs call only the
    abstraction's actions, waiting and stopping; the abstraction's other tools
    (``robot_solve_ik``) are called as tools."""

    name = "sequence"

    def tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        acting = _acting(action)
        rest = tuple(t for t in action.tools if t not in acting)
        return SENSING + rest + ("code_run",) + CONTROL + TASK

    def code_tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        return _acting(action) + CONTROL

    def hand_back(self, action: Optional[ActionAbstraction]) -> str:
        return "program_end"

    def dialect(self, action: Optional[ActionAbstraction]) -> str:
        return "program"

    def params(self, action: Optional[ActionAbstraction]) -> Dict[str, Dict[str, Any]]:
        out = super().params(action)
        out.setdefault("code_run", {})["blind"] = True
        return out
