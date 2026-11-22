"""The per-action form: the model calls the action tools itself, one action per
arm per reply, and control comes back after each."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ruikit.interfaces.base import CONTROL, SENSING, TASK, Interface

if TYPE_CHECKING:
    from ruikit.actions.base import ActionAbstraction


class PerAction(Interface):
    """The shared tools with one action abstraction's: perceive, act, wait or
    stop, read and end the task."""

    name = "per_action"

    def tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        return SENSING + action.tools + CONTROL + TASK

    def hand_back(self, action: Optional[ActionAbstraction]) -> str:
        return action.hand_back


class PerActionRule:
    """The per-action form at run time: one action per arm per reply, and nothing after a wait."""

    def _sequenced(self, calls) -> str:
        """Why a reply of a tool-calling interface is not one step of it, or "".

        Such an interface hands control back after each action: one action per
        arm per reply (two arms may act together).  robot_wait inside a reply
        blocks until it is done, so [step, wait, step] would chain several
        actions into one reply -- an agent on the step interface (``rui.step/v1``)
        did exactly that on WK5, ten steps a reply -- which is what the program
        form is for."""
        if self.profile.channel != "tools":
            return ""
        acted: Dict[str, int] = {}
        waited = False
        for name, args in calls:
            spec = self._spec_of(name)
            if spec is None:
                continue
            if name == "robot_wait":
                waited = True
            elif spec.controls == "arm":
                if waited:
                    return f"{name} comes after robot_wait in the same reply"
                arm = str((args or {}).get("arm") or "default")
                acted[arm] = acted.get(arm, 0) + 1
                if acted[arm] > 1:
                    return (f"this reply starts {acted[arm]} actions on the {arm} arm; "
                            "this interface takes one action per arm per reply")
        return ""
