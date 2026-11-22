"""The free form (RUI-Free): every action abstraction's tools side by side, a
program among them, the model choosing one each time.

The aids come with it, as on every column of the main table, and what the
aids leave out is left out here too: another way of moving the arm and asking
another model.  Each abstraction's tools behave as in its own interface: the
same step sizes, one action per arm per reply."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ruikit.interfaces.base import CONTROL, GRIPPER, SENSING, TASK, Interface

if TYPE_CHECKING:
    from ruikit.actions.base import ActionAbstraction


class Free(Interface):
    """The moves, the gripper they share, the acts, a program; then the aids,
    the task's own tools last."""

    name = "free"

    def tools(self, action: Optional[ActionAbstraction] = None) -> Tuple[str, ...]:
        from ruikit.actions import ACTIONS
        from ruikit.harness.components.aids import aid_tools

        def of(kind):
            return tuple(t for a in ACTIONS.values() if a.kind == kind for t in a.tools
                         if t != GRIPPER)

        aids = aid_tools()
        ending = tuple(t for t in aids if t.startswith("task_"))      # giving up
        return (SENSING + of("move") + (GRIPPER,) + of("act") + ("code_run",) + CONTROL
                + tuple(t for t in aids if t not in ending) + TASK + ending)

    def code_tools(self, action: Optional[ActionAbstraction] = None) -> Tuple[str, ...]:
        """A program may call everything else but giving up."""
        return tuple(t for t in self.tools(action) if t not in ("code_run", "task_give_up"))

    def params(self, action: Optional[ActionAbstraction] = None) -> Dict[str, Dict[str, Any]]:
        """Every abstraction's own settings, and the aids' in their default
        implementations."""
        from ruikit.actions import ACTIONS
        from ruikit.harness.components import COMPONENTS
        from ruikit.harness.components.aids import aid_components

        out: Dict[str, Dict[str, Any]] = {}
        for a in ACTIONS.values():
            for tool, settings in a.params.items():
                out.setdefault(tool, {}).update(settings)
        for name in aid_components():
            im = COMPONENTS[name].implementation(None)
            for tool, settings in (im.params if im is not None else {}).items():
                out.setdefault(tool, {}).update(settings)
        return out
