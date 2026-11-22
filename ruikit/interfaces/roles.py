"""Where each tool belongs in the design space: shared by every interface, an
action abstraction's, the program form's own, a Harness component's, or an
extra outside the design space."""

from __future__ import annotations

from typing import Any, List, Mapping, Tuple

from ruikit.interfaces.base import SHARED

#: (English, Chinese), in the order the listings use
ROLES: Mapping[str, Tuple[str, str]] = {
    "shared": ("shared by every interface", "共享工具"),
    "action": ("action abstractions", "动作抽象"),
    "form": ("interaction forms", "交互形式"),
    "component": ("Harness components", "Harness 组件"),
    "extra": ("extras, outside the design space", "设计空间之外"),
}

#: the tools an interaction form brings itself
FORM_TOOLS: Mapping[str, Tuple[str, ...]] = {"program": ("code_run",)}


def _homes() -> Mapping[str, Mapping[str, Tuple[str, ...]]]:
    """{role: {name: tools}} for every role but the shared tools."""
    from ruikit.actions import ACTIONS
    from ruikit.extras import EXTRAS
    from ruikit.harness.components import COMPONENTS

    return {"action": {a.name: a.tools for a in ACTIONS.values()},
            "form": dict(FORM_TOOLS),
            "component": {c.name: c.tools for c in COMPONENTS.values()},
            "extra": {e.name: e.tools for e in EXTRAS.values()}}


def role_of(tool_name: str) -> str:
    """'shared', 'action', 'form', 'component' or 'extra' -- '' for a tool in none."""
    if tool_name in SHARED:
        return "shared"
    for role, homes in _homes().items():
        if any(tool_name in tools for tools in homes.values()):
            return role
    return ""


def where(tool_name: str) -> str:
    """The abstractions, form, component or extra a tool belongs to ('' for a shared one)."""
    return ", ".join(name for homes in _homes().values() for name, tools in homes.items()
                     if tool_name in tools)


def layout_problems(registry: Mapping[str, Any]) -> List[str]:
    """Every registered tool has exactly one role: shared, or in action
    abstractions (several may share one: robot_gripper), or in one form,
    component or extra -- and every tool placed is registered."""
    homes = _homes()
    out = []
    for name in registry:
        roles = [role for role, by_name in homes.items()
                 for tools in by_name.values() if name in tools]
        if name in SHARED:
            roles.append("shared")
        if not roles:
            out.append(f"{name} has no place in the design space")
        elif len(set(roles)) > 1:
            out.append(f"{name} is in more than one role: {', '.join(sorted(set(roles)))}")
        elif roles[0] != "action" and len(roles) > 1:
            out.append(f"{name} is in two {ROLES[roles[0]][0]}")
    placed = list(SHARED) + [t for by_name in homes.values() for tools in by_name.values()
                             for t in tools]
    for name in dict.fromkeys(placed):
        if name not in registry:
            out.append(f"{name} is placed but not registered")
    return out
