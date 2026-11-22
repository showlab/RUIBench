"""Make: an agent builds its own tools from the ones it has.

``toolkit_define`` turns a small program into a new tool, callable like any
other for the rest of the episode.  The rules:

* a custom tool can only call tools this interface already offers -- it
  composes, it gains nothing (object-capability: powers are handed down, never
  invented);
* it runs in the same sandbox as ``code_run``, each call a fresh namespace;
* it costs exactly its inner calls plus its compute time;
* its name starts with ``my_`` and cannot shadow a built-in; it cannot define
  further tools or run ``code_run``;
* it lives for this episode only;
* every definition and call is logged.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, all_tools, tool
from ruikit.harness.components.base import Component

CUSTOM_NAME = re.compile(r"^my_[a-z0-9_]{1,40}$")
MAX_CUSTOM_TOOLS = 20
_JSON_TYPES = {"string": str, "number": (int, float), "integer": int, "boolean": bool,
               "array": (list, tuple), "object": dict}


def _meta(name: str) -> bool:
    return name == "code_run" or name.startswith("toolkit_") or name.startswith("task_")


class CustomTool:
    """A tool an agent defined in this episode."""

    def __init__(self, name: str, description: str, schema: dict, code: str,
                 allowed: List[str], defined_at: float) -> None:
        self.name = name
        self.description = description
        self.schema = schema
        self.code = code
        self.allowed = allowed
        self.defined_at = defined_at
        self.calls = 0

    @property
    def fields(self) -> List[str]:
        return list((self.schema.get("properties") or {}).keys())

    def input_schema(self) -> dict:
        return dict(self.schema)

    def _check(self, args: Mapping[str, Any]) -> None:
        props = self.schema.get("properties") or {}
        unknown = sorted(set(args) - set(props))
        if unknown:
            raise ToolError("invalid_args", f"{self.name} takes no argument {unknown[0]!r}")
        for req in self.schema.get("required", []):
            if req not in args:
                raise ToolError("invalid_args", f"{self.name}: {req!r} is required")
        for key, value in args.items():
            kind = (props.get(key) or {}).get("type")
            if kind in _JSON_TYPES and not isinstance(value, _JSON_TYPES[kind]):
                raise ToolError("invalid_args", f"{self.name}: {key!r} should be a {kind}")

    def invoke(self, ctx, args: Optional[Mapping[str, Any]]) -> ToolResult:
        from ruikit.interfaces.sandbox.code_run import run_program

        args = dict(args or {})
        try:
            self._check(args)
        except ToolError as exc:
            return ToolResult.fail(exc)
        self.calls += 1
        out = run_program(ctx, self.code, self.allowed, custom_args=args)
        data = {"result": out.get("value"), "stdout": out.get("stdout", ""),
                "tool_calls": out.get("calls", 0), "compute_s": out.get("compute_s", 0.0),
                "isolation": out.get("isolation")}
        if out.get("error"):
            return ToolResult(data=data, text=f"{self.name} raised an error", error="tool_failed",
                              hint=out["error"].strip().splitlines()[-1][:300])
        return ToolResult(data=data, text=f"{self.name} finished")


@dataclass(frozen=True)
class ListParams:
    family: Optional[str] = param("only this family (observe, perceive, act, ...)", default=None)


@tool(name="toolkit_list", family="make", params=ListParams, source="agent",
      cost=Cost("declared", 0.0), examples=[{}, {"family": "perceive"}],
      description="""List the tools available in this configuration -- those you can call
      directly, those callable only from code_run, and the tools you defined yourself -- each
      with a one-line summary. Use it to discover capabilities before writing a program or a
      custom tool. Costs no simulated time.""")
def toolkit_list(ctx, p: ListParams) -> ToolResult:
    registry = all_tools()
    rows = []
    for where, names in (("direct", ctx.profile.tools), ("code", ctx.profile.code_tools)):
        for name in names:
            spec = registry[name].spec
            if p.family and spec.family != p.family:
                continue
            rows.append({"name": name, "where": where, "family": spec.family,
                         "summary": spec.description.split(". ")[0][:160]})
    for name, custom in ctx.custom_tools().items():
        rows.append({"name": name, "where": "custom", "family": "make",
                     "summary": custom.description[:160]})
    return ToolResult(data={"tools": rows}, text=f"{len(rows)} tools")


@dataclass(frozen=True)
class DefineParams:
    name: str = param("the new tool's name: my_ followed by lowercase letters, digits, _")
    description: str = param("what it does, its arguments and what it returns (a sentence or more)")
    input_schema: str = param(
        'JSON Schema of its arguments, as a JSON string: {"type": "object", "properties": '
        '{...}, "required": [...]}')
    code: str = param("Python defining run(**arguments); call tools as in code_run and return "
                      "a JSON-friendly value")


@tool(name="toolkit_define", family="make", params=DefineParams, source="agent",
      cost=Cost("declared", 0.0),
      examples=[{"name": "my_hover", "description": "Move the right gripper 10 cm above a point.",
                 "input_schema": '{"type": "object", "properties": {"xyz": {"type": "array"}}, '
                                 '"required": ["xyz"]}',
                 "code": "def run(xyz):\n    return robot.move_to(xyz_m=[xyz[0], xyz[1], "
                         "xyz[2] + 0.1])['data']"}],
      description="""Define a new tool for the rest of this episode from a Python run(**args)
      function that calls the tools this configuration offers (the same functions as in
      code_run). It can only combine existing tools -- it gets no new powers -- and it costs
      exactly what its inner calls cost plus its compute time. The name must start with 'my_';
      the input_schema is a JSON Schema object. Once defined, call it by name like any other
      tool. At most 20 custom tools; they are forgotten when the episode ends.""")
def toolkit_define(ctx, p: DefineParams) -> ToolResult:
    if not CUSTOM_NAME.match(p.name):
        raise ToolError("invalid_args", "the name must match my_[a-z0-9_]{1,40}")
    if p.name in all_tools():
        raise ToolError("invalid_args", f"{p.name!r} is a built-in tool")
    custom = ctx.custom_tools()
    if p.name not in custom and len(custom) >= MAX_CUSTOM_TOOLS:
        raise ToolError("invalid_args", f"at most {MAX_CUSTOM_TOOLS} custom tools")
    if len(p.description.strip()) < 20:
        raise ToolError("invalid_args", "describe the tool in at least a sentence")
    try:
        schema = json.loads(p.input_schema)
    except json.JSONDecodeError as exc:
        raise ToolError("invalid_args", f"input_schema is not valid JSON: {exc.msg}")
    if not isinstance(schema, dict) or schema.get("type") != "object" \
            or not isinstance(schema.get("properties", {}), dict):
        raise ToolError("invalid_args", "input_schema must be {'type': 'object', 'properties': {...}}")
    try:
        tree = compile(p.code, f"<{p.name}>", "exec")
    except SyntaxError as exc:
        raise ToolError("invalid_args", f"syntax error on line {exc.lineno}: {exc.msg}")
    if "run" not in tree.co_names and "def run" not in p.code:
        raise ToolError("invalid_args", "the code must define run(**arguments)")
    allowed = [n for n in list(ctx.profile.tools) + list(ctx.profile.code_tools) if not _meta(n)]
    custom[p.name] = CustomTool(p.name, " ".join(p.description.split()), schema, p.code,
                                allowed, ctx.sim_time)
    return ToolResult(data={"name": p.name, "callable_tools": allowed},
                      text=f"defined {p.name}; call it like any other tool")


@dataclass(frozen=True)
class RemoveParams:
    name: str = param("a tool you defined")


@tool(name="toolkit_remove", family="make", params=RemoveParams, source="agent",
      cost=Cost("declared", 0.0), examples=[{"name": "my_hover"}],
      description="""Remove a tool you defined earlier in this episode with toolkit_define, for
      instance to redefine it differently. Built-in tools cannot be removed, and removing a tool
      does not undo anything it already did. Costs no simulated time and does not touch the
      robot.""")
def toolkit_remove(ctx, p: RemoveParams) -> ToolResult:
    if ctx.custom_tools().pop(p.name, None) is None:
        raise ToolError("invalid_args", f"no custom tool {p.name!r}")
    return ToolResult(data={"name": p.name}, text=f"removed {p.name}")


COMPONENT = Component("make", "自定义工具", "new tools made of the others, for this episode",
                      ("toolkit_define", "toolkit_list", "toolkit_remove"))
