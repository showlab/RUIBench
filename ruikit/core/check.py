"""``ruikit check``: the contract checks a tool must pass.

Static, no simulator needed:

* the spec itself (``spec_problems``): provider-legal name, a real
  description, declared source / controls / tracks, strict-compatible schema,
  examples that validate;
* **the module never reaches past its ToolContext**: no import of the task
  cards, scorer, runner, event scripts, executor or the benchmark's own
  perception services, and no use of the context's private episode handle.
  A tool that needs something new asks for a new door on the context, in
  review, instead of quietly taking the simulator.

Runtime checks (same frame -> same answer, stop means stopped, not-visible
means not reported) live in ``tests/ruikit``.
"""

from __future__ import annotations

import ast
import inspect
from typing import Dict, List, Optional

from ruikit.core.tool import Tool, all_tools, spec_problems

#: a tool module may not import these (the world's private parts)
FORBIDDEN_IMPORTS = ("ruibench.tasks", "ruibench.scoring", "ruibench.runner",
                     "ruibench.events", "ruibench.executor",
                     "ruibench.policies", "ruibench.engines.isaac", "ruibench.engines.mock")
#: modules allowed to touch the episode's private parts: sessions ticked by it
PRIVILEGED_MODULES = ("ruikit.extras.policy",)


def module_problems(module) -> List[str]:
    out = []
    try:
        source = inspect.getsource(module)
    except (OSError, TypeError):
        return out
    tree = ast.parse(source)
    for node in ast.walk(tree):
        names = []
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = [node.module]
        for n in names:
            if n.startswith(FORBIDDEN_IMPORTS):
                out.append(f"{module.__name__} imports {n} (world internals)")
        if isinstance(node, ast.Attribute) and node.attr == "_ep" \
                and module.__name__ not in PRIVILEGED_MODULES:
            out.append(f"{module.__name__} line {node.lineno}: uses ctx._ep (private)")
    return out


def check_tool(t: Tool) -> List[str]:
    problems = spec_problems(t.spec)
    mod = inspect.getmodule(t.fn)
    if mod is not None:
        problems += module_problems(mod)
    return problems


def check_all(names: Optional[List[str]] = None) -> Dict[str, List[str]]:
    tools = all_tools()
    out = {}
    seen_modules = set()
    for name, t in tools.items():
        if names and name not in names:
            continue
        problems = spec_problems(t.spec)
        mod = inspect.getmodule(t.fn)
        if mod is not None and mod.__name__ not in seen_modules:
            seen_modules.add(mod.__name__)
            problems += module_problems(mod)
        out[name] = problems
    # every tool has one place in the design space (``ruikit.interfaces.roles``)
    from ruikit.interfaces.roles import layout_problems

    for problem in layout_problems(all_tools()):
        name = problem.split(" ", 1)[0]
        if names is None or name in names:
            out.setdefault(name, []).append(problem)
    return out
