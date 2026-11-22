"""The tool contract and the registry.

One definition per tool: a function ``fn(ctx, params) -> ToolResult`` plus a
:class:`ToolSpec`.  The spec carries what a model reads (name, description,
parameter schema, examples) and what the gate enforces (which arm resource it
writes, which sensors it reads, where its information comes from, which tracks
may use it, how its time is charged).

**Registered is not enabled.**  Every tool known to the library is in
:data:`REGISTRY`; an agent sees only the tools its interface lists
(``ruikit.interfaces``).  Adding a file never changes an existing experiment.

Writing a tool::

    @dataclass(frozen=True)
    class LookParams:
        cameras: Optional[List[str]] = param("cameras to capture; default: all", default=None)

    @tool(name="camera_look", family="observe", params=LookParams,
          description="...", source="sensor", reads=("cameras",))
    def camera_look(ctx, p: LookParams) -> ToolResult: ...

Tools from other packages register the same way and are discovered through
the ``ruikit.tools`` entry-point group (:func:`load_plugins`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, is_dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from ruikit.core.params import ArgError, NoParams, describe_params, json_schema, parse_args
from ruikit.core.result import ToolError, ToolResult

FAMILIES: Tuple[str, ...] = ("observe", "perceive", "act", "delegate", "compute",
                             "code", "task", "memory", "make")
#: one character per family, for the tables
FAMILY_GLYPH: Mapping[str, str] = dict(zip(FAMILIES, "看认动托算写问记造"))
CONTROLS: Tuple[str, ...] = ("none", "arm", "both_arms")
#: where a tool's information comes from.  ``vision``: the cameras' pictures and
#: depth, read by the perception backend (every vision_* tool, and the planners
#: that work on what it measured); ``model``: an outside model the agent consults
#: (a VLM, a VLA, Jev) -- an extra, never in a base interface; ``synthetic``: the
#: mock engine's test double
SOURCES: Tuple[str, ...] = ("sensor", "vision", "synthetic", "model", "controller", "task",
                            "agent", "oracle")
RETURNS: Tuple[str, ...] = ("result", "handle", "result_or_handle")
TRACKS: Tuple[str, ...] = ("controlled", "open", "diagnostic")
NAME_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
#: descriptions shorter than this read like "Gets the stock price" -- the
#: example of a *poor* description in Anthropic's tool guide
MIN_DESCRIPTION_CHARS = 120


@dataclass(frozen=True)
class Cost:
    """How a call is charged.

    ``physical``: simulated time is what the motion actually takes.
    ``declared``: a fixed number of simulated seconds per call (a perception
    query, a render).  ``measured``: the call's real wall-clock time is charged
    (an external model API), and so is ``usd`` when the backend reports it.
    """

    kind: str = "declared"
    seconds: float = 0.0
    usd: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in ("physical", "declared", "measured"):
            raise ValueError(f"unknown cost kind {self.kind!r}")


#: what the result of a motion holds -- the same for every tool that moves an arm
MOTION_DATA = ("handle, status ('done', 'failed', 'cancelled' or 'running'), tcp {xyz_m, rpy_deg}, "
               "gripper_closed, holding_something, residual_cm and residual_deg (how far from "
               "its target it ended), and short_by_m [dx, dy, dz] (target minus tcp) when it "
               "stopped short; with wait=false only handle and status")


@dataclass(frozen=True)
class ToolSpec:
    name: str
    family: str
    description: str
    params: type = NoParams
    examples: Tuple[Mapping[str, Any], ...] = ()
    controls: str = "none"
    reads: Tuple[str, ...] = ()
    source: str = "sensor"
    returns: str = "result"
    cost: Cost = field(default_factory=Cost)
    tracks: Tuple[str, ...] = ("controlled", "open")
    version: str = "1"
    #: shown only in code mode: the one-line signature summary
    code_name: str = ""
    #: what the result's ``data`` holds, key by key, as a program reads it:
    #: shown with the function in code mode (a tool-calling agent sees the
    #: result itself).  Without it, a program had to print a result before it
    #: could use one
    data: str = ""

    def __post_init__(self) -> None:
        problems = spec_problems(self)
        if problems:
            raise ValueError(f"tool {self.name!r}: " + "; ".join(problems))
        if not self.code_name:
            family, _, rest = self.name.partition("_")
            object.__setattr__(self, "code_name", f"{family}.{rest}" if rest else self.name)

    def input_schema(self, strict: bool = False, hide: Tuple[str, ...] = ()) -> dict:
        return json_schema(self.params, strict=strict, hide=hide)

    def signature_doc(self) -> str:
        return describe_params(self.params)


def spec_problems(spec: ToolSpec) -> list:
    """Everything wrong with a spec, as messages (empty: fine).  Used when a
    tool is registered and again by ``ruikit check``."""
    out = []
    if not NAME_RE.match(spec.name):
        out.append(f"name must match {NAME_RE.pattern} (providers reject dots)")
    if spec.family not in FAMILIES:
        out.append(f"family must be one of {FAMILIES}")
    if spec.controls not in CONTROLS:
        out.append(f"controls must be one of {CONTROLS}")
    if spec.source not in SOURCES:
        out.append(f"source must be one of {SOURCES}")
    if spec.returns not in RETURNS:
        out.append(f"returns must be one of {RETURNS}")
    if not spec.tracks or any(t not in TRACKS for t in spec.tracks):
        out.append(f"tracks must be a non-empty subset of {TRACKS}")
    if spec.source == "oracle" and set(spec.tracks) - {"diagnostic"}:
        out.append("an oracle tool may only be offered on the diagnostic track")
    if len(spec.description.strip()) < MIN_DESCRIPTION_CHARS:
        out.append(f"description needs at least {MIN_DESCRIPTION_CHARS} characters: say units, "
                   "frame, whether it blocks, and what it does not do")
    if not is_dataclass(spec.params):
        out.append("params must be a dataclass")
    else:
        try:
            json_schema(spec.params)
            json_schema(spec.params, strict=True)
        except TypeError as exc:
            out.append(f"params: {exc}")
        for i, example in enumerate(spec.examples):
            try:
                parse_args(spec.params, example)
            except ArgError as exc:
                out.append(f"example {i} does not validate: {exc}")
    if spec.family in ("observe", "perceive", "task", "memory", "compute") \
            and spec.controls != "none":
        out.append(f"a {spec.family} tool must not control the robot")
    return out


@dataclass(frozen=True)
class Tool:
    spec: ToolSpec
    fn: Callable[[Any, Any], ToolResult]

    @property
    def name(self) -> str:
        return self.spec.name

    def invoke(self, ctx, raw_args: Optional[Mapping[str, Any]]) -> ToolResult:
        """Validate, run, and turn a :class:`ToolError` into an error result."""
        try:
            params = parse_args(self.spec.params, raw_args)
        except ArgError as exc:
            return ToolResult.fail(ToolError("invalid_args", str(exc),
                                             hint="check the tool's parameter schema"))
        try:
            out = self.fn(ctx, params)
        except ToolError as exc:
            return ToolResult.fail(exc)
        if not isinstance(out, ToolResult):
            raise TypeError(f"tool {self.name} returned {type(out).__name__}, not ToolResult")
        return out


#: every tool the library knows, by name -- registered, not enabled
REGISTRY: Dict[str, Tool] = {}


def register(t: Tool, replace: bool = False) -> Tool:
    if t.name in REGISTRY and not replace and REGISTRY[t.name].fn is not t.fn:
        raise ValueError(f"tool {t.name!r} is already registered")
    REGISTRY[t.name] = t
    return t


def tool(*, name: str, family: str, description: str, params: type = NoParams,
         examples: Iterable[Mapping[str, Any]] = (), controls: str = "none",
         reads: Iterable[str] = (), source: str = "sensor", returns: str = "result",
         cost: Optional[Cost] = None, tracks: Iterable[str] = ("controlled", "open"),
         version: str = "1", data: str = ""):
    """Register ``fn(ctx, params) -> ToolResult`` as a tool."""
    spec = ToolSpec(name=name, family=family, description=" ".join(description.split()),
                    params=params, examples=tuple(examples), controls=controls,
                    reads=tuple(reads), source=source, returns=returns,
                    cost=cost or Cost(), tracks=tuple(tracks), version=version,
                    data=" ".join(data.split()))

    def wrap(fn):
        register(Tool(spec, fn))
        fn.tool_spec = spec
        return fn

    return wrap


def get_tool(name: str) -> Tool:
    load_builtin_tools()
    try:
        return REGISTRY[name]
    except KeyError:
        raise KeyError(f"no tool named {name!r}; known: {sorted(REGISTRY)}") from None


def all_tools() -> Dict[str, Tool]:
    load_builtin_tools()
    return dict(sorted(REGISTRY.items()))


_BUILTINS_LOADED = False


#: the modules that hold the built-in tools: importing one registers its tools
BUILTIN_TOOL_MODULES: Tuple[str, ...] = (
    "ruikit.perception.camera", "ruikit.perception.proprioception", "ruikit.perception.vision",
    "ruikit.harness.components.pointer",
    "ruikit.harness.components.track", "ruikit.harness.components.marks", "ruikit.actions.common",
    "ruikit.actions.pose", "ruikit.actions.delta", "ruikit.actions.step", "ruikit.actions.point",
    "ruikit.actions.joint", "ruikit.extras.motion",
    "ruikit.actions.skill", "ruikit.extras.policy", "ruikit.harness.components.planning.reach",
    "ruikit.harness.components.planning.grasps", "ruikit.harness.components.planning.paths",
    "ruikit.extras.models", "ruikit.interfaces.sandbox.code_run",
    "ruikit.env.task", "ruikit.harness.components.memory", "ruikit.extras.make")


def load_builtin_tools() -> None:
    """Import the modules of the built-in tools (idempotent)."""
    global _BUILTINS_LOADED
    if _BUILTINS_LOADED:
        return
    _BUILTINS_LOADED = True
    import importlib

    for name in BUILTIN_TOOL_MODULES:
        importlib.import_module(name)


def load_plugins(group: str = "ruikit.tools") -> list:
    """Import tool plugins advertised by installed packages.  Discovery only
    imports them; an interface still has to list a tool before anyone sees it."""
    from importlib.metadata import entry_points

    loaded = []
    for ep in entry_points().select(group=group) if hasattr(entry_points(), "select") \
            else entry_points().get(group, ()):
        ep.load()
        loaded.append(ep.name)
    return loaded
