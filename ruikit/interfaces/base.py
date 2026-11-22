"""An interface: an interaction form applied to an action abstraction.

The form decides how the abstraction's tools reach the model -- one action a
reply (``per_action``), called from inside a program (``program``), or every
abstraction's side by side (``free``) -- and with it the tools offered, what a
program may call, the channel the model speaks in and when control comes back.
What results is a :class:`Profile`: the interface as a Harness hands it to the
model and as every episode log records it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple

from ruikit.core.tool import TRACKS, all_tools
from ruikit.perception.settings import PERCEPTION_DEFAULTS, parse_perception

if TYPE_CHECKING:
    from ruikit.actions.base import ActionAbstraction

#: what every interface offers, the same in all: the cameras, the robot's own
#: readings and the perception tools; waiting and stopping; the task
SENSING: Tuple[str, ...] = ("camera_look", "camera_zoom", "robot_state",
                            "vision_detect", "vision_segment", "vision_lift", "vision_pose")
CONTROL: Tuple[str, ...] = ("robot_wait", "robot_stop")
TASK: Tuple[str, ...] = ("task_info", "task_demo", "task_events", "task_done")
SHARED: Tuple[str, ...] = SENSING + CONTROL + TASK
#: the tool several action abstractions share
GRIPPER = "robot_gripper"

CHANNELS = ("tools", "json", "text", "code")
HAND_BACK = ("when_done", "each_step", "chunk_end", "program_end")
PERCEPTION = ("synthetic", "model", "privileged")


@dataclass(frozen=True)
class Profile:
    """An interface, resolved from its id: what the model may call and how.

    ``tools`` is what the model calls; ``code_tools`` what a program run by
    ``code_run`` may call.  ``params`` fixes settings of the tools (and, under
    ``perception``, of the perception backend).  ``as_dict`` is the header of
    every episode log."""
    id: str
    #: what the agent is told it is using
    title: str = ""
    track: str = "controlled"
    tools: Tuple[str, ...] = ()
    code_tools: Tuple[str, ...] = ()
    channel: str = "tools"
    hand_back: str = "when_done"
    #: the observation policy: what rides on the result of a reply that moved
    #: an arm (``ruikit.harness.components.observation``)
    push: Mapping[str, Any] = field(default_factory=dict)
    #: which variant of the tool descriptions
    docs: str = "default"
    perception: str = "model"
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: the action abstraction; ``program`` for a program over one, '' for an
    #: interface offering several
    dialect: str = ""
    #: the Harness components and extras attached, as written in the id ('marks:masks')
    plugins: Tuple[str, ...] = ()
    #: the published work it reproduces, and how it differs
    reproduces: str = ""
    differences: str = ""
    #: an operation guide the Harness adds to what the agent is told ('' none; ``+guide``)
    guide: str = ""

    def tool_params(self, name: str) -> Mapping[str, Any]:
        return dict(self.params.get(name, {}))

    def allows(self, name: str) -> bool:
        return name in self.tools or name in self.code_tools

    def callable_from_code(self, name: str) -> bool:
        return name in self.code_tools

    def as_dict(self) -> dict:
        return {"id": self.id, "title": self.title, "track": self.track,
                "tools": list(self.tools), "code_tools": list(self.code_tools),
                "channel": self.channel, "hand_back": self.hand_back, "push": dict(self.push),
                "docs": self.docs, "perception": self.perception,
                "params": {k: dict(v) for k, v in self.params.items()},
                "dialect": self.dialect, "plugins": list(self.plugins)}

    def perception_settings(self) -> dict:
        """The perception backend as configured: detector, depth, noise,
        calibration -- or, for the control condition, ``{"perception": "privileged"}``."""
        if self.perception == "privileged":
            return {"perception": "privileged"}
        return {**PERCEPTION_DEFAULTS, **dict(self.params.get("perception", {}))}


class Interface:
    """An interaction form: how an action abstraction's tools are handed to the model."""

    name = ""
    channel = "tools"

    def tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        """What the model calls."""
        raise NotImplementedError

    def code_tools(self, action: Optional[ActionAbstraction]) -> Tuple[str, ...]:
        """What a program may call (none, unless the form runs programs)."""
        return ()

    def hand_back(self, action: Optional[ActionAbstraction]) -> str:
        return "when_done"

    def dialect(self, action: Optional[ActionAbstraction]) -> str:
        return action.name if action is not None else ""

    def params(self, action: Optional[ActionAbstraction]) -> Dict[str, Dict[str, Any]]:
        return {k: dict(v) for k, v in action.params.items()} if action is not None else {}

    def profile(self, interface_id: str, action: Optional[ActionAbstraction],
                **described: str) -> Profile:
        """The interface over ``action``, named ``interface_id``; ``described``
        gives its title, what it reproduces and how it differs."""
        from ruikit.harness.components.observation import PUSH

        return Profile(id=interface_id, tools=self.tools(action),
                       code_tools=self.code_tools(action), channel=self.channel,
                       hand_back=self.hand_back(action), push=dict(PUSH),
                       params=self.params(action), dialect=self.dialect(action), **described)


def profile_problems(prof: Profile) -> List[str]:
    """What is wrong with an interface: unknown settings or tools, a tool not
    allowed on its track, a program that could call code_run."""
    from ruikit.actions import ACTIONS

    out = []
    if prof.track not in TRACKS:
        out.append(f"track must be one of {TRACKS}")
    if prof.channel not in CHANNELS:
        out.append(f"channel must be one of {CHANNELS}")
    if prof.hand_back not in HAND_BACK:
        out.append(f"hand_back must be one of {HAND_BACK}")
    if prof.perception not in PERCEPTION:
        out.append(f"perception must be one of {PERCEPTION}")
    dialects = tuple(ACTIONS) + ("program",)
    if prof.dialect and prof.dialect not in dialects:
        out.append(f"dialect must be one of {dialects}")
    try:
        parse_perception(",".join(f"{k}={v}" for k, v in prof.params.get("perception", {}).items()))
    except ValueError as exc:
        out.append(f"params.perception: {exc}")
    registry = all_tools()
    for name in prof.tools + prof.code_tools:
        t = registry.get(name)
        if t is None:
            out.append(f"unknown tool {name!r}")
        elif prof.track not in t.spec.tracks:
            out.append(f"tool {name!r} is not allowed on the {prof.track} track")
    if len(set(prof.tools)) != len(prof.tools):
        out.append("a tool is listed twice")
    if prof.channel == "code" and "code_run" not in prof.tools:
        out.append("a code-channel profile must offer code_run")
    if prof.code_tools and "code_run" not in prof.tools:
        out.append("code_tools without code_run")
    if "code_run" in prof.code_tools or any(n.startswith("toolkit_") for n in prof.code_tools):
        out.append("code_tools must not include code_run or toolkit_* (no recursion)")
    for name in prof.params:
        if name == "perception":                 # the perception backend's, not a tool's
            continue
        if name not in prof.tools and name not in prof.code_tools:
            out.append(f"params for {name!r}, which the profile does not offer")
    return out
