"""A profile's tools in each provider's native format.

Same tools, three spellings: Anthropic (``input_schema``), OpenAI Responses
(``parameters``, ``strict``) and Gemini (``parameters``), plus MCP's own
(``inputSchema``).  Names are already legal for every provider (no dots), and
the order is fixed so prompt caches keep hitting.

Strict mode (every property required, optional ones nullable, no extra
properties) is what OpenAI's and Anthropic's strict tool use want; Claude
Opus 5.5 does not accept forced ``tool_choice``, so a loop that needs a tool
call uses ``auto`` with strict tools instead.
"""

from __future__ import annotations

import re
from typing import List, Optional

from ruikit.core.tool import MOTION_DATA, get_tool
from ruikit.interfaces import Profile, load_profile


def _variant(description: str, docs: str) -> str:
    return description


#: what an aid's description says of a move the interface may not offer, and
#: what it says instead where it does not (the aids go on every interface)
ELSEWHERE = (
    ("plan_motion", "Returns up to 12 waypoints to send with robot_waypoints; it does not "
     "move the robot.", "robot_waypoints",
     "Returns up to 12 waypoints; it does not move the robot -- follow them with this "
     "interface's own moves."),
    ("plan_grasps", "execute a candidate with robot_move_to and robot_gripper.",
     "robot_move_to", "carry a candidate out with this interface's own moves."),
    ("vision_point", "Use it to get a pixel for robot_move_to_pixel or vision_lift when",
     "robot_move_to_pixel", "Use it to get a pixel for vision_lift when"),
)


#: what the perception tools say of their errors, and what they say instead
#: under the privileged control (``@privileged``): exact, as they answer
PRIVILEGED = (
    ("vision_detect", "-- the box is a detector's, a few pixels off around where the robot's "
     "estimate puts the object.", "-- the box is where the object is in the picture (this "
     "interface's perception is privileged: the simulator's own values)."),
    ("vision_lift", "It is a measurement, not the truth: it meets boxes that stand in for each "
     "thing where the robot's estimate puts it (one box for a doll, several for a board with "
     "holes or a tray with walls), adds about 1 mm + 0.3 % of the distance of depth noise, and "
     "places the point with the camera's calibration, which is off by about 3 mm and 0.3 "
     "degrees (5 mm more a metre away).", "This interface's perception is privileged: the ray "
     "meets boxes that stand in for each thing where it is (one box for a doll, several for a "
     "board with holes or a tray with walls), with no depth noise and the camera's exact pose."),
    ("vision_pose", "It is the robot's estimate, not the truth: each object is off by its own "
     "fixed error (about 5 mm, sometimes 1-1.5 cm; 3 degrees; 5 % in size) that every "
     "perception tool shares, plus a couple of millimetres that change with the viewpoint and "
     "the camera's calibration error.", "This interface's perception is privileged: these are "
     "the simulator's own values, exact."),
)


def _as_configured(name: str, description: str, profile: Profile) -> str:
    """A tool's description with this profile's own settings in it, naming no
    tool the profile does not offer."""
    text = _configured(name, description, profile)
    for tool_name, said, needs, instead in ELSEWHERE:
        if name == tool_name and not profile.allows(needs):
            text = text.replace(said, instead)
    if profile.perception == "privileged":
        for tool_name, said, instead in PRIVILEGED:
            if name == tool_name:
                text = text.replace(said, instead)
    return text


def _configured(name: str, description: str, profile: Profile) -> str:
    """A tool's description with this profile's own settings in it.  Without them, an
    agent on the step interface was told 0.5 / 2 / 4 cm steps where its steps were
    0.25 / 1 / 4, and one on the program interface knew its functions only by example."""
    if name == "robot_step":
        from ruikit.actions.step import STEP_DEFAULTS

        s = {**STEP_DEFAULTS, **profile.tool_params("robot_step")}
        cm = " / ".join(f"{100 * s[k + '_m']:g}" for k in ("micro", "fine", "coarse"))
        deg = " / ".join(f"{s[k + '_deg']:g}" for k in ("micro", "fine", "coarse"))
        return re.sub(r"\(by default [^)]*\)",
                      f"(in this interface {cm} cm for micro / fine / coarse, and {deg} degrees "
                      "for rotate and tilt)", description)
    if name == "code_run" and profile.code_tools:
        if profile.tool_params("code_run").get("blind"):
            from ruikit.interfaces.sandbox.code_run import BLIND_DESCRIPTION

            return BLIND_DESCRIPTION + "\n\n" + code_reference(profile, blind=True)
        return description + "\n\n" + code_reference(profile)
    impl = profile.tool_params(name).get("impl")
    from ruikit.harness.components import BY_IMPLEMENTATION

    if (name, impl) in BY_IMPLEMENTATION:              # a component's other implementation
        return BY_IMPLEMENTATION[(name, impl)]
    if name == "vision_marks" and profile.tool_params(name).get("style") in ("masks", "boxes"):
        style = profile.tool_params(name)["style"]
        return description.replace("Draw numbered marks on a copy of a frame",
                                   "Draw numbered marks on a copy of a frame -- each thing's "
                                   + ("outline" if style == "masks" else "box")
                                   + " in a colour of its own, its number in the middle")
    if name == "model_decide" and profile.tool_params(name).get("backend", "jev") != "jev":
        from ruikit.extras.models import DECIDE_BY_MODEL

        return DECIDE_BY_MODEL
    if profile.perception == "model":
        from ruikit.perception.service.client import SCORES
        from ruikit.perception.vision import FROM_PIXELS, from_pixels_described

        settings = profile.perception_settings()
        text = FROM_PIXELS.get(name)                    # the same tool, answered from pixels
        if text is None:
            return description
        low, doubtful = SCORES.get(settings["backend"], SCORES["sam3"])
        text = text.replace("{low}", f"{low:g}").replace("{doubtful}", f"{doubtful:g}")
        return from_pixels_described(name, text, settings)
    return description


def code_reference(profile: Profile, blind: bool = False) -> str:
    """The functions a program can call, documented as fully as the tools are for
    the tool-calling interfaces -- the same tools, so the same words; ``blind``:
    what they return left out, for they return nothing."""
    if blind:
        lines = ["The functions a program can call. Each is one of the tools, with the same "
                 "parameters and rules; it returns nothing (an empty dict) and raises nothing "
                 "when its action fails. Arguments go by name or in the order shown:"]
    else:
        lines = ["The functions a program can call. Each is one of the tools, with the same "
                 "parameters, rules and result (a dict: ok, text, data, ...; a failed call "
                 "raises ToolCallError). Arguments go by name or in the order shown; '-> data' "
                 "lists what the result's data holds:"]
    if not blind and any(MOTION_DATA in get_tool(name).spec.data for name in profile.code_tools):
        lines.append("A motion's result -- that of every function that moves an arm -- holds as "
                     "data: " + MOTION_DATA + ".")
    for name in profile.code_tools:
        spec = get_tool(name).spec
        family, _, rest = name.partition("_")
        fields = [line.split(":", 1)[0] for line in spec.signature_doc().splitlines() if line]
        lines.append(f"\n{family}.{rest or name}({', '.join(fields)})")
        lines.append("    " + " ".join(_as_configured(name, spec.description, profile).split()))
        for line in spec.signature_doc().splitlines():
            lines.append("    - " + line)
        if spec.data and not blind:
            lines.append("    -> data: " + spec.data.replace(MOTION_DATA, "a motion's result (above)"))
    return "\n".join(lines)


def tool_entries(profile: Profile, strict: bool = False) -> List[dict]:
    out = []
    for name in profile.tools:
        spec = get_tool(name).spec
        out.append({"name": name,
                    "description": _as_configured(name, _variant(spec.description, profile.docs),
                                                  profile),
                    "schema": spec.input_schema(strict=strict),
                    "examples": [dict(e) for e in spec.examples] if profile.docs != "no_examples"
                    else []})
    return out


def anthropic_tools(profile: Profile | str, strict: bool = False,
                    with_examples: bool = True) -> List[dict]:
    prof = load_profile(profile) if isinstance(profile, str) else profile
    tools = []
    for e in tool_entries(prof, strict):
        t = {"name": e["name"], "description": e["description"], "input_schema": e["schema"]}
        if strict:
            t["strict"] = True
        if with_examples and e["examples"]:
            t["input_examples"] = e["examples"]
        tools.append(t)
    return tools


def openai_tools(profile: Profile | str, strict: bool = True) -> List[dict]:
    prof = load_profile(profile) if isinstance(profile, str) else profile
    return [{"type": "function", "name": e["name"], "description": e["description"],
             "parameters": e["schema"], "strict": strict} for e in tool_entries(prof, strict)]


def gemini_tools(profile: Profile | str) -> List[dict]:
    prof = load_profile(profile) if isinstance(profile, str) else profile
    decls = []
    for e in tool_entries(prof, strict=False):
        decls.append({"name": e["name"], "description": e["description"],
                      "parameters": _gemini_schema(e["schema"])})
    return [{"function_declarations": decls}]


def _gemini_schema(schema: dict) -> dict:
    """Gemini's function schema has no additionalProperties; drop it."""
    if not isinstance(schema, dict):
        return schema
    out = {k: v for k, v in schema.items() if k != "additionalProperties"}
    if "properties" in out:
        out["properties"] = {k: _gemini_schema(v) for k, v in out["properties"].items()}
    if "items" in out:
        out["items"] = _gemini_schema(out["items"])
    if "anyOf" in out:
        out["anyOf"] = [_gemini_schema(v) for v in out["anyOf"]]
    return out


def mcp_tools(profile: Profile | str, extra: Optional[List[dict]] = None) -> List[dict]:
    """``tools/list`` entries.  No ``outputSchema``: a structured result makes
    some clients drop the image content (Codex issue 10334)."""
    prof = load_profile(profile) if isinstance(profile, str) else profile
    out = []
    for e in tool_entries(prof, strict=False):
        spec = get_tool(e["name"]).spec
        out.append({"name": e["name"], "title": e["name"].replace("_", " "),
                    "description": e["description"], "inputSchema": e["schema"],
                    "annotations": {"readOnlyHint": spec.controls == "none"
                                    and spec.family not in ("act", "delegate", "make", "code"),
                                    "destructiveHint": False,
                                    "openWorldHint": spec.source == "model"}})
    return out + list(extra or [])
