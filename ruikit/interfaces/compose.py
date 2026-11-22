"""Composing an interface in its id: Harness components after ``+``, the
perception backend after ``@``.

    rui.pose/v1+marks:masks+memory      two components, the marks drawn as outlines
    rui.pose/v1+aids                    the aids: every component that helps the agent
                                        remember, see, find, follow and plan
    rui.pose/v1+aids-memory             the aids less one group (the ablations)
    rui.pose/v1+pull                    no pictures pushed after a move
    rui.pose/v1+guide                   with the operation guide
    rui.pose/v1+motion                  an extra (``ruikit.extras``): several waypoints a call
    rui.pose/v1@owlv2-sam2,depth=mono   perception by OWLv2 + SAM 2.1, depth from RGB alone
    rui.pose/v1@privileged              the control condition: the simulator's own values

The id is kept as written, defaults left out, so every log says what was
attached.  A component whose tools are open-track only (a second model, a
planner backed by one) makes the interface an open-track one.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Mapping, Optional, Tuple

from ruikit.core.tool import all_tools
from ruikit.interfaces.base import PERCEPTION, Profile
from ruikit.perception.settings import parse_perception, perception_label

if TYPE_CHECKING:
    from ruikit.harness.components.base import Component, Implementation


def attachable() -> Mapping[str, Component]:
    """Everything an id may attach after a ``+``: the Harness components, then the extras."""
    from ruikit.extras import EXTRAS
    from ruikit.harness.components import COMPONENTS

    return {**COMPONENTS, **EXTRAS}


def parse_component(token: str) -> Tuple[Component, Optional[Implementation]]:
    """'marks:masks' -> (the marks component, its masks implementation)."""
    name, _, impl = token.partition(":")
    found = attachable()
    component = found.get(name)
    if component is None:
        raise ValueError(f"no component {name!r}; there are {', '.join(found)}")
    if component.unavailable:
        raise ValueError(f"component {name!r} cannot be offered: {component.unavailable}")
    return component, component.implementation(impl or None)


def _named(prof: Profile, label: str) -> str:
    """The id with ``+label`` added before any ``@`` perception setting."""
    base_id, at, perception = prof.id.partition("@")
    return f"{base_id}+{label}" + (f"@{perception}" if at else "")


def with_component(prof: Profile, token: str) -> Profile:
    """The interface with one more component or extra ('marks', 'marks:masks',
    'ask:openai'): its tools added -- to what a program may call, on a code
    channel -- and the settings its implementation fixes.  'aids', 'pull' and
    'guide' are composed by their own functions."""
    if token == "aids" or token.startswith("aids-"):
        return with_aids(prof, token)
    if token == "pull":
        return with_pull(prof)
    if token == "guide" or token.startswith("guide:"):
        from ruikit.harness.components.guides import DEFAULT_GUIDE

        return with_guide(prof, token.partition(":")[2] or DEFAULT_GUIDE)
    component, impl = parse_component(token)
    tools = impl.tools if impl is not None and impl.tools is not None else component.tools
    registry = all_tools()
    top, inner = list(prof.tools), list(prof.code_tools)
    for name in tools:
        # a program calls them -- but not a tool that makes tools, nor one that
        # ends the episode (task_done is the program interface's own, too)
        if prof.channel == "code" and not name.startswith(("toolkit_", "task_")):
            if name not in inner:
                inner.append(name)
        elif name not in top:
            top.append(name)
    params = {k: dict(v) for k, v in prof.params.items()}
    for tool_name, settings in ((impl.params if impl is not None else {}) or {}).items():
        params.setdefault(tool_name, {}).update(settings)
    track = prof.track
    if any(track not in registry[n].spec.tracks for n in tools if n in registry):
        track = "open"                  # a second model or a model planner: not controlled
    label = component.name + (f":{impl.name}" if impl is not None
                              and impl.name != component.default else "")
    return replace(prof, id=_named(prof, label), tools=tuple(top), code_tools=tuple(inner),
                   params=params, track=track, plugins=prof.plugins + (label,))


def with_aids(prof: Profile, token: str) -> Profile:
    """The interface with the aids ('aids'), or with all but some groups of them
    ('aids-memory'): their components attached one by one, the id saying only
    which groups were taken away."""
    from ruikit.harness.components.aids import aid_components, parse_aids

    out = prof
    for name in aid_components(parse_aids(token)):
        out = with_component(out, name)
    return replace(out, id=_named(prof, token), plugins=prof.plugins + (token,))


def with_pull(prof: Profile) -> Profile:
    """The interface without pictures pushed after its moves: the agent looks
    when it chooses to, with camera_look."""
    return replace(prof, id=_named(prof, "pull"), push={**prof.push, "cameras": None},
                   plugins=prof.plugins + ("pull",))


def with_guide(prof: Profile, name: str) -> Profile:
    """The interface with an operation guide: general practice the Harness
    adds to what the agent is told, the same for every task
    (``ruikit/configs/guides/<name>.md``)."""
    from ruikit.harness.components.guides import DEFAULT_GUIDE, guide_text

    guide_text(name)                          # a guide that is not there: said here
    label = "guide" if name == DEFAULT_GUIDE else f"guide:{name}"
    return replace(prof, id=_named(prof, label), guide=name, plugins=prof.plugins + (label,))


def with_perception(prof: Profile, perception: str) -> Profile:
    """The same interface with another perception backend: from pixels
    (``model``, the default) with a detector and depth settings
    (``owlv2-sam2,depth=mono,noise=2``); ``synthetic``, the mock engine's test
    double, refused where pictures are rendered; or ``privileged``, the control
    condition, every perception tool answering with the simulator's own values
    (never a default).  The id says which, defaults left out, so a log and a
    table never mix two conditions."""
    base_id = prof.id.split("@")[0]
    if perception in ("synthetic", "privileged"):
        return replace(prof, id=f"{base_id}@{perception}", perception=perception)
    if perception not in PERCEPTION and not perception.strip():
        raise ValueError("an empty perception setting")
    settings = {**prof.params.get("perception", {}), **parse_perception(perception)}
    label = perception_label(settings)
    params = {k: dict(v) for k, v in prof.params.items()}
    if settings:
        params["perception"] = settings
    return replace(prof, id=base_id + (f"@{label}" if label else ""), perception="model",
                   params=params)


def compose(prof: Profile, tokens: Tuple[str, ...], perception: Optional[str]) -> Profile:
    """``prof`` with each ``+`` token attached in order, then the perception set."""
    for token in tokens:
        prof = with_component(prof, token)
    if perception is not None:
        prof = with_perception(prof, perception)
    return prof
