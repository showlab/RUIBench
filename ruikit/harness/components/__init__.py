"""The Harness components: what a Harness adds to an interface besides the
action tools -- memory, the aids that help an agent see, find, follow and
plan, giving up -- each in its own module, attached as ``+<name>``.  The
operation guides (``guides``) and the observation policy (``observation``)
are components too, attached as ``+guide`` and ``+pull``."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from ruikit.harness.components import check, give_up, marks, memory, pointer, track
from ruikit.harness.components.base import Component, Implementation
from ruikit.harness.components.planning import grasps, paths, reach

#: the components, by the name an interface id attaches them with
COMPONENTS: Mapping[str, Component] = {c.name: c for c in (
    memory.COMPONENT, marks.COMPONENT, pointer.COMPONENT, track.COMPONENT,
    grasps.COMPONENT, paths.COMPONENT, reach.COMPONENT, give_up.COMPONENT,
    check.COMPONENT)}

#: what the agent is told of a tool in a component's other implementations,
#: by (tool, implementation)
BY_IMPLEMENTATION: Dict[Tuple[str, str], str] = {
    ("vision_point", "vlm"): pointer.VLM_DESCRIPTION,
    ("vision_track", "template"): track.TEMPLATE_DESCRIPTION,
}

__all__ = ["BY_IMPLEMENTATION", "COMPONENTS", "Component", "Implementation"]
