"""The aids: the components every interface of the main table carries, in the groups the
ablation takes away one at a time (+aids-<group>)."""

from __future__ import annotations

from typing import Mapping, Tuple

from ruikit.harness.components import COMPONENTS

#: the aids: the components every interface of the main table carries, in the
#: groups the ablation takes away one at a time (``+aids-<group>``).  Each in
#: its default implementation; none moves the arm another way or asks another
#: model (``AIDS_LEFT_OUT``)
AID_GROUPS: Mapping[str, Tuple[str, ...]] = {
    "memory": ("memory",),               # notes, the pictures again, the scene record
    "marks": ("marks",),                 # numbered dots on a picture
    "pointer": ("pointer",),             # a pixel on what a description names: the detector
    "track": ("track",),                 # keep following one thing
    "plan": ("grasps", "paths", "reach"),    # where to grasp, a way round, can it get there
}


#: in the aids whatever group is taken away: ending the episode early, saying why
AID_ALWAYS: Tuple[str, ...] = ("give_up",)


#: what the aids never include, and why
AIDS_LEFT_OUT: Mapping[str, str] = {
    "motion": "another way of moving the arm (several waypoints, a chunk, back home)",
    "make": "a tool made of moves: a whole sequence in one call",
    "ask": "another model's answer",
    "decide": "another model's answer",
    "imagine": "another model's answer",
    "policy": "another model moving the arm",
}


def parse_aids(token: str) -> Tuple[str, ...]:
    """'aids' -> (), 'aids-memory' -> ('memory',): the groups taken away."""
    head, *removed = token.split("-")
    if head != "aids":
        raise ValueError(f"not an aids token: {token!r}")
    for group in removed:
        if group not in AID_GROUPS:
            raise ValueError(f"no aids group {group!r}; there are {', '.join(AID_GROUPS)} "
                             "(no aids at all: the interface's own id, without +aids)")
    if len(set(removed)) != len(removed):
        raise ValueError(f"{token!r} names a group twice")
    return tuple(removed)


def aid_components(removed: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    """The components of the aids less the ``removed`` groups, in a fixed order."""
    out = [name for group, names in AID_GROUPS.items() if group not in removed for name in names]
    return tuple(out) + AID_ALWAYS


def aid_tools(removed: Tuple[str, ...] = ()) -> Tuple[str, ...]:
    """The tools of the aids less the ``removed`` groups."""
    return tuple(t for name in aid_components(removed) for t in COMPONENTS[name].tools)
