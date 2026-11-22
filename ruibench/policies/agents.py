"""An agent from a short spec -- the same parser for the CLI and the workers.

    reference                          the task's scripted reference
    ignore_change                      a negative control around it
    reference+misperceive=0.02         a stress agent: objects seen ~2 cm off
    reference+serial                   a stress agent: one arm at a time
    reference+ignore_rules             a stress agent: plays as if the game had no rules
    reference+greedy                   a stress agent: plays for its own lines only
    first_frame_only+misperceive=0.03  both
"""

from __future__ import annotations

from typing import List, Tuple

from ..core.errors import ConfigError
from .base import Agent
from .negative_controls import CONTROL_CHECKS, wrap
from .stress import Greedy, IgnoreRules, Misperceiving, Serial

MODIFIERS = ("misperceive", "serial", "ignore_rules", "greedy")

#: what each part of a spec does to the reference, in words
EFFECTS = {
    "do_nothing": "never moves",
    "first_frame_only": "only ever looks at the first picture",
    "ignore_feedback": "ignores the right/wrong feedback",
    "ignore_change": "ignores the changed instruction",
    "ignore_demonstration": "never watches the demonstration",
}


def parse_agent_spec(spec: str) -> Tuple[str, List[Tuple[str, float]]]:
    """``"reference+misperceive=0.03"`` -> ``("reference", [("misperceive", 0.03)])``."""
    base, *parts = (spec or "reference").split("+")
    base = base or "reference"
    if base != "reference" and base not in CONTROL_CHECKS:
        raise ConfigError(f"unknown agent {base!r} in {spec!r}; have reference, "
                          + ", ".join(sorted(CONTROL_CHECKS)))
    modifiers = []
    for part in parts:
        key, _, value = part.partition("=")
        if key not in MODIFIERS:
            raise ConfigError(f"unknown agent modifier {key!r} in {spec!r}; have: "
                              + ", ".join(MODIFIERS))
        modifiers.append((key, float(value or (0.02 if key == "misperceive" else 1.0))))
    return base, modifiers


def describe_agent(spec: str) -> str:
    """What the agent does differently from the reference, in a few words."""
    base, modifiers = parse_agent_spec(spec)
    words = [] if base == "reference" else [EFFECTS[base]]
    for key, value in modifiers:
        if key == "misperceive":
            words.append(f"sees every object about {value * 100:g} cm from where it is")
        elif key == "serial":
            words.append("uses one arm at a time")
        elif key == "ignore_rules":
            words.append("ignores the rules of the game")
        elif key == "greedy":
            words.append("plays for its own lines and never blocks")
    return " and ".join(words) or "the scripted reference"


def build_agent(spec: str, inst) -> Agent:
    if inst.reference_factory is None:
        raise ConfigError(f"{inst.task_id} has no scripted reference yet")
    base, modifiers = parse_agent_spec(spec)
    ref = inst.reference_factory(inst)
    agent = ref if base == "reference" else wrap(base, ref)
    for key, value in modifiers:
        if key == "misperceive":
            agent = Misperceiving(agent, sigma=value, seed=inst.seed)
        elif key == "serial":
            agent = Serial(agent)
        elif key == "ignore_rules":
            agent = IgnoreRules(agent)
        elif key == "greedy":
            agent = Greedy(agent)
    return agent
