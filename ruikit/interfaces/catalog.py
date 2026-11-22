"""The named interfaces, and loading an interface by its id.

``ruikit/configs/interfaces.yaml`` names each interface by its interaction
form and action abstraction, from which its tools follow, and gives the title
the agent is told.  An id may compose more onto a named interface
(``ruikit.interfaces.compose``).  An interface may also be a YAML file of its
own, listing its tools; it may ``extends:`` another interface and take what a
program may call from one (``code_from:``).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

import yaml

from ruikit.core.paths import CONFIGS
from ruikit.interfaces.base import Interface, Profile, profile_problems
from ruikit.interfaces.compose import compose
from ruikit.interfaces.free import Free
from ruikit.interfaces.per_action import PerAction
from ruikit.interfaces.program import Program
from ruikit.interfaces.sequence import Sequence

#: the interaction forms, by name
FORMS: Mapping[str, Interface] = {f.name: f for f in (PerAction(), Program(), Sequence(), Free())}

CATALOG = CONFIGS / "interfaces.yaml"


@lru_cache(maxsize=1)
def _catalog() -> Mapping[str, Mapping[str, Any]]:
    return yaml.safe_load(CATALOG.read_text(encoding="utf-8")) or {}


def list_profiles() -> List[str]:
    """The ids of the named interfaces, in the catalog's order."""
    return list(_catalog())


def interface_for(form: str, action: Optional[str] = None) -> str:
    """The id of the named interface of a form over an action abstraction
    ('per_action', 'pose' -> 'rui.pose/v1')."""
    for interface_id, entry in _catalog().items():
        if entry.get("form") == form and entry.get("action") == action:
            return interface_id
    raise KeyError(f"no named interface of the {form} form over {action!r}")


def form_of(interface_id: str) -> Tuple[str, str]:
    """The (form, action) of a named interface; ('', '') for one listing its own tools."""
    entry = _catalog().get(interface_id) or {}
    return str(entry.get("form") or ""), str(entry.get("action") or "")


def load_profile(interface_id: str, _seen: Tuple[str, ...] = ()) -> Profile:
    """An interface by its id -- ``rui.pose/v1``, composed or not
    (``rui.pose/v1+aids@owlv2-sam2``) -- or by its YAML file."""
    if ("@" in interface_id or "+" in interface_id) and not Path(interface_id).exists():
        spec, at, perception = interface_id.partition("@")
        base, *tokens = spec.split("+")
        prof = compose(load_profile(base, _seen), tuple(tokens), perception if at else None)
    elif interface_id in _catalog():
        prof = _named(interface_id)
    else:
        prof = _from_file(interface_id, _seen)
    problems = profile_problems(prof)
    if problems:
        raise ValueError(f"profile {prof.id!r}: " + "; ".join(problems))
    return prof


def _named(interface_id: str) -> Profile:
    entry = dict(_catalog()[interface_id])
    form = entry.pop("form", None)
    if form is not None:
        from ruikit.actions import ACTIONS

        action = entry.pop("action", None)
        return FORMS[form].profile(interface_id, ACTIONS[action] if action else None, **entry)
    from ruikit.harness.components.observation import PUSH

    return _profile({"push": dict(PUSH), **entry, "id": interface_id}, interface_id)


def _from_file(path_text: str, seen: Tuple[str, ...]) -> Profile:
    path = Path(path_text)
    if path.suffix not in (".yaml", ".yml") or not path.exists():
        raise FileNotFoundError(f"no profile {path_text!r}: neither named in {CATALOG} nor a "
                                "YAML file")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parent = data.pop("extends", None)
    if parent:
        if parent in seen:
            raise ValueError(f"profile {path_text!r}: circular extends through {parent!r}")
        data = _merge(load_profile(parent, seen + (path_text,)).as_dict(), data)
    code_from = data.pop("code_from", None)
    if code_from and not data.get("code_tools"):
        data["code_tools"] = list(load_profile(code_from, seen + (path_text,)).tools)
    return _profile(data, path_text)


def _profile(data: Dict[str, Any], where: str) -> Profile:
    unknown = sorted(set(data) - set(Profile.__dataclass_fields__))
    if unknown:
        raise ValueError(f"profile {where!r}: unknown keys {unknown}")
    return Profile(**{**data, "tools": tuple(data.get("tools") or ()),
                      "code_tools": tuple(data.get("code_tools") or ()),
                      "plugins": tuple(data.get("plugins") or ())})


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        if key == "params":
            merged = {k: dict(v) for k, v in (base.get("params") or {}).items()}
            for tool_name, settings in (value or {}).items():
                merged.setdefault(tool_name, {}).update(settings or {})
            out["params"] = merged
        elif key == "push":
            out["push"] = {**(base.get("push") or {}), **(value or {})}
        else:
            out[key] = value
    return out
