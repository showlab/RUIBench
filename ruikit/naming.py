"""The interfaces' names, as the paper gives them, and their identifiers.

The paper crosses six action abstractions with two interaction forms into 12
fixed RUIs and adds RUI-Free, which offers every abstraction in both forms:
13 interfaces.  A per-action RUI is named by its abstraction (``skill``), a
program RUI by the prefix ``program`` (``program-skill``), RUI-Free
``rui-free``.  These names are what a person types and reads: command-line
values, the columns of the results, reports and documentation.

Each RUI also has a versioned identifier, ``rui.code-skill/v1``: the catalog's
key (``ruikit/configs/interfaces.yaml``), the name of a result's folder and the
words the agent reads in its prompt.  The identifiers stay as the paper's runs
had them -- the prompt is part of the frozen protocol -- so ``program-skill``
and ``rui.code-skill/v1`` name the same interface.  The one irregular id is the
program over the pose, ``rui.code/v1``; ``rui.code-pose/v1`` is accepted for it.

The paper's RUIs carry the aids, the tools every RUI shares on top of its own
(24 in all, ``ruikit design``): the name ``program-skill`` runs
``rui.code-skill/v1+aids``.  RUI-Free has them built in.  The paper's ablation
adds the action-sequence variant (``sequence-delta``: several actions a reply,
nothing seen in between), which is not one of the 13.

Older names -- ``code-skill``, ``program`` for the program over the pose,
``code``, ``free`` -- are still read, in results written before and on the
command line, and mean the same interfaces.
"""

from __future__ import annotations

from typing import Dict, Tuple

#: the six action abstractions, in the paper's order
ABSTRACTIONS: Tuple[str, ...] = ("joint", "delta", "step", "pose", "point", "skill")
#: the interaction forms, as the paper writes them
FORMS: Tuple[str, ...] = ("per-action", "program", "free")
PER_ACTION: Tuple[str, ...] = ABSTRACTIONS
PROGRAM: Tuple[str, ...] = tuple(f"program-{a}" for a in ABSTRACTIONS)
FREE = "rui-free"
#: the 13 RUIs, in the order of the paper's main table
RUIS: Tuple[str, ...] = PER_ACTION + PROGRAM + (FREE,)
#: the action-sequence variant of the paper's ablation (not one of the 13)
SEQUENCE: Tuple[str, ...] = tuple(f"sequence-{a}" for a in ABSTRACTIONS)
NAMES: Tuple[str, ...] = RUIS + SEQUENCE

#: name -> the catalog's id (without components)
IDS: Dict[str, str] = {
    **{a: f"rui.{a}/v1" for a in ABSTRACTIONS},
    **{f"program-{a}": ("rui.code/v1" if a == "pose" else f"rui.code-{a}/v1") for a in ABSTRACTIONS},
    FREE: "rui.free/v1",
    **{f"sequence-{a}": f"rui.seq-{a}/v1" for a in ABSTRACTIONS},
}
_BY_ID: Dict[str, str] = {v: k for k, v in IDS.items()}
#: ids accepted besides the catalog's own
ID_ALIASES: Dict[str, str] = {"rui.code-pose/v1": "rui.code/v1", "rui.program/v1": "rui.code/v1",
                              **{f"rui.program-{a}/v1": IDS[f"program-{a}"] for a in ABSTRACTIONS}}
#: older names -> names
LEGACY: Dict[str, str] = {
    "program": "program-pose", "code": "program-pose", "code-pose": "program-pose",
    **{f"code-{a}": f"program-{a}" for a in ABSTRACTIONS if a != "pose"},
    **{f"seq-{a}": f"sequence-{a}" for a in ABSTRACTIONS},
    "free": FREE, "rui_free": FREE, "ruifree": FREE,
}
_LABEL = {"joint": "Joint", "delta": "Delta", "step": "Step", "pose": "Pose", "point": "Point",
          "skill": "Skill"}


def _base(text: str) -> str:
    """An id without its components and perception: ``rui.pose/v1+aids@x`` -> ``rui.pose/v1``."""
    return text.split("@")[0].split("+")[0]


def is_name(text: str) -> bool:
    """Whether ``text`` is a name (or an older one) rather than an id or a file."""
    t = text.strip().lower()
    return t in IDS or t in LEGACY


def name_of(text: str) -> str:
    """The name of an interface given by name, older name or id (composed or
    not): ``code-skill``, ``rui.code-skill/v1+aids`` -> ``program-skill``.
    Raises ``ValueError`` for anything else (an interface of its own, a file)."""
    t = text.strip()
    low = t.lower()
    if low in IDS:
        return low
    if low in LEGACY:
        return LEGACY[low]
    base = _base(t)
    base = ID_ALIASES.get(base, base)
    if base in _BY_ID:
        return _BY_ID[base]
    raise ValueError(f"{text!r} is not one of the named interfaces: "
                     + ", ".join(NAMES))


def id_of(text: str, aids: bool = True) -> str:
    """The id to run for a name: ``program-skill`` -> ``rui.code-skill/v1+aids``
    (the paper's RUI, with the aids; ``aids=False`` for the bare interface).
    RUI-Free carries its aids built in.  An id is returned as given, with an
    accepted alias (``rui.code-pose/v1``) put back to the catalog's id."""
    t = text.strip()
    if is_name(t):
        name = name_of(t)
        base = IDS[name]
        return base + ("+aids" if aids and name != FREE else "")
    base = _base(t)
    return ID_ALIASES[base] + t[len(base):] if base in ID_ALIASES else t


def form_of(name: str) -> str:
    """``per-action``, ``program``, ``sequence`` or ``free``."""
    n = name_of(name)
    if n == FREE:
        return "free"
    if n.startswith("program-"):
        return "program"
    if n.startswith("sequence-"):
        return "sequence"
    return "per-action"


def abstraction_of(name: str) -> str:
    """The action abstraction (``''`` for RUI-Free)."""
    n = name_of(name)
    return "" if n == FREE else n.split("-")[-1]


def label(name: str) -> str:
    """How a table or a page writes it: ``Skill``, ``Program-Skill``,
    ``Sequence-Delta``, ``RUI-Free``."""
    n = name_of(name)
    if n == FREE:
        return "RUI-Free"
    form = form_of(n)
    a = _LABEL[abstraction_of(n)]
    return a if form == "per-action" else f"{form.capitalize()}-{a}"
