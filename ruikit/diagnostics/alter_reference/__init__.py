"""Other ways to solve a task than its reference.  Each contributor keeps theirs in a folder of their
own, ``alter_reference/<name>/``.

A task's reference (``reference_factory``, in ``ruibench/tasks/impl/``) is one way to solve it, and
it is frozen with the task.  An alternative decides differently -- another order, another plan,
another way of finding something out -- and the scorer takes it all the same.  Like a reference it
reads the simulator unless it says otherwise; nothing here shapes a benchmark episode, so it is
outside the freeze (``ruibench/tasks/freeze.py``).

``make("alt:wk2_other_crossing", inst)`` gives the policy, ``ways("WK2")`` the names for a task.
``coverage --reference alt:wk2_other_crossing`` runs one through an interface's tools.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from importlib import import_module
from typing import Callable, Dict, List

#: the authors' packages; each has ``WAYS``, a list of ``Way``
AUTHORS = ("alt",)


@dataclass(frozen=True)
class Way:
    task: str
    name: str
    #: the policy for a task instance
    make: Callable
    #: one line: how it differs from the reference
    what: str = ""
    #: how it did on Isaac (easy, 8 s a reply, code-joint and code-delta), as its author last ran it
    isaac: str = "not run on Isaac"


def adopt(cls, inst):
    """A subclass of the task's reference policy, built with the reference's own arguments."""
    reference = inst.reference_factory(inst)
    policy = cls.__new__(cls)
    policy.__dict__.update(reference.__dict__)
    return policy


def _all() -> Dict[str, Way]:
    out = {}
    for author in AUTHORS:
        for way in import_module(f"{__name__}.{author}").WAYS:
            out[f"{author}:{way.name}"] = way
    return out


def ways(task: str) -> List[str]:
    return [name for name, way in _all().items() if way.task == task]


def make(name: str, inst):
    """The policy of the way called ``name`` for this task instance."""
    way = _all().get(name)
    if way is None:
        raise KeyError(f"no alternative {name!r}; there are {sorted(_all())}")
    if way.task != inst.task_id:
        raise ValueError(f"{name} is for {way.task}, not {inst.task_id}")
    policy = way.make(inst)
    policy.reset()
    return policy


def install(name: str, inst) -> None:
    """From now on this task instance's reference is the way called ``name``; the expressors,
    which ask the instance for its reference, then act that way."""
    original = copy.copy(inst)
    inst.reference_factory = lambda _inst: make(name, original)
