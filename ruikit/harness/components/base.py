"""A Harness component: tools that attach to any interface, often in several
implementations, one chosen per run so that it can be swapped while everything
else stays (the ablations)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Tuple


@dataclass(frozen=True)
class Implementation:
    """One implementation of a component: the tools it offers (``None``: the
    component's own) and the per-tool settings it fixes."""
    name: str
    zh: str
    en: str
    tools: Optional[Tuple[str, ...]] = None
    params: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    #: what has to run for it to be offered (a server, a key)
    needs: str = ""


@dataclass(frozen=True)
class Component:
    """Tools that attach to an interface as ``+<name>`` or ``+<name>:<implementation>``."""
    name: str
    zh: str
    en: str
    tools: Tuple[str, ...]
    implementations: Tuple[Implementation, ...] = ()
    default: str = ""
    #: why it cannot be offered in this benchmark's setting, if it cannot
    unavailable: str = ""
    #: older names of implementations -> their names now
    aliases: Mapping[str, str] = field(default_factory=dict)

    def implementation(self, name: Optional[str]) -> Optional[Implementation]:
        """The implementation called ``name`` (the default for ``None``); a
        model behind it may follow a slash: ``vlm/openai``."""
        if not self.implementations:
            if name:
                raise ValueError(f"component {self.name!r} has no implementations to choose "
                                 f"from (asked for {name!r})")
            return None
        name = self.aliases.get(name, name) if name else self.default
        for im in self.implementations:
            if im.name == name:
                return im
        head, _, rest = name.partition("/")
        for im in self.implementations:
            if im.name == head and rest:
                return _with_backend(im, rest)
        raise ValueError(f"component {self.name!r} has no implementation {name!r}; it has "
                         + ", ".join(i.name for i in self.implementations))


def _with_backend(im: Implementation, backend: str) -> Implementation:
    params = {tool: {**settings, "backend": backend} for tool, settings in im.params.items()}
    return Implementation(f"{im.name}/{backend}", im.zh, im.en, im.tools, params, im.needs)


def backends(tool: str, names: Tuple[Tuple[str, str, str], ...],
             needs: str) -> Tuple[Implementation, ...]:
    """One implementation per model route: ``(name, zh, en)`` each, all of the same tool."""
    return tuple(Implementation(n, zh, en, params={tool: {"backend": n}}, needs=needs)
                 for n, zh, en in names)
