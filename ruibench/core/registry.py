"""A tiny name -> factory registry, used for engines, interfaces and tasks.

Keeping this in one place means every extension point in RUIBench is
discovered the same way: decorate, then look up by string.
"""

from __future__ import annotations

from typing import Callable, Dict, Generic, Iterator, TypeVar

from .errors import ConfigError

T = TypeVar("T")


class Registry(Generic[T]):
    def __init__(self, what: str) -> None:
        self._what = what
        self._items: Dict[str, T] = {}

    def register(self, name: str, item: T | None = None):
        """Use as ``@registry.register("name")`` or ``registry.register("name", obj)``."""
        if item is not None:
            self._put(name, item)
            return item

        def deco(obj: T) -> T:
            self._put(name, obj)
            return obj

        return deco

    def _put(self, name: str, item: T) -> None:
        if name in self._items and self._items[name] is not item:
            raise ConfigError(f"duplicate {self._what} {name!r}")
        self._items[name] = item

    def get(self, name: str) -> T:
        try:
            return self._items[name]
        except KeyError:
            raise ConfigError(
                f"unknown {self._what} {name!r}; available: {', '.join(sorted(self._items)) or '(none)'}"
            ) from None

    def names(self) -> list[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._items))

    def __len__(self) -> int:
        return len(self._items)
