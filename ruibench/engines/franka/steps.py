"""What a session on the real robot is doing, for the operator.

Each of the agent's calls is said as it starts and as it answers, a program's calls
among them, numbered: ``#4.2`` is the second call of the agent's fourth, and a
program's source is said with it.  The episode tells the engine (``Episode._invoke``,
on a real engine only).  The lines go to the session's standard output, its
``server.log``, marked; the runner follows them for the operator
(``ruikit.experiments.real``), and the window says the call under way (``view.py``).
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Optional

#: how the runner tells these lines among whatever else the session's log holds
MARK = "@@ "
#: a call's arguments and its answer, on a line each
LINE = 200
PROGRAM_LINES = 60


def _brief(args: Optional[Mapping[str, Any]]) -> str:
    """A call's arguments, as they would be written."""
    return ", ".join(f"{k}={v!r}" for k, v in dict(args or {}).items() if k != "source")[:LINE]


class Steps:
    """The account (the module docstring)."""

    def __init__(self) -> None:
        self._calls = 0                      # the agent's calls so far
        self._inner = 0                      # the calls of the present one
        self._doing = ""
        self._since = time.monotonic()

    def _say(self, t: float, line: str) -> None:
        print(f"{MARK}{t:6.1f} s  {line}", flush=True)

    def _number(self, inner: bool) -> str:
        return f"#{self._calls}.{self._inner}" if inner else f"#{self._calls}"

    def began(self, t: float, name: str, args: Optional[Mapping[str, Any]], inner: bool) -> None:
        if inner:
            self._inner += 1
        else:
            self._calls, self._inner = self._calls + 1, 0
        self._doing = f"{self._number(inner)} {name} {_brief(args)}".rstrip()
        self._say(t, self._doing)
        source = str(dict(args or {}).get("source") or "").splitlines()
        for line in source[:PROGRAM_LINES]:
            print(f"{MARK}{'':10}| {line}", flush=True)
        if len(source) > PROGRAM_LINES:
            print(f"{MARK}{'':10}| ... {len(source) - PROGRAM_LINES} more lines", flush=True)

    def ended(self, t: float, result: Any, inner: bool) -> None:
        text = " ".join(str(getattr(result, "text", "") or "").split())
        error = getattr(result, "error", None)
        arrow = f"!! {error}:" if error else "->"
        self._say(t, f"{self._number(inner)} {arrow} {text}"[:LINE])
        self._doing = f"#{self._calls} going on" if inner else ""
        self._since = time.monotonic()

    def doing(self) -> str:
        """What is under way, for the window."""
        return self._doing or (f"waiting for the agent: {time.monotonic() - self._since:.0f} s"
                               f" since its last call" if self._calls else
                               "waiting for the agent's first call")


__all__ = ["MARK", "Steps"]
