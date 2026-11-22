"""What a tool returns, and how it fails.

Two channels, as every vision tool-use system ends up with:
structured ``data`` plus one short ``text`` line for everything a model reads,
and ``images`` only for what it has to *look* at.  Every result an episode
hands back also carries the events since the previous call, the current
simulated time and the budget left -- a client that drops server
notifications still misses nothing.

Errors are results too (MCP's ``isError``), with a code from a fixed list and
a message built only from facts the agent could observe, plus a legal next
step.  A tool raises :class:`ToolError`; the episode turns it into a result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional

#: every error code a tool may return, with what it means to the agent
ERROR_CODES: Mapping[str, str] = {
    "invalid_args": "the arguments do not match the tool's schema",
    "unreachable": "the target is outside the arm's reach",
    "blocked": "the motion stopped short of its target",
    "not_visible": "the pixel or object is not visible in that frame",
    "stale_frame": "the frame id does not exist or has expired",
    "stale_handle": "the handle does not exist or has expired",
    "resource_busy": "the arm is already executing another action",
    "not_in_profile": "the tool is not available in this configuration",
    "not_supported": "the station or engine cannot do this",
    "backend_unavailable": "the model or service behind this tool is not configured",
    "budget_exhausted": "a budget of this episode is used up",
    "episode_over": "the episode has ended",
    "rate_limited": "too many calls of this tool",
    "tool_failed": "the tool failed for a reason stated in the message",
}


class ToolError(Exception):
    """Raised by a tool; becomes an error result the model can act on."""

    def __init__(self, code: str, message: str, hint: str = "",
                 data: Optional[Mapping[str, Any]] = None) -> None:
        if code not in ERROR_CODES:
            raise ValueError(f"unknown tool error code {code!r}")
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint
        self.data = dict(data or {})


@dataclass
class Image:
    """One picture in a result.  ``png`` holds the encoded bytes when the engine
    renders pixels; the mock engine has none and says so in ``note``."""

    frame_id: str
    camera: str
    width: int
    height: int
    png: Optional[bytes] = None
    path: Optional[str] = None
    note: str = ""

    def as_json(self) -> dict:
        out = {"frame_id": self.frame_id, "camera": self.camera,
               "width": self.width, "height": self.height}
        if self.path:
            out["path"] = self.path
        if self.note:
            out["note"] = self.note
        return out


@dataclass
class ToolResult:
    data: Dict[str, Any] = field(default_factory=dict)
    text: str = ""
    images: List[Image] = field(default_factory=list)
    error: Optional[str] = None
    hint: str = ""
    # -- stamped by the episode on the way out ------------------------------
    events: List[dict] = field(default_factory=list)
    sim_time: float = 0.0
    budget: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def fail(cls, err: ToolError) -> "ToolResult":
        return cls(data=dict(err.data), text=err.message, error=err.code, hint=err.hint)

    def as_json(self) -> dict:
        """The whole result as JSON values: images by frame id."""
        out: Dict[str, Any] = {"ok": not self.is_error}
        if self.is_error:
            out["error"] = self.error
        if self.text:
            out["text"] = self.text
        if self.hint:
            out["hint"] = self.hint
        if self.data:
            out["data"] = self.data
        if self.images:
            out["images"] = [im.as_json() for im in self.images]
        out["sim_time"] = round(self.sim_time, 3)
        if self.events:
            out["events"] = list(self.events)
        if self.budget:
            out["budget"] = dict(self.budget)
        return out

    def summary(self) -> str:
        """One line for logs and the text channel."""
        head = f"[{self.error}] " if self.is_error else ""
        tail = f" (hint: {self.hint})" if self.hint else ""
        return f"{head}{self.text}{tail}"
