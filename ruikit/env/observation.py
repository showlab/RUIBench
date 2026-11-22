"""The observation policy: fresh pictures of every camera after each reply that
moved an arm (push), or only on request (pull)."""

from __future__ import annotations

from typing import List

from ruikit.core.result import ToolResult


class Observation:
    """Pushing the cameras' pictures after a reply that moved an arm."""

    def _push(self, results: List[ToolResult], moved: bool = False) -> None:
        """Every interface shows the world after a reply that moved an arm, as
        Show-Harness does after each step (it pushes agentview + wrist after
        each action): the cameras' pictures ride on the reply's last result.
        They cost the cameras' usual time, not a reply (agents on the step
        interface spent a third of their replies only looking).  ``+pull``
        leaves it to the agent."""
        if self.over or not results:
            return
        cams = (self.profile.push or {}).get("cameras")
        if not cams or not moved:
            return
        # a program interface has camera_look among its program's functions, not its tools
        look = self._invoke("camera_look", {"cameras": None if cams == "all" else list(cams)},
                            from_code="camera_look" not in self.profile.tools)
        if look.is_error:
            return
        last = results[-1]
        last.images = list(last.images or []) + list(look.images or [])
        last.data = {**(last.data or {}),
                     "pushed": {"frames": (look.data or {}).get("frames"),
                                "note": "the cameras after this reply's moves"}}
