"""The Python SDK: the profile's tools as methods.

For researchers writing their own loop, and for smolagents / LangGraph /
Agents-SDK wrappers::

    from ruikit import Episode
    from ruikit.serve.sdk import Client

    ep = Episode("DS2", level="easy", profile="rui.pose/v1")
    robot = Client(ep)
    f = robot.camera.look(cameras=["front"]).data["frames"][0]["frame_id"]
    pots = robot.vision.detect(frame_id=f, query="pot").data["objects"]

Names match the tools exactly (``robot.robot.move_to`` is ``robot_move_to``);
every call goes through ``Episode.call`` -- the same validation, clock, budget
and log as the MCP server.  Only tools the profile offers are attached.
"""

from __future__ import annotations

from typing import Any, Dict


class _Family:
    def __init__(self, client: "Client", family: str) -> None:
        self._client = client
        self._family = family

    def __repr__(self) -> str:
        names = [k for k in vars(self) if not k.startswith("_")]
        return f"<{self._family}: {', '.join(names)}>"


class Client:
    def __init__(self, episode) -> None:
        self.episode = episode
        for name in episode.profile.tools:
            family, _, rest = name.partition("_")
            fam = getattr(self, family, None)
            if not isinstance(fam, _Family):
                fam = _Family(self, family)
                setattr(self, family, fam)
            setattr(fam, rest, self._bind(name))

    def _bind(self, name: str):
        def call(**kwargs: Any):
            return self.episode.call(name, kwargs)

        call.__name__ = name
        return call

    def call(self, name: str, **kwargs: Any):
        """Call any offered tool by its full name (also custom ``my_*`` tools)."""
        return self.episode.call(name, kwargs)

    def batch(self, *calls: Dict[str, Any]):
        """Several calls as one model reply: ``batch({"tool": ..., "args": {...}}, ...)``."""
        return self.episode.call_batch([(c["tool"], c.get("args") or {}) for c in calls])
