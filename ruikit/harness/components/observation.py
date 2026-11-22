"""The observation policy: after each reply that moved an arm, fresh pictures of
every camera ride on its last result (push, the default); with ``+pull`` the
agent looks when it chooses to, with camera_look.  The pushing itself is the
episode's (``ruikit.env.observation``)."""

from __future__ import annotations

from typing import Any, Mapping

#: what an interface pushes after a reply that moved an arm
PUSH: Mapping[str, Any] = {"cameras": "all", "history": 2, "proprio": True, "events": True}
