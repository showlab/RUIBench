"""The action abstractions, one module each, through one shared controller.

These cover the action granularities the published interfaces use: joint
targets (Inspect Robots, yad-use), semantic steps (Show-Harness), relative
moves (Embody direct, Inspect ``move_by``), absolute poses (Robocurve, Inspect
``move_to``, CaP-X ``goto_pose``), pixels (MOKA, Gemini-ER) and ready-made
skills (SayCan, ROS-MCP).

Every motion goes through the benchmark's executor and its limits; none can
place an object, change the physics or reset anything.  A motion either blocks
(default; at most 20 s of simulated time) or, with ``wait=false``, returns a
handle at once for ``robot_wait`` / ``robot_stop``.  In one model reply,
motions on different arms run at the same time.
"""

from __future__ import annotations

from typing import Mapping

from ruikit.actions import delta, joint, point, pose, skill, step
from ruikit.actions.base import ActionAbstraction

#: the action abstractions, from the arm's joints to a whole act
ACTIONS: Mapping[str, ActionAbstraction] = {a.name: a for a in (
    joint.ABSTRACTION, delta.ABSTRACTION, step.ABSTRACTION, pose.ABSTRACTION,
    point.ABSTRACTION, skill.ABSTRACTION)}

__all__ = ["ACTIONS", "ActionAbstraction"]
