"""Giving up: ending the episode early, saying why (the tool is the task's own,
``task_give_up``; every set of aids carries it)."""

from __future__ import annotations

from ruikit.harness.components.base import Component

COMPONENT = Component("give_up", "放弃", "end the episode, saying why", ("task_give_up",))
