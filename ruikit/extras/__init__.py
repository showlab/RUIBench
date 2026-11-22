"""Tools outside the design space: other ways of moving the arm (several waypoints
in one call, a chunk of deltas, back home), tools an agent makes of the others,
another model's answer, a VLA policy.  They attach to an interface as the
components do (``+motion``, ``+ask:openai``), never as part of the aids."""

from __future__ import annotations

from typing import Mapping

from ruikit.extras import make, models, motion, policy
from ruikit.harness.components.base import Component

#: the extras, by the name an interface id attaches them with
EXTRAS: Mapping[str, Component] = {c.name: c for c in (
    models.ASK, models.DECIDE, motion.COMPONENT, make.COMPONENT, models.IMAGINE,
    policy.COMPONENT)}

__all__ = ["EXTRAS"]
