"""The episode's limits: replies, tool calls, simulated and wall-clock time."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

#: a reply of a tool-calling agent is a smaller thing than one of the
#: benchmark's model calls, which carries a whole list of commands: a look, a
#: find, an estimate and a few moves take about this many replies
REPLIES_PER_CALL = 10


@dataclass
class Budget:
    """When an episode ends unfinished.  Left unset, the limits come from the
    task: its model-call budget (``max_model_calls``) times ``REPLIES_PER_CALL``
    replies, and times T_ref seconds of simulated time, never past the
    benchmark's own cap on an episode (``BudgetConfig.max_sim_seconds``) --
    past either, the episode is over and failed."""

    max_tool_calls: int = 800
    max_sim_seconds: Optional[float] = None
    max_wall_seconds: float = 3600.0
    max_model_turns: Optional[int] = None
    max_replies: Optional[int] = None

    def as_dict(self) -> dict:
        return {"max_tool_calls": self.max_tool_calls, "max_sim_seconds": self.max_sim_seconds,
                "max_wall_seconds": self.max_wall_seconds, "max_model_turns": self.max_model_turns,
                "max_replies": self.max_replies}


class EpisodeBudget:
    """Checking and reporting what is left of the budget."""

    def _check_budget(self) -> Optional[str]:
        b = self.budget
        if self.tool_calls >= b.max_tool_calls:
            return "budget_calls"
        if b.max_replies is not None and self.replies >= b.max_replies:
            return "budget_replies"
        if b.max_model_turns is not None and self.model_turns > b.max_model_turns:
            return "budget_turns"
        if self.sim_time > b.max_sim_seconds:
            return "timeout"
        if time.time() - self._wall_start > b.max_wall_seconds:
            return "wall_timeout"
        return None

    def budget_left(self) -> dict:
        b = self.budget
        out = {"tool_calls": b.max_tool_calls - self.tool_calls,
               "sim_seconds": round(b.max_sim_seconds - self.sim_time, 1)}
        if b.max_replies is not None:
            out["replies"] = b.max_replies - self.replies
        if b.max_model_turns is not None:
            out["model_turns"] = b.max_model_turns - self.model_turns
        return out
