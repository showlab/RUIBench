"""CR6: the arch held longer at each point of the carry.  The same pinch points and the same fold as
the reference, so a shallow difference: no other fold of the shirt passes the scorer's fit as
surely."""

from __future__ import annotations

from typing import List

from ruibench.interfaces.commands import Command, Wait
from ruibench.tasks.impl.cr6_fold import FoldPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class SlowCarry(FoldPolicy):
    name = "fold_cloth_slow_carry"

    def _carry(self) -> List[Command]:
        return [Wait(0.2) if isinstance(c, Wait) else c for c in super()._carry()]


WAY = Way("CR6", "cr6_slow_carry", lambda inst: adopt(SlowCarry, inst), "a slower carry over",
          isaac="passes: 4/4; a shallow difference")
