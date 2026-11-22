"""WK3: uneven gaps.  The reference spaces the dominoes by its route; this one moves every other one
a little along the chain, a tenth of the way toward its next neighbour.  The scorer asks only that
the bell ring; the chain may not carry on every seed."""

from __future__ import annotations

from ruibench.tasks.impl.wk3_dominoes import DominoPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class UnevenGaps(DominoPolicy):
    name = "dominoes_uneven_gaps"
    SHARE = 0.1


def _make(inst):
    policy = adopt(UnevenGaps, inst)
    moved = list(policy.places)
    for i in range(1, len(moved) - 1, 2):
        x, y, yaw = moved[i]
        nx, ny, _ = moved[i + 1]
        moved[i] = (x + (nx - x) * policy.SHARE, y + (ny - y) * policy.SHARE, yaw)
    policy.places = moved
    return policy


WAY = Way("WK3", "wk3_uneven_gaps", _make, "every other domino moved a little along the chain",
          isaac="passes: 4/4")
