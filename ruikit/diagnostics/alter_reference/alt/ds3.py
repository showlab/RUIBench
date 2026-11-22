"""DS3: the shells swept in the reverse order; the sweep stops at the first hot reading, so it covers
different shells from the reference's."""

from __future__ import annotations

from ruibench.tasks.impl.ds3_treasure import TreasurePolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class ReverseSweep(TreasurePolicy):
    name = "treasure_reverse_sweep"


def _make(inst):
    policy = adopt(ReverseSweep, inst)
    policy.names = list(reversed(policy.names))
    return policy


WAY = Way("DS3", "ds3_reverse_sweep", _make, "sweeps the shells from the other end", isaac="passes: 4/4")
