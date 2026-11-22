"""CL1: the tune at another tempo, the bars held shorter and no rest between them.  The scorer asks
for the notes in order, not for a tempo.  A shallow difference: the same bars, struck the same way."""

from __future__ import annotations

from ruibench.tasks.impl.cl1_play_it_back import TunePolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class QuickTempo(TunePolicy):
    name = "play_it_back_quick"


def _make(inst):
    policy = adopt(QuickTempo, inst)
    policy.hold, policy.rest = 0.25, 0.1
    return policy


WAY = Way("CL1", "cl1_quick_tempo", _make, "another tempo", isaac="passes: 4/4; a shallow difference")
