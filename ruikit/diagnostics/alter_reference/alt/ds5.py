"""DS5: the beakers tested in the reverse order, so a reading comes first from the other end and
elimination, when it decides, falls on another beaker."""

from __future__ import annotations

from ruibench.tasks.impl.ds5_litmus import LitmusPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class ReverseTests(LitmusPolicy):
    name = "litmus_reverse_tests"


def _make(inst):
    policy = adopt(ReverseTests, inst)
    policy.lets = list(reversed(policy.lets))
    return policy


WAY = Way("DS5", "ds5_reverse_tests", _make, "tests the beakers from the other end", isaac="passes: 4/4")
