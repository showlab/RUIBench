"""CR1: the blocks in the reverse order, the cross first.  The scorer asks only that each go through
its own hole."""

from __future__ import annotations

from ruibench.tasks.impl.cr1_shape_sorter import SorterPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class CrossFirst(SorterPolicy):
    name = "shape_sorter_cross_first"
    ORDER = tuple(reversed(SorterPolicy.ORDER))


WAY = Way("CR1", "cr1_cross_first", lambda inst: adopt(CrossFirst, inst), "blocks in the reverse order",
          isaac="passes: 4/4")
