"""CR2: short strokes.  The reference turns the bulb as far as is left or the wrist goes in one
stroke; this one turns it at most 55 degrees a stroke, so it lets go and takes hold again in between.
The scorer counts the bulb's net turn, however many strokes make it.  For the easy level's quarter
turn: the two turns of the hard level would pass the policy's limit of 8 strokes."""

from __future__ import annotations

import math

from ruibench.tasks.impl.cr2_bulb import ScrewPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class ShortStrokes(ScrewPolicy):
    name = "screw_in_short_strokes"
    CAP = math.radians(55)

    def _left(self) -> float:
        return min(super()._left(), self.CAP)


WAY = Way("CR2", "cr2_short_strokes", lambda inst: adopt(ShortStrokes, inst),
          "more, shorter strokes", isaac="passes: 4/4, one grip more than the reference")
