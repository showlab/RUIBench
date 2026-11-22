"""DI6: the cover set aside on the table's other side.  The same cover is lifted: which one it is
stays the reference's knowledge, so this is a shallow difference."""

from __future__ import annotations

import math

from ruibench.interfaces.commands import Wait
from ruibench.tasks.impl.di6_carousel import DISC_R, TABLE, TABLE_AT, FollowPolicy, ahead
from ruikit.diagnostics.alter_reference import Way, adopt


class OtherSide(FollowPolicy):
    name = "follow_other_side"

    def _act(self, world, ctx):
        if self._phase != "cover":
            return super()._act(world, ctx)
        c = world.obj(self._cover)
        if not c.is_held and math.dist(c.pose.pos[:2], ahead(*TABLE_AT)) > DISC_R + 0.02:
            self._phase, self._sub = "cake", None
            return [Wait(0.3)]
        aside = ahead(TABLE_AT[0] - 0.02, TABLE_AT[1] - DISC_R - 0.07)
        return self._carry(world, ctx, self._cover, (aside[0], aside[1], TABLE))


WAY = Way("DI6", "di6_other_side", lambda inst: adopt(OtherSide, inst), "the cover set aside on the other side",
          isaac="passes: 4/4; a shallow difference")
