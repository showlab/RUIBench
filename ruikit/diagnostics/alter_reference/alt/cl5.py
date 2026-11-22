"""CL5: each place gets the farthest free piece of its kind, where the reference takes the nearest.
The scorer takes any piece of the right kind in a place."""

from __future__ import annotations

import math

from ruibench.tasks.impl.cl5_tangram import CARD_TOP, PIECES, SYMMETRY, TangramPolicy, fill_order, places
from ruikit.diagnostics.alter_reference import Way, adopt


class FarthestPiece(TangramPolicy):
    name = "tangram_farthest_piece"

    def _plan(self, world, ctx) -> None:
        if not self._watched(ctx):
            return super()._plan(world, ctx)
        free = {name: kind for name, kind, _ in PIECES}
        spots = places(self.shape)
        for kind, (x, y), yaw in (spots[i] for i in fill_order(spots)):
            name = max((n for n, k in free.items() if k == kind),
                       key=lambda n: math.dist(world.obj(n).pose.pos[:2], (x, y)))
            free.pop(name)
            self.queue.append((name, (x, y, CARD_TOP), yaw, SYMMETRY[kind]))


WAY = Way("CL5", "cl5_farthest_piece", lambda inst: adopt(FarthestPiece, inst),
          "the farthest piece of a kind for each place", isaac="passes: 4/4")
