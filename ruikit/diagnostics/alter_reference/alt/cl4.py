"""CL4: the stones nearest the well first, dropped round the well from the other side.  The reference
takes them by name; the scorer asks only that the ball end up in the dish."""

from __future__ import annotations

import math
from typing import List, Optional

from ruibench.core.types import Pose
from ruibench.interfaces.commands import MoveTo, SetGripper, Wait
from ruibench.tasks.impl.cl4_pitcher import (
    CARRY_Z,
    GRASP_DOWN,
    TABLE,
    WELL_H,
    WELL_R,
    PitcherPolicy,
    pebble_name,
    pebbles,
    proud,
    well_xy,
)
from ruikit.diagnostics.alter_reference import Way, adopt


class NearestStoneFirst(PitcherPolicy):
    name = "pitcher_nearest_stone_first"

    def reset(self) -> None:
        super().reset()
        self._order: Optional[List[str]] = None

    def commands(self, world, ctx):
        from ruibench.policies.scripted import grasp_pose

        if ctx.extras.get("executor_busy"):
            return []
        if self._phase == "ball" or proud(world) or not self._watched(ctx):
            return super().commands(world, ctx)
        wx, wy = well_xy()
        if self._order is None:
            names = [pebble_name(k) for k in range(pebbles(self.level))]
            self._order = sorted(names, key=lambda n: math.dist(world.obj(n).pose.pos[:2], (wx, wy)))
        # as the reference goes on: with stones left the ball stands or it does not
        if self._k >= len(self._order):
            return super().commands(world, ctx)
        name = self._order[self._k]
        if f"{name}_in_the_well" in world.events_fired:
            self._k += 1
            return [Wait(0.3)]
        obj = world.obj(name)
        if not obj.is_held:
            g = grasp_pose(obj)
            return [SetGripper(self.arm, 0.0),
                    MoveTo(self.arm, Pose((g.pos[0], g.pos[1], CARRY_Z), g.quat)),
                    MoveTo(self.arm, g), SetGripper(self.arm, 1.0), Wait(0.4)]
        angle = math.pi + 2 * math.pi * self._k / max(1, len(self._order))
        off = WELL_R - 0.024
        over = (wx + off * math.cos(angle), wy + off * math.sin(angle), TABLE + WELL_H + 0.048)
        hand = world.arm(self.arm).eef_pose.pos
        return [MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), GRASP_DOWN)),
                MoveTo(self.arm, Pose(over, GRASP_DOWN)), SetGripper(self.arm, 0.0), Wait(1.0)]


WAY = Way("CL4", "cl4_nearest_stone_first", lambda inst: adopt(NearestStoneFirst, inst),
          "stones by distance to the well, dropped from the other side", isaac="passes: 4/4")
