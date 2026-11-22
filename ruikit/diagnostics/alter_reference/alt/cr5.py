"""CR5: the fingers across the other way when they press.  The reference presses with the fingers'
width across the brick; a square brick takes the same press a quarter turn on, and the scorer asks
only that each brick be home.  A long brick stays as the reference has it (its studs are not square
to the fingers the other way)."""

from __future__ import annotations

import math
from typing import List

from ruibench.core.geometry import yaw_of
from ruibench.core.types import Pose
from ruibench.interfaces.commands import Command, MoveTo
from ruibench.policies.scripted import turned_down
from ruibench.tasks.impl.cr5_bricks import LONG, BuildPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class QuarterPress(BuildPolicy):
    name = "press_bricks_quarter_press"

    def _press(self, world, name: str) -> List[Command]:
        cmds = super()._press(world, name)
        if LONG[name] != 2:
            return cmds
        out = []
        for c in cmds:
            if isinstance(c, MoveTo):
                turned = turned_down(yaw_of(c.pose.quat) + math.pi / 2)
                c = MoveTo(c.arm, Pose(c.pose.pos, turned))
            out.append(c)
        return out


WAY = Way("CR5", "cr5_quarter_press", lambda inst: adopt(QuarterPress, inst),
          "square bricks pressed a quarter turn on", isaac="passes: 4/4")
