"""DI4: back up to the hover over the middle of the bench after a touch, where the reference stays
over the toy it touched.  Going home between calls brushes no toy on the way to the next.

Not the DI4 way the training data used: at 8 s a reply a turn-by-turn policy answers a call too
late, as the reference does.  The data's DI4 is a program that listens and a reply that taps."""

from __future__ import annotations

from ruibench.core.types import Pose
from ruibench.interfaces.commands import MoveTo
from ruibench.tasks.impl.di4_simon_says import SimonPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class HomeBetween(SimonPolicy):
    name = "simon_home_between"

    def commands(self, world, ctx):
        lifting = self._lift
        cmds = super().commands(world, ctx)
        if lifting is not None and cmds:
            names = [t for t in self.toys]
            xs = [world.obj(t).pose.pos for t in names]
            mid = (sum(p[0] for p in xs) / len(xs), sum(p[1] for p in xs) / len(xs), lifting.pos[2])
            cmds = cmds + [MoveTo(self.arm, Pose(mid, lifting.quat))]
        return cmds


WAY = Way("DI4", "di4_home_between", lambda inst: adopt(HomeBetween, inst),
          "back over the middle of the bench between calls")
