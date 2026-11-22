"""DI5: the hand stays over the hole it just struck instead of going home, and hops to the next from
there.  A head coming down over a mole counts from wherever it came (``Moles.tick``).

Not the DI5 way the training data used: at 8 s a reply a turn-by-turn policy swings at moles already
gone.  The data's DI5 is a program that watches and a reply that whacks."""

from __future__ import annotations

from ruibench.interfaces.commands import Done, MoveTo
from ruibench.tasks.impl.di5_whack_a_mole import WhackPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class Hop(WhackPolicy):
    name = "whack_hop"

    def commands(self, world, ctx):
        cmds = super().commands(world, ctx)
        if any(isinstance(c, Done) for c in cmds) or self._rested:
            return cmds
        moves = [c for c in cmds if isinstance(c, MoveTo)]
        if not moves:
            return cmds
        # the reference's three moves an arm: over, down, home.  Home becomes back up over the hole
        per_arm = {}
        for c in moves:
            per_arm.setdefault(c.arm, []).append(c)
        out = []
        for c in cmds:
            if isinstance(c, MoveTo) and len(per_arm[c.arm]) == 3 and c is per_arm[c.arm][2]:
                c = per_arm[c.arm][0]
            out.append(c)
        return out


WAY = Way("DI5", "di5_hop", lambda inst: adopt(Hop, inst), "stays over the last hole between moles")
