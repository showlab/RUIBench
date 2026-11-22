"""CL2: the dolls in the reverse order at the hard level, carried higher.  At the easy level only the
carrying height differs: there is one doll to put to bed."""

from __future__ import annotations

from ruibench.tasks.impl.cl2_names import NameBinder
from ruikit.diagnostics.alter_reference import Way, adopt


class OtherOrder(NameBinder):
    name = "name_binder_other_order"

    def commands(self, world, ctx):
        from ruibench.policies.scripted import SequentialPickAndPlacePolicy

        if self._inner is None:
            self._inner = SequentialPickAndPlacePolicy(list(reversed(self._moves(world, ctx))), arm=self.arm,
                                                       lift_height=0.16)
        return self._inner.commands(world, ctx)


WAY = Way("CL2", "cl2_other_order", lambda inst: adopt(OtherOrder, inst),
          "dolls in the reverse order, carried higher", isaac="passes: 4/4; on easy only the height differs")
