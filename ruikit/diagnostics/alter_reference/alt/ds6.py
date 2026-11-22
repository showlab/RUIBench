"""DS6: the splints tried from the robot's right, where the reference goes from its left."""

from __future__ import annotations

from typing import Optional

from ruibench.tasks.impl.ds6_flame import SALTS, FlameTestPolicy, splint_name
from ruikit.diagnostics.alter_reference import Way, adopt


class FromTheRight(FlameTestPolicy):
    name = "flame_test_from_the_right"

    def _next(self, world, colour: str) -> Optional[str]:
        names = sorted((splint_name(s) for s in SALTS), key=lambda n: world.obj(n).pose.pos[1])
        left = [n for n in names if n not in self._used]
        known = [n for n in left if self._burns.get(n) == colour]
        if known:
            return known[0]
        fresh = [n for n in left if n not in self._burns]
        return fresh[0] if fresh else None


WAY = Way("DS6", "ds6_from_the_right", lambda inst: adopt(FromTheRight, inst),
          "tries the splints from the other side", isaac="passes: 4/4")
