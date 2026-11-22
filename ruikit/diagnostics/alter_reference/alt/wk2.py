"""WK2: another safe crossing.  The reference's search tries the loads in sorted order, which fixes
its plan; trying them in the reverse order lands on another shortest plan (easy: wolf, bucket, goat
instead of goat, bucket, wolf)."""

from __future__ import annotations

from collections import deque

from ruibench.tasks.impl.wk2_river import FAR, NEAR, FerrymanPolicy, safe
from ruikit.diagnostics.alter_reference import Way, adopt


def other_crossing(names, rules) -> list:
    start, goal = (NEAR, frozenset()), (FAR, frozenset(names))
    prev = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        if state == goal:
            break
        farmer, far = state
        here = far if farmer == FAR else frozenset(names) - far
        for load in [None] + sorted(here, reverse=True):
            moved = far if load is None else (far - {load} if farmer == FAR else far | {load})
            nxt = (NEAR if farmer == FAR else FAR, frozenset(moved))
            if nxt not in prev and safe(nxt, names, rules):
                prev[nxt] = (state, load)
                queue.append(nxt)
    plan, state = [], goal
    while prev.get(state):
        state, load = prev[state]
        plan.append(load)
    return plan[::-1]


class OtherCrossing(FerrymanPolicy):
    name = "ferryman_other_crossing"

    def reset(self) -> None:
        super().reset()
        self._plan = other_crossing(self.names, self.rules)


WAY = Way("WK2", "wk2_other_crossing", lambda inst: adopt(OtherCrossing, inst),
          "another shortest safe plan", isaac="passes: 4/4")
