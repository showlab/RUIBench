"""CL6: another way to the king.  The reference takes the shortest route every turn; this one's first
move is another legal one from which the king can still be reached, the hand clear at both ends, and
from there it goes the shortest way.  Nothing limits the number of moves.  The route it lands on is
not always longer (seed 12: h3, f4, g6, h8 against the reference's f2, g4, e5, g6, h8)."""

from __future__ import annotations

from typing import Optional

from ruibench.tasks.impl.cl6_chess import KING, KNIGHT, KNIGHT_STEPS, KnightPolicy, on_board, route, square_of, turn_for
from ruikit.diagnostics.alter_reference import Way, adopt


class LongerWay(KnightPolicy):
    name = "knight_longer_way"

    def reset(self) -> None:
        super().reset()
        self._went_round = False

    def _next(self, world, at) -> Optional[object]:
        best = super()._next(world, at)
        if self._went_round or best is None:
            return best
        self._went_round = True
        king = world.obj(KING)
        now = square_of(king.pose.pos) if on_board(king.pose.pos) else None
        if now is None:
            return best
        other = self.kings[1] if now == self.kings[0] else self.kings[0]
        taken = {s for n, s in self.where.items() if n not in (KNIGHT, KING)}
        where = {n: s for n, s in self.where.items() if n not in (KNIGHT, KING)}
        where[KNIGHT], where[KING] = at, now
        for dx, dy in KNIGHT_STEPS:
            to = (at[0] + dx, at[1] + dy)
            if to in (best, now, other) or to in taken:
                continue
            legal = route(at, (to, to), taken, hobbled=True, moving=False)
            onward = route(to, (other, now), taken, hobbled=True)
            if legal and legal[1] == to and onward and turn_for(at, to, where):
                return to
        return best


WAY = Way("CL6", "cl6_longer_way", lambda inst: adopt(LongerWay, inst),
          "a first move off the shortest route",
          isaac="weak: 6/8 without errors; with errors it failed 5 times on code-joint")
