"""DI4, Simon says, on the real station: the program says the calls, and keeps the score.

The calls are simulation's: for a seed, the same which-toy and with-or-without "Simon says" at
the same times (``impl.di4_simon_says.schedule``), the toys the card's, in the order of simulation's
own.  They come by themselves, nobody says anything, and the agent reads them as it does there: as
the instruction changing, with its next tool result.  Three things differ, for a real robot:

* the game's clock starts at the agent's first call: an agent's program takes seconds to start, and
  simulation's two T_ref to look at the toys are for looking;
* a call the agent thought through is still said, each in its turn, not skipped: nothing ticks here
  while it thinks, where in simulation the thinking itself is ticked through;
* nothing knows where the toys are but the card, so a touch is the hand over the toy's spot on the
  table, down to within reach of its top, a little more widely than simulation has it: the
  operator's own verdict is the result, and this record is what the operator is shown.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

from ...core.config import DEFAULT_T_REF_SECONDS
from ...core.types import WorldState
from ...engines.base import Engine
from ...engines.franka.steps import MARK
from ...events.actions import SetInstruction
from ...scoring import Custom, Ever, Finally, Scorer
from ..impl.di4_simon_says import (
    CALLS, GAME_OVER, LEVEL_TOYS, SIMON, TOUCH_ABOVE, Simon, caught_out, obeyed, schedule,
)

#: how far round its spot a toy can be touched: laid out by hand, it is not where the card says
SPOT_MARGIN_M = 0.02


@dataclass(frozen=True)
class Toy:
    """What the calls call a toy, and where it stands (station frame, metres)."""
    name: str
    called: str
    at: Tuple[float, float]
    half: Tuple[float, float]
    top: float

    def touched_by(self, tcp: Sequence[float]) -> bool:
        return (abs(tcp[0] - self.at[0]) <= self.half[0] + SPOT_MARGIN_M
                and abs(tcp[1] - self.at[1]) <= self.half[1] + SPOT_MARGIN_M
                and tcp[2] <= self.top + TOUCH_ABOVE)


def calls_of(toys: Sequence[Toy], level: str, seed: int) -> List[Tuple[float, bool, str]]:
    """Simulation's calls for this seed, each of its toys standing for one of the card's."""
    named = {sim: toy.name for sim, toy in zip(LEVEL_TOYS[level], toys)}
    return [(when, simon, named[toy]) for when, simon, toy in schedule(level, seed)]


def call_text(simon: bool, toy: Toy) -> str:
    body = f"touch {toy.called}."
    return SIMON + body if simon else body[0].upper() + body[1:]


class RealSimon(Simon):
    """Says the calls, and records every touch against the call it fell in (the module docstring)."""

    def __init__(self, calls, toys: Sequence[Toy], gap: float) -> None:
        super().__init__(calls, [t.name for t in toys], gap)
        self.real_toys = {t.name: t for t in toys}
        self._began = None

    def tick(self, engine: Engine, world: WorldState) -> None:
        if self._began is None:
            self._began = world.sim_time
        t = world.sim_time - self._began
        k = self.current(t)
        if k > self._said:                   # one a tick: each is heard, in its turn
            self._said += 1
            _, simon, name = self.calls[self._said]
            text = call_text(simon, self.real_toys[name])
            self.act(engine, world, f"call_{self._said}", SetInstruction(text))
            print(f"{MARK}{t:6.1f} s  the program calls: {text}", flush=True)
        tcp = world.arm(world.arm_ids[0]).eef_pose.pos
        for name, toy in self.real_toys.items():
            now = toy.touched_by(tcp)
            if now and not self._in.get(name) and k >= 0 and not self._over:
                self.note(engine, world, f"touched_{k}_{name}")
                print(f"{MARK}{t:6.1f} s  the hand touches {toy.called}", flush=True)
            self._in[name] = now
        if not self._over and t >= self.calls[-1][0] + self.gap:
            self._over = True
            self.note(engine, world, GAME_OVER)
            self.say(world, "simon", "done", "That's the end of the game.")
            print(f"{MARK}{t:6.1f} s  the game is over", flush=True)


def simon_game(task_id: str, toys: Sequence[Toy], level: str, seed: int):
    """The scorer and the reactor of a Simon says game on these toys."""
    calls = calls_of(toys, level, seed)
    gap = CALLS[level][1] * DEFAULT_T_REF_SECONDS
    won = Custom(lambda w: GAME_OVER in w.events_fired and obeyed(w, calls)
                 and not caught_out(w, calls), "every Simon call done, no other")
    # no violation ends it early, as in simulation: where a toy is is known only roughly here
    scorer = Scorer(task_id=task_id, level=level, success=Finally(won) & Ever(Custom(
        lambda w: obeyed(w, calls) and GAME_OVER not in w.events_fired,
        "every Simon call done before the end")))
    return scorer, lambda: RealSimon(calls, toys, gap)


__all__ = ["RealSimon", "Toy", "call_text", "calls_of", "simon_game"]
