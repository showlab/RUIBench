"""DS4 "the fake coin" -- Discovery.

Knob: how many coins there are to tell apart (3 -> 9).

Gold coins that look the same -- a row of three, or three rows of three --
a toy balance and a small dish.  "One of
the coins is fake -- it is lighter.  Put the fake one in the dish."  Nothing
in any picture says which: only the balance does.  Three coins take one
weighing (one against one: the lighter side, or the one left out); nine take
two (three against three, then one against one of the lighter three).  More
weighings than twice the fewest (``ALLOWANCE``) is a violation, so trying
every pair is not a way round; a real coin in the dish, at any moment, is a
failure (v2.3: before, only the last picture was looked at, and coins could
be tried in the dish one by one).

The balance is a toy pan balance whose pans sit on top and stay level, one
going down as the other goes up, the beam between them showing which way.
What it shows is worked out from what lies in the pans (``Balance``), so where
a coin sits in a pan does not matter and a few grams always tip it.  A coin
still in the hand weighs nothing yet.

Which coin is fake is drawn so that every place is fake exactly
once in any n consecutive seeds: a guess is right one time in n.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import Pose, WorldState
from ...engines.base import Engine
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, In, KnownBefore, Milestone, Never, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

BEAM, HINGE, DISH = "beam", "balance_hinge", "dish"
SLIDES = {-1: "pan_left_slide", 1: "pan_right_slide"}
BALANCE_AT = (0.50, 0.0)
#: where the coins lie, in reading order; 10 cm apart across the fixed camera's
#: view, room for the open fingers beside a coin 4.5 cm across its corners (they
#: close across the view: turned along it, the fingers hid a held coin from the
#: camera).  The balance stands behind them, where its pans hide none of them
COLUMNS_Y = (-0.10, 0.0, 0.10)
ROWS_X = {3: (0.20,), 9: (0.27, 0.20, 0.13)}
DISH_AT = (0.30, -0.30)
#: the dish: a small white porcelain bowl (v8; a yellow toy plate until then),
#: 12 cm across and 3 cm deep, over a hidden floor and ring
DISH_R, DISH_HZ = 0.06, 0.015
COIN_MASS, FAKE_MASS = 0.012, 0.007
#: how many weighings count as fair: twice the fewest
ALLOWANCE = {3: 2, 9: 4}
#: the beam turns at most this fast (rad/s), the pans go with it
TURN_RATE = 0.25
#: a reading counts once the pans have not changed for this long, nothing in the hand
SETTLE = 1.0
#: where coins lie in a pan: side by side along x, the fingers closing across
#: the row (stacked, they came apart on the mock as the pan moved)
PAN_SLOTS = (0.0, -0.045, 0.045)
FRAME = Frame((0.025, 0.600), (-0.380, 0.380), 0.16)


def fake_index(seed: int, n: int) -> int:
    """Every place fake exactly once in any n consecutive seeds."""
    group, place = divmod(seed, n)
    order = list(range(n))
    SeedBundle(group).stream(f"fake/{n}").shuffle(order)
    return order[place]


def pivot() -> Tuple[float, float, float]:
    x, y = ahead(*BALANCE_AT)
    return (x, y, T.BALANCE_PIVOT_Z)


def build_scene(n: int, fake: int):
    coins = [f"coin{k}" for k in range(n)]
    homes = [ahead(x, y) for x in ROWS_X[n] for y in COLUMNS_Y]
    bodies = [T.gold_coin(name, xy, TABLE, mass=FAKE_MASS if k == fake else COIN_MASS)
              for k, (name, xy) in enumerate(zip(coins, homes))]
    beam, hinge, pans, slides, stand = T.balance("", ahead(*BALANCE_AT), HINGE,
                                                 (SLIDES[-1], SLIDES[1]))
    bodies += [beam, *pans, stand]
    bodies.append(dish(DISH, ahead(*DISH_AT)))
    return station_scene(f"ds4_fake_coin_{n}", bodies, frame=FRAME,
                         articulations=[hinge, *slides]), \
        coins, dict(zip(coins, homes))


def dish(name: str, xy) -> "BodySpec":
    """A porcelain bowl: drawn round, what a coin meets a hidden floor and a
    ring of hidden walls inside it."""
    from ...engines import parts as P
    from ...engines.base import BodySpec

    r, hz = DISH_R, DISH_HZ
    floor = -hz + 0.16 * 2 * hz                       # the bowl's inside floor (``parts._bowl_profile``)
    parts = [P.cylinder((0, 0, (-hz + floor) / 2), 0.55 * r, (floor + hz) / 2, "grey", collides="hidden")]
    parts += [q[:5] + ("hidden",) for q in P.ring((0, 0, 0), 0.9 * r, 0.9 * r - 0.004, hz, "grey",
                                                  segments=12)]
    parts.append(P.lathe((0, 0, 0), "bowl", r, hz, "ceramic:white"))
    return BodySpec(name, "compound", (r, r, hz), Pose((xy[0], xy[1], TABLE + hz)),
                    category="dish", color="ceramic:white", fixed=True,
                    attrs={"parts": tuple(parts), "container": "open", "wall": 0.004,
                           "graspable": False, "footprint": "round"})


def held_things(world: WorldState) -> set:
    return {a.holding for r in world.robots.values() for a in r.arms.values()}


def pan_contents(world: WorldState, coins: Sequence[str]) -> Dict[int, frozenset]:
    """The coins lying in each pan (-1: the -y pan, +1: the +y pan); a coin in
    the hand is in neither."""
    held = held_things(world)
    out = {}
    for side in (-1, 1):
        cx, cy, cz = T.pan_centre(pivot(), world.articulations.get(SLIDES[side], 0.0), side)
        out[side] = frozenset(
            c for c in coins if c not in held
            and math.hypot(world.obj(c).pose.pos[0] - cx, world.obj(c).pose.pos[1] - cy) < T.PAN_RADIUS
            and -0.03 < world.obj(c).pose.pos[2] - cz < 0.06)
    return out


class Balance(Reactor):
    """Tips the beam toward the heavier pan, turning at ``TURN_RATE``; counts
    the weighings.  A weighing is a settled reading with coins in both pans.
    Coins added to the last one's pans, or taken off them, are still that
    weighing; anything else -- a coin that was not in it, a coin moved to the
    other pan, after some were taken off -- is a new one."""

    def __init__(self, coins: Sequence[str], masses: Dict[str, float]) -> None:
        super().__init__()
        self.coins, self.masses = list(coins), dict(masses)
        self.weighings = 0
        self._last: Optional[Dict[int, frozenset]] = None
        self._since: Optional[float] = None
        self._seen: Optional[Dict[int, frozenset]] = None
        self._command = 0.0
        self._t = 0.0

    def tick(self, engine: Engine, world: WorldState) -> None:
        pans = pan_contents(world, self.coins)
        weight = {side: sum(self.masses[c] for c in pans[side]) for side in pans}
        target = 0.0
        if weight[-1] > weight[1] + 1e-6:
            target = T.BALANCE_TILT          # the -y pan down, the +y pan up
        elif weight[1] > weight[-1] + 1e-6:
            target = -T.BALANCE_TILT
        step = TURN_RATE * max(world.sim_time - self._t, 0.0)
        self._t = world.sim_time
        self._command += max(-step, min(step, target - self._command))
        engine.set_articulation_target(HINGE, self._command)
        lift = T.BALANCE_ARM * math.sin(self._command)
        engine.set_articulation_target(SLIDES[1], lift)
        engine.set_articulation_target(SLIDES[-1], -lift)

        # the weighings
        if pans != self._seen:
            self._seen, self._since = pans, world.sim_time
        held = bool(held_things(world) & set(self.coins))
        settled = (not held and pans[-1] and pans[1]
                   and world.sim_time - self._since >= SETTLE)
        if not settled or pans == self._last:
            return
        last = self._last
        if last is not None and all(pans[s] <= last[s] for s in pans):
            return                                    # coins taken off: still that weighing
        self._last = pans
        if last is not None and all(last[s] <= pans[s] for s in pans):
            return                                    # coins added to it: still that weighing
        self.weighings += 1
        self.note(engine, world, f"weighing_{self.weighings}",
                  f"{sorted(pans[-1])} against {sorted(pans[1])}")


def found_by_weighings(coins: Sequence[str], masses: Dict[str, float]):
    """What the balance has shown so far leaves open for the light coin: each
    settled reading (coins in both pans, none in a hand, ``SETTLE`` seconds alike,
    as ``Balance`` counts them) keeps the coins that, were they the light one,
    would tip the beam the way it tipped.  A balanced weighing of two rules out
    both and leaves the third (``KnownBefore``)."""
    fake = min(coins, key=lambda c: masses[c])

    def tilt(pans, light: str) -> int:
        m = {c: FAKE_MASS if c == light else COIN_MASS for c in coins}
        d = sum(m[c] for c in pans[-1]) - sum(m[c] for c in pans[1])
        return (d > 1e-6) - (d < -1e-6)

    def narrow(w: WorldState, left: frozenset, memo: dict) -> frozenset:
        pans = pan_contents(w, coins)
        if pans != memo.get("pans"):
            memo["pans"], memo["since"] = pans, w.sim_time
        settled = (pans[-1] and pans[1] and not (held_things(w) & set(coins))
                   and w.sim_time - memo["since"] >= SETTLE)
        if not settled or pans == memo.get("read"):
            return left
        memo["read"] = pans
        shown = tilt(pans, fake)
        return frozenset(c for c in left if tilt(pans, c) == shown)
    return narrow


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DS4")
    n = int(spec.knob.levels[level])
    fake_at = fake_index(seed, n)
    scene, coins, homes = build_scene(n, fake_at)
    fake = coins[fake_at]
    masses = {c: FAKE_MASS if c == fake else COIN_MASS for c in coins}
    allowance = ALLOWANCE[n]
    in_dish = {c: In(c, DISH) for c in coins}

    def weighed(k: int):
        return Custom(lambda w, k=k: f"weighing_{k}" in w.events_fired, f"{k} weighing(s)")

    done = Custom(lambda w: in_dish[fake].holds(w) and AtRest(fake).holds(w)
                  and not any(in_dish[c].holds(w) for c in coins if c != fake),
                  "the fake coin, and only it, at rest in the dish")
    real_in = Custom(lambda w: any(in_dish[c].holds(w) for c in coins if c != fake),
                     "a real coin in the dish")
    milestones = (
        Milestone("first_weighing", weighed(1)),
        Milestone("a_coin_in_the_dish", Custom(lambda w: any(in_dish[c].holds(w) for c in coins),
                                               "a coin in the dish")),
        Milestone("fake_in_dish", done),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="One of the coins is fake -- it is lighter. Put the fake one in the dish.",
        scorer=Scorer(task_id="DS4", level=level, success=Finally(done), milestones=milestones,
                      violations=(("weighed_within_allowance", Never(weighed(allowance + 1))),
                                  # the light coin into the dish only once the weighings had
                                  # found it: the first coin in, blind, was right one time in n
                                  ("fake_found_first",
                                   KnownBefore(fake, tuple(coins), found_by_weighings(coins, masses),
                                               in_dish[fake], "which coin is the light one")),
                                  ("no_real_coin_in_the_dish", Never(real_in)))),
        milestones=milestones,
        reactor_factory=lambda c=tuple(coins), m=masses: Balance(c, m),
        reference_factory=lambda inst, c=tuple(coins), h=homes: WeighPolicy(c, h),
        metadata={"knob": spec.knob.field, "knob_value": n, "allowance": allowance,
                  # hidden from the agent; here so the gap-filled control can name it
                  "fake": fake, "max_model_calls": 40 + 12 * n},
        extra_metrics={"weighings": lambda traj: sum(
            1 for k in range(1, 20) if traj and f"weighing_{k}" in traj[-1].events_fired)},
    )


class WeighPolicy(ScriptedPolicy):
    """The classic way: three against three, then one against one (nine
    coins); one against one (three).  Coins go into a pan side by side, come
    back to their own place afterwards, and are told apart only by that
    place.  The beam is read off its angle, as the cameras show it."""

    name = "weigh"

    def __init__(self, coins: Sequence[str], homes, arm: str = RIGHT) -> None:
        self.coins, self.homes, self.arm = list(coins), dict(homes), arm
        self.reset()

    def reset(self) -> None:
        self._candidates = list(self.coins)
        self._jobs: List[tuple] = []
        self._fake: Optional[str] = None
        self._on_pan: Dict[int, List[str]] = {-1: [], 1: []}
        self._waited = False

    def _split(self) -> Tuple[List[str], List[str], List[str]]:
        k = len(self._candidates) // 3 if len(self._candidates) > 3 else 1
        c = self._candidates
        return c[:k], c[k:2 * k], c[2 * k:]

    def _move(self, world: WorldState, coin: str, where) -> List[Command]:
        from ...policies.scripted import _pick_and_place_sequence

        obj = world.obj(coin)
        if where == "home":
            x, y = self.homes[coin]
            target, height = (x, y, TABLE), 0.0
        elif where == "dish":
            dish = world.obj(DISH)
            target, height = dish.pose.pos, dish.size[2]
        else:
            x, y, z = T.pan_centre(pivot(), world.articulations.get(SLIDES[where], 0.0), where)
            target, height = (x + PAN_SLOTS[len(self._on_pan[where])], y, z), 0.0
        cmds = _pick_and_place_sequence(self.arm, obj, target, height, 0.08, 0.004)
        for side in self._on_pan:
            if coin in self._on_pan[side]:
                self._on_pan[side].remove(coin)
        if where in (-1, 1):
            self._on_pan[where].append(coin)
        return cmds

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._jobs:
            kind, *args = self._jobs.pop(0)
            if kind == "move":
                return self._move(world, *args)
            return self._read(world)
        if self._fake is not None:
            if world.obj(self._fake).pose.pos[2] < TABLE + 0.05 and \
                    In(self._fake, DISH).holds(world):
                return [Done(f"{self._fake} is the fake")]
            return self._move(world, self._fake, "dish")
        if any(self._on_pan.values()):              # clear the pans first
            self._jobs = [("move", c, "home") for s in (-1, 1) for c in reversed(self._on_pan[s])]
            return self.commands(world, ctx)
        left, right, _ = self._split()
        self._jobs = ([("move", c, -1) for c in left] + [("move", c, 1) for c in right]
                      + [("read",)])
        return self.commands(world, ctx)

    def _read(self, world: WorldState) -> List[Command]:
        """Wait for the beam, then keep the lighter group (or the rest)."""
        angle = world.articulations.get(HINGE, 0.0)
        if abs(angle) < 0.6 * T.BALANCE_TILT and self._jobs == [] and not self._waited:
            self._waited = True
            self._jobs = [("read",)]
            return [Wait(2.0)]
        self._waited = False
        left, right, rest = self._split()
        if angle > 0.6 * T.BALANCE_TILT:            # the +y pan up: lighter there
            self._candidates = right
        elif angle < -0.6 * T.BALANCE_TILT:
            self._candidates = left
        else:
            self._candidates = rest
        if len(self._candidates) == 1:
            self._fake = self._candidates[0]
        return [Wait(0.5)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: no weighing, the first coin into the dish -- right one time
        in n."""
        return self._move(world, self.coins[0], "dish") + [Done("guessed")]


def secrets(inst) -> Tuple[str, ...]:
    """The light coin (``TaskInstance.secrets``)."""
    return (inst.metadata["fake"],)
