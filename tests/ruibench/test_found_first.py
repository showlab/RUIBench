"""DS3, DS4, DS5: the hidden answer found out before it is acted on (``KnownBefore``).
A test that tells the answer, or rules every other candidate out, counts; acting
first does not, even when the guess is right."""

from dataclasses import replace
from types import SimpleNamespace

from ruibench.engines.mock import MockEngine
from ruibench.scoring import Custom, KnownBefore
from ruibench.tasks.instance import build
from ruibench.core.config import RunConfig

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def snaps(*steps):
    """Snapshots for the predicate alone: what the world showed, and whether the act came."""
    return [SimpleNamespace(shown=shown, acted=acted, events_fired=()) for shown, acted in steps]


def rule(answer, candidates):
    def narrow(w, left, memo):
        for c in w.shown:
            left = frozenset((c,)) if c == answer else left - {c}
        return left
    return KnownBefore(answer, tuple(candidates), narrow, Custom(lambda w: w.acted, "acted"), "the answer")


def test_known_before_asks_the_answer_found_or_the_rest_ruled_out_before_the_act():
    r = rule("b", "abc")
    assert r.evaluate(snaps(((), False), (("b",), False), ((), True)))            # tested itself
    assert r.evaluate(snaps((("a",), False), (("c",), False), ((), True)))        # the others ruled out
    assert r.evaluate(snaps((("a", "c"), True)))                                  # in the same snapshot
    assert not r.evaluate(snaps(((), True), (("b",), False)))                     # acted, then tested
    assert not r.evaluate(snaps((("a",), False), ((), True)))                     # one of two left
    assert r.evaluate(snaps((("a",), False), ((), False)))                        # never acted: not its call


def test_ds5_the_waters_read_leave_the_acid():
    from ruibench.tasks.impl import ds5_litmus as D

    held = D.contents("easy", 0)
    acid = next(l for l, k in held.items() if k == D.ACID)
    waters = [l for l, k in held.items() if k != D.ACID]
    narrow = D.found_by_readings(held, D.ACID)
    left, memo = frozenset(held), {}
    for k, letter in enumerate(waters):
        left = narrow(SimpleNamespace(events_fired=tuple(f"strip{i + 1}_read_{w}"
                                                         for i, w in enumerate(waters[:k + 1]))),
                      left, memo)
    assert left == {acid}
    # the acid read itself is enough, whatever was read before
    left = D.found_by_readings(held, D.ACID)(SimpleNamespace(events_fired=(f"strip1_read_{acid}",)),
                                              frozenset(held), {})
    assert left == {acid}


def test_ds4_a_balanced_weighing_of_two_leaves_the_third(monkeypatch):
    from ruibench.tasks.impl import ds4_fake_coin as D

    coins = ["coin_a", "coin_b", "coin_c"]
    masses = {"coin_a": D.COIN_MASS, "coin_b": D.COIN_MASS, "coin_c": D.FAKE_MASS}
    monkeypatch.setattr(D, "held_things", lambda w: set())
    monkeypatch.setattr(D, "pan_contents", lambda w, coins: w.pans)
    narrow = D.found_by_weighings(coins, masses)
    left, memo = frozenset(coins), {}
    pans = {-1: frozenset({"coin_a"}), 1: frozenset({"coin_b"})}
    for t in (0.0, 0.5, D.SETTLE + 0.1):                     # settles, then reads
        left = narrow(SimpleNamespace(pans=pans, sim_time=t), left, memo)
    assert left == {"coin_c"}
    # a weighing that tips: the lighter pan's coin
    left, memo = frozenset(coins), {}
    pans = {-1: frozenset({"coin_a"}), 1: frozenset({"coin_c"})}
    for t in (0.0, D.SETTLE + 0.1):
        left = narrow(SimpleNamespace(pans=pans, sim_time=t), left, memo)
    assert left == {"coin_c"}
    # coins still moving between pans tell nothing yet
    left = narrow(SimpleNamespace(pans={-1: frozenset({"coin_b"}), 1: frozenset()}, sim_time=9.0),
                  frozenset(coins), {})
    assert left == set(coins)


def test_ds3_cold_over_the_other_shells_leaves_the_coins():
    from ruibench.tasks.impl import ds3_treasure as D

    inst = build("DS3", "easy", 0, CFG)
    names, places = D.shell_names("easy"), D.spots("easy")
    under = D.hidden_under("easy", 0)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    world = engine.reset()
    narrow = D.found_by_the_detector(names, places, under)
    left, memo = frozenset(names), {}
    for k, (x, y) in enumerate(places):
        if k == under:
            continue
        detector = engine.state().obj(D.DETECTOR)
        dz = D.SAND_TOP + 0.05 - detector.pose.pos[2]
        engine.teleport(D.DETECTOR, replace(detector.pose, pos=(x - D.COIL_X, y, detector.pose.pos[2] + dz)))
        left = narrow(engine.state(), left, memo)
    assert left == {names[under]}
    # held high above the sand, the detector shows nothing and rules nothing out
    detector = engine.state().obj(D.DETECTOR)
    engine.teleport(D.DETECTOR, replace(detector.pose, pos=(detector.pose.pos[0], detector.pose.pos[1],
                                                            D.SAND_TOP + D.RANGE + 0.2)))
    assert narrow(engine.state(), frozenset(names), {}) == set(names)
