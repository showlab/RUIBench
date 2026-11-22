"""CL1 "play it back" (v3, two arms; v6, mallets in hand): the tune is only in
the film; the hard level moves the colours and adds chords, which only both
hands at once play."""

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.types import Pose
from ruibench.engines.mock import MockEngine
from ruibench.policies import build_agent
from ruibench.runner import RunPlan, run_plan
from ruibench.tasks.impl import cl1_play_it_back as C
from ruibench.tasks.impl._common import LEFT, RIGHT
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def run(level, agent, seeds=(0, 1)):
    return run_plan(RunPlan(["CL1"], levels=(level,), seeds=seeds),
                    agent_factory=lambda i: build_agent(agent, i), config=CFG)


@pytest.mark.parametrize("level", ["easy", "hard"])
def test_the_film_is_needed_and_enough(level):
    assert all(r.success for r in run(level, "reference"))
    assert not any(r.success for r in run(level, "ignore_demonstration"))


def test_one_hand_at_a_time_cannot_play_a_chord():
    assert all(r.success for r in run("easy", "reference+serial"))
    hard = run("hard", "reference+serial")
    assert not any(r.success for r in hard)
    assert all(r.log.task_metrics["steps_played"] > C.STEPS for r in hard)   # the chords came apart


def test_the_tunes_and_the_bars_are_what_the_levels_promise():
    for seed in range(12):
        easy = C.tune_for("easy", seed)
        hands = [C.side(C.COLOURS.index(next(iter(s)))) for s in easy]
        assert len(easy) == C.STEPS and all(len(s) == 1 for s in easy)
        assert all(a != b for a, b in zip(hands, hands[1:]))               # the hands take turns
        assert C.colours_for("easy", seed, easy) == C.COLOURS
        hard = C.tune_for("hard", seed)
        assert sum(len(s) == 2 for s in hard) == C.CHORDS["hard"]
        bars = C.colours_for("hard", seed, hard)
        assert sorted(bars) == sorted(C.COLOURS)
        assert all(a != b for a, b in zip(bars, C.COLOURS))                 # no bar keeps its colour
        where = {c: C.side(i) for i, c in enumerate(bars)}
        assert all({where[c] for c in s} == {LEFT, RIGHT} for s in hard if len(s) == 2)
        for a, b in zip(hard, hard[1:]):
            assert not (a & b)                                              # no bar twice in a row
    inst = build("CL1", "hard", 0, CFG)
    assert inst.spec.arms == 2 and len(inst.scene.robots[0].arms) == 2
    # every bar within its own hand's reach: 25 cm to the side of its arm
    for body in inst.scene.bodies:
        if body.name in C.KEYS:
            arm_y = 0.30 if body.attrs["hand"] == LEFT else -0.30
            assert abs(body.pose.pos[1] - arm_y) <= 0.25


def test_a_chord_is_two_hands_within_the_window():
    group = C.group
    assert group([(1.0, "red", LEFT), (1.2, "blue", RIGHT)]) == [(1.0, frozenset({"red", "blue"}))]
    # too far apart, or the same hand: two notes
    assert len(group([(1.0, "red", LEFT), (1.0 + C.CHORD_WINDOW + 0.1, "blue", RIGHT)])) == 2
    assert len(group([(1.0, "red", LEFT), (1.1, "orange", LEFT)])) == 2
    # a third bar struck at once does not join a chord
    assert len(group([(1.0, "red", LEFT), (1.1, "blue", RIGHT), (1.2, "pink", RIGHT)])) == 2


def test_copying_the_hands_works_only_while_the_bars_stay_put():
    """Playing the bars where the film's hands went, whatever their colour:
    right on the film's xylophone, wrong once the colours have moved."""

    class Copycat(C.TunePolicy):
        def keys(self, world, ctx):
            if self._keys is None:
                film = ctx.extras["demonstrations"][0].states
                first = film[0]
                where = {first.obj(k).attrs["note"]: k for k in C.KEYS}
                self._keys = [sorted(where[c] for c in step) for step in C.heard(film)]
            return self._keys

    for level, ok in (("easy", True), ("hard", False)):
        copies = run_plan(RunPlan(["CL1"], levels=(level,), seeds=(0, 1)),
                          agent_factory=lambda i: Copycat(), config=CFG)
        assert all(r.success is ok for r in copies), level
    for seed in range(6):
        easy, hard = build("CL1", "easy", seed, CFG), build("CL1", "hard", seed, CFG)
        assert easy.scene.structure_key() == hard.scene.structure_key()
        assert not any(c in easy.instruction.lower() for c in C.COLOURS)


def test_the_hands_start_with_mallets_and_only_a_mallet_head_sounds_a_bar():
    inst = build("CL1", "easy", 0, CFG)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    world = engine.state()
    keyboard = inst.reactor_factory()
    for arm in (LEFT, RIGHT):
        assert world.arm(arm).holding == C.mallet_name(arm)
        # the head waits outboard of its own bars, over none of them
        assert not any(C.on_bar(world, k, C.head(world, arm), 0.05) for k in C.KEYS)
    key = C.KEYS[5]
    x, y, _ = world.obj(key).pose.pos
    # the closed hand itself on the bar does nothing: the mallet is let go first
    engine.set_gripper(RIGHT, 0.0)
    engine.advance(0.3)
    engine.teleport(C.mallet_name(RIGHT), Pose((x - C.HEAD_X, y, C.BAR_TOP + C.HEAD_R)))
    for _ in range(5):
        engine.advance(0.1)
        keyboard.tick(engine, engine.state())
    assert C.played(engine.state()) == []
    # a held mallet's head coming down on it plays it, once
    engine.load(inst.scene, 0)
    keyboard = inst.reactor_factory()
    arm, up, down = C.press_poses(engine.state(), key)
    for pose in (up, down, down, down, up):
        engine.set_arm_target(arm, pose)
        engine.advance(0.4)
        keyboard.tick(engine, engine.state())
    assert C.played(engine.state()) == [frozenset({engine.state().obj(key).attrs["note"]})]
    engine.advance(0.5)                                  # let up, the bar springs back
    assert C.all_up(engine.state())
