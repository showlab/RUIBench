"""DI5 "whack-a-mole" (v6: the hands start with the mallets): the moles come
up where nothing said they would, and only a held mallet's head whacks one."""

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.types import Pose
from ruibench.engines.mock import MockEngine
from ruibench.policies import build_agent
from ruibench.runner import RunPlan, run_plan
from ruibench.tasks.impl import di5_whack_a_mole as W
from ruibench.tasks.impl import _toys as T
from ruibench.tasks.impl._common import LEFT, RIGHT
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def run(level, agent, seeds=(0, 1)):
    return run_plan(RunPlan(["DI5"], levels=(level,), seeds=seeds),
                    agent_factory=lambda i: build_agent(agent, i), config=CFG)


@pytest.mark.parametrize("level", ["easy", "hard"])
def test_the_mallets_whack_the_moles_and_nothing_is_known_in_advance(level):
    results = run(level, "reference")
    assert all(r.success for r in results)
    assert all("first_whack" in r.log.score["milestones_hit"] for r in results)
    assert not any(r.success for r in run(level, "do_nothing"))
    assert not any(r.success for r in run(level, "first_frame_only"))


def test_the_levels_put_moles_up_as_they_say():
    for seed in range(6):
        for _, moles in W.schedule("hard", seed):
            assert {m.split("_")[1][:-1] for m in moles} == {"left", "right"}
        assert all(len(p) == 1 for _, p in W.schedule("easy", seed))
    inst = build("DI5", "hard", 0, CFG)
    assert inst.instruction == "Whack the moles with the mallets!"
    # each hand starts holding its mallet, the head waiting over the
    # cabinet's outer edge on its own side, over no hole (v6)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    world = engine.state()
    cabinet = next(b for b in inst.scene.bodies if b.name == "cabinet")
    holes = list(W.build_scene("hard")[1].values())
    for arm, arm_y in ((LEFT, 0.30), (RIGHT, -0.30)):
        m = world.obj(W.mallet_name(arm))
        assert m.is_held and world.arm(arm).holding == W.mallet_name(arm)
        hx, hy, hz = W.head(world, arm)
        assert abs(hy - arm_y) <= 0.25 and abs(hy - cabinet.pose.pos[1]) < cabinet.size[1]
        assert hz > T.MOLE_TOP + T.MOLE_RISE                  # over a risen mole
        assert all(abs(hy - y) > T.MOLE_HOLE + W.HEAD_R for _, y in holes)


def _game(start_up=False):
    inst = build("DI5", "easy", 0, CFG)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    return inst, engine, inst.reactor_factory()


def test_only_a_held_mallet_whacks():
    inst, engine, moles = _game()
    engine.set_gripper(LEFT, 0.0)                        # the left hand lets go of its mallet
    for _ in range(int((W.game_start(CFG.t_ref_seconds) + 1) / 0.1)):
        engine.advance(0.1)
        moles.tick(engine, engine.state())
    world = engine.state()
    assert W.STARTED in world.events_fired                # the game starts on its own
    assert not world.obj(W.mallet_name(LEFT)).is_held
    up = [n for n in moles.holes if W.mole_up(world, n)]
    assert len(up) == 1
    x, y = moles.holes[up[0]]
    # a mallet lying on the mole (not held) does not whack it
    top = W.mole_top(world, up[0])
    name = W.mallet_name(LEFT)
    engine.teleport(name, Pose((x - W.HEAD_X, y, top + W.HEAD_R)))
    for _ in range(5):
        engine.advance(0.1)
        moles.tick(engine, engine.state())
    assert not moles.whacked
