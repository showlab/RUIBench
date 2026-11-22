"""CL2 "the dolls' names" (v3): the films are shot at another table in
another light, and the six dolls are two look-alike pairs and two others."""

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.errors import ConfigError
from ruibench.engines.isaac.environment import LOOK_KEYS, check_look
from ruibench.engines.mock import MockEngine
from ruibench.policies import build_agent
from ruibench.runner import RunPlan, run_plan
from ruibench.runner.demonstration import play
from ruibench.tasks.impl import cl2_names as C
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def test_the_films_are_shot_elsewhere_and_the_scene_stays_the_same():
    looks = set()
    for seed in range(12):
        for level in ("easy", "hard"):
            inst = build("CL2", level, seed, CFG)
            for film in inst.demonstrations:
                assert film.look == C.film_look(seed)
                assert set(film.look) <= set(LOOK_KEYS) and film.look["table"] != "wood:oak"
                # the look is not the scene's structure: one stage serves film and episode
                assert film.scene.structure_key() == inst.scene.structure_key()
            looks.add(C.film_look(seed)["table"])
    assert len(looks) == len(C.FILM_LOOKS)            # every look is used


class LookRecordingEngine(MockEngine):
    """The mock, told the look as the Isaac engine is."""

    def __init__(self):
        super().__init__()
        self.looks = []

    def set_look(self, look=None):
        self.looks.append(dict(look) if look else None)


def test_the_film_is_shot_in_its_look_and_the_look_is_put_back():
    inst = build("CL2", "hard", 4, CFG)
    film = inst.demonstrations[0]
    engine = LookRecordingEngine()
    watched = play(engine, film, cameras=(), sized=lambda scene: scene)
    assert engine.looks == [film.look, None]           # for the film, then the table's own
    assert watched.look == film.look and watched.as_log()["look"] == film.look

    class Broken:
        def reset(self):
            pass

        def commands(self, world, ctx):
            raise RuntimeError("the demonstrator broke")

    from dataclasses import replace

    engine = LookRecordingEngine()
    with pytest.raises(RuntimeError):
        play(engine, replace(film, demonstrator=Broken), cameras=(), sized=lambda scene: scene)
    assert engine.looks == [film.look, None]           # put back even so


def test_a_run_logs_where_the_films_were_shot_and_still_works():
    results = run_plan(RunPlan(["CL2"], levels=("hard",), seeds=(0, 1)),
                       agent_factory=lambda i: build_agent("reference", i), config=CFG)
    assert all(r.success for r in results)
    for r in results:
        assert [d["look"] for d in r.log.demonstrations] == [C.film_look(r.log.seed)] * 2
    blind = run_plan(RunPlan(["CL2"], levels=("hard",), seeds=(0, 1, 2, 3)),
                     agent_factory=lambda i: build_agent("ignore_demonstration", i), config=CFG)
    assert sum(r.success for r in blind) <= 1


def test_a_look_is_checked():
    assert check_look(None)["table"] == "wood:oak"
    with pytest.raises(ConfigError):
        check_look({"table": "marble"})
    with pytest.raises(ConfigError):
        check_look({"tablecloth": "fabric:red"})
    with pytest.raises(ConfigError):
        check_look({"light_colour": (2.0, 0.0, 0.0)})
    with pytest.raises(ConfigError):
        check_look({"light": 50.0})


def test_two_pairs_of_dolls_look_alike_and_can_still_be_told_apart():
    assert len(C.KINDS) == 6 and len(set(C.KINDS)) == 6
    paired = [k for pair in C.LOOKALIKES for k in pair]
    assert len(paired) == len(set(paired)) == 4 and set(paired) <= set(C.KINDS)
    for a, b in C.LOOKALIKES:
        # the same clothes ...
        assert a.split(" in ", 1)[1] == b.split(" in ", 1)[1]
        # ... other hair, and from the other family's scan
        assert a.split(" in ", 1)[0] != b.split(" in ", 1)[0]
        families = {C.LOOKS[k][0].rsplit("_", 1)[0] for k in (a, b)}
        assert families == {"doll_family", "doll_modern"}
