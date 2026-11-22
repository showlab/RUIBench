"""Every implemented task, checked the way the design says a task is "done".

Three-layer solvability, layer 1: a privileged scripted reference finishes it,
which is what shows the scene and the success criterion are valid at all.  Then
the negative controls, each judged **only against the information it removes**.
"""

import math
from dataclasses import replace

import pytest

from ruibench.core.config import RunConfig
from ruibench.policies import wrap
from ruibench.runner import RunPlan, run_plan
from ruibench.tasks.impl._common import TABLE as TABLE_TOP
from ruibench.tasks.instance import build, implemented

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
SEEDS = (0, 1, 2)


def run(task, level, agent=None, seeds=SEEDS):
    factory = ((lambda i: i.reference_factory(i)) if agent is None
               else (lambda i: wrap(agent, i.reference_factory(i))))
    return run_plan(RunPlan([task], levels=(level,), seeds=seeds),
                    agent_factory=factory, config=CFG)


@pytest.mark.parametrize("task", implemented())
@pytest.mark.parametrize("level", ["easy", "hard"])
def test_scripted_reference_solves_every_level(task, level):
    """Layer 1.  A level a privileged reference cannot finish is a broken task,
    not a hard one.  And on its way the wrist camera meets nothing: its
    housing is solid on Isaac, where the hand stops at whatever it meets,
    while the mock's hand passes through (v16: the FR3's camera moved to the
    hand's robot side, and WK3's came down on the pendulum's frame -- every
    seed failed on Isaac)."""
    from ruibench.tasks.clearance import camera_touches

    results = run(task, level)
    assert all(r.success for r in results), [
        (r.score.level, r.score.failure.value, r.score.violations) for r in results]
    first = results[0]
    assert not camera_touches(build(task, level, first.log.seed).scene, first.trajectory)


@pytest.mark.parametrize("task", implemented())
@pytest.mark.parametrize("level", ["easy", "hard"])
def test_doing_nothing_never_works(task, level):
    assert not any(r.success for r in run(task, level, "do_nothing"))


@pytest.mark.parametrize("task", ["CL3", "WK5", "WK3", "WK1", "CL2"])
def test_static_tasks_are_solvable_from_the_first_frame(task):
    """**This control is supposed to pass here.**

    CL3, WK5 and WK1 are static: the geometry is fully visible at t=0 and a
    committed open-loop plan really does finish them.  A `first_frame_only`
    control that failed everywhere would be removing something other than what
    it claims -- and would falsely 'validate' these tasks.
    """
    assert all(r.success for r in run(task, "easy", "first_frame_only"))


def test_responsiveness_needs_a_second_look():
    """DI4's event is invisible at t=0, so a first-frame plan must fail it."""
    assert not any(r.success for r in run("DI4", "easy", "first_frame_only"))


def test_active_perception_cannot_be_planned_blind():
    """DS1's deciding evidence is hidden at t=0 at both levels (v6), so an
    open-loop plan can only guess.  Judged against chance (three presents,
    five), not against zero."""
    for level, presents in (("easy", 3), ("hard", 5)):
        results = run("DS1", level, "first_frame_only", seeds=tuple(range(3 * presents)))
        assert sum(1 for r in results if r.success) == 3, level
    assert all(r.success for r in run("DS1", "hard", seeds=tuple(range(8))))


@pytest.mark.parametrize("task", ["CL3", "WK5", "WK1", "DS1", "CL2", "WK4", "DI1"])
def test_ignoring_change_only_matters_where_there_is_change(task):
    """No event, no effect.  The control checks one thing and is not asked to
    fail everywhere."""
    assert all(r.success for r in run(task, "easy", "ignore_change"))


def test_ds2_the_egg_is_hidden_until_its_place_is_lifted_and_evenly_placed():
    from collections import Counter

    from ruibench.tasks.impl.ds2_egg_hunt import hidden_under

    for level, n in (("easy", 3), ("hard", 4)):
        # every place (by where it stands) hides the egg equally often
        counts = Counter(hidden_under(seed, n) for seed in range(4 * n))
        assert set(counts.values()) == {4}                      # exactly uniform
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine

    seed = next(s for s in range(20) if build("DS2", "hard", s, CFG).metadata["kinds"][
        build("DS2", "hard", s, CFG).metadata["under"]] == "pot")
    inst = build("DS2", "hard", seed, CFG)
    engine = MockEngine()
    engine.track_visibility = True
    engine.load(inst.scene, seed)
    assert engine.state().obj("egg").visible_from == ()
    places = [p for p, k in inst.metadata["kinds"].items() if k == "pot"]
    others = [p for p in places if p != inst.metadata["under"]]
    lifted = lambda name: Pose(tuple(v + d for v, d in zip(engine.state().obj(name).pose.pos,
                                                            (0.0, 0.0, 0.15))))
    engine.teleport(others[0], lifted(others[0]))
    assert engine.state().obj("egg").visible_from == ()        # the wrong place
    engine.teleport(inst.metadata["under"], lifted(inst.metadata["under"]))
    assert "agentview" in engine.state().obj("egg").visible_from
    # blind, the plan lifts the first place: right one time in four
    blind = run("DS2", "hard", "first_frame_only", seeds=tuple(range(8)))
    assert sum(r.success for r in blind) == 2
    # a pot hides the egg as often as a box does
    kinds = Counter(build("DS2", "hard", seed, CFG).metadata["kinds"][
        build("DS2", "hard", seed, CFG).metadata["under"]] for seed in range(8))
    assert set(kinds.values()) == {4}


def test_cl2_a_name_names_a_doll_by_its_look_not_its_place():
    from ruibench.tasks.impl.cl2_names import DOLLS, NAMES

    moved, pairs = 0, set()
    for seed in range(20):
        easy, hard = build("CL2", "easy", seed, CFG), build("CL2", "hard", seed, CFG)
        assert easy.scene.structure_key() == hard.scene.structure_key()
        names = hard.metadata["names"]
        first, second = list(names)
        assert first != second and set(names) <= set(NAMES)
        assert easy.metadata["names"] == {first: names[first]}
        assert easy.instruction == f"Put {first} to bed."
        assert [d.caption for d in hard.demonstrations] == [f"This is {first}.", f"This is {second}."]
        kind = {b.name: b.attrs["kind"] for b in hard.scene.bodies if b.name in DOLLS}
        named = next(d for d in DOLLS if kind[d] == names[first])
        film = {b.name: b.pose.pos for b in hard.demonstrations[0].scene.bodies}
        now = {b.name: b.pose.pos for b in hard.scene.bodies}
        moved += film[named] != now[named]
        pairs.add((first, second))
    assert moved >= 15                  # where it lay in the film says little
    assert len(pairs) >= 10             # the names themselves are drawn
    guesses = run("CL2", "easy", "ignore_demonstration", seeds=tuple(range(12)))
    assert sum(r.success for r in guesses) <= 4        # one doll in six


def test_cl2_hard_sends_each_name_to_its_own_place_and_a_swap_fails():
    """Both names are needed, and mixing them up is a failure, not half a
    success."""
    from dataclasses import replace

    from ruibench.core.types import Pose
    from ruibench.tasks.impl.cl2_names import BED, DOLLS, SOFA

    inst = build("CL2", "hard", 3, CFG)
    names = inst.metadata["names"]
    first, second = list(names)
    assert inst.instruction == f"Put {first} to bed and {second} on the sofa."
    kind = {b.name: b.attrs["kind"] for b in inst.scene.bodies if b.name in DOLLS}
    momo = next(d for d in DOLLS if kind[d] == names[first])
    kiki = next(d for d in DOLLS if kind[d] == names[second])
    world = _initial_world(inst)
    bed, sofa = world.obj(BED), world.obj(SOFA)

    def put(state, doll, where):
        o = state.objects[doll]
        z = where.pose.pos[2] - where.size[2] + 0.02 + o.size[2]     # on the mattress, the seat
        objects = dict(state.objects)
        objects[doll] = replace(o, pose=Pose((where.pose.pos[0], where.pose.pos[1], z), o.pose.quat))
        return replace(state, objects=objects)

    placed = inst.scorer.milestones[-1].predicate
    right = put(put(world, momo, bed), kiki, sofa)
    swapped = put(put(world, kiki, bed), momo, sofa)
    assert placed.holds(right) and not placed.holds(swapped)


def test_ds1_no_name_shows_from_where_the_cameras_start():
    """If the names could be read from where the cameras start, the task
    would measure nothing -- at either level (v6: the easy level's used to
    face the fixed camera, and its film showed no looking)."""
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl.ds1_whose_present import CARDS

    for level in ("easy", "hard"):
        inst = build("DS1", level, 0, CFG)
        engine = MockEngine()
        engine.track_visibility = True
        engine.load(inst.scene, 0)
        engine.reset()
        cards = [b.name for b in inst.scene.bodies if b.name in CARDS.values()]
        assert len(cards) == {"easy": 3, "hard": 5}[level]
        for card in cards:
            assert card not in engine.visible_objects("agentview"), (level, card)
            assert card not in engine.visible_objects("wrist"), (level, card)


def test_wk1_instances_are_solvable_and_the_knob_changes_the_plan():
    """The symbolic search proves feasibility and gives the minimum move count;
    the hard level must actually need more moves, or the knob is decorative."""
    from ruibench.tasks.impl.wk1_hanoi import GOAL, layout, mid_games, shortest_plan

    easy = build("WK1", "easy", 0, CFG).metadata["minimum_moves"]
    assert easy == 3
    for seed in range(6):
        hard = build("WK1", "hard", seed, CFG).metadata
        assert hard["rings"] == 4 and 7 <= hard["minimum_moves"] <= 11
    # a game in progress: never the standard start, never three on one peg with the fourth home
    assert len(mid_games()) == 26 and all(len(set(s)) > 1 for s in mid_games())
    assert {layout("hard", s) for s in range(12)} <= set(mid_games())
    assert len(shortest_plan((0, 0, 0, 0), (GOAL,) * 4)) == 15


def test_cl3_keeps_the_same_objects_on_the_table_at_both_levels():
    """Only which objects are requested changes -- otherwise clutter would vary
    with the knob and the contrast would stop being about horizon length."""
    easy = build("CL3", "easy", 0, CFG)
    hard = build("CL3", "hard", 0, CFG)
    assert [b.name for b in easy.scene.bodies] == [b.name for b in hard.scene.bodies]
    assert easy.scene.structure_key() == hard.scene.structure_key()    # one simulator for both
    assert len(easy.metadata["order"]) < len(hard.metadata["order"])


def test_cl3_card_shows_the_house_and_blocks_of_a_colour_are_interchangeable():
    """The card draws each block of the house in its colour, bottom up; the
    house goes up in the order that holds it up; either cube of a colour fills
    a slot; a block too many on the board fails the house."""
    from dataclasses import replace

    from ruibench.core.types import Pose
    from ruibench.tasks.impl.cl3_house import (BLOCKS, CARD, CARD_HALF, design_for, house_is,
                                               slot_pose, slots_for)

    inst = build("CL3", "hard", 1, CFG)
    order = inst.metadata["order"]
    assert order[2].endswith("plank") and order[5].endswith("plank")
    assert order[0] == order[1] != order[3] == order[4]            # a colour per storey
    assert order[6].endswith("cube") and order[6] not in (order[0], order[3])   # the chimney
    from ruibench.core.geometry import quat_conj, rotate

    def on_face(scene):
        """The card's marks drawn on its face (the rest are tucked inside it),
        in the leaning card's own frame."""
        card = next(b for b in scene.bodies if b.name == CARD)
        out = {}
        for m in scene.bodies:
            if m.name.startswith("card_"):
                d = tuple(a - b for a, b in zip(m.pose.pos, card.pose.pos))
                out[m.name] = rotate(quat_conj(m.pose.quat), d)[0] > CARD_HALF[0]
        return out

    marks = {b.name: b for b in inst.scene.bodies if b.name.startswith("card_")}
    assert all(on_face(inst.scene).values())                        # all seven drawn
    assert marks["card_mark0"].color == marks["card_mark1"].color != marks["card_mark3"].color
    easy = build("CL3", "easy", 1, CFG)
    tucked = [name for name, drawn in on_face(easy.scene).items() if not drawn]
    # the upper storey and the chimney: their colours, their outlines, the smoke
    assert len(tucked) == 4 + 4 + 1
    colours, planks = design_for(1)
    slots = slots_for(7, colours, planks)
    assert [s.kind for s in slots] == order
    assert all(slot_pose(a)[2] <= slot_pose(b)[2] for a, b in zip(slots, slots[1:]))
    world = _initial_world(inst)
    assert not house_is(slots).holds(world)
    # every block of the house put in its slot: built -- with either cube
    free = {k: [n for n in BLOCKS if BLOCKS[n] == k] for k in set(BLOCKS.values())}
    objects = dict(world.objects)
    placed = []
    for slot in slots:
        name = free[slot.kind].pop()
        yaw = 1.5707963 if slot.across else 0.0
        quat = (0.7071068 if slot.across else 1.0, 0.0, 0.0, 0.7071068 if slot.across else 0.0)
        objects[name] = replace(objects[name], pose=Pose(slot_pose(slot), quat))
        placed.append(name)
    built = replace(world, objects=objects)
    assert house_is(slots).holds(built)
    spare = next(n for n in BLOCKS if n not in placed)
    x, y, z = slot_pose(slots[0])
    objects[spare] = replace(objects[spare], pose=Pose((x + 0.04, y, z)))
    assert not house_is(slots).holds(replace(world, objects=objects))


def test_wk3_the_chain_runs_as_far_as_the_dominoes_stand_in_reach_of_each_other():
    from ruibench.tasks.impl.wk3_dominoes import BELL, HOUSE, DominoPolicy, route

    easy, hard = build("WK3", "easy", 0, CFG), build("WK3", "hard", 0, CFG)
    assert HOUSE in {b.name for b in hard.scene.bodies} - {b.name for b in easy.scene.bodies}
    assert not easy.scene.markers and not hard.scene.markers          # nothing painted
    assert (easy.metadata["knob_value"], hard.metadata["knob_value"]) == ("straight", "corner")
    assert "when you say you are done" in hard.instruction
    # the corner turns a quarter, a little at each domino
    assert [round(p[2], 3) for p in route("hard")][-1] == round(3.14159265 / 2, 3)
    (done,) = run("WK3", "hard", seeds=(0,))
    assert done.success and done.log.task_metrics["fallen"] == 5

    class ThirdOff(DominoPolicy):
        """Puts the third domino 2 cm to the side of where the chain runs."""

        def __init__(self, places):
            places = list(places)
            x, y, yaw = places[2]
            places[2] = (x + 0.02, y, yaw)
            super().__init__(places)

    (off,) = run_plan(RunPlan(["WK3"], levels=("hard",), seeds=(0,)),
                      agent_factory=lambda i: ThirdOff(i.reference_factory(i).places), config=CFG)
    assert not off.success and off.log.task_metrics["fallen"] == 2
    assert "bell_rang" not in off.log.score["milestones_hit"]
    assert BELL in {b.name for b in hard.scene.bodies}


def test_wk2_the_second_rule_brings_the_goat_back():
    for seed in range(5):
        easy, hard = build("WK2", "easy", seed, CFG), build("WK2", "hard", seed, CFG)
        assert (easy.metadata["crossings"], hard.metadata["crossings"]) == (5, 7)
        assert (easy.metadata["plan"].count("goat"), hard.metadata["plan"].count("goat")) == (1, 3)
        assert "cabbage" in hard.instruction and "cabbage" not in easy.instruction
        names = lambda inst: {b.name for b in inst.scene.bodies}
        assert names(easy) - names(hard) == {"bucket"}
        assert names(hard) - names(easy) == {"cabbage"}


def test_wk2_crossing_without_the_rules_leaves_a_pair_alone():
    from ruibench.policies import build_agent

    for level in ("easy", "hard"):
        (blind,) = run_plan(RunPlan(["WK2"], levels=(level,), seeds=(1,)),
                            agent_factory=lambda i: build_agent("reference+ignore_rules", i),
                            config=CFG)
        assert not blind.success and "nothing_eaten" in blind.score.violations, level
        (ok,) = run("WK2", level, seeds=(1,))
        assert ok.success and not ok.score.violations, level


def test_wk6_the_positions_are_the_ones_the_levels_promise():
    from ruibench.tasks.impl.wk6_tic_tac_toe import (best_x, mistakes, play_out, puzzles, threats,
                                                     value, winner)

    assert len(puzzles("easy")) == 208 and len(puzzles("hard")) == 20
    for xs, os_, m in puzzles("easy"):
        x, o = frozenset(xs), frozenset(os_)
        assert m is None and threats(x, o) and not threats(o, x)
        assert winner(x | {best_x(x, o)}, o) == "X"
    assert [m for _, _, m in puzzles("hard")][:4] == [1, 2, 1, 2]
    for xs, os_, m in puzzles("hard"):
        x, o = frozenset(xs), frozenset(os_)
        # no win at the start: perfect play draws; the opponent's mistake gives it
        assert value(x, o, True) == 0
        assert play_out(xs, os_, None, 0) == (None, 0)
        assert play_out(xs, os_, m, 0) == ("X", m)
    x, o = frozenset({4, 0}), frozenset({8})
    assert all(not threats(x, o | {c}) for c in mistakes(x, o))


def test_wk6_greed_loses_the_hard_game_and_a_taken_cell_is_a_foul():
    from ruibench.policies import build_agent
    from ruibench.tasks.impl.wk6_tic_tac_toe import TicTacToePolicy

    for level, ok in (("easy", True), ("hard", False)):
        greedy = run_plan(RunPlan(["WK6"], levels=(level,), seeds=(0, 1)),
                          agent_factory=lambda i: build_agent("reference+greedy", i), config=CFG)
        assert all(r.success is ok for r in greedy), level
    (lost,) = run_plan(RunPlan(["WK6"], levels=("hard",), seeds=(2,)),
                       agent_factory=lambda i: build_agent("reference+greedy", i), config=CFG)
    assert "x_won" not in lost.log.score["milestones_hit"] and not lost.score.violations

    class OnAnO(TicTacToePolicy):
        """Puts its X on a cell the opponent already has."""

        def choose(self, x, o):
            return min(o)

    (foul,) = run_plan(RunPlan(["WK6"], levels=("easy",), seeds=(0,)),
                       agent_factory=lambda i: OnAnO(i.reference_factory(i).board), config=CFG)
    assert not foul.success and foul.score.violations == ["fair_play"]


def test_di1_springing_buttons_need_both_hands_down_together():
    from ruibench.interfaces.commands import Done, MoveTo, Wait
    from ruibench.policies import build_agent
    from ruibench.tasks.impl._common import LEFT, RIGHT
    from ruibench.tasks.impl.di1_launch import PressBothPolicy

    for level, serial_ok in (("easy", True), ("hard", False)):
        serial = run_plan(RunPlan(["DI1"], levels=(level,), seeds=(0, 1)),
                          agent_factory=lambda i: build_agent("reference+serial", i), config=CFG)
        assert all(r.success is serial_ok for r in serial), level
        assert all(r.success for r in run("DI1", level, seeds=(0, 1)))
        # v7: a press lights its button's lamp -- for good at the easy level,
        # only while held at the hard one, where the two are never lit together
        names = [str(e.get("name")) for e in serial[0].log.events]
        assert {"left_lamp_on", "right_lamp_on"} <= set(names), level
        assert ("left_lamp_off" in names) is not serial_ok, level

    class Tap(PressBothPolicy):
        """Both at once, down for a moment, and straight back up.

        The moment is one turn, not nothing: how far a button is pushed is
        read from where the fingertips are when the world is sampled, so a
        hand that goes down and straight back up inside one queue is never
        seen fully down at either level -- that would test the sampler, not
        the latch.  One turn is still far short of the hard level's 0.5 s
        hold, which is what this is here to tell apart."""

        def commands(self, world, ctx):
            if ctx.extras.get("executor_busy"):
                return []
            arms = [LEFT, RIGHT]
            self._step += 1
            if self._step == 1:
                return self._press(world, arms)
            if self._step == 2:                  # a Wait does not gate an arm's own queue
                return [MoveTo(a, self._poses(world, a)[0]) for a in arms]
            return [Wait(2.0), Done("tapped")]

    for level, ok in (("easy", True), ("hard", False)):
        (tap,) = run_plan(RunPlan(["DI1"], levels=(level,), seeds=(0,)),
                          agent_factory=lambda i: Tap(), config=CFG)
        assert tap.success is ok, level
        assert "both_down" in tap.log.score["milestones_hit"]


def test_di2_the_train_keeps_moving_and_a_blind_plan_mostly_misses_it():
    from ruibench.engines.base import joint_program
    from ruibench.tasks.impl.di2_train import AMPLITUDE

    inst = build("DI2", "hard", 0, CFG)
    joint, other = inst.scene.articulations
    track = [joint_program(joint, t / 10) for t in range(400)]
    # the two wagons go together
    assert all(joint_program(other, t / 10) == joint_program(joint, t / 10) for t in range(0, 400, 7))
    assert max(track) - min(track) > 1.9 * AMPLITUDE            # it really travels
    assert all(r.success for r in run("DI2", "hard", seeds=(0, 1)))
    blind = run("DI2", "hard", "first_frame_only", seeds=tuple(range(6)))
    assert sum(r.success for r in blind) <= 3
    easy = build("DI2", "easy", 0, CFG).metadata["train_speed_m_s"]
    assert inst.metadata["train_speed_m_s"] == pytest.approx(16 * easy, rel=0.02)
    # the engine rides along, painted on
    engine = next(b for b in inst.scene.bodies if b.name == "engine")
    assert engine.attrs["attached_to"] == "red_wagon" and engine.attrs["collides"] is False


def test_di2_the_first_picture_shows_the_wagon_where_it_is():
    """The wagon starts where its program has it -- not at the middle of the
    track, dashing to its seeded place once the clock runs -- and the engine
    painted on it goes with it."""
    from ruibench.engines.base import joint_program
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl.di2_train import ENGINE, WAGON

    starts = set()
    for seed in range(4):
        inst = build("DI2", "hard", seed, CFG)
        joint = inst.scene.articulations[0]
        middle = next(b for b in inst.scene.bodies if b.name == WAGON).pose.pos[1]
        engine = MockEngine()
        engine.load(inst.scene, seed)
        first = engine.reset()
        wagon_y = first.obj(WAGON).pose.pos[1]
        # (the mock jitters every body's layout by up to 4 mm)
        assert wagon_y == pytest.approx(middle + joint_program(joint, 0.0), abs=0.005)
        assert first.obj(ENGINE).pose.pos[1] - wagon_y == pytest.approx(
            next(b for b in inst.scene.bodies if b.name == ENGINE).pose.pos[1] - middle, abs=1e-6)
        engine.advance(0.2)                       # no dash: it moves at the train's speed
        moved = abs(engine.state().obj(WAGON).pose.pos[1] - wagon_y)
        assert moved <= inst.metadata["train_speed_m_s"] * 0.2 + 1e-3
        starts.add(round(wagon_y, 3))
    assert len(starts) == 4                       # the seed decides where it is


def test_wk4_two_arms_set_the_places_the_custom_puts_them():
    """v6: two arms, a whole set on each one's tray -- so where a piece starts
    says nothing about where it goes -- and the custom decides which arm does
    what: the fork, on the guest's left, is the right arm's."""
    from ruibench.tasks.impl._common import LEFT, RIGHT
    from ruibench.tasks.impl.wk4_set_table import (PLACES, SEATS, arm_for, in_place, place_xy,
                                                   set_for, tray_spots)

    easy, hard = build("WK4", "easy", 2, CFG), build("WK4", "hard", 2, CFG)
    assert easy.scene.structure_key() != hard.scene.structure_key()   # one mat, or two
    assert (easy.instruction, hard.instruction) == (
        "Set the table for the guest across from you.",
        "Set the table for the two guests across from you.")
    assert (easy.metadata["pieces_to_place"], hard.metadata["pieces_to_place"]) == (3, 6)
    assert not easy.scene.markers                  # nothing on the mats says where
    # the easy level's one place is shared out by the custom, not by the trays
    middle = SEATS["easy"]["middle"]
    assert arm_for(place_xy(middle, "fork")) == RIGHT
    assert all(arm_for(place_xy(middle, k)) == LEFT for k in ("knife", "spoon"))
    assert not any(b.name.startswith("cup") for b in hard.scene.bodies)      # v10: no cups
    for arm in (LEFT, RIGHT):                      # a whole set on each tray
        assert set(tray_spots(arm, 2)) == set(PLACES)
    for seat in SEATS["hard"].values():            # a place is one arm's at the hard level
        assert len({arm_for(place_xy(seat, k)) for k in PLACES}) == 1
    (one,) = run("WK4", "easy", seeds=(2,))
    end = one.trajectory[-1]
    assert one.success and set_for(end, middle)
    # both hands placed something: the milestone says so
    assert "both_hands" in one.log.score["milestones_hit"]
    (done,) = run("WK4", "hard", seeds=(2,))
    hit = done.log.score["milestones_hit"]
    assert done.success
    assert hit["first_piece"] <= hit["all_placed"] <= hit["table_set"]


def test_every_built_task_builds_at_both_levels():
    """v3 is being built (v3.md §5): planned tasks are allowed, but a card that
    says implemented, or names a module, has to build."""
    from ruibench.tasks import main_board

    for t in main_board():
        if t.impl:
            for level in ("easy", "hard"):
                assert build(t.id, level, 0, CFG).scene.bodies, t.id


def _sorting_instance(seed=0):
    """Six cubes into the basket of their colour: v0's CL3 at its hard level,
    kept as the fixture the bin helpers were tuned on."""
    from ruibench.core.rng import SeedBundle
    from ruibench.policies.scripted import SequentialPickAndPlacePolicy
    from ruibench.scoring import AllOfT, Finally, In, Scorer
    from ruibench.tasks.impl._common import ARM, BIN_XY, basket, cube, grid_layout, table_scene
    from ruibench.tasks.instance import TaskInstance
    from ruibench.tasks.loader import get

    s = SeedBundle(seed)
    spots = grid_layout(s, "cubes", 6)
    colors = [("red", "blue")[i % 2] for i in range(6)]
    s.stream("colors").shuffle(colors)
    bodies = [cube(f"cube{i}", spots[i], colors[i]) for i in range(6)]
    bodies += [basket("basket_red", BIN_XY[0], "red"), basket("basket_blue", BIN_XY[1], "blue")]
    pairs = [(f"cube{i}", f"basket_{colors[i]}") for i in range(6)]
    success = AllOfT(tuple(Finally(In(obj, bin_)) for obj, bin_ in pairs))
    return TaskInstance(
        spec=get("CL3"), level="hard", seed=seed, scene=table_scene("sorting", bodies),
        instruction="Sort the cubes into the basket of their own colour.",
        scorer=Scorer(task_id="CL3", level="hard", success=success, milestones=()),
        reference_factory=lambda inst: SequentialPickAndPlacePolicy(pairs, arm=ARM))


def _run_instance(inst):
    from ruibench.clock import make_clock
    from ruibench.engines import make_engine
    from ruibench.interfaces import make_interface
    from ruibench.runner.episode import EpisodeRunner

    engine = make_engine("mock")
    try:
        return EpisodeRunner(engine, make_interface(CFG.interface), inst.reference_factory(inst),
                             make_clock(CFG.clock), CFG).run(inst)
    finally:
        engine.close()


def test_things_let_go_over_one_spot_in_a_bin_stack_and_the_top_is_not_in_it(monkeypatch):
    """The mock stacks like PhysX does.  Letting every cube go over a bin's
    centre built a three-high tower on Isaac, and its top cube ended above the
    rim -- CL3 hard failed there while the mock, which laid things side by side,
    kept passing.  With the old drop point the mock must fail the same way."""
    import ruibench.policies.scripted as scripted

    monkeypatch.setattr(scripted, "drop_spot",
                        lambda world, obj, target, taken=(): tuple(target.pose.pos[:2]))
    r = _run_instance(_sorting_instance())
    heights = sorted(round(o.pose.pos[2], 3) for n, o in r.trajectory[-1].objects.items()
                     if n.startswith("cube"))
    assert not r.success
    assert heights[-1] == pytest.approx(0.920, abs=0.002)     # what Isaac showed


def test_references_put_things_side_by_side_in_a_bin():
    from ruibench.policies.scripted import container_spots, drop_spot

    world = _initial_world(_sorting_instance())
    bin_ = world.obj("basket_red")
    spots = container_spots(bin_)
    assert len(set(spots)) == 8
    assert all(bin_.contains_xy(p, margin=-0.03) for p in spots)   # clear of the walls
    first = drop_spot(world, world.obj("cube0"), bin_)
    second = drop_spot(world, world.obj("cube1"), bin_, taken=[first])
    assert ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5 > 0.09


def test_a_reference_never_calls_something_put_away_that_the_scorer_does_not():
    """IC1 once spent its whole call budget idle: the sorter asked the scorer's
    question ("is it in a bin?"), its pick-and-place a looser one, and an object
    balanced above the rim satisfied only the second."""
    from ruibench.core.types import Pose
    from ruibench.policies.scripted import placed_on
    from ruibench.scoring import inside

    world = _initial_world(_sorting_instance())
    bin_ = world.obj("basket_red")
    cube = world.obj("cube0")
    for z in (0.832, 0.876, 0.920):
        moved = replace(cube, pose=Pose((bin_.pose.pos[0], bin_.pose.pos[1], z)))
        assert placed_on(moved, bin_) == inside(moved, bin_)


def test_a_reference_puts_a_crooked_thing_down_by_its_lowest_corner():
    """WK2's goat, held crooked after its grasp slipped, came down on one hoof
    above where its half height put the deck; pressed on, it flew off the raft
    when the fingers opened (Isaac, v9)."""
    from ruibench.core.types import Pose
    from ruibench.policies.scripted import reach_below

    cube = _initial_world(_sorting_instance()).obj("cube0")
    sx, sy, sz = cube.size

    def tilted(degrees):
        a = math.radians(degrees)
        return replace(cube, pose=Pose(cube.pose.pos, (math.cos(a / 2), math.sin(a / 2), 0.0, 0.0)))

    assert reach_below(cube) == sz
    assert reach_below(tilted(5)) == sz                        # a tremor is not crooked
    assert reach_below(tilted(20)) == pytest.approx(sy * math.sin(math.radians(20))
                                                    + sz * math.cos(math.radians(20)))


def _initial_world(inst):
    from ruibench.engines import make_engine

    engine = make_engine("mock")
    engine.load(inst.scene, inst.seed)
    return engine.reset()


def test_ds1_names_are_not_read_from_across_the_table():
    """v8: the easy tags lie on top, face up -- but a script hand a centimetre
    tall is read from within 45 cm, not from the fixed camera; the hard
    labels are underneath, facing the table."""
    from ruibench.engines.base import external_camera_pose, legible_from
    from ruibench.tasks.impl.ds1_whose_present import CARDS, NAMES, OWNER, owners_for

    for level in ("easy", "hard"):
        scene = build("DS1", level, 0).scene
        eye = external_camera_pose(scene).pos
        for card in (b for b in scene.bodies if b.name in CARDS.values()):
            assert not legible_from(card, card.pose.pos, card.pose.quat, eye)
            assert card.attrs["text"] in card.name.title()
            if level == "easy":                          # face up, and read from close by
                above = tuple(card.pose.pos[:2]) + (card.pose.pos[2] + 0.3,)
                assert legible_from(card, card.pose.pos, card.pose.quat, above)
        n = len(NAMES[level])
        assert all(OWNER in owners_for(seed, level) and len(set(owners_for(seed, level))) == n
                   for seed in range(12))
    assert "Tim" in NAMES["hard"] and "Tim" not in NAMES["easy"]


def test_ds1_a_label_underneath_is_read_once_turned_to_the_camera():
    """The hard level's label reads upright to the fixed camera from the
    present held bottom-first towards it -- and not from the table."""
    from ruibench.core.geometry import compose, relative_pose
    from ruibench.engines.base import external_camera_pose, legible_from
    from ruibench.tasks.impl.ds1_whose_present import CARDS, OWNER, show_pose, GRASP_HEIGHT

    inst = build("DS1", "hard", 0)
    scene = inst.scene
    eye = external_camera_pose(scene).pos
    card = next(b for b in scene.bodies if b.name == CARDS[OWNER])
    box = next(b for b in scene.bodies if b.name == card.attrs["attached_to"])
    from ruibench.policies.scripted import turned_down
    grasp = type(box.pose)((box.pose.pos[0], box.pose.pos[1], box.pose.pos[2] + GRASP_HEIGHT),
                           turned_down(box.attrs["grasp_yaw"]))
    # the present and its label as the hand holds them at the showing pose
    held_box = compose(show_pose(), relative_pose(grasp, box.pose))
    held_card = compose(held_box, relative_pose(box.pose, card.pose))
    assert not legible_from(card, card.pose.pos, card.pose.quat, eye)
    assert legible_from(card, held_card.pos, held_card.quat, eye)


def test_a_painted_mark_travels_with_its_object():
    """A name card is carried with its present rather than left floating."""
    from ruibench.engines import make_engine
    from ruibench.interfaces.commands import MoveTo, SetGripper
    from ruibench.core.types import GRASP_DOWN, Pose
    from ruibench.executor import Executor

    inst = build("DS1", "hard", 0)
    box_name = "present1"
    card_name = next(b.name for b in inst.scene.bodies if b.attrs.get("attached_to") == box_name)
    engine = make_engine("mock")
    engine.load(inst.scene, 0)
    world = engine.reset()
    box, card = world.obj(box_name), world.obj(card_name)
    before = [p - b for p, b in zip(card.pose.pos, box.pose.pos)]
    ex = Executor(engine, inst.scene.arm_ids)
    x, y, z = box.pose.pos
    ex.submit_all([SetGripper("robot0/right", 0.0),
                   MoveTo("robot0/right", Pose((x, y, z), GRASP_DOWN)),
                   SetGripper("robot0/right", 1.0),
                   MoveTo("robot0/right", Pose((x, y, z + 0.15), GRASP_DOWN))])
    ex.tick(8.0)
    world = engine.state()
    box, card = world.obj(box_name), world.obj(card_name)
    assert box.pose.pos[2] > z + 0.1                         # it was lifted
    after = [p - b for p, b in zip(card.pose.pos, box.pose.pos)]
    assert all(abs(a - b) < 1e-6 for a, b in zip(after, before))


def test_only_visual_marks_may_be_carried_along():
    from ruibench.core.errors import ConfigError
    from ruibench.engines.base import BodySpec, attachments

    block = BodySpec("block")
    mark = BodySpec("mark", attrs={"attached_to": "block", "collides": False})
    assert attachments([block, mark]) == {"mark": "block"}
    with pytest.raises(ConfigError):
        attachments([block, BodySpec("lid", attrs={"attached_to": "block"})])
    with pytest.raises(ConfigError):
        attachments([BodySpec("mark", attrs={"attached_to": "nowhere", "collides": False})])


def test_a_mark_is_read_only_from_in_front_of_its_face():
    """The shared legibility rule: grazing views do not count, and the face's
    normal turns with the body.  From DS1's start pose the Isaac wrist camera
    caught 34 pixels of the hard level's patch a few degrees behind its plane,
    and the reference 'saw' the answer without moving."""
    import math

    from ruibench.core.types import Pose
    from ruibench.engines.base import BodySpec, legible_from

    mark = BodySpec("mark", pose=Pose((0.0, 0.0, 0.0)),
                    attrs={"facing": (-1.0, 0.0, 0.0), "facing_cos_min": 0.45})
    here, upright = (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0)
    assert legible_from(mark, here, upright, (-0.3, 0.0, 0.2))           # in front
    assert not legible_from(mark, here, upright, (0.03, 0.0, 0.7))       # grazing, from above
    assert not legible_from(mark, here, upright, (0.3, 0.0, 0.2))        # behind
    half_turn = (math.cos(math.pi / 2), 0.0, 0.0, math.sin(math.pi / 2)) # the body turned 180 deg
    assert legible_from(mark, here, half_turn, (0.3, 0.0, 0.2))
    assert legible_from(BodySpec("plain"), here, upright, (0.3, 0.0, 0.2))


def test_ds1_the_hand_as_it_hangs_reads_the_tags():
    """v8, the advisor: the easy level needs no turning or tipping of the hand.
    Over the row, pointing down, the wrist camera reads every tag -- and at
    the hard level, with the names underneath, none."""
    import math

    from ruibench.engines.mock import MockEngine
    from ruibench.policies.scripted import turned_down
    from ruibench.core.geometry import yaw_of
    from ruibench.tasks.impl.ds1_whose_present import CARDS, look_down_pose

    look = look_down_pose()
    assert max(abs(a - b) for a, b in zip(look.quat, turned_down(yaw_of(look.quat)))) < 1e-6 or \
        max(abs(a + b) for a, b in zip(look.quat, turned_down(yaw_of(look.quat)))) < 1e-6
    for level in ("easy", "hard"):
        for seed in range(3):
            inst = build("DS1", level, seed)
            cards = [b.name for b in inst.scene.bodies if b.name in CARDS.values()]
            engine = MockEngine()
            engine.track_visibility = True
            engine.load(inst.scene, seed)
            engine.reset()
            engine.set_arm_target("robot0/right", look)
            engine.advance(6.0)
            seen = set(engine.visible_objects("wrist"))
            if level == "easy":
                assert set(cards) <= seen, (level, seed, set(cards) - seen)
            else:
                assert not set(cards) & seen, (level, seed)


def test_ds1_the_reference_looks_once_or_turns_presents_over():
    """Easy: one look, then Tom's present to the mat.  Hard: presents turned
    over one at a time, nearest the mat first, until Tom's label shows."""
    easy = run("DS1", "easy", seeds=tuple(range(6)))
    assert all(r.success for r in easy)
    assert {sum(1 for t in r.log.turns if '"seconds": 0.8' in t["raw_response"]) for r in easy} == {1}
    hard = run("DS1", "hard", seeds=tuple(range(10)))
    assert all(r.success for r in hard)
    shown = [sum(1 for t in r.log.turns if '"seconds": 1.0' in t["raw_response"]) for r in hard]
    assert min(shown) == 1 and max(shown) == 5, shown


# -- v3 prototypes (2026-09-19) ---------------------------------------------------------
def _agent_run(task, level, spec, seeds):
    from ruibench.policies.agents import build_agent

    return run_plan(RunPlan([task], levels=(level,), seeds=seeds),
                    agent_factory=lambda i: build_agent(spec, i), config=CFG)


def test_ds4_weighs_as_few_times_as_the_puzzle_needs_and_a_right_guess_does_not_count():
    for level, n, fewest in (("easy", 3, 1), ("hard", 9, 2)):
        seeds = tuple(range(n))
        results = run("DS4", level, seeds=seeds)
        assert all(r.success for r in results)
        assert {r.log.task_metrics["weighings"] for r in results} == {fewest}
        blind = run("DS4", level, "first_frame_only", seeds=seeds)
        # every place is the fake once in n seeds: the guess that is right is put in
        # the dish before any weighing found it, and fails
        assert not any(r.success for r in blind)
        assert sum("fake_found_first" in r.score.violations for r in blind) == 1


def test_ds4_the_balance_tips_toward_the_heavier_pan():
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import _toys as T
    from ruibench.tasks.impl import ds4_fake_coin as D

    inst = build("DS4", "easy", 0, CFG)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    coins = [b.name for b in inst.scene.bodies if b.name.startswith("coin")]
    fake = inst.metadata["fake"]
    real = [c for c in coins if c != fake]
    masses = {c: D.FAKE_MASS if c == fake else D.COIN_MASS for c in coins}
    balance = D.Balance(coins, masses)
    for side, coin in ((-1, fake), (1, real[0])):
        x, y, z = T.pan_centre(D.pivot(), 0.0, side)
        engine.teleport(coin, Pose((x, y, z + T.COIN_HALF[1])))
    for _ in range(80):
        engine.advance(0.1)
        balance.tick(engine, engine.state())
    # the fake (lighter) in the -y pan: the +y pan goes down, the -y pan up
    joints = engine.state().articulations
    assert joints[D.HINGE] < -0.5 * T.BALANCE_TILT
    assert joints[D.SLIDES[1]] < 0 < joints[D.SLIDES[-1]]


def test_cl4_the_water_comes_up_a_step_a_stone_and_the_ball_is_out_of_reach():
    from ruibench.tasks.impl import cl4_pitcher as C

    for level in ("easy", "hard"):
        results = run("CL4", level, seeds=(0, 1))
        assert all(r.success for r in results)
        assert all(r.log.task_metrics["stones_used"] == C.GAME[level] for r in results)
        blind = run("CL4", level, "ignore_demonstration", seeds=(0, 1))
        assert not any(r.success for r in blind)
        for r in blind:
            ball = r.log.trajectory[-1]["objects"]["ball"]["pos"]
            assert ball[2] < C.TABLE + C.JAR_H         # never came up out of the jar
        # the same things on the bench every seed; only their places change
        assert len({build("CL4", level, s, CFG).scene.structure_key() for s in range(4)}) == 1

    # down in the jar the ball is out of reach, and only the water brings it up
    inst = build("CL4", "easy", 0, CFG)
    ball = next(b for b in inst.scene.bodies if b.name == "ball")
    assert ball.attrs["graspable_later"]
    assert ball.pose.pos[2] < C.TABLE + C.JAR_H - 4 * C.BALL_R
    assert C.JAR_R * 2 < 0.208                         # narrower than the hand is long
    assert C.TABLE + C.END_TOP + C.BALL_R > C.TABLE + C.JAR_H + 0.008

def test_wk2_the_raft_takes_one_passenger():
    """Two aboard: it stays, and says so -- a tap does not send it either."""
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import wk2_river as R

    inst = build("WK2", "hard", 0, CFG)
    names = R.things("hard")
    engine = MockEngine()
    engine.load(inst.scene, 0)
    ferry = R.Ferry(names)
    raft = engine.state().obj(R.RAFT).pose.pos
    for k, n in enumerate(names[:2]):
        engine.teleport(n, Pose((raft[0], raft[1] + (k - 0.5) * 0.06, R.DECK + 0.05)))
    for _ in range(40):
        engine.advance(0.1)
        ferry.tick(engine, engine.state())
    assert "raft_refused" in [r.name for r in ferry.records]
    assert R.raft_side(engine.state()) == R.NEAR


def test_wk2_a_passenger_aboard_sends_the_raft_and_a_tap_sends_it_empty():
    """v9: no bells.  Put aboard, the raft sets off once the hand is away --
    not with the passenger it has just brought; empty, it goes on a tap."""
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.policies.scripted import turned_down
    from ruibench.tasks.impl import wk2_river as R

    inst = build("WK2", "easy", 0, CFG)
    names = R.things("easy")
    engine = MockEngine()
    engine.load(inst.scene, 0)
    ferry = R.Ferry(names)

    def run(seconds):
        for _ in range(int(seconds / 0.1)):
            engine.advance(0.1)
            ferry.tick(engine, engine.state())

    raft = engine.state().obj(R.RAFT).pose.pos
    engine.teleport(R.GOAT, Pose((raft[0], raft[1] + R.SEAT_Y, R.DECK + 0.06)))
    run(0.5)
    assert R.raft_side(engine.state()) == R.NEAR          # the hand is still over it
    engine.set_arm_target("robot0/right", Pose((raft[0] - 0.2, raft[1], R.TABLE + 0.3),
                                                turned_down(0.0)))
    run(10.0)
    assert R.raft_side(engine.state()) == R.FAR and R.GOAT in R.passengers(engine.state(), names)
    run(5.0)                                              # it does not take the goat back
    assert R.raft_side(engine.state()) == R.FAR
    far = engine.state().obj(R.RAFT).pose.pos
    engine.teleport(R.GOAT, Pose((far[0] + 0.12, far[1] + 0.2, R.TABLE + 0.06)))
    run(1.0)
    assert R.raft_side(engine.state()) == R.FAR           # empty, it waits
    x, y = far[0], far[1] + R.TAP_Y
    engine.set_arm_target("robot0/right", Pose((x, y, R.DECK + R.FINGERTIP + 0.004),
                                                turned_down(R.TAP_YAW)))
    run(3.0)
    engine.set_arm_target("robot0/right", Pose((x, y, R.TABLE + 0.3), turned_down(R.TAP_YAW)))
    run(10.0)
    assert R.raft_side(engine.state()) == R.NEAR


def test_wk2_nothing_crosses_but_on_the_raft():
    """Carried over the water by hand, the goat breaks the rules -- even with
    nothing left to eat it."""
    from ruibench.interfaces.commands import Done
    from ruibench.policies.base import ScriptedPolicy
    from ruibench.policies.scripted import PlaceAtPointPolicy
    from ruibench.tasks.impl import wk2_river as R

    class ByHand(ScriptedPolicy):
        name = "by_hand"

        def __init__(self, inst):
            self.homes = inst.reference_factory(inst).homes
            self.sub = None

        def reset(self):
            pass

        def commands(self, world, ctx):
            if ctx.extras.get("executor_busy"):
                return []
            if self.sub is None:
                x, y = self.homes[R.GOAT][R.FAR]
                self.sub = PlaceAtPointPolicy(R.GOAT, (x, y, R.TABLE), tolerance=0.02,
                                              carry_z=R.TABLE + R.HOVER)
            if R.zone(world.obj(R.GOAT).pose.pos[0]) == R.FAR and not world.obj(R.GOAT).is_held:
                return [Done("across by hand")]
            return [c for c in self.sub.commands(world, ctx) if not isinstance(c, Done)]

    [result] = run_plan(RunPlan(["WK2"], levels=("easy",), seeds=(0,)),
                        agent_factory=lambda i: ByHand(i), config=CFG)
    assert result.log.score["violations"] == ["nothing_eaten"]
    breach = R.RiverRules(R.things("easy"), R.RULES["easy"]).first_breach(result.trajectory)
    assert breach and "without the raft" in breach[1]


def test_cl5_the_silhouette_is_filled_by_either_of_its_tilings():
    """v2.3.4: the fish's two large triangles make a square, and laid across
    its other diagonal they fill the silhouette as well -- all seven in place;
    the cat and the boat have one tiling each.  A large triangle turned off
    its place in either tiling still fails."""
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import cl5_tangram as C

    assert [len(C.tilings(s)) for s in ("cat", "fish", "boat")] == [1, 2, 1]
    seed = next(s for s in range(40) if C.shapes_for(s)[0] == "fish")
    inst = build("CL5", "hard", seed, CFG)
    engine = MockEngine()
    engine.load(inst.scene, seed)
    engine.reset()

    def lay(where):
        free = {kind: [n for n, k, _ in C.PIECES if k == kind] for _, k, _ in C.PIECES for kind in [k]}
        for kind, (x, y), yaw in where:
            name = free[kind].pop(0)
            engine.teleport(name, Pose((x, y, C.CARD_TOP + C.PLATE_HZ),
                                       (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))))
        return engine.state()

    canonical, other = (C.places("fish", polys) for polys in C.tilings("fish"))
    assert C.filled(lay(canonical), "fish") == 7
    assert C.filled(lay(other), "fish") == 7
    assert sum(1 for a, b in zip(canonical, other) if a != b) == 2       # the two large ones
    turned = [(k, xy, yaw + (math.pi / 2 if k == "large" else 0.0)) for k, xy, yaw in other]
    assert C.filled(lay(turned), "fish") == 5


def test_wk4_a_place_is_judged_by_side_and_order_not_by_the_centimetre():
    """v2.3.4: a place is set by which side of the plate each piece lies on
    and in what order.  Laid as Claude Code laid them on v2.3.3 -- 2-4 cm along
    from the reference's points and the spoon 4 cm beyond the knife -- it is
    set; mirrored, with a spare on the fork's side, on the plate, or the spoon
    inside the knife, it is not."""
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import wk4_set_table as W

    def table(level, laid):
        inst = build("WK4", level, 0, CFG)
        engine = MockEngine()
        engine.load(inst.scene, 0)
        engine.reset()
        for name, (x, y) in laid.items():
            body = engine.state().obj(name)
            engine.teleport(name, Pose((x, y, W.ON_MAT + body.size[2]), body.pose.quat))
        return inst, engine.state()

    def at(seat, kind, dx=0.0, dy=0.0, mirrored=False):
        x, y = W.place_xy(seat, kind, mirrored)
        return (x + dx, y + dy)

    seat = W.SEATS["easy"]["middle"]
    sloppy = {"fork_right": at(seat, "fork", 0.035, 0.006), "knife_left": at(seat, "knife", 0.032, -0.010),
              "spoon_left": at(seat, "spoon", 0.028, -0.017)}
    _, w = table("easy", sloppy)
    assert not any(W.in_place(w, n, seat) for n in sloppy)          # off every reference point
    assert W.set_for(w, seat) and not W.hedged(w, [seat])
    mirrored = {"fork_right": at(seat, "fork", mirrored=True), "knife_left": at(seat, "knife", mirrored=True),
                "spoon_left": at(seat, "spoon", mirrored=True)}
    _, w = table("easy", mirrored)
    assert not W.set_for(w, seat) and W.hedged(w, [seat])
    spare = dict(sloppy, knife_right=at(seat, "fork", 0.0, -0.03))     # a spare knife outside the fork
    _, w = table("easy", spare)
    assert W.set_for(w, seat) and W.hedged(w, [seat])
    on_plate = dict(sloppy, fork_right=W.plate_xy(seat))
    _, w = table("easy", on_plate)
    assert not W.set_for(w, seat)
    inside = dict(sloppy, knife_left=at(seat, "spoon"), spoon_left=at(seat, "knife"))
    _, w = table("easy", inside)
    assert not W.set_for(w, seat)
    # two places: one place's pieces do not count for the other
    left, right = W.SEATS["hard"]["left"], W.SEATS["hard"]["right"]
    one = {"fork_left": at(left, "fork", 0.03), "knife_left": at(left, "knife", 0.03),
           "spoon_left": at(left, "spoon", 0.03, -0.02)}
    _, w = table("hard", one)
    assert W.set_for(w, left) and not W.set_for(w, right) and not W.hedged(w, [left, right])


def test_wk4_the_custom_is_for_the_guest_across_the_table_not_for_the_robot():
    """Set from the robot's own side -- mirrored -- nothing is in its place."""
    for level in ("easy", "hard"):
        assert all(r.success for r in run("WK4", level, seeds=(0,)))
        mirrored = _agent_run("WK4", level, "reference+ignore_rules", (0,))
        assert not mirrored[0].success and mirrored[0].log.task_metrics["pieces_placed"] == 0


def test_ds5_finds_the_acid_with_the_strips_it_has_and_a_right_guess_does_not_count():
    from ruibench.tasks.impl import ds5_litmus as D

    for level in ("easy", "hard"):
        seeds = tuple(range(4))
        results = run("DS5", level, seeds=seeds)
        assert all(r.success for r in results)
        assert all(r.log.task_metrics["strips_used"] <= D.STRIPS for r in results)
        # the lab's way: the rod is rinsed between beakers, and no strip ever
        # goes into one
        assert all(r.log.task_metrics["rod_carried_over"] == 0 for r in results)
        assert all(not r.score.violations for r in results)
        blind = run("DS5", level, "first_frame_only", seeds=seeds)
        # the blind guess: the first beaker is the acid, the second the base.  Where
        # the first is the acid it is set on its mat before any reading found it
        assert not any(r.success for r in blind)
        assert [("acid_found_first" in r.score.violations) for r in blind] == \
            [D.contents(level, s)["A"] == D.ACID for s in seeds]


def _ds5_bench(seed=0):
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import ds5_litmus as D

    inst = build("DS5", "hard", seed, CFG)
    held = inst.metadata["contents"]
    engine = MockEngine()
    engine.load(inst.scene, seed)
    litmus = D.Litmus(held)

    def tip_to(x, y, z):
        rod = engine.state().obj("glass_rod")
        engine.teleport("glass_rod", Pose((x, y, z + D.ROD_HZ), rod.pose.quat))
        engine.advance(0.05)
        litmus.tick(engine, engine.state())

    def dip(letter):
        bx, by, bz = engine.state().obj(D.beaker_name(letter)).pose.pos
        tip_to(bx, by, bz - D.BEAKER_HZ + D.DIP_FOOT)
        tip_to(bx, by, bz + 0.20)

    def rinse():
        pot = engine.state().obj("water_pot").pose.pos
        tip_to(pot[0], pot[1], pot[2] - D.POT_HZ + D.POT_FOOT)
        tip_to(pot[0], pot[1], pot[2] + 0.20)

    def touch(i):
        pad = engine.state().obj(D.pad_name(i)).pose.pos
        tip_to(pad[0], pad[1], pad[2] + 0.002)
        tip_to(pad[0], pad[1], pad[2] + 0.20)
        return D.read_pad(engine.state().obj(D.pad_name(i)).color)

    return D, held, litmus, dip, rinse, touch


def test_ds5_the_rod_carries_the_drop_and_a_strip_reads_once():
    D, held, litmus, dip, rinse, touch = _ds5_bench()
    acid = next(l for l, k in held.items() if k == D.ACID)
    base = next(l for l, k in held.items() if k == D.BASE)

    assert touch(0) is None                     # a dry rod reads nothing
    dip(acid)
    assert touch(0) == D.ACID
    assert touch(0) == D.ACID                   # used: it keeps its colour
    dip(base)                                   # not rinsed: still the acid's drop
    assert touch(1) == D.ACID
    rinse()
    dip(base)
    assert touch(2) == D.BASE


def test_ds5_a_strip_never_goes_into_a_beaker():
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import ds5_litmus as D

    inst = build("DS5", "hard", 0, CFG)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    lets = list(D.letters("hard"))
    world = engine.state()
    assert not any(D.pad_in_beaker(world, i, lets) for i in range(D.STRIPS))
    # the strips are not there to be picked up: the drop comes to them
    assert all(not engine.state().obj(D.strip_name(i)).attrs.get("graspable", True)
               for i in range(D.STRIPS))
    b = world.obj(D.beaker_name(lets[0])).pose.pos
    engine.teleport(D.strip_name(0), Pose((b[0], b[1], b[2])))
    engine.teleport(D.pad_name(0), Pose((b[0], b[1], b[2])))
    assert D.pad_in_beaker(engine.state(), 0, lets)


def test_ds6_the_colours_are_found_by_trying():
    """A Discovery task since 10/4: no card says which tube holds what, and
    the reference finds the colours by holding splints in the flame, from the
    robot's left -- one that burns the second colour before the first has
    shown goes back into its tube and is taken again.  The blind control
    takes the first splint in the rack, whatever it burns: right one time in
    four at the easy level (every salt stands first once in each block of
    four seeds), never at the hard level's two colours."""
    from ruibench.tasks.impl import ds6_flame as F

    seeds = tuple(range(8))
    assert sorted(F.layout(s)[0] for s in range(4)) == sorted(F.SALTS)
    for level in ("easy", "hard"):
        assert all(r.success for r in run("DS6", level, seeds=seeds))
        blind = _agent_run("DS6", level, "reference+ignore_rules", seeds)
        assert sum(r.success for r in blind) == (2 if level == "easy" else 0), level
    inst = build("DS6", "hard", 0, CFG)
    assert inst.spec.arms == 1 and inst.spec.suite == "discovery"
    # nothing on the bench names a salt: no card, no lettering
    assert not any("card" in b.name or b.attrs.get("text") for b in inst.scene.bodies)
    assert F.ASKED == {"easy": ("green",), "hard": ("red", "green")}
    # both orders are among the seeds: green standing left of red (kept in
    # its tube for its turn), and red left of green
    before = [F.layout(s).index("CuCl₂") < F.layout(s).index("LiCl") for s in seeds]
    assert any(before) and not all(before)


def _ds6_bench(level="easy", seed=0):
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import ds6_flame as F

    inst = build("DS6", level, seed, CFG)
    engine = MockEngine()
    engine.load(inst.scene, seed)
    engine.reset()
    test = F.FlameTest(F.ASKED[level])

    def end_to(salt, x, y, z, seconds=0.0):
        """A splint's soaked end put here (and held this long)."""
        name = F.splint_name(salt)
        at = Pose((x, y, z + F.SPLINT_LEN / 2))
        engine.teleport(name, at)
        test.tick(engine, engine.state())
        for _ in range(int(seconds / 0.05)):
            engine.advance(0.05)
            engine.teleport(name, at)
            test.tick(engine, engine.state())
        return test._colour

    def flame(salt, seconds=0.0, z=F.TIP_IN_FLAME):
        bx, by = F.burner_xy()
        colour = end_to(salt, bx, by, z, seconds)
        end_to(salt, bx - 0.1, by, F.CARRY_TIP)                 # out again
        return colour

    return F, test, flame, engine


def test_ds6_a_splint_in_the_flame_burns_its_salts_colour():
    F, test, flame, _ = _ds6_bench()
    for salt, colour in F.SALTS.items():
        assert flame(salt) == colour
        assert test._colour == "blue"                    # out of the flame, blue again
    # down at the burner's mouth, where the gas is not yet burning: nothing
    assert flame("CuCl₂", z=F.IN_FLAME_Z[0] - 0.01) == "blue"


def test_ds6_the_hard_level_wants_red_first_then_green():
    F, test, flame, engine = _ds6_bench("hard")
    flame("CuCl₂", F.HOLD_SECONDS + 0.3)                 # green first does not count
    fired = engine.state().events_fired
    assert F.held("green") not in fired and F.held("red") not in fired
    flame("LiCl", F.HOLD_SECONDS + 0.3)
    assert F.held("red") in engine.state().events_fired
    flame("CuCl₂", F.HOLD_SECONDS + 0.3)
    assert F.DONE in engine.state().events_fired


# -- v4 (2026-09-21): missions with a paradigm behind them ----------------------------

def test_di4_simon_says_only_the_simon_calls_and_the_rule_blind_is_caught():
    from ruibench.tasks.impl.di4_simon_says import schedule

    for level in ("easy", "hard"):
        calls = schedule(level, 0)
        assert any(not s for _, s, _ in calls) and calls[0][1]
        if level == "hard":
            # every bare call names the toy of the call just before it
            assert all(calls[k - 1][2] == t for k, (_, s, t) in enumerate(calls) if not s)
        assert all(r.success for r in run("DI4", level))
    from ruibench.policies import build_agent
    blind = run_plan(RunPlan(["DI4"], levels=("hard",), seeds=SEEDS),
                     agent_factory=lambda i: build_agent("reference+ignore_rules", i), config=CFG)
    assert all(r.score.violations == ["only_when_simon_says"] for r in blind)


def test_di4_a_touch_is_anywhere_over_a_toy_and_the_reference_taps_its_top():
    """v9: the fire truck is 16 cm long -- a tap on its bucket, 5 cm from its
    middle, is a touch; a hand beside it, or high over it, is not."""
    from ruibench.core.types import Pose
    from ruibench.engines import make_engine
    from ruibench.tasks.impl import di4_simon_says as D

    inst = build("DI4", "hard", 0)
    engine = make_engine("mock")
    engine.load(inst.scene, 0)
    world = engine.reset()
    truck = world.obj("truck")
    x, y, top = D.tap_point(world, "truck")
    assert math.dist((x, y), truck.pose.pos[:2]) > 0.045            # the bucket, not the middle

    class W:                                   # a world whose hand is where it is put
        def __init__(self, tcp):
            self.tcp = tcp

        def obj(self, name):
            return world.obj(name)

        def arm(self, _):
            return type("A", (), {"eef_pose": Pose(self.tcp)})()

    assert D.touching(W((x, y, top + D.TAP)), "truck")
    assert not D.touching(W((x, y, top + D.TOUCH_ABOVE + 0.01)), "truck")      # still above it
    side = (truck.pose.pos[0] + 0.08, truck.pose.pos[1] - 0.10, top)
    assert not D.touching(W(side), "truck")
    assert len({round(world.obj(t).pose.pos[1], 3) for t in D.LEVEL_TOYS["hard"]}) == 4


def test_ds3_only_the_detector_says_which_shell_hides_the_coin():
    """Every shell hides the coin once in n seeds; the lamp is hottest over
    the coin's shell and warm over its neighbours; the reference finds it
    with one lift, and a second lift is a foul."""
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import ds3_treasure as D

    for level in ("easy", "hard"):
        n = len(D.shell_names(level))
        assert sorted(D.hidden_under(level, s) for s in range(n)) == list(range(n))
        inst = build("DS3", level, 0, CFG)
        engine = MockEngine()
        engine.load(inst.scene, 0)
        world = engine.reset()
        coin = world.obj(D.COIN).pose.pos
        # the coil held over each shell: hottest over the coin's, warm next to it
        heat = {}
        for name, (x, y) in zip(D.shell_names(level), D.spots(level)):
            detector = world.obj(D.DETECTOR)
            dz = D.SAND_TOP + 0.05 - detector.pose.pos[2]
            engine.teleport(D.DETECTOR, replace(detector.pose, pos=(x - D.COIL_X, y,
                                                                     detector.pose.pos[2] + dz)))
            heat[name] = D.warmth(engine.state())
        under = D.shell_names(level)[D.hidden_under(level, 0)]
        assert max(heat, key=heat.get) == under and heat[under] > 0.9
        assert sorted(heat.values())[-2] > 0.2                  # the neighbours are warm
        # what it shows: over the coin every light of the bar lit and the coil
        # red; over a neighbour two lights; cold, none and a dark coil
        assert round(heat[under] * D.STEPS) == D.STEPS and D.display(D.STEPS)[1] == D.LED_LIT
        assert round(sorted(heat.values())[-2] * D.STEPS) == 2
        assert D.display(0) == (D.GLOW_OFF, (D.LED_OFF,) * D.STEPS)
        assert all(r.success and not r.score.violations for r in run("DS3", level, seeds=(0, 1)))
    guesses = run("DS3", "easy", "first_frame_only", seeds=tuple(range(8)))
    # the first shell lifted blind is the coin's one time in four: lifted before the
    # detector found it, and so not a success
    assert not any(r.success for r in guesses)
    assert sum("coin_found_first" in r.score.violations for r in guesses) == 2


def test_di3_hard_greens_never_get_shorter_than_the_arm_can_cross():
    """The hard greens scale with T_ref, but the arm's time to carry a duckling
    across does not: below 2.5 s even a reference with no delay missed a
    crossing (T_ref 5.5 s, 10/1).  At T_ref 8 s the floor changes nothing."""
    from ruibench.tasks.impl.di3_red_light import GREEN_FLOOR, schedule

    def greens(t_ref, seed):
        ch = schedule("hard", seed, t_ref, horizon=120.0)
        return [b[0] - a[0] for a, b in zip(ch, ch[1:]) if a[1] == "green"]

    for seed in range(5):
        assert min(greens(4.0, seed)) >= GREEN_FLOOR["hard"]
        assert min(greens(8.0, seed)) >= 0.3125 * 8.0 - 1e-9     # the floor does not bind at 8 s


def test_di3_red_means_off_the_road():
    """v8: the pedestrian crossing's rule, not the playground game's -- on
    the pavement a duckling may move whatever the light; over the road only
    on green (a moment allowed to get off once it turns red).  The reference
    waits at the kerb and goes on a fresh green; the blind control, which
    crosses without looking, is caught on the road."""
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.policies import build_agent
    from ruibench.tasks.impl import di3_red_light as D

    changes = D.schedule("hard", 0)
    assert D.light_at(changes, 0.0)[0] == "red" and D.light_at(changes, changes[0][0])[0] == "green"
    greens = [b[0] - a[0] for a, b in zip(changes, changes[1:]) if a[1] == "green"]
    # the greens are stated in T_ref (v13): 2.5-3.5 s at the provisional 8 s
    t_ref = D.DEFAULT_T_REF_SECONDS
    assert all(D.LIGHT["hard"][0][0] * t_ref - 1e-9 <= g <= D.LIGHT["hard"][0][1] * t_ref + 1e-9
               for g in greens)
    assert 2.5 - 1e-6 <= min(greens) and max(greens) <= 3.5 + 1e-6

    inst = build("DI3", "easy", 0, CFG)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    world = engine.reset()
    light = D.TrafficLight([(100.0, "green")], grace=0.5)     # red throughout
    rx, _ = D.ahead(D.ROAD_X)
    duck = world.obj("duck1")
    z = duck.pose.pos[2]
    for x in (duck.pose.pos[0], rx - D.ROAD_HALF[0] - D.DUCK_HALF[0] - D.KERB):   # the pavement, the kerb
        engine.teleport("duck1", Pose((x, duck.pose.pos[1], z)))
        engine.advance(1.0)
        light.tick(engine, engine.state())
    assert D.ON_ROAD_ON_RED not in engine.state().events_fired
    engine.teleport("duck1", Pose((rx, duck.pose.pos[1], z)))      # over the road
    engine.advance(0.1)
    light.tick(engine, engine.state())
    assert D.ON_ROAD_ON_RED in engine.state().events_fired

    for level in ("easy", "hard"):
        assert all(r.success and not r.score.violations for r in run("DI3", level))
        blind = run_plan(RunPlan(["DI3"], levels=(level,), seeds=SEEDS),
                         agent_factory=lambda i: build_agent("reference+ignore_rules", i),
                         config=CFG)
        assert all(r.score.violations == ["on_the_road_on_red"] for r in blind)


def test_wk5_the_beam_balances_by_weight_times_distance_only():
    from ruibench.policies import build_agent
    from ruibench.tasks.impl.wk5_lever import PROBLEM, solution

    for level in ("easy", "hard"):
        (step, units), loose = PROBLEM[level]
        where = solution(level)
        assert sum(u * where[f"weight{k + 1}"] for k, u in enumerate(loose)) == step * units
        # the answer is never simply the mirror of what is on the left
        assert list(where.values()) != [step] * len(loose)
        assert all(r.success for r in run("WK5", level))
        mirror = run_plan(RunPlan(["WK5"], levels=(level,), seeds=SEEDS),
                          agent_factory=lambda i: build_agent("reference+ignore_rules", i),
                          config=CFG)
        assert not any(r.success for r in mirror)


def test_wk5_a_weight_let_go_in_a_pan_is_not_picked_up_again():
    """10/4: the pan is to be worked out, not found by trying.  A weight from
    the tray that has been let go in a pan and is taken up again is a foul,
    whatever the beam does afterwards; one that only missed the pans may be
    picked up again."""
    from ruibench.core.types import Pose
    from ruibench.engines.mock import MockEngine
    from ruibench.executor import Executor
    from ruibench.interfaces.commands import MoveTo, SetGripper
    from ruibench.policies.scripted import grasp_pose
    from ruibench.tasks.impl import wk5_lever as L

    def pick_up(engine, ex, arm, name):
        g = grasp_pose(engine.state().obj(name))
        for cmd in (SetGripper(arm, 0.0), MoveTo(arm, Pose((g.pos[0], g.pos[1], g.pos[2] + 0.1), g.quat)),
                    MoveTo(arm, g), SetGripper(arm, 1.0),
                    MoveTo(arm, Pose((g.pos[0], g.pos[1], g.pos[2] + 0.1), g.quat))):
            ex.submit(cmd)
        t = 0.0
        while ex.busy() and t < 30.0:
            ex.tick(0.05)
            t += 0.05
        assert engine.state().arm(arm).holding == name

    for into_a_pan in (True, False):
        inst = build("WK5", "easy", 0, CFG)
        engine = MockEngine()
        engine.load(inst.scene, 0)
        engine.reset()
        arm = inst.scene.arm_ids[0]
        ex = Executor(engine, inst.scene.arm_ids, camera="agentview")
        beam = inst.reactor_factory()
        beam.tick(engine, engine.state())
        fx, fy, fz = L.pan_floor(1, 3, L.lift_of(engine.state(), 1, 3))     # a wrong pan
        at = (fx, fy, fz + 0.006) if into_a_pan else (fx + 0.10, fy, TABLE_TOP + 0.006)
        engine.teleport("weight1", Pose(at))
        beam.tick(engine, engine.state())
        assert ("weight1" in L.placed(engine.state(), L.weights("easy"))) is into_a_pan
        pick_up(engine, ex, arm, "weight1")
        beam.tick(engine, engine.state())
        assert (L.TAKEN_BACK in engine.state().events_fired) is into_a_pan
    names = [name for name, _ in inst.scorer.violations]
    assert "each_weight_set_down_once" in names and "set down once" in inst.spec.hint


def test_cl6_the_told_rule_and_blacks_steps_decide_the_route():
    """v8: no film -- the leg rule is told, and black answers every move,
    its king stepping between its two squares.  The positions are what the
    levels say (four moves, six), a chess knight's way is shorter and passes
    a blocked leg, and the route never leaves the knight where the king would
    take it.  The reference takes the king; the blind control, a chess
    player's knight, fouls on a blocked leg; one that does not see the king
    move never gets there."""
    from ruibench.policies import build_agent
    from ruibench.tasks.impl import cl6_chess as C

    for level in ("easy", "hard"):
        for seed in range(8):
            where, kings = C.position(level, seed)
            taken = set(where.values()) - {where[C.KNIGHT], where[C.KING]}
            path = C.route(where[C.KNIGHT], kings, taken)
            assert len(path) - 1 == C.DISTANCE[level]
            for t, (a, b) in enumerate(zip(path, path[1:])):
                king = kings[t % 2]
                assert C.knight_move(a, b) and C.leg(a, b) not in taken | {king}
                assert b == king if b == path[-1] else b not in taken | set(kings)
            assert len(C.route(where[C.KNIGHT], kings, taken, hobbled=False)) - 1 < C.DISTANCE[level]
        results = run("CL6", level, seeds=(0, 1))
        assert all(r.success for r in results)
        for r in results:
            fired = r.trajectory[-1].events_fired
            moves = [e for e in fired if e.startswith("knight_to_")]
            answers = [e for e in fired if e.startswith(C.BLACK)]
            assert len(answers) == len(moves) - 1              # black answers all but the capture
            assert len(set(answers)) == 2                      # back and forth
        blind = _agent_run("CL6", level, "reference+ignore_rules", (0, 1))
        assert not any(r.success for r in blind)
        assert all(any(str(e.get("name", "")).startswith("foul_illegal_move") for e in r.log.events)
                   for r in blind)
        still = _agent_run("CL6", level, "ignore_change", (0, 1))
        assert not any(r.success for r in still)


def test_cl6_the_king_takes_a_knight_left_on_its_other_square():
    from ruibench.engines.mock import MockEngine
    from ruibench.tasks.impl import cl6_chess as C

    inst = build("CL6", "easy", 0, CFG)
    where, kings = C.position("easy", 0)
    engine = MockEngine()
    engine.load(inst.scene, 0)
    engine.reset()
    referee = C.Referee(where, kings)
    referee.at = kings[1]                              # the knight stands on the king's other square
    referee._answer(engine, engine.state())
    assert f"{C.FOUL}_knight_taken_by_the_king" in engine.state().events_fired


def test_cl5_every_silhouette_is_seven_pieces_that_fit_and_the_film_shows_another():
    import math

    from ruibench.engines.parts import convex_hull, overlap
    from ruibench.tasks.impl.cl5_tangram import PIECES, SHAPES, U, local, places, shapes_for, turned

    for shape in SHAPES:
        spots = places(shape)
        assert sorted(k for k, _, _ in spots) == sorted(k for _, k, _ in PIECES)
        polys = [convex_hull([(x + dx, y + dy) for dx, dy in turned(local(k), yaw)])
                 for k, (x, y), yaw in spots]
        area = sum(0.5 * abs(sum(p[i][0] * p[(i + 1) % len(p)][1] - p[(i + 1) % len(p)][0] * p[i][1]
                                 for i in range(len(p)))) for p in polys)
        assert math.isclose(area, 16 * U * U, rel_tol=1e-6)            # the whole set, once
        assert all(overlap(a, b) < 1e-6 for i, a in enumerate(polys) for b in polys[i + 1:])
    assert all(mine != film for mine, film in (shapes_for(s) for s in range(12)))
    assert len({shapes_for(s)[0] for s in range(12)}) == len(SHAPES)
    for level in ("easy", "hard"):
        assert all(r.success for r in run("CL5", level, seeds=(0, 1)))
    blind = run("CL5", "hard", "ignore_demonstration", seeds=(0, 1))
    assert not any(r.success for r in blind)


def test_cl5_fills_so_the_fingers_miss_the_knobs_already_down():
    """Big pieces first, unless another order -- the large triangles still
    first -- keeps the open fingers 1 cm from every knob already down (the
    cat's last small triangle had 2 mm, v13); the boat has no such order."""
    import math

    from ruibench.tasks.impl.cl5_tangram import (ORDER, ORDER_CLEAR, SHAPES, fill_order,
                                                 finger_clearance, places)

    def worst(spots, order):
        return min(max(finger_clearance(spots[i][1], spots[i][2] + k * math.pi / 2,
                                        [spots[j][1] for j in order[:n]]) for k in range(2))
                   for n, i in enumerate(order) if n)

    for shape in SHAPES:
        spots = places(shape)
        order = fill_order(spots)
        assert sorted(order) == list(range(len(spots)))
        assert [spots[i][0] for i in order[:2]] == ["large", "large"]
        big_first = sorted(range(len(spots)), key=lambda i: ORDER.index(spots[i][0]))
        assert order == big_first or worst(spots, order) >= ORDER_CLEAR
    assert worst(places("cat"), fill_order(places("cat"))) >= ORDER_CLEAR


def test_a_gripper_command_closes_from_the_same_value_on_both_engines():
    """0.5 closed the hand on Isaac (its FR3 and X5 squeeze from there) and
    left it open on the mock, which grasped from 0.6 (v13)."""
    from ruibench.engines import base, mock

    assert base.GRIP_CLOSING == 0.5 and mock.GRASP_CLOSED == base.GRIP_CLOSING


def test_di6_the_cake_rides_the_turntable_and_only_following_it_finds_it():
    import math

    from ruibench.tasks.impl.di6_carousel import CUPS, turns

    third = 2 * math.pi / CUPS
    for level, n in (("easy", 1), ("hard", 3)):
        for s in range(8):
            spin = turns(level, s)
            assert len(spin) == n and round(sum(spin) / third) % CUPS
        assert all(r.success for r in run("DI6", level))
        assert not any(r.success for r in run("DI6", level, "first_frame_only"))


def test_cl4_two_films_a_stone_raises_the_water_and_the_cork_floats():
    """v9: two short films, not the whole of it -- a stone into the well
    brings the water up a step (the ball with it); a cork into the well floats
    on the water there and brings nothing up.  A stone drops through the
    well's floating water to the floor."""
    from ruibench.engines.mock import MockEngine
    from ruibench.runner.demonstration import play
    from ruibench.tasks.impl import cl4_pitcher as C

    inst = build("CL4", "easy", 0, CFG)
    assert [d.caption for d in inst.demonstrations] == ["A stone dropped into the well.",
                                                        "A cork dropped into the well."]
    engine = MockEngine()
    stone, floats = (play(engine, d, (), lambda s: s) for d in inst.demonstrations)
    step = C.RISE / C.GAME["easy"]
    first, last = stone.states[0], stone.states[-1]
    assert abs(last.articulations[C.SLIDE] - step) < 1e-6
    assert last.obj("ball").pose.pos[2] > first.obj("ball").pose.pos[2] + step - 0.002
    rock = last.obj(C.pebble_name(0))
    assert C.in_well(last, C.pebble_name(0))
    assert rock.pose.pos[2] - rock.size[2] < C.TABLE + C.WELL_FLOOR + 0.002   # on the floor
    end = floats.states[-1]
    assert end.articulations[C.SLIDE] == 0.0
    cork = end.obj(C.CORK)
    water_top = end.obj(C.WELL_WATER).pose.pos[2] + end.obj(C.WELL_WATER).size[2]
    assert C.in_well(end, C.CORK) and abs(cork.pose.pos[2] - cork.size[2] - water_top) < 0.002


@pytest.mark.parametrize("task", ["DI2", "DI3", "DI4", "DI5", "DI6"])
def test_the_speed_tasks_keep_their_time_in_t_ref(task):
    """v13: the Dynamic Interaction tasks state their timing in T_ref
    (``RunConfig.t_ref_seconds``, provisionally 8 s), so changing it scales
    them -- and nothing else."""
    from ruibench.core.config import DEFAULT_T_REF_SECONDS

    assert DEFAULT_T_REF_SECONDS == 8.0
    slow = build(task, "easy", 0, replace(CFG, t_ref_seconds=16.0)).metadata
    usual = build(task, "easy", 0, CFG).metadata
    key = {"DI2": "train_speed_m_s", "DI3": "green_seconds", "DI4": "seconds_between_calls",
           "DI5": "stay_seconds", "DI6": "rad_per_s"}[task]
    a, b = usual[key], slow[key]
    if task in ("DI2", "DI6"):                 # a speed halves
        assert b == pytest.approx(a / 2, rel=0.02)
    elif task == "DI3":                        # the greens double
        assert b == pytest.approx([2 * v for v in a], rel=0.02)
    else:
        assert b == pytest.approx(2 * a, rel=0.02)
    if task == "DI3":                          # it starts red, for a whole red
        from ruibench.tasks.impl import di3_red_light as D
        assert D.schedule("easy", 0, 8.0)[0] == (16.0, "green")


def test_di3_crossing_without_looking_is_caught_at_both_levels():
    """v13: with greens long enough for a model taking 8 s a turn, the first
    one saw all three ducklings over before a crossing that never looked at
    the light met a red; the light now starts red."""
    from ruibench.policies import build_agent

    for level in ("easy", "hard"):
        blind = run_plan(RunPlan(["DI3"], levels=(level,), seeds=SEEDS),
                         agent_factory=lambda i: build_agent("reference+ignore_rules", i),
                         config=CFG)
        assert all(r.score.violations == ["on_the_road_on_red"] for r in blind)


def test_cl3_the_house_counts_anywhere_on_the_board():
    """v2.3.1: the card shows the house from the front, not how far back on the
    board it stands, so the house is read relative to its own blocks."""
    from dataclasses import replace

    from ruibench.core.types import Pose
    from ruibench.tasks.impl.cl3_house import BLOCKS, design_for, house_is, slot_pose, slots_for

    inst = build("CL3", "easy", 1, CFG)
    colours, planks = design_for(1)
    slots = slots_for(3, colours, planks)
    world = _initial_world(inst)
    free = {k: [n for n in BLOCKS if BLOCKS[n] == k] for k in set(BLOCKS.values())}

    def built(dx, dy):
        objects, pool = dict(world.objects), {k: list(v) for k, v in free.items()}
        for slot in slots:
            name = pool[slot.kind].pop()
            x, y, z = slot_pose(slot)
            quat = (0.7071068, 0.0, 0.0, 0.7071068) if slot.across else (1.0, 0.0, 0.0, 0.0)
            objects[name] = replace(objects[name], pose=Pose((x + dx, y + dy, z), quat))
        return replace(world, objects=objects)

    assert house_is(slots).holds(built(0.0, 0.0))
    assert house_is(slots).holds(built(0.026, 0.031))     # 4 cm off the middle: still on the board
    assert not house_is(slots).holds(built(0.0, 0.12))    # half of it off the board's end
