"""Harness semantics: cancellation, event triggers, clocks, negative controls."""

import pytest

from ruibench.clock import ControlledLatencyClock, MeasuredClock, ZeroLatencyClock
from ruibench.core.config import RunConfig
from ruibench.core.errors import ConfigError
from ruibench.core.types import Pose
from ruibench.engines import make_engine
from ruibench.engines.base import BodySpec, SceneSpec
from ruibench.events import Event, EventScheduler, ModelCallTrigger, SetInstruction
from ruibench.executor import Executor, HandleStatus
from ruibench.policies import wrap
from ruibench.runner import RunPlan, run_plan


@pytest.fixture
def engine():
    e = make_engine("mock")
    e.load(SceneSpec("t", bodies=(BodySpec("cube", pose=Pose((0.1, 0.0, 0.82))),)), seed=0)
    return e


def test_cancelled_work_never_resumes(engine):
    """Gate B: after a cancel, the old queue must not come back to life."""
    ex = Executor(engine, ["robot0/right"])
    handles = [ex.submit(c) for c in [
        __import__("ruibench.interfaces.commands", fromlist=["MoveTo"]).MoveTo(
            "robot0/right", Pose((0.3, 0.3, 1.0))),
        __import__("ruibench.interfaces.commands", fromlist=["MoveTo"]).MoveTo(
            "robot0/right", Pose((-0.2, 0.3, 1.0))),
    ]]
    ex.tick(0.2)
    ex.cancel("robot0/right")
    assert all(h.status is HandleStatus.CANCELLED for h in handles)
    ex.tick(3.0)
    assert not ex.busy()
    assert all(h.status is HandleStatus.CANCELLED for h in handles)


def test_main_board_events_cannot_fire_on_a_model_call_count():
    """Tying an event to the k-th model call makes a fast interface meet a
    different world from a slow one -- the exact confound the suite avoids."""
    event = Event("bad", ModelCallTrigger(3), SetInstruction("x"))
    with pytest.raises(ConfigError, match="not allowed on the main board"):
        EventScheduler([event])
    EventScheduler([Event("ok", ModelCallTrigger(3), SetInstruction("x"), diagnostic=True)])


def test_clock_semantics():
    assert ZeroLatencyClock().sim_seconds_for(12.0) == 0.0
    assert ControlledLatencyClock(20.0).sim_seconds_for(3.0) == 20.0   # same world for everyone
    assert MeasuredClock().sim_seconds_for(3.5) == 3.5


def test_di4_reference_succeeds_at_both_levels():
    """Layer 1 of the three-layer solvability check: the task is solvable at all."""
    cfg = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
    results = run_plan(RunPlan(["DI4"], levels=("easy", "hard"), seeds=(0, 1, 2)),
                       agent_factory=lambda i: i.reference_factory(i), config=cfg)
    assert all(r.success for r in results), [
        (r.score.level, r.score.failure.value) for r in results]


@pytest.mark.parametrize("control,expected", [
    ("do_nothing", "the task requires acting at all"),
    ("ignore_change", "the task requires noticing the mid-episode change"),
])
def test_negative_controls_fail_rp3(control, expected):
    """Each control is checked only against the information it removes."""
    cfg = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
    results = run_plan(RunPlan(["DI4"], levels=("easy",), seeds=(0, 1)),
                       agent_factory=lambda i: wrap(control, i.reference_factory(i)),
                       config=cfg)
    assert not any(r.success for r in results), expected


def test_ignore_change_fails_for_the_right_reason():
    """Not merely failing -- failing because the calls it never heard were
    never done (DI4, Simon says: every call is a change of instruction)."""
    cfg = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
    r = run_plan(RunPlan(["DI4"], levels=("easy",), seeds=(0,)),
                 agent_factory=lambda i: wrap("ignore_change", i.reference_factory(i)),
                 config=cfg)[0]
    assert not r.success and not r.score.violations
    assert "game_over" in r.log.milestones_hit
    assert "a_simon_call_done" not in r.log.milestones_hit


def test_event_fires_on_a_physical_milestone():
    """No task on the v4 board uses a milestone-triggered event any more (the
    old DI4's change of mind did); the scheduler still has to honour one."""
    from ruibench.engines import make_engine
    from ruibench.events import Event, MilestoneTrigger, SetInstruction
    from ruibench.events.scheduler import EventScheduler
    from ruibench.tasks.instance import build

    engine = make_engine("mock")
    engine.load(build("DI4", "easy", 0).scene, seed=0)
    sched = EventScheduler([Event("change", MilestoneTrigger("grasped"), SetInstruction("Now this."))])
    assert not sched.tick(engine, engine.state(), (), 0)
    engine.advance(0.5)
    fired = sched.tick(engine, engine.state(), ("grasped",), 1)
    assert [r.name for r in fired] == ["change"] and "milestone" in fired[0].trigger
    assert engine.state().instruction == "Now this." and engine.state().instruction_version == 1
    assert not sched.tick(engine, engine.state(), ("grasped",), 2)          # once


def test_latency_shows_up_as_observation_age():
    cfg = RunConfig(engine="mock", interface="I0_direct", clock="controlled",
                    clock_latency_seconds=15.0)
    r = run_plan(RunPlan(["DI4"], levels=("easy",), seeds=(0,)),
                 agent_factory=lambda i: i.reference_factory(i), config=cfg)[0]
    assert r.score.metrics["observation_age_mean"] == pytest.approx(15.0, abs=0.5)


def test_a_move_does_not_wait_for_a_twitching_gripper(engine):
    """On a physics backend, fingers closed on an object keep twitching while
    the arm moves.  A move must complete on pose alone -- tying it to the
    gripper made every carry time out after 20 s on Isaac."""
    from ruibench.interfaces.commands import MoveTo, SetGripper

    engine.gripper_settled = lambda arm: False          # fingers never "stop"
    ex = Executor(engine, ["robot0/right"])
    move = ex.submit(MoveTo("robot0/right", Pose((0.2, 0.1, 1.0))))
    ex.tick(5.0)
    assert move.status is HandleStatus.DONE

    grip = ex.submit(SetGripper("robot0/right", 1.0))
    ex.tick(1.0)
    assert grip.status is not HandleStatus.DONE, "a gripper action waits for the fingers"


def test_a_blocked_move_fails_at_once_rather_than_timing_out(engine):
    """On a physics backend an object in the way stops the arm.  Waiting out a
    20 s timeout would change what a timing-sensitive task measures."""
    from ruibench.interfaces.commands import MoveTo

    goal = Pose((0.4, 0.3, 1.0))
    engine.arm_at_target = lambda arm, *a, **k: False
    real_state = engine.state
    frozen = real_state()

    class Frozen:                         # the arm never moves again
        def __getattr__(self, name):
            return getattr(frozen, name)

    engine.state = lambda: frozen
    ex = Executor(engine, ["robot0/right"])
    move = ex.submit(MoveTo("robot0/right", goal))
    ex.tick(3.0)
    assert move.status is HandleStatus.FAILED
    assert "blocked" in move.detail
    assert move.finished_at - move.started_at < 2.0, "fails in about a second, not twenty"


def test_a_move_blocked_by_its_turn_says_how_far_it_is_turned(engine):
    """A hand a millimetre from its target but turned short of it -- an FR3
    wrist at the end of its travel -- is not there, and the message says what
    is missing: not "0.1 cm short", as if something were in the way."""
    import math

    from ruibench.core.geometry import quat_mul
    from ruibench.interfaces.commands import MoveTo

    frozen = engine.state()
    here = frozen.arm("robot0/right").eef_pose
    turn = (math.cos(0.1), 0.0, 0.0, math.sin(0.1))      # 0.2 rad about z
    goal = Pose((here.pos[0] + 0.001, here.pos[1], here.pos[2]), quat_mul(turn, here.quat))
    engine.arm_at_target = lambda arm, *a, **k: False
    engine.state = lambda: frozen
    ex = Executor(engine, ["robot0/right"])
    move = ex.submit(MoveTo("robot0/right", goal))
    ex.tick(3.0)
    assert move.status is HandleStatus.FAILED
    assert move.detail == "blocked: stopped 0.1 cm and 11 degrees short of the target"


def test_a_hand_turning_in_place_is_not_blocked(engine):
    """A wrist turning in place moves the hand less than a millimetre a second:
    turning toward the target is progress, and the move goes on to its end
    (v1.17: on Isaac the joint interface's turns of CR2's bulb were stopped as
    blocked mid-turn)."""
    import dataclasses
    import math

    from ruibench.core.geometry import quat_mul
    from ruibench.interfaces.commands import MoveTo

    frozen = engine.state()
    start = frozen.arm("robot0/right").eef_pose
    about_z = lambda a: (math.cos(a / 2), 0.0, 0.0, math.sin(a / 2))   # noqa: E731
    goal = Pose(start.pos, quat_mul(about_z(1.5), start.quat))
    t0 = engine.sim_time

    def turned() -> Pose:                         # 0.6 rad/s about z, in place
        return Pose(start.pos, quat_mul(about_z(min(1.5, 0.6 * (engine.sim_time - t0))),
                                        start.quat))

    class Turning:
        def __getattr__(self, name):
            return getattr(frozen, name)

        def arm(self, arm):
            return dataclasses.replace(frozen.arm(arm), eef_pose=turned())

    engine.arm_at_target = lambda arm, *a, **k: turned().angle_to(goal) < 0.01
    engine.state = Turning
    ex = Executor(engine, ["robot0/right"])
    move = ex.submit(MoveTo("robot0/right", goal))
    ex.tick(3.0)
    assert move.status is HandleStatus.DONE, move.detail
    assert move.finished_at - move.started_at > 2.4, "turned all the way, 2.5 s"


def test_requested_rates_reach_the_engine_or_fail_loudly():
    """A rate the config asks for is either applied or refused -- never just
    written into the log while the engine runs at its own."""
    from ruibench.runner.harness import engine_options

    assert engine_options(RunConfig(engine="mock")) == {}
    assert engine_options(RunConfig(engine="isaac", sim_hz=240.0)) == {"sim_hz": 240.0}
    with pytest.raises(ConfigError):
        engine_options(RunConfig(engine="mock", sim_hz=240.0))


def test_the_log_records_what_the_engine_ran_with():
    cfg = RunConfig(engine="mock", camera_width=128, camera_height=96)
    r = run_plan(RunPlan(["WK5"], levels=("easy",), seeds=(0,)),
                 agent_factory=lambda i: i.reference_factory(i), config=cfg)[0]
    assert r.log.engine_setup["name"] == "mock"
    assert r.log.config["sim_hz"] is None           # asked for nothing in particular

    from ruibench.runner.episode import _sized
    from ruibench.tasks.instance import build

    scene = _sized(build("WK5").scene, cfg)         # what the engine is handed
    assert {(c.width, c.height) for c in scene.cameras} == {(128, 96)}
