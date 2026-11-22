"""v15: the benchmark-side leaks and loopholes the RUIKit collaborator's audit
found, each checked the way the audit
asked: what is hidden is not told, what is private stays with the harness,
what is late is refused, what physics would not do is flagged, and an episode
can be scored again from its log alone."""

import json
import math

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.types import (ArmReading, ObjectState, Pose, VisibilityNotComputed,
                                 WorldState)
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def _engine(task, level="easy", seed=0):
    from ruibench.engines.mock import MockEngine

    inst = build(task, level, seed, CFG)
    engine = MockEngine()
    engine.load(inst.scene, seed)
    engine.reset()
    return inst, engine


# -- §4-5: what an interface under test is given ---------------------------------------
def test_an_interface_under_test_gets_the_public_extras_and_readings_only():
    from ruibench.clock import make_clock
    from ruibench.engines import make_engine
    from ruibench.interfaces.base import (Expressivity, Interface, InterfaceContract,
                                          ModelTurn, ParsedAction)
    from ruibench.interfaces.commands import Done
    from ruibench.policies.base import Agent
    from ruibench.runner.demonstration import Film
    from ruibench.runner.episode import PUBLIC_EXTRAS, EpisodeRunner

    seen = []

    class Probe(Interface):
        id, name, source = "IX", "probe", "test"

        @property
        def contract(self):
            return InterfaceContract(Expressivity.ABSOLUTE, Expressivity.ABSOLUTE,
                                     Expressivity.ABSOLUTE, True, False, False, "none",
                                     ("detect", "segment", "depth_at", "pose_of"), "none",
                                     "per_step")

        def system_prompt(self, ctx):
            return ""

        def observe(self, ctx):
            seen.append(ctx)
            return ModelTurn(ctx.turn_index, "")

        def parse(self, raw, ctx):
            return ParsedAction([Done("probe")])

    class Quiet(Agent):
        name = "quiet"

        def act(self, ctx, turn):
            return ""

    assert Probe.is_benchmark_interface
    for task in ("CL2", "DS1"):
        inst = build(task, "hard", 0, CFG)
        engine = make_engine("mock")
        try:
            EpisodeRunner(engine, Probe(), Quiet(), make_clock("zero_latency"), CFG).run(inst)
        finally:
            engine.close()
        ctx = seen[-1]
        assert set(ctx.extras) == set(PUBLIC_EXTRAS), sorted(ctx.extras)
        films = ctx.extras["demonstrations"]
        assert all(isinstance(f, Film) and not hasattr(f, "states") for f in films)
        assert inst.demonstrations == () or films
        for reading in ctx.observation.proprioception.values():
            assert isinstance(reading, ArmReading) and not hasattr(reading, "holding")


def test_a_held_things_name_stays_with_the_harness():
    from ruibench.core.types import ArmState

    held = ArmState("robot0/right", Pose((0, 0, 1)), 1.0, holding="present2")
    assert held.public() == ArmReading("robot0/right", Pose((0, 0, 1)), 1.0, True, False)


# -- §6: "seen by none" is not "not worked out" --------------------------------------
def test_sight_lines_not_worked_out_are_not_read_as_an_answer():
    o = ObjectState("x", Pose((0, 0, 0)))
    assert o.visible_from is None
    with pytest.raises(VisibilityNotComputed):
        o.seen("agentview")
    inst, engine = _engine("DI6")
    assert engine.state().obj("cake").visible_from is None      # DI6 does not ask
    engine.track_visibility = True
    assert engine.state().obj("cake").visible_from == ()


# -- §7-8: the executor --------------------------------------------------------------
def _executor(task="DS4"):
    from ruibench.executor import Executor

    inst, engine = _engine(task)
    return inst, engine, Executor(engine, inst.scene.arm_ids, camera="agentview")


def test_work_decided_before_a_cancel_is_refused_after_it():
    from ruibench.executor.handles import HandleStatus
    from ruibench.interfaces.commands import MoveTo

    inst, engine, ex = _executor()
    arm = inst.scene.arm_ids[0]
    start = engine.state().arm(arm).eef_pose
    g = ex.generation(arm)
    ex.cancel(arm)
    assert ex.generation(arm) == g + 1
    late = ex.submit(MoveTo(arm, start.translated((0.1, 0.0, 0.0))), generation=g)
    assert late.status is HandleStatus.REJECTED and late.detail.startswith("stale generation")
    ex.tick(1.0)
    assert engine.state().arm(arm).eef_pose.distance_to(start) < 1e-6
    fresh = ex.submit(MoveTo(arm, start.translated((0.05, 0.0, 0.0))), generation=g + 1)
    assert fresh.status is not HandleStatus.REJECTED


def test_what_the_controller_cannot_do_is_refused_and_the_rest_is_done():
    from ruibench.core.geometry import quat_mul
    from ruibench.executor.handles import HandleStatus
    from ruibench.interfaces.commands import MoveDelta, MoveTo

    inst, engine, ex = _executor()
    arm = inst.scene.arm_ids[0]
    start = engine.state().arm(arm).eef_pose
    slow = ex.submit(MoveTo(arm, start, speed=0.5))
    assert slow.status is HandleStatus.REJECTED and "speed" in slow.detail
    assert ex.submit(MoveDelta(arm, frame="nowhere")).status is HandleStatus.REJECTED
    # a turn about the world's vertical, in place
    ex.submit(MoveDelta(arm, drot=(0.0, 0.0, 0.3)))
    ex.tick(1.0)
    now = engine.state().arm(arm).eef_pose
    turn = (math.cos(0.15), 0.0, 0.0, math.sin(0.15))
    assert now.distance_to(start) < 1e-6
    assert all(abs(a - b) < 1e-6 for a, b in zip(now.quat, quat_mul(turn, start.quat)))
    # 5 cm along the hand's own z: the tool points down, so down
    ex.submit(MoveDelta(arm, dpos=(0.0, 0.0, 0.05), frame="eef"))
    ex.tick(2.0)
    after = engine.state().arm(arm).eef_pose
    assert after.pos[2] == pytest.approx(now.pos[2] - 0.05, abs=2e-3)
    # the camera's x is the picture's right
    ex.submit(MoveDelta(arm, dpos=(0.03, 0.0, 0.0), frame="camera"))
    ex.tick(2.0)
    moved = tuple(b - a for a, b in zip(after.pos, engine.state().arm(arm).eef_pose.pos))
    from ruibench.core.geometry import rotate
    right = rotate(engine.camera_pose("agentview").quat, (1.0, 0.0, 0.0))
    assert sum(m * r for m, r in zip(moved, right)) == pytest.approx(0.03, abs=2e-3)


def test_a_refused_command_is_told_a_scripted_agent_at_once_and_any_other_at_its_next_turn():
    """The executor's "no" does not stay in the log.  A scripted agent that
    asks for a speed the controller has not is a broken script, and the run
    stops saying so (a reference that carried at half speed had every carry
    refused and nobody told, 10/5).  Any other agent finds what was refused,
    and why, in its next turn."""
    from ruibench.clock import make_clock
    from ruibench.core.errors import ConfigError
    from ruibench.engines import make_engine
    from ruibench.interfaces import make_interface
    from ruibench.interfaces.commands import Done, MoveTo
    from ruibench.interfaces.direct import encode
    from ruibench.policies.base import Agent, ScriptedPolicy
    from ruibench.runner.episode import EpisodeRunner

    inst = build("DS4", "easy", 0, CFG)

    def half_speed(ctx):
        arm = ctx.arm_ids[0]
        up = ctx.extras["world"].arm(arm).eef_pose.translated((0.0, 0.0, 0.05))
        return [MoveTo(arm, up, speed=0.5)]

    class Scripted(ScriptedPolicy):
        name = "half_speed"

        def commands(self, world, ctx):
            return half_speed(ctx)

    told = []

    class Speaks(Agent):          # writes the direct interface's JSON itself
        name = "speaks"

        def act(self, ctx, turn):
            told.append(ctx.extras["refused"])
            return encode(half_speed(ctx) if ctx.turn_index == 0 else [Done("heard")])

    def run(agent):
        engine = make_engine("mock")
        try:
            return EpisodeRunner(engine, make_interface("I0_direct"), agent,
                                 make_clock("zero_latency"), CFG).run(inst)
        finally:
            engine.close()

    with pytest.raises(ConfigError, match="speed 0.5 is not supported"):
        run(Scripted())
    result = run(Speaks())
    assert told[0] == ()
    assert [r["kind"] for r in told[1]] == ["MoveTo"] and "speed" in told[1][0]["why"]
    rejected = [h for h in result.log.handles if h["status"] == "rejected"]
    assert len(rejected) == 1                       # and it is in the log, as before



def _run_out(ex, limit=40.0):
    t = 0.0
    while ex.busy() and t < limit:
        ex.tick(0.05)
        t += 0.05


def test_a_wait_is_a_pause_where_it_stands():
    """"Move down, wait 0.3 s, open" opens 0.3 s after the hand is down (a Wait
    used to be a timer from when it was submitted, and the gripper opened the
    moment the hand arrived: the RUIKit collaborator, WK3)."""
    from ruibench.interfaces.commands import MoveTo, SetGripper, Wait

    inst, engine, ex = _executor()
    arm = inst.scene.arm_ids[0]
    start = engine.state().arm(arm).eef_pose
    move = ex.submit(MoveTo(arm, start.translated((0.0, 0.0, -0.05))))
    assert ex.submit(Wait(0.3)) is None
    grip = ex.submit(SetGripper(arm, 0.0))
    _run_out(ex)
    assert move.succeeded and grip.succeeded
    assert grip.started_at >= move.finished_at + 0.3 - 1e-6
    assert grip.started_at <= move.finished_at + 0.3 + 0.05 + 1e-6
    # on its own, the world runs that long
    t0 = engine.sim_time
    ex.submit(Wait(0.5))
    assert ex.busy() and ex.pending_wait == pytest.approx(0.5)
    _run_out(ex)
    assert engine.sim_time - t0 == pytest.approx(0.5, abs=1e-6)
    # a long one is not an action that times out
    t0 = engine.sim_time
    ex.submit(Wait(30.0))
    _run_out(ex, limit=60.0)
    assert engine.sim_time - t0 == pytest.approx(30.0, abs=1e-6)


def test_a_wait_holds_every_arm_until_all_have_got_to_it():
    from ruibench.interfaces.commands import MoveTo, SetGripper, Wait

    inst, engine, ex = _executor("WK4")
    left, right = inst.scene.arm_ids[:2]
    far = ex.submit(MoveTo(left, engine.state().arm(left).eef_pose.translated((0.0, 0.0, 0.08))))
    ex.submit(MoveTo(right, engine.state().arm(right).eef_pose.translated((0.0, 0.0, 0.01))))
    ex.submit(Wait(0.2))
    grip = ex.submit(SetGripper(right, 1.0))
    _run_out(ex)
    assert far.succeeded and grip.started_at >= far.finished_at + 0.2 - 1e-6


def test_a_wait_for_one_arm_holds_only_that_arm():
    """Two hands' work in one batch, each with its own pause (WK4's reference
    sets the table with both): neither waits for the other."""
    from ruibench.interfaces.commands import MoveTo, SetGripper, Wait
    from ruibench.interfaces.direct import _CODECS, encode

    inst, engine, ex = _executor("WK4")
    left, right = inst.scene.arm_ids[:2]
    far = ex.submit(MoveTo(left, engine.state().arm(left).eef_pose.translated((0.0, 0.0, 0.08))))
    ex.submit(Wait(1.0, left))
    near = ex.submit(MoveTo(right, engine.state().arm(right).eef_pose.translated((0.0, 0.0, 0.01))))
    ex.submit(Wait(0.2, right))
    grip = ex.submit(SetGripper(right, 1.0))
    _run_out(ex)
    assert grip.started_at < far.finished_at
    assert grip.started_at >= near.finished_at + 0.2 - 1e-6
    # through I0's JSON and back, the arm is kept
    import json
    [d] = json.loads(encode([Wait(0.3, right)]))
    assert _CODECS[d["op"]](d) == Wait(0.3, right)


def test_a_cancel_drops_a_pause_and_what_follows_it():
    from ruibench.executor.handles import HandleStatus
    from ruibench.interfaces.commands import MoveTo, Wait

    inst, engine, ex = _executor()
    arm = inst.scene.arm_ids[0]
    start = engine.state().arm(arm).eef_pose
    ex.submit(Wait(1.0))
    later = ex.submit(MoveTo(arm, start.translated((0.05, 0.0, 0.0))))
    ex.tick(0.2)
    ex.cancel(arm)
    _run_out(ex)
    assert later.status is HandleStatus.CANCELLED and not ex.busy() and ex.pending_wait == 0
    assert engine.state().arm(arm).eef_pose.distance_to(start) < 1e-6

# -- §9: the physics audit -----------------------------------------------------------
def _still(t, pos, vel=(0.0, 0.0, 0.0), joints=None):
    from ruibench.core.types import ArmState, RobotState

    arms = {"right": ArmState("robot0/right", Pose((0.0, 0.0, 1.2)))}
    return WorldState(t, {"box": ObjectState("box", Pose(pos), lin_vel=vel)},
                      {"robot0": RobotState("robot0", arms)}, articulations=joints or {})


def test_the_audit_flags_what_physics_does_not_do():
    from ruibench.scoring.audit import audit

    rest = [_still(t / 10, (0.5, 0.0, 0.85)) for t in range(10)]
    assert audit(rest) == []
    teleported = rest + [_still(1.0, (0.5, 0.2, 0.85))]            # 20 cm, at rest, untouched
    assert any(f.startswith("jumped:box") for f in audit(teleported))
    thrown = rest + [_still(1.0, (1.5, 0.0, 0.85), vel=(10.0, 0, 0))]
    assert any(f.startswith("too_fast:box") for f in audit(thrown))
    assert any(f.startswith("nan:box") for f in audit(rest + [_still(1.0, (math.nan, 0, 0))]))
    bent = [_still(0.0, (0.5, 0, 0.85), joints={"lid": 0.4})]
    assert any(f.startswith("joint:lid") for f in audit(bent, {"lid": (0.0, 0.3)}))
    # the mock has no physics: its straight drops are not jumps
    assert audit(teleported, dynamics=False) == []


def test_a_flagged_episode_is_a_failure():
    from ruibench.scoring import Custom, Finally, Scorer
    from ruibench.scoring.failure import FailureLabel

    scorer = Scorer("T", "easy", Finally(Custom(lambda w: True, "anything")))
    rest = [_still(t / 10, (0.5, 0.0, 0.85)) for t in range(10)]
    assert scorer.score(rest).success
    flagged = scorer.score(rest + [_still(1.0, (0.5, 0.2, 0.85))])
    assert not flagged.success and flagged.failure is FailureLabel.PHYSICS_FLAG and flagged.flags


def test_a_leaked_secret_voids_the_episode():
    from ruibench.scoring import Custom, Finally, Scorer
    from ruibench.scoring.failure import FailureLabel

    scorer = Scorer("T", "easy", Finally(Custom(lambda w: True, "anything")))
    rest = [_still(t / 10, (0.5, 0.0, 0.85)) for t in range(10)]
    leaked = scorer.score(rest, {"leaks": 2})
    assert not leaked.success and leaked.failure is FailureLabel.SECRET_LEAK
    assert leaked.flags[0] == "secret_leak:2"
    assert scorer.score(rest, {"leaks": 0}).success


def test_references_on_the_mock_earn_no_flags():
    from ruibench.policies.agents import build_agent
    from ruibench.runner import RunPlan, run_plan

    for task in ("WK3", "DI1", "DI2", "CL5"):
        [r] = run_plan(RunPlan([task], levels=("hard",), seeds=(0,)),
                       agent_factory=lambda inst: build_agent("reference", inst), config=CFG)
        assert r.success and not r.score.flags, (task, r.score.flags)


def test_every_task_body_has_its_physx_caps():
    from ruibench.engines.isaac import scene_builder as S

    assert 0 < S.MAX_DEPENETRATION <= 5.0 and S.MAX_CONTACT_IMPULSE > 0
    src = open(S.__file__).read()
    assert "max_depenetration_velocity=MAX_DEPENETRATION" in src
    assert "max_contact_impulse=MAX_CONTACT_IMPULSE" in src


# -- §11: scored again from the log alone ---------------------------------------------
def test_an_episode_scores_again_from_its_log_alone(tmp_path):
    from ruibench.cli import main
    from ruibench.policies.agents import build_agent
    from ruibench.runner import RunPlan, run_plan
    from ruibench.runner.snapshot import is_whole, rescore

    out = tmp_path / "x.jsonl"
    run_plan(RunPlan(["DS2", "DI4"], levels=("hard",), seeds=(1,)),
             agent_factory=lambda inst: build_agent("reference", inst), config=CFG, out=str(out))
    logs = [json.loads(line) for line in out.read_text().splitlines()]
    for log in logs:
        assert log["schema"] == "ruibench-log/2" and is_whole(log["trajectory"])
        new, diffs = rescore(log)
        assert new.success == log["score"]["success"] and not diffs
    assert main(["rescore", str(out)]) == 0
    # tampered: the egg never reached the basket -- the log's own trajectory says so
    log = logs[0]
    for snap in log["trajectory"][len(log["trajectory"]) // 2:]:
        snap["objects"]["egg"]["pos"] = log["trajectory"][0]["objects"]["egg"]["pos"]
    new, diffs = rescore(log)
    assert not new.success and any(d.startswith("success") for d in diffs)
    out.write_text(json.dumps(log) + "\n")
    assert main(["rescore", str(out)]) == 1



def test_an_event_fired_again_is_kept():
    """A mole whacked twice is two whacks: the snapshots keep every event the
    engine fired, repeats included (the first Isaac DI5 logs of v15 lost them),
    and a list that starts over is kept whole."""
    from dataclasses import replace

    from ruibench.runner.snapshot import dump_all, load_all

    inst, engine = _engine("DI5")
    w = engine.state()
    fired = [(), ("mole_left0_up",), ("mole_left0_up", "mole_left0_whacked"),
             ("mole_left0_up", "mole_left0_whacked", "mole_left0_up", "mole_left0_whacked"),
             ("restart",)]
    traj = [replace(w, sim_time=0.1 * i, events_fired=f) for i, f in enumerate(fired)]
    back = load_all(json.loads(json.dumps(dump_all(traj, inst.scene))), inst.scene)
    assert [b.events_fired for b in back] == fired

# -- §12: secrets and the canary --------------------------------------------------------
def test_every_secret_is_specific_and_carries_the_canary():
    from ruibench.core.canary import CANARY
    from ruibench.tasks.instance import implemented
    from ruibench.tasks.loader import get

    for task in implemented():
        for level in ("easy", "hard"):
            for seed in range(4):
                inst = build(task, level, seed, CFG)
                shown = " ".join([inst.instruction, inst.hint]
                                 + [d.caption for d in inst.demonstrations]
                                 + [b.category for b in inst.scene.bodies]
                                 + [str(b.color) for b in inst.scene.bodies]).lower()
                assert CANARY in inst.secrets
                for s in inst.secrets:
                    assert len(s) >= 4 and s.lower() not in shown, (task, level, seed, s)
                if get(task).secrets:
                    assert len(inst.secrets) > 1, (task, "a card that names secrets gives some")


def test_the_canary_is_in_the_cards_and_the_logs(tmp_path):
    from pathlib import Path

    from ruibench.core.canary import CANARY_GUID
    from ruibench.policies.agents import build_agent
    from ruibench.runner import RunPlan, run_plan
    from ruibench.tasks import loader

    for card in Path(loader.CARDS_DIR).glob("*.yaml"):
        assert CANARY_GUID in card.read_text().splitlines()[0], card
    out = tmp_path / "x.jsonl"
    run_plan(RunPlan(["WK1"], levels=("easy",), seeds=(0,)),
             agent_factory=lambda inst: build_agent("reference", inst), config=CFG, out=str(out))
    assert json.loads(out.read_text().splitlines()[0])["provenance"]["canary"] == CANARY_GUID


# -- §15-16: task fixes ------------------------------------------------------------------
def test_the_tart_under_its_cover_is_hidden_on_the_mock_too():
    """§15: drawn opaque shells block sight lines (``sight_boxes``)."""
    for level in ("easy", "hard"):
        for seed in range(3):
            inst, engine = _engine("DI6", level, seed)
            for cam in (c.name for c in inst.scene.cameras):
                assert "cake" not in engine.visible_objects(cam), (level, seed, cam)


def test_a_beaker_carried_over_the_strips_does_not_dip_them():
    """§16: a strip is in a beaker when its pad is over the beaker's floor and
    under its rim -- not whenever it is under the rim of a beaker in the air."""
    from dataclasses import replace

    from ruibench.tasks.impl import ds5_litmus as D

    inst, engine = _engine("DS5", "hard", 2)
    w = engine.state()
    letters = D.letters("hard")
    pad = w.obj(D.pad_name(0)).pose.pos
    name = D.beaker_name(letters[0])
    beaker = w.obj(name)

    def with_beaker_at(pos):
        return replace(w, objects={**w.objects, name: replace(beaker, pose=Pose(pos, beaker.pose.quat))})

    over = with_beaker_at((pad[0], pad[1], pad[2] + 0.15 + D.BEAKER_HZ))
    assert not D.pad_in_beaker(over, 0, letters)
    around = with_beaker_at((pad[0], pad[1], pad[2] + D.BEAKER_HZ - 0.004))
    assert D.pad_in_beaker(around, 0, letters)
    tip = D.rod_tip(w)
    assert not D.in_beaker(with_beaker_at((tip[0], tip[1], tip[2] + 0.15 + D.BEAKER_HZ)), tip,
                           letters[0])
