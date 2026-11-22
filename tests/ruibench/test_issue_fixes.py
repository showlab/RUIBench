"""The second round of the RUIKit collaborator's benchmark-side findings
(#17-#21), each checked the way it
asks: a closed hand pushed onto something is not holding it, the fingers'
real opening is reported, WK5 starts where its load puts it, every body is
found in Isaac's segmentation whatever its name's case, and depth on CL2's
bed and sofa reads where things rest on them."""

import math

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.types import Pose
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def _engine(task, level="easy", seed=0):
    from ruibench.engines.mock import MockEngine
    from ruibench.executor import Executor

    inst = build(task, level, seed, CFG)
    engine = MockEngine()
    engine.load(inst.scene, seed)
    engine.reset()
    return inst, engine, Executor(engine, inst.scene.arm_ids, camera="agentview")


def _run_out(ex, limit=30.0):
    t = 0.0
    while ex.busy() and t < limit:
        ex.tick(0.05)
        t += 0.05


# -- #21: holding ---------------------------------------------------------------------
def test_a_closed_hand_pushed_onto_something_does_not_hold_it():
    """The collaborator's check: closed empty fingers pressed on a body, then
    lifted -- ``holding`` is None and the body stays where it was."""
    from ruibench.interfaces.commands import MoveTo, SetGripper

    inst, engine, ex = _engine("DS4")
    arm = inst.scene.arm_ids[0]
    coin = next(b.name for b in inst.scene.bodies if "coin" in b.category)
    at = engine.state().obj(coin).pose.pos
    hand = engine.state().arm(arm).eef_pose
    above = Pose((at[0], at[1], at[2] + 0.10), hand.quat)
    on = Pose((at[0], at[1], at[2]), hand.quat)
    for cmd in (MoveTo(arm, above), SetGripper(arm, 1.0), MoveTo(arm, on), MoveTo(arm, above)):
        ex.submit(cmd)
    _run_out(ex)
    w = engine.state()
    assert w.arm(arm).holding is None and not w.obj(coin).is_held
    assert math.dist(w.obj(coin).pose.pos, at) < 1e-3
    # and taken the usual way -- open, down, close -- it is held
    for cmd in (SetGripper(arm, 0.0), MoveTo(arm, on), SetGripper(arm, 1.0), MoveTo(arm, above)):
        ex.submit(cmd)
    _run_out(ex)
    assert engine.state().arm(arm).holding == coin


# -- #17: the fingers' opening ---------------------------------------------------------------
def test_the_fingers_opening_is_measured_not_commanded():
    from ruibench.engines.base import arm_camera_model
    from ruibench.interfaces.commands import MoveTo, SetGripper

    inst, engine, ex = _engine("DS4")
    arm = inst.scene.arm_ids[0]
    widest = arm_camera_model(inst.scene, arm).max_opening
    assert engine.state().arm(arm).opening == pytest.approx(widest)
    assert engine.state().arm(arm).public().opening == pytest.approx(widest)
    coin = next(b for b in inst.scene.bodies if "coin" in b.category)
    at = engine.state().obj(coin.name).pose.pos
    hand = engine.state().arm(arm).eef_pose
    for cmd in (MoveTo(arm, Pose(at, hand.quat)), SetGripper(arm, 1.0)):
        ex.submit(cmd)
    _run_out(ex)
    reading = engine.state().arm(arm)
    assert reading.holding == coin.name and reading.gripper == pytest.approx(1.0)
    assert reading.opening == pytest.approx(2 * coin.size[0], abs=1e-3)   # a round coin, across
    ex.submit(SetGripper(arm, 0.0))
    ex.submit(MoveTo(arm, hand))
    ex.submit(SetGripper(arm, 1.0))
    _run_out(ex)
    assert engine.state().arm(arm).opening == 0.0                      # closed on nothing


# -- #18: WK5 starts leaning ----------------------------------------------------------------
@pytest.mark.parametrize("level", ["easy", "hard"])
def test_wk5_starts_where_its_load_puts_the_beam(level):
    from ruibench.tasks.impl import wk5_lever as W

    inst, engine, _ = _engine("WK5", level)
    w = engine.state()
    assert w.articulations[W.HINGE] == pytest.approx(W.TILT)              # the left down
    (step, _), _ = W.PROBLEM[level]
    assert W.placed(w, ["left_weight"]) == {"left_weight": (-1, step)}
    beam = inst.reactor_factory()
    for _ in range(40):                                                   # 2 s: nothing turns
        beam.tick(engine, engine.state())
        engine.advance(0.05)
    assert engine.state().articulations[W.HINGE] == pytest.approx(W.TILT)


# -- #19: names in Isaac's segmentation -------------------------------------------------------
def test_every_body_is_found_in_segmentation_whatever_its_case():
    """Isaac hands class names back lower-cased: WK5's ``pan_L1`` and DS5's
    ``beaker_A`` came back as ``pan_l1`` and ``beaker_a`` and were never seen."""
    from dataclasses import replace

    from ruibench.core.errors import ConfigError
    from ruibench.engines.isaac.engine import IsaacEngine
    from ruibench.tasks.instance import implemented

    for task in implemented():
        for level in ("easy", "hard"):
            scene = build(task, level, 0, CFG).scene
            names = IsaacEngine._segmentation_names(scene)
            assert all(names[b.name.lower()] == b.name for b in scene.bodies), task
    scene = build("WK5", "easy", 0, CFG).scene
    assert IsaacEngine._segmentation_names(scene)["pan_l1"] == "pan_L1"
    twin = replace(scene.bodies[0], name=scene.bodies[0].name.upper())
    with pytest.raises(ConfigError, match="differ only in case"):
        IsaacEngine._segmentation_names(replace(scene, bodies=tuple(scene.bodies) + (twin,)))


# -- #20: CL2's bed and sofa -------------------------------------------------------------------
def test_depth_on_the_bed_and_sofa_reads_where_things_rest():
    """The collaborator's check, on the geometry depth is read from (RUIKit's
    estimated world uses the same boxes): the ray from the fixed camera to the
    middle of the mattress (and of the seat) meets it within 5 mm of where a
    doll laid there rests."""
    from ruibench.engines.sight import sight_boxes
    from ruibench.tasks.impl import cl2_names as C
    from ruibench.tasks.impl._common import TABLE

    inst, engine, _ = _engine("CL2", "hard")
    world = engine.state()
    specs = {b.name: b for b in inst.scene.bodies}
    boxes = [box for n, o in world.objects.items()
             for box in sight_boxes(specs[n].shape, specs[n].size, o.pose, specs[n].attrs)]
    eye = engine.camera_pose("agentview").pos

    def first_hit(target):
        """Where the ray from the camera through ``target`` first meets a box
        or the table -- what a depth reading along it gives."""
        d = [t - e for t, e in zip(target, eye)]
        best = (TABLE - eye[2]) / d[2]
        for box in boxes:
            t0, t1 = 0.0, best
            for k in range(3):
                if abs(d[k]) < 1e-12:
                    if not box[k][0] <= eye[k] <= box[k][1]:
                        t0, t1 = 1.0, 0.0
                    continue
                a, b = (box[k][0] - eye[k]) / d[k], (box[k][1] - eye[k]) / d[k]
                t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
            if t0 <= t1:
                best = t0
        return [e + best * v for e, v in zip(eye, d)]

    for name, floor, middle in ((C.BED, C.BED_FLOOR, (0.0, 0.0)), (C.SOFA, C.SOFA_FLOOR, (0.0, -0.35))):
        body = specs[name]
        (x, y, _), (hx, hy, _) = body.pose.pos, body.size
        spot = (x + middle[0] * hx, y + middle[1] * hy, TABLE + floor)
        lifted = first_hit(spot)
        assert abs(lifted[2] - spot[2]) < 0.005, (name, lifted, spot)
    # and a doll the reference lays in the bed rests on the mattress
    from ruibench.policies.agents import build_agent
    from ruibench.runner import RunPlan, run_plan

    [r] = run_plan(RunPlan(["CL2"], levels=("easy",), seeds=(0,)),
                   agent_factory=lambda inst: build_agent("reference", inst), config=CFG)
    end = r.trajectory[-1]
    doll = next(n for n in C.DOLLS if end.obj(C.BED).contains_xy(end.obj(n).pose.pos))
    bottom = end.obj(doll).pose.pos[2] - end.obj(doll).size[2]
    assert r.success and bottom == pytest.approx(TABLE + C.BED_FLOOR, abs=0.005)


def test_a_mock_hand_holds_only_what_its_fingertips_reach():
    """v1.2: the box test alone let a hand whose fingertips ended above a 1 cm
    coin take it -- a pick that closes on air on Isaac worked on the mock.  The
    fingertips (8 mm below the tool centre point) must reach 2 mm into it."""
    from ruibench.engines.mock import FINGERTIP_OVERLAP
    from ruibench.interfaces.commands import MoveTo, SetGripper

    for into, held in ((0.0, False), (FINGERTIP_OVERLAP + 0.002, True)):
        inst, engine, ex = _engine("DS4")
        arm = inst.scene.arm_ids[0]
        coin = next(b for b in inst.scene.bodies if "coin" in b.category)
        at = engine.state().obj(coin.name).pose.pos
        hand = engine.state().arm(arm).eef_pose
        tcp_z = at[2] + coin.size[2] + 0.008 - into         # the fingertips `into` below its top
        above = Pose((at[0], at[1], at[2] + 0.10), hand.quat)
        grip = Pose((at[0], at[1], tcp_z), hand.quat)
        for cmd in (MoveTo(arm, above), SetGripper(arm, 0.0), MoveTo(arm, grip),
                    SetGripper(arm, 1.0), MoveTo(arm, above)):
            ex.submit(cmd)
        _run_out(ex)
        assert (engine.state().arm(arm).holding == coin.name) is held, (into, held)


def test_the_mock_hand_can_stop_where_it_meets_things_as_on_isaac():
    """RUIBENCH_MOCK_HAND_COLLIDES=1 (off by default): a hand pointing down
    stops where it would newly meet a body or put its fingertips into the
    table, and the executor calls the move blocked -- a program practised on
    the mock skimmed a balance pan's rim and stopped short there on Isaac
    (RUI-Zero I-013).  Open fingers straddling a coin still go down to it."""
    from ruibench.executor.handles import HandleStatus
    from ruibench.interfaces.commands import MoveTo, SetGripper

    def descend(collides, grip, depth):
        inst, engine, ex = _engine("DS4")
        engine.hand_collides = collides
        arm = inst.scene.arm_ids[0]
        coin = next(b for b in inst.scene.bodies if "coin" in b.category)
        at = engine.state().obj(coin.name).pose.pos
        hand = engine.state().arm(arm).eef_pose
        ex.submit(MoveTo(arm, Pose((at[0], at[1], at[2] + 0.10), hand.quat)))
        ex.submit(SetGripper(arm, grip))
        down = ex.submit(MoveTo(arm, Pose((at[0], at[1], at[2] + depth), hand.quat)))
        _run_out(ex)
        return down, engine.state().arm(arm).eef_pose.pos[2] - at[2]

    # closed fingers pressed onto the coin: through it when off, stopped on it when on
    down, z = descend(False, 1.0, 0.0)
    assert down.status is HandleStatus.DONE and abs(z) < 1e-3
    down, z = descend(True, 1.0, 0.0)
    assert down.status is HandleStatus.FAILED and "blocked" in down.detail
    assert z > 0.005 + 0.008 - 0.002          # fingertips (8 mm below) on the coin's top
    # open fingers straddle it and reach the grip height (fingertips 2 mm off the table)
    down, z = descend(True, 0.0, 0.005)
    assert down.status is HandleStatus.DONE, down.detail
    # and never into the table
    down, z = descend(True, 0.0, -0.01)
    assert down.status is not HandleStatus.DONE


def test_what_the_mock_hand_holds_collides_too():
    """Under the switch a held thing is as solid as on Isaac: a coin brought
    down onto another stops there (an X lowered beside a placed X, WK6:
    RUI-Zero I-015)."""
    from ruibench.executor.handles import HandleStatus
    from ruibench.interfaces.commands import MoveTo, SetGripper

    def stack(collides):
        inst, engine, ex = _engine("DS4")
        engine.hand_collides = collides
        arm = inst.scene.arm_ids[0]
        coins = [b for b in inst.scene.bodies if "coin" in b.category][:2]
        a = engine.state().obj(coins[0].name).pose.pos
        b = engine.state().obj(coins[1].name).pose.pos
        hand = engine.state().arm(arm).eef_pose
        grip = 0.009 - coins[0].size[2]                 # 9 mm above its bottom
        for cmd in (MoveTo(arm, Pose((a[0], a[1], a[2] + 0.10), hand.quat)), SetGripper(arm, 0.0),
                    MoveTo(arm, Pose((a[0], a[1], a[2] + grip), hand.quat)), SetGripper(arm, 1.0),
                    MoveTo(arm, Pose((a[0], a[1], a[2] + 0.10), hand.quat)),
                    MoveTo(arm, Pose((b[0], b[1], b[2] + 0.10), hand.quat))):
            ex.submit(cmd)
        _run_out(ex)
        assert engine.state().arm(arm).holding == coins[0].name
        down = ex.submit(MoveTo(arm, Pose((b[0], b[1], b[2] + grip), hand.quat)))   # into it
        _run_out(ex)
        return down

    assert stack(False).status is HandleStatus.DONE
    down = stack(True)
    assert down.status is HandleStatus.FAILED and "blocked" in down.detail



def test_the_task_levels_the_collision_switch_is_unsound_for_are_listed():
    """``mock.HAND_COLLIDES_UNSOUND`` says where practice with the switch on
    would mislead: the reference itself is stopped there (RUI-Zero reads it to
    decide).  Its keys are task-levels of the main board; its counts make sense."""
    from ruibench.engines import mock
    from ruibench.tasks.loader import main_board

    cells = {(t.id, level) for t in main_board() for level in t.knob.levels}
    assert mock.HAND_COLLIDES_UNSOUND, "an empty list would say the switch is sound everywhere"
    for cell, (stopped, run, why) in mock.HAND_COLLIDES_UNSOUND.items():
        assert cell in cells, cell
        assert 0 < stopped <= run and why
