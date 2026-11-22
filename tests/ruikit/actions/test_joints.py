"""The joint Dialect (v1.17): the robots' own kinematics, joint targets on both
engines' terms, and the two interfaces that use them."""
import random

import pytest

from ruibench.core.types import GRASP_DOWN, Pose
from ruibench.engines import kinematics as K
from ruibench.engines import robots
from ruibench.tasks.instance import build
from ruikit import Episode
from ruikit.interfaces import load_profile
from ruikit.actions import ACTIONS


def _arm(task, seed, arm="right"):
    scene = build(task, level="easy", seed=seed).scene
    robot = scene.robots[0]
    return scene, robots.model_of(robot), K.arm_frame(robot, arm, scene.table_height)


@pytest.mark.parametrize("task,seed", [("CL2", 0), ("WK4", 15)])
def test_the_ready_posture_is_where_the_station_starts_the_hand(task, seed):
    """The kinematics were measured on Isaac at the ready posture: through them
    the ready joints put the tool where the station starts it (to its rounding)."""
    scene, model, base = _arm(task, seed)
    kin = K.KINEMATICS[model.name]
    tcp = K.forward(model, base, kin.ready)
    start = robots.arm_start_pose(scene, scene.robots[0], "right")
    assert tcp.distance_to(start) < 0.001 and tcp.angle_to(start) < 0.002


@pytest.mark.parametrize("task,seed", [("CL2", 0), ("WK4", 15)])
def test_the_inverse_kinematics_finds_a_pose_the_joints_can_take(task, seed):
    """Joints drawn inside their limits, put through the kinematics and back:
    the inverse finds joints for every pose, within 0.1 mm and the limits."""
    _, model, base = _arm(task, seed)
    kin = K.KINEMATICS[model.name]
    rng = random.Random(3)
    for _ in range(8):
        q = [rng.uniform(lo, hi) for lo, hi in zip(kin.lower, kin.upper)]
        goal = K.forward(model, base, q)
        sol = K.inverse(model, base, goal, seed=kin.ready)
        assert sol.ok and K.within_limits(kin, sol.joints)
        back = K.forward(model, base, sol.joints)
        assert back.distance_to(goal) < 1.5e-4 and back.angle_to(goal) < 2e-3


def test_the_x5s_reach_from_its_joints_is_the_measured_ceiling():
    """Straight down, 45 cm out, the X5's joints reach 10 cm over the table
    and not 12: what v1.15 measured with the hand driven by poses."""
    scene, model, base = _arm("WK4", 15)
    at = lambda z: Pose((base.pos[0] + 0.45, base.pos[1], scene.table_height + z), GRASP_DOWN)  # noqa: E731
    assert K.inverse(model, base, at(0.09), seed=K.KINEMATICS["arx5"].ready).ok
    assert not K.inverse(model, base, at(0.12), seed=K.KINEMATICS["arx5"].ready).ok


@pytest.mark.parametrize("task,seed", [("CL2", 0), ("WK4", 15)])
def test_solve_ik_then_move_joints_puts_the_hand_there(task, seed):
    """robot_solve_ik gives the angles, robot_move_joints takes them: the hand
    arrives where it was asked to go, and robot_state reads the joints back."""
    ep = Episode(task, level="easy", seed=seed, profile="rui.joint/v1", engine="mock")
    try:
        base = ep.station.to_station(robots.arm_base(ep.scene.robots[0], "right",
                                                     ep.station.table_height))
        goal = [base[0] + 0.35, base[1] + 0.05, 0.08]
        r = ep.call("robot_solve_ik", {"xyz_m": goal, "rpy_deg": [0, 0, 30]})
        assert r.data["reachable"] and r.data["residual_mm"] < 0.2, r.text
        moved = ep.call("robot_move_joints", {"joints_deg": r.data["joints_deg"]})
        assert moved.data["status"] == "done", moved.text
        state = ep.call("robot_state", {}).data["arms"]
        right = next(a for a in state if a["arm"] == "right")
        assert right["tcp"]["xyz_m"] == pytest.approx(goal, abs=0.002)
        assert right["joints_deg"] == pytest.approx(r.data["joints_deg"], abs=0.1)
    finally:
        ep.close()


def test_joint_targets_outside_the_limits_or_miscounted_are_refused():
    ep = Episode("CL2", level="easy", seed=0, profile="rui.joint/v1", engine="mock")
    try:
        r = ep.call("robot_move_joints", {"joints_deg": [0, 0, 0, 0, 0, 90, 45]})   # joint 4 > -8.7
        assert r.error == "unreachable" and "joint 4" in r.text and "limits" in r.text, r.text
        assert len(r.data["limits_deg"]) == 7
        r = ep.call("robot_move_joints", {"joints_deg": [0, -45, 0, -163, 0, 117, 172.8]})
        assert r.error is None, r.text            # 172.8 is the limit as printed: at it, not past
        r = ep.call("robot_move_joints", {"joints_deg": [0, -45, 0, -163, 0, 117]})
        assert r.error == "invalid_args" and "7 joints" in r.text, r.text
        r = ep.call("robot_solve_ik", {"xyz_m": [2.0, 0.0, 0.1]})
        assert not r.data["reachable"] and r.data["residual_mm"] > 100, r.text
    finally:
        ep.close()


def test_the_joints_are_an_abstraction_of_their_own():
    """rui.joint/v1 is the shared tools with the joint moves; rui.code-joint/v1
    calls the same moves from programs; RUI-Free offers them beside the others."""
    assert ACTIONS["joint"].tools == ("robot_solve_ik", "robot_move_joints", "robot_gripper")
    joint, code = load_profile("rui.joint/v1"), load_profile("rui.code-joint/v1")
    assert joint.dialect == "joint" and set(ACTIONS["joint"].tools) <= set(joint.tools)
    assert set(code.code_tools) == set(joint.tools) and code.channel == "code"
    free = load_profile("rui.free/v1")
    assert {"robot_solve_ik", "robot_move_joints"} <= set(free.tools) & set(free.code_tools)


def test_the_joint_reference_goes_along_the_pose_interfaces_straight_lines():
    """A joint move between two far poses swings the hand along a curve (it
    knocked WK3's dominoes over on Isaac): the reference cuts a move until each
    piece's joint path keeps within 5 mm of the straight line."""
    from ruikit.diagnostics.reference.joint import JointExpressor, _off_line
    from ruikit.experiments.results import column_of

    ep = Episode("CL2", level="easy", seed=0, profile="rui.joint/v1", engine="mock")
    try:
        arm = ep.scene.arm_ids[0]
        now = ep.engine.state().arm(arm).eef_pose
        far = now.translated((0.20, -0.25, -0.15))

        class Move:                                   # what the reference commands
            pass
        c = Move()
        c.arm = arm
        steps = JointExpressor()._steps(ep, c, far)
        assert len(steps) > 1 and steps[-1] == far
        model, base, kin = ep.station.kinematics("right")
        q, a = ep.engine.arm_joints(arm), now
        for b in steps:
            qb = K.inverse(model, base, b, seed=q).joints
            for t in (0.25, 0.5, 0.75):
                p = K.forward(model, base, [x + (y - x) * t for x, y in zip(q, qb)]).pos
                assert _off_line(p, a.pos, b.pos) <= JointExpressor.TOL_M + 1e-6
            q, a = qb, b
        assert column_of("rui.joint/v1") == "joint"
    finally:
        ep.close()


def test_a_program_over_a_dialect_moves_both_arms_at_once():
    """As the pose program does: a round's moves on two arms start together
    (wait=False) and are waited for together -- not one after the other."""
    from ruikit.diagnostics.reference.program import _ProgramEpisode

    ep = Episode("WK4", level="easy", seed=0, profile="rui.code-joint/v1", engine="mock")
    try:
        prog = _ProgramEpisode(ep)
        state = ep.engine.state()
        calls = []
        for side in ("left", "right"):
            tcp = ep.station.describe_pose(state.arm(f"robot0/{side}").eef_pose)
            r = prog.call("robot_solve_ik", {"arm": side, "xyz_m": tcp["xyz_m"],
                                             "rpy_deg": tcp["rpy_deg"]})
            calls.append(("robot_move_joints", {"arm": side, "joints_deg": r.data["joints_deg"]}))
        out = prog.call_batch(calls)
        assert all(not r.is_error for r in out), [r.text for r in out]
        source = ep.log[-1]["args"]["source"]
        assert source.count("'wait': False") == 2 and "robot.wait(handles=" in source, source
    finally:
        ep.close()


def test_a_program_solves_and_moves_in_one_run():
    """Over a program, the joint reference solves and moves in one program, the
    solved joints passed on inside it: one reply where the joint interface
    takes two."""
    from ruikit.diagnostics.reference.program import _Out, _ProgramEpisode

    ep = Episode("CL2", level="easy", seed=0, profile="rui.code-joint/v1", engine="mock")
    try:
        arm = ep.scene.arm_ids[0]
        base = ep.station.to_station(robots.arm_base(ep.scene.robots[0], "right",
                                                     ep.station.table_height))
        goal = [base[0] + 0.35, base[1] + 0.05, 0.08]
        before = len(ep.log)
        solved, moved = _ProgramEpisode(ep).call_rounds(
            [[("robot_solve_ik", {"xyz_m": goal, "rpy_deg": [0, 0, 30]})],
             [("robot_move_joints", {"joints_deg": _Out(0, "joints_deg")})]])
        assert solved.data["reachable"] and moved.data["status"] == "done", moved.text
        assert [row["tool"] for row in ep.log[before:]] == ["code_run"]
        tcp = ep.station.describe_pose(ep.engine.state().arm(arm).eef_pose)["xyz_m"]
        assert tcp == pytest.approx(goal, abs=0.002)
    finally:
        ep.close()
