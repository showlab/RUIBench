"""The real station's engine (``ruibench.engines.franka``), without the robot.

The fakes stand in for the NUC's server and the cameras (``engines.franka.fake``):
the engine, its control thread and its rectification run as on the robot.  A
fast rig keeps the moves short.
"""

import dataclasses
import json
import math
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
cv2 = pytest.importorskip("cv2")

from ruibench.core.errors import ConfigError  # noqa: E402
from ruibench.core.geometry import relative_pose  # noqa: E402
from ruibench.core.types import GRASP_DOWN, Pose  # noqa: E402
from ruibench.engines import ENGINES, is_real, kinematics, results_grade  # noqa: E402
from ruibench.engines.franka import calibration, rig as rig_mod  # noqa: E402
from ruibench.engines.franka.arm import MAX_RESTARTS, FrankaArm, from_wire  # noqa: E402
from ruibench.engines.franka.cameras import Intrinsics, Rectifier, _distorted  # noqa: E402
from ruibench.engines.franka.engine import FrankaEngine, RobotFault  # noqa: E402
from ruibench.engines.franka.fake import MEASURED, FakeCamera, FakeServer, flange_at  # noqa: E402
from ruibench.engines.franka.model import (  # noqa: E402
    HOME_JOINTS, LINE_UPPER, PANDA, WRIST_CLEAR_RAD,
)
from ruibench.engines.franka.route import along, solve  # noqa: E402
from ruibench.engines.franka.view import VIEW_ENV  # noqa: E402
from ruibench.engines.franka.scene import station_scene  # noqa: E402

ARM = "robot0/right"
FAST = rig_mod.Motion(control_hz=100, max_speed_m_s=1.0, max_turn_rad_s=4.0, descent_speed_m_s=1.0,
                      joint_speed_rad_s=3.0, min_move_seconds=0.1, follow_limit_m=0.5,
                      follow_limit_rad=3.0, joint_speed_limit_rad_s=30.0)
QUICK_GRIP = rig_mod.GripperRig(stable_s=0.1, min_s=0.1, max_s=1.5)


def _engine(sag_m=0.003, motion=FAST, server=FakeServer, **given):
    rig = dataclasses.replace(rig_mod.load(), motion=motion, gripper=QUICK_GRIP)
    eng = FrankaEngine(rig=rig, calibration=calibration.provisional(),
                       connect=lambda: server(lag_s=0.02, sag_m=sag_m, gripper_speed=1.0, **given),
                       cameras=lambda name, c: FakeCamera(c.serial, c.width, c.height, c.fps))
    scene = station_scene("test", [("lemon", "lemon")], cal=eng.calibration)
    scene = dataclasses.replace(scene, cameras=tuple(
        dataclasses.replace(c, width=640 if c.kind == "external" else 512,
                            height=480 if c.kind == "external" else 512, depth=True)
        for c in scene.cameras))
    eng.load(scene, 0)
    eng.reset()
    return eng


def _settle(eng, cap=5.0, gripper=False):
    waited = 0.0
    done = (lambda: eng.gripper_settled(ARM)) if gripper else (lambda: eng.arm_at_target(ARM))
    while not done() and waited < cap:
        eng.advance(0.02)
        waited += 0.02
    return done()


@pytest.fixture(autouse=True)
def _no_calibration_of_this_machine(tmp_path, monkeypatch):
    """What is tested is the provisional station, whatever the lab has measured."""
    monkeypatch.setattr(calibration, "PATH", tmp_path / "not-measured.json")


@pytest.fixture
def engine():
    eng = _engine()
    yield eng
    eng.close()


def test_the_real_station_is_an_engine_of_its_own():
    """Registered beside the simulator and the mock; its results are real ones, kept apart."""
    assert set(ENGINES.names()) == {"isaac", "mock", "franka"}
    assert is_real("franka") and not is_real("mock") and not is_real("isaac")
    assert results_grade("franka")


def test_importing_the_real_station_needs_no_robot_libraries():
    code = ("import ruibench.engines.franka, ruibench.tasks.real, sys;"
            "print([m for m in ('zerorpc', 'pyrealsense2', 'gevent') if m in sys.modules])")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert out.stdout.strip() == "[]", out.stderr


def test_the_tool_centre_point_is_where_the_joints_put_it():
    """Flange to tool centre point (the Franka Hand turned 45 degrees on the flange,
    10.34 cm along it) agrees with the model's kinematics, both ways."""
    for q in ((0.0, -0.5058, 0.0, -2.6069, 0.0, 2.0712, 0.8071),
              (0.4, -0.2, 0.1, -2.2, 0.3, 2.4, -0.5)):
        tcp = calibration.flange_to_tcp(flange_at(q))
        model = kinematics.forward(PANDA, Pose(), q)
        assert tcp.distance_to(model) < 1e-6 and tcp.angle_to(model) < 1e-6
        assert calibration.tcp_to_flange(tcp).distance_to(flange_at(q)) < 1e-9


def test_a_rectified_pixel_is_the_models_pinhole():
    """A point projects onto the rectified picture where a centred pinhole of the
    model's focal length puts it -- through the D405's distorted lens too."""
    raw = MEASURED["218622273043"]
    rect = Rectifier(raw, 512, 512, 439.8)
    assert rect.uncovered == 0.0
    for x, y in ((0.0, 0.0), (0.3, -0.2), (-0.5, 0.45)):
        u, v = 439.8 * x + 256 - 0.5, 439.8 * y + 256 - 0.5        # its pixel, by index
        # the raw pixel the rectifier reads there, undistorted, is the same ray
        rx, ry = rect.map_x[int(round(v)), int(round(u))], rect.map_y[int(round(v)), int(round(u))]
        xd, yd = (rx - raw.ppx) / raw.fx, (ry - raw.ppy) / raw.fy
        k1, k2, p1, p2, k3 = raw.coeffs
        r2 = xd * xd + yd * yd
        f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        ux = xd * f + 2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd)
        uy = yd * f + 2 * p2 * xd * yd + p1 * (r2 + 2 * yd * yd)
        assert abs(ux - x) < 2e-3 and abs(uy - y) < 2e-3


def test_the_inverse_distortion_is_solved_exactly():
    xu = np.linspace(-0.6, 0.6, 7)[None, :].repeat(3, 0)
    yu = np.linspace(-0.6, 0.6, 3)[:, None].repeat(7, 1)
    coeffs = MEASURED["218622273043"].coeffs
    xd, yd = _distorted(xu, yu, coeffs)
    k1, k2, p1, p2, k3 = coeffs
    r2 = xd * xd + yd * yd
    f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
    assert np.allclose(xd * f + 2 * p1 * xd * yd + p2 * (r2 + 2 * xd * xd), xu, atol=1e-9)
    assert np.allclose(yd * f + 2 * p2 * xd * yd + p1 * (r2 + 2 * yd * yd), yu, atol=1e-9)


def test_a_lens_wider_than_the_camera_is_refused():
    with pytest.raises(ConfigError, match="sees past its picture"):
        eng = FrankaEngine(rig=rig_mod.load(), calibration=calibration.provisional(),
                           connect=FakeServer,
                           cameras=lambda name, c: FakeCamera(
                               c.serial, intrinsics=Intrinsics(640, 480, 900.0, 900.0, 320, 240)))
        try:
            eng.load(station_scene("test", cal=eng.calibration), 0)
        finally:
            eng.close()


def test_the_arm_starts_about_where_the_labs_home_has_it(engine):
    st = engine.state()
    arm = st.arm(ARM)
    assert arm.eef_pose.distance_to(Pose((0.38, 0.0, 0.29))) < 0.002
    assert abs(arm.joints[0]) < 0.05 and abs(arm.joints[2]) < 0.05        # the elbow up, not aside
    assert arm.eef_pose.angle_to(Pose((0, 0, 0), GRASP_DOWN)) < 0.02
    assert st.objects == {} and arm.holding is None and len(arm.joints) == 7
    assert arm.opening == pytest.approx(0.08, abs=1e-3) and arm.gripper == 0.0


def test_a_move_arrives_although_the_arm_sags(engine):
    """The impedance controller rests short of its setpoint; the settling closes it."""
    goal = Pose((0.45, 0.10, 0.12), GRASP_DOWN)
    engine.set_arm_target(ARM, goal)
    assert _settle(engine)
    here = engine.state().arm(ARM).eef_pose
    assert here.distance_to(goal) <= 0.002


def test_a_straight_descent_arrives(engine):
    """A line follows the joints as the robot's controller moves them, whose
    redundancy may go anywhere inside the limits (a check that wanted each point's
    joints near the last's stopped a 5 cm descent 4 cm short)."""
    start = engine.state().arm(ARM).eef_pose
    goal = Pose((start.pos[0], start.pos[1], start.pos[2] - 0.05), GRASP_DOWN)
    engine.set_arm_target(ARM, goal)
    assert _settle(engine)


# -- what keeps the robot out of its reflexes (2026-11-02) ------------------------------
class _Recording(FakeServer):
    """Keeps the setpoints it is sent; ``slow_once`` delays one reading of the flange."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.sent, self.ahead, self.slow_once = [], [], 0.0

    def update_desired_ee_pose(self, pose):
        super().update_desired_ee_pose(pose)
        self.sent.append(from_wire(pose))
        self.ahead.append(self.sent[-1].distance_to(self.flange))

    def get_ee_pose(self):
        slow, self.slow_once = self.slow_once, 0.0
        if slow:
            time.sleep(slow)
        return super().get_ee_pose()


def _the_server(eng):
    while eng.arm.server is None:
        time.sleep(0.01)
    return eng.arm.server


def _base_and_start():
    base = Pose((0.0, 0.0, -0.0187))
    q = list(HOME_JOINTS)
    return base, q, kinematics.forward(PANDA, base, q)


def test_a_line_stops_short_of_the_wrists_singularity():
    """The hand bent forward takes joint 6 toward pi, where the robot's controller
    swings joints 3, 5 and 7 past their speed: the calibration's first run, whose
    check (a solver at every point, with restarts) let such views through."""
    from ruibench.engines.franka.calibrate import turned_down

    base, q, here = _base_and_start()
    goal = Pose((0.60, 0.0, 0.25), turned_down(0, 45, 0))
    way = along(PANDA, base, q, here, goal, rig_mod.Motion())
    assert way.end is not None and not way.whole and "joint 6" in way.stopped
    assert way.joints[5] <= LINE_UPPER[5] <= math.pi - WRIST_CLEAR_RAD + 1e-3
    back = along(PANDA, base, way.joints, way.end, here, rig_mod.Motion())
    assert back.whole                                     # and it may come back from there


def test_the_hand_gets_down_to_the_table_all_over_where_things_are_laid_out():
    """From the start, a line above the place and a line down to a grasp's height, all
    over what the real task's card asks of the operator, on the lab's calibrated table."""
    lab = json.loads(Path(calibration.__file__).with_name("calibration.json").read_text())
    base, _, first = _base_and_start()
    base = Pose((0.0, 0.0, -lab["table_z_m"]), base.quat)
    kin = kinematics.KINEMATICS[PANDA.name]
    start, solved = solve(kin, PANDA.tcp_from_body, HOME_JOINTS, relative_pose(base, first))
    assert solved
    for x in (0.38, 0.45, 0.55):
        for y in (-0.25, 0.0, 0.25):
            q, here = start, first
            for goal in (Pose((x, y, 0.15), GRASP_DOWN), Pose((x, y, 0.03), GRASP_DOWN)):
                way = along(PANDA, base, q, here, goal, rig_mod.Motion())
                assert way.whole, f"({x}, {y}, {goal.pos[2]}): {way.stopped}"
                q, here = way.joints, goal


def test_a_line_takes_as_long_as_its_joints_need():
    """Its joints' fastest, reckoned along the line, sets its time at full speed
    and past it, and the arm it is sent to keeps within that."""
    motion = dataclasses.replace(rig_mod.Motion(), joint_speed_rad_s=0.2)
    eng = _engine(sag_m=0.0, motion=motion)
    try:
        server = _the_server(eng)
        start = eng.state().arm(ARM).eef_pose
        goal = Pose((start.pos[0] - 0.06, start.pos[1] + 0.10, start.pos[2] - 0.04), start.quat)
        way = eng.route(goal)
        assert way.whole
        hand_alone = 1.875 * start.distance_to(goal) / motion.max_speed_m_s
        assert way.seconds > 1.1 * hand_alone             # a joint is what limits it
        server.fastest = [0.0] * 7
        eng.set_arm_target(ARM, goal)
        assert _settle(eng, cap=way.seconds + 3.0)
        assert max(server.fastest) <= motion.joint_speed_rad_s * 1.15, server.fastest
        assert not server.reflexes
    finally:
        eng.close()


def test_a_late_answer_pauses_a_move_instead_of_jumping_it():
    """A reading 0.2 s late is ten periods' travel the path must not make up at once:
    it faulted the arm on the calibration's first run."""
    motion = dataclasses.replace(FAST, max_speed_m_s=0.3, control_hz=50)
    eng = _engine(motion=motion, server=_Recording)
    try:
        server = _the_server(eng)
        start = eng.state().arm(ARM).eef_pose
        goal = Pose((start.pos[0] + 0.12, start.pos[1], start.pos[2]), start.quat)
        eng.set_arm_target(ARM, goal)
        eng.advance(0.3)
        server.sent.clear()
        server.slow_once = 0.2
        assert _settle(eng)                               # no fault, and it gets there
        jumps = [a.distance_to(b) for a, b in zip(server.sent, server.sent[1:])]
        assert max(jumps) <= 3 * motion.max_speed_m_s / motion.control_hz + 0.002
    finally:
        eng.close()


def test_a_hand_held_back_stops_pushing():
    """The setpoint runs ahead of a hand that something holds, by the stiffness'
    newtons a centimetre: it comes back to the hand before the robot's reflex, and
    the executor finds the move blocked."""
    motion = dataclasses.replace(FAST, max_speed_m_s=0.2, follow_limit_m=0.02, follow_lag_s=0.0)
    eng = _engine(motion=motion, server=_Recording, floor_z=0.33)
    try:
        server = _the_server(eng)
        start = eng.state().arm(ARM).eef_pose
        eng.set_arm_target(ARM, Pose((start.pos[0], start.pos[1], 0.05), start.quat))
        assert not _settle(eng, cap=1.5)
        here = from_wire(server.get_ee_pose())
        assert not eng.arm.busy() and server.sent[-1].distance_to(here) < 0.012
        assert max(server.ahead) < 0.03, max(server.ahead)    # never much past the limit
        assert here.pos[2] >= 0.33 - 1e-3 and not eng.arm_at_target(ARM)
    finally:
        eng.close()


class _SlowToStart(FakeServer):
    """Its joint controller takes this long to start, and may be gone as it does."""

    start_s = 0.0
    gone = 0

    def start_joint_impedance(self, Kq=None, Kqd=None):
        time.sleep(self.start_s)
        super().start_joint_impedance(Kq, Kqd)
        with self._lock:
            if self.gone:
                self.gone -= 1
                self.policy = None


def _a_little_joint_move(eng):
    q = list(eng.arm_joints(ARM))
    q[0] += 0.05
    eng.set_arm_joints(ARM, q)
    return q


def test_a_controller_slow_to_start_is_not_a_server_gone_quiet():
    """Over 1 s to start one, on 2026-11-04: going home faulted with "the server stopped
    answering" while the server was doing as it was told."""
    eng = _engine(server=_SlowToStart)
    try:
        server = _the_server(eng)
        server.start_s = 1.3
        q = _a_little_joint_move(eng)
        assert _settle(eng)
        assert eng.arm_joints(ARM)[0] == pytest.approx(q[0], abs=0.01)
    finally:
        eng.close()


def test_a_joint_controller_gone_as_it_starts_is_started_again():
    eng = _engine(server=_SlowToStart)
    try:
        server = _the_server(eng)
        server.gone = MAX_RESTARTS
        q = _a_little_joint_move(eng)
        assert _settle(eng)
        assert eng.arm_joints(ARM)[0] == pytest.approx(q[0], abs=0.01)
        server.gone = 1                                   # and it does not go on for ever
        _a_little_joint_move(eng)
        with pytest.raises(RobotFault, match="controller was lost"):
            _settle(eng)
    finally:
        eng.close()


def test_a_joint_controller_lost_on_the_way_still_faults_the_arm():
    """It has moved the arm, and a reflex may be why: no going on by itself."""
    motion = dataclasses.replace(FAST, joint_speed_rad_s=0.1)
    eng = _engine(motion=motion, server=_SlowToStart)
    try:
        server = _the_server(eng)
        _a_little_joint_move(eng)
        eng.advance(0.3)
        with server._lock:
            server.policy = None
        with pytest.raises(RobotFault, match="no controller running"):
            eng.advance(0.3)
    finally:
        eng.close()


class _Runaway(FakeServer):
    """Its joint 1 sets off by itself when told to."""

    runaway = False

    def get_joint_positions(self):
        q = super().get_joint_positions()
        if self.runaway:
            with self._lock:
                self.q[0] += 0.05
        return q


def test_a_joint_running_away_faults_the_arm_and_leaves_it_held():
    motion = dataclasses.replace(FAST, joint_speed_limit_rad_s=1.2)
    eng = _engine(motion=motion, server=_Runaway)
    try:
        server = _the_server(eng)
        server.runaway = True
        with pytest.raises(RobotFault, match="joint 1 ran at"):
            for _ in range(100):
                eng.advance(0.02)
        server.runaway = False
        time.sleep(0.1)
        here = from_wire(server.get_ee_pose())
        assert server.setpoint.distance_to(here) < 0.01        # the last word: stay here
    finally:
        eng.close()


def test_a_lost_controller_is_started_where_the_arm_is_and_the_move_is_over():
    """A reflex ends the controller.  What the path had the arm do meanwhile is not
    caught up on when it is back; and reflexes that go on fault the arm."""
    motion = dataclasses.replace(FAST, max_speed_m_s=0.2)
    eng = _engine(motion=motion, server=_Recording)
    try:
        server = _the_server(eng)
        start = eng.state().arm(ARM).eef_pose
        goal = Pose((start.pos[0] + 0.15, start.pos[1], start.pos[2]), start.quat)
        for _ in range(MAX_RESTARTS):
            eng.set_arm_target(ARM, goal)
            eng.advance(0.3)
            with server._lock:
                server.policy = None
            at = from_wire(server.get_ee_pose())
            eng.advance(0.3)
            assert server.policy == "cartesian" and not eng.arm.busy()
            assert from_wire(server.get_ee_pose()).distance_to(at) < 0.01
            assert not eng.arm_at_target(ARM)
        eng.set_arm_target(ARM, goal)
        eng.advance(0.1)
        with server._lock:
            server.policy = None
        with pytest.raises(RobotFault, match="controller was lost"):
            eng.advance(0.3)
    finally:
        eng.close()


def test_what_the_arm_did_is_kept(tmp_path):
    arm = FrankaArm("fake", 0, FAST, connect=lambda: FakeServer(lag_s=0.02, sag_m=0.0),
                    flight=tmp_path / "flight" / "arm.jsonl")
    start = arm.start()
    arm.move(start.flange.translated((0.02, 0.0, 0.0)))
    arm.note("a word")
    time.sleep(0.3)
    arm.close()
    rows = [json.loads(line) for line in (tmp_path / "flight" / "arm.jsonl").read_text().splitlines()]
    moving = [r for r in rows if "q" in r and not r["done"]]
    assert len(moving) >= 5 and len(moving[0]["q"]) == 7 and len(moving[0]["set"]) == 7
    assert any(r.get("note") == "a word" for r in rows)


# -- the operator's pause --------------------------------------------------------------
def test_a_pause_stops_the_arm_and_the_clock_and_the_move_goes_on():
    motion = dataclasses.replace(FAST, max_speed_m_s=0.2)
    eng = _engine(motion=motion, server=_Recording)
    try:
        server = _the_server(eng)
        start = eng.state().arm(ARM).eef_pose
        goal = Pose((start.pos[0] + 0.15, start.pos[1], start.pos[2]), start.quat)
        eng.set_arm_target(ARM, goal)
        eng.advance(0.5)
        eng.pause()
        clock, at = eng.sim_time, eng.state().arm(ARM).eef_pose
        time.sleep(0.3)
        server.sent.clear()
        time.sleep(0.3)
        held = eng.state().arm(ARM).eef_pose
        assert eng.paused and eng.sim_time == pytest.approx(clock, abs=1e-6)
        assert held.distance_to(at) < 0.02 and held.distance_to(goal) > 0.03
        assert not eng.arm_at_target(ARM)
        assert server.sent and all(s.distance_to(server.sent[0]) < 1e-9 for s in server.sent)
        eng.resume()
        assert not eng.paused and _settle(eng)
        assert eng.state().arm(ARM).eef_pose.distance_to(goal) <= 0.002
        assert eng.sim_time - clock < 3.0
    finally:
        eng.close()


def test_a_pause_does_not_lose_a_joint_move():
    motion = dataclasses.replace(FAST, joint_speed_rad_s=1.0, joint_speed_limit_rad_s=30.0)
    eng = _engine(motion=motion)
    try:
        q = list(eng.arm_joints(ARM))
        q[0] += 0.5
        goal = eng.set_arm_joints(ARM, q)
        eng.advance(0.3)
        eng.pause()
        time.sleep(0.2)
        held = eng.arm_joints(ARM)
        time.sleep(0.2)
        assert abs(eng.arm_joints(ARM)[0] - held[0]) < 0.02 and abs(held[0] - q[0]) > 0.05
        eng.resume()
        assert _settle(eng)
        assert eng.state().arm(ARM).eef_pose.distance_to(goal) <= 0.002
    finally:
        eng.close()


def test_what_is_asked_during_a_pause_waits_for_its_end():
    eng = _engine(holding_width=0.05)
    try:
        start = eng.state().arm(ARM).eef_pose
        eng.pause()
        goal = Pose((start.pos[0] + 0.05, start.pos[1], start.pos[2]), start.quat)
        eng.set_arm_target(ARM, goal)
        eng.set_gripper(ARM, 1.0)
        time.sleep(0.4)
        arm = eng.state().arm(ARM)
        assert arm.eef_pose.distance_to(start) < 0.005 and arm.opening > 0.07
        eng.resume()
        assert _settle(eng) and _settle(eng, gripper=True)
        arm = eng.state().arm(ARM)
        assert arm.eef_pose.distance_to(goal) <= 0.002
        assert arm.opening == pytest.approx(0.05, abs=2e-3)
    finally:
        eng.close()


def test_a_held_arm_gives_way_to_a_push():
    """Pushed further than the watching allows, a pause takes the new place: it does
    not push back harder and harder until the robot's reflex."""
    motion = dataclasses.replace(FAST, follow_limit_m=0.02, follow_lag_s=0.0)
    eng = _engine(motion=motion, server=_Recording, sag_m=0.0)
    try:
        server = _the_server(eng)
        eng.pause()
        time.sleep(0.1)
        with server._lock:
            server.q[0] += 0.15                           # about 5 cm along y
            server.flange = flange_at(server.q)
        time.sleep(0.1)
        assert server.sent[-1].distance_to(from_wire(server.get_ee_pose())) < 0.02
        eng.resume()
        eng.advance(0.3)
        assert not eng.arm.busy() or _settle(eng)
    finally:
        eng.close()


def test_a_tool_waits_out_a_pause(engine):
    """A pause longer than the executor's patience with a hand that does not move: the
    move is neither blocked nor timed out, and no limit is charged for it."""
    ep = _episode(engine, clock="measured")
    wall, t0 = time.monotonic(), ep.sim_time

    def operator():
        time.sleep(0.1)
        engine.pause()
        time.sleep(1.5)
        engine.resume()

    hand = threading.Thread(target=operator)
    hand.start()
    moved = ep.call("robot_move_to", {"arm": "right", "xyz_m": [0.40, 0.05, 0.15],
                                      "rpy_deg": [0, 0, 0]})
    hand.join()
    assert moved.data["status"] == "done", moved.text
    wall = time.monotonic() - wall
    assert wall >= 1.5 and ep.sim_time - t0 <= wall - 1.4


def test_the_window_shows_both_cameras_and_p_pauses_and_goes_on(monkeypatch):
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv(VIEW_ENV, "1")

    def until(done):
        for _ in range(200):
            if done():
                return True
            time.sleep(0.02)
        return False

    eng = _engine()
    try:
        view = eng.view
        assert view is not None and until(lambda: set(view._shown) == {"agentview", "wrist"})
        assert view._shown["wrist"].shape == (512, 512, 3) and not view.problem
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        assert until(lambda: eng.paused)
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_p))
        assert until(lambda: not eng.paused)
        pygame.event.post(pygame.event.Event(pygame.QUIT))
        time.sleep(0.1)
        assert view._thread.is_alive()                    # no way out of a pause without it
    finally:
        eng.close()


def test_f_in_the_window_fails_the_episode_and_nothing_moves_the_arm_again(tmp_path, monkeypatch):
    pygame = pytest.importorskip("pygame")
    from ruibench.engines.franka.engine import FAIL_FILE_ENV

    word = tmp_path / "failed"
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv(VIEW_ENV, "1")
    monkeypatch.setenv(FAIL_FILE_ENV, str(word))
    eng = _engine()
    try:
        eng.set_arm_target(ARM, Pose((0.6, 0.2, 0.1), GRASP_DOWN))
        eng.advance(0.05)
        eng.pause()                                # from a pause too: what waits on it returns
        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        for _ in range(200):
            if word.exists():
                break
            time.sleep(0.02)
        assert word.exists() and not eng.paused and "FAILED" in eng._view_state()[0]
        time.sleep(0.2)
        held = eng.state().arm(ARM).eef_pose
        eng.advance(0.4)
        assert eng.state().arm(ARM).eef_pose.distance_to(held) < 0.01      # the move does not go on
        eng.set_arm_target(ARM, Pose((0.45, -0.2, 0.2), GRASP_DOWN))       # a program's next move
        eng.set_gripper(ARM, 1.0)
        eng.advance(0.4)
        assert eng.state().arm(ARM).eef_pose.distance_to(held) < 0.01
        assert eng.state().arm(ARM).opening > 0.07
    finally:
        eng.close()


def test_without_the_runner_f_does_nothing(monkeypatch):
    from ruibench.engines.franka.engine import FAIL_FILE_ENV

    pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv(VIEW_ENV, "1")
    monkeypatch.delenv(FAIL_FILE_ENV, raising=False)
    eng = _engine()
    try:
        assert eng.view is not None and eng.view._fail is None     # going home, the calibration
    finally:
        eng.close()


def test_the_runner_ends_a_failed_episode_and_asks_only_why(tmp_path, monkeypatch):
    from ruikit.experiments import real
    from ruikit.experiments.plans import Job
    from ruikit.serve import session

    folder = tmp_path / "cell"
    folder.mkdir()
    asked, said = [], []
    monkeypatch.setattr(session, "_ask", lambda f, req, timeout=0: asked.append(req) or {"ended": 1})
    follow = real.Follow(folder, say=said.append)
    follow.look()
    assert asked == [] and said == []              # nobody has pressed it
    (folder / real.FAILED_BY_KEY).write_text("now")
    follow.look()
    follow.look()
    for _ in range(100):
        if asked:
            break
        time.sleep(0.01)
    assert len(asked) == 1 and asked[0]["calls"][0][0] == "task_done"       # once
    assert len([s for s in said if "F:" in s]) == 1

    job = Job("claude-opus-5-5", "R0", "easy", 0, "rui.pose/v1+aids", "claude-code", "measured")
    monkeypatch.setattr(real, "ask", lambda *a: pytest.fail("the verdict is given already"))
    monkeypatch.setattr("builtins.input", lambda prompt: "it never found the block")
    row = real.judge(job, {"reason": "agent_done", "replies": 3}, folder, say=said.append)
    assert row["success"] is False and row["judged_by"] == "operator"
    assert row["verdict"]["failed_by_key"] and row["verdict"]["note"] == "it never found the block"
    assert json.loads((folder / "result.json").read_text())["success"] is False


def test_a_move_out_of_the_workspace_stops_at_its_edge(engine):
    ws = engine.rig.workspace
    engine.set_arm_target(ARM, Pose((0.45, 0.0, -0.05), GRASP_DOWN))     # into the table
    assert not _settle(engine, cap=1.5)
    assert engine.state().arm(ARM).eef_pose.pos[2] >= ws.z[0] - 0.003
    engine.set_arm_target(ARM, Pose((1.2, 0.0, 0.2), GRASP_DOWN))        # past its reach
    assert not _settle(engine, cap=1.5)
    assert engine.state().arm(ARM).eef_pose.pos[0] <= ws.x[1] + 0.003


def test_what_follows_a_blocked_move_does_not_run_on_a_real_robot(monkeypatch):
    from ruibench.executor import Executor
    from ruibench.executor.handles import HandleStatus
    from ruibench.interfaces.commands import MoveTo

    def two_legs(real):
        eng = _engine(floor_z=0.35)                # what stops the first leg, 30 cm over the table
        try:
            if not real:
                monkeypatch.setattr(eng, "capabilities",
                                    dataclasses.replace(eng.capabilities, real=False))
            ex = Executor(eng, [ARM])
            first = ex.submit(MoveTo(ARM, Pose((0.38, 0.0, 0.10), GRASP_DOWN)))
            second = ex.submit(MoveTo(ARM, Pose((0.38, 0.0, 0.05), GRASP_DOWN)))
            for _ in range(200):
                ex.tick(0.05)
                if first.status is HandleStatus.FAILED:
                    break
            assert first.status is HandleStatus.FAILED and "blocked" in first.detail
            ex.tick(0.05)
            return second.status
        finally:
            eng.close()

    assert two_legs(real=True) is HandleStatus.FAILED      # it never starts
    assert two_legs(real=False) is HandleStatus.RUNNING    # simulation's, as its results have it


def test_going_down_near_the_table_is_slower_than_the_rest_of_the_way():
    from ruibench.engines.franka.arm import FrankaArm

    arm = FrankaArm("nowhere", 0, rig_mod.load().motion)
    high, low = Pose((0.4, 0.0, 0.25), GRASP_DOWN), Pose((0.4, 0.0, 0.15), GRASP_DOWN)
    assert arm._seconds(high, low) == pytest.approx(1.875 * 0.10 / 0.08)   # its fastest, 0.08 m/s
    assert arm._seconds(low, high) == pytest.approx(1.875 * 0.10 / 0.15)   # up, as fast as ever
    far_up = Pose((0.4, 0.0, 0.45), GRASP_DOWN)
    assert arm._seconds(far_up, Pose((0.4, 0.0, 0.35), GRASP_DOWN)) == pytest.approx(1.25)


def test_a_press_does_not_fly_the_next_move_under_its_line():
    eng = _engine(floor_z=0.20)
    try:
        here = eng.state().arm(ARM).eef_pose
        eng.set_arm_target(ARM, Pose((here.pos[0], here.pos[1], 0.02), GRASP_DOWN))   # on the floor
        for _ in range(150):
            eng.advance(0.02)
        pressed = eng.arm._bias_p[2]
        assert pressed < -0.008                    # it has been pushing down
        held = eng.state().arm(ARM).eef_pose.pos
        eng.arm.move(eng._to_flange(Pose((held[0], held[1] + 0.03, held[2]), GRASP_DOWN)))
        for _ in range(100):
            eng.advance(0.02)
            if eng.arm.reading().path_done:
                break
        assert eng.arm.reading().path_done
        assert abs(eng.arm._bias_p[2]) <= 0.0065   # what a free hand needs, and no more
    finally:
        eng.close()


def test_a_stop_holds_where_the_arm_is(engine):
    engine.set_arm_target(ARM, Pose((0.6, 0.2, 0.1), GRASP_DOWN))
    engine.advance(0.1)
    engine.stop_arm(ARM)
    held = engine.state().arm(ARM).eef_pose
    engine.advance(0.3)
    assert engine.state().arm(ARM).eef_pose.distance_to(held) < 0.01
    assert engine.arm_at_target(ARM, 0.01, 0.05)


def test_a_joint_move_goes_where_the_joints_put_the_hand(engine):
    q = list(engine.arm_joints(ARM))
    q[0] += 0.3
    goal = engine.set_arm_joints(ARM, q)
    assert _settle(engine)
    assert engine.state().arm(ARM).eef_pose.distance_to(goal) <= 0.002


def test_the_gripper_is_told_by_its_fingers():
    eng = _engine(holding_width=0.05)
    try:
        eng.set_gripper(ARM, 1.0)
        assert _settle(eng, gripper=True)
        arm = eng.state().arm(ARM)
        assert arm.gripper == 1.0 and arm.opening == pytest.approx(0.05, abs=2e-3)
        eng.set_gripper(ARM, 0.0)
        assert _settle(eng, gripper=True)
        assert eng.state().arm(ARM).opening == pytest.approx(0.08, abs=2e-3)
    finally:
        eng.close()


def test_the_pictures_are_the_models_and_the_cameras_where_calibrated(engine):
    front, wrist = engine.render("agentview"), engine.render("wrist")
    assert front.image.shape == (480, 640, 3) and front.depth.shape == (480, 640)
    assert wrist.image.shape == (512, 512, 3)
    cal = engine.calibration
    assert engine.camera_pose("agentview").distance_to(
        Pose((cal.front.pos[0], cal.front.pos[1], cal.front.pos[2] - cal.table_z_m))) < 1e-9
    assert engine.visible_objects("agentview") == ()


# -- the real tasks and the episode on the real engine --------------------------------
def test_a_real_task_is_one_setup_judged_by_the_operator():
    from ruibench.core.canary import CANARY
    from ruibench.tasks import real

    inst = real.build("R0")
    assert inst.instruction == "Put the lemon on the plate."
    assert {b.category for b in inst.scene.bodies} == {"lemon", "plate"}
    assert inst.scene.robots[0].model == "panda" and CANARY in inst.secrets
    assert not inst.scorer.success.evaluate([]) and "operator" in inst.scorer.success.describe()
    with pytest.raises(ConfigError):
        real.build("R0", level="hard")
    with pytest.raises(ConfigError, match="no real task"):
        real.build("WK1")
    assert inst.seed == 0
    with pytest.raises(ConfigError, match="has no seed 1"):
        real.build("R0", seed=1)


def test_a_seed_is_a_table_of_its_own():
    from ruibench.tasks import real
    from ruibench.tasks.impl.ds2_egg_hunt import PLACES, hidden_under

    card = real.get("DS2")
    assert real.build("DS2").seed == min(card.seeds) == 15       # rui's first test seed
    assert real.build("DS2", seed=16).seed == 16
    with pytest.raises(ConfigError, match=r"has no seed 0: its card has \[15, 16, 17\]"):
        real.build("DS2", seed=0)
    # hidden where simulation's seed of the same number hides it, each place once
    sleeve = {-0.17: "white", 0.0: "orange", 0.17: "grey"}
    for seed, words in card.seeds.items():
        assert f"under the {sleeve[PLACES['easy'][hidden_under(seed, 3)][1]]} sleeve" in words
        assert card.setup in card.setup_of(seed) and words in card.setup_of(seed)


def _episode(engine, **kw):
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parents[1] / "ruikit"))
    import painted_mock
    from ruikit import Episode
    from ruikit.extras.backends import VISION
    from ruikit.interfaces import load_profile
    from ruibench.tasks.real import build

    task = kw.pop("task", "R0")
    prof = load_profile(kw.pop("profile", "rui.pose/v1"))
    prof = dataclasses.replace(prof, perception="model", params={
        **prof.params, "perception": {**dict(prof.params.get("perception", {})),
                                      "backend": "colour-test"}})
    vision = painted_mock.ColourVision(build(task).scene)
    VISION["colour-test"] = lambda: vision
    try:
        return Episode(task, profile=prof, engine=engine, **kw)
    finally:
        VISION.pop("colour-test", None)


def test_a_real_robot_has_the_measured_clock_only(engine):
    with pytest.raises(ValueError, match="measured clock"):
        _episode(engine, clock="fixed", think_seconds=8.0)


def test_an_episode_runs_on_the_real_engine(engine):
    import time

    ep = _episode(engine, clock="measured")
    info = ep.call("task_info", {}).data
    conv = info["conventions"]
    assert conv["cameras"] == ["front", "wrist"] and conv["estimates"].startswith("this is a real")
    assert "just show in its top corners" in conv["wrist_camera"]
    assert ep.perception_info["sensors"] == "real" and ep.perception.real_sensors
    look = ep.call("camera_look", {"cameras": None})
    assert [(f["width"], f["height"]) for f in look.data["frames"]] == [(640, 480), (512, 512)]
    lifted = ep.call("vision_lift", {"frame_id": "latest.front", "u": 320, "v": 300})
    assert lifted.error is None and "np." not in lifted.text    # plain numbers, as in simulation
    t0 = ep.sim_time
    time.sleep(0.3)                                        # the model thinks: the world goes on
    moved = ep.call("robot_move_to", {"arm": "right", "xyz_m": [0.40, 0.05, 0.15],
                                      "rpy_deg": [0, 0, 0]})
    assert moved.data["status"] == "done", moved.text
    assert ep.log[-1]["think_s"] >= 0.25 and ep.sim_time - t0 < 0.3 + 2.0  # charged, never twice
    ep.call("task_done", {"report": "done"})
    assert ep.outcome.reason == "agent_done" and not ep.outcome.success   # the operator judges
    json.dumps(ep.log)                                      # what the log writes as it ends


def test_the_simulated_station_says_what_it_said():
    """The real robot's words override nothing of the simulated FR3's."""
    from ruikit.env.station import Station
    from ruibench.tasks.instance import build

    st = Station(build("WK1", "easy", 0).scene)
    assert "fingertips show in the bottom corners" in st.conventions("model")["wrist_camera"]
    assert st.conventions("model")["estimates"].startswith("as on a real robot")


# -- the agents' prompt and the plan ------------------------------------------------------
def test_the_real_prompt_says_what_is_real():
    from ruikit.harness.session_agent import PROMPT, REAL_PROMPT

    assert "simulated lab (Isaac Sim)" in PROMPT and "real robot" not in PROMPT.split("\n")[0][:60]
    assert "simulated" not in REAL_PROMPT and "by the wall clock" in REAL_PROMPT
    keep = PROMPT.split("How replies count:")[1]
    assert REAL_PROMPT.endswith(keep)                       # the rules, word for word


def test_the_operator_follows_the_agents_calls_and_a_programs_steps(engine, capsys):
    from ruibench.engines.franka.steps import MARK

    ep = _episode(engine, profile="rui.code/v1", clock="measured")
    assert "first call" in engine.steps.doing()
    ep.call("task_info", {})
    ep.call("code_run", {"source": "here = robot.state()\nrobot.move_to(xyz_m=[0.4, 0.05, 0.15])\n"
                                   "result = 'ran'"})
    assert "waiting for the agent" in engine.steps.doing()
    said = [line[len(MARK):].strip() for line in capsys.readouterr().out.splitlines()
            if line.startswith(MARK)]
    calls = [line.split("  ", 1)[1] if " s  " in line else line for line in said]
    assert calls[0].startswith("#1 task_info") and calls[1].startswith("#1 ->")
    assert calls[2] == "#2 code_run"
    assert calls[3:6] == ["| here = robot.state()", "| robot.move_to(xyz_m=[0.4, 0.05, 0.15])",
                          "| result = 'ran'"]
    assert calls[6] == "#2.1 robot_state" and calls[7].startswith("#2.1 ->")
    assert calls[8].startswith("#2.2 robot_move_to xyz_m=") and "done" in calls[9]
    assert calls[10].startswith("#2 ->") and len(calls) == 11     # no look of the episode's own


def test_the_runner_shows_the_session_and_the_agent_as_they_go(tmp_path):
    import json as j

    from ruikit.experiments.real import Follow

    said = []
    follow = Follow(tmp_path, say=said.append)
    follow.look()                                               # nothing there yet
    (tmp_path / "server.log").write_text("[transformers] noise\n@@   1.0 s  #1 robot_state\n@@ par")
    with open(tmp_path / "agent.jsonl", "w") as fh:
        fh.write(j.dumps({"type": "assistant", "message": {"content": [
            {"type": "thinking", "thinking": ""}, {"type": "text", "text": "I see  a lemon."},
            {"type": "tool_use", "name": "x"}]}}) + "\n")
        fh.write(j.dumps({"type": "item.completed", "item": {
            "type": "agent_message", "text": "Codex words."}}) + "\n")
        fh.write(j.dumps({"type": "item.completed", "item": {"type": "mcp_tool_call"}}) + "\n")
    follow.look()
    assert said == ["     1.0 s  #1 robot_state", "   agent says: I see a lemon.",
                    "   agent says: Codex words."]
    said.clear()
    with open(tmp_path / "server.log", "a") as fh:              # the rest of a line, when it comes
        fh.write("t\n")
    follow.look()
    assert said == ["   part"]
    said.clear()
    (tmp_path / "server.log").write_text("@@ again\n")          # a new try begins the file again
    follow.look()
    assert said == ["   again"]


def _a_judged_episode(out, cell="R0_easy_s0_rui.code-v1+aids", at="2026-11-02T13:00:27+00:00"):
    folder = out / "claude-code-claude-opus-5-5@measured" / cell
    (folder / "images").mkdir(parents=True)
    (folder / "video.mp4").write_bytes(b"film")
    (folder / "images" / "f1.front.jpg").write_bytes(b"jpeg")
    (folder / "prompt.txt").write_text("put it there")
    (folder / "session.json").write_text(json.dumps({"pid": 77, "token": "secret-of-the-session"}))
    (folder / "result.json").write_text(json.dumps({
        "harness": "claude-code", "agent_cli": "2.1", "model": "claude-opus-5-5", "effort": "high",
        "task": "R0", "profile": "rui.code/v1+aids", "success": True, "reason": "agent_done",
        "sim_seconds": 45.8, "wall_seconds": 60.0, "agent_seconds": 50.0, "replies": 2,
        "tool_calls": 2, "summary": "it is on the plate",
        "usage": {"input_tokens": 1234, "cache_read_tokens": 1000, "output_tokens": 56, "usd": 0.15},
        "provenance": {"commit": "26746688fbb0", "frozen": "v2.4.0"},
        "verdict": {"at": at, "note": "a clean one", "criterion": "on the plate"}}))
    answer = {"ok": True, "text": "program finished after 1 tool call(s)",
              "data": {"stdout": "printed this", "result": "returned that"}}
    rows = [{"header": {}},
            {"tool": "code_run", "args": {"source": "robot.move_to([0.4, 0.0, 0.2])"}, "think_s": 3.0,
             "t0": 5.6, "t1": 7.0, "results": [answer],
             "inner": [{"tool": "robot_move_to", "args": {"xyz_m": [0.4, 0.0, 0.2]}, "t0": 5.7,
                        "t1": 6.9, "result": {"text": "move_to: done"}}]},
            {"tool": "task_done", "args": {"report": "done"}, "think_s": 1.0, "t0": 8.0, "t1": 8.0,
             "results": [{"ok": False, "error": "blocked", "text": "not really"}]},
            {"outcome": {}}]
    (folder / f"{cell}.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (folder / "agent.jsonl").write_text(json.dumps({"type": "assistant", "message": {"content": [
        {"type": "text", "text": "I see the lemon."}]}}) + "\n")
    return folder


def test_a_days_episodes_go_up_with_what_they_did_and_what_they_cost(tmp_path):
    pytest.importorskip("huggingface_hub")
    from ruikit.experiments import real_upload as U

    folder = _a_judged_episode(tmp_path)
    _a_judged_episode(tmp_path, "R0_easy_s0_rui.pose-v1+aids", at="2026-11-03T13:00:27+00:00")
    (tmp_path / "flight").mkdir()
    (tmp_path / "flight" / "1102-205905-77.jsonl").write_text("{}")
    (tmp_path / "flight" / "1102-205905-78.jsonl").write_text("{}")
    assert U.episodes(tmp_path, "1102") == [folder]             # that day's alone
    assert U.episodes(tmp_path, "1101") == []
    assert U.episodes(tmp_path, "1102", ["R0"]) == [folder]     # and a suite's, of a day of several
    assert U.episodes(tmp_path, "1102", ["DS2"]) == []
    uploads = U.to_upload("1102-PickAndPlace", tmp_path, [folder])
    at = "1102-PickAndPlace/claude-code-claude-opus-5-5@measured/R0_easy_s0_rui.code-v1+aids"
    assert {f"{at}/{name}" for name in ("video.mp4", "result.json", "agent.jsonl", "flight.jsonl",
                                        "images/f1.front.jpg", "details.md")} <= set(uploads)
    assert uploads[f"{at}/flight.jsonl"].name == "1102-205905-77.jsonl"
    assert not [path for path in uploads if path.endswith("session.json")]
    page = uploads[f"{at}/details.md"].decode()
    for words in ("| Episode's clock | 45.8 s |", "1,234 (1,000)", "| Output tokens | 56 |", "$0.15",
                  "a clean one", "`code_run` x1", "`robot_move_to` x1", "`task_done` x1",
                  "```python\nrobot.move_to([0.4, 0.0, 0.2])\n```", "printed this", "returned that",
                  "#1.1 `robot_move_to`", "-> move_to: done", "!! blocked: not really",
                  "- I see the lemon."):
        assert words in page, words
    day = uploads["1102-PickAndPlace/README.md"].decode()
    assert "1 of 1 episodes succeeded" in day and "rui.code/v1+aids | success | 45.8" in day
    assert "(claude-code-claude-opus-5-5%40measured/R0_easy_s0_rui.code-v1%2Baids/video.mp4)" in day


def test_nothing_goes_up_to_a_repository_open_to_anyone():
    pytest.importorskip("huggingface_hub")
    from types import SimpleNamespace

    from ruikit.experiments import real_upload as U

    class Hub:
        def __init__(self, private=False, gated=False):
            self.private, self.gated, self.made, self.commits = private, gated, None, []

        def create_repo(self, repo, **kw):
            self.made = kw

        def repo_info(self, repo, **kw):
            return SimpleNamespace(private=self.private, gated=self.gated)

        def file_exists(self, repo, name, **kw):
            return False

        def create_commit(self, repo, operations, **kw):
            self.commits.append([op.path_in_repo for op in operations])

    repo = "someone/real-runs"                  # RUIBENCH_REAL_DATASET names it; no default in the release
    with pytest.raises(ValueError, match="RUIBENCH_REAL_DATASET"):
        U.upload({"d/README.md": b"x"}, "m", repo="", api=Hub())
    hub = Hub()
    with pytest.raises(SystemExit, match="open to anyone"):
        U.upload({"d/README.md": b"x"}, "m", repo=repo, api=hub)
    assert hub.made["private"] is True and not hub.commits
    hub = Hub(private=True)
    assert U.upload({"d/README.md": b"x"}, "m", repo=repo, api=hub).endswith("/datasets/" + repo)
    assert hub.commits == [["d/README.md", "README.md"]]
    hub = Hub(gated="manual")
    U.upload({"d/README.md": b"x"}, "m", repo=repo, api=hub)
    assert hub.commits


def test_what_the_operator_must_answer_stands_out_on_a_terminal(monkeypatch, capsys):
    from ruibench.engines.franka import terminal
    from ruikit.experiments import real

    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "s")
    assert real.ask("s skips, q stops:", "sq") == "s"
    assert asked[-1] == "\n >>> s skips, q stops:  "                # a file gets the words alone
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True, raising=False)
    assert real.ask("s skips, q stops:", "sq") == "s"
    assert asked[-1] == "\n\033[1;30;43m >>> s skips, q stops: \033[0m "
    assert terminal.alarm("x") == "\033[1;31mx\033[0m"
    monkeypatch.setenv("NO_COLOR", "1")
    assert terminal.notice("x") == "x"


def test_the_operator_judges_and_a_run_left_unjudged_is_judged_on_resume(tmp_path, monkeypatch):
    import json

    from ruikit.experiments import plans, real

    job = plans.Job("claude-opus-5-5", "R0", "easy", 0, "rui.pose/v1+aids", "claude-code",
                    "measured")
    folder = job.folder(tmp_path)
    folder.mkdir(parents=True)
    (folder / "result.json").write_text(json.dumps(
        {"success": False, "reason": "agent_done", "declared_done": True, "replies": 9}))
    assert real.todo([job], tmp_path) == [job]                  # it ran, nobody judged it
    answers = iter(["", "y", "on the plate, inside the rim"])   # Enter is no answer: asked again
    monkeypatch.setattr("builtins.input", lambda *_: next(answers))
    row = real.judge(job, real._result(folder), folder, say=lambda *_: None)
    assert row["success"] is True and row["judged_by"] == "operator"
    assert row["scored_success"] is False
    assert real._result(folder)["verdict"]["note"] == "on the plate, inside the rim"
    assert real.todo([job], tmp_path) == []


def test_the_real_plan_runs_both_agents_on_every_interface():
    from ruikit.experiments import plans

    jobs = plans.select(plans.load_plan("real"), names=["bring-up"])
    assert len(jobs) == 26 and {j.clock for j in jobs} == {"measured"}
    assert {j.harness for j in jobs} == {"claude-code", "codex"}
    assert {(j.task, j.level, j.seed) for j in jobs} == {("R0", "easy", 0)}


def test_the_suites_are_set_as_the_simulated_experiment_is():
    import re

    from ruibench.core.config import BudgetConfig
    from ruibench.tasks.instance import build
    from ruibench.tasks.real import get
    from ruikit.experiments import plans
    from ruikit.harness.session_agent import effort_of

    real = {e.name: e for e in plans.load_plan("real")}["suites"]
    sim = {e.name: e for e in plans.load_plan("paper")}["rui"]
    assert ([(r.harness, r.model) for r in real.agents]
            == [(r.harness, r.model) for r in sim.agents])
    assert {effort_of(r.harness, r.model) for r in real.agents} == {"high"}
    assert real.profiles == sim.profiles and real.levels == sim.levels == ["easy"]
    assert real.record and sim.record
    assert real.tasks and set(real.tasks) <= set(sim.tasks)
    for task in real.tasks:
        card = get(task)
        assert card.mirrors == task
        assert card.max_model_calls == build(task, "easy", 15).metadata.get(
            "max_model_calls", BudgetConfig.max_model_calls)
        for seed in card.seeds:                                 # what it tells the operator to use
            for named in re.findall(r"[\w/.-]+\.(?:pdf|jpg)", card.setup_of(seed)):
                assert (Path(__file__).parents[2] / named).exists(), named


def test_where_the_block_is_stays_with_the_operator(engine):
    from ruibench.tasks.real import get

    card = get("DS2")
    ep = _episode(engine, clock="measured", task="DS2", seed=15)
    assert ep.budget.max_replies == 1200 and ep.budget.max_sim_seconds == 960.0   # DS2's, easy
    assert ep.instruction == card.instruction and ep.inst.hint == card.hint
    assert "orange" not in card.instruction + card.hint
    said = json.dumps(ep.call("task_info", {}).data)
    assert "orange" not in said and card.seeds[15] not in said and card.setup not in said
    ep.call("task_done", {"report": "done"})


def test_simon_says_is_simulations_game_said_by_the_program():
    from ruibench.tasks.impl.di4_simon_says import LEVEL_TOYS, schedule
    from ruibench.tasks.instance import build as simulated
    from ruibench.tasks.real import build, get
    from ruibench.tasks.real.simon import RealSimon, call_text, calls_of

    card, sim = get("DI4"), simulated("DI4", "easy", 15)
    assert card.instruction == sim.instruction and card.hint == sim.hint
    assert card.success == get_card_success("DI4")
    named = dict(zip(LEVEL_TOYS["easy"], [t.name for t in card.toys]))
    for seed in card.seeds:
        assert calls_of(card.toys, "easy", seed) == [
            (when, simon, named[toy]) for when, simon, toy in schedule("easy", seed)]
    assert [c[0] for c in calls_of(card.toys, "easy", 15)] == [16.0, 36.0, 56.0, 76.0]
    inst = build("DI4", seed=15)
    assert isinstance(inst.reactor_factory(), RealSimon) and build("R0").reactor_factory is None
    assert {b.category for b in inst.scene.bodies} == {"toy"}
    doll = card.toys[0]
    assert call_text(True, doll) == "Simon says: touch the doll."
    assert call_text(False, doll) == "Touch the doll."


def get_card_success(task_id):
    from ruibench.tasks.loader import get

    return " ".join(str(get(task_id).success).split())


def _a_quick_game(toys):
    from ruibench.tasks.real.simon import RealSimon

    calls = [(0.2, True, toys[0].name), (0.7, False, toys[1].name), (1.2, True, toys[2].name)]
    return RealSimon(calls, toys, 0.5)


def test_a_call_is_heard_by_a_call_that_moves_nothing_and_none_is_lost(engine):
    from ruibench.tasks.real import get

    toys = get("DI4").toys
    ep = _episode(engine, clock="measured", task="DI4", seed=15)
    ep._reactor = _a_quick_game(toys)
    assert ep.call("task_events", {"since": 0}).events == []       # the game's clock starts here
    time.sleep(0.3)
    heard = ep.call("task_events", {"since": 0}).events            # no time charged, and it hears
    assert [e["text"] for e in heard] == ["Simon says: touch the doll."]
    time.sleep(1.2)                                                # a long thought: two calls go by
    said = []
    for _ in range(3):
        said += [e["text"] for e in ep.call("robot_state", {"arm": "right"}).events]
    assert said[:2] == ["Touch the race car.", "Simon says: touch the milk stone."]


def test_the_game_is_won_by_touching_what_simon_says_and_nothing_else(engine):
    from ruibench.tasks.real import build, get
    from ruibench.tasks.real.simon import Toy, simon_game
    from ruibench.tasks.impl import di4_simon_says as D

    toys = get("DI4").toys
    assert toys[0].touched_by((0.48, -0.13, 0.11)) and not toys[0].touched_by((0.48, -0.13, 0.20))
    assert not toys[0].touched_by((0.48, 0.0, 0.05)) and toys[1].touched_by((0.47, 0.01, 0.05))
    calls = [(0.0, True, "doll"), (1.0, False, "race_car")]
    fired = ["call_0", "touched_0_doll", "call_1", D.GAME_OVER]
    assert D.obeyed(_Fired(fired), calls) and not D.caught_out(_Fired(fired), calls)
    caught = fired[:3] + ["touched_1_race_car", D.GAME_OVER]
    assert D.caught_out(_Fired(caught), calls)
    scorer, reactor = simon_game("DI4", toys, "easy", 15)
    assert callable(reactor) and not scorer.success.evaluate([_Fired([])])
    assert isinstance(toys[0], Toy) and build("DI4", seed=16).seed == 16


class _Fired:
    def __init__(self, events):
        self.events_fired = tuple(events)


def test_the_hands_body_stays_over_the_table(engine):
    from ruibench.core.geometry import quat_mul

    half = math.radians(-90.0) / 2.0
    level = quat_mul(GRASP_DOWN, (math.cos(half), math.sin(half), 0.0, 0.0))
    assert engine._allowed(Pose((0.5, 0.0, 0.008), GRASP_DOWN))    # pointing down: as ever
    assert not engine._allowed(Pose((0.5, 0.0, 0.05), level))      # level: its body in the table
    assert engine._allowed(Pose((0.5, 0.0, 0.15), level))


def test_the_fingers_close_at_the_rigs_speed_and_force_on_any_server():
    asked = []

    class Told(FakeServer):
        def grasp(self, speed, force):
            asked.append((speed, force))
            super().grasp(speed, force)

    class Older(FakeServer):                       # as zerorpc says a call the server has not
        def grasp(self, speed, force):
            error = RuntimeError("grasp")
            error.name = "NameError"
            raise error

    for server in (Told, Older):
        eng = _engine(server=server)
        try:
            eng.set_gripper(ARM, 1.0)
            assert _settle(eng, gripper=True) and eng.state().arm(ARM).opening < 0.01
        finally:
            eng.close()
    assert asked == [(QUICK_GRIP.grasp_speed_m_s, QUICK_GRIP.grasp_force_n)]


def test_a_hand_whose_client_died_is_a_fault_and_not_a_grip(engine, monkeypatch):
    from ruibench.engines.franka import check

    server = _the_server(engine)
    server.gripper_speed = 0.0                     # it goes on answering, with the last width
    engine._on_the_nuc = True
    asked = []
    monkeypatch.setattr(check, "hand", lambda: asked.append(1) or (True, "running"))
    engine.set_gripper(ARM, 1.0)
    assert _settle(engine, gripper=True) and len(asked) == 1       # alive: an object in the way
    engine.gripper_settled(ARM)
    assert len(asked) == 1                                         # asked once a command
    monkeypatch.setattr(check, "hand", lambda: (False, "the client has died"))
    engine.set_gripper(ARM, 0.0)
    engine.set_gripper(ARM, 1.0)
    with pytest.raises(RobotFault, match="the client has died"):
        _settle(engine, gripper=True)


def test_the_nuc_says_whether_the_hands_client_runs(monkeypatch):
    from types import SimpleNamespace

    from ruibench.engines.franka import check

    said = []

    def answers(code):
        def run(argv, **kw):
            said.append(argv)
            return SimpleNamespace(returncode=code)
        monkeypatch.setattr(check.subprocess, "run", run)

    monkeypatch.setattr(check, "NUC", "")              # no NUC named: the hand is not asked about
    answers(1)
    assert check.hand()[0] and not said
    monkeypatch.setattr(check, "NUC", "robot-pc")      # RUIBENCH_FRANKA_NUC_SSH
    answers(0)
    assert check.hand()[0] and said[-1][-3:] == ["pgrep", "-x", "franka_hand_cli"]
    assert "robot-pc" in said[-1]
    answers(1)
    alive, why = check.hand()
    assert not alive and "launch_gripper.py gripper=franka_hand" in why
    answers(255)                                   # ssh could not ask: nothing is known of it
    assert check.hand()[0]


def test_the_table_is_not_laid_out_for_a_hand_that_does_not_answer(monkeypatch):
    from ruibench.engines.franka import check, engine as engine_module
    from ruikit.experiments import real
    from ruikit.experiments.plans import Job

    job = Job("claude-opus-5-5", "R0", "easy", 0, "rui.pose/v1+aids", "claude-code", "measured")
    went_home, said, answers = [], [], iter([(False, "it died"), (True, "running")])
    monkeypatch.setattr(check, "hand", lambda: next(answers))
    monkeypatch.setattr(engine_module, "home", lambda: went_home.append(1))
    monkeypatch.setattr(real, "ask", lambda asked, *a: said.append(asked) or "g")
    assert real.lay_out(job, 1, 1, say=said.append) == "g"
    assert went_home == [1] and any("it died" in s for s in said)
    assert any("runs again" in s for s in said) and any("laid out" in s for s in said)


def test_a_tasks_picture_comes_by_task_demo_and_moves_nothing(engine):
    from ruibench.tasks.real import get
    from ruikit.experiments import plans
    from ruikit.interfaces import load_profile

    card = get("CL3")
    ep = _episode(engine, clock="measured", task="CL3", seed=15)
    assert ep.call("task_info", {}).data["demonstrations"] == 1
    shown = ep.call("task_demo", {"index": 0})
    assert shown.error is None and shown.data["picture"] and shown.data["frames_total"] == 1
    assert shown.data["caption"] == card.caption and shown.images[0].png
    assert (shown.images[0].width, shown.images[0].height) == (640, 480)
    frame = shown.data["frame_ids"][0]
    assert ep.call("camera_zoom", {"frame_id": frame, "box": [200, 100, 440, 400]}).error is None
    assert "task_demo" in card.instruction
    ep.call("task_done", {"report": "done"})
    # and every interface of the suites can ask for it
    for profile in {e.name: e for e in plans.load_plan("real")}["suites"].profiles:
        prof = load_profile(profile)
        assert "task_demo" in (*prof.tools, *prof.code_tools), profile


def test_a_try_counts_for_nothing_and_a_seed_has_its_own_cells(tmp_path, monkeypatch):
    from ruikit.experiments import real
    from ruikit.experiments import results as cells

    ran = []
    monkeypatch.setattr(real, "TEST_OUT", tmp_path / "tries")
    monkeypatch.setattr(real, "preflight", lambda *a, **k: [])
    monkeypatch.setattr(real, "run", lambda jobs, out, **k: ran.append((jobs, out)) or [])
    args = ["--tasks", "DS2", "--harness", "claude-code", "--out", str(tmp_path / "real")]

    assert real.main(args + ["--interfaces", "pose", "free", "--test"]) == 0
    assert real.main(args + ["--test"]) == 0                     # again: nothing stops a try
    for jobs, out in ran:
        assert [cells.column_of(j.profile) for j in jobs] == ["pose"]
        assert out.parent == tmp_path / "tries" and not (tmp_path / "real").exists()

    ran.clear()
    assert real.main(args + ["--interfaces", "pose"]) == 0
    assert real.main(args + ["--interfaces", "pose", "--seed", "16"]) == 0
    (first,), (second,) = ran[0][0], ran[1][0]
    assert first.seed == 15 and "_s15_" in first.stem and second.seed == 16 and "_s16_" in second.stem
    with pytest.raises(ConfigError, match="has no seed 3"):
        real.main(args + ["--seed", "3"])


def test_a_tasks_episodes_stay_together_for_the_one_table_laid_out():
    from ruikit.experiments import real
    from ruikit.experiments import results as cells
    from ruikit.experiments.plans import Job

    jobs = [Job(m, t, "easy", 0, p, h, "measured")
            for h, m in (("claude-code", "claude-opus-5-5"), ("codex", "gpt-6-astra"))
            for t in ("DS2", "R0") for p in ("rui.pose/v1+aids", "rui.free/v1")]
    assert [(j.task, j.harness, cells.column_of(j.profile))
            for j in real.in_order_asked(jobs, ["free", "pose"])] == [
        ("DS2", h, c) for h in ("claude-code", "codex") for c in ("rui-free", "pose")] + [
        ("R0", h, c) for h in ("claude-code", "codex") for c in ("rui-free", "pose")]


def test_the_script_runs_every_interface_in_its_own_order():
    import re

    from ruikit.experiments import plans, real
    from ruikit.experiments import results as cells

    script = Path(__file__).parents[2] / "scripts" / "real_run.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    listed = re.search(r"INTERFACES=\((.*?)\n\)", script.read_text(), re.S).group(1)
    named = [w for line in listed.splitlines() for w in line.split("#")[0].split()]
    jobs = plans.select(plans.load_plan("real"), names=["bring-up"], columns=named,
                        harnesses=["claude-code"])
    assert len(jobs) == len(named) == 13
    assert [cells.column_of(j.profile) for j in real.in_order_asked(jobs, named)] == named
    assert real.in_order_asked(jobs, []) == jobs                # none named: the plan's
    some = plans.select(plans.load_plan("real"), names=["bring-up"],
                        columns=["free", "code", "pose"], harnesses=["claude-code"])
    assert [cells.column_of(j.profile) for j in real.in_order_asked(some, ["free", "code", "pose"])
            ] == ["rui-free", "program-pose", "pose"]          # the older names are read too


def test_the_first_moves_are_asked_for_once(tmp_path, monkeypatch):
    from ruibench.engines.franka import calibrate as C

    monkeypatch.setattr(C, "MOVES_PASSED", tmp_path / "moves-passed")
    monkeypatch.setattr("builtins.input", lambda *_: "n")       # not yet: asked, and refused
    assert C.main(["moves", "--once"]) == 1
    (tmp_path / "moves-passed").write_text("today\n")
    monkeypatch.setattr("builtins.input", lambda *_: pytest.fail("asked again"))
    assert C.main(["moves", "--once"]) == 0


# -- calibration ---------------------------------------------------------------------
class _PaintedStation:
    """A station whose cameras paint the board on the table from wherever they are:
    known wrist mount, board pose, square size, fixed camera and depth error."""

    SQUARE = 0.0307
    TABLE_Z = 0.0432
    DEPTH_LONG = 1.004                     # what the depth camera reads, of what is there

    def __init__(self):
        from ruibench.core.geometry import camera_quat, quat_mul
        from ruibench.engines.franka import calibrate as C

        def about(axis, deg):
            a = math.radians(deg) / 2
            return (math.cos(a), *(c * math.sin(a) for c in axis))

        self.C, self.server = C, None
        self.board = C.Board(marker_share=0.73)
        self.art = self.board.charuco().generateImage((1400, 1400), marginSize=0, borderBits=1)
        # printed side up (its z into the table), 60 cm out, off to one side
        self.on_table = Pose((0.55, 0.02, self.TABLE_Z), (0, 1, 0, 0))
        # the wrist camera on the flange: turned as the real one, a little off true
        self.mount = Pose((0.062, -0.018, 0.012), quat_mul(about((0, 0, 1), -45 - 90 + 2),
                                                           about((1, 0, 0), 3)))
        eye, target = (1.10, 0.03, 0.62), (0.50, 0.0, self.TABLE_Z)
        self.front = Pose(eye, camera_quat(eye, target))

    def connect(self):
        self.server = FakeServer(lag_s=0.02, sag_m=0.002, gripper_speed=1.0)
        return self.server

    def camera(self, name, c):
        return _PaintedCamera(self, name, c.width, c.height, c.fps)

    def picture(self, pose, k_raw, width, height):
        from ruibench.core.geometry import compose, inverse

        C = self.C
        t = C._matrix(compose(inverse(pose), self.on_table))
        px = np.array([[0, 0], [1400, 0], [1400, 1400], [0, 1400]], float)
        where = np.c_[px / 350.0 * self.SQUARE, np.zeros(4)]
        uvw = (k_raw @ (t[:3, :3] @ where.T + t[:3, 3:4])).T
        if (uvw[:, 2] <= 0).any():
            return np.full((height, width, 3), 150, dtype=np.uint8)
        h = cv2.getPerspectiveTransform((px - 0.5).astype(np.float32),
                                        (uvw[:, :2] / uvw[:, 2:]).astype(np.float32))
        out = cv2.warpPerspective(self.art, h, (width, height), flags=cv2.INTER_AREA,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=170)
        return cv2.cvtColor(out, cv2.COLOR_GRAY2RGB)

    def depth(self, pose, k_raw, width, height):
        m = self.C._matrix(pose)
        us, vs = np.meshgrid((np.arange(width) - k_raw[0, 2]) / k_raw[0, 0],
                             (np.arange(height) - k_raw[1, 2]) / k_raw[1, 1])
        down = (np.stack([us, vs, np.ones_like(us)], axis=-1) @ m[:3, :3].T)[..., 2]
        return ((self.TABLE_Z - m[2, 3]) / down * self.DEPTH_LONG).astype(np.float32)


class _PaintedCamera:
    def __init__(self, station, name, width=640, height=480, fps=30, depth=True):
        self.station, self.name, self.has_depth = station, name, depth
        self.width, self.height = width, height
        wide = 392.0 if name == "wrist" else 605.0
        scale = width / 640.0
        self.intrinsics = Intrinsics(width, height, wide * scale, wide * scale,
                                     width / 2.0 - 0.5, height / 2.0 - 0.5)
        self.depth_scale = 1.0

    def start(self):
        return self

    def capture(self):
        from ruibench.core.geometry import compose
        from ruibench.engines.franka.arm import from_wire

        s, i = self.station, self.intrinsics
        k = np.array([[i.fx, 0, i.ppx], [0, i.fy, i.ppy], [0, 0, 1]])
        pose = (compose(from_wire(s.server.get_ee_pose()), s.mount) if self.name == "wrist"
                else s.front)
        colour = s.picture(pose, k, self.width, self.height)
        return colour, s.depth(pose, k, self.width, self.height) if self.has_depth else None

    def close(self):
        pass


@pytest.mark.skipif(not hasattr(cv2, "calibrateHandEye") or not hasattr(cv2, "aruco")
                    or not hasattr(cv2.aruco, "CharucoDetector"),
                    reason="needs OpenCV 4.8-4.x (5.0 has no calibrateHandEye)")
def test_a_run_finds_the_board_and_measures_the_station(tmp_path, monkeypatch):
    """The whole of ``calibrate run`` on a station of painted pictures, where every
    answer is known: it searches for the board, sees it round about, and comes to
    the wrist camera's mount, the fixed camera, the table, a square's size and the
    depth's error."""
    from ruibench.engines.franka import calibrate as C

    painted = _PaintedStation()
    monkeypatch.setattr(calibration, "PATH", tmp_path / "calibration.json")
    monkeypatch.setattr(C, "STILL_SECONDS", 0.05)
    monkeypatch.setattr(C, "RealSenseCamera", lambda serial, w, h, fps, depth=True: _PaintedCamera(
        painted, "agentview", w, h, fps, depth))
    rig = dataclasses.replace(rig_mod.load(), motion=FAST, gripper=QUICK_GRIP)
    engine = FrankaEngine(rig=rig, calibration=calibration.provisional(),
                          connect=painted.connect, cameras=painted.camera)
    said = []
    folder = C.capture(tmp_path / "run", C.Board(), say=said.append, engine=engine)
    cal = C.solve(folder, say=said.append)
    text = "\n".join(said)
    assert cal.fit["square_m"] == pytest.approx(painted.SQUARE, rel=0.005), text
    assert cal.fit["board"]["marker_share"] == pytest.approx(0.73, abs=0.01), text
    assert cal.wrist.distance_to(painted.mount) < 0.003, text
    assert math.degrees(cal.wrist.angle_to(painted.mount)) < 0.3, text
    assert cal.table_z_m == pytest.approx(painted.TABLE_Z, abs=0.001), text
    assert cal.front.distance_to(painted.front) < 0.004, text
    assert math.degrees(cal.front.angle_to(painted.front)) < 0.3, text
    for name in ("wrist", "agentview"):
        assert cal.depth_scale[name] == pytest.approx(1.0 / painted.DEPTH_LONG, abs=0.003), text
    assert calibration.load().measured == cal.measured
    assert not calibration.load().provisional


@pytest.mark.skipif(not hasattr(cv2, "calibrateHandEye") or not hasattr(cv2, "aruco")
                    or not hasattr(cv2.aruco, "CharucoDetector"),
                    reason="needs OpenCV 4.8-4.x (5.0 has no calibrateHandEye)")
def test_a_run_cut_short_keeps_what_it_saw(tmp_path, monkeypatch):
    from ruibench.engines.franka import calibrate as C

    def no_camera(*_, **__):
        raise RuntimeError("the camera went away")

    painted = _PaintedStation()
    monkeypatch.setattr(C, "STILL_SECONDS", 0.02)
    monkeypatch.setattr(C, "FIRST_VIEWS", C.FIRST_VIEWS[:3])
    monkeypatch.setattr(C, "VIEWS", ())
    monkeypatch.setattr(C, "RealSenseCamera", no_camera)
    rig = dataclasses.replace(rig_mod.load(), motion=FAST, gripper=QUICK_GRIP)
    engine = FrankaEngine(rig=rig, calibration=calibration.provisional(),
                          connect=painted.connect, cameras=painted.camera)
    with pytest.raises(RuntimeError, match="went away"):
        C.capture(tmp_path / "run", C.Board(), say=lambda *_: None, engine=engine)
    views = json.loads((tmp_path / "run" / "views.json").read_text())
    assert len(views) == 3 and len(views[0]["joints"]) == 7
    with pytest.raises(SystemExit, match="cut short after 3 views"):
        C.solve(tmp_path / "run", say=lambda *_: None)
