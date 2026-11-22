"""The real station as an engine: the lab's Franka Panda and its RealSense cameras.

What the rest of RUIBench asks of an engine, answered by the robot:

* **time** is the wall clock from the reset: ``advance`` waits, it does not
  step -- the arm moves on its own control thread meanwhile (``arm``), and the
  world goes on whether or not a model is thinking (``capabilities.real``).  Only
  the operator stops it: ``pause`` holds the arm and the clock together, so that
  no move finds itself blocked or timed out by it and no limit is charged for it;
* **the arm** is the tool centre point the moves take, in the world (the
  station frame: the table top at z = 0 under the base).  A move is followed
  before it starts as the robot's own controller would follow it (``route``):
  inside the workspace, tilted no further than the rig allows, the joints inside
  their limits.  It runs as far as it may, slowly enough for its joints: a move
  that may not go all the way stops where it must, and the executor finds it
  blocked short, as in simulation;
* **the pictures** are the cameras' (``cameras``), through the model's lens;
  the cameras stand where the calibration puts them (``calibration``);
* **the world's state** is the robot alone -- its hand, fingers and joints.
  Nothing knows where the things on the table are: an operator judges how an
  episode went.

On the real robot the operator has a window on it all (``view``): both cameras,
live, and the key that pauses.

``RUIBENCH_FRANKA_FAKE=1`` stands the robot and cameras in with the fakes of
``engines.franka.fake``: the engine runs, nothing moves, nothing counts.
"""

from __future__ import annotations

import math
from itertools import product
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ...core.errors import ConfigError
from ...core.geometry import compose, inverse, rotate
from ...core.types import (
    ArmState, CameraFrame, Pose, RobotState, WorldState,
)
from .. import kinematics, robots
from ..base import (
    ARRIVAL_POS_TOL, ARRIVAL_ROT_TOL, GRIP_CLOSING, HORIZONTAL_APERTURE, Engine,
    EngineCapabilities, SceneSpec, camera_apertures, camera_focal_length,
)
from . import calibration as calibration_mod
from . import rig as rig_mod
from .arm import Fault, FrankaArm
from .cameras import RealSenseCamera, Rectifier
from .model import HAND_BODY_M, HOME_JOINTS, PANDA
from .route import Route, along
from .steps import Steps
from .view import VIEW_ENV, LiveView

#: stand the robot and cameras in with fakes (``engines.franka.fake``)
FAKE_ENV = "RUIBENCH_FRANKA_FAKE"
#: where what the real arm did is kept (``arm.Flight``), a file for each connection
FLIGHTS = Path("runs/real/flight")
#: how often a wait looks whether the operator's pause has ended
PAUSE_POLL_SECONDS = 0.02
#: where F in the operator's window leaves word for the runner: set for an episode's session
FAIL_FILE_ENV = "RUIBENCH_FRANKA_FAIL_FILE"
#: how far the hand's body stays over the table
HAND_OVER_TABLE_M = 0.005
#: how long the reset may take to bring the arm to its start
RESET_SECONDS = 40.0
#: a share of a rectified picture the camera never saw, black at its edges,
#: past which the model's lens is refused for that camera
MAX_UNCOVERED = 0.002


class RobotFault(RuntimeError):
    """The robot cannot go on: it stopped answering, faulted, or was asked for
    something unsafe.  It ends the episode, which is never scored."""


class _Clock:
    """The wall clock from its start, stopped while the operator has the station paused."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t0 = time.monotonic()
        self._stopped_at: Optional[float] = None

    def start(self) -> None:
        with self._lock:
            self._t0 = self._stopped_at if self._stopped else time.monotonic()

    @property
    def _stopped(self) -> bool:
        return self._stopped_at is not None

    @property
    def stopped(self) -> bool:
        with self._lock:
            return self._stopped

    def read(self) -> float:
        with self._lock:
            return (self._stopped_at if self._stopped else time.monotonic()) - self._t0

    def stop(self) -> None:
        with self._lock:
            if not self._stopped:
                self._stopped_at = time.monotonic()

    def go_on(self) -> None:
        with self._lock:
            if self._stopped:
                self._t0 += time.monotonic() - self._stopped_at
                self._stopped_at = None


class FrankaEngine(Engine):
    capabilities = EngineCapabilities(
        name="franka", concurrent_arms=False, springs=False, actuated_joints=False,
        rendering=True, mesh_assets=False, max_camera_size=640,
        photoreal=True, results_grade=True, dynamics=True, cloth=False, real=True,
    )

    def __init__(self, rig: Optional[rig_mod.Rig] = None,
                 calibration: Optional[calibration_mod.Calibration] = None,
                 connect: Optional[Callable[[], object]] = None,
                 cameras: Optional[Callable[[str, rig_mod.CameraRig], object]] = None) -> None:
        """``connect`` makes the NUC server's client, ``cameras(name, rig)`` a
        camera: the robot's by default, the fakes' under ``RUIBENCH_FRANKA_FAKE``."""
        self.fake = os.environ.get(FAKE_ENV, "") == "1"
        #: the lab's own robot, whose NUC can be asked about the hand
        self._on_the_nuc = connect is None and not self.fake
        self.rig = rig or rig_mod.load()
        self.calibration = calibration or calibration_mod.load()
        if self.fake:
            from .fake import FakeCamera, FakeServer

            connect = connect or FakeServer
            cameras = cameras or (lambda name, c: FakeCamera(c.serial, c.width, c.height, c.fps))
        self._connect = connect
        self._make_camera = cameras or (lambda name, c: RealSenseCamera(c.serial, c.width,
                                                                        c.height, c.fps))
        self.control_hz = self.rig.motion.control_hz
        #: an engine with ground truth tracks sight lines for the tasks that score
        #: seeing; this one knows no thing, so none
        self.track_visibility = False
        self.arm: Optional[FrankaArm] = None
        self.view: Optional[LiveView] = None      # the operator's window
        self._failed = False                      # the operator has failed this episode (F)
        self.steps = Steps()                      # what the agent is doing, for the operator
        self._scene: Optional[SceneSpec] = None
        self._arm_id = ""
        self._base = Pose()                       # the robot base in the world
        self._origin = (0.0, 0.0, 0.0)            # the station frame's origin in the world
        self._cameras: Dict[str, object] = {}     # rig camera name -> camera
        self._rectifiers: Dict[str, Rectifier] = {}
        self._clock = _Clock()
        self._target: Optional[Pose] = None       # where the hand was sent (world)
        self._grip_value = 0.0                    # the gripper command: 0 open, 1 closed
        self._grip_since = 0.0
        self._grip_from = 0.0                     # the width when it was given
        self._grip_moves = False                  # does the command need the fingers to move?
        self._grip_asked = False                  # has the NUC been asked why they did not?
        self._widths: List[Tuple[float, float]] = []
        self._instruction = ""
        self._instruction_version = 0
        self._events_fired: List[str] = []

    def describe(self) -> dict:
        out = super().describe()
        out.update(rig=self.rig.describe(), fake=self.fake,
                   calibration={**self.calibration.as_dict(),
                                "provisional": self.calibration.provisional},
                   cameras={name: {"serial": self.rig.cameras[name].serial,
                                   "raw": r.raw.as_dict(), "size": [r.width, r.height],
                                   "focal_px": round(r.focal_px, 2)}
                            for name, r in self._rectifiers.items()})
        return out

    @property
    def arm_id(self) -> str:
        """The arm, as the scene names it (``robot0/right``)."""
        return self._arm_id

    @property
    def base(self) -> Pose:
        """The robot's base frame in the world: the frame Polymetis reports in."""
        return self._base

    # -- lifecycle ---------------------------------------------------------------
    def load(self, scene: SceneSpec, seed: int) -> None:
        """Take the real station's scene: connect the arm and start the cameras
        the first time, and check that the scene is this station's."""
        robot = self._station_robot(scene)
        self._scene = scene
        self._arm_id = f"{robot.name}/{robot.arms[0]}"
        self._base = kinematics.arm_frame(robot, robot.arms[0], scene.table_height)
        self._origin = (self._base.pos[0], self._base.pos[1], float(scene.table_height))
        built_for = scene.table_height - self._base.pos[2]
        if abs(built_for - self.calibration.table_z_m) > 1e-3:
            raise ConfigError(f"the scene puts the table {built_for:.4f} m above the robot base, "
                              f"the calibration {self.calibration.table_z_m:.4f} m: build it "
                              "with engines.franka.scene.station_scene")
        if self.arm is None:
            # the real robot's alone: a stand-in has nothing to tell
            flight = None if self._connect else FLIGHTS / (
                datetime.now().strftime("%m%d-%H%M%S-") + f"{os.getpid()}.jsonl")
            self.arm = FrankaArm(self.rig.host, self.rig.port, self.rig.motion, self._connect,
                                 flight, gripper=self.rig.gripper)
            self._wrap(self.arm.start)
        for cam in scene.cameras:
            if cam.name not in self.rig.cameras:
                raise ConfigError(f"the scene's camera {cam.name!r} is no camera of the rig "
                                  f"(rig.yaml has {sorted(self.rig.cameras)})")
            if cam.name not in self._cameras:
                made = self._make_camera(cam.name, self.rig.cameras[cam.name])
                self._cameras[cam.name] = made.start()
            self._rectifiers[cam.name] = self._rectifier(scene, cam)
        self._open_view(scene)

    def _open_view(self, scene: SceneSpec) -> None:
        """The operator's window: on the real robot unless told not to, on a stand-in
        if told to."""
        wanted = os.environ.get(VIEW_ENV, "")
        if self.view is None and self._cameras and (
                wanted == "1" or (wanted != "0" and self._connect is None)):
            self.view = LiveView(f"RUIBench real station: {scene.name}", self.live_pictures,
                                 self._view_state, self.toggle_pause, self.steps.doing,
                                 self.fail_by_operator if os.environ.get(FAIL_FILE_ENV) else None
                                 ).start()

    def live_pictures(self) -> Dict[str, object]:
        """The cameras' newest pictures, as the tools get them, with no depth."""
        return {name: self._rectifiers[name].colour(camera.colour())
                for name, camera in list(self._cameras.items()) if name in self._rectifiers}

    def _view_state(self) -> Tuple[str, str]:
        """What the window's top strip says, and which state it says it in."""
        try:
            self.arm.check()
        except Fault as exc:
            return f"ARM FAULT: {exc}"[:100], "fault"
        if self._failed:
            return "FAILED by the operator: the arm holds, the episode is ending", "fault"
        if self.paused:
            return f"PAUSED at {self.sim_time:.0f} s: the arm holds, P goes on", "paused"
        return f"running: {self.sim_time:.0f} s", "running"

    def _station_robot(self, scene: SceneSpec):
        if len(scene.robots) != 1 or len(scene.robots[0].arms) != 1:
            raise ConfigError("the real station has one robot with one arm")
        robot = scene.robots[0]
        if robots.model_of(robot).name != PANDA.name:
            raise ConfigError(f"the real station's robot is the {PANDA.name!r} model, not "
                              f"{robot.model!r}: build the scene with "
                              "engines.franka.scene.station_scene")
        return robot

    def _rectifier(self, scene: SceneSpec, cam) -> Rectifier:
        """The camera's raw picture onto the model's lens for it, as wide as
        the scene asks; refused where that lens sees past the raw picture."""
        across, _ = camera_apertures(scene, cam)
        focal_px = cam.width * camera_focal_length(scene, cam) / (across or HORIZONTAL_APERTURE)
        r = Rectifier(self._cameras[cam.name].intrinsics, cam.width, cam.height, focal_px)
        if r.uncovered > MAX_UNCOVERED:
            raise ConfigError(f"camera {cam.name!r}: the model's lens sees past its picture "
                              f"({r.uncovered:.1%} of {cam.width}x{cam.height} at "
                              f"{focal_px:.0f} px would be black)")
        return r

    def reset(self) -> WorldState:
        """Open the hand and bring the arm to its start, joint by joint, in the
        posture sought from the lab's HOME: an episode's moves leave the arm's
        redundancy anywhere, and the next one starts from the same joints.  The
        clock starts there.  The things on the table are the operator's."""
        assert self._scene is not None, "call load() first"
        robot = self._scene.robots[0]
        start = robots.arm_start_pose(self._scene, robot, robot.arms[0])
        self.set_gripper(self._arm_id, 0.0)
        joints = self.arm_joints(self._arm_id)
        sol = kinematics.inverse(PANDA, self._base, start, seed=HOME_JOINTS)
        if not sol.ok:
            raise RobotFault(f"the start pose {start.pos} is out of the arm's reach")
        if not self._joint_path_allowed(joints, sol.joints):
            raise RobotFault("the way to the start leaves the workspace: bring the arm home "
                             "first (on the NUC: cd franka_server && python go_home_client.py)")
        self._clock.start()
        self.set_arm_joints(self._arm_id, sol.joints)
        while not (self.arm_at_target(self._arm_id) and self.gripper_settled(self._arm_id)):
            if self.sim_time > RESET_SECONDS:
                raise RobotFault(f"the arm did not reach its start within {RESET_SECONDS:g} s")
            self.advance(0.05)
        self._clock.start()
        self._instruction, self._instruction_version, self._events_fired = "", 0, []
        return self.state()

    def close(self) -> None:
        if self.view is not None:
            self.view.close()
            self.view = None
        if self.arm is not None:
            self.arm.close()
            self.arm = None
        for cam in self._cameras.values():
            cam.close()
        self._cameras.clear()

    # -- time ----------------------------------------------------------------------
    @property
    def sim_time(self) -> float:
        return self._clock.read()

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` of the world go by: the arm runs meanwhile.  A pause is no
        part of them: what is left of them goes by when it ends."""
        end = self.sim_time + max(0.0, seconds)
        while True:
            left = end - self.sim_time
            if left <= 0 and not self._clock.stopped:
                break
            time.sleep(PAUSE_POLL_SECONDS if self._clock.stopped else left)
            self._wrap(self.arm.check)
        self._wrap(self.arm.check)

    @property
    def paused(self) -> bool:
        return self._clock.stopped

    def pause(self) -> None:
        """The operator's pause: the arm holds where it is, and the clock stops with
        it (the module docstring)."""
        self._clock.stop()
        self._wrap(self.arm.pause)

    def resume(self) -> None:
        self._wrap(self.arm.resume)
        self._clock.go_on()

    def toggle_pause(self) -> None:
        (self.resume if self.paused else self.pause)()

    def fail_by_operator(self) -> None:
        """F: the operator fails this episode.  The arm stops where it is, a pause ends so that
        what waits on it can return, and nothing moves it again, a program's next moves
        among them; the runner, told by a file, ends the episode."""
        if self._failed:
            return
        self._failed = True
        self.arm.note("the operator fails this episode")
        self._wrap(self.arm.hold)
        if self.paused:
            self.resume()
        Path(os.environ[FAIL_FILE_ENV]).write_text(datetime.now().isoformat(timespec="seconds"))

    def _wrap(self, call, *args):
        try:
            return call(*args)
        except Fault as exc:
            raise RobotFault(str(exc)) from None

    # -- what the robot reads ------------------------------------------------------
    def _tcp(self, flange: Pose) -> Pose:
        return compose(self._base, calibration_mod.flange_to_tcp(flange))

    def state(self) -> WorldState:
        r = self._wrap(self.arm.reading)
        self._note_width(r.t, r.width)
        robot, _, part = self._arm_id.partition("/")
        arm = ArmState(name=self._arm_id, eef_pose=self._tcp(r.flange), gripper=self._grip_value,
                       holding=None, busy=not (self.arm_at_target(self._arm_id)
                                               and self.gripper_settled(self._arm_id)),
                       opening=r.width, joints=r.joints)
        return WorldState(sim_time=self.sim_time, objects={},
                          robots={robot: RobotState(robot, {part: arm})},
                          instruction=self._instruction,
                          instruction_version=self._instruction_version,
                          events_fired=tuple(self._events_fired))

    def arm_joints(self, arm: str) -> Optional[Tuple[float, ...]]:
        return self._wrap(self.arm.reading).joints

    def render(self, camera: str) -> CameraFrame:
        spec = next(c for c in self._scene.cameras if c.name == camera)
        rect = self._rectifiers[camera]
        try:
            colour, depth = self._cameras[camera].capture()
        except RuntimeError as exc:
            raise RobotFault(f"camera {camera!r}: {exc}") from None
        scale = float(self.calibration.depth_scale.get(camera, 1.0))
        return CameraFrame(camera=camera, width=rect.width, height=rect.height,
                           captured_at=self.sim_time, image=rect.colour(colour),
                           depth=rect.depth(depth) * scale if spec.depth else None)

    def camera_pose(self, camera: str) -> Pose:
        spec = next(c for c in self._scene.cameras if c.name == camera)
        if spec.kind == "wrist":
            flange = self._wrap(self.arm.reading).flange
            return compose(compose(self._base, flange), self.calibration.wrist)
        return compose(self._base, self.calibration.front)

    def visible_objects(self, camera: str) -> Tuple[str, ...]:
        """Nothing is known of the things on the table, sight lines least."""
        return ()

    # -- the arm -------------------------------------------------------------------
    def set_arm_target(self, arm: str, pose: Pose) -> None:
        """Send the tool centre point to ``pose`` (world), as far along the
        straight line as it may go and as slowly as its joints need (the module
        docstring)."""
        self._target = pose
        if self._failed:
            self._wrap(self.arm.hold)
            return
        way = self.route(pose)
        if way.stopped:
            self.arm.note(f"a line to {[round(v, 3) for v in pose.pos]} stops short: {way.stopped}")
        if way.end is None:
            self._wrap(self.arm.hold)
            return
        self._wrap(self.arm.move, self._to_flange(way.end), way.seconds)

    def set_arm_joints(self, arm: str, joints: Sequence[float]) -> Pose:
        """Joint targets, inside the robot's limits, along a path in joint space;
        refused -- the arm holds, and the executor finds it blocked -- where the
        path would take the hand out of the workspace."""
        kin = kinematics.arm_kinematics(PANDA.name)
        if len(joints) != len(kin.joints):
            raise ValueError(f"{arm} has {len(kin.joints)} joints, not {len(joints)}")
        q1 = tuple(min(max(float(v), lo), hi) for v, lo, hi in zip(joints, kin.lower, kin.upper))
        goal = kinematics.forward(PANDA, self._base, q1)
        self._target = goal
        if self._failed or not self._joint_path_allowed(self.arm_joints(arm), q1):
            self._wrap(self.arm.hold)
            return goal
        self._wrap(self.arm.move_joints, q1, self._to_flange(goal))
        return goal

    def _joint_path_allowed(self, q0: Sequence[float], q1: Sequence[float]) -> bool:
        """Does the straight path in joint space keep the hand allowed (``_allowed``)?"""
        steps = max(2, int(max(abs(b - a) for a, b in zip(q0, q1)) / 0.05) + 1)
        return all(self._allowed(kinematics.forward(
            PANDA, self._base, [a + (b - a) * i / steps for a, b in zip(q0, q1)]))
            for i in range(1, steps + 1))

    def _to_flange(self, tcp: Pose) -> Pose:
        """The flange's pose in the base frame that puts the tool centre point at ``tcp``."""
        return calibration_mod.tcp_to_flange(compose(inverse(self._base), tcp))

    def _allowed(self, tcp: Pose) -> bool:
        """Inside the workspace (station frame), tilted no further than the rig allows, and
        the hand's body over the table: a level hand (``skill_pick``'s side, ``skill_turn``'s
        tip) stands 10 cm below its tool centre point, and pressing on the table is a reflex
        here where in simulation it is only a hand held back."""
        ws = self.rig.workspace
        here = tuple(p - o for p, o in zip(tcp.pos, self._origin))
        axis = rotate(tcp.quat, (0.0, 0.0, 1.0))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, -axis[2]))))
        lowest = min(rotate(tcp.quat, corner)[2] for corner in product(*HAND_BODY_M))
        return (ws.contains(here) and tilt <= ws.max_tilt_deg + 1e-6
                and here[2] + lowest >= HAND_OVER_TABLE_M)

    def route(self, goal: Pose) -> Route:
        """The straight line from the hand's target to ``goal`` (world), followed
        from the joints as the robot's own controller would (``route.along``)."""
        r = self._wrap(self.arm.reading)
        return along(PANDA, self._base, r.joints, self._tcp(r.aim), goal, self.rig.motion,
                     self._allowed)

    def joints_for(self, pose: Pose, seed: Optional[Sequence[float]] = None
                   ) -> Optional[Tuple[float, ...]]:
        """Joints that put the tool centre point at ``pose`` (world), inside their
        limits, sought from ``seed`` (else from where they are), the straight way
        there in joint space keeping the hand allowed; None where there are none."""
        here = self.arm_joints(self._arm_id)
        sol = kinematics.inverse(PANDA, self._base, pose, seed=seed or here)
        return sol.joints if sol.ok and self._joint_path_allowed(here, sol.joints) else None

    def arm_at_target(self, arm: str, pos_tol: float = ARRIVAL_POS_TOL,
                      rot_tol: float = ARRIVAL_ROT_TOL) -> bool:
        if self._target is None or self._wrap(self.arm.busy):
            return False
        here = self._tcp(self._wrap(self.arm.reading).flange)
        return here.distance_to(self._target) <= pos_tol and here.angle_to(self._target) <= rot_tol

    def stop_arm(self, arm: str) -> None:
        """Hold where the hand is now: target and setpoint both."""
        self._wrap(self.arm.hold)
        self._target = self._tcp(self._wrap(self.arm.reading).flange)

    # -- the gripper ---------------------------------------------------------------
    def set_gripper(self, arm: str, value: float) -> None:
        """0 open, 1 closed; ``GRIP_CLOSING`` and above close on what is there,
        below it the fingers open to (1 - value) of their full opening.  A
        command that changes nothing -- closing a closed hand, opening an open
        one -- is not sent: the Franka Hand's client can die of a command that
        arrives while it is carrying one out (``single_arm_server.py``'s note)."""
        value = float(min(1.0, max(0.0, value)))
        r = self._wrap(self.arm.reading)
        g = self.rig.gripper
        close = value >= GRIP_CLOSING
        width = None if value <= 0.0 else (1.0 - value) * g.open_width_m
        if close:
            self._grip_moves = self._grip_value < GRIP_CLOSING
        else:
            self._grip_moves = abs(r.width - (g.open_width_m if width is None else width)) > 0.005
        self._grip_value, self._grip_since, self._grip_from = value, self.sim_time, r.width
        self._widths, self._grip_asked = [], False
        self._grip_moves = self._grip_moves and not self._failed
        if self._grip_moves:
            self._wrap(self.arm.grip, close, None if close else width)

    def gripper_settled(self, arm: str) -> bool:
        """The fingers have stopped: they moved since the command (if it asked
        them to) and then held still for ``stable_s`` -- not before ``min_s``,
        and in any case by ``max_s``: the Franka Hand reports its width late,
        so the first readings after a command are the old ones.  Fingers that did not
        move by then may belong to a hand whose client has died, while the robot server
        goes on with the last width: that is a fault, and not a grip."""
        g = self.rig.gripper
        r = self._wrap(self.arm.reading)
        self._note_width(r.t, r.width)
        now = time.monotonic()
        since = self.sim_time - self._grip_since
        moved = any(abs(w - self._grip_from) > 0.005 for _, w in self._widths)
        if since >= g.max_s:
            if self._grip_moves and not moved and self._on_the_nuc and not self._grip_asked:
                self._grip_asked = True
                from .check import hand

                alive, said = hand()
                if not alive:
                    self.arm.note(f"fault: {said}")
                    raise RobotFault(said)
            return True
        if since < g.min_s:
            return False
        recent = [w for t, w in self._widths if now - t <= g.stable_s]
        covered = any(now - t >= g.stable_s for t, _ in self._widths)
        stable = covered and bool(recent) and max(recent) - min(recent) <= g.stable_m
        return stable and (moved or not self._grip_moves)

    def _note_width(self, t: float, width: float) -> None:
        if not self._widths or self._widths[-1][0] != t:
            self._widths.append((t, width))
            del self._widths[:-200]

    # -- the world -----------------------------------------------------------------
    def set_articulation_target(self, joint: str, value: float) -> None:
        raise NotImplementedError("the real station drives no joint of a toy")

    def set_instruction(self, text: str, initial: bool = False) -> None:
        self._instruction = text
        if not initial:
            self._instruction_version += 1

    def note_call(self, name: str, args, inner: bool) -> None:
        """The agent's call, or one inside it, starts: said to the operator (``steps.py``)."""
        self.steps.began(self.sim_time, name, args, inner)

    def note_answer(self, result, inner: bool) -> None:
        self.steps.ended(self.sim_time, result, inner)

    def note_event(self, name: str) -> None:
        self._events_fired.append(name)


def home(rig: Optional[rig_mod.Rig] = None,
         calibration: Optional[calibration_mod.Calibration] = None) -> None:
    """Bring the arm to its start with the hand open, as an episode's reset does:
    between episodes, so that the operator can lay out the next table."""
    from ...core.config import RunConfig
    from ...runner.episode import _sized
    from .scene import station_scene

    engine = FrankaEngine(rig, calibration)
    try:
        engine.load(_sized(station_scene("home", cal=engine.calibration), RunConfig()), 0)
        engine.reset()
    finally:
        engine.close()


__all__ = ["FAIL_FILE_ENV", "FAKE_ENV", "FrankaEngine", "RobotFault", "home"]
