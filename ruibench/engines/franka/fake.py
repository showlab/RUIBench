"""The real station without the robot: the NUC's server and the cameras, in the process.

For the tests and for a dry run of the real runner (``RUIBENCH_FRANKA_FAKE=1``):
the engine, its control thread and its rectification run as on the robot,
and nothing moves.  Never a result.

* ``FakeServer`` answers the FrankaInterface API.  A Cartesian setpoint becomes
  joint targets from where the joints are, as the robot's own inverse kinematics
  has it (``route.solve``); the joints follow their targets with a lag, and the
  flange rests ``sag_m`` under its setpoint, as an impedance controller sags under
  a hand it does not quite know.  What would trip the robot's reflexes -- a joint
  past the server's limit on its place or speed -- goes in ``reflexes``; the arm
  stops short of ``floor_z``, as on something in the way.  The fingers close on
  ``holding_width`` or on nothing.  It answers a setpoint with no controller
  running as the real server does.
* ``FakeCamera`` delivers a grey picture and a table ``depth_m`` away.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Optional, Sequence

from ...core.geometry import compose
from ...core.types import Pose
from .. import kinematics
from .arm import from_wire, to_wire
from .calibration import flange_to_tcp
from .cameras import Intrinsics
from .model import (
    HAND_ON_FLANGE_RAD, HOME_JOINTS, PANDA, PANDA_KINEMATICS, SERVER_LOWER, SERVER_SPEED,
    SERVER_UPPER,
)
from .route import solve

#: the flange in the hand's frame
_HAND_TO_FLANGE = Pose((0.0, 0.0, 0.0), (math.cos(-HAND_ON_FLANGE_RAD / 2), 0.0, 0.0,
                                         math.sin(-HAND_ON_FLANGE_RAD / 2)))


def flange_at(q: Sequence[float]) -> Pose:
    """Where the joints put the flange, in the base frame."""
    return compose(kinematics.body_in_base(PANDA_KINEMATICS, q), _HAND_TO_FLANGE)


class FakeServer:
    """``single_arm_server.py``, in the process (module docstring)."""

    def __init__(self, joints: Sequence[float] = HOME_JOINTS, lag_s: float = 0.05,
                 sag_m: float = 0.003, holding_width: Optional[float] = None,
                 gripper_speed: float = 0.1, floor_z: Optional[float] = None) -> None:
        self.q = list(map(float, joints))
        self.flange = flange_at(self.q)
        self.lag_s, self.sag_m, self.gripper_speed = lag_s, sag_m, gripper_speed
        self.holding_width, self.floor_z = holding_width, floor_z
        self.reflexes: list = []
        self.fastest = [0.0] * len(self.q)                # each joint's speed so far, rad/s
        self.width, self.width_goal = 0.08, 0.08
        self.policy: Optional[str] = None
        self.setpoint: Optional[Pose] = None
        self.q_goal = list(self.q)
        self._t = time.monotonic()
        self._lock = threading.Lock()
        self.calls = 0

    def _advance(self) -> None:
        now = time.monotonic()
        dt, self._t = now - self._t, now
        k = 1.0 - math.exp(-dt / self.lag_s) if self.lag_s > 0 else 1.0
        if self.policy is not None:
            q = [a + (b - a) * k for a, b in zip(self.q, self.q_goal)]
            flange = flange_at(q)
            if self.floor_z is None or flange.pos[2] >= self.floor_z:
                self._reflexes(q, dt)
                self.q, self.flange = q, flange
        step = self.gripper_speed * dt
        closing_on = self.holding_width and self.width_goal < self.width
        floor = self.holding_width if closing_on else 0.0002
        goal = max(self.width_goal, floor)
        self.width += max(-step, min(step, goal - self.width))

    def _reflexes(self, q: Sequence[float], dt: float) -> None:
        for j, (a, b) in enumerate(zip(self.q, q)):
            speed = abs(b - a) / dt if dt > 0 else 0.0
            self.fastest[j] = max(self.fastest[j], speed)
            if speed > SERVER_SPEED[j] and len(self.reflexes) < 20:
                self.reflexes.append(f"joint {j + 1} at {speed:.2f} rad/s")
            if not SERVER_LOWER[j] <= b <= SERVER_UPPER[j] and len(self.reflexes) < 20:
                self.reflexes.append(f"joint {j + 1} at {b:.3f} rad")

    # -- the FrankaInterface API -----------------------------------------------------
    def get_ee_pose(self):
        with self._lock:
            self.calls += 1
            self._advance()
            return to_wire(self.flange)

    def get_joint_positions(self):
        with self._lock:
            self._advance()
            return list(self.q)

    def get_gripper_position(self):
        with self._lock:
            self._advance()
            return [self.width]

    def start_cartesian_impedance(self, Kx, Kxd):
        with self._lock:
            self._advance()
            self.policy, self.q_goal = "cartesian", list(self.q)
            self.setpoint = self.flange.translated((0.0, 0.0, self.sag_m))

    def start_joint_impedance(self, Kq=None, Kqd=None):
        with self._lock:
            self._advance()
            self.policy, self.q_goal = "joint", list(self.q)

    def terminate_current_policy(self):
        with self._lock:
            self._advance()
            if self.policy is None:
                raise RuntimeError("terminate_current_policy: no controller running")
            self.policy = None

    def update_desired_ee_pose(self, pose):
        with self._lock:
            self._advance()
            if self.policy != "cartesian":
                raise RuntimeError("Tried to update policy with no controller running: "
                                   "start_cartesian_impedance first")
            self.setpoint = from_wire(pose)
            q, solved = solve(PANDA_KINEMATICS, PANDA.tcp_from_body, self.q, flange_to_tcp(
                self.setpoint.translated((0.0, 0.0, -self.sag_m))))
            if solved:                                    # or Polymetis skips the update, silently
                self.q_goal = list(q)

    def update_desired_joint_pos(self, pos):
        with self._lock:
            self._advance()
            if self.policy != "joint":
                raise RuntimeError("Tried to update policy with no controller running: "
                                   "start_joint_impedance first")
            self.q_goal = [float(v) for v in pos]

    def control_gripper(self, close: bool):
        with self._lock:
            self._advance()
            self.width_goal = 0.0 if close else 0.08

    def grasp(self, speed: float, force: float):
        self.control_gripper(True)

    def set_gripper_position(self, width: float):
        with self._lock:
            self._advance()
            self.width_goal = min(0.08, max(0.0, float(width)))

    def close(self):
        pass


#: the station's cameras' colour lenses at 640 x 480, as they reported them (2026-10-27)
MEASURED = {
    "327122079691": Intrinsics(640, 480, 605.103, 604.575, 320.322, 254.783),     # D435, front
    "218622273043": Intrinsics(640, 480, 392.103, 391.646, 321.093, 238.658,      # D405, wrist
                               (-0.05515, 0.06124, -0.00074, 0.00109, -0.02021),
                               "inverse_brown_conrady"),
}


class FakeCamera:
    """A RealSense that delivers a grey picture and a table ``depth_m`` away,
    through the lens the station's camera of that serial has."""

    def __init__(self, serial: str = "327122079691", width: int = 640, height: int = 480,
                 fps: int = 30, intrinsics: Optional[Intrinsics] = None,
                 depth_m: float = 0.6) -> None:
        self.serial, self.width, self.height, self.fps = serial, width, height, fps
        self.intrinsics = intrinsics or MEASURED.get(serial) or MEASURED["327122079691"]
        self.depth_scale, self.depth_m = 0.001, depth_m

    def start(self) -> "FakeCamera":
        return self

    def colour(self):
        import numpy as np

        return np.full((self.height, self.width, 3), 128, dtype=np.uint8)

    def capture(self):
        import numpy as np

        return self.colour(), np.full((self.height, self.width), self.depth_m, dtype=np.float32)

    def close(self) -> None:
        pass


__all__ = ["FakeCamera", "FakeServer", "HOME_JOINTS", "MEASURED", "flange_at"]
