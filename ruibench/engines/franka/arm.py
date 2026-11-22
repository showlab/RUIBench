"""The arm and its hand through the NUC's ZeroRPC server, from one control thread.

The NUC runs Polymetis behind ``franka_server/single_arm_server.py``, which
serves the FrankaInterface API (Show-Harness ``core/franka/franka_interface.py``):
the flange's pose (``panda_link8``, base frame, quaternion x-y-z-w) and the
joints, the finger width, Cartesian and joint impedance and their setpoints,
the gripper.

One thread owns the connection: ZeroRPC's event loop belongs to the thread
that made the client, and a call from any other thread fails (``LoopExit``,
as the Show-Harness teleop found).  At ``control_hz`` it reads the arm and
sends the setpoint of the motion in progress; the engine only hands it
motions and reads what it last read.  A move so goes on at its own pace while
the episode renders, perceives or waits on a model, and no reading is older
than a control period.

Motions, all of the flange in the base frame:

* a **move** -- from the setpoint to a goal in a straight line and a slerp,
  minimum-jerk in time, as fast as ``max_speed_m_s`` and ``max_turn_rad_s``
  let its fastest instant be, never quicker than ``min_move_seconds`` nor than
  the joints need (``route``: the engine says how long that is);
* a **joint move** -- joint impedance along a minimum-jerk path in joint
  space, the farthest joint at ``joint_speed_rad_s`` at most; then Cartesian
  impedance again, brought to where those joints put the flange;
* **holding** -- the end of every motion: the goal, the setpoint nudged by
  what still separates the flange from it (``settle_gain``, within
  ``settle_limit_*``), so that a free hand comes to rest within the 2 mm the
  executor calls arrived, and one pressed on something pushes no harder than
  the stiffness times that limit.

A motion's clock is the control thread's: a late answer from the server, or a
controller starting, pauses the motion instead of making its next setpoint jump
ahead.  A setpoint further from the one before than that lets it travel is a
bug, and faults the arm instead of being sent.

The thread watches the arm every period, whatever it was asked (``rig.yaml``):

* the hand further from its setpoint than ``follow_limit_*`` and what its speed
  lets it trail by (``follow_lag_s``), or a joint going on past the server's
  limit less ``WATCH_MARGIN_RAD``: something holds it, or a reflex stopped it.
  The arm holds where it is, nothing pushed, and the executor finds the move
  blocked;
* a joint faster than ``joint_speed_limit_rad_s``: no motion asks for that.  The
  arm holds where it is and faults.

A server error that lost the controller means a reflex ended it: it is started
again where the arm is (Show-Harness's ``FrankaSession.start_impedance``), the
arm holds there, and the motion is over; after ``MAX_RESTARTS`` of those the arm
faults.  A joint controller gone as it starts has moved nothing yet: it is started
again and the path goes on, counted with those.  Any other error, or no answer,
faults the arm, and the engine ends the episode.  A fault leaves the last setpoint
where the arm is.

The operator can pause the arm (``pause``, ``resume``; the key is ``view``'s).  It
holds where it is at the next period, whatever it was asked, and gives way to a
push rather than push back harder and harder.  What was under way goes on afterwards
from where the arm is, and what is asked meanwhile waits, the hand's commands too;
one the hand has been given runs its course, there being nothing to stop it with.

``flight`` keeps what the arm did as it happens: every period of a motion, once
a second at rest, and what the watching did.  The robot's own account of a
reflex goes to the NUC's terminal alone; this is the rest of the story.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from ...core.geometry import quat_conj, quat_mul
from ...core.types import Pose
from .model import SERVER_LOWER, SERVER_UPPER
from .rig import GripperRig, Motion

#: how long the engine waits for the control thread's first reading
CONNECT_SECONDS = 15.0
#: a ZeroRPC call that takes longer than this has failed
CALL_TIMEOUT_SECONDS = 5
#: no reading for this long: the server has stopped answering ...
QUIET_SECONDS = 1.0
#: ... unless it is ending a controller and compiling the next, two calls ZeroRPC's timeout
#: bounds.  0.4 s until 2026-11-03; on 2026-11-04 over 1 s, and the driver took that for silence
STARTING_SECONDS = 2.0 * CALL_TIMEOUT_SECONDS
#: what the server says when no controller consumes setpoints: a joint move or
#: a reflex ended it (Show-Harness ``franka_atomic_controller``)
LOST_CONTROLLER = ("no controller running", "with no controller", "start_joint_impedance",
                   "start_cartesian_impedance")
#: a tick runs a motion's clock on by this many periods at most
LATEST_TICK_PERIODS = 2.0
#: a joint this near the server's limit, and still going, is stopped: past where a
#: line may take it (``model.LINE_*``), short of the server's reflex
WATCH_MARGIN_RAD = 0.05
#: ... going on by more than this in a period, not an encoder's jitter
WATCH_GOING_RAD = 0.002
#: as much of the setpoint's offset as a free hand needs: it rests 3.5 mm off it (middle)
FREE_BIAS_M = 0.006
#: a joint's speed is read over this long
SPEED_WINDOW_SECONDS = 0.1
#: reflexes the driver mends in an arm's life before it stops mending them
MAX_RESTARTS = 2


class Fault(RuntimeError):
    """The arm stopped answering, or failed in a way the driver cannot mend."""


@dataclass(frozen=True)
class Reading:
    """What the arm last told the control thread, and what it was sent."""
    t: float                          # time.monotonic() when the control thread had it
    flange: Pose                      # base frame
    joints: Tuple[float, ...]
    width: float                      # between the fingers, m
    #: where the motion has the flange now: its path, or its goal once there.
    #: The setpoint sent is this, nudged by what holds the flange at it
    aim: Pose
    mode: str                         # "cartesian" or "joint"
    path_done: bool                   # the motion's path has run out: holding


def to_wire(pose: Pose) -> list:
    """A pose as the server takes it: [x, y, z, qx, qy, qz, qw]."""
    w, x, y, z = pose.quat
    return [*map(float, pose.pos), float(x), float(y), float(z), float(w)]


def from_wire(v: Sequence[float]) -> Pose:
    x, y, z, qx, qy, qz, qw = (float(c) for c in v[:7])
    return Pose((x, y, z), _unit((qw, qx, qy, qz)))


def _unit(q) -> Tuple[float, float, float, float]:
    n = math.sqrt(sum(c * c for c in q)) or 1.0
    q = tuple(c / n for c in q)
    return q if q[0] >= 0 else tuple(-c for c in q)


def min_jerk(s: float) -> float:
    """Minimum-jerk progress at time fraction ``s``: still at both ends, its
    fastest instant 1.875 times its mean speed."""
    s = min(1.0, max(0.0, s))
    return s * s * s * (10.0 - 15.0 * s + 6.0 * s * s)


def slerp(q0, q1, f: float):
    dot = sum(a * b for a, b in zip(q0, q1))
    if dot < 0.0:                                     # the shorter way round
        q1, dot = tuple(-c for c in q1), -dot
    if dot > 0.9995:
        return _unit(tuple(a + (b - a) * f for a, b in zip(q0, q1)))
    theta = math.acos(min(1.0, dot))
    s0, s1 = math.sin((1 - f) * theta), math.sin(f * theta)
    return _unit(tuple((a * s0 + b * s1) / math.sin(theta) for a, b in zip(q0, q1)))


def rotvec(q) -> Tuple[float, float, float]:
    """The rotation vector (radians) of a unit quaternion."""
    w, x, y, z = _unit(q)
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (0.0, 0.0, 0.0)
    angle = 2.0 * math.atan2(s, w)
    return (x / s * angle, y / s * angle, z / s * angle)


def from_rotvec(r) -> Tuple[float, float, float, float]:
    a = math.sqrt(sum(c * c for c in r))
    if a < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    k = math.sin(a / 2.0) / a
    return (math.cos(a / 2.0), r[0] * k, r[1] * k, r[2] * k)


def _lost_controller(exc: Exception) -> bool:
    return any(s in str(exc).lower() for s in LOST_CONTROLLER)


def _clip(v, limit: float):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(v) if n <= limit or n < 1e-12 else tuple(c * limit / n for c in v)


@dataclass
class _Path:
    """A motion: of the flange (``start``/``goal`` poses) or of the joints
    (``q0``/``q1``, ending at ``goal``, where those joints put the flange)."""
    seconds: float
    start: Pose
    goal: Pose
    q0: Optional[Tuple[float, ...]] = None
    q1: Optional[Tuple[float, ...]] = None
    elapsed: float = 0.0                 # by the control thread's clock
    #: the setpoint's offset the motion began with (``FrankaArm._fade_bias``)
    bias0: Optional[Tuple[float, float, float]] = None
    #: a joint controller has taken a setpoint of this path
    answered: bool = False

    @property
    def fraction(self) -> float:
        return 1.0 if self.seconds <= 0 else min(1.0, self.elapsed / self.seconds)

    @property
    def joint(self) -> bool:
        return self.q1 is not None


class Flight:
    """What the arm did, in a file of one JSON object a line, kept as it happens:
    a process that is killed leaves it all but the last second.  A full disk stops
    the keeping, never the arm."""

    FLUSH_SECONDS = 1.0

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._flushed = self._at_rest = 0.0

    def _write(self, entry: dict) -> None:
        now = time.time()
        with self._lock:
            if self._file is None:
                return
            try:
                self._file.write(json.dumps({"t": round(now, 3), **entry}) + "\n")
                if now - self._flushed >= self.FLUSH_SECONDS or "note" in entry:
                    self._file.flush()
                    self._flushed = now
            except (OSError, ValueError):
                self._file = None

    def note(self, text: str) -> None:
        self._write({"note": text})

    def tick(self, reading: "Reading", setpoint: Pose) -> None:
        now = time.time()
        if reading.path_done and now - self._at_rest < 1.0:
            return
        self._at_rest = now

        def pose(p: Pose) -> list:
            return [round(v, 5) for v in (*p.pos, *p.quat)]

        self._write({"q": [round(v, 5) for v in reading.joints], "flange": pose(reading.flange),
                     "set": pose(setpoint), "mode": reading.mode, "done": reading.path_done})

    def close(self) -> None:
        with self._lock:
            if self._file is not None:
                self._file.close()
                self._file = None


class FrankaArm:
    """The arm and its hand, driven from a control thread (module docstring).

    ``connect`` makes the server's client in the control thread: a ZeroRPC
    client by default, ``engines.franka.fake.FakeServer`` in tests.  ``flight``
    is where what the arm did is kept (``Flight``)."""

    def __init__(self, host: str, port: int, motion: Motion,
                 connect: Optional[Callable[[], object]] = None,
                 flight: Optional[Path] = None, gripper: GripperRig = GripperRig()) -> None:
        self.host, self.port, self.motion, self.gripper = host, int(port), motion, gripper
        self.flight = Flight(flight) if flight else None
        self._connect = connect or self._zerorpc
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._fault: Optional[str] = None
        self._reading: Optional[Reading] = None
        self._path: Optional[_Path] = None
        self._grip: Optional[Tuple[bool, Optional[float]]] = None
        #: a stop the control thread has yet to act on: hold where the arm is
        self._stop_pending = False
        #: ... and start the next setpoint from there, however far ahead the
        #: last one ran, with nothing pushed
        self._restart_pending = False
        #: the setpoint's offset from the goal that holds the flange at it
        self._bias_p = (0.0, 0.0, 0.0)
        self._bias_r = (0.0, 0.0, 0.0)
        self._mode = "cartesian"
        self._sent_speed = 0.0                # of the setpoint, m/s
        self._pause_wanted = False            # the operator's word ...
        self._paused = False                  # ... and the control thread's doing of it
        self._held: Optional[Pose] = None     # where a pause holds the flange ...
        self._held_q: Tuple[float, ...] = ()  # ... and the joints
        self._restarts = 0                    # the reflexes mended
        self._starting = False                # a controller is starting: no reading till it is up
        self.server = None                    # the control thread's, and only its

    def _zerorpc(self):
        import zerorpc

        client = zerorpc.Client(heartbeat=20, timeout=CALL_TIMEOUT_SECONDS)
        client.connect(f"tcp://{self.host}:{self.port}")
        return client

    # -- lifecycle (engine thread) ---------------------------------------------------
    def start(self) -> Reading:
        """Connect, start Cartesian impedance where the arm is, and hold it."""
        self._thread = threading.Thread(target=self._run, name="franka-control", daemon=True)
        self._thread.start()
        if not self._ready.wait(CONNECT_SECONDS):
            self._stop.set()
            raise Fault(f"no answer from the Franka server at {self.host}:{self.port} within "
                        f"{CONNECT_SECONDS:g} s: is single_arm_server.py running on the NUC?")
        self.check()
        return self.reading()

    def close(self) -> None:
        """Hold where the arm is, and let go of the connection."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5.0)

    # -- commands (engine thread) ----------------------------------------------------
    def move(self, goal: Pose, at_least: float = 0.0) -> float:
        """Send the flange to ``goal`` (base frame), from where the motion has it,
        taking ``at_least`` this long (what its joints need: ``route``); returns
        how long it takes."""
        with self._lock:
            self._check_locked()
            start = self._start_locked()
            seconds = max(self._seconds(start, goal), at_least)
            self._path = _Path(seconds, start, goal)
            self._reading = replace(self._reading, path_done=False)   # busy from now on
            return seconds

    def move_joints(self, q1: Sequence[float], goal: Pose) -> float:
        """Joint impedance to ``q1``, then Cartesian impedance holding ``goal``
        (where ``q1`` puts the flange); returns how long the joint path takes."""
        with self._lock:
            self._check_locked()
            q0 = self._reading.joints
            q1 = tuple(float(v) for v in q1)
            far = max(abs(b - a) for a, b in zip(q0, q1))
            m = self.motion
            seconds = max(m.min_move_seconds, 1.875 * far / m.joint_speed_rad_s)
            self._path = _Path(seconds, self._start_locked(), goal, q0, q1)
            self._reading = replace(self._reading, path_done=False)
            return seconds

    def _start_locked(self) -> Pose:
        """Where a new motion starts: where the last one has the flange -- or,
        after a stop the control thread has not acted on yet, where the flange is."""
        if self._stop_pending:
            self._stop_pending = False                 # the new motion replaces the hold
            return self._reading.flange
        return self._reading.aim

    def hold(self) -> None:
        """Stop where the arm is now: no path, nothing pushed."""
        with self._lock:
            self._stop_pending = self._restart_pending = True

    def grip(self, close: bool, width: Optional[float] = None) -> None:
        """Close the fingers on what is between them, or open them (to ``width``)."""
        with self._lock:
            self._grip = (bool(close), None if width is None else float(width))

    def pause(self) -> None:
        """Stop where the arm is, until ``resume`` (the module docstring)."""
        with self._lock:
            self._pause_wanted = True

    def resume(self) -> None:
        with self._lock:
            self._pause_wanted = False

    def note(self, text: str) -> None:
        """Keep a word with what the arm did (``Flight``)."""
        if self.flight is not None:
            self.flight.note(text)

    # -- readings (engine thread) ----------------------------------------------------
    def reading(self) -> Reading:
        with self._lock:
            self._check_locked()
            return self._reading

    def busy(self) -> bool:
        """A motion's path is still running (or a joint move has not handed back)."""
        with self._lock:
            return self._path is not None and not self._reading.path_done

    def check(self) -> None:
        with self._lock:
            self._check_locked()

    def _check_locked(self) -> None:
        if self._fault:
            raise Fault(self._fault)
        if self._reading is None:
            raise Fault("the arm has not been read yet")
        age = time.monotonic() - self._reading.t
        limit = STARTING_SECONDS if self._starting else QUIET_SECONDS
        if age > limit and self._thread is not None and self._thread.is_alive():
            raise Fault(f"the arm's last reading is {age:.1f} s old: the server stopped answering")

    def _seconds(self, start: Pose, goal: Pose) -> float:
        m = self.motion
        dist = math.dist(start.pos, goal.pos)
        turn = start.angle_to(goal)
        descent = start.pos[2] - goal.pos[2] if goal.pos[2] < m.descent_below_m else 0.0
        return max(m.min_move_seconds, 1.875 * dist / m.max_speed_m_s,
                   1.875 * turn / m.max_turn_rad_s, 1.875 * descent / m.descent_speed_m_s)

    # -- the control thread ----------------------------------------------------------
    def _run(self) -> None:
        try:
            self.server = self._connect()
            flange = from_wire(self.server.get_ee_pose())
            self._restart_cartesian()
            now = time.monotonic()
            with self._lock:
                self._reading = Reading(now, flange, self._joints(), self._width(), flange,
                                        "cartesian", True)
                self._path, self._mode = _Path(0.0, flange, flange), "cartesian"
                self._sent, self._ticked_at = flange, now
                self._speed_from = (self._reading.joints, now)
            self._ready.set()
            period = 1.0 / float(self.motion.control_hz)
            while not self._stop.is_set():
                t0 = time.monotonic()
                self._tick(period)
                self._sleep(max(0.0, period - (time.monotonic() - t0)))
            self._hold_where_it_is()
        except BaseException as exc:                    # the thread must say why it ended
            with self._lock:
                self._fault = self._fault or (str(exc) if isinstance(exc, Fault)
                                              else f"{type(exc).__name__}: {exc}")
            self.note(f"fault: {self._fault}")
            self._ready.set()
            self._hold_where_it_is()
        finally:
            try:
                if self.server is not None and hasattr(self.server, "close"):
                    self.server.close()
            except Exception:
                pass
            if self.flight is not None:
                self.flight.close()

    def _sleep(self, seconds: float) -> None:
        """A pause that lets ZeroRPC's heartbeat run on this thread's loop."""
        try:
            import gevent

            gevent.sleep(seconds)
        except ImportError:                             # the fake server needs no loop
            time.sleep(seconds)

    def _tick(self, period: float) -> None:
        now = time.monotonic()
        flange = from_wire(self.server.get_ee_pose())
        joints, width = self._joints(), self._width()
        self._watch_speed(joints)
        if self._pause_tick(now, flange, joints, width, period):
            return
        strayed = self._strayed(flange, joints)
        if strayed:
            self.note(f"holding where it is: {strayed}")
        with self._lock:
            grip, self._grip = self._grip, None
            if self._stop_pending or strayed:
                # stop where the arm is: the safest setpoint there is
                self._stop_pending, self._restart_pending = False, True
                self._path = _Path(0.0, flange, flange)
            path, restart, self._restart_pending = self._path, self._restart_pending, False
        if restart:
            self._unbias()
        if grip is not None:
            close, width_to = grip
            if close:
                self._grasp()
            elif width_to is None:
                self.server.control_gripper(False)
            else:
                self.server.set_gripper_position(float(width_to))
        path.elapsed += min(now - self._ticked_at, LATEST_TICK_PERIODS * period)
        self._ticked_at = now
        aim, done = self._step(path, flange, period, jump_ok=restart)
        reading = Reading(time.monotonic(), flange, joints, width, aim, self._mode, done)
        with self._lock:
            self._reading, self._starting = reading, False
        if self.flight is not None:
            self.flight.tick(reading, self._sent)

    def _grasp(self) -> None:
        """Close the fingers at the rig's speed and force.  A server that has no such call
        yet closes them its own way, at 0.5 m/s."""
        try:
            self.server.grasp(self.gripper.grasp_speed_m_s, self.gripper.grasp_force_n)
        except Exception as exc:
            if getattr(exc, "name", "") != "NameError":
                raise
            self.note("the robot server has no grasp call: closing at its own speed")
            self.server.control_gripper(True)

    def _pause_tick(self, now: float, flange: Pose, joints: Tuple[float, ...], width: float,
                    period: float) -> bool:
        """The operator's pause, on or just ended; True while the arm does nothing else."""
        with self._lock:
            wanted = self._pause_wanted
        if not wanted and not self._paused:
            return False
        if not wanted:
            self._paused = False
            self.note("the operator goes on")
            self._go_on(flange, joints)
            return False
        if not self._paused:
            self._paused = True
            self._held, self._held_q = flange, joints
            self._unbias()
            self.note("the operator pauses")
        elif (self._mode == "cartesian"
              and math.dist(flange.pos, self._held.pos) > self.motion.follow_limit_m):
            self._held = flange
        if self._mode == "cartesian":
            self._send(self._held, period, jump_ok=True)
        else:
            self.server.update_desired_joint_pos(list(self._held_q))
        self._ticked_at = now
        reading = replace(self._reading, t=time.monotonic(), flange=flange, joints=joints,
                          width=width, aim=self._held)
        with self._lock:
            self._reading, self._starting = reading, False
        if self.flight is not None:
            self.flight.tick(reading, self._held)
        return True

    def _go_on(self, flange: Pose, joints: Tuple[float, ...]) -> None:
        """What a pause interrupted, or what was asked during it, from where the arm
        is: what was left of it, no faster than it was going, smooth from rest."""
        m = self.motion
        with self._lock:
            path = self._path
            left = path.seconds * (1.0 - min_jerk(path.fraction))
            if path.joint:
                far = max(abs(b - a) for a, b in zip(joints, path.q1))
                seconds = max(left, m.min_move_seconds, 1.875 * far / m.joint_speed_rad_s)
                self._path = _Path(seconds, flange, path.goal, joints, path.q1)
            else:
                self._path = _Path(max(left, self._seconds(flange, path.goal)), flange, path.goal)
            self._restart_pending = True
            self._reading = replace(self._reading, path_done=False)

    def _watch_speed(self, joints: Tuple[float, ...]) -> None:
        """No motion asks a joint for this: stop the arm (``_run`` holds it)."""
        then, at = self._speed_from
        now = time.monotonic()
        if now - at < SPEED_WINDOW_SECONDS:
            return
        self._speed_from = (joints, now)
        speeds = [abs(b - a) / (now - at) for a, b in zip(then, joints)]
        fastest = max(speeds)
        if fastest > self.motion.joint_speed_limit_rad_s:
            raise Fault(f"joint {speeds.index(fastest) + 1} ran at {fastest:.2f} rad/s: no move "
                        f"asks for more than {self.motion.joint_speed_rad_s:g}")

    def _strayed(self, flange: Pose, joints: Tuple[float, ...]) -> str:
        """Why the arm should stop where it is, if it should (the module docstring)."""
        m = self.motion
        for j, (a, b, lo, hi) in enumerate(zip(self._reading.joints, joints, SERVER_LOWER,
                                               SERVER_UPPER)):
            if ((b < lo + WATCH_MARGIN_RAD and b < a - WATCH_GOING_RAD)
                    or (b > hi - WATCH_MARGIN_RAD and b > a + WATCH_GOING_RAD)):
                return f"joint {j + 1} at {b:.3f} rad is going past its limit"
        if self._mode != "cartesian":
            return ""
        off, turned = math.dist(flange.pos, self._sent.pos), flange.angle_to(self._sent)
        if (off > m.follow_limit_m + m.follow_lag_s * self._sent_speed
                or turned > m.follow_limit_rad):
            return (f"the hand is {off * 1000:.0f} mm and {math.degrees(turned):.1f} degrees "
                    "from its setpoint")
        return ""

    def _step(self, path: _Path, flange: Pose, period: float,
              jump_ok: bool = False) -> Tuple[Pose, bool]:
        """Send this period's setpoint; returns where the motion has the flange
        now and whether its path is done."""
        m = self.motion
        f = path.fraction
        if path.joint and f < 1.0:
            if self._mode != "joint":
                self._switch("joint")
                return flange, False
            s = min_jerk(f)
            try:
                self.server.update_desired_joint_pos([a + (b - a) * s
                                                      for a, b in zip(path.q0, path.q1)])
                path.answered = True
            except Exception as exc:
                if path.answered or not _lost_controller(exc):
                    raise
                self._joint_controller_gone(exc, flange)
            return flange, False
        if self._mode != "cartesian":
            # the joints are there (or their path was cut short): hold the flange
            # where they put it, with Cartesian impedance again, from where it is
            # once that controller is up
            self._switch("cartesian")
            self._unbias()
            flange = from_wire(self.server.get_ee_pose())
            with self._lock:
                if self._path is path:
                    self._path = _Path(self._seconds(flange, path.goal), flange, path.goal)
            self._sent = flange
            self._send(flange, period, jump_ok=True)
            return flange, False
        if f < 1.0:
            s = min_jerk(f)
            aim = Pose(tuple(a + (b - a) * s for a, b in zip(path.start.pos, path.goal.pos)),
                       slerp(path.start.quat, path.goal.quat, s))
            self._fade_bias(path, s)
        else:
            # holding: close what is left between the flange and the goal
            gain = m.settle_gain * period
            err_p = tuple(g - c for g, c in zip(path.goal.pos, flange.pos))
            err_r = rotvec(quat_mul(path.goal.quat, quat_conj(flange.quat)))
            self._bias_p = _clip(tuple(b + gain * e for b, e in zip(self._bias_p, err_p)),
                                 m.settle_limit_m)
            self._bias_r = _clip(tuple(b + gain * e for b, e in zip(self._bias_r, err_r)),
                                 m.settle_limit_rad)
            aim = path.goal
        setpoint = Pose(tuple(p + b for p, b in zip(aim.pos, self._bias_p)),
                        _unit(quat_mul(from_rotvec(self._bias_r), aim.quat)))
        self._send(setpoint, period, jump_ok=jump_ok)
        return aim, f >= 1.0

    def _fade_bias(self, path: _Path, s: float) -> None:
        """What pressing on something built up of the setpoint's offset goes over the next
        move.  Kept, it flew that move 1 cm under its line, and a hand coming off an edge fell
        by that much; what holds a free hand up stays (``FREE_BIAS_M``)."""
        if path.bias0 is None:
            path.bias0 = self._bias_p
        keep = _clip(path.bias0, FREE_BIAS_M)
        self._bias_p = tuple(k + (b - k) * (1.0 - s) for k, b in zip(keep, path.bias0))

    def _unbias(self) -> None:
        self._bias_p, self._bias_r = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)

    def _send(self, setpoint: Pose, period: float, jump_ok: bool = False) -> None:
        m = self.motion
        # the most a motion's clock lets it travel from one tick to the next at
        # full speed, with room for the settling's nudges
        step = math.dist(setpoint.pos, self._sent.pos)
        turn = setpoint.angle_to(self._sent)
        most = (LATEST_TICK_PERIODS + 1.0) * period
        if not jump_ok and (step > most * m.max_speed_m_s + 0.002
                            or turn > most * m.max_turn_rad_s + 0.01):
            raise Fault(f"refused a setpoint {step * 1000:.1f} mm / {math.degrees(turn):.1f} deg "
                        "from the last one")
        try:
            self.server.update_desired_ee_pose(to_wire(setpoint))
        except Exception as exc:
            if not _lost_controller(exc):
                raise
            self._controller_lost(exc)
            return
        self._sent, self._sent_speed = setpoint, 0.0 if jump_ok else step / period

    def _controller_lost(self, exc: Exception) -> None:
        """A reflex ended the controller: start it again where the arm is and hold
        there.  The motion is over, what it had the arm do is not to be caught up on."""
        self._count_lost(exc)
        self._restart_cartesian()
        flange = from_wire(self.server.get_ee_pose())
        with self._lock:
            self._path = _Path(0.0, flange, flange)
        self._unbias()
        self.server.update_desired_ee_pose(to_wire(flange))
        self._sent, self._sent_speed = flange, 0.0

    def _joint_controller_gone(self, exc: Exception, flange: Pose) -> None:
        """Gone before it took a setpoint, so nothing has moved on it (Polymetis ends a
        controller at once when it has not heard from the robot for 1 s): the path goes on
        from where the joints are, under a controller started again."""
        self._count_lost(exc)
        self._switch("joint")
        self._go_on(flange, self._joints())

    def _count_lost(self, exc: Exception) -> None:
        self._restarts += 1
        self.note(f"the controller was lost ({self._restarts}): {exc}")
        if self._restarts > MAX_RESTARTS:
            raise Fault(f"the controller was lost {self._restarts} times, last: {exc}")

    def _switch(self, mode: str) -> None:
        began = time.monotonic()
        if mode == "joint":
            self._end_controller()
            self.server.start_joint_impedance(None, None)
        else:
            self._restart_cartesian()
        self._mode = mode
        self.note(f"the {mode} controller is up after {time.monotonic() - began:.2f} s")

    def _restart_cartesian(self) -> None:
        m = self.motion
        self._end_controller()
        self.server.start_cartesian_impedance(list(m.stiffness), list(m.damping))
        self._mode = "cartesian"

    def _end_controller(self) -> None:
        """The first half of starting one: no reading comes until it is up."""
        with self._lock:
            self._starting = True
        try:
            self.server.terminate_current_policy()
        except Exception:                               # nothing running is fine
            pass

    def _hold_where_it_is(self) -> None:
        """Leaving: the setpoint where the arm is, so nothing moves on."""
        try:
            if self._mode == "cartesian":
                self.server.update_desired_ee_pose(self.server.get_ee_pose())
            else:
                self.server.update_desired_joint_pos(list(self.server.get_joint_positions()))
        except Exception:
            pass

    def _joints(self) -> Tuple[float, ...]:
        return tuple(float(v) for v in self.server.get_joint_positions())

    def _width(self) -> float:
        return float(list(self.server.get_gripper_position())[0])


__all__ = ["CONNECT_SECONDS", "Fault", "Flight", "FrankaArm", "MAX_RESTARTS", "Reading",
           "from_wire", "min_jerk", "slerp", "to_wire"]
