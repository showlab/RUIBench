"""The execution stack shared by all four interfaces.

It is the *only* thing that talks to the engine during an episode.  Interfaces
emit commands; the executor queues them per arm, drives the engine, and records
what happened to each one.

Two semantics matter enough to be tested directly (Gate B in the TODO):

* **cancellation is final** -- after ``cancel()`` the previous queue must not
  come back to life.  Enforced with a per-arm generation counter: a command
  admitted under generation *g* is dropped if the arm has moved on to *g+1*.
* **two arms run concurrently** -- ``wait_any`` / ``wait_all`` are offered to
  every interface, so a sequential-call interface can still express serial
  bimanual work and a program interface can express parallel work.

A ``Wait`` is a pause at its place in what was submitted: what comes after it
starts once every arm has done what came before it and ``seconds`` more have
passed; on its own, the world runs that long.  A ``Wait`` with an arm pauses
only that arm.  (It used to be a timer from when it was submitted, which the
work around it did not wait for: "move down, wait 0.3 s, open" opened the
moment the hand arrived -- the RUIKit collaborator's WK3 finding, v15.)
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, List, Mapping, Optional, Sequence, Union

from ..core.types import Pose
from ..engines.base import NEAR_POS_TOL, NEAR_ROT_TOL, Engine
from ..interfaces.commands import (
    Cancel, Command, Done, MoveDelta, MoveJoints, MoveTo, SetGripper, Wait,
)
from .handles import ActionHandle, HandleStatus

#: An action that has not finished by this many seconds is marked FAILED so a
#: bad IK target cannot stall an episode forever.
DEFAULT_ACTION_TIMEOUT = 20.0

#: A move that has stopped short -- less than this much motion over this long --
#: is blocked, and fails at once instead of waiting out the timeout.  One that
#: stopped within the engine's near tolerance (``NEAR_POS_TOL``) is done.  On a
#: physics backend a descent onto an object that is in the way otherwise ate
#: twenty seconds of simulated time, which on a timing-sensitive task is not a
#: detail but a different experiment.  Turning toward the target is progress
#: too: a wrist turning in place moves the hand less than a millimetre a second
#: (v1.17: on Isaac the joint interface's turns of CR2's bulb, over a second
#: long, were stopped as blocked mid-turn; so was any move's turn left over a
#: second after its travel).  A hand that only shakes in contact is not.
BLOCKED_SECONDS = 1.0
BLOCKED_TOLERANCE = 0.001        # m
BLOCKED_TURN = math.radians(1.0)

_pauses = itertools.count(1)


class Executor:
    def __init__(self, engine: Engine, arm_ids: Sequence[str],
                 action_timeout: float = DEFAULT_ACTION_TIMEOUT,
                 camera: Optional[str] = None) -> None:
        self.engine = engine
        self.arm_ids = tuple(arm_ids)
        self.action_timeout = action_timeout
        #: the camera a ``MoveDelta(frame="camera")`` is in: the run's first
        self.camera = camera
        #: where a started move is going (a ``MoveDelta``'s, worked out when it starts)
        self._goal: Dict[str, Pose] = {}
        self._queues: Dict[str, List[ActionHandle]] = {a: [] for a in self.arm_ids}
        self._payload: Dict[str, Command] = {}
        self._active: Dict[str, Optional[ActionHandle]] = {a: None for a in self.arm_ids}
        self._generation: Dict[str, int] = {a: 0 for a in self.arm_ids}
        #: a queued pause's handle id -> the ``Wait`` it belongs to (one per arm)
        self._pause_of: Dict[str, str] = {}
        self.history: List[ActionHandle] = []
        #: handle ids count from h1 in each executor, so in each episode
        self._ids = itertools.count(1)
        self._motion: Dict[str, List[tuple]] = {}      # handle id -> [(t, pose)]
        self.done_requested: Optional[Done] = None

    # -- submission --------------------------------------------------------
    def generation(self, arm: str) -> int:
        """How many times ``arm``'s work has been cancelled: a caller that
        notes it when it starts can submit under it, and a cancel in between
        turns what it submits away (``submit``)."""
        return self._generation[arm]

    def submit(self, cmd: Command,
               generation: Union[None, int, Mapping[str, int]] = None) -> Optional[ActionHandle]:
        """Accept one command.  Returns a handle for motion commands, None for
        everything else (service calls, waits, done).

        ``generation``: the arm's generation the caller started under (one
        number, or one per arm).  Work decided before a cancel and submitted
        after it -- a late chunk of actions, a program's sub-call -- is
        rejected, never run (v15; until then it ran as new work).  None: the
        current generation."""
        if isinstance(cmd, Done):
            self.done_requested = cmd
            return None
        if isinstance(cmd, Wait):
            if cmd.arm is not None and cmd.arm not in self._queues:
                return self._reject(cmd, str(cmd.arm), f"unknown arm {cmd.arm!r}; have {list(self.arm_ids)}")
            self._pause(cmd, generation)
            return None
        if isinstance(cmd, Cancel):
            self.cancel(cmd.arm, cmd.handle_id)
            return None

        arm = getattr(cmd, "arm", None)
        if arm not in self._queues:
            return self._reject(cmd, str(arm), f"unknown arm {arm!r}; have {list(self.arm_ids)}")
        started = generation.get(arm) if isinstance(generation, Mapping) else generation
        if started is not None and started != self._generation[arm]:
            return self._reject(cmd, arm, f"stale generation: decided under generation {started}, "
                                          f"and {arm} has been cancelled since "
                                          f"(now {self._generation[arm]})")
        refused = self._unsupported(cmd)
        if refused:
            return self._reject(cmd, arm, refused)
        h = ActionHandle(arm=arm, kind=type(cmd).__name__,
                         submitted_at=self.engine.sim_time,
                         generation=self._generation[arm], id=self._next_id())
        self._queues[arm].append(h)
        self._payload[h.id] = cmd
        self.history.append(h)
        return h

    def _pause(self, cmd: Wait, generation) -> None:
        """A ``Wait``: a pause in every arm's queue, all of them started
        together once each arm has got to it (``_start_pending``) -- or, for
        a ``Wait`` with an arm, in that arm's alone.  Not an action anyone
        holds a handle to, nor one the log lists."""
        mark = f"w{next(_pauses)}"
        for arm in ((cmd.arm,) if cmd.arm else self.arm_ids):
            started = generation.get(arm) if isinstance(generation, Mapping) else generation
            if started is not None and started != self._generation[arm]:
                continue                        # decided before a cancel: dropped with the rest
            h = ActionHandle(arm=arm, kind="Wait", submitted_at=self.engine.sim_time,
                             generation=self._generation[arm], id=self._next_id())
            self._queues[arm].append(h)
            self._payload[h.id] = cmd
            self._pause_of[h.id] = mark

    def _next_id(self) -> str:
        return f"h{next(self._ids)}"

    def submit_all(self, cmds: Sequence[Command],
                   generation: Union[None, int, Mapping[str, int]] = None) -> List[ActionHandle]:
        return [h for h in (self.submit(c, generation) for c in cmds) if h is not None]

    def _reject(self, cmd: Command, arm: str, why: str) -> ActionHandle:
        h = ActionHandle(arm=arm, kind=type(cmd).__name__, submitted_at=self.engine.sim_time,
                         status=HandleStatus.REJECTED, detail=why, id=self._next_id())
        self.history.append(h)
        return h

    def _unsupported(self, cmd: Command) -> str:
        """Why a command cannot be run as it asks -- said, never silently
        dropped (v15: a MoveTo's speed and a MoveDelta's turn and frame were
        ignored, and an interface believed it had turned the wrist)."""
        if isinstance(cmd, MoveTo) and abs(float(cmd.speed) - 1.0) > 1e-9:
            return (f"speed {cmd.speed} is not supported: every move runs at the shared "
                    "controller's speed (MoveTo.speed must be 1.0)")
        if isinstance(cmd, MoveDelta):
            if cmd.frame not in ("world", "eef", "camera"):
                return f"unknown frame {cmd.frame!r}; have world, eef, camera"
            if cmd.frame == "camera" and self._camera_pose() is None:
                return "no camera to move in: the run names none, or the engine cannot say where it is"
        return ""

    # -- cancellation ------------------------------------------------------
    def cancel(self, arm: Optional[str] = None, handle_id: Optional[str] = None) -> List[ActionHandle]:
        """Drop queued work and stop motion.  Cancelled work never resumes."""
        targets = [arm] if arm else list(self.arm_ids)
        cancelled: List[ActionHandle] = []
        now = self.engine.sim_time
        for a in targets:
            if handle_id is not None:
                pool = [h for h in self._queues[a] if h.id == handle_id]
                if self._active[a] and self._active[a].id == handle_id:
                    pool.append(self._active[a])
            else:
                pool = list(self._queues[a]) + ([self._active[a]] if self._active[a] else [])
            for h in pool:
                h.status = HandleStatus.CANCELLED
                h.finished_at = now
                cancelled.append(h)
                self._payload.pop(h.id, None)
            if handle_id is None:
                self._queues[a].clear()
                if self._active[a] is not None:
                    self._active[a] = None
                    self.engine.stop_arm(a)
                self._generation[a] += 1        # nothing from the old generation runs again
            else:
                self._queues[a] = [h for h in self._queues[a] if h.id != handle_id]
                if self._active[a] and self._active[a].id == handle_id:
                    self._active[a] = None
                    self.engine.stop_arm(a)
        return cancelled

    # -- stepping ----------------------------------------------------------
    def tick(self, seconds: float, step: float = 0.05) -> None:
        """Advance the world by ``seconds``, running queued actions as it goes."""
        remaining = seconds
        while remaining > 1e-9:
            dt = min(step, remaining)
            self._start_pending()
            self.engine.advance(dt)
            self._finish_completed()
            remaining -= dt

    def _start_pending(self) -> None:
        now = self.engine.sim_time
        for arm in self.arm_ids:
            if self._active[arm] is not None or not self._queues[arm]:
                continue
            mark = self._pause_of.get(self._queues[arm][0].id)
            if mark is not None and not self._all_at(mark):
                continue                        # a pause the other arms have not got to yet
            h = self._queues[arm].pop(0)
            if h.generation != self._generation[arm] or h.status is HandleStatus.CANCELLED:
                continue                        # stale: a cancel happened after submission
            cmd = self._payload.get(h.id)
            if cmd is None:
                continue
            if not isinstance(cmd, Wait):
                self._apply(arm, cmd, h)
            h.status = HandleStatus.RUNNING
            h.started_at = now
            self._active[arm] = h

    def _all_at(self, mark: str) -> bool:
        """Has every arm that has this pause got to it: nothing running, and
        the pause first in its queue (or started)?  An arm whose pause a
        cancel dropped does not hold the others up."""
        for arm in self.arm_ids:
            active = self._active[arm]
            if active is not None and self._pause_of.get(active.id) == mark:
                continue
            mine = [h for h in self._queues[arm] if self._pause_of.get(h.id) == mark]
            if not mine:
                continue
            if active is not None or self._queues[arm][0] is not mine[0]:
                return False
        return True

    def _apply(self, arm: str, cmd: Command, handle: Optional[ActionHandle] = None) -> None:
        if isinstance(cmd, MoveTo):
            self.engine.set_arm_target(arm, cmd.pose)
        elif isinstance(cmd, MoveJoints):
            # where those joints put the hand: what "short of it" is measured against
            goal = self.engine.set_arm_joints(arm, tuple(cmd.joints))
            if handle is not None:
                self._goal[handle.id] = goal
        elif isinstance(cmd, MoveDelta):
            goal = self._delta_goal(self.engine.state().arm(arm).eef_pose, cmd)
            if handle is not None:
                self._goal[handle.id] = goal
            self.engine.set_arm_target(arm, goal)
        elif isinstance(cmd, SetGripper):
            self.engine.set_gripper(arm, cmd.value)

    def _camera_pose(self) -> Optional[Pose]:
        if not self.camera:
            return None
        try:
            return self.engine.camera_pose(self.camera)
        except (NotImplementedError, StopIteration, KeyError, AttributeError):
            return None

    def _delta_goal(self, cur: Pose, cmd: MoveDelta) -> Pose:
        """Where a relative move goes from ``cur``: its offset and its turn (a
        rotation vector, radians) in the world's axes, the hand's own, or the
        camera's (optical: x right in the picture, y down, z away)."""
        from ..core.geometry import quat_mul, rotate

        if cmd.frame == "eef":
            axes = cur.quat
        elif cmd.frame == "camera":
            axes = self._camera_pose().quat
        else:
            axes = (1.0, 0.0, 0.0, 0.0)
        dpos = rotate(axes, tuple(cmd.dpos))
        rv = rotate(axes, tuple(cmd.drot))
        angle = math.sqrt(sum(v * v for v in rv))
        turn = (1.0, 0.0, 0.0, 0.0) if angle < 1e-12 else (
            math.cos(angle / 2), *(math.sin(angle / 2) * v / angle for v in rv))
        return Pose(tuple(p + d for p, d in zip(cur.pos, dpos)), quat_mul(turn, cur.quat))

    def _finish_completed(self) -> None:
        now = self.engine.sim_time
        for arm in self.arm_ids:
            h = self._active[arm]
            if h is None:
                continue
            if h.id in self._pause_of:
                # a pause is over when its time is: never blocked, never timed out
                if now - _started(h, now) >= self._payload[h.id].seconds - 1e-9:
                    h.status, h.finished_at = HandleStatus.DONE, now
                    self._active[arm] = None
                continue
            done = (self.engine.gripper_settled(arm) if h.kind == "SetGripper"
                    else self.engine.arm_at_target(arm))
            if done:
                h.status, h.finished_at = HandleStatus.DONE, now
                self._active[arm] = None
                self._motion.pop(h.id, None)
            elif h.kind in ("MoveTo", "MoveJoints", "MoveDelta") and self._blocked(arm, h, now):
                self._active[arm] = None
                self._motion.pop(h.id, None)
                if self.engine.arm_at_target(arm, NEAR_POS_TOL, NEAR_ROT_TOL):
                    # at rest just short of the target: as close as it gets
                    h.status, h.finished_at, h.detail = HandleStatus.DONE, now, ""
                else:
                    h.status, h.finished_at = HandleStatus.FAILED, now
                    self.engine.stop_arm(arm)
                    self._drop_what_follows(arm, now)
            elif now - _started(h, now) > self.action_timeout:
                h.status, h.finished_at = HandleStatus.FAILED, now
                h.detail = "action timeout"
                self._active[arm] = None
                self.engine.stop_arm(arm)
                self._drop_what_follows(arm, now)

    def _drop_what_follows(self, arm: str, now: float) -> None:
        """On a real robot what was queued behind a move that failed does not run: the
        descent after an approach that was stopped would drive the hand on into what stopped
        it, after the agent has been told it is blocked.  In simulation it runs, as it has in
        every result there."""
        if not getattr(getattr(self.engine, "capabilities", None), "real", False):
            return
        for h in self._queues[arm]:
            h.status, h.finished_at = HandleStatus.FAILED, now
            self._payload.pop(h.id, None)
        self._queues[arm].clear()

    def _blocked(self, arm: str, h: ActionHandle, now: float) -> bool:
        """Has this move stopped short of its target?"""
        here = self.engine.state().arm(arm).eef_pose
        pos = here.pos
        trail = self._motion.setdefault(h.id, [])
        trail.append((now, here))
        del trail[:-64]
        if now - _started(h, now) < BLOCKED_SECONDS:
            return False
        earlier = [p for (ts, p) in trail if now - ts >= BLOCKED_SECONDS]
        if not earlier:
            return False
        if here.distance_to(earlier[-1]) >= BLOCKED_TOLERANCE:
            return False
        goal = self._goal.get(h.id) or getattr(self._payload.get(h.id), "pose", None)
        if goal is not None and earlier[-1].angle_to(goal) - here.angle_to(goal) >= BLOCKED_TURN:
            return False                              # still turning toward its target
        short = (sum((a - b) ** 2 for a, b in zip(pos, goal.pos)) ** 0.5) if goal else float("nan")
        # the turn too, when it is what is missing: a hand a millimetre from its
        # target but turned short of it (an FR3 wrist at the end of its travel
        # near +-180 degrees) used to read "0.1 cm short", as if something were
        # in the way (v1.1, from RUI-Zero's CL2 runs on Isaac)
        turn = here.angle_to(goal) if goal else 0.0
        gap = (f"{short * 100:.1f} cm and {math.degrees(turn):.0f} degrees" if turn > NEAR_ROT_TOL
               else f"{short * 100:.1f} cm")
        h.detail = f"blocked: stopped {gap} short of the target"
        return True

    # -- waits -------------------------------------------------------------
    def wait_all(self, handles: Sequence[ActionHandle], timeout: float = 30.0) -> None:
        self._wait(handles, timeout, mode="all")

    def wait_any(self, handles: Sequence[ActionHandle], timeout: float = 30.0) -> None:
        self._wait(handles, timeout, mode="any")

    def _wait(self, handles: Sequence[ActionHandle], timeout: float, mode: str) -> None:
        deadline = self.engine.sim_time + timeout
        check = (lambda: all(h.done for h in handles)) if mode == "all" \
            else (lambda: any(h.done for h in handles))
        while not check() and self.engine.sim_time < deadline:
            self.tick(0.05)

    # -- state -------------------------------------------------------------
    def busy(self, arm: Optional[str] = None) -> bool:
        arms = [arm] if arm else list(self.arm_ids)
        return any(self._active[a] is not None or self._queues[a] for a in arms)

    @property
    def pending_wait(self) -> float:
        """How much pausing is still to come, at least: the longest, over the
        arms, of what is left of a started pause and the pauses queued."""
        now, most = self.engine.sim_time, 0.0
        for arm in self.arm_ids:
            left = 0.0
            active = self._active[arm]
            if active is not None and active.id in self._pause_of:
                left += max(0.0, self._payload[active.id].seconds - (now - _started(active, now)))
            left += sum(self._payload[h.id].seconds for h in self._queues[arm]
                        if h.id in self._pause_of and h.id in self._payload)
            most = max(most, left)
        return most

    def goal_of(self, handle: ActionHandle) -> Optional[Pose]:
        """Where a move was going: its command's pose, or the pose a relative
        move worked out to when it started.  None for what is not a move."""
        return self._goal.get(handle.id) or getattr(self._payload.get(handle.id), "pose", None)

    def residual_motion(self) -> Dict[str, float]:
        """Distance each arm still has to travel -- 'leftover old action', one of
        the mechanism-figure channels."""
        st = self.engine.state()
        out = {}
        for arm in self.arm_ids:
            h = self._active[arm]
            goal = None if h is None else (
                self._goal.get(h.id) or getattr(self._payload.get(h.id, None), "pose", None))
            out[arm] = 0.0 if h is None else float(
                st.arm(arm).eef_pose.distance_to(goal or st.arm(arm).eef_pose))
        return out


def _started(h: ActionHandle, now: float) -> float:
    """When an action started.  Not ``h.started_at or now``: an episode starts at
    t = 0.0, which is falsy, and that spelling made every action begun at the
    first instant immune to both the timeout and the blocked check."""
    return h.started_at if h.started_at is not None else now
