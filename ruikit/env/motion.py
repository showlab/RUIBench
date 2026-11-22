"""What a tool may do with the arms: start, sequence, wait for and stop motions,
and report how they ended."""

from __future__ import annotations

import math
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
)

from ruibench.core.types import Pose
from ruibench.engines.base import (
    ARRIVAL_POS_TOL,
    ARRIVAL_ROT_TOL,
)
from ruibench.executor.handles import HandleStatus
from ruibench.interfaces.commands import MoveJoints, MoveTo, SetGripper
from ruikit.core.result import ToolError, ToolResult
from ruikit.perception.proprioception import holding_something


class DeferredResult(ToolResult):
    """A motion submitted inside a batch: its outcome is known only when the
    whole reply's motions have run (``Episode.call_batch``)."""

    def __init__(self, finalize: Callable[[], ToolResult]) -> None:
        super().__init__(text="submitted with the other calls of this reply")
        self._finalize = finalize

    def finalize(self) -> ToolResult:
        return self._finalize()


class MotionContext:
    """The arms through the shared controller: moves, sequences, the gripper,
    waits and stops, and the motion result."""

    # -- acting ------------------------------------------------------------------
    def check_free(self, arm: str) -> None:
        busy = self._ep.arm_busy(arm)
        if busy is not None:
            raise ToolError("resource_busy", f"arm {arm!r} is still running {busy.id} "
                            f"({busy.label})", hint=f"robot_wait on {busy.id}, or robot_stop it",
                            data={"handle": busy.id})

    def move(self, arm: str, poses: Sequence[Pose], label: str,
             gripper_after: Optional[Dict[int, float]] = None, speed: float = 1.0):
        """Submit poses (and gripper changes after given indices) as one handle."""
        info = self.station.arm(arm)
        self.check_free(info.name)
        cmds: List[Any] = []
        for i, pose in enumerate(poses):
            cmds.append(MoveTo(info.arm_id, pose, speed))
            if gripper_after and i in gripper_after:
                cmds.append(SetGripper(info.arm_id, gripper_after[i]))
        target = poses[-1] if poses else None
        return self._ep.start_motion(info.name, cmds, label, target)

    def move_joints(self, arm: str, joints: Sequence[float], label: str):
        """Submit a joint target as one handle; its target, for the residual,
        is where those joints put the hand."""
        from ruibench.engines.kinematics import forward

        info = self.station.arm(arm)
        self.check_free(info.name)
        model, base, _ = self.station.kinematics(info.name)
        return self._ep.start_motion(info.name, [MoveJoints(info.arm_id, tuple(joints))], label,
                                     forward(model, base, joints))

    def sequence(self, arm: str, steps: Sequence[Any], label: str,
                 target: Optional[Pose] = None):
        """Submit a multi-step motion as one handle: each step is a list of
        ``(kind, value)`` -- ``("move", Pose)``, ``("grip", 0..1)`` -- or a
        number of seconds to hold still."""
        info = self.station.arm(arm)
        self.check_free(info.name)
        out = []
        for step in steps:
            if isinstance(step, (int, float)):
                out.append(float(step))
                continue
            cmds = []
            for kind, value in step:
                if kind == "move":
                    cmds.append(MoveTo(info.arm_id, value))
                elif kind == "grip":
                    cmds.append(SetGripper(info.arm_id, float(value)))
                else:
                    raise ValueError(f"unknown step kind {kind!r}")
            out.append(cmds)
        return self._ep.start_sequence(info.name, out, label, target)

    def add_session(self, session) -> None:
        self._ep.sessions.append(session)

    def run_until(self, done, cap: float) -> bool:
        return self._ep._run_until(done, cap)

    def executor_generation(self, arm: str) -> int:
        """The executor's count of cancels on this arm: a session notes it when
        it starts and submits under it, so the executor itself turns away
        anything decided before a stop."""
        return self._ep.executor.generation(self.station.arm(arm).arm_id)

    def generation(self, arm: str) -> int:
        return self._ep._generation[self.station.arm(arm).name]

    def grip(self, arm: str, value: float, label: str):
        info = self.station.arm(arm)
        self.check_free(info.name)
        return self._ep.start_motion(info.name, [SetGripper(info.arm_id, value)], label)

    def motion_result(self, handle, wait: bool = True,
                      cap: Optional[float] = None) -> ToolResult:
        from ruikit.env.handles import BLOCKING_CAP_SECONDS

        cap = BLOCKING_CAP_SECONDS if cap is None else cap
        if not wait:
            return ToolResult(data={"handle": handle.id, "status": handle.status},
                              text=f"{handle.label}: started as {handle.id}; "
                                   f"robot_wait on it or robot_stop it")
        if self._ep._batch is not None and self._ep._depth <= 1:
            # a reply's own motions run together; a program inside the reply
            # still gets each of its calls' answers when that call is done
            self._ep._batch.append(handle)
            return DeferredResult(lambda: self._report(handle))
        self._ep._run_until(lambda: handle.done, cap)
        return self._report(handle)

    def _report(self, handle) -> ToolResult:
        info = self.station.arm(handle.arm)
        st = self._ep.engine.state().arm(info.arm_id)
        data = {"handle": handle.id, "status": handle.status,
                "tcp": self.station.describe_pose(st.eef_pose),
                "gripper_closed": st.gripper >= 0.5,
                "holding_something": holding_something(st)}
        status = handle.status
        goal = handle.target
        if status == "failed":
            # a skill is several moves, and what stopped is one of them: the residual
            # and short_by_m are to *its* target.  Against the skill's last pose a
            # descent blocked 1.5 cm short read "10 cm short, upward"
            part = next((p for p in handle.parts if p.status is HandleStatus.FAILED), None)
            goal = (self._ep.executor.goal_of(part) if part is not None else None) or goal
        if goal is not None:
            miss = st.eef_pose.distance_to(goal)
            data["residual_cm"] = round(miss * 100, 2)
            data["residual_deg"] = round(math.degrees(st.eef_pose.angle_to(goal)), 1)
        if status == "done":
            short = handle.target is not None and (
                data["residual_cm"] > ARRIVAL_POS_TOL * 100 + 0.01
                or math.radians(data["residual_deg"]) > ARRIVAL_ROT_TOL + 0.002)
            if not short:
                return ToolResult(data=data, text=f"{handle.label}: done")
            # a free hand ends within 2 mm; one that stalled a little short of its
            # target counts as arrived -- say so, it is the only feel of contact
            # there is (seen on CL2 through the step interface)
            aim = self.station.describe_pose(handle.target)["xyz_m"]
            data["short_by_m"] = [round(a - b, 4) for a, b in zip(aim, data["tcp"]["xyz_m"])]
            return ToolResult(data=data, text=f"{handle.label}: done, but it stopped "
                              f"{data['residual_cm'] * 10:.1f} mm / {data['residual_deg']} deg "
                              f"short and held still",
                              hint="a free hand ends within 2 mm and about 1 degree of its "
                                   "target: something is in the way of the fingers or of what "
                                   "they hold (short_by_m points at it), or the arm is at the "
                                   "end of its reach")
        if status == "running":
            return ToolResult(data=data, text=f"{handle.label}: still running after the "
                              f"blocking limit", hint=f"robot_wait on {handle.id}")
        detail = handle.detail or status
        code = "blocked" if "blocked" in detail else "tool_failed"
        if status == "cancelled":
            return ToolResult(data=data, text=f"{handle.label}: stopped")
        hint = "the arm stopped where tcp says; choose a reachable, unobstructed target"
        if code == "blocked" and goal is not None:
            # no contact sensing: what is known is which way the rest of the move went
            aim = self.station.describe_pose(goal)["xyz_m"]
            data["short_by_m"] = [round(a - b, 4) for a, b in zip(aim, data["tcp"]["xyz_m"])]
            hint = ("the arm stopped where tcp says; short_by_m (target minus tcp) points at "
                    "what was in the way of the fingers or of what they hold")
            if data.get("residual_cm", 0.0) < 0.5 and data.get("residual_deg", 0.0) > 3.0:
                # there, but not turned: an FR3 wrist at the end of its travel near
                # yaw +-180 read as "0.1 cm short", as if something were in the way
                hint = ("the hand reached the place but could not turn that far (residual_deg): "
                        "near yaw +-180 degrees the wrist can be at the end of its travel; the "
                        "fingers close along the same line turned 180 degrees")
        return ToolResult(data=data, text=f"{handle.label}: {detail}", error=code, hint=hint)

    def stop(self, arm: Optional[str] = None, handle: Optional[str] = None) -> List[str]:
        return self._ep.stop(arm, handle)

    def handle(self, hid: str):
        h = self._ep.handles.get(hid)
        if h is None:
            raise ToolError("stale_handle", f"no handle {hid!r} in this episode")
        return h

    def wait_handles(self, hids: Iterable[str], mode: str, cap: float) -> List[dict]:
        ids: List[str] = []
        for h in hids:                    # 'running': every action running now (on WK5, agents on
            if h == "running":            # the pose interface guessed the ids of moves just sent)
                ids += [x.id for x in self._ep.handles.values() if not x.done]
            else:
                ids.append(h)
        hs = [self.handle(h) for h in dict.fromkeys(ids)]
        check = (lambda: all(h.done for h in hs)) if mode == "all" else \
            (lambda: any(h.done for h in hs))
        self._ep._run_until(check, cap)
        return [{"handle": h.id, "status": h.status, "label": h.label} for h in hs]

    def wait_event(self, cap: float) -> List[dict]:
        """Run until an event the agent has not been given yet -- with a
        result, or by an earlier wait of the same program.  One that came while
        it was thinking, or while its program was moving the arm, counts and
        ends the wait at once.  (Counted from the call, such an event was
        waited past for 6-15 s.)"""
        start = max(self._ep._events_seen, self._ep._events_waited)
        self._ep._run_until(lambda: len(self._ep._events) > start, cap)
        self._ep._events_waited = len(self._ep._events)
        return self._ep.events(start)
