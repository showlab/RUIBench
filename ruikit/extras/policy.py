"""A VLA behind a tool: a policy session that keeps acting at its own rate, or one proposed
chunk."""

from __future__ import annotations

from dataclasses import dataclass

from ruibench.core.types import Pose
from ruibench.interfaces.commands import MoveTo, SetGripper
from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.env.station import quat_from_rpy_deg, rpy_deg_from_quat
from ruikit.extras.backends import POLICIES, get_backend
from ruikit.harness.components.base import Component


class PolicySession:
    """A running VLA: asks its backend for a chunk at ``rate_hz`` from fresh
    frames, and plays each chunk on its arm -- until done, timed out or
    stopped.  Every chunk carries the arm's generation at start; after a stop
    the generation moves on and anything the backend still returns is dropped
    (the late-result hole the benchmark's executor has, closed here)."""

    def __init__(self, ctx, arm: str, instruction: str, backend, max_seconds: float) -> None:
        self.arm = arm
        self.instruction = instruction
        self.backend = backend
        self.deadline = ctx.sim_time + max_seconds
        self.generation = ctx.generation(arm)
        self.executor_generation = ctx.executor_generation(arm)
        self.next_infer = ctx.sim_time
        self.stopped = False
        self.chunks = 0
        backend.reset()

    def tick(self, episode, dt: float) -> None:
        from ruikit.env.context import ToolContext

        if self.stopped or episode._generation[self.arm] != self.generation:
            self.stopped = True
            episode.sessions.remove(self)
            return
        if episode.sim_time >= self.deadline:
            self.close(episode)
            return
        if episode.sim_time + 1e-9 < self.next_infer or episode.arm_busy(self.arm):
            return
        ctx = ToolContext(episode, "policy_run")
        cams = [c for c in self.backend.cameras if c in episode.station.cameras] or \
            sorted(episode.station.cameras)[:1]
        frames = {c: ctx.capture(c).image for c in cams}
        rows = self.backend.infer(self.instruction, frames, ctx.arm_state(self.arm))
        self.next_infer = episode.sim_time + max(1.0 / self.backend.rate_hz, self.backend.latency_s)
        if episode._generation[self.arm] != self.generation or not rows:
            return
        now = ctx.tcp(self.arm)
        pos, rpy = list(now.pos), list(rpy_deg_from_quat(now.quat))
        info = episode.station.arm(self.arm)
        cmds = []
        for row in rows:
            pos = [a + b for a, b in zip(pos, row[:3])]
            rpy = [a + b for a, b in zip(rpy, row[3:6])]
            cmds.append(MoveTo(info.arm_id, Pose(tuple(pos), quat_from_rpy_deg(tuple(rpy)))))
            if len(row) > 6:
                cmds.append(SetGripper(info.arm_id, 1.0 if row[6] >= 0.5 else 0.0))
        try:
            episode.start_motion(self.arm, cmds, f"policy chunk {self.chunks}",
                                 generation=self.executor_generation)
        except ToolError:
            self.close(episode)              # stopped, or the controller refused: done
            return
        self.chunks += 1

    def request_stop(self, episode) -> None:
        self.stopped = True

    def close(self, episode) -> None:
        self.stopped = True
        if self in episode.sessions:
            episode.sessions.remove(self)


@dataclass(frozen=True)
class PolicyRunParams:
    instruction: str = param("what the policy should do, in its own words ('pick up the cup')")
    arm: str = param("which arm", default="right")
    max_seconds: float = param("stop the policy after this long", default=20.0,
                               minimum=0.5, maximum=120.0)
    wait: bool = param("true: return when the policy stops; false: a handle at once",
                       default=True)


@tool(name="policy_run",
      data="arm, chunks, running, elapsed_s, backend",
      family="delegate", params=PolicyRunParams, controls="arm",
      source="model", reads=("cameras", "proprioception"), returns="result_or_handle",
      cost=Cost("physical"), tracks=("open",),
      examples=[{"instruction": "put the teddy bear in the red wagon", "max_seconds": 15}],
      description="""Hand one arm to a fine-tuned vision-language-action policy with a short
      instruction. The policy keeps acting on its own at its own rate from fresh camera frames
      -- also while you think -- until max_seconds pass or robot_stop is called; after a stop
      nothing it computed earlier can move the arm. It knows only what its training taught it;
      judge success from your own look afterwards. Returns when it stops (wait=true) or a
      handle to wait for.""")
def policy_run(ctx, p: PolicyRunParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    ctx.check_free(arm)
    backend = get_backend(POLICIES, "policy", ctx.params.get("backend"))
    session = PolicySession(ctx, arm, p.instruction, backend, p.max_seconds)
    ctx.add_session(session)
    t0 = ctx.sim_time
    if p.wait:
        ctx.run_until(lambda: session.stopped, p.max_seconds + 1.0)
        session.stopped = True
    return ToolResult(data={"arm": arm, "chunks": session.chunks, "running": not p.wait,
                            "elapsed_s": round(ctx.sim_time - t0, 3),
                            "backend": getattr(backend, "name", "?")},
                      text=f"policy on {arm}: {session.chunks} chunk(s)"
                           + (" and still running" if not p.wait else ""))


@dataclass(frozen=True)
class ProposeParams:
    instruction: str = param("what the policy should do")
    arm: str = param("which arm", default="right")


@tool(name="policy_propose",
      data="arm, actions (rows as robot_chunk takes them), backend",
      family="delegate", params=ProposeParams, controls="none",
      source="model", reads=("cameras", "proprioception"), returns="result",
      cost=Cost("declared", 0.1), tracks=("open",),
      examples=[{"instruction": "grasp the red block"}],
      description="""Ask the fine-tuned policy what it would do next, without doing it: returns
      one action chunk (rows of [dx, dy, dz, droll, dpitch, dyaw, gripper], station frame,
      metres and degrees) computed from fresh frames. Execute it with robot_chunk, edit it
      first, or ignore it. It does not move the robot.""")
def policy_propose(ctx, p: ProposeParams) -> ToolResult:
    arm = ctx.station.arm(p.arm).name
    backend = get_backend(POLICIES, "policy", ctx.params.get("backend"))
    cams = [c for c in backend.cameras if c in ctx.station.cameras] or ctx.cameras()[:1]
    frames = {c: ctx.capture(c).image for c in cams}
    rows = backend.infer(p.instruction, frames, ctx.arm_state(arm))
    return ToolResult(data={"arm": arm, "actions": rows, "backend": getattr(backend, "name", "?")},
                      text=f"proposed {len(rows)} step(s)")


COMPONENT = Component(
    "policy", "VLA 策略", "hand the arm to a vision-language-action policy",
    ("policy_run", "policy_propose"),
    unavailable="needs a policy fine-tuned for these robots, outside the benchmark's "
                "no-fine-tuning setting")
