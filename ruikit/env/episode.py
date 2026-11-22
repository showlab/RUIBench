"""One episode, as the tool server holds it.

It owns the world (a RUIBench task on one engine), advances it, runs tool
calls, delivers events and, at the end, scores the physical trajectory.  An
agent never touches this object: it calls tools through ``Episode.call`` (or
the SDK / MCP server built on it), and each tool sees only a
:class:`~ruikit.env.context.ToolContext`.

**Lockstep clock.**  Between two tool calls the simulation is paused.  When a
call arrives, the world first advances by the agent's thinking time for that
gap -- events fire and running motions continue meanwhile -- and then the call
runs.  How thinking is charged is the experiment's choice:

=========  ==============================================================
zero       nothing (diagnostic: separates "thought wrong" from "too slow")
fixed      ``think_seconds`` per model turn (a loop calls ``model_turn``;
           an external client's call after a pause counts as a new turn)
measured   the wall-clock gap since the previous result, or the generation
           time a model proxy reports through ``model_turn(seconds)``
=========  ==============================================================

The world's evolution is then the same as if the simulator had run the whole
time, without leaving Isaac spinning while a model thinks.

On a real robot (``EngineCapabilities.real``) nothing pauses: the world runs on
the wall clock, so ``measured`` is the only clock, and the time thought is
charged by having gone by.
"""

from __future__ import annotations

import itertools
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

from ruibench.core.config import BudgetConfig, RunConfig
from ruibench.events import EventScheduler
from ruibench.executor import Executor
from ruibench.executor.handles import HandleStatus
from ruibench.runner.demonstration import play
from ruibench.runner.episode import (
    SETTLE_AFTER_DONE,
    _sized,
    observed_cameras,
)
from ruibench.scoring import MilestoneTracker
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import get_tool, load_builtin_tools
from ruikit.diagnostics.oracle import Oracle
from ruikit.env.budget import REPLIES_PER_CALL, Budget, EpisodeBudget
from ruikit.env.clock import EpisodeClock
from ruikit.env.ending import EpisodeEnd, Outcome
from ruikit.env.frames import FrameStore
from ruikit.env.handles import (
    BLOCKING_CAP_SECONDS,
    EpisodeMotion,
    Handle,
)
from ruikit.env.log import EpisodeLogging
from ruikit.env.observation import Observation
from ruikit.env.station import Station
from ruikit.env.world import EpisodeWorld
from ruikit.interfaces import Profile, load_profile
from ruikit.interfaces.per_action import PerActionRule
from ruikit.perception.backends import depth_camera_needed, open_perception


#: what a toy's clock is given before each of an agent's calls on a real robot (``_invoke``)
REAL_TICK_SECONDS = 0.05

class Episode(EpisodeEnd, EpisodeWorld, PerActionRule, Observation, EpisodeLogging, EpisodeClock,
              EpisodeBudget, EpisodeMotion):
    """A task, an engine, a profile and a clock -- the server side of one run."""

    def __init__(self, task: str, level: str = "easy", seed: int = 0,
                 profile: Union[str, Profile] = "rui.pose/v1", engine: Union[str, Any] = "mock",
                 clock: str = "zero", think_seconds: float = 0.0, think_per: str = "turn",
                 budget: Optional[Budget] = None, log_dir: Optional[str] = None,
                 record_dir: Optional[str] = None, play_out: bool = False,
                 end_at_success: bool = False, end_at_violation: bool = True,
                 camera_size: Tuple[int, int] = (512, 512), record_every: float = 0.5,
                 record_size: int = 256, image_dir: Optional[str] = None,
                 agent: str = "agent") -> None:
        """``record_dir``: film the episode (a frame every ``record_every``
        simulated seconds, ``record_size`` pixels wide); ``image_dir``: keep
        every picture the agent was shown, by frame id; ``agent``: who is
        playing, for the logs.

        Reaching the goal does not end an episode: the agent's ``task_done`` or
        ``task_give_up`` does, or a limit, and it is judged on the world then.
        An episode that ended the moment it succeeded would tell the agent it
        had: a wrong try still running is a "not yet", and trying candidates one
        by one would beat the task (v2.3).  ``end_at_success``: end it then all
        the same -- for the diagnostics, where a scripted reference plays.

        A rule of the task broken ends it at once (``end_at_violation``): every
        rule is a "never", so the episode is lost then, and playing on would only
        spend the agent's tokens (v2.3.4)."""
        from ruibench.engines import is_real, make_engine
        from ruibench.runner.recorder import FrameRecorder
        from ruibench.tasks import real as real_tasks
        from ruibench.tasks.instance import build as build_simulated

        load_builtin_tools()
        #: a physical robot (``EngineCapabilities.real``): its world keeps its own
        #: time, and its tasks are the ones set up on it (``ruibench.tasks.real``)
        self.real = is_real(engine)
        build = real_tasks.build if self.real else build_simulated
        self.profile = profile if isinstance(profile, Profile) else load_profile(profile)
        # only what can run is offered: a listed tool with no backend behind it
        # cost agents replies on calls that could only fail
        from ruikit.extras.backends import unavailable

        self.unavailable = unavailable(self.profile)
        if self.unavailable:
            self.profile = replace(
                self.profile,
                tools=tuple(t for t in self.profile.tools if t not in self.unavailable),
                code_tools=tuple(t for t in self.profile.code_tools if t not in self.unavailable))
        if clock not in ("zero", "fixed", "measured"):
            raise ValueError(f"clock must be zero, fixed or measured, not {clock!r}")
        if self.real and clock != "measured":
            raise ValueError(f"a real robot's world goes on while the model thinks, as long as "
                             f"it really thinks: the measured clock, not {clock!r}")
        self.clock = clock
        self.think_seconds = float(think_seconds)
        if think_per not in ("turn", "reply"):
            raise ValueError(f"think_per must be turn or reply, not {think_per!r}")
        # "reply": under the fixed clock every reply costs think_seconds -- each
        # tool call or batch is one answer from the model, and the world moves on
        # meanwhile; "turn": only a turn the loop announces (model_turn) does
        self.think_per = think_per
        self.budget = budget or Budget()
        self.play_out = play_out
        self.end_at_success = end_at_success
        self.end_at_violation = end_at_violation
        engine_name = engine if isinstance(engine, str) else getattr(
            getattr(engine, "capabilities", None), "name", type(engine).__name__)
        self.cfg = RunConfig(run_id="ruikit", engine=engine_name,
                             interface=f"ruikit:{self.profile.id}", agent="ruikit",
                             clock={"zero": "zero_latency", "fixed": "controlled",
                                    "measured": "measured"}[clock],
                             clock_latency_seconds=self.think_seconds,
                             camera_width=camera_size[0], camera_height=camera_size[1],
                             record_dir=record_dir, record_every=record_every,
                             record_size=record_size, play_out=play_out)
        self.inst = build(task, level=level, seed=seed, config=self.cfg)
        calls = int(self.inst.metadata.get("max_model_calls", BudgetConfig.max_model_calls))
        limits = {}
        if self.budget.max_model_turns is None:
            limits["max_model_turns"] = calls       # the task's own turn budget, as the runner's
        if self.budget.max_replies is None:
            limits["max_replies"] = calls * REPLIES_PER_CALL
        if self.budget.max_sim_seconds is None:
            limits["max_sim_seconds"] = round(min(calls * self.cfg.t_ref_seconds,
                                                  self.cfg.budget.max_sim_seconds), 1)
        if limits:
            self.budget = replace(self.budget, **limits)
        self._own_engine = isinstance(engine, str)
        self.engine = make_engine(engine) if isinstance(engine, str) else engine
        if hasattr(self.engine, "track_visibility"):
            self.engine.track_visibility = bool(self.inst.metadata.get("needs_visibility"))
        self._recorder = (FrameRecorder(record_dir, self.cfg.record_every, self.cfg.record_size)
                          if record_dir else None)
        if self._recorder is not None:
            self._recorder.start(f"{task}_{level}_s{seed}_{self.profile.id.replace('/', '-')}")
        cameras = observed_cameras(self.inst.scene, self.cfg)
        self._scene_cameras = cameras
        #: the perception backend's settings (``ruikit.perception.backends``)
        self.perception_settings = self.profile.perception_settings()
        renders = bool(getattr(getattr(self.engine, "capabilities", None), "rendering", False))
        from_pixels = depth_camera_needed(self.profile, renders)

        def sized(sc):
            sc = _sized(sc, self.cfg)
            if from_pixels:                 # RGB-D cameras: the backend needs the depth image
                sc = replace(sc, cameras=tuple(replace(c, depth=True) for c in sc.cameras))
            return sc

        self._watched = [play(self.engine, demo, cameras, sized, self._recorder, i)
                         for i, demo in enumerate(self.inst.demonstrations)]
        self.scene = sized(self.inst.scene)
        self.engine.load(self.scene, seed)
        self.engine.reset()
        if hasattr(self.engine, "set_instruction"):
            self.engine.set_instruction(self.inst.instruction, initial=True)
        self.executor = Executor(self.engine, self.scene.arm_ids)
        self.scheduler = EventScheduler(self.inst.events, episode_start=self.engine.sim_time)
        self.tracker = MilestoneTracker(tuple(self.inst.milestones))
        self._feedback = self.inst.feedback_factory() if self.inst.feedback_factory else None
        self._reactor = self.inst.reactor_factory() if self.inst.reactor_factory else None
        self.traj = [self.engine.state()]
        self.tracker.update(self.traj[0])

        self.station = Station(self.scene)
        self.frames = FrameStore()
        self.perception, self.perception_info = open_perception(
            self.profile, self.scene, self.station, seed, renders, real=self.real)
        self.handles: Dict[str, Handle] = {}
        self._handle_ids = itertools.count(1)
        self._generation: Dict[str, int] = {a: 0 for a in self.station.arms}
        self.memory: Dict[str, Any] = {"notes": {}}
        self.custom_tools: Dict[str, Any] = {}
        self.sessions: List[Any] = []
        self.sandbox = None
        self._events: List[dict] = []
        self._events_seen = 0
        #: the events a wait has already ended on (a program's results are not
        #: stamped one by one, so ``_events_seen`` stands still while it runs)
        self._events_waited = 0
        self._delivered = 0
        self._instruction_version = self.traj[0].instruction_version
        self._done_report: Optional[str] = None
        self._given_up: Optional[str] = None
        self.outcome: Optional[Outcome] = None
        self.tool_calls = 0
        self.replies = 0                  # top-level calls and batches: what the agent said
        self.model_turns = 0
        self.usd = 0.0
        self.log: List[dict] = []
        # what nothing shown to the agent may contain -- the task's answer-bearing
        # names, hidden mappings and canary; a hit voids the run
        self.secrets = tuple(s for s in (getattr(self.inst, "secrets", None) or ()) if s)
        self.leaks: List[dict] = []
        self.log_dir = Path(log_dir) if log_dir else None
        self.image_dir = Path(image_dir) if image_dir else None
        self.agent = agent
        self._wall_start = time.time()
        self._started_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        self._score: Optional[dict] = None
        self._scoring: dict = {}
        self._last_return_wall: Optional[float] = None
        self._pending_think = 0.0
        self._waited = 0.0
        self._turn_marked = False
        self._batch: Optional[list] = None
        self._depth = 0
        # the calls a program (or an agent's own tool) makes inside one reply,
        # kept with that reply's log row
        self._inner: Optional[List[dict]] = None

    # ------------------------------------------------------------ agent side
    @property
    def over(self) -> bool:
        return self.outcome is not None

    @property
    def sim_time(self) -> float:
        return self.engine.sim_time

    @property
    def instruction(self) -> str:
        return self.traj[-1].instruction


    def call(self, name: str, args: Optional[Mapping[str, Any]] = None) -> ToolResult:
        """Run one tool call from the agent."""
        return self._top_level(lambda: self._invoke(name, args, from_code=False), name, args)

    def call_batch(self, calls: Sequence[Tuple[str, Mapping[str, Any]]]) -> List[ToolResult]:
        """The calls of one model reply: submitted in order, motions on
        different arms run together, and the reply returns when every blocking
        motion it started has finished."""
        if len(calls) == 1:
            return [self.call(*calls[0])]
        broken = self._sequenced(calls)

        def run():
            if broken:
                # the whole reply is refused: nothing in it runs
                return [ToolResult.fail(ToolError("invalid_args", broken, hint=(
                    "send one action per arm in a reply and the next one after it returns; "
                    "a reply may still hold one action for each arm and any looking")))
                        for _ in calls]
            self._batch = []
            try:
                results = [self._invoke(n, a, from_code=False) for n, a in calls]
                pending = [h for h in self._batch]
            finally:
                self._batch = None
            if pending:
                self._run_until(lambda: all(h.done for h in pending), BLOCKING_CAP_SECONDS)
            return [r.finalize() if hasattr(r, "finalize") else r for r in results]

        results = self._top_level(run, "batch", {"calls": [{"tool": n, "args": dict(a or {})}
                                                            for n, a in calls]})
        return results if isinstance(results, list) else [results] * len(calls)


    def _spec_of(self, name: str):
        try:
            return get_tool(name).spec
        except Exception:
            return None

    # ----------------------------------------------------------- internals
    def _top_level(self, run: Callable, name: str, args):
        if self.over:
            return self._stamp(ToolResult.fail(ToolError(
                "episode_over", f"the episode has ended ({self.outcome.reason})")))
        wall0 = time.time()
        think = self._think_charge(wall0)
        stop = self._check_budget()
        if stop:
            self.finish(stop)
            return self._stamp(ToolResult.fail(ToolError(
                "budget_exhausted", f"budget used up: {stop}")))
        if think > 0 and not self.real:
            # on a real robot the time thought has passed already, on the world's
            # own clock: it is charged by having gone by
            self.advance(think)
        t0 = self.sim_time
        self.tool_calls += 1
        self.replies += 1
        self._inner = []
        # an arm moving into this reply, or sent moving in it: the world after it is pushed
        before = len(self.executor.history)
        moving = any(self.arm_busy(a) is not None for a in self.station.arms)
        try:
            out = run()
        finally:
            inner, self._inner = self._inner, None
        results = out if isinstance(out, list) else [out]
        moved = moving or any(h.status is not HandleStatus.REJECTED
                              for h in self.executor.history[before:])
        self._push(results, moved)
        self._after_call(t0)
        stamped = [self._stamp(r) for r in results]
        names = ([c["tool"] for c in args.get("calls", ())] if name == "batch" else [name])
        for n, r in zip(names, stamped):
            self.scan(n, r)
        self._log(name, args, think, t0, stamped, time.time() - wall0, inner)
        if self.over:
            # the call that ended the episode (task_done, a limit) finished it -- and wrote
            # the logs -- before its own row existed: write them again with it
            self._write_log()
        self._last_return_wall = time.time()
        return stamped if isinstance(out, list) else stamped[0]


    def scan(self, name: str, result: ToolResult) -> None:
        """Look for the task's secrets in a result on its way to the agent --
        a reply, or a call's answer inside a program.  A hit is recorded, and
        fails the run: it saw what it must not have."""
        if not self.secrets or not hasattr(result, "as_json"):
            return
        blob = json.dumps(result.as_json(), ensure_ascii=False, default=str).lower()
        for secret in self.secrets:
            if secret.lower() in blob:
                self.leaks.append({"t": round(self.sim_time, 3), "tool": name, "secret": secret})


    def _invoke(self, name: str, args, from_code: bool) -> ToolResult:
        # on a real robot an operator follows the agent's calls, a program's among them;
        # the looks the episode itself takes after a reply (_push) are none of the agent's
        if not self.real or self._inner is None:
            return self._call_tool(name, args, from_code)
        inner = self._depth > 0
        if self._reactor is not None:
            # a toy's clock has gone on while the agent thought, and nothing ticked it: it has
            # its say before each call, a look or a question too, or an agent asking what was
            # said would never hear (in simulation each reply ticks the world through)
            self.advance(REAL_TICK_SECONDS)
        self.engine.note_call(name, args, inner)
        result = self._call_tool(name, args, from_code)
        self.engine.note_answer(result, inner)
        return result

    def _call_tool(self, name: str, args, from_code: bool) -> ToolResult:
        from ruikit.env.context import ToolContext

        custom = self.custom_tools.get(name)
        if custom is not None:
            self._depth += 1                 # its calls are nested, as a program's are
            try:
                return custom.invoke(ToolContext(self, name, from_code), args)
            finally:
                self._depth -= 1
        allowed = self.profile.callable_from_code(name) if from_code else name in self.profile.tools
        if not allowed:
            offered = self.profile.code_tools if from_code else self.profile.tools
            why = self.unavailable.get(name)
            return ToolResult.fail(ToolError(
                "not_in_profile", f"{name!r} is not available here" + (f": {why}" if why else ""),
                hint=f"available: {', '.join(offered)}"))
        tool = get_tool(name)
        ctx = ToolContext(self, name, from_code)
        wall0 = time.time()
        self._depth += 1
        try:
            result = tool.invoke(ctx, args)
        finally:
            self._depth -= 1
        cost = tool.spec.cost
        if cost.kind == "declared" and cost.seconds > 0:
            self.advance(cost.seconds)
        elif cost.kind == "measured":
            self.advance(time.time() - wall0)
        return result

    def _after_call(self, t0: float) -> None:
        if self.over:
            return
        if (self.end_at_success and not self.play_out and self.sim_time > t0
                and self.inst.scorer.success.evaluate(self.traj)):
            self.finish("success")
            return
        if (self.end_at_violation and not self.play_out and self.sim_time > t0
                and self.broken_rule() is not None):
            self.finish("rule_broken")
            return
        if (self._done_report is not None or self._given_up is not None):
            if self.executor.busy():
                self._run_until(lambda: not self.executor.busy(), BLOCKING_CAP_SECONDS + 5)
            if self._done_report is not None:
                settle = SETTLE_AFTER_DONE
                show = getattr(self._reactor, "finale_seconds", 0.0)
                if show > 0:
                    self._reactor.finale(self.engine, self.engine.state())
                    settle = max(settle, show)
                self.advance(min(settle, max(0.0, self.budget.max_sim_seconds - self.sim_time)))
                self.finish("agent_done")
            else:
                self.finish("gave_up")


    def broken_rule(self) -> Optional[str]:
        """The first of the task's rules broken so far, or None.  Each is a
        "never" -- once broken it stays broken -- so the episode is a failure
        from then on, whatever follows."""
        for name, pred in getattr(self.inst.scorer, "violations", ()) or ():
            if not pred.evaluate(self.traj):
                return name
        return None

    # -- diagnostics ---------------------------------------------------------------
    def oracle(self) -> "Oracle":
        """Privileged access for the scripted references of the diagnostic
        track, which check that a task is solvable, physically and through an
        interface's tools.  Never handed to a tool."""
        return Oracle(self)
