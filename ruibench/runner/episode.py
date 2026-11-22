"""The episode loop.

Implements the ordering fixed by the design (details §5.3):

1. load the frozen scene and initial state from the seed; the hidden scoring
   predicates and the event script go to the evaluation process only;
2. initialise executor, clock and event queue, capture the first frame;
3. issue the model request carrying the observation timestamp;
4. advance physics, the current action, local control and events on the clock,
   recording when each message became available;
5. hand the result to the executor, recording accepted / rejected / queued,
   the handle, and the instruction version in force;
6. keep evaluating process predicates -- recovery is allowed, but a violation
   that already happened stays on the record;
7. stop on strict success, unrecoverable failure, timeout or budget;
8. store the trajectory, cost, failure reason and scorer output.

The one invariant worth stating out loud: **the world advances while the model
thinks**, and how much it advances is the clock's business alone.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field, replace
from typing import List, Mapping, Optional, Sequence, Tuple

from ..core.config import PLAY_OUT_SECONDS, RunConfig
from ..core.errors import ConfigError
from ..core.types import CameraFrame, Observation, WorldState
from ..engines.base import Engine
from ..events import EventScheduler
from ..executor import Executor
from ..executor.handles import HandleStatus
from ..interfaces.base import Interface, ModelTurn, TurnContext
from ..interfaces.commands import Done
from ..policies.base import Agent, ScriptedPolicy
from ..scoring import EpisodeScore, MilestoneTracker
from ..tasks.instance import TaskInstance
from . import heartbeat
from . import snapshot
from .demonstration import play
from .log import EpisodeLog
from .recorder import FrameRecorder

#: how often the world is snapshotted into the trajectory.  Temporal predicates
#: ("stable for 3 s", "never touched X") are evaluated on these snapshots, so it
#: bounds their time resolution.
SNAPSHOT_DT = 0.1

#: cap on how long one turn's actions may run before the model is asked again
MAX_TURN_SECONDS = 25.0

#: everything an interface under test finds in ``TurnContext.extras`` (v15):
#: the simulator's ground truth, the level and its knob, and the world states
#: behind a demonstration's frames go only to the harness's own test double
PUBLIC_EXTRAS = ("executor_busy", "residual_motion", "feedback", "demonstrations", "refused")

#: After the agent declares it is finished, the world keeps running for this
#: long before scoring.  Success is judged from the physical trajectory, never
#: from the declaration (v0 §2.1), and terminal criteria such as "at rest for
#: 2 s" need time to become true -- or to be falsified by a slow topple.
SETTLE_AFTER_DONE = 3.0


def observed_cameras(scene, cfg: RunConfig) -> Tuple[str, ...]:
    """The scene's cameras the run shows: those it names, and -- when it asks
    for "wrist" -- every wrist camera, so a two-arm scene shows both hands."""
    names = tuple(c.name for c in scene.cameras
                  if c.name in cfg.cameras or (c.kind == "wrist" and "wrist" in cfg.cameras))
    return names or tuple(cfg.cameras)


def _sized(scene, cfg: RunConfig):
    """``scene`` with every camera at the run's resolution.

    Resolution is a property of the run, not of the task: the same task is shown
    to every model at the size the run chose, and the log's config says which.
    The one exception is a robot whose fixed camera is a real sensor with its
    own picture (``RobotModel.camera_size``: the ARX station's head camera is
    RoboDojo's 640 x 480 one); the log's scene says so, camera by camera.
    """
    from ..engines.robots import model_of

    fixed = model_of(scene.robots[0]).camera_size if scene.robots else None
    cameras = tuple(
        replace(c, width=fixed[0], height=fixed[1]) if fixed and c.kind == "external"
        and c.pose is None else replace(c, width=cfg.camera_width, height=cfg.camera_height)
        for c in scene.cameras)
    return replace(scene, cameras=cameras)


@dataclass
class EpisodeResult:
    score: EpisodeScore
    log: EpisodeLog
    trajectory: List[WorldState] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.score.success


class EpisodeRunner:
    def __init__(self, engine: Engine, interface: Interface, agent: Agent,
                 clock, config: RunConfig) -> None:
        self._feedback = None
        self._reactor = None
        self._recorder: Optional[FrameRecorder] = None
        self._cameras: Tuple[str, ...] = tuple(config.cameras)
        self._delivered = 0
        self._beat_task, self._beat_seed = "", None
        self.engine = engine
        self.interface = interface
        self.agent = agent
        self.clock = clock
        self.config = config

    # ---------------------------------------------------------------- run
    def run(self, inst: TaskInstance, condition: str = "main") -> EpisodeResult:
        cfg = self.config
        # A task may ask for a larger call budget: a six-object sorting task
        # legitimately needs more turns than a single pick-and-place, and the
        # cost column is where that shows up, not the success rate.
        max_calls = int(inst.metadata.get("max_model_calls", cfg.budget.max_model_calls))
        log = EpisodeLog(
            run_id=cfg.run_id, task_id=inst.task_id, level=inst.level, seed=inst.seed,
            interface=self.interface.id, agent=getattr(self.agent, "name", "agent"),
            engine=type(self.engine).__name__, clock=self.clock.describe(),
            config=cfg.as_dict(), engine_setup=self.engine.describe(), condition=condition,
            instruction=inst.instruction, hint=inst.hint, scene=inst.scene.summary(),
        )
        log.stamp("start")

        # -- 0. coverage gap: the interface cannot express this task at all ---
        gap = self.interface.coverage_gap(inst.spec)
        if gap:
            score = inst.scorer.score([], {"coverage_gap": gap})
            log.score = score.as_dict()
            log.stamp("end")
            return EpisodeResult(score, log, [])

        # -- 1-2. load, reset, wire up ---------------------------------------
        if hasattr(self.engine, "track_visibility"):
            self.engine.track_visibility = bool(inst.metadata.get("needs_visibility"))
        self._recorder = (FrameRecorder(cfg.record_dir, cfg.record_every, cfg.record_size)
                          if cfg.record_dir else None)
        if self._recorder is not None:
            self._recorder.start(f"{inst.task_id}_{inst.level}_s{inst.seed}_{self.interface.id}"
                                 f"_{condition}_{log.agent}")
        self._cameras = observed_cameras(inst.scene, cfg)
        # the demonstrations come first, on the same engine and cameras
        watched = [play(self.engine, demo, self._cameras, lambda sc: _sized(sc, cfg),
                        self._recorder, i) for i, demo in enumerate(inst.demonstrations)]
        log.demonstrations = [w.as_log() for w in watched]
        self.engine.load(_sized(inst.scene, cfg), inst.seed)
        self.engine.reset()
        if hasattr(self.engine, "set_instruction"):
            self.engine.set_instruction(inst.instruction, initial=True)
        looking = cfg.cameras[0] if cfg.cameras else "agentview"
        executor = Executor(self.engine, inst.scene.arm_ids, camera=looking)
        # what an interface under test is told, and what only the harness's
        # own test double (I0, for the scripted agents) is: the simulator's
        # ground truth, the level and its knob, and the world states behind a
        # demonstration's frames all carry answers
        under_test = getattr(self.interface, "is_benchmark_interface", True)
        films = tuple(w.public() for w in watched) if under_test else tuple(watched)
        scheduler = EventScheduler(inst.events, episode_start=self.engine.sim_time)
        tracker = MilestoneTracker(tuple(inst.milestones))
        self._feedback = inst.feedback_factory() if inst.feedback_factory else None
        self._reactor = inst.reactor_factory() if inst.reactor_factory else None
        self._beat_task, self._beat_seed = f"{inst.task_id} {inst.level}", inst.seed
        self._delivered = 0
        self.agent.reset()

        traj: List[WorldState] = [self.engine.state()]
        tracker.update(traj[0])
        turns: List[ModelTurn] = []
        parse_errors = 0
        #: what the executor refused of the last turn's commands
        refused: tuple = ()
        wall_start = time.time()
        stop_reason = ""
        succeeded_at: Optional[float] = None       # ``RunConfig.play_out``

        # -- 3-7. the loop ----------------------------------------------------
        while True:
            if len(turns) >= max_calls:
                stop_reason = "budget_calls"
                break
            if self.engine.sim_time > cfg.budget.max_sim_seconds:
                stop_reason = "timeout"
                break
            if time.time() - wall_start > cfg.budget.max_wall_seconds:
                stop_reason = "wall_timeout"
                break
            if sum(t.usd for t in turns) > cfg.budget.max_usd:
                stop_reason = "budget_usd"
                break

            world = self.engine.state()
            obs = self._observe(world, public=under_test)
            # Deliver any bits that came due since the last turn.  One shared
            # path for every interface: the program interface receives exactly
            # what the step-by-step ones do.
            said = self._said()
            if said or self._feedback is not None:
                pending = said[self._delivered:]
                self._delivered = len(said)
                obs = replace(obs, feedback=tuple(m.as_text() for m in said))
                for msg in pending:
                    self.interface.on_feedback(None, msg.as_text())
            ctx = TurnContext(
                observation=obs, task_id=inst.task_id, instruction=world.instruction,
                hint=inst.hint,
                arm_ids=inst.scene.arm_ids, turn_index=len(turns),
                sim_time=self.engine.sim_time,
                budget_left={"calls": max_calls - len(turns),
                             "usd": cfg.budget.max_usd - sum(t.usd for t in turns),
                             "sim_seconds": cfg.budget.max_sim_seconds - self.engine.sim_time},
                history=turns,
                extras={**({} if under_test else
                           {"world": world, "level": inst.level,
                            "knob": inst.metadata.get("knob_value")}),
                        # what a program interface can query about its own
                        # in-flight work; a step-by-step interface learns the
                        # same thing from the picture being out of date
                        "executor_busy": executor.busy(),
                        "residual_motion": executor.residual_motion(),
                        "feedback": tuple(self._said()),
                        # what the executor would not run of the last turn's
                        # commands, and why: ({"kind", "arm", "why"}, ...)
                        "refused": refused,
                        # watched before the episode began: frames and captions
                        "demonstrations": films},
            )

            refused = ()                        # told once
            turn = self.interface.observe(ctx)
            turn.requested_at = self.engine.sim_time
            turn.observed_at = obs.sim_time

            # -- 4. the world runs while the model thinks --------------------
            t0 = time.time()
            heartbeat.beat(heartbeat.AGENT, task=inst.task_id, seed=inst.seed, turn=len(turns))
            raw = self.agent.act(ctx, turn)
            heartbeat.beat(heartbeat.SIM, task=inst.task_id, seed=inst.seed, turn=len(turns))
            wall = time.time() - t0 + getattr(self.agent, "latency_seconds", 0.0)
            turn.latency_seconds = wall
            turn.usd = getattr(self.agent, "usd_per_call", 0.0)
            self._advance(self.clock.sim_seconds_for(wall, len(turns)),
                          executor, scheduler, tracker, traj, len(turns))
            turn.returned_at = self.engine.sim_time
            turn.raw_response = raw
            turns.append(turn)

            # -- 5. hand the result to the executor --------------------------
            parsed = self.interface.parse(raw, ctx)
            if parsed.coverage_gap:
                score = inst.scorer.score(traj, {"coverage_gap": parsed.coverage_gap})
                log.score = score.as_dict()
                log.stamp("end")
                return EpisodeResult(score, log, traj)
            if parsed.parse_error:
                parse_errors += 1
                self._advance(SNAPSHOT_DT, executor, scheduler, tracker, traj, len(turns))
                continue

            handles = executor.submit_all(parsed.commands)
            # What the executor will not run as asked -- an arm the scene has
            # not, a speed or a frame the shared controller has not
            # (``Executor.submit``) -- is not run as something else, and is
            # not kept quiet either.  The harness's own scripted agents write
            # their commands themselves: one refused is a broken script, and
            # it stops the run (a cloth reference carried at half speed, every
            # carry was refused, and the hands let go where they had pinched
            # for a reason only the log knew, 10/5).  Any other agent finds
            # it in its next turn's ``extras["refused"]``.
            refused = tuple({"kind": h.kind, "arm": h.arm, "why": h.detail}
                            for h in handles if h.status is HandleStatus.REJECTED)
            if refused and isinstance(self.agent, ScriptedPolicy):
                raise ConfigError(
                    f"{inst.task_id} {inst.level} seed {inst.seed}: the scripted agent "
                    f"({type(self.agent).__name__}) asked for what the executor does not run -- "
                    + "; ".join(f"{r['kind']} on {r['arm']}: {r['why']}" for r in refused))

            # -- run the submitted actions -----------------------------------
            # Submitted work runs to completion whatever the interface's re-entry
            # style.  An interface that wants to get back in sooner says so by
            # submitting a shorter batch -- which is a property of the action
            # language.  Re-entering on a fixed timer instead would charge a
            # non-blocking interface extra model calls for a harness constant,
            # and that cost would then show up in the results as if it were real.
            budget = min(MAX_TURN_SECONDS,
                         max(0.0, cfg.budget.max_sim_seconds - self.engine.sim_time))
            elapsed = 0.0
            if handles or executor.pending_wait > 0:
                while elapsed < budget and (executor.busy() or executor.pending_wait > 0):
                    self._advance(SNAPSHOT_DT, executor, scheduler, tracker, traj, len(turns))
                    elapsed += SNAPSHOT_DT
            else:
                # the agent said nothing actionable; let a little time pass so the
                # world can move on rather than spinning on an identical state
                self._advance(min(SNAPSHOT_DT * 5, budget), executor, scheduler,
                              tracker, traj, len(turns))

            # -- 7. stop conditions ------------------------------------------
            if inst.scorer.success.evaluate(traj):
                if not cfg.play_out:
                    stop_reason = "success"
                    break
                if succeeded_at is None:
                    succeeded_at = self.engine.sim_time
                if self.engine.sim_time - succeeded_at > PLAY_OUT_SECONDS:
                    stop_reason = "success"
                    break
            if (executor.done_requested is not None and not executor.busy()
                    and executor.pending_wait <= 0):
                settle = min(SETTLE_AFTER_DONE,
                             max(0.0, cfg.budget.max_sim_seconds - self.engine.sim_time))
                show = getattr(self._reactor, "finale_seconds", 0.0)
                if show > 0:
                    # the agent's time is over; the show gets all of its own
                    self._reactor.finale(self.engine, self.engine.state())
                    settle = max(settle, show)
                self._advance(settle, executor, scheduler, tracker, traj, len(turns))
                stop_reason = "agent_done"
                break

        # -- 8. score and log --------------------------------------------------
        if self._recorder is not None:
            self._recorder.final(self.engine, self._cameras)
        metrics = self._metrics(turns, traj, scheduler, executor, wall_start)
        scoring = {
            "timed_out": stop_reason in ("timeout", "wall_timeout"),
            "budget_exhausted": stop_reason in ("budget_calls", "budget_usd"),
            "parse_errors": parse_errors,
            # the physics audit's motion checks need a physics engine
            "dynamics": bool(getattr(self.engine.capabilities, "dynamics", True)),
        }
        score = inst.scorer.score(traj, {**scoring, "metrics": metrics})

        log.turns = [self._turn_dict(t) for t in turns]
        log.events = [r.as_dict() for r in sorted(
            list(scheduler.records) + list(getattr(self._reactor, "records", ())),
            key=lambda r: r.sim_time)]
        log.handles = [h.as_dict() for h in executor.history]
        log.feedback = [m.as_dict() for m in self._said()]
        log.milestones_hit = dict(tracker.hit_at)
        log.trajectory = snapshot.dump_all(traj, inst.scene)
        log.score = score.as_dict()
        log.scoring = scoring
        log.metrics = metrics
        log.task_metrics = {name: fn(traj) for name, fn in inst.extra_metrics.items()}
        log.frames = list(self._recorder.records) if self._recorder is not None else []
        log.stamp("end")
        return EpisodeResult(score, log, traj)

    # ------------------------------------------------------------- helpers
    def _observe(self, world: WorldState, public: bool = True) -> Observation:
        frames = {name: self.engine.render(name) for name in self._cameras}
        if self._recorder is not None:
            self._recorder.observation(frames)
        # an interface under test feels whether its fingers hold something,
        # never what: a held body's name can be the answer (DS1's "present2")
        return Observation(
            sim_time=world.sim_time, frames=frames,
            proprioception={a: world.arm(a).public() if public else world.arm(a)
                            for a in world.arm_ids},
            instruction=world.instruction, instruction_version=world.instruction_version,
        )

    def _advance(self, seconds: float, executor: Executor, scheduler: EventScheduler,
                 tracker: MilestoneTracker, traj: List[WorldState], model_calls: int) -> None:
        """Advance the world in snapshot-sized slices, ticking events and
        milestones at each one so their ordering in time is auditable."""
        remaining = max(0.0, seconds)
        while remaining > 1e-9:
            dt = min(SNAPSHOT_DT, remaining)
            heartbeat.beat(heartbeat.SIM, task=self._beat_task, seed=self._beat_seed,
                           turn=model_calls, sim_time=round(self.engine.sim_time, 2))
            executor.tick(dt)
            world = self.engine.state()
            tracker.update(world)
            scheduler.tick(self.engine, world, tracker.hit, model_calls)
            if self._reactor is not None:
                self._reactor.tick(self.engine, world)
            if self._feedback is not None:
                self._feedback.tick(world)
            traj.append(self.engine.state())
            if self._recorder is not None:
                self._recorder.tick(self.engine, self._cameras)
            remaining -= dt

    def _said(self) -> list:
        """Every message so far, from the feedback channel and the toys, in
        the order they were said: one path to every interface."""
        out = list(getattr(self._feedback, "messages", ()))
        out += list(getattr(self._reactor, "messages", ()))
        return sorted(out, key=lambda m: m.sim_time)

    @staticmethod
    def _turn_dict(t: ModelTurn) -> dict:
        return {"turn": t.turn_index, "prompt_chars": len(t.prompt),
                "raw_response": t.raw_response, "requested_at": t.requested_at,
                "observed_at": t.observed_at, "returned_at": t.returned_at,
                "observation_age": t.observation_age,
                "latency_seconds": t.latency_seconds, "usd": t.usd}

    def _metrics(self, turns, traj, scheduler, executor, wall_start) -> dict:
        ages = [t.observation_age for t in turns]
        response = self._response_times(scheduler, executor)
        return {
            "model_calls": len(turns),
            "sim_seconds": round(traj[-1].sim_time if traj else 0.0, 3),
            "wall_seconds": round(time.time() - wall_start, 3),
            "usd_model": round(sum(t.usd for t in turns), 5),
            "usd": round(sum(t.usd for t in turns), 5),
            "observation_age_mean": round(sum(ages) / len(ages), 3) if ages else 0.0,
            "observation_age_max": round(max(ages), 3) if ages else 0.0,
            "latency_mean": round(sum(t.latency_seconds for t in turns) / len(turns), 3)
            if turns else 0.0,
            "response_time": response,
            "actions_submitted": len(executor.history),
            "actions_cancelled": sum(1 for h in executor.history
                                     if h.status.value == "cancelled"),
        }

    @staticmethod
    def _response_times(scheduler, executor) -> Optional[float]:
        """Simulated seconds from an event firing to the first action issued
        after it -- the response-time channel of the mechanism figure."""
        if not scheduler.records:
            return None
        fired = scheduler.records[0].sim_time
        after = [h.submitted_at for h in executor.history if h.submitted_at >= fired]
        return round(min(after) - fired, 3) if after else None
