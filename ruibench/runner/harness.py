"""Running many episodes: a grid over tasks, levels, seeds and conditions.

Deliberately simple and sequential.  Parallelism belongs one level up (one
process per worker over disjoint seed ranges) because the API rate limit, not
the CPU, is the binding constraint: roughly 7,440 episodes x ~20 requests x
~20 s is on the order of 827 hours of cumulative waiting, about 103 hours at
8-way concurrency (v0 §6).  Confirm the concurrency and the rate limit during
the pilot, not after the main run has started.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Tuple

from ..core.config import RunConfig
from ..clock import make_clock
from ..engines import make_engine
from ..interfaces import make_interface
from ..policies.base import Agent
from ..tasks.instance import TaskInstance, build
from .episode import EpisodeResult, EpisodeRunner
from .log import EpisodeLog, write_jsonl


@dataclass
class RunPlan:
    """What to run.  ``conditions`` names the paired conditions to add."""

    task_ids: Sequence[str]
    levels: Sequence[str] = ("easy", "hard")
    seeds: Sequence[int] = (0,)
    interface: str = "I0_direct"
    conditions: Sequence[str] = ("main",)
    #: (task, level, seed) to run instead of every task x level x seed -- a
    #: split's seeds differ from task to task (``tasks/splits.py``)
    missions: Optional[Sequence[Tuple[str, str, int]]] = None

    def cells(self) -> List[Tuple[str, str, int, str]]:
        if self.missions is not None:
            return [(t, lv, s, c) for t, lv, s in self.missions for c in self.conditions]
        return [(t, lv, s, c) for t in self.task_ids for lv in self.levels
                for s in self.seeds for c in self.conditions]


_ENGINES: dict = {}


def engine_options(cfg: RunConfig) -> dict:
    """The run config's requests that go to the engine's constructor.

    Only what the config actually sets: ``None`` leaves the backend's own
    default, and the log records what the engine ran with either way.  A
    backend that cannot honour a request is an error, not a silent no-op.
    """
    import inspect

    from ..core.errors import ConfigError
    from ..engines import ENGINES

    wanted = {k: v for k, v in (("sim_hz", cfg.sim_hz), ("control_hz", cfg.control_hz))
              if v is not None}
    accepted = inspect.signature(ENGINES.get(cfg.engine)()).parameters
    refused = sorted(set(wanted) - set(accepted))
    if refused:
        raise ConfigError(f"engine {cfg.engine!r} does not take {', '.join(refused)}")
    return wanted


def _engine_for(cfg: RunConfig, keep: bool):
    """A fresh engine per episode, or one kept alive across episodes.

    Keeping it is what makes the Isaac backend affordable: starting the
    simulator costs tens of seconds, loading a scene costs a few.
    """
    options = engine_options(cfg)
    if not keep:
        return make_engine(cfg.engine, **options)
    key = (cfg.engine, tuple(sorted(options.items())))
    if key not in _ENGINES:
        _ENGINES[key] = make_engine(cfg.engine, **options)
    return _ENGINES[key]


def run_plan(plan: RunPlan, agent_factory: Callable[[TaskInstance], Agent],
             config: Optional[RunConfig] = None,
             out: Optional[str | Path] = None,
             progress: Optional[Callable[[str], None]] = None,
             keep_engine: bool = False) -> List[EpisodeResult]:
    cfg = config or RunConfig(interface=plan.interface)
    results: List[EpisodeResult] = []
    for task_id, level, seed, condition in plan.cells():
        inst = build(task_id, level=level, seed=seed, config=cfg)
        engine = _engine_for(cfg, keep_engine)
        interface = make_interface(plan.interface)
        clock = make_clock(cfg.clock, **({"latency_seconds": cfg.clock_latency_seconds}
                                         if cfg.clock == "controlled" else {}))
        agent = agent_factory(inst)
        runner = EpisodeRunner(engine, interface, agent, clock, cfg)
        try:
            result = runner.run(inst, condition=condition)
        finally:
            if not keep_engine:
                engine.close()
        results.append(result)
        if progress:
            mark = "ok " if result.success else ("gap" if result.score.coverage_gap else "-- ")
            progress(f"{mark} {task_id} {level:4s} seed={seed} "
                     f"progress={result.score.progress:.2f} {result.score.failure.value}")
    if out:
        write_jsonl(out, [r.log for r in results])
    return results
