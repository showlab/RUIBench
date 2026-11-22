"""Run-level configuration.

Every field here is written into the episode log, because a result is only
comparable to another result taken under the same configuration (design doc
details §4.3: "each evaluation records model id, inference config, prompt
hash, service version, sensor permissions, date").
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Mapping, Optional, Tuple

# T_ref is a *configurable hyper-parameter*, not a hard-coded constant, but it
# is a benchmark-level setting: within one run it is identical for every model
# (design doc v0 §3.1a).  The speed tasks (DI2-DI6) state their timing in it,
# so changing it scales them and nothing else.  PROVISIONAL: 8 s, a typical
# frontier model's effective response time assumed until Gate C measures one
# with the real API (the user, 2026-09-25); v0 had 20 s as a placeholder.  At
# 8 s the scripted reference taking 8 s a turn (``--clock controlled
# --latency 8``) solves every easy level and almost none of the hard ones (at
# most 2 seeds in 20), which the same reference at zero latency solves.
DEFAULT_T_REF_SECONDS = 8.0


#: how long ``RunConfig.play_out`` lets an episode go on after its success
PLAY_OUT_SECONDS = 30.0


@dataclass(frozen=True)
class BudgetConfig:
    """Per-episode stopping conditions.  Exhausting any of them ends the episode."""

    max_model_calls: int = 60
    #: generous by default: 60 calls at 20 s a call is already 1,200 s of pure
    #: thinking under the measured or controlled clock.
    max_sim_seconds: float = 1800.0
    max_wall_seconds: float = 1800.0
    max_usd: float = 5.0


@dataclass(frozen=True)
class RunConfig:
    """The full description of one evaluation run."""

    run_id: str = "dev"
    engine: str = "mock"
    interface: str = "I0_direct"
    agent: str = "scripted"
    clock: str = "zero_latency"
    clock_latency_seconds: float = 0.0
    t_ref_seconds: float = DEFAULT_T_REF_SECONDS
    #: ``None`` keeps the backend's own rates.  Either way the log's
    #: ``engine_setup`` records what the engine actually ran at.
    control_hz: Optional[float] = None
    sim_hz: Optional[float] = None
    #: Applied to every camera of every scene by the episode runner.
    camera_width: int = 512
    camera_height: int = 512
    #: Save camera frames for the episode viewer under this directory: every
    #: observation, plus one every ``record_every`` simulated seconds (0: the
    #: observations only).  ``None`` records nothing.
    record_dir: Optional[str] = None
    record_every: float = 0.5
    record_size: int = 256
    cameras: Tuple[str, ...] = ("agentview", "wrist")
    #: For showing an episode, not for scoring one: after the task first
    #: succeeds, let the agent finish what it is doing (up to
    #: ``PLAY_OUT_SECONDS`` more) instead of stopping there -- a reference that
    #: puts its tools down and folds its arms away is seen doing it.  Scored
    #: on where things end, as ever.
    play_out: bool = False
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    split: str = "dev"
    model_id: Optional[str] = None
    model_snapshot: Optional[str] = None
    inference_config: Mapping[str, object] = field(default_factory=dict)
    prompt_hash: Optional[str] = None
    services_version: str = "0.1.0"
    notes: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return asdict(self)
