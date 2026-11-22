"""Engine backends.  Pick one by name; everything above never knows which."""

from __future__ import annotations

from ..core.registry import Registry
from .base import (
    ArticulationSpec, BodySpec, CameraSpec, Engine, EngineCapabilities,
    MarkerSpec, RobotSpec, SceneSpec,
)

ENGINES: Registry = Registry("engine")


def _isaac():
    from .isaac import IsaacEngine
    return IsaacEngine


def _mock():
    from .mock import MockEngine
    return MockEngine


def _franka():
    from .franka import FrankaEngine
    return FrankaEngine


#: The engine results are reported on.  Frontier models are never fine-tuned on
#: our images, so rendering fidelity is part of the experiment: whatever we show
#: them has to look like the world.
ENGINES.register("isaac", _isaac)
#: No physics and no pixels -- CI, scorer tests, negative controls, and task
#: logic before it goes to Isaac.  Never a result.  (A MuJoCo/robosuite
#: development backend existed until 2026-09-17; it could not be installed next
#: to Isaac and nothing tested it, so it was removed.)
ENGINES.register("mock", _mock)
#: The lab's Franka Panda and its RealSense cameras: the real station.  Its
#: results are real-robot results, kept apart from the simulation's.
ENGINES.register("franka", _franka)


def make_engine(name: str, **kwargs) -> Engine:
    """Instantiate a backend by name: ``isaac``, ``mock`` or ``franka``."""
    return ENGINES.get(name)()(**kwargs)


def results_grade(name: str) -> bool:
    """May this backend's episodes appear on a leaderboard?"""
    return bool(ENGINES.get(name)().capabilities.results_grade)


def is_real(engine) -> bool:
    """Is ``engine`` -- a backend's name or an engine -- a physical robot
    (``EngineCapabilities.real``)?  Asked of the class, so a name needs no robot."""
    caps = (ENGINES.get(engine)().capabilities if isinstance(engine, str)
            else getattr(engine, "capabilities", None))
    return bool(getattr(caps, "real", False))


__all__ = [
    "ENGINES", "make_engine", "results_grade", "is_real", "Engine", "EngineCapabilities",
    "SceneSpec", "BodySpec", "ArticulationSpec", "MarkerSpec", "CameraSpec", "RobotSpec",
]
