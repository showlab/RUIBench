"""Engine-agnostic core: value types, seeds, config, registry, errors."""

from .config import DEFAULT_T_REF_SECONDS, BudgetConfig, RunConfig
from .errors import (
    BudgetExceeded, ConfigError, CoverageGap, EngineNotAvailable,
    FrozenAssetError, RUIBenchError,
)
from .registry import Registry
from .rng import SeedBundle, derive_seed
from .types import (
    ArmState, CameraFrame, GRASP_DOWN, IDENTITY_QUAT, Observation, ObjectState, Pose,
    Quat, RobotState, Vec3, WorldState,
)

__all__ = [
    "DEFAULT_T_REF_SECONDS", "BudgetConfig", "RunConfig",
    "BudgetExceeded", "ConfigError", "CoverageGap", "EngineNotAvailable",
    "FrozenAssetError", "RUIBenchError", "Registry", "SeedBundle", "derive_seed",
    "ArmState", "CameraFrame", "GRASP_DOWN", "IDENTITY_QUAT", "Observation", "ObjectState",
    "Pose", "Quat", "RobotState", "Vec3", "WorldState",
]
