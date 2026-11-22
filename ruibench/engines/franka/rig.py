"""The real station's settings (``rig.yaml``): how the robot and its cameras are
reached, and the limits the arm moves within."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple

import yaml

from ...core.errors import ConfigError

PATH = Path(__file__).with_name("rig.yaml")
#: overrides the NUC's address: ``<host>`` or ``<host>:<port>``
NUC_ENV = "RUIBENCH_FRANKA_NUC"


@dataclass(frozen=True)
class CameraRig:
    """One RealSense: which, and the raw stream rectified to the model's lens."""
    serial: str
    width: int = 640
    height: int = 480
    fps: int = 30


@dataclass(frozen=True)
class Motion:
    control_hz: float = 50.0
    max_speed_m_s: float = 0.15
    max_turn_rad_s: float = 0.6
    descent_speed_m_s: float = 0.08
    descent_below_m: float = 0.30
    joint_speed_rad_s: float = 0.6
    min_move_seconds: float = 0.4
    max_slowdown: float = 4.0
    max_move_seconds: float = 15.0
    stiffness: Tuple[float, ...] = (750.0, 750.0, 750.0, 15.0, 15.0, 15.0)
    damping: Tuple[float, ...] = (37.0, 37.0, 37.0, 2.0, 2.0, 2.0)
    settle_gain: float = 3.0
    settle_limit_m: float = 0.01
    settle_limit_rad: float = 0.05
    follow_limit_m: float = 0.03
    follow_lag_s: float = 0.05
    follow_limit_rad: float = 0.3
    joint_speed_limit_rad_s: float = 1.2


@dataclass(frozen=True)
class Workspace:
    """Where the tool centre point may go (station frame, metres), and how far
    its axis may tilt from pointing straight down."""
    x: Tuple[float, float] = (0.15, 0.80)
    y: Tuple[float, float] = (-0.45, 0.45)
    z: Tuple[float, float] = (0.008, 0.60)
    max_tilt_deg: float = 100.0

    def clamp(self, p) -> Tuple[float, float, float]:
        return tuple(min(max(float(v), lo), hi) for v, (lo, hi) in zip(p, (self.x, self.y, self.z)))

    def contains(self, p, slack: float = 1e-6) -> bool:
        return all(lo - slack <= float(v) <= hi + slack
                   for v, (lo, hi) in zip(p, (self.x, self.y, self.z)))


@dataclass(frozen=True)
class GripperRig:
    open_width_m: float = 0.08
    stable_m: float = 0.0005
    stable_s: float = 0.3
    min_s: float = 0.4
    max_s: float = 3.0
    grasp_speed_m_s: float = 0.05
    grasp_force_n: float = 60.0


@dataclass(frozen=True)
class Rig:
    host: str = "172.16.0.2"
    port: int = 4242
    cameras: Mapping[str, CameraRig] = field(default_factory=dict)
    motion: Motion = field(default_factory=Motion)
    workspace: Workspace = field(default_factory=Workspace)
    gripper: GripperRig = field(default_factory=GripperRig)

    def describe(self) -> dict:
        """What an episode's log keeps of the rig."""
        return {"nuc": f"{self.host}:{self.port}",
                "cameras": {name: {"serial": c.serial, "stream": [c.width, c.height, c.fps]}
                            for name, c in self.cameras.items()},
                "motion": dict(vars(self.motion)), "workspace": dict(vars(self.workspace)),
                "gripper": dict(vars(self.gripper))}


def _section(cls, data: Optional[Mapping], name: str):
    data = dict(data or {})
    known = set(cls.__dataclass_fields__)
    unknown = set(data) - known
    if unknown:
        raise ConfigError(f"rig.yaml: unknown keys in {name}: {sorted(unknown)}")
    return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in data.items()})


def load(path: Optional[Path] = None) -> Rig:
    """The rig as ``rig.yaml`` describes it, the NUC's address overridden by
    ``RUIBENCH_FRANKA_NUC``."""
    raw = yaml.safe_load(Path(path or PATH).read_text(encoding="utf-8")) or {}
    nuc = dict(raw.get("nuc") or {})
    host, port = str(nuc.get("host", Rig.host)), int(nuc.get("port", Rig.port))
    override = os.environ.get(NUC_ENV, "").strip()
    if override:
        host, _, given = override.partition(":")
        port = int(given) if given else port
    cameras: Dict[str, CameraRig] = {
        name: _section(CameraRig, spec, f"cameras.{name}")
        for name, spec in (raw.get("cameras") or {}).items()}
    return Rig(host=host, port=port, cameras=cameras,
               motion=_section(Motion, raw.get("motion"), "motion"),
               workspace=_section(Workspace, raw.get("workspace"), "workspace"),
               gripper=_section(GripperRig, raw.get("gripper"), "gripper"))


__all__ = ["CameraRig", "GripperRig", "Motion", "NUC_ENV", "Rig", "Workspace", "load"]
