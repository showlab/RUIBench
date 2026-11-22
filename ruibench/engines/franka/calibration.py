"""Where the real station's table and cameras are, relative to the robot.

Measured by ``python -m ruibench.engines.franka.calibrate`` and kept in
``calibration.json`` beside this file; every episode's log carries the
calibration it ran with (``FrankaEngine.describe``).

* ``table_z_m`` -- the table top's height in the robot's base frame (the
  frame Polymetis reports in: ``panda_link0``).  The station frame every tool
  speaks is the base frame lowered by it: origin on the table top under the
  robot base, x forward, y left, z up.
* ``front`` -- the fixed camera in the base frame.
* ``wrist`` -- the wrist camera in the flange's frame (``panda_link8``): it
  rides the hand.
* ``depth_scale`` -- by camera, what corrects its depth to the table's plane.

Camera poses are in the optical convention (+z along the view, image x right,
y down) and quaternions (w, x, y, z), as everywhere in RUIBench.

Until the station is calibrated the values are provisional -- the table from
the Franka setup's contact measurement (2026-06-20: the flange 0.1539 m above
the base with the closed fingertips, 11.23 cm below it, on the table), the
cameras roughly where they stand -- and ``measured`` is None: the engine runs,
and every position perception reports is off by as much as the guess.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Optional

from ...core.geometry import camera_quat, compose, inverse
from ...core.types import Pose
from .model import HAND_ON_FLANGE_RAD, PANDA, TCP_FROM_FLANGE_M

PATH = Path(__file__).with_name("calibration.json")


#: the tool centre point in the flange's frame: the Franka Hand turned on the
#: flange, its tool centre point along its axis
FLANGE_TO_TCP = compose(
    Pose((0.0, 0.0, 0.0), (math.cos(HAND_ON_FLANGE_RAD / 2), 0.0, 0.0,
                           math.sin(HAND_ON_FLANGE_RAD / 2))),
    Pose((0.0, 0.0, TCP_FROM_FLANGE_M)))


@dataclass(frozen=True)
class Calibration:
    table_z_m: float
    front: Pose
    wrist: Pose
    #: when it was measured (ISO date), None while provisional
    measured: Optional[str] = None
    #: by camera, what its depth is multiplied by to read the table where the board
    #: says it is: a RealSense's depth runs long or short by a fraction of a percent
    depth_scale: Mapping[str, float] = field(default_factory=dict)
    #: how well it fits: residuals, the poses and board it was measured with
    fit: Mapping[str, object] = field(default_factory=dict)

    @property
    def provisional(self) -> bool:
        return self.measured is None

    def as_dict(self) -> dict:
        def pose(p: Pose) -> dict:
            return {"pos": [round(v, 6) for v in p.pos], "quat_wxyz": [round(v, 7) for v in p.quat]}

        return {"table_z_m": round(self.table_z_m, 6), "front": pose(self.front),
                "wrist": pose(self.wrist), "measured": self.measured,
                "depth_scale": dict(self.depth_scale), "fit": dict(self.fit)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Calibration":
        def pose(p: Mapping) -> Pose:
            return Pose(tuple(p["pos"]), tuple(p["quat_wxyz"]))

        return cls(table_z_m=float(d["table_z_m"]), front=pose(d["front"]),
                   wrist=pose(d["wrist"]), measured=d.get("measured"),
                   depth_scale=dict(d.get("depth_scale") or {}), fit=dict(d.get("fit") or {}))


def provisional() -> Calibration:
    """The station before it is measured: the table from the contact
    measurement, the fixed camera about where it stands across the table, the
    wrist camera where the model's nominal mount puts it."""
    table = 0.1539 - 0.1123
    eye, target = (1.15, 0.0, table + 0.55), (0.45, 0.0, table)
    return Calibration(
        table_z_m=round(table, 4),
        front=Pose(eye, camera_quat(eye, target)),
        wrist=compose(FLANGE_TO_TCP, PANDA.camera_from_tcp))


def load(path: Optional[Path] = None) -> Calibration:
    """The station's calibration, or the provisional one where none is saved."""
    path = Path(path or PATH)
    if not path.exists():
        return provisional()
    return Calibration.from_dict(json.loads(path.read_text(encoding="utf-8")))


def save(cal: Calibration, path: Optional[Path] = None) -> Path:
    path = Path(path or PATH)
    path.write_text(json.dumps(cal.as_dict(), indent=1) + "\n", encoding="utf-8")
    return path


def tcp_to_flange(tcp: Pose) -> Pose:
    """The flange's pose that puts the tool centre point at ``tcp``."""
    return compose(tcp, inverse(FLANGE_TO_TCP))


def flange_to_tcp(flange: Pose) -> Pose:
    return compose(flange, FLANGE_TO_TCP)


__all__ = ["Calibration", "FLANGE_TO_TCP", "PATH", "TCP_FROM_FLANGE_M", "flange_to_tcp", "load",
           "provisional", "save", "tcp_to_flange"]
