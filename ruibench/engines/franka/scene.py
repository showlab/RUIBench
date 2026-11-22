"""The real station as a scene: what the tasks on it are built on."""

from __future__ import annotations

from typing import Iterable, Optional, Tuple

from ...core.types import Pose
from ..base import BodySpec, CameraSpec, RobotSpec, SceneSpec
from . import calibration as calibration_mod
from .model import PANDA

#: the station frame is the world: the table top at z = 0, under the robot base
TABLE_HEIGHT = 0.0


def station_scene(name: str, things: Iterable[Tuple[str, str]] = (),
                  cal: Optional[calibration_mod.Calibration] = None) -> SceneSpec:
    """The lab's Panda with its fixed and wrist cameras, its base where the
    calibration puts it under the table top, and the things a task names as
    (name, category) -- declared, never placed: nothing knows where they are,
    and the categories are the words perception lists everything by."""
    cal = cal or calibration_mod.load()
    robot = RobotSpec("robot0", PANDA.name, ("right",),
                      Pose((0.0, 0.0, -float(cal.table_z_m))))
    return SceneSpec(
        name=name, robots=(robot,),
        bodies=tuple(BodySpec(n, category=c, attrs={"declared": True}) for n, c in things),
        cameras=(CameraSpec("agentview", "external"),
                 CameraSpec("wrist", "wrist", attached_to="robot0/right")),
        table_height=TABLE_HEIGHT, table=None)


__all__ = ["TABLE_HEIGHT", "station_scene"]
