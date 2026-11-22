"""A perception backend: what answers the perception tools in one episode."""

from __future__ import annotations

from typing import List, Optional, Sequence

from ruibench.core.types import Pose
from ruikit.env.frames import Frame


class PerceptionBackend:
    """Answers about one frame, and only about what it shows: the things a
    word names (``detect``), a thing outlined at a pixel or in a box
    (``segment_at``, ``segment_box``), a pixel lifted to a point (``lift``), a
    thing's pose (``pose``); and, for the Harness components, a thing's pixels
    and points and every point the frame measured (``mask_of``, ``box_of``,
    ``points_of``, ``scene_points``).  Object ids are frame-local
    (``f12.front#3``); ``resolve`` says which body an id names, for the
    episode's own bookkeeping only."""

    def believed(self, frame: Frame) -> Pose:
        """Where the robot thinks the frame's camera was: its calibration."""
        raise NotImplementedError

    def detect(self, frame: Frame, query: str = "") -> List[dict]:
        raise NotImplementedError

    def segment_at(self, frame: Frame, u: float, v: float) -> Optional[dict]:
        raise NotImplementedError

    def segment_box(self, frame: Frame, box: Sequence[int]) -> Optional[dict]:
        raise NotImplementedError

    def lift(self, frame: Frame, u: float, v: float) -> dict:
        raise NotImplementedError

    def ray_at_height(self, frame: Frame, u: float, v: float, height: float):
        raise NotImplementedError

    def pose(self, frame: Frame, obj_id: str) -> dict:
        raise NotImplementedError

    def label(self, name: str) -> dict:
        raise NotImplementedError

    def resolve(self, frame: Frame, obj_id: str) -> str:
        raise NotImplementedError

    def mask_of(self, frame: Frame, obj_id: str):
        raise NotImplementedError

    def box_of(self, frame: Frame, obj_id: str):
        raise NotImplementedError

    def points_of(self, frame: Frame, obj_id: str):
        raise NotImplementedError

    def scene_points(self, frame: Frame):
        raise NotImplementedError
