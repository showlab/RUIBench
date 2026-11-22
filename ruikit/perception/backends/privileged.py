"""Privileged perception: the simulator's own values, without error -- the control
condition, never the default."""

from __future__ import annotations

from typing import Optional

from ruikit.env.frames import Frame
from ruikit.perception.backends.synthetic import SyntheticPerception


class PrivilegedPerception(SyntheticPerception):
    """The control condition ``@privileged``: every perception tool
    answers with the simulator's own values -- where each thing is, how it is
    turned, how big it is, the cameras' poses -- with no error, and its pixels
    where the engine segments them (Isaac).  What a frame shows is still what
    its camera sees.  Asked for by name only, never a default; the log records
    the condition."""

    def __init__(self, scene, station, seed: int, params: Optional[dict] = None) -> None:
        super().__init__(scene, station, seed, params)
        self.scale = 0.0

    def mask_of(self, frame: Frame, obj_id: str):
        """The thing's own pixels, from the engine's segmentation of the frame."""
        return (getattr(frame, "masks", None) or {}).get(self.resolve(frame, obj_id))
