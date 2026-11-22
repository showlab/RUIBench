"""What perception from pixels measures: a detection, a measured view, and the thresholds
they share."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from ruibench.core.geometry import rotate
from ruikit.core.result import ToolError

#: a mask with fewer measured pixels than this has nothing to say
MIN_POINTS = 12


#: a mask is the table top when it stands no higher than this over it -- or,
#: seen from afar, than three times what the smoothed depth can tell
FLAT_M = 0.002


#: two masks are the same thing from this overlap (IoU) up
SAME_MASK_IOU = 0.5


#: a thing's top and bottom, as percentiles of its points' heights
TOP_PCT, LOW_PCT = 92.0, 4.0


#: asked for along with the task's words when everything is listed: what
#: answers to no word of the task's is still a thing (a Hanoi ring is no
#: "ring" to the detector, but it is an "object")
GENERIC = "object"


def _np():
    try:
        import numpy as np
    except ImportError:                     # pragma: no cover - numpy comes with Isaac
        raise ToolError("not_supported", "perception from pixels needs numpy") from None
    return np


def _rot(quat):
    """The rotation matrix of a unit quaternion (w, x, y, z)."""
    np = _np()
    return np.array([rotate(quat, e) for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]).T


@dataclass
class Detection:
    id: str
    crop: object                       # bool: its pixels inside its box (the robot's own taken out)
    shape: Tuple[int, int]             # the picture's rows and columns
    box: Tuple[int, int, int, int]
    scores: Dict[str, float]           # phrase -> the detector's score for it
    centre: Tuple[int, int]
    area: int = 0
    colour: Optional[str] = None
    #: the picture's edge cuts it: its box and its pose are of the part in view
    cut: bool = False

    @property
    def mask(self):
        """Its pixels, as a picture-sized bool array."""
        full = _np().zeros(self.shape, dtype=bool)
        full[self.box[1]:self.box[3], self.box[0]:self.box[2]] = self.crop
        return full

    def covers(self, row: int, col: int) -> bool:
        u0, v0, u1, v1 = self.box
        return u0 <= col < u1 and v0 <= row < v1 and bool(self.crop[row - v0, col - u0])

    @property
    def label(self) -> str:
        """The phrase it answered best -- any before the catch-all."""
        named = {p: s for p, s in self.scores.items() if p != GENERIC}
        return max(named.items(), key=lambda kv: kv[1])[0] if named else GENERIC

    def described(self, phrase: str = "", spelling: Optional[Dict[str, str]] = None) -> dict:
        """What a tool says of it; ``score`` is the detector's for ``phrase``,
        or for its own label.  A label that is one of the task's words is
        spelt as the task spells it ("X piece")."""
        score = self.scores.get(phrase) or self.scores.get(self.label)
        out = {"id": self.id, "label": (spelling or {}).get(self.label, self.label)}
        if score is not None:
            out["score"] = round(score, 2)
        if self.colour and self.colour not in self.label:
            out["color"] = self.colour
        if self.cut:
            out["cut_by_frame"] = True
        return out


@dataclass
class View:
    """One frame as the robot measures it."""
    depth: object                      # HxW metres along the optical axis, NaN: no reading
    points: object                     # HxWx3 world coordinates, by the believed camera pose
    height: object                     # HxW metres above the fitted table top
    robot: object                      # bool HxW: the robot's own body
    plane: Tuple[float, float, float]  # the table top: z = a x + b y + c
    sigma: float                       # what the smoothed depth can tell, at the table (m)
    dets: List[Detection] = field(default_factory=list)    # the frame's (shared with the store)
    asked: set = field(default_factory=set)
