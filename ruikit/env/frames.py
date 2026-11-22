"""Captured frames.

Every picture an agent receives gets a ``frame_id`` (``f12.front``).  A frame
keeps what a real camera would have told the robot at that instant -- the
image, where the camera stood, its intrinsics -- plus, on the server side only,
the snapshot the synthetic perception backend answers from: which bodies were
in view and where everything was.  Pixels are always interpreted in the frame
they came from (a pixel picked in an old picture is lifted with that picture's
depth, not today's), which is what makes pointing at a moving target honest.

The snapshot never leaves the server: tools read it through
``ruikit.perception.backends.synthetic``, which answers only about what was visible.
"""

from __future__ import annotations

import io
import itertools
from dataclasses import dataclass, field
from typing import Dict, Mapping, Optional, Tuple

from ruibench.core.types import ObjectState, Pose

from ruikit.core.result import ToolError

#: images kept per episode; older frames keep their metadata but lose pixels
KEEP_IMAGES = 200


@dataclass
class Frame:
    id: str
    camera: str                      # agent-facing name: front / wrist / head / wrist_left ...
    scene_camera: str                # the engine's name
    sim_time: float
    width: int
    height: int
    pose: Pose                       # camera pose, optical convention (+z forward, x right, y down)
    focal_mm: float
    apertures: Tuple[float, Optional[float]]
    image: object = None             # engine image (array / tensor); None on the mock
    #: what an RGB-D camera adds: metres along the optical axis a pixel (HxW),
    #: when the episode asked its cameras for it (``perception: model``)
    depth: object = None
    #: the robot's own readings at that instant: arm id -> (tool pose, finger
    #: gap or None) -- what it knows of where its hands were in the picture
    arms: Mapping[str, tuple] = field(default_factory=dict)
    # -- server side only: the synthetic backend's view of this instant -----
    visible: Tuple[str, ...] = ()
    objects: Mapping[str, ObjectState] = field(default_factory=dict)
    #: frame-local ids -> body name, filled the first time the frame is queried
    ids: Dict[str, str] = field(default_factory=dict)
    #: body name -> its pixels (bool HxW), only for the privileged condition
    masks: Mapping[str, object] = field(default_factory=dict)

    @property
    def has_pixels(self) -> bool:
        return self.image is not None

    def intrinsics(self) -> dict:
        across, down = self.apertures
        fx = self.width * self.focal_mm / across
        fy = self.height * self.focal_mm / down if down else fx
        return {"fx": fx, "fy": fy, "cx": self.width / 2.0, "cy": self.height / 2.0}


class FrameStore:
    def __init__(self) -> None:
        self._frames: Dict[str, Frame] = {}
        self._order: list = []
        self._counter = itertools.count(1)

    def new_id(self, camera: str) -> str:
        return f"f{next(self._counter)}.{camera}"

    def add(self, frame: Frame) -> Frame:
        self._frames[frame.id] = frame
        self._order.append(frame.id)
        if len(self._order) > KEEP_IMAGES:
            old = self._frames.get(self._order[-KEEP_IMAGES - 1])
            if old is not None:
                old.image = None
                old.depth = None
        return frame

    def get(self, frame_id: str) -> Frame:
        if frame_id == "latest" or frame_id.startswith("latest."):
            return self._latest(frame_id)
        frame = self._frames.get(frame_id)
        if frame is None:
            raise ToolError("stale_frame", f"no frame {frame_id!r} in this episode",
                            hint="take a new picture with camera_look and use its frame_id")
        return frame

    def _latest(self, alias: str) -> Frame:
        """'latest' is the newest picture; 'latest.<camera>' that camera's newest --
        so a reply can look and read the picture in one go (on DS4, an agent on
        the pointing interface had to guess the next frame id)."""
        camera = alias[len("latest."):] if alias != "latest" else None
        for fid in reversed(self._order):
            if camera is None or fid.split(".", 1)[-1] == camera:
                return self._frames[fid]
        raise ToolError("stale_frame", f"no {camera + ' ' if camera else ''}picture taken yet",
                        hint="take one with camera_look first (it can be earlier in the same "
                             "reply)")

    def last(self, n: int = 1):
        return [self._frames[i] for i in self._order[-n:]]

    def __len__(self) -> int:
        return len(self._order)


def to_array(image):
    """An engine image (numpy array, torch tensor, None) as a uint8 HxWx3 array."""
    if image is None:
        return None
    try:
        import numpy as np
    except ImportError:          # pragma: no cover - numpy ships with the dev extra
        return None
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    arr = np.asarray(image)
    if arr.ndim != 3:
        return None
    if arr.dtype != np.uint8:
        scale = 255.0 if arr.max() <= 1.0 else 1.0
        arr = np.clip(arr * scale, 0, 255).astype(np.uint8)
    return arr[..., :3]


def encode_png(image, box: Optional[Tuple[int, int, int, int]] = None,
               upscale_to: Optional[int] = None) -> Optional[bytes]:
    """PNG bytes of an engine image, optionally cropped to ``box`` (u0, v0, u1, v1)
    and scaled up so the crop's long side is ``upscale_to`` pixels.  Scaling
    adds no information -- it only spares the model from squinting."""
    arr = to_array(image)
    if arr is None:
        return None
    from PIL import Image as PILImage

    img = PILImage.fromarray(arr)
    if box is not None:
        img = img.crop(box)
        if upscale_to and max(img.size) < upscale_to:
            k = upscale_to / max(img.size)
            img = img.resize((max(1, int(img.size[0] * k)), max(1, int(img.size[1] * k))),
                             PILImage.BICUBIC)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
