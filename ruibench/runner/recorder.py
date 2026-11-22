"""Camera frames for the episode viewer.

What the model looked at is the observation, so a recorded run saves every
observation handed to the agent, the final state the episode was scored on,
and, between them, a frame every ``every`` simulated seconds (0: none) --
what happened while the model was thinking.  JPEG,
downscaled.  The log lists each frame with its time, camera and kind; paths are
relative to ``RunConfig.record_dir``.

An engine that renders no pixels (the mock) records nothing, and the viewer
shows the plan view alone.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import List, Mapping, Optional, Sequence

from ..core.types import CameraFrame

OBSERVATION = "observation"
BETWEEN = "between"
FINAL = "final"
DEMONSTRATION = "demonstration"


class FrameRecorder:
    def __init__(self, root: str | Path, every: float = 0.5, size: int = 256,
                 quality: int = 82) -> None:
        self.root = Path(root)
        self.every = float(every)
        self.size = int(size)
        self.quality = int(quality)
        self.records: List[dict] = []
        self._dir: Optional[Path] = None
        self._prefix = ""
        self._last = float("-inf")

    def start(self, key: str) -> None:
        """Begin an episode.  A retried episode starts from an empty folder."""
        self._prefix = re.sub(r"[^A-Za-z0-9_.-]+", "-", key)
        self._dir = self.root / self._prefix
        if self._dir.exists():
            shutil.rmtree(self._dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._last = float("-inf")
        self.records = []

    def observation(self, frames: Mapping[str, CameraFrame]) -> None:
        """The frames the agent is about to be shown."""
        self._save(frames, OBSERVATION)

    def demonstration(self, index: int, frames: Mapping[str, CameraFrame]) -> None:
        """A frame of a demonstration played before the episode."""
        self._save(frames, DEMONSTRATION, prefix=f"demo{index}-")

    def tick(self, engine, cameras: Sequence[str]) -> None:
        """Called as the world advances; renders a frame when one is due."""
        if self.every <= 0 or not engine.capabilities.rendering:
            return
        if engine.sim_time - self._last < self.every - 1e-9:
            return
        self._save({name: engine.render(name) for name in cameras}, BETWEEN)

    def final(self, engine, cameras: Sequence[str]) -> None:
        """The world as the episode was scored -- after the agent stopped and
        things settled.  Saved even when nothing else between turns is."""
        if engine.capabilities.rendering:
            self._save({name: engine.render(name) for name in cameras}, FINAL)

    def _save(self, frames: Mapping[str, CameraFrame], kind: str, prefix: str = "") -> None:
        if self._dir is None:
            return
        for name, frame in frames.items():
            image = to_image(frame.image, self.size)
            if image is None:
                continue
            file = f"{prefix}{frame.captured_at:09.3f}-{name}.jpg"
            image.save(self._dir / file, quality=self.quality)
            record = {"t": round(frame.captured_at, 3), "camera": name,
                      "kind": kind, "path": f"{self._prefix}/{file}"}
            if prefix:
                record["demonstration"] = int(prefix[4:-1])
            self.records.append(record)
            if kind != DEMONSTRATION:
                self._last = frame.captured_at


def to_image(pixels, size: int):
    """A camera tensor or array as a PIL image no wider than ``size``; None if
    there are no pixels (or no Pillow to encode them)."""
    if pixels is None:
        return None
    try:
        import numpy as np
        from PIL import Image
    except ImportError:                          # pragma: no cover - optional dependency
        return None
    if hasattr(pixels, "detach"):
        pixels = pixels.detach().cpu().numpy()
    array = np.asarray(pixels)
    if array.ndim != 3 or array.shape[2] < 3:
        return None
    image = Image.fromarray(array[:, :, :3].astype("uint8"))
    if max(image.size) > size:
        image.thumbnail((size, size))
    return image
