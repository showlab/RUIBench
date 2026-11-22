"""The station's RealSense cameras: a colour picture and depth, as the robot model's lens.

Each camera streams colour and depth (``rig.yaml``) on a thread of its own,
which keeps the newest frames -- a picture asked for is the world now, not one
queued up a second ago.  Depth is aligned to the colour picture by the camera's
own calibration (``rs.align``).

What a tool gets is the picture through the robot model's lens
(``engines.franka.model``): rectified -- the colour lens's distortion undone,
the principal point put in the middle, square pixels -- and cut to the size and
field of view the model gives the camera.  Every frame then means what a frame
means in simulation: a pinhole of ``camera_focal_length`` (or the wrist's) on
the shared 20.955 mm sensor, centred, so the perception tools read it as they
read Isaac's.  A lens wider than the sensor's picture is refused when the
camera starts: it would need pixels the camera never saw.

The camera's factory intrinsics are kept: the D405's colour lens follows the
inverse Brown-Conrady model (its coefficients take a distorted point to the
undistorted one, ``rs2_deproject_pixel_to_point``), the D435's has none.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

#: how long a camera may take to deliver its first frames
START_SECONDS = 10.0
#: a frame older than this is no picture of now: the stream has stalled
STALE_SECONDS = 1.0
#: frames dropped while the exposure settles
WARMUP_FRAMES = 15


@dataclass(frozen=True)
class Intrinsics:
    """A camera's colour lens, as the camera reports it (pixel centres at
    integers, OpenCV's convention): focal lengths, principal point, and the
    five distortion coefficients of its model."""
    width: int
    height: int
    fx: float
    fy: float
    ppx: float
    ppy: float
    coeffs: Tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    model: str = "none"

    def as_dict(self) -> dict:
        return {"size": [self.width, self.height], "fx": round(self.fx, 3), "fy": round(self.fy, 3),
                "ppx": round(self.ppx, 3), "ppy": round(self.ppy, 3),
                "coeffs": [round(c, 6) for c in self.coeffs], "model": self.model}


class Rectifier:
    """Remaps a camera's raw picture onto a centred pinhole of focal length
    ``focal_px`` and ``width`` x ``height`` pixels.  A pixel's middle is at
    +0.5 as RUIBench's perception reads it (``Frame.intrinsics``)."""

    def __init__(self, raw: Intrinsics, width: int, height: int, focal_px: float) -> None:
        import numpy as np

        self.raw, self.focal_px = raw, float(focal_px)
        self.width, self.height = int(width), int(height)
        us = (np.arange(self.width) + 0.5 - self.width / 2.0) / self.focal_px
        vs = (np.arange(self.height) + 0.5 - self.height / 2.0) / self.focal_px
        xu, yu = np.meshgrid(us, vs)
        xd, yd = _distorted(xu, yu, raw.coeffs) if raw.model == "inverse_brown_conrady" \
            else (xu, yu)
        self.map_x = (xd * raw.fx + raw.ppx).astype(np.float32)
        self.map_y = (yd * raw.fy + raw.ppy).astype(np.float32)
        outside = ((self.map_x < -0.5) | (self.map_x > raw.width - 0.5)
                   | (self.map_y < -0.5) | (self.map_y > raw.height - 0.5))
        #: the share of the rectified picture the camera never saw
        self.uncovered = float(outside.mean())

    def colour(self, image):
        import cv2

        return cv2.remap(image, self.map_x, self.map_y, cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    def depth(self, depth):
        """Depth lies along the optical axis, which rectifying does not turn:
        each value moves with its pixel, never mixed with a neighbour's."""
        import cv2

        return cv2.remap(depth, self.map_x, self.map_y, cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _distorted(xu, yu, coeffs: Sequence[float], iterations: int = 20):
    """The distorted normalised coordinates whose undistortion is (xu, yu),
    under the inverse Brown-Conrady model: solved by fixed point, as the model
    gives the undistorted point of a distorted one."""
    k1, k2, p1, p2, k3 = (float(c) for c in coeffs)
    xd, yd = xu.copy(), yu.copy()
    for _ in range(iterations):
        r2 = xd * xd + yd * yd
        f = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        ux = xd * f + 2.0 * p1 * xd * yd + p2 * (r2 + 2.0 * xd * xd)
        uy = yd * f + 2.0 * p2 * xd * yd + p1 * (r2 + 2.0 * yd * yd)
        xd, yd = xd - (ux - xu), yd - (uy - yu)
    return xd, yd


class RealSenseCamera:
    """One RealSense: colour and aligned depth, the newest kept by a reader thread."""

    def __init__(self, serial: str, width: int = 640, height: int = 480, fps: int = 30,
                 depth: bool = True) -> None:
        """``depth``: False streams colour alone, the only way a camera on a USB 2
        port reaches its largest pictures (for the calibration)."""
        self.serial, self.fps = str(serial), int(fps)
        self.width, self.height = int(width), int(height)
        self.depth = bool(depth)
        self.intrinsics: Optional[Intrinsics] = None
        self.depth_scale = 0.001
        self._pipeline = None
        self._align = None
        self._lock = threading.Lock()
        self._latest = None                   # (monotonic time, frameset)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[str] = None

    def start(self) -> "RealSenseCamera":
        import pyrealsense2 as rs

        config = rs.config()
        config.enable_device(self.serial)
        config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
        if self.depth:
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
        self._pipeline = rs.pipeline()
        try:
            profile = self._pipeline.start(config)
        except RuntimeError as exc:
            raise RuntimeError(f"RealSense {self.serial} would not start {self.width}x{self.height}"
                               f"@{self.fps}: {exc} (plugged in? on a USB 3 port?)") from None
        if self.depth:
            self._align = rs.align(rs.stream.color)
            self.depth_scale = float(profile.get_device().first_depth_sensor().get_depth_scale())
        k = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(k.width, k.height, k.fx, k.fy, k.ppx, k.ppy,
                                     tuple(float(c) for c in k.coeffs),
                                     str(k.model).split(".")[-1])
        for _ in range(WARMUP_FRAMES):
            self._pipeline.wait_for_frames(int(START_SECONDS * 1000))
        self._thread = threading.Thread(target=self._read, name=f"realsense-{self.serial}",
                                        daemon=True)
        self._thread.start()
        deadline = time.monotonic() + START_SECONDS
        while self._latest is None and time.monotonic() < deadline and not self._error:
            time.sleep(0.01)
        if self._latest is None:
            raise RuntimeError(f"RealSense {self.serial} delivered no frame: {self._error}")
        return self

    def _read(self) -> None:
        while not self._stop.is_set():
            try:
                frames = self._pipeline.wait_for_frames(1000)
            except RuntimeError as exc:
                self._error = str(exc)
                continue
            frames.keep()
            with self._lock:
                self._latest = (time.monotonic(), frames)

    def _newest(self):
        with self._lock:
            latest = self._latest
        if latest is None or time.monotonic() - latest[0] > STALE_SECONDS:
            raise RuntimeError(f"RealSense {self.serial} has delivered nothing for "
                               f"{STALE_SECONDS:g} s: {self._error or 'the stream stalled'}")
        return latest[1]

    def colour(self):
        """The newest colour picture alone, which costs no aligning: for showing."""
        import numpy as np

        return np.asanyarray(self._newest().get_color_frame().get_data()).copy()

    def capture(self):
        """(colour HxWx3 uint8 RGB, depth HxW float32 metres -- 0 where none, None
        for a camera streaming colour alone), both in the raw colour camera's pixels."""
        import numpy as np

        if not self.depth:
            return self.colour(), None
        aligned = self._align.process(self._newest())
        colour = np.asanyarray(aligned.get_color_frame().get_data()).copy()
        depth = np.asanyarray(aligned.get_depth_frame().get_data()).astype(np.float32)
        return colour, depth * self.depth_scale

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        if self._pipeline is not None:
            try:
                self._pipeline.stop()
            except RuntimeError:
                pass


__all__ = ["Intrinsics", "RealSenseCamera", "Rectifier"]
