"""The ``model`` perception backend: the ``vision_*`` tools answered from
pixels, with nothing from the simulator that a real RGB-D camera would not
hand over.

What it is given, a frame: the picture, a depth image, the camera's
intrinsics, and what the robot knows of itself -- where its calibration puts
the camera and where its joint readings put its hands.  What it never reads:
where a thing is, what it is called, which pixels are whose (``Frame.objects``
and ``Frame.visible`` are the synthetic backend's; ``resolve`` alone looks at
them, for scripted agents and scoring, and no tool calls it).

How a question is answered:

* **detect** -- an open-vocabulary detector and a segmenter on the picture
  (the vision backend: SAM 3 unless the interface names another detector).
  The words are the agent's query; to list everything, the task's public
  category words.  A detection is then turned away where depth says it is no
  thing on the table: the robot itself (its hands, wrists and base, where its
  own readings put them), the table top, what lies below it.
* **pose** -- the mask's pixels, lifted by depth, as a box standing on what is
  under the thing: the middle and heading of its footprint, its extent.  Only
  the side the camera sees is measured.
* **lift** -- the depth at a pixel.

What is wrong with the answers is what is wrong on a real robot.  The
camera's calibration is off by the synthetic backend's amount (3 mm, 0.3
degrees: the same draw for the same seed), and the depth by its sensor noise
(1 mm + 0.3 % of the distance a pixel, fixed for a viewpoint, so looking twice
averages nothing away).  The rest are the models' own: a thing missed, two
merged, a wrong word, a mask cut short.  On a real robot (``sensors: real``)
neither error is added: its depth camera and its calibration err as they err.

What can be swapped, for the ablations (``ruikit.perception.settings``): the
detector (``backend``), where depth comes from (``depth``: ``sensor``, the
camera's; ``mono``, a monocular model on the RGB picture whose scale is fitted
to the table top the robot knows), and how much worse than a real robot's the
depth noise and the calibration are (``noise``, ``calib``: 1, 2 or 4 times).
Never better: a perception more exact than a real robot's would hand the agent
positions it could not have.

Needs numpy, and a camera that delivers depth (``CameraSpec.depth``) -- or, with
``depth=mono``, a picture and a vision server with a depth model: Isaac.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ruibench.core.geometry import compose, rotate
from ruibench.core.types import Pose
from ruikit.core.result import ToolError
from ruikit.env.frames import Frame, to_array
from ruikit.perception.backends.base import PerceptionBackend
from ruikit.perception.backends.geometry import (
    _dilate,
    _erode,
    _footprint,
    _iou,
    _normal,
    _tilt,
)
from ruikit.perception.backends.noise import (
    CAMERA_POS_M,
    CAMERA_ROT_DEG,
    _gauss,
    _rotvec,
)
from ruikit.perception.backends.pixels_depth import (
    DepthMeasurement,
)
from ruikit.perception.backends.pixels_detect import (
    DetectionLookup,
)
from ruikit.perception.backends.pixels_types import (
    FLAT_M,
    GENERIC,
    LOW_PCT,
    MIN_POINTS,
    SAME_MASK_IOU,
    TOP_PCT,
    Detection,
    View,
    _np,
)

#: the surface a thing stands on is looked for this many pixels round its mask,
#: and taken for its bottom when it is no further than this under the lowest
#: point seen of the thing itself
RING_PX = 4
SUPPORT_BELOW_M = 0.03


class PixelPerception(DetectionLookup, DepthMeasurement, PerceptionBackend):
    def __init__(self, scene, station, seed: int, params: Optional[dict] = None,
                 backend=None) -> None:
        from ruibench.engines import robots

        self.station = station
        self.seed = int(seed)
        params = dict(params or {})
        # an interface may make perception worse, never better than a real robot's:
        # error_scale, the older single setting, is both; noise and calib each on its own
        both = max(1.0, float(params.get("error_scale", 1.0)))
        self.noise_scale = max(1.0, float(params.get("noise", both)))
        self.calib_scale = max(1.0, float(params.get("calib", both)))
        self.scale = self.noise_scale          # kept for the callers that read it
        #: a real robot's cameras (``EngineCapabilities.real``): their depth and
        #: their calibration err as they err, and nothing is added to either
        self.real_sensors = str(params.get("sensors", "simulated")) == "real"
        if self.real_sensors and (self.noise_scale > 1.0 or self.calib_scale > 1.0):
            raise ToolError("invalid_args", "noise and calib make a simulated sensor worse; a "
                            "real robot's cameras are as they are",
                            hint="drop noise= and calib= from the interface id")
        self.depth_source = str(params.get("depth", "sensor"))
        if self.depth_source not in ("sensor", "mono"):
            raise ToolError("invalid_args", f"depth must be sensor or mono, not {self.depth_source!r}")
        #: the scale fitted to the table, per frame (monocular depth), for the log
        self.mono_scales: Dict[str, float] = {}
        self.backend = backend
        #: the task's public category words: what "everything" is looked for by
        self.vocabulary = tuple(dict.fromkeys(
            str(b.category) for b in scene.bodies
            if getattr(b, "category", None) and str(b.category) != "object"))
        self._spelling = {word.lower(): word for word in self.vocabulary}
        self.bodies = {b.name: b for b in scene.bodies}     # for ``resolve`` only
        self._robots = [(r, robots.model_of(r)) for r in scene.robots]
        self._bases = [robots.arm_base(r, a, scene.table_height)
                       for r, m in self._robots for a in m.arm_mounts]
        self._reach = self._robots[0][1].reach if self._robots else (0.2, 0.8, 0.4, 0.5)
        self._views: Dict[str, View] = {}
        #: a frame's detections and the phrases asked of it, kept as long as the
        #: episode: an id from an earlier picture stays good (the measured
        #: arrays of a view are heavy, and worked out again when needed)
        self._found: Dict[str, Tuple[List[Detection], set]] = {}

    # -- what the robot knows of its cameras -------------------------------------
    def _g(self, *key) -> float:
        return self.calib_scale * _gauss(self.seed, *key)

    def believed(self, frame: Frame) -> Pose:
        """Where the robot thinks the camera was: its calibration, off by the
        camera's own fixed error -- the synthetic backend's, draw for draw.  On
        a real robot the frame's pose is its calibration already."""
        if self.real_sensors:
            return frame.pose
        t = tuple(CAMERA_POS_M * self._g("camera", frame.camera, "t", i) for i in range(3))
        r = tuple(math.radians(CAMERA_ROT_DEG) * self._g("camera", frame.camera, "r", i)
                  for i in range(3))
        return compose(frame.pose, Pose(t, _rotvec(r)))

    def ray_at_height(self, frame: Frame, u: float, v: float, height: float):
        """Where the pixel's ray crosses the horizontal plane ``height`` metres
        above the table (world coordinates), by the robot's own geometry."""
        self._inside(frame, u, v)
        pose = self.believed(frame)
        k = frame.intrinsics()
        d = ((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], 1.0)
        n = math.sqrt(sum(c * c for c in d))
        d = rotate(pose.quat, tuple(c / n for c in d))
        z = self.station.table_height + height
        if abs(d[2]) < 1e-9 or (z - pose.pos[2]) / d[2] <= 0:
            raise ToolError("not_visible", f"the ray of pixel ({u:g}, {v:g}) never reaches "
                            f"{height:.3f} m above the table", hint="pick a lower height or "
                            "another pixel")
        t = (z - pose.pos[2]) / d[2]
        return tuple(o + t * c for o, c in zip(pose.pos, d))

    @staticmethod
    def _inside(frame: Frame, u: float, v: float) -> None:
        if not (0 <= u <= frame.width and 0 <= v <= frame.height):
            raise ToolError("not_visible", f"pixel ({u:g}, {v:g}) is outside the "
                            f"{frame.width}x{frame.height} image of {frame.id}")


    # -- the answers ------------------------------------------------------------------
    def detect(self, frame: Frame, query: str = "") -> List[dict]:
        query = " ".join(query.lower().split())
        self._ask(frame, [query] if query else self.vocabulary + (GENERIC,))
        words = [w for w in query.replace("-", " ").split() if w]
        out = []
        for d in sorted(self.view(frame).dets, key=lambda d: (d.centre[0], d.centre[1])):
            if query:
                text = f"{d.label} {d.colour or ''}".lower().replace("-", " ")
                if query not in d.scores and not all(w in text.split() for w in words):
                    continue
            out.append({**d.described(query, self._spelling), "box": list(d.box),
                        "center": list(d.centre)})
        return out

    def _segmented(self, frame: Frame, found: Optional[dict]) -> Optional[Detection]:
        np = _np()
        if not found:
            return None
        view = self.view(frame)
        mask = self._a_thing(view, np.asarray(found["mask"], dtype=bool))
        if mask is None:
            return None
        same = next((d for d in view.dets if _iou(d.mask, mask) > SAME_MASK_IOU), None)
        if same is not None:
            return same
        d = self._detection(frame, mask, {})
        d.id = f"{frame.id}#{len(view.dets) + 1}"
        view.dets.append(d)
        return d

    def segment_at(self, frame: Frame, u: float, v: float) -> Optional[dict]:
        self._inside(frame, u, v)
        view = self.view(frame)
        at = (min(int(v), view.depth.shape[0] - 1), min(int(u), view.depth.shape[1] - 1))
        inside = [d for d in view.dets if d.covers(*at)]
        d = min(inside, key=lambda d: d.area) if inside else self._segmented(
            frame, self._backend().segment(to_array(frame.image), point=(float(u), float(v))))
        if d is None or not d.covers(*at):
            return None
        return {**d.described(spelling=self._spelling), "box": list(d.box), "area_px": d.area}

    def segment_box(self, frame: Frame, box: Sequence[int]) -> Optional[dict]:
        d = self._segmented(frame, self._backend().segment(to_array(frame.image),
                                                           box=[float(b) for b in box]))
        if d is None:
            return None
        w = min(d.box[2], box[2]) - max(d.box[0], box[0])
        h = min(d.box[3], box[3]) - max(d.box[1], box[1])
        inter = max(0, w) * max(0, h)
        union = ((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                 + (box[2] - box[0]) * (box[3] - box[1]) - inter)
        return {**d.described(spelling=self._spelling), "box": list(d.box), "area_px": d.area,
                "iou": round(inter / union, 2) if union else 0.0}

    def lift(self, frame: Frame, u: float, v: float) -> dict:
        """The depth at a pixel, placed with the camera's calibration."""
        np = _np()
        self._inside(frame, u, v)
        view = self.view(frame)
        rows, cols = view.depth.shape
        j, i = min(int(v), rows - 1), min(int(u), cols - 1)
        z = view.depth[j, i]
        if not np.isfinite(z):
            near = view.depth[max(0, j - 2):j + 3, max(0, i - 2):i + 3]
            if not np.isfinite(near).any():
                raise ToolError("not_visible", f"no depth at pixel ({u:g}, {v:g}) of {frame.id}",
                                hint="a depth camera reads nothing along the edges of things "
                                     "and far away: pick a pixel inside a surface")
            z = float(np.nanmedian(near))
        k = frame.intrinsics()
        pose = self.believed(frame)
        local = ((u - k["cx"]) / k["fx"] * z, (v - k["cy"]) / k["fy"] * z, float(z))
        point = tuple(p + d for p, d in zip(pose.pos, rotate(pose.quat, local)))
        out = {"xyz_m": [round(c, 4) for c in self.station.to_station(point)],
               "normal": [round(c, 3) for c in _normal(view.points, j, i, pose.pos)]}
        a, b, c = view.plane
        over_table = point[2] - (a * point[0] + b * point[1] + c)
        if abs(over_table) < max(FLAT_M, 3.0 * view.sigma) and not view.robot[j, i]:
            out["on"] = "table"
        else:
            self._ask(frame, self.vocabulary + (GENERIC,))
            over = [d for d in view.dets if d.covers(j, i)]
            if over:
                label = min(over, key=lambda d: d.area).label
                out["on"] = self._spelling.get(label, label)
        return out

    def pose(self, frame: Frame, obj_id: str) -> dict:
        """A detection as a box standing on what is under it: the middle and
        heading of its footprint, its extent -- from the side the camera sees."""
        np = _np()
        d = self._by_id(frame, obj_id)
        view = self.view(frame)
        mask = d.mask
        measured = mask & np.isfinite(view.depth)
        inner = _erode(measured)
        if inner.sum() >= MIN_POINTS:
            measured = inner                                # edge pixels mix two depths
        pts, heights = view.points[measured], view.height[measured]
        top = float(np.percentile(heights, TOP_PCT))
        low = float(np.percentile(heights, LOW_PCT))
        ring = _dilate(mask, RING_PX) & ~mask & np.isfinite(view.depth) & ~view.robot
        bottom = low
        if ring.sum() >= MIN_POINTS:
            under = float(np.median(view.height[ring]))
            if low - under <= SUPPORT_BELOW_M:
                bottom = min(low, under)
        bottom = max(0.0, min(bottom, top - 1e-3))
        (cx, cy), (along, across), yaw = _footprint(pts[:, :2])
        a, b, c = view.plane
        mid = a * cx + b * cy + c + (top + bottom) / 2
        tilt, toward = _tilt(pts, heights, top, bottom, view)
        out = {**d.described(spelling=self._spelling),
               "xyz_m": [round(v, 4) for v in self.station.to_station((cx, cy, mid))],
               "yaw_deg": round(math.degrees(yaw), 1), "tilt_deg": round(tilt, 1),
               "extent_m": [round(along, 4), round(across, 4), round(top - bottom, 4)]}
        if toward is not None:
            out["tilt_toward_xy"] = [round(toward[0], 2), round(toward[1], 2)]
        return out


