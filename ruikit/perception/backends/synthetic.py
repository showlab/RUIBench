"""The ``synthetic`` perception backend.

Answers the ``vision_*`` tools from the simulator's own bookkeeping instead of
a learned model -- cheap, deterministic, the same for every interface -- under
the rules that stop it from knowing more than a camera could:

* **only what the frame shows.**  An object counts only if the engine's
  visibility pass put it in that frame (Isaac: its own segmentation pixels;
  mock: sight lines).  Under a flower pot, behind a box, out of the picture:
  not reported, not locatable, not posable.
* **ids are frame-local.**  ``f12.front#3`` is "the third thing in frame 12,
  counting left to right".  The next frame numbers its objects afresh, so the
  identity of two look-alike domes across a spin is something the agent has to
  follow with its own eyes (DI6), not read off a stable id.
* **labels are public category words** (``pot``, ``coin``, ``doll``) and a
  plain colour when the body has one.  Internal names (``card_tom``,
  ``splint_cucl2``) never leave the server.
* **nothing is exact, and no tool is more exact than another.**  Every
  answer -- boxes, depth, poses, sizes, the camera calibration -- comes from
  one perceived copy of the world: each object sits off by its own fixed error
  (place, heading, tilt, size) and each camera's calibration by its own, like
  a real robot's.  Tools agree with each other, so checking one against another
  cannot get back to the truth (depth at 1 mm and exact boxes had let the
  agents grasp within 1.3 mm of the true centres).
* **noise is a function of the state**, never re-drawn: the fixed errors plus
  a part hashed from the view (and per pixel for depth).  Asking a hundred
  times returns the same answer a hundred times, so averaging buys nothing.

The backend still leans on simulator knowledge for *what* a thing is, so it
stands in only on the mock engine, which renders no pictures; wherever
pictures are rendered, the ``model`` backend (SAM 3, or a box detector with
SAM 2.1) answers the same tools from pixels alone.  The ``privileged``
backend is this one without the errors.
"""

from __future__ import annotations

import math
import re
from typing import List, Optional, Sequence, Tuple

from ruibench.core.geometry import (
    compose,
    quat_conj,
    quat_mul,
    relative_pose,
    rotate,
    yaw_of,
)
from ruibench.core.types import Pose
from ruibench.engines.base import collides, project
from ruibench.engines.sight import box_bounds, occluder_boxes
from ruikit.core.result import ToolError
from ruikit.env.frames import Frame
from ruikit.perception.backends.base import PerceptionBackend
from ruikit.perception.backends.noise import (
    CAMERA_POS_M,
    CAMERA_ROT_DEG,
    DEPTH_FRAC,
    DEPTH_M,
    _gauss,
    _rotvec,
    _uniform,
)

_SIMPLE_COLOR = re.compile(r"^[a-z][a-z ]{1,20}$")

#: How far off the robot's perception is: a real robot's, not the simulator's.
#: Each object's own fixed error (standard deviations, per axis):
OBJECT_POS_M = (0.005, 0.005, 0.003)
OBJECT_YAW_DEG = 3.0
OBJECT_TILT_DEG = 2.0
OBJECT_SIZE_FRAC = 0.05
#: what changes with the viewpoint (vision_pose):
VIEW_POS_M = 0.002
VIEW_YAW_DEG = 1.0
#: each edge of a detection box, per frame (pixels)
BOX_PX = 2.0


class SyntheticPerception(PerceptionBackend):
    def __init__(self, scene, station, seed: int, params: Optional[dict] = None) -> None:
        self.station = station
        self.seed = int(seed)
        # an interface may make perception worse, never better than a real robot's
        self.scale = max(1.0, float((params or {}).get("error_scale", 1.0)))
        self.bodies = {b.name: b for b in scene.bodies}

    # -- the perceived world: where the robot thinks things are -------------------
    def _g(self, *key) -> float:
        return self.scale * _gauss(self.seed, *key)

    def seen(self, name: str, pose: Pose) -> Pose:
        """Where perception puts this object: the true pose moved by the object's
        own fixed error in place, heading and tilt -- the same for every tool and
        every camera, so no tool can check another back to the truth."""
        pos = tuple(p + sd * self._g("object", name, "pos", i)
                    for i, (p, sd) in enumerate(zip(pose.pos, OBJECT_POS_M)))
        yaw = math.radians(OBJECT_YAW_DEG) * self._g("object", name, "yaw")
        tilt = math.radians(OBJECT_TILT_DEG) * self._g("object", name, "tilt")
        axis = 2 * math.pi * _uniform(self.seed, "object", name, "tilt axis")
        turn = quat_mul(_rotvec((0.0, 0.0, yaw)),
                        _rotvec((tilt * math.cos(axis), tilt * math.sin(axis), 0.0)))
        return Pose(pos, quat_mul(turn, pose.quat))

    def size(self, name: str, obj=None) -> tuple:
        """The object's half extents as perception sees them (each off by a few %).
        A cloth's are those of the box round it as it lies now (``obj``: its
        state), not of the cloth laid flat."""
        true = obj.size if obj is not None and obj.points is not None else self.bodies[name].size
        return tuple(v * (1.0 + OBJECT_SIZE_FRAC * self._g("object", name, "size", i))
                     for i, v in enumerate(true))

    def believed(self, frame: Frame) -> Pose:
        """Where the robot thinks the camera was for this frame: its calibration,
        off by the camera's own fixed error -- never the simulator's pose."""
        t = tuple(CAMERA_POS_M * self._g("camera", frame.camera, "t", i) for i in range(3))
        r = tuple(math.radians(CAMERA_ROT_DEG) * self._g("camera", frame.camera, "r", i)
                  for i in range(3))
        return compose(frame.pose, Pose(t, _rotvec(r)))

    def _as_believed(self, frame: Frame, point) -> tuple:
        """A point measured through this frame's camera, where the robot puts it."""
        local = relative_pose(frame.pose, Pose(tuple(point), (1.0, 0.0, 0.0, 0.0))).pos
        return compose(self.believed(frame), Pose(local, (1.0, 0.0, 0.0, 0.0))).pos

    def _turn_believed(self, frame: Frame) -> tuple:
        return quat_mul(self.believed(frame).quat, quat_conj(frame.pose.quat))

    # -- geometry of one frame ------------------------------------------------
    def _boxes(self, frame: Frame, name: str):
        body = self.bodies.get(name)
        obj = frame.objects.get(name)
        if body is None or obj is None:
            return []
        pose, size = self.seen(name, obj.pose), self.size(name, obj)
        try:
            return occluder_boxes(body.shape, size, pose, body.attrs or {})
        except Exception:                   # a shape the sight module does not model
            return [box_bounds(body.shape, size, pose)]

    def _project(self, frame: Frame, point):
        return project(frame.pose, point, frame.width, frame.height, frame.focal_mm,
                       frame.apertures)

    def _pixel_box(self, frame: Frame, name: str):
        us, vs = [], []
        for (x0, x1), (y0, y1), (z0, z1) in self._boxes(frame, name):
            for x in (x0, x1):
                for y in (y0, y1):
                    for z in (z0, z1):
                        uv = self._project(frame, (x, y, z))
                        if uv is not None:
                            us.append(uv[0])
                            vs.append(uv[1])
        if not us:
            return None
        j = [BOX_PX * self._g("box", frame.id, name, i) for i in range(4)]
        u0, u1 = max(0.0, min(us) + j[0]), min(float(frame.width), max(us) + j[2])
        v0, v1 = max(0.0, min(vs) + j[1]), min(float(frame.height), max(vs) + j[3])
        if u1 <= u0 + 1 or v1 <= v0 + 1:
            return None
        return (int(u0), int(v0), int(math.ceil(u1)), int(math.ceil(v1)))

    def label(self, name: str) -> dict:
        body = self.bodies.get(name)
        out = {"label": str(getattr(body, "category", "object") or "object")}
        color = getattr(body, "color", None)
        if isinstance(color, str) and _SIMPLE_COLOR.match(color) and color not in out["label"]:
            out["color"] = color
        return out

    def _ensure_ids(self, frame: Frame) -> List[Tuple[str, str, Tuple[int, int, int, int]]]:
        """Number the frame's visible bodies left to right (then top to bottom)."""
        rows = []
        for name in frame.visible:
            if name not in self.bodies:
                continue
            box = self._pixel_box(frame, name)
            if box is None:
                continue
            rows.append((((box[0] + box[2]) / 2, (box[1] + box[3]) / 2), name, box))
        rows.sort(key=lambda r: (round(r[0][0]), round(r[0][1]), r[1]))
        if not frame.ids:
            for k, (_, name, _) in enumerate(rows, start=1):
                frame.ids[f"{frame.id}#{k}"] = name
        by_name = {v: k for k, v in frame.ids.items()}
        return [(by_name[name], name, box) for _, name, box in rows if name in by_name]

    def mask_of(self, frame: Frame, obj_id: str):
        """No pixels on the mock: the test double has boxes only."""
        return None

    def box_of(self, frame: Frame, obj_id: str):
        for _, oid, box in self._ensure_ids(frame):
            if oid == obj_id:
                return list(box)
        raise ToolError("not_visible", f"frame {frame.id} has no object {obj_id!r}")

    def points_of(self, frame: Frame, obj_id: str):
        """The test double's points of a thing: its perceived box's top face and
        sides, sampled (the mock renders no depth)."""
        import numpy as np

        pose = self.pose(frame, obj_id)
        x, y, z = self.station.to_world(pose["xyz_m"])
        L, W, H = pose["extent_m"]
        yaw = math.radians(pose["yaw_deg"])
        c, s = math.cos(yaw), math.sin(yaw)
        pts = []
        for i in np.linspace(-0.5, 0.5, 9):
            for j in np.linspace(-0.5, 0.5, 9):
                for k in (0.5, 0.25, 0.0, -0.25):
                    if k != 0.5 and abs(i) < 0.5 and abs(j) < 0.5:
                        continue                       # inside: only the top and the sides
                    pts.append((x + c * i * L - s * j * W, y + s * i * L + c * j * W, z + k * H))
        pts = np.asarray(pts, dtype=np.float64)
        return pts, pts[:, 2] - self.station.table_height

    def scene_points(self, frame: Frame):
        import numpy as np

        out = [self.points_of(frame, oid)[0] for _, oid, _ in self._ensure_ids(frame)]
        pts = np.concatenate(out) if out else np.zeros((0, 3))
        return pts, pts[:, 2] - self.station.table_height

    def resolve(self, frame: Frame, obj_id: str) -> str:
        prefix, _, k = obj_id.partition("#")
        if prefix == "latest" or prefix == "latest." + frame.id.split(".", 1)[-1]:
            obj_id = f"{frame.id}#{k}"                # 'latest.wrist#2' in the frame it names
        if not obj_id.startswith(frame.id + "#"):
            raise ToolError("invalid_args", f"{obj_id!r} is not an object of frame {frame.id}",
                            hint="object ids are frame-local: detect in this frame first")
        self._ensure_ids(frame)
        name = frame.ids.get(obj_id)
        if name is None:
            raise ToolError("not_visible", f"frame {frame.id} has no object {obj_id!r}",
                            hint="use an id returned by vision_detect or vision_segment")
        return name

    # -- the four answers -----------------------------------------------------
    def detect(self, frame: Frame, query: str = "") -> List[dict]:
        words = [w for w in re.split(r"[^a-z0-9]+", query.lower()) if w]
        out = []
        for obj_id, name, box in self._ensure_ids(frame):
            lab = self.label(name)
            text = f"{lab['label']} {lab.get('color', '')}".lower()
            if words and not all(w in text for w in words):
                continue
            out.append({"id": obj_id, **lab, "box": list(box),
                        "center": [(box[0] + box[2]) // 2, (box[1] + box[3]) // 2]})
        return out

    def ray(self, frame: Frame, u: float, v: float, pose: Optional[Pose] = None):
        """The pixel's ray from the camera's true pose (what the sensor measures
        along), or from ``pose`` (the robot's belief, for pointing into the air)."""
        pose = pose or frame.pose
        k = frame.intrinsics()
        d = ((u - k["cx"]) / k["fx"], (v - k["cy"]) / k["fy"], 1.0)
        n = math.sqrt(sum(c * c for c in d))
        return pose.pos, rotate(pose.quat, tuple(c / n for c in d))

    def cast(self, frame: Frame, u: float, v: float):
        """Nearest surface along the pixel's ray: (distance, point, normal, body or None)."""
        if not (0 <= u <= frame.width and 0 <= v <= frame.height):
            raise ToolError("not_visible", f"pixel ({u:g}, {v:g}) is outside the "
                            f"{frame.width}x{frame.height} image of {frame.id}")
        origin, d = self.ray(frame, u, v)
        best = None
        for name, body in self.bodies.items():
            if name not in frame.objects or not collides(body):
                continue
            for box in self._boxes(frame, name):
                hit = _slab(origin, d, box)
                if hit is not None and (best is None or hit[0] < best[0]):
                    best = (hit[0], hit[1], name)
        table = self.station.table_height
        if d[2] < -1e-9:
            t = (table - origin[2]) / d[2]
            if t > 0 and (best is None or t < best[0]):
                best = (t, (0.0, 0.0, 1.0), None)
        if best is None:
            raise ToolError("not_visible", f"nothing at pixel ({u:g}, {v:g}) of {frame.id}")
        t, normal, name = best
        point = tuple(o + t * c for o, c in zip(origin, d))
        return t, point, normal, name

    def ray_at_height(self, frame: Frame, u: float, v: float, height: float):
        """Where the pixel's ray crosses the horizontal plane ``height`` metres
        above the table (world coordinates), or a not_visible error."""
        if not (0 <= u <= frame.width and 0 <= v <= frame.height):
            raise ToolError("not_visible", f"pixel ({u:g}, {v:g}) is outside the "
                            f"{frame.width}x{frame.height} image of {frame.id}")
        origin, d = self.ray(frame, u, v, self.believed(frame))     # the robot's own geometry
        z = self.station.table_height + height
        if abs(d[2]) < 1e-9 or (z - origin[2]) / d[2] <= 0:
            raise ToolError("not_visible", f"the ray of pixel ({u:g}, {v:g}) never reaches "
                            f"{height:.3f} m above the table", hint="pick a lower height or "
                            "another pixel")
        t = (z - origin[2]) / d[2]
        return tuple(o + t * c for o, c in zip(origin, d))

    def lift(self, frame: Frame, u: float, v: float) -> dict:
        """Depth at a pixel: the distance along its ray to the perceived surface,
        with the sensor's per-pixel noise, placed with the camera's calibration."""
        t, _, normal, name = self.cast(frame, u, v)
        origin, d = self.ray(frame, u, v)
        t += (DEPTH_M + DEPTH_FRAC * t) * self._g("depth", frame.id, round(u), round(v))
        point = self._as_believed(frame, tuple(o + t * c for o, c in zip(origin, d)))
        normal = rotate(self._turn_believed(frame), normal)
        out = {"xyz_m": [round(c, 4) for c in self.station.to_station(point)],
               "normal": [round(c, 3) for c in normal]}
        if name is not None and name in frame.visible:
            out["on"] = self.label(name)["label"]
        elif name is None:
            out["on"] = "table"
        return out

    def segment_at(self, frame: Frame, u: float, v: float) -> Optional[dict]:
        _, _, _, name = self.cast(frame, u, v)
        if name is None or name not in frame.visible:
            return None
        for obj_id, n, box in self._ensure_ids(frame):
            if n == name:
                return {"id": obj_id, **self.label(name), "box": list(box),
                        "area_px": (box[2] - box[0]) * (box[3] - box[1])}
        return None

    def segment_box(self, frame: Frame, box: Sequence[int]) -> Optional[dict]:
        best, best_iou = None, 0.0
        for obj_id, name, b in self._ensure_ids(frame):
            iou = _iou(b, box)
            if iou > best_iou:
                best, best_iou = (obj_id, name, b), iou
        if best is None or best_iou < 0.1:
            return None
        obj_id, name, b = best
        return {"id": obj_id, **self.label(name), "box": list(b),
                "area_px": (b[2] - b[0]) * (b[3] - b[1]), "iou": round(best_iou, 2)}

    def pose(self, frame: Frame, obj_id: str) -> dict:
        """The perceived pose, a little different from each viewpoint, placed with
        the camera's calibration."""
        name = self.resolve(frame, obj_id)
        obj = frame.objects[name]
        seen = self.seen(name, obj.pose)
        key = ("view", name, tuple(round(p, 3) for p in obj.pose.pos),
               tuple(round(p, 2) for p in frame.pose.pos))
        pos = self._as_believed(frame, tuple(p + VIEW_POS_M * self._g(*key, i)
                                             for i, p in enumerate(seen.pos)))
        quat = quat_mul(self._turn_believed(frame), seen.quat)
        yaw = math.degrees(yaw_of(quat)) + VIEW_YAW_DEG * self._g(*key, "yaw")
        up = rotate(quat, (0.0, 0.0, 1.0))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, up[2]))))
        out = {"id": obj_id, **self.label(name),
               "xyz_m": [round(c, 4) for c in self.station.to_station(pos)],
               "yaw_deg": round((yaw + 180.0) % 360.0 - 180.0, 1),
               "tilt_deg": round(tilt, 1),
               "extent_m": [round(2 * v, 4) for v in self.size(name, obj)]}
        if tilt >= 1.0:
            # which way the top leans: a beam's top leans toward the end that went down
            a = self.station.to_station(pos)
            b = self.station.to_station(tuple(p + u for p, u in zip(pos, up)))
            dx, dy = b[0] - a[0], b[1] - a[1]
            n = math.hypot(dx, dy) or 1.0
            out["tilt_toward_xy"] = [round(dx / n, 2), round(dy / n, 2)]
        return out


def _slab(origin, d, box):
    """Ray/axis-aligned-box intersection: (t, outward normal) of the entry face."""
    t0, t1, normal = 0.0, float("inf"), None
    for i, (lo, hi) in enumerate(box):
        if abs(d[i]) < 1e-12:
            if origin[i] < lo or origin[i] > hi:
                return None
            continue
        a, b = (lo - origin[i]) / d[i], (hi - origin[i]) / d[i]
        n = [0.0, 0.0, 0.0]
        n[i] = -1.0 if d[i] > 0 else 1.0
        if a > b:
            a, b = b, a
        if a > t0:
            t0, normal = a, tuple(n)
        t1 = min(t1, b)
        if t0 > t1:
            return None
    if normal is None:              # the camera is inside the box
        return None
    return t0, normal


def _iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area if area > 0 else 0.0
