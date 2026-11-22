"""A mock engine with a camera, for testing perception from pixels without
Isaac: each body is painted one flat colour and the depth image is ray-cast
from the bodies' boxes, the table top and the open hand.  ``ColourVision``
stands in for the detector and the segmenter: a thing is the pixels of its
colour, and its word is its category.  So what these tests check is the half
that is ours -- depth to positions, the filters, the plumbing -- not the models.
"""
import dataclasses

import numpy as np

from ruibench.core.geometry import rotate
from ruibench.core.types import CameraFrame
from ruibench.engines.base import camera_apertures, camera_focal_length, collides
from ruibench.engines.mock import MockEngine
from ruibench.engines.robots import model_of
from ruibench.engines.sight import occluder_boxes

TABLE, ROBOT = (150, 120, 90), (200, 200, 205)


def colour_of(k: int):
    """A distinct colour for the k-th body (never the table's or the robot's)."""
    return (40 + (k * 67) % 200, 30 + (k * 131) % 210, 20 + (k * 29) % 220)


class PixelMock(MockEngine):
    capabilities = dataclasses.replace(MockEngine.capabilities, rendering=True)

    def render(self, camera: str) -> CameraFrame:
        spec = next(c for c in self._scene.cameras if c.name == camera)
        w, h = spec.width, spec.height
        pose = self.camera_pose(camera)
        focal = camera_focal_length(self._scene, spec)
        across, down = camera_apertures(self._scene, spec)
        fx = w * focal / across
        fy = h * focal / down if down else fx
        us = (np.arange(w) + 0.5 - w / 2) / fx
        vs = (np.arange(h) + 0.5 - h / 2) / fy
        cam = np.stack([np.broadcast_to(us[None, :], (h, w)), np.broadcast_to(vs[:, None], (h, w)),
                        np.ones((h, w))], axis=-1)
        norm = np.linalg.norm(cam, axis=-1, keepdims=True)
        R = np.array([rotate(pose.quat, e) for e in ((1.0, 0, 0), (0, 1.0, 0), (0, 0, 1.0))]).T
        d = (cam / norm) @ R.T
        o = np.asarray(pose.pos)
        best = np.full((h, w), np.inf)
        image = np.zeros((h, w, 3), dtype=np.uint8)
        image[:] = (60, 60, 70)                              # the room

        def paint(boxes, colour):
            nonlocal best
            for box in boxes:
                t0 = np.zeros((h, w))
                t1 = np.full((h, w), np.inf)
                for i, (lo, hi) in enumerate(box):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        a = (lo - o[i]) / d[..., i]
                        b = (hi - o[i]) / d[..., i]
                    near, far = np.minimum(a, b), np.maximum(a, b)
                    t0, t1 = np.maximum(t0, near), np.minimum(t1, far)
                hit = (t0 <= t1) & (t0 > 1e-6) & (t0 < best)
                best = np.where(hit, t0, best)
                image[hit] = colour

        top = self._scene.table_height
        paint([((-2.0, 2.0), (-2.0, 2.0), (top - 0.05, top))], TABLE)
        world = self.state()
        for k, body in enumerate(self._scene.bodies):
            if not collides(body) or body.name not in world.objects:
                continue
            paint(occluder_boxes(body.shape, body.size, world.objects[body.name].pose,
                                 body.attrs or {}), colour_of(k))
        for robot in self._scene.robots:                     # the open hand, pointing down
            model = model_of(robot)
            for arm in robot.arms:
                tcp = world.arm(f"{robot.name}/{arm}").eef_pose.pos
                paint([((tcp[0] + p.x[0], tcp[0] + p.x[1]), (tcp[1] + p.y[0], tcp[1] + p.y[1]),
                        (tcp[2] + p.bottom, tcp[2] + p.bottom + p.height))
                       for p in model.open_hand], ROBOT)
        #: the true depth of the last picture, for test doubles of a monocular model
        self.last_depth = np.where(np.isfinite(best), best / norm[..., 0], np.nan)
        depth = np.where(np.isfinite(best), best / norm[..., 0], np.inf) if spec.depth else None
        return CameraFrame(camera=camera, width=w, height=h, captured_at=self.sim_time,
                           image=image, depth=depth)


class ColourVision:
    """The detector and the segmenter of the painted world."""

    name = "colour"

    def __init__(self, scene) -> None:
        self.words = {colour_of(k): str(b.category) for k, b in enumerate(scene.bodies)}
        self.calls = []

    def info(self) -> dict:
        return {"backend": self.name, "detector": "flat colours", "segmenter": "flat colours"}

    def _masks(self, image):
        flat = image.reshape(-1, 3)
        for colour in {tuple(int(v) for v in c) for c in np.unique(flat, axis=0)}:
            yield colour, (image == np.array(colour, dtype=np.uint8)).all(axis=-1)

    def detect(self, image, phrases):
        self.calls.append(tuple(phrases))
        out = []
        for colour, mask in self._masks(image):
            word = "robot" if colour == ROBOT else "table" if colour == TABLE else self.words.get(colour)
            for phrase in phrases:
                if word and (phrase == word.lower() or phrase in word.lower().split()):
                    vs, us = np.nonzero(mask)
                    out.append({"phrase": phrase, "score": 0.9, "mask": mask, "mask_score": 1.0,
                                "box": [int(us.min()), int(vs.min()), int(us.max()) + 1,
                                        int(vs.max()) + 1]})
        return out

    def segment(self, image, point=None, box=None):
        if point is None:
            point = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        colour = image[int(point[1]), int(point[0])]
        mask = (image == colour).all(axis=-1)
        vs, us = np.nonzero(mask)
        return {"mask": mask, "score": 1.0,
                "box": [int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1]}


def episode(task: str, level: str = "easy", seed: int = 0, profile: str = "rui.pose/v1", **kw):
    """An episode on the painted mock, answered from pixels by ``ColourVision``."""
    from ruikit import Episode
    from ruikit.extras.backends import VISION
    from ruikit.interfaces import load_profile

    box = {}

    def make():
        return box["vision"]

    prof = load_profile(profile)
    settings = {**dict(prof.params.get("perception", {})), "backend": "colour-test"}
    prof = dataclasses.replace(prof, perception="model",
                               params={**prof.params, "perception": settings})
    from ruibench.core.config import RunConfig
    from ruibench.tasks.instance import build

    vision = kw.pop("vision", ColourVision)
    engine = kw.pop("engine", None) or PixelMock()
    box["vision"] = vision(build(task, level=level, seed=seed, config=RunConfig()).scene)
    box["vision"].engine = engine
    VISION["colour-test"] = make
    try:
        ep = Episode(task, level=level, seed=seed, profile=prof, engine=engine, **kw)
    finally:
        VISION.pop("colour-test", None)
    ep.vision = box["vision"]
    return ep
