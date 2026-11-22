"""Detection: asking the vision service about a frame and naming what it found."""

from __future__ import annotations

from typing import Dict, Sequence

from ruikit.core.result import ToolError
from ruikit.env.frames import Frame, to_array
from ruikit.perception.backends.geometry import (
    _colour,
    _iou,
)
from ruikit.perception.backends.pixels_types import (
    FLAT_M,
    GENERIC,
    MIN_POINTS,
    SAME_MASK_IOU,
    TOP_PCT,
    Detection,
    View,
    _np,
)

#: a mask lies below the table (the floor, a wall behind) when its middle is this
#: far under it
BELOW_M = 0.01


#: a detection is the robot when this much of it is
SELF_FRACTION = 0.5


#: what the picture's edge cuts down to a sliver narrower than this (pixels)
#: is not reported: there is nothing of it to measure
SLIVER_PX = 12


class DetectionLookup:
    """Detections: asking the service, and finding a thing again by its frame-local id."""

    # -- detections ----------------------------------------------------------------
    def _backend(self):
        if self.backend is None:
            from ruikit.extras.backends import VISION, get_backend
            from ruikit.perception.service.client import DEFAULT_BACKEND

            self.backend = get_backend(VISION, "vision", DEFAULT_BACKEND)
        return self.backend

    def _ask(self, frame: Frame, phrases: Sequence[str]) -> None:
        """Run the detector for ``phrases`` on this frame (once each) and file
        what survives among the frame's detections."""
        np = _np()
        view = self.view(frame)
        todo = [p for p in dict.fromkeys(p.strip().lower() for p in phrases) if p
                and p not in view.asked]
        if not todo:
            return
        image = to_array(frame.image)
        if image is None:
            raise ToolError("not_supported", f"frame {frame.id} has no pixels left",
                            hint="take a new picture with camera_look")
        answers = self._backend().detect(image, todo)
        view.asked.update(todo)
        fresh = []
        for found in answers:
            mask = np.asarray(found["mask"], dtype=bool)
            kept = self._a_thing(view, mask)
            if kept is None:
                continue
            same = next((d for d in view.dets if _iou(d.mask, kept) > SAME_MASK_IOU), None)
            if same is None:
                same = next((d for d in fresh if _iou(d.mask, kept) > SAME_MASK_IOU), None)
            if same is not None:
                same.scores[found["phrase"]] = max(same.scores.get(found["phrase"], 0.0),
                                                   float(found["score"]))
                continue
            fresh.append(self._detection(frame, kept, {found["phrase"]: float(found["score"])}))
        # numbered left to right, after the ones this frame already has
        fresh.sort(key=lambda d: (d.centre[0], d.centre[1]))
        for d in fresh:
            d.id = f"{frame.id}#{len(view.dets) + 1}"
            view.dets.append(d)

    def _a_thing(self, view: View, mask):
        """The mask, without the robot's own pixels -- or None where depth says
        it is no thing on the table."""
        np = _np()
        if mask.shape != view.depth.shape:
            return None
        measured = mask & np.isfinite(view.depth)
        n = int(measured.sum())
        if n < MIN_POINTS:
            return None
        own = measured & view.robot
        if own.sum() >= SELF_FRACTION * n:
            return None
        mask = mask & ~view.robot
        heights = view.height[measured & ~view.robot]
        if heights.size < MIN_POINTS:
            return None
        if float(np.median(heights)) < -BELOW_M:
            return None                                     # the floor, a wall behind
        if float(np.percentile(heights, TOP_PCT)) < max(FLAT_M, 3.0 * view.sigma):
            return None                                     # the table top
        vs, us = np.nonzero(mask)
        rows, cols = mask.shape
        if ((us.min() <= 0 or us.max() >= cols - 1 or vs.min() <= 0 or vs.max() >= rows - 1)
                and min(us.max() - us.min(), vs.max() - vs.min()) + 1 < SLIVER_PX):
            return None                                     # a sliver at the picture's edge
        return mask

    def _detection(self, frame: Frame, mask, scores: Dict[str, float]) -> Detection:
        np = _np()
        vs, us = np.nonzero(mask)
        box = (int(us.min()), int(vs.min()), int(us.max()) + 1, int(vs.max()) + 1)
        centre = (int(round(float(us.mean()))), int(round(float(vs.mean()))))
        if not mask[min(centre[1], mask.shape[0] - 1), min(centre[0], mask.shape[1] - 1)]:
            k = int(np.argmin((us - centre[0]) ** 2 + (vs - centre[1]) ** 2))
            centre = (int(us[k]), int(vs[k]))                # a ring's middle is not the ring
        rows, cols = mask.shape
        cut = box[0] <= 0 or box[1] <= 0 or box[2] >= cols or box[3] >= rows
        return Detection(id="", crop=mask[box[1]:box[3], box[0]:box[2]].copy(),
                         shape=(rows, cols), box=box, scores=scores, centre=centre,
                         area=int(mask.sum()), colour=_colour(to_array(frame.image), mask),
                         cut=cut)

    def _by_id(self, frame: Frame, obj_id: str) -> Detection:
        prefix, _, k = obj_id.partition("#")
        if prefix == "latest" or prefix == "latest." + frame.id.split(".", 1)[-1]:
            obj_id = f"{frame.id}#{k}"                # 'latest.wrist#2' in the frame it names
        if not obj_id.startswith(frame.id + "#"):
            raise ToolError("invalid_args", f"{obj_id!r} is not an object of frame {frame.id}",
                            hint="object ids are frame-local: detect in this frame first")
        found = next((d for d in self.view(frame).dets if d.id == obj_id), None)
        if found is None:
            raise ToolError("not_visible", f"frame {frame.id} has no object {obj_id!r}",
                            hint="use an id returned by vision_detect or vision_segment")
        return found

    # -- for the Harness components, and for scripted agents and scoring ----------
    def mask_of(self, frame: Frame, obj_id: str):
        """The pixels of a thing found in this frame (bool HxW)."""
        return self._by_id(frame, obj_id).mask

    def box_of(self, frame: Frame, obj_id: str):
        """Its pixel box [u0, v0, u1, v1]."""
        return list(self._by_id(frame, obj_id).box)

    def points_of(self, frame: Frame, obj_id: str):
        """The measured points of a thing found in this frame: (N x 3 world
        coordinates, their heights over the table) -- what the depth camera
        saw of it, by the robot's calibration."""
        np = _np()
        view = self.view(frame)
        mask = self._by_id(frame, obj_id).mask & np.isfinite(view.depth) & ~view.robot
        return view.points[mask].astype(np.float64), view.height[mask].astype(np.float64)

    def scene_points(self, frame: Frame):
        """Everything the frame's depth saw standing on the table -- not the
        table top, not the robot itself: (N x 3 world points, heights)."""
        np = _np()
        view = self.view(frame)
        keep = np.isfinite(view.depth) & ~view.robot & (view.height > max(FLAT_M, 3.0 * view.sigma))
        return view.points[keep].astype(np.float64), view.height[keep].astype(np.float64)

    def resolve(self, frame: Frame, obj_id: str) -> str:
        """The body a detection is, by where the simulator has the bodies --
        what a scripted agent needs to follow its script, and what scoring the
        detections needs.  No tool calls this."""
        from ruibench.engines.base import project
        from ruibench.engines.sight import box_bounds

        d = self._by_id(frame, obj_id)
        best, best_iou = None, 0.0
        for name in frame.visible:
            body, state = self.bodies.get(name), frame.objects.get(name)
            if body is None or state is None:
                continue
            us, vs = [], []
            # (a cloth's box is the one round it as it lies now)
            size = state.size if state.points is not None else body.size
            (x0, x1), (y0, y1), (z0, z1) = box_bounds(body.shape, size, state.pose)
            for x in (x0, x1):
                for y in (y0, y1):
                    for z in (z0, z1):
                        uv = project(frame.pose, (x, y, z), frame.width, frame.height,
                                     frame.focal_mm, frame.apertures)
                        if uv is not None:
                            us.append(uv[0])
                            vs.append(uv[1])
            if not us:
                continue
            box = (min(us), min(vs), max(us), max(vs))
            w = min(d.box[2], box[2]) - max(d.box[0], box[0])
            h = min(d.box[3], box[3]) - max(d.box[1], box[1])
            inter = max(0.0, w) * max(0.0, h)
            union = ((d.box[2] - d.box[0]) * (d.box[3] - d.box[1])
                     + (box[2] - box[0]) * (box[3] - box[1]) - inter)
            iou = inter / union if union else 0.0
            if iou > best_iou:
                best, best_iou = name, iou
        if best is None or best_iou < 0.25:
            raise ToolError("not_visible", f"{obj_id} is no body of the scene")
        return best

    def label(self, name: str) -> dict:
        body = self.bodies.get(name)
        return {"label": str(getattr(body, "category", "object") or "object")}

    def _ensure_ids(self, frame: Frame):
        """(id, body, box) of this frame's detections that are bodies (``resolve``)."""
        self._ask(frame, self.vocabulary + (GENERIC,))
        out = []
        for d in self.view(frame).dets:
            try:
                out.append((d.id, self.resolve(frame, d.id), d.box))
            except ToolError:
                continue
        return out
