"""Depth from the camera: a measured view of a frame, from the depth sensor or predicted
from the picture alone."""

from __future__ import annotations

import hashlib

from ruikit.core.result import ToolError
from ruikit.env.frames import Frame, to_array
from ruikit.perception.backends.geometry import (
    TABLE_BAND_M,
    _fit_table,
    _median,
)
from ruikit.perception.backends.noise import DEPTH_FRAC, DEPTH_M
from ruikit.perception.backends.pixels_types import (
    View,
    _np,
    _rot,
)

#: depth further than this is the room, not the table (the cameras clip at 5 m)
MAX_DEPTH_M = 4.5


#: a jump in depth between neighbours larger than this (plus 2 % of the
#: distance) leaves no reading on either side: a stereo camera's holes along
#: the edges of things
EDGE_JUMP_M = 0.02


EDGE_JUMP_FRAC = 0.02


#: the depth image is smoothed by the median of this many pixels a side before
#: anything is measured in it, as depth pipelines do
SMOOTH = 5


#: nothing on the table reaches this high: above it is the arm
ARM_ABOVE_M = 0.35


#: the arm's base, as a column of this radius
BASE_RADIUS_M = 0.13


#: the hand's parts, taken this much larger
SELF_MARGIN_M = 0.012


#: monocular depth: the table top is the pixels within this share of the depth
#: the robot's calibration expects of it, once the scale is roughly right
MONO_TABLE_SHARE = 0.03


#: ... and the scale is fitted only when this many such pixels are seen
MONO_MIN_TABLE_PX = 500


#: the table round the arms' reach that the fit looks at (metres)
MONO_REACH_MARGIN = 0.10


#: with monocular depth a thing is told from the table top by standing at least
#: this far over it, however rough the predicted table is
MONO_FLAT_MAX_M = 0.01


class DepthMeasurement:
    """Measuring a frame's depth: the points, the table plane and the robot's own body."""

    # -- one frame, measured --------------------------------------------------------
    def view(self, frame: Frame) -> View:
        hit = self._views.get(frame.id)
        if hit is not None:
            return hit
        np = _np()
        if self.depth_source == "mono":
            return self._mono_view(frame)
        if frame.depth is None:
            raise ToolError("not_supported", f"frame {frame.id} has no depth image: perception "
                            "from pixels needs an RGB-D camera", hint="run on Isaac with "
                            "perception: model (the mock renders nothing)")
        true = np.asarray(frame.depth, dtype=np.float64)
        height_px, width_px = true.shape
        # the sensor: noise fixed for a viewpoint, holes where depth jumps
        digest = hashlib.sha256(repr((self.seed, "depth", frame.camera,
                                      tuple(round(p, 2) for p in frame.pose.pos))).encode())
        rng = np.random.default_rng(int.from_bytes(digest.digest()[:8], "little"))
        noise = rng.standard_normal(true.shape)
        ok = np.isfinite(true) & (true > 0) & (true < MAX_DEPTH_M)
        z = np.where(ok, true, np.nan)
        jump = EDGE_JUMP_M + EDGE_JUMP_FRAC * np.nan_to_num(z, nan=0.0)
        edge = np.zeros(true.shape, dtype=bool)
        for axis in (0, 1):
            d = np.abs(np.diff(z, axis=axis))
            big = np.nan_to_num(d, nan=np.inf) > np.minimum(np.take(jump, range(d.shape[axis]), axis),
                                                           np.take(jump, range(1, d.shape[axis] + 1), axis))
            if axis == 0:
                edge[:-1] |= big
                edge[1:] |= big
            else:
                edge[:, :-1] |= big
                edge[:, 1:] |= big
        if not self.real_sensors:                   # a real camera's depth is noisy already
            z = z + self.noise_scale * (DEPTH_M + DEPTH_FRAC * z) * noise
        z[edge | ~ok] = np.nan
        z = _median(z, SMOOTH)
        return self._measured(frame, z, None)

    def _rays(self, frame: Frame):
        """Per pixel, the direction of its ray in the world (z-depth units: a point
        is the camera's position plus z times it), by the robot's calibration."""
        np = _np()
        k = frame.intrinsics()
        us = (np.arange(frame.width) + 0.5 - k["cx"]) / k["fx"]
        vs = (np.arange(frame.height) + 0.5 - k["cy"]) / k["fy"]
        cam = np.stack([np.broadcast_to(us[None, :], (frame.height, frame.width)),
                        np.broadcast_to(vs[:, None], (frame.height, frame.width)),
                        np.ones((frame.height, frame.width))], axis=-1)
        pose = self.believed(frame)
        return cam @ _rot(pose.quat).T, np.asarray(pose.pos)

    def _mono_view(self, frame: Frame) -> View:
        """Depth from the picture alone: the monocular model's, its scale fitted
        so that the table top lies where the robot's calibration expects it.
        A robot with one RGB camera knows its table; it does not know the depth
        of anything on it."""
        np = _np()
        image = to_array(frame.image)
        if image is None:
            raise ToolError("not_supported", f"frame {frame.id} has no pixels left",
                            hint="take a new picture with camera_look")
        pred = np.asarray(self._backend().depth(image), dtype=np.float64)
        if pred.shape != (frame.height, frame.width):
            raise ToolError("tool_failed", f"the depth model answered {pred.shape} for a "
                            f"{frame.width}x{frame.height} picture")
        rays, origin = self._rays(frame)
        table = self.station.table_height
        with np.errstate(divide="ignore", invalid="ignore"):
            expected = (table - origin[2]) / rays[..., 2]       # the table's z-depth, per pixel
        ok = np.isfinite(pred) & (pred > 0.05) & np.isfinite(expected) & (expected > 0) \
            & (expected < MAX_DEPTH_M)
        # only where the ray meets the table the robot works on: its arms' reach,
        # and a margin -- what the robot knows of its own station, not of any thing
        hit = origin + rays * expected[..., None]
        near_x, far_x, side, _ = self._reach
        reach = np.zeros(pred.shape, dtype=bool)
        for bx, by, _ in self._bases:
            reach |= ((hit[..., 0] > bx + near_x - MONO_REACH_MARGIN)
                      & (hit[..., 0] < bx + far_x + MONO_REACH_MARGIN)
                      & (np.abs(hit[..., 1] - by) < side + MONO_REACH_MARGIN))
        ok &= reach
        if ok.sum() < MONO_MIN_TABLE_PX:
            raise ToolError("not_visible", f"frame {frame.id} shows too little of the table "
                            "to fit the depth's scale to", hint="look from higher up, or with "
                            "the fixed camera")
        ratio = expected[ok] / pred[ok]
        scale = float(np.median(ratio))
        # a monocular model's depth is off by more than one scale: it tilts and
        # bends the table across the picture.  The correction is fitted where the
        # table top is -- a smooth (quadratic) field of the ratio of the depth the
        # calibration expects to the depth predicted -- and applied everywhere:
        # a thing on the table shares the correction of the table round it
        rows, cols = pred.shape
        vv, uu = np.mgrid[0:rows, 0:cols]
        un, vn = (uu - cols / 2) / cols, (vv - rows / 2) / rows
        basis = np.stack([np.ones_like(un), un, vn, un * un, un * vn, vn * vn], axis=-1)
        field = np.full(pred.shape, scale)
        share = MONO_TABLE_SHARE
        for _ in range(3):                  # the table's pixels, once the depth is near
            near = ok & (np.abs(field * pred - expected) < share * expected)
            if near.sum() < MONO_MIN_TABLE_PX:
                break
            coef, *_ = np.linalg.lstsq(basis[near], expected[near] / pred[near], rcond=None)
            field = basis @ coef
            share = max(0.01, share * 0.6)
        scale = float(np.median(field[ok]))
        self.mono_scales[frame.id] = round(scale, 4)
        z = np.where(np.isfinite(pred) & (pred > 0), field * pred, np.nan)
        z[(z <= 0) | (z >= MAX_DEPTH_M)] = np.nan
        z = _median(z, SMOOTH)
        return self._measured(frame, z, "mono")

    def _measured(self, frame: Frame, z, source) -> View:
        """A view from a depth image (metres along the optical axis, NaN: no
        reading): lifted with the robot's calibration, the robot's own body
        and the table top found."""
        np = _np()
        height_px, width_px = z.shape
        # lifted with the robot's calibration
        k = frame.intrinsics()
        us = (np.arange(width_px) + 0.5 - k["cx"]) / k["fx"]
        vs = (np.arange(height_px) + 0.5 - k["cy"]) / k["fy"]
        cam = np.stack([us[None, :] * z, vs[:, None] * z, z], axis=-1)
        pose = self.believed(frame)
        points = cam @ _rot(pose.quat).T + np.asarray(pose.pos)
        table = self.station.table_height
        robot = self._own_body(frame, points, table)
        plane = _fit_table(points, robot, table)
        height = points[..., 2] - (plane[0] * points[..., 0] + plane[1] * points[..., 1] + plane[2])
        near = np.abs(height) < TABLE_BAND_M
        z_table = float(np.nanmedian(z[near])) if near.any() else float(np.nanmedian(z))
        sigma = self.noise_scale * (DEPTH_M + DEPTH_FRAC * z_table) / (0.6 * SMOOTH)
        if source == "mono" and near.any():
            # what the predicted depth can tell is how smooth it lies on the
            # table: the spread of the table's own heights -- but a thing a
            # centimetre over it is a thing, however rough the prediction
            spread = 1.4826 * float(np.nanmedian(np.abs(height[near & ~robot])))
            sigma = max(sigma, min(spread, MONO_FLAT_MAX_M / 3.0))
        dets, asked = self._found.setdefault(frame.id, ([], set()))
        out = View(depth=z.astype(np.float32), points=points.astype(np.float32),
                   height=height.astype(np.float32), robot=robot, plane=plane, sigma=sigma,
                   dets=dets, asked=asked)
        self._views[frame.id] = out
        while len(self._views) > 8:
            self._views.pop(next(iter(self._views)))
        return out

    def _own_body(self, frame: Frame, points, table: float):
        """The pixels that are the robot: its hands, wrists and cameras where
        the joint readings put them, the arm's base, and whatever stands higher
        than a thing on the table can."""
        np = _np()
        z = points[..., 2]
        own = np.nan_to_num(z, nan=-np.inf) > table + ARM_ABOVE_M
        for bx, by, _ in self._bases:
            own |= np.hypot(points[..., 0] - bx, points[..., 1] - by) < BASE_RADIUS_M
        for robot, model in self._robots:
            for arm in robot.arms:
                state = (frame.arms or {}).get(f"{robot.name}/{arm}")
                if state is None:
                    continue
                tcp, opening = state
                local = (points - np.asarray(tcp.pos)) @ _rot(tcp.quat)
                x, y, up = local[..., 0], local[..., 1], -local[..., 2]
                m = SELF_MARGIN_M
                half = (model.max_opening if opening is None else float(opening)) / 2
                inward = model.max_opening / 2 - half
                top = 0.0
                for part in model.open_hand:
                    ys = part.y
                    if part.name == "finger":
                        ys = ((ys[0] - inward, ys[1] - inward) if ys[0] >= 0
                              else (ys[0] + inward, ys[1] + inward))
                    own |= ((x > part.x[0] - m) & (x < part.x[1] + m) & (y > ys[0] - m)
                            & (y < ys[1] + m) & (up > part.bottom - m)
                            & (up < part.bottom + part.height + m))
                    top = max(top, part.bottom + part.height)
                half_w, column = model.wrist_column
                own |= ((np.abs(x) < half_w + m) & (np.abs(y) < half_w + m) & (up > top - m)
                        & (up < top + column))
        return own
