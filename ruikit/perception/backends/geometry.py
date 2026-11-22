"""Numerics on masks, depth and points: overlap, filtering, the table plane, footprints,
tilt, colour."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from ruikit.perception.backends.pixels_types import (
    View,
    _np,
)

#: the table top is fitted to what lies within this of where the calibration
#: puts it
TABLE_BAND_M = 0.03


#: plain colour words, by hue (degrees) -- given when most of a mask agrees
HUES = ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (255, "blue"),
        (290, "purple"), (335, "pink"), (360, "red"))


COLOUR_SHARE = 0.6


# -- small numerics ---------------------------------------------------------------
def _iou(a, b) -> float:
    np = _np()
    inter = float(np.logical_and(a, b).sum())
    union = float(a.sum()) + float(b.sum()) - inter
    return inter / union if union else 0.0


def _shifted(a, dy: int, dx: int, fill):
    np = _np()
    out = np.full(a.shape, fill, dtype=a.dtype)
    h, w = a.shape
    ys, yd = (slice(dy, h), slice(0, h - dy)) if dy >= 0 else (slice(0, h + dy), slice(-dy, h))
    xs, xd = (slice(dx, w), slice(0, w - dx)) if dx >= 0 else (slice(0, w + dx), slice(-dx, w))
    out[yd, xd] = a[ys, xs]
    return out


def _median(z, size: int):
    """The median of each pixel's ``size`` x ``size`` neighbourhood, over the
    readings there are; no reading stays none."""
    np = _np()
    r = size // 2
    stack = np.stack([_shifted(z, dy, dx, np.nan) for dy in range(-r, r + 1)
                      for dx in range(-r, r + 1)])
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        out = np.nanmedian(stack, axis=0)
    out[~np.isfinite(z)] = np.nan
    return out


def _erode(mask):
    out = mask.copy()
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        out &= _shifted(mask, dy, dx, False)
    return out


def _dilate(mask, n: int):
    out = mask.copy()
    for _ in range(n):
        grown = out.copy()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            grown |= _shifted(out, dy, dx, False)
        out = grown
    return out


def _fit_table(points, robot, table: float) -> Tuple[float, float, float]:
    """The table top as the camera measures it: the plane through what lies
    near where the calibration puts it (z = a x + b y + c, world).  The
    calibration alone is off by a centimetre across the table."""
    np = _np()
    sub = points[::3, ::3].reshape(-1, 3)
    sub = sub[np.isfinite(sub).all(axis=1) & ~robot[::3, ::3].reshape(-1)]
    near = sub[np.abs(sub[:, 2] - table) < TABLE_BAND_M]
    plane = (0.0, 0.0, table)
    for _ in range(4):
        if len(near) < 200:
            return plane
        A = np.c_[near[:, 0], near[:, 1], np.ones(len(near))]
        sol, *_ = np.linalg.lstsq(A, near[:, 2], rcond=None)
        if abs(sol[0]) > 0.05 or abs(sol[1]) > 0.05:           # not a table top: 3 degrees
            return plane
        plane = (float(sol[0]), float(sol[1]), float(sol[2]))
        res = sub[:, 2] - (sub[:, 0] * plane[0] + sub[:, 1] * plane[1] + plane[2])
        spread = max(0.001, 1.4826 * float(np.median(np.abs(res[np.abs(res) < TABLE_BAND_M]))))
        near = sub[np.abs(res) < 2.5 * spread]
    return plane


def _footprint(xy):
    """The smallest rectangle round a thing's points, seen from above:
    (middle), (length along its heading, across it), heading in radians --
    of its longer side, between -90 and 90 degrees."""
    np = _np()
    xy = np.asarray(xy, dtype=np.float64)
    centre = np.median(xy, axis=0)
    spread = np.abs(xy - centre)
    keep = (spread <= np.percentile(spread, 99.0, axis=0) + 1e-9).all(axis=1)
    xy = xy[keep] if keep.sum() >= 3 else xy
    hull = _hull(xy)
    best = None
    for k in range(len(hull)):
        e = hull[(k + 1) % len(hull)] - hull[k]
        n = float(np.hypot(e[0], e[1]))
        if n < 1e-9:
            continue
        ux, uy = e[0] / n, e[1] / n
        along = hull @ np.array([ux, uy])
        across = hull @ np.array([-uy, ux])
        area = (along.max() - along.min()) * (across.max() - across.min())
        if best is None or area < best[0]:
            best = (area, ux, uy, along.min(), along.max(), across.min(), across.max())
    if best is None:
        return (float(centre[0]), float(centre[1])), (0.0, 0.0), 0.0
    _, ux, uy, a0, a1, c0, c1 = best
    mid_a, mid_c = (a0 + a1) / 2, (c0 + c1) / 2
    cx, cy = mid_a * ux - mid_c * uy, mid_a * uy + mid_c * ux
    length, width, yaw = a1 - a0, c1 - c0, math.atan2(uy, ux)
    if width > length:
        length, width, yaw = width, length, yaw + math.pi / 2
    yaw = (yaw + math.pi / 2) % math.pi - math.pi / 2
    return (float(cx), float(cy)), (float(length), float(width)), float(yaw)


def _hull(xy):
    """Convex hull (Andrew's chain), counter-clockwise."""
    np = _np()
    pts = np.unique(np.round(xy, 5), axis=0)
    if len(pts) < 3:
        return pts
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]

    def half(points):
        out = []
        for p in points:
            while len(out) >= 2 and ((out[-1][0] - out[-2][0]) * (p[1] - out[-2][1])
                                     - (out[-1][1] - out[-2][1]) * (p[0] - out[-2][0])) <= 0:
                out.pop()
            out.append(p)
        return out

    lower, upper = half(pts), half(pts[::-1])
    return np.array(lower[:-1] + upper[:-1])


def _normal(points, j: int, i: int, eye) -> Tuple[float, float, float]:
    """The surface's normal at a pixel: the plane through the readings round
    it, facing the camera."""
    np = _np()
    near = points[max(0, j - 3):j + 4, max(0, i - 3):i + 4].reshape(-1, 3)
    near = near[np.isfinite(near).all(axis=1)]
    if len(near) < 6:
        return (0.0, 0.0, 1.0)
    centred = near - near.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    n = vt[-1]
    if float(n @ (np.asarray(eye) - near.mean(axis=0))) < 0:
        n = -n
    return tuple(float(v) for v in n)


def _tilt(pts, heights, top: float, bottom: float, view: View):
    """How far a thing's top leans from the table's, in degrees, and which way
    (a horizontal unit vector) -- when its top is a face flat enough to say."""
    np = _np()
    band = heights > top - max(0.003, 0.2 * (top - bottom))
    if band.sum() < 40:
        return 0.0, None
    face = pts[band]
    A = np.c_[face[:, 0], face[:, 1], np.ones(len(face))]
    sol, *_ = np.linalg.lstsq(A, face[:, 2], rcond=None)
    res = face[:, 2] - A @ sol
    if float(np.sqrt(np.mean(res ** 2))) > 0.0015 + 2.0 * view.sigma:
        return 0.0, None                                    # no flat face to read
    n = np.array([-(sol[0] - view.plane[0]), -(sol[1] - view.plane[1]), 1.0])
    n /= np.linalg.norm(n)
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(n[2])))))
    if tilt < 1.0:
        return tilt, None
    flat = math.hypot(float(n[0]), float(n[1])) or 1.0
    return tilt, (float(n[0]) / flat, float(n[1]) / flat)


def _colour(image, mask) -> Optional[str]:
    """A plain colour word when most of the mask's pixels agree on one."""
    np = _np()
    if image is None or not mask.any():
        return None
    rgb = image[mask].astype(np.float64) / 255.0
    hi, lo = rgb.max(axis=1), rgb.min(axis=1)
    span = hi - lo
    sat = np.where(hi > 0, span / np.maximum(hi, 1e-9), 0.0)
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    hue = np.zeros(len(rgb))
    nz = span > 1e-9
    rm, gm, bm = nz & (hi == r), nz & (hi == g) & (hi != r), nz & (hi == b) & (hi != r) & (hi != g)
    hue[rm] = (60 * ((g - b)[rm] / span[rm])) % 360
    hue[gm] = 60 * ((b - r)[gm] / span[gm]) + 120
    hue[bm] = 60 * ((r - g)[bm] / span[bm]) + 240
    names = np.full(len(rgb), "", dtype=object)
    grey = sat < 0.22
    names[grey & (hi > 0.8)] = "white"
    names[grey & (hi <= 0.8) & (hi > 0.25)] = "grey"
    names[grey & (hi <= 0.25)] = "black"
    last = 0
    for limit, word in HUES:
        names[~grey & (hue >= last) & (hue < limit)] = word
        last = limit
    brown = ~grey & (hue >= 10) & (hue < 50) & ((hi < 0.65) | (sat < 0.55))
    names[brown] = "brown"
    words, counts = np.unique(names, return_counts=True)
    k = int(counts.argmax())
    return str(words[k]) if words[k] and counts[k] >= COLOUR_SHARE * len(rgb) else None
