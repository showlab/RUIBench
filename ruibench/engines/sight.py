"""Lines of sight, by geometry -- the same answer for every engine that has no
renderer to ask, and for the design checks that run without one.

Bodies are boxes: their upright, world-axis-aligned bounds.  A container is its
walls and its floor (``attrs["container"] == "open"``) or its walls and its
roof (``"inverted"``: a cup put upside down over something), so what is inside
one is hidden or not the way it would be; its floor is ``base.container_floor``
high.  A mesh may say what it is solid as, in boxes (``attrs["proxy"]``:
CL2's doll bed is its mattress, pillows, head- and footboard -- as a tray
with 6 cm walls round a 6 mm floor, it gave depth readings 3 cm under the
mattress, bench-issue #20).  A toy built from parts is the
bounds of each solid part (a roof block, a stack of slabs narrowing to its
ridge): a stand with three pegs is not a wall.  A point is in sight of an eye when
the segment between them crosses no box.
"""

from __future__ import annotations

import math
from typing import List, Sequence, Tuple

from ..core.geometry import yaw_of
from ..core.types import Pose, Vec3

Box = Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]

#: points per side of each face sampled by :func:`facing_samples`
SAMPLES_PER_SIDE = 4


def box_bounds(shape: str, size: Vec3, pose: Pose) -> Box:
    """World-axis-aligned bounds of an upright body."""
    x, y, z = pose.pos
    hx, hy, hz = size
    if shape in ("cylinder", "sphere"):
        ex, ey = hx, hx
    else:
        yaw = yaw_of(pose.quat)
        c, s = abs(math.cos(yaw)), abs(math.sin(yaw))
        ex, ey = c * hx + s * hy, s * hx + c * hy
    return ((x - ex, x + ex), (y - ey, y + ey), (z - hz, z + hz))


def part_boxes(parts, pose: Pose, drawn: bool = False, turned: bool = False) -> List[Box]:
    """World-axis-aligned bounds of each solid part of a body at ``pose`` --
    or, ``drawn``, of each part that is only drawn.  ``turned``: a part is
    covered as the body holds it now -- a lid turned up on its hinge leans as
    a part built leaning does, and is covered slab by slab (``box_pieces``),
    not by the bounds of the whole board, which fill the wedge under it."""
    from ..core.geometry import quat_mul, rotate
    from .parts import box_pieces, part_corners

    out = []
    for part in parts:
        if bool(part[5]) == drawn:
            continue
        if turned:
            kind, centre, dims, rot, color, collides = part
            now = (kind, rotate(pose.quat, centre), dims, quat_mul(pose.quat, rot), color, collides)
            groups = [list(part_corners(piece)) for piece in box_pieces(now)]
        else:
            groups = [[rotate(pose.quat, c) for c in part_corners(piece)]
                      for piece in box_pieces(part)]
        for pts in groups:
            out.append(tuple((min(p[i] for p in pts) + pose.pos[i],
                              max(p[i] for p in pts) + pose.pos[i]) for i in range(3)))
    return out


def occluder_boxes(shape: str, size: Vec3, pose: Pose, attrs, turned: bool = False) -> List[Box]:
    from .base import container_floor

    if shape == "compound" and attrs.get("parts"):
        return part_boxes(attrs["parts"], pose, turned=turned)
    if attrs.get("proxy"):
        return part_boxes(attrs["proxy"], pose, turned=turned)
    (x0, x1), (y0, y1), (z0, z1) = box_bounds(shape, size, pose)
    kind = attrs.get("container")
    wall = float(attrs.get("wall") or 0.0)
    if kind in ("open", "inverted") and wall:
        walls = [((x0, x1), (y0, y0 + wall), (z0, z1)), ((x0, x1), (y1 - wall, y1), (z0, z1)),
                 ((x0, x0 + wall), (y0, y1), (z0, z1)), ((x1 - wall, x1), (y0, y1), (z0, z1))]
        lid = ((x0, x1), (y0, y1), (z0, z0 + container_floor(attrs))) if kind == "open" else \
            ((x0, x1), (y0, y1), (z1 - wall, z1))
        return walls + [lid]
    return [((x0, x1), (y0, y1), (z0, z1))]


#: what is drawn and not seen through: a surface of revolution (a cloche's
#: dome, a pot) in anything but glass or flame
_SEE_THROUGH = ("glass:", "fire:", "flame:")


def sight_boxes(shape: str, size: Vec3, pose: Pose, attrs) -> List[Box]:
    """What blocks a line of sight to a body's far side: what it collides with
    (``occluder_boxes``) and, for a body that says its look is opaque
    (``attrs["opaque"]``), the drawn shells of that look.  A cloche is drawn
    over a hidden ring of blocks it stands on, and a sight line down through
    the ring saw the tart under it -- on the mock, whose sight lines are these
    boxes (DI6, v15; Isaac asks its renderer)."""
    boxes = occluder_boxes(shape, size, pose, attrs)
    if attrs.get("opaque") and attrs.get("parts"):
        shells = [p for p in attrs["parts"] if not p[5] and str(p[0]).startswith("lathe:")
                  and not str(p[4]).startswith(_SEE_THROUGH)]
        boxes += part_boxes(shells, pose, drawn=True)
    return boxes


def facing_samples(box: Box, eye: Vec3, n: int = SAMPLES_PER_SIDE) -> List[Vec3]:
    """Points spread over the faces of a box that face the eye."""
    lo, hi = [r[0] for r in box], [r[1] for r in box]
    ts = [(i + 0.5) / n for i in range(n)]
    out: List[Vec3] = []
    for axis in range(3):
        for value, side in ((lo[axis], -1.0), (hi[axis], 1.0)):
            if (eye[axis] - value) * side <= 0:
                continue                                  # faces away
            a, b = [k for k in range(3) if k != axis]
            for u in ts:
                for v in ts:
                    p = [0.0, 0.0, 0.0]
                    p[axis] = value
                    p[a] = lo[a] + u * (hi[a] - lo[a])
                    p[b] = lo[b] + v * (hi[b] - lo[b])
                    out.append(tuple(p))
    return out


def crosses(p: Vec3, eye: Vec3, box: Box) -> bool:
    """Does the segment from p to the eye pass through the box?"""
    t0, t1 = 1e-4, 1.0
    for k in range(3):
        d = eye[k] - p[k]
        lo, hi = box[k]
        if abs(d) < 1e-12:
            if not lo <= p[k] <= hi:
                return False
            continue
        a, b = (lo - p[k]) / d, (hi - p[k]) / d
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
        if t0 > t1:
            return False
    return True


def hidden_fraction(points: Sequence[Vec3], eye: Vec3, boxes: Sequence[Box]) -> float:
    if not points:
        return 0.0
    return sum(any(crosses(p, eye, b) for b in boxes) for p in points) / len(points)
