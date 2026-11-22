"""Small, dependency-free geometry used by task code and scripted references."""

from __future__ import annotations

import math
from typing import Tuple

from .types import Quat, Vec3


def _normalize(v) -> Tuple[float, float, float]:
    n = math.sqrt(sum(c * c for c in v))
    if n < 1e-12:
        raise ValueError("cannot normalise a zero vector")
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def matrix_to_quat(m) -> Quat:
    """Rotation matrix (rows) -> unit quaternion (w, x, y, z)."""
    (m00, m01, m02), (m10, m11, m12), (m20, m21, m22) = m
    trace = m00 + m11 + m22
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return (0.25 / s, (m21 - m12) * s, (m02 - m20) * s, (m10 - m01) * s)
    if m00 > m11 and m00 > m22:
        s = 2.0 * math.sqrt(1.0 + m00 - m11 - m22)
        return ((m21 - m12) / s, 0.25 * s, (m01 + m10) / s, (m02 + m20) / s)
    if m11 > m22:
        s = 2.0 * math.sqrt(1.0 + m11 - m00 - m22)
        return ((m02 - m20) / s, (m01 + m10) / s, 0.25 * s, (m12 + m21) / s)
    s = 2.0 * math.sqrt(1.0 + m22 - m00 - m11)
    return ((m10 - m01) / s, (m02 + m20) / s, (m12 + m21) / s, 0.25 * s)


def look_at_quat(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0)) -> Quat:
    """Tool orientation whose z axis points from ``eye`` at ``target``.

    The wrist camera looks along the tool axis, so this is how a viewpoint is
    aimed.  The tool y axis is kept horizontal -- the aim tilts by pitching --
    and a target straight below gives exactly ``GRASP_DOWN``.
    """
    z = _normalize(tuple(t - e for t, e in zip(target, eye)))
    hint = up if abs(sum(a * b for a, b in zip(z, up))) < 0.99 else (1.0, 0.0, 0.0)
    y = _normalize(_cross(z, hint))
    x = _cross(y, z)
    # columns are the tool axes expressed in the world frame
    rows = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    q = matrix_to_quat(rows)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def rotate(q: Quat, v: Vec3) -> Vec3:
    w, x, y, z = q
    cx = y * v[2] - z * v[1]
    cy = z * v[0] - x * v[2]
    cz = x * v[1] - y * v[0]
    return (v[0] + 2 * (w * cx + y * cz - z * cy),
            v[1] + 2 * (w * cy + z * cx - x * cz),
            v[2] + 2 * (w * cz + x * cy - y * cx))


def yaw_of(q: Quat) -> float:
    """Rotation about world z of an upright orientation, in radians."""
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def camera_quat(eye: Vec3, target: Vec3, up: Vec3 = (0.0, 0.0, 1.0)) -> Quat:
    """Orientation of a camera at ``eye`` looking at ``target``.

    Optical convention -- x to the right of the image, y down, z along the
    view -- which is what Isaac Lab calls ``"ros"``.  Not :func:`look_at_quat`:
    that one aims a *tool*, whose axes are assigned differently.
    """
    z = _normalize(tuple(t - e for t, e in zip(target, eye)))
    x = _normalize(_cross(z, up))
    y = _cross(z, x)
    rows = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    q = matrix_to_quat(rows)
    n = math.sqrt(sum(c * c for c in q))
    return tuple(c / n for c in q)


def quat_mul(a: Quat, b: Quat) -> Quat:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def quat_conj(q: Quat) -> Quat:
    return (q[0], -q[1], -q[2], -q[3])


def relative_pose(parent, child):
    """``child`` expressed in ``parent``'s frame (both :class:`Pose`)."""
    from .types import Pose

    inv = quat_conj(parent.quat)
    offset = rotate(inv, tuple(c - p for c, p in zip(child.pos, parent.pos)))
    return Pose(offset, quat_mul(inv, child.quat))


def compose(parent, relative):
    """The world pose of something held at ``relative`` in ``parent``'s frame."""
    from .types import Pose

    offset = rotate(parent.quat, relative.pos)
    return Pose(tuple(p + o for p, o in zip(parent.pos, offset)),
                quat_mul(parent.quat, relative.quat))


def inverse(pose):
    """The pose that undoes ``pose``: ``compose(pose, inverse(pose))`` is identity."""
    from .types import Pose

    inv = quat_conj(pose.quat)
    return Pose(tuple(-v for v in rotate(inv, pose.pos)), inv)
