"""Measurement error of a believable size: the camera's calibration and depth errors,
and hashed draws so that the same thing errs the same way each time."""

from __future__ import annotations

import hashlib
import math
import random

#: each camera's calibration (a wrist camera's rides with the hand):
CAMERA_POS_M = 0.003


CAMERA_ROT_DEG = 0.3


#: depth, per pixel, along the ray: DEPTH_M + DEPTH_FRAC x the distance
DEPTH_M = 0.001


DEPTH_FRAC = 0.003


def _gauss(*key) -> float:
    """A standard normal draw that depends only on ``key``."""
    digest = hashlib.sha256(repr(key).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little")).gauss(0.0, 1.0)


def _uniform(*key) -> float:
    digest = hashlib.sha256(repr(("u",) + key).encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "little")).random()


def _rotvec(r) -> tuple:
    """The quaternion of a small rotation vector (radians)."""
    a = math.sqrt(sum(c * c for c in r))
    if a < 1e-12:
        return (1.0, 0.0, 0.0, 0.0)
    k = math.sin(a / 2) / a
    return (math.cos(a / 2), r[0] * k, r[1] * k, r[2] * k)
