"""The hand's [roll, pitch, yaw] reading (``ruikit.env.station``)."""

import math
import random
from types import SimpleNamespace

from ruibench.core.geometry import quat_mul, rotate
from ruibench.engines.robots import arm_start_pose
from ruibench.tasks.impl._common import TABLE, TWO_ARMS
from ruikit.env.station import quat_from_rpy_deg, rpy_deg_from_quat


def _turned(q, axis, deg):
    n = math.sqrt(sum(a * a for a in axis))
    s = math.sin(math.radians(deg) / 2)
    return quat_mul((math.cos(math.radians(deg) / 2), *(a / n * s for a in axis)), q)


def _angle(a, b):
    return math.degrees(2 * math.acos(min(1.0, abs(sum(x * y for x, y in zip(a, b))))))


def test_the_x5_start_reads_the_same_whichever_way_it_sags():
    """The X5's hands start at pitch 90 degrees, where roll and yaw turn about
    one axis.  A sag of a hundredth of a degree read [180, 90, 180] and a
    twentieth [90, 90, -90], and pitching down 90 degrees from those pointed
    the hand up, or left it level (v2.3.4)."""
    for arm in ("left", "right"):
        start = arm_start_pose(SimpleNamespace(table_height=TABLE), TWO_ARMS, arm)
        for axis, deg in (((0, 1, 0), 0.01), ((0, 1, 0), -0.01), ((1, 0, 0), 0.05),
                          ((1, 0, 0), -0.3), ((0, 0, 1), 0.05), ((1, 1, 0), 0.4)):
            roll, pitch, yaw = rpy_deg_from_quat(_turned(start.quat, axis, deg))
            assert roll == 0.0 and abs(pitch - 90.0) < 0.5 and abs(yaw) < 0.5
            down = quat_from_rpy_deg((roll, pitch - 90.0, yaw))
            assert rotate(down, (0.0, 0.0, 1.0))[2] < -0.999


def test_the_reading_rebuilds_the_hand():
    rng = random.Random(0)
    for _ in range(2000):
        rpy = (rng.uniform(-180, 180), rng.uniform(-89.0, 89.0), rng.uniform(-180, 180))
        q = quat_from_rpy_deg(rpy)
        assert _angle(q, quat_from_rpy_deg(rpy_deg_from_quat(q))) < 0.01
    for _ in range(2000):                     # within half a degree of the lock: roll is let go
        rpy = (rng.uniform(-180, 180), rng.choice((90.0, -90.0)) + rng.uniform(-0.5, 0.5),
               rng.uniform(-180, 180))
        q = quat_from_rpy_deg(rpy)
        assert _angle(q, quat_from_rpy_deg(rpy_deg_from_quat(q))) < 1.2
