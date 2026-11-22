"""Each arm's joints: forward and inverse kinematics, the same on every backend.

An arm is a chain of revolute joints, written as a product of exponentials
(Lynch & Park, *Modern Robotics*, ch. 4 and 6): every joint's screw axis in the
arm's base frame, and the grasp body's pose there, all at the ready posture.
The numbers are measured on Isaac from the articulation itself (``KINEMATICS``),
so the mock, the inverse kinematics offered to agents and the simulator agree.
An arm's base frame is where its articulation stands: ``arm_base``, turned as
the robot's base pose.

Pure Python, like the rest of the core.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from ..core.geometry import compose, matrix_to_quat, relative_pose
from ..core.types import Pose, Vec3

Mat = Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]
Screw = Tuple[float, float, float, float, float, float]


@dataclass(frozen=True)
class ArmKinematics:
    """One arm model's joints, base to wrist."""
    joints: Tuple[str, ...]
    #: the posture the screw axes and ``body`` are given at (rad)
    ready: Tuple[float, ...]
    #: each joint's screw axis at ``ready``, base frame: (wx, wy, wz, vx, vy, vz)
    screws: Tuple[Screw, ...]
    #: the grasp body at ``ready``, base frame
    body: Pose
    lower: Tuple[float, ...]
    upper: Tuple[float, ...]


def _revolute(screws) -> Tuple[Screw, ...]:
    """Measured axes as exact turning axes: the direction a unit vector, the
    moment at right angles to it (no pitch)."""
    out = []
    for sc in screws:
        n = math.sqrt(sum(c * c for c in sc[:3]))
        w = tuple(c / n for c in sc[:3])
        along = sum(a * b for a, b in zip(sc[3:], w))
        out.append(w + tuple(v - along * c for v, c in zip(sc[3:], w)))
    return tuple(out)


#: measured on Isaac at the ready posture (v1.17): each joint's axis from the articulation's
#: own Jacobian (its direction) through the child link's origin, in the arm's base frame;
#: checked against the simulator at five other postures of each arm -- within 0.001 mm and
#: 0.001 degrees (both X5 arms the same)
KINEMATICS: Dict[str, ArmKinematics] = {
    "fr3": ArmKinematics(
        joints=("fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4", "fr3_joint5", "fr3_joint6", "fr3_joint7"),
        ready=(0.0, -0.7921, 0.0, -2.839, 0.0, 2.0469, 0.7854),
        screws=_revolute((
            (1e-07, 0.0, 1.0000004, 0.0, 0.0, 0.0),
            (3e-07, 0.9999999, 0.0, -0.3330001, 1e-07, 0.0),
            (-0.7118298, 1e-07, 0.7023523, 0.0, -0.2370395, 0.0),
            (1e-07, -1.0000002, 2e-07, 0.6136696, 1e-07, 0.1669943),
            (0.8887873, 1e-07, -0.4583199, -1e-07, 0.5513851, -1e-07),
            (4e-07, -1.0000002, 1e-07, 0.5109997, 2e-07, -0.2121117),
            (2e-07, 4e-07, -1.0, -3e-07, 0.3001118, 1e-07),
        )),
        body=Pose((0.3001117, 2e-07, 0.4039996),
                  (-2e-07, 1.0000001, -6e-07, -1e-07)),
        lower=(-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159),
        upper=(2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)),
    "arx5": ArmKinematics(
        joints=("joint1", "joint2", "joint3", "joint4", "joint5", "joint6"),
        ready=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        screws=_revolute((
            (1e-07, 0.0, 1.0000001, 0.0, 0.0, 0.0),
            (1e-07, 1.0000004, 0.0, -0.1005001, 0.0, 0.02),
            (2e-07, -1.0000001, -7.3e-06, 0.1005001, -1.8e-06, 0.2440001),
            (1e-07, -1.0, -7.1e-06, 0.1565, 0.0, -0.001),
            (1e-07, 7.3e-06, -1.0, 0.0004993, 0.0687501, 5e-07),
            (1.0, 0.0, 0.0, 0.0, 0.1565001, 0.0005004),
        )),
        body=Pose((0.0977, -0.0005004, 0.1565001),
                  (1.0, 0.0, 0.0, 0.0)),
        lower=(-2.87, 0.0, 0.0, -1.57, -1.57, -1.57),
        upper=(3.14, 3.66, 3.14, 1.57, 1.57, 1.57)),
}

#: inverse kinematics: how close counts as there, and how hard it tries
IK_POS_TOL = 1e-4               # m
IK_ROT_TOL = 1e-3               # rad
IK_ITERATIONS = 150
IK_RESTARTS = 12
IK_DAMPING = 0.02
IK_MAX_STEP = 0.3               # rad per joint per iteration


@dataclass(frozen=True)
class IKSolution:
    joints: Tuple[float, ...]
    pos_err: float              # m, left between the tool and the goal
    rot_err: float              # rad
    ok: bool                    # within ``IK_POS_TOL`` and ``IK_ROT_TOL``


# -- small linear algebra ----------------------------------------------------------
def _quat_matrix(q) -> Mat:
    w, x, y, z = q
    return ((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)),
            (2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)),
            (2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)))


def _mat_mul(a: Mat, b: Mat) -> Mat:
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3))
                 for i in range(3))


def _mat_vec(a: Mat, v) -> Vec3:
    return tuple(sum(a[i][k] * v[k] for k in range(3)) for i in range(3))


def _cross(a, b) -> Vec3:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _transpose(a: Mat) -> Mat:
    return tuple(tuple(a[j][i] for j in range(3)) for i in range(3))


def _exp(screw: Screw, theta: float) -> Tuple[Mat, Vec3]:
    """The motion of turning ``theta`` about a unit screw axis (Rodrigues)."""
    w, v = screw[:3], screw[3:]
    s, c = math.sin(theta), math.cos(theta)
    k = ((0.0, -w[2], w[1]), (w[2], 0.0, -w[0]), (-w[1], w[0], 0.0))
    k2 = _mat_mul(k, k)
    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    rot = tuple(tuple(eye[i][j] + s * k[i][j] + (1 - c) * k2[i][j] for j in range(3))
                for i in range(3))
    g = tuple(tuple(eye[i][j] * theta + (1 - c) * k[i][j] + (theta - s) * k2[i][j]
                    for j in range(3)) for i in range(3))
    return rot, _mat_vec(g, v)


def _then(a: Tuple[Mat, Vec3], b: Tuple[Mat, Vec3]) -> Tuple[Mat, Vec3]:
    (ra, pa), (rb, pb) = a, b
    return _mat_mul(ra, rb), tuple(x + y for x, y in zip(_mat_vec(ra, pb), pa))


def _rotvec(r: Mat) -> Vec3:
    """The rotation vector of a rotation matrix."""
    cos = max(-1.0, min(1.0, (r[0][0] + r[1][1] + r[2][2] - 1.0) / 2.0))
    angle = math.acos(cos)
    if angle < 1e-9:
        return (0.0, 0.0, 0.0)
    if math.pi - angle < 1e-6:
        # half a turn: the axis from the matrix's largest diagonal
        i = max(range(3), key=lambda k: r[k][k])
        axis = [0.0, 0.0, 0.0]
        axis[i] = math.sqrt(max(0.0, (r[i][i] + 1.0) / 2.0))
        for j in range(3):
            if j != i:
                axis[j] = r[j][i] / (2.0 * axis[i])
        return tuple(angle * a for a in axis)
    k = angle / (2.0 * math.sin(angle))
    return (k * (r[2][1] - r[1][2]), k * (r[0][2] - r[2][0]), k * (r[1][0] - r[0][1]))


def _solve(a: List[List[float]], b: List[float]) -> List[float]:
    """``a x = b`` for a small symmetric positive-definite ``a`` (Gauss)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        m[col], m[piv] = m[piv], m[col]
        p = m[col][col]
        for r in range(col + 1, n):
            f = m[r][col] / p
            if f:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


# -- kinematics --------------------------------------------------------------------
def arm_kinematics(model_name: str) -> Optional[ArmKinematics]:
    return KINEMATICS.get(model_name)


def register_kinematics(model_name: str, kin: ArmKinematics) -> ArmKinematics:
    """Add an arm defined outside this module to ``KINEMATICS``: the real
    station's (``engines.franka.model``), held to its own robot's limits."""
    KINEMATICS[model_name] = kin
    return kin


def arm_frame(robot, arm: str, table_height: float) -> Pose:
    """Where an arm's articulation stands: its base frame in the world."""
    from .robots import arm_base

    return Pose(arm_base(robot, arm, table_height), robot.base_pose.quat)


def _chain(kin: ArmKinematics, q: Sequence[float], tool: Optional[Pose] = None):
    """The grasp body -- or the ``tool`` frame on it -- in the base frame at
    ``q``, and every joint's axis there (the space Jacobian's columns)."""
    t = (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), (0.0, 0.0, 0.0))
    axes = []
    for screw, angle, ready in zip(kin.screws, q, kin.ready):
        r, p = t
        w = _mat_vec(r, screw[:3])
        axes.append((w, tuple(a + b for a, b in zip(_mat_vec(r, screw[3:]), _cross(p, w)))))
        t = _then(t, _exp(screw, angle - ready))
    end = kin.body if tool is None else compose(kin.body, tool)
    return _then(t, (_quat_matrix(end.quat), end.pos)), axes


def body_in_base(kin: ArmKinematics, q: Sequence[float]) -> Pose:
    (r, p), _ = _chain(kin, q)
    return Pose(p, matrix_to_quat(r))


def forward(model, base: Pose, q: Sequence[float]) -> Pose:
    """The tool centre point, in the world, of an arm of ``model`` whose base
    frame is ``base`` and whose joints are at ``q`` (rad)."""
    from .robots import tcp_from_body

    kin = KINEMATICS[model.name]
    return tcp_from_body(compose(base, body_in_base(kin, q)), model)


def within_limits(kin: ArmKinematics, q: Sequence[float], slack: float = 1e-6) -> bool:
    return all(lo - slack <= v <= hi + slack for v, lo, hi in zip(q, kin.lower, kin.upper))


def inverse(model, base: Pose, tcp: Pose, seed: Optional[Sequence[float]] = None) -> IKSolution:
    """Joints that put the tool centre point at ``tcp`` (world), inside the
    joint limits: damped least squares from ``seed`` (else the ready posture),
    then from postures drawn at random -- the same draw for the same goal.  When
    none gets there, the nearest the first start comes: how close the arm gets
    from where it is, not a posture across the table that misses by a little
    less."""
    kin = KINEMATICS[model.name]
    goal = relative_pose(base, tcp)
    goal_r = _quat_matrix(goal.quat)
    starts = [tuple(seed)] if seed is not None else []
    starts.append(kin.ready)
    rng = random.Random(repr(tuple(round(v, 6) for v in goal.pos + goal.quat)))
    starts += [tuple(rng.uniform(lo, hi) for lo, hi in zip(kin.lower, kin.upper))
               for _ in range(IK_RESTARTS)]
    first: Optional[IKSolution] = None
    for start in starts:
        sol = _descend(kin, goal.pos, goal_r, [min(max(v, lo), hi) for v, lo, hi
                                               in zip(start, kin.lower, kin.upper)],
                       model.tcp_from_body)
        if sol.ok:
            return sol
        first = first or sol
    return first


def _descend(kin: ArmKinematics, goal_p: Vec3, goal_r: Mat, q: List[float],
             tool: Pose) -> IKSolution:
    """Damped least squares on the tool frame, the joints kept in their limits."""
    n = len(q)
    lam2 = IK_DAMPING ** 2
    for _ in range(IK_ITERATIONS):
        (r, p), axes = _chain(kin, q, tool)
        ep = [g - c for g, c in zip(goal_p, p)]
        er = list(_rotvec(_mat_mul(goal_r, _transpose(r))))
        pos_err = math.sqrt(sum(e * e for e in ep))
        rot_err = math.sqrt(sum(e * e for e in er))
        if pos_err < IK_POS_TOL and rot_err < IK_ROT_TOL:
            return IKSolution(tuple(q), pos_err, rot_err, True)
        # the body point's linear velocity and the angular velocity, per joint
        cols = [tuple(a + b for a, b in zip(v, _cross(w, p))) + tuple(w) for w, v in axes]
        e = ep + er
        # less damping close to the goal, so the last tenth of a millimetre comes too
        damp = lam2 * min(1.0, (pos_err + rot_err) / 0.01) ** 2 + 1e-9
        # a joint at a limit that the step would push further is held there and
        # the others take its share: clamped instead, it stalled the descent, and
        # the restarts found far postures for points a centimetre apart (CL4 on
        # Isaac: the FR3's elbow at its stop, the hand lifted out of the pitcher)
        held: set = set()
        while True:
            jjt = [[sum(cols[k][i] * cols[k][j] for k in range(n) if k not in held)
                    + (damp if i == j else 0.0) for j in range(6)] for i in range(6)]
            y = _solve(jjt, e)
            dq = [0.0 if k in held else sum(cols[k][i] * y[i] for i in range(6)) for k in range(n)]
            out = {k for k in range(n) if k not in held and (
                (q[k] <= kin.lower[k] and dq[k] < 0) or (q[k] >= kin.upper[k] and dq[k] > 0))}
            if not out:
                break
            held |= out
        big = max(abs(d) for d in dq)
        if big > IK_MAX_STEP:
            dq = [d * IK_MAX_STEP / big for d in dq]
        q = [min(max(v + d, lo), hi) for v, d, lo, hi in zip(q, dq, kin.lower, kin.upper)]
    (r, p), _ = _chain(kin, q, tool)
    pos_err = math.dist(goal_p, p)
    rot_err = math.sqrt(sum(e * e for e in _rotvec(_mat_mul(goal_r, _transpose(r)))))
    return IKSolution(tuple(q), pos_err, rot_err, pos_err < IK_POS_TOL and rot_err < IK_ROT_TOL)


__all__ = ["ArmKinematics", "IKSolution", "KINEMATICS", "arm_kinematics", "arm_frame",
           "body_in_base", "forward", "inverse", "register_kinematics", "within_limits"]
