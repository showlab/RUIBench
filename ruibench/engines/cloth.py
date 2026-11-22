"""Cloth: a body that is not rigid.

A cloth is a body like any other in a scene -- a name, a category, a colour,
a mass, a place on the table -- whose ``shape`` is ``"cloth"`` and whose
``attrs["cloth"]`` says what it is made of (:class:`ClothSpec`): the points it
is simulated with, lying flat, and the triangles between them.  To everything
that only asks where a body is and how big (the checks, sight lines, the
perception tools' boxes) it is the flat box round those points.

Its state is an ``ObjectState`` with ``points``: where a sparse set of its
points, the *marks*, are now, in ``ClothSpec.marks``' order.  Its ``pose`` is
the middle of the box round the marks (in the orientation it started in),
its ``size`` that box's half extents -- both follow the cloth as it is folded
--, its ``lin_vel`` the velocity of its fastest mark, and ``held_by`` a hand
that has hold of it.

**The grasp is a rule, not friction** (:func:`pinched`): fingers that close
with their tips at the cloth take the patch of it under them, however many
layers, and carry it until they open.  The same rule on every engine.  On
Isaac the patch is then attached to the hand and PhysX's particle cloth does
the rest; where there is no physics (the mock) :class:`FlatCloth` stands in:
what is let go at B after being taken at A folds the cloth, flat, along the
perpendicular bisector of AB.

Nothing here imports an engine.  Frame of a cloth's own points: x away from
the robot, y to its left, metres, as the cloth lies at yaw 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.geometry import quat_conj, rotate
from ..core.types import ObjectState, Pose, Vec3
from .base import BodySpec

Vec2 = Tuple[float, float]
Line = Tuple[Vec2, Vec2]

SHAPE = "cloth"
#: the grid a cloth is simulated on (m), and how many grid steps apart the
#: marks are each way: 3 cm
CELL = 0.005
MARK_EVERY = 6

#: The patch a closing hand takes, in the tool frame, from where the
#: fingertips end (they end ``TIP_BELOW_TCP`` past the tool centre point on
#: both hands): this far either side across the fingers' travel and along it,
#: from this far up the fingers to this far beyond their tips -- and only if
#: the nearest point of it is within ``PINCH_NEAR`` of the tips: a hand that
#: closes a few centimetres over a cloth closes on air.
TIP_BELOW_TCP = 0.008
PINCH_ACROSS, PINCH_ALONG = 0.014, 0.011
PINCH_ABOVE, PINCH_BELOW = 0.015, 0.020
PINCH_NEAR = 0.010


@dataclass(frozen=True)
class ClothSpec:
    """What a cloth is made of (``BodySpec.attrs["cloth"]``)."""

    #: every simulated point, lying flat, in the cloth's own frame
    nodes: Tuple[Vec2, ...]
    #: the triangles between them (node numbers)
    triangles: Tuple[Tuple[int, int, int], ...]
    #: the nodes whose positions are the cloth's state: reported, logged, scored
    marks: Tuple[int, ...]
    #: named key points -- corners, sleeve ends -- as (name, node); each is a mark
    keys: Tuple[Tuple[str, int], ...] = ()
    #: the grid's step: a point is ``cell / 2`` thick each way
    cell: float = CELL
    #: the colour of its other side; None: as the side that starts up
    back: Optional[str] = None
    #: the springs between neighbouring points, and what damps them
    stretch_stiffness: float = 1.0e4
    bend_stiffness: float = 1.0
    shear_stiffness: float = 100.0
    spring_damping: float = 5.0e-4
    #: against the table, the fingers and itself
    friction: float = 0.8

    def __post_init__(self) -> None:
        marks = set(self.marks)
        missing = [name for name, node in self.keys if node not in marks]
        if missing:
            raise ValueError(f"key points that are not marks: {missing}")
        if len(marks) != len(self.marks):
            raise ValueError("a mark is listed twice")

    @property
    def radius(self) -> float:
        """Half a point's thickness: how far a layer's middle lies above what
        it rests on."""
        return self.cell / 2

    def key(self, name: str) -> int:
        """Where in ``marks`` (and in ``ObjectState.points``) a key point is."""
        node = dict(self.keys)[name]
        return self.marks.index(node)

    def bounds(self) -> Tuple[float, float, float, float]:
        xs, ys = [p[0] for p in self.nodes], [p[1] for p in self.nodes]
        return min(xs), max(xs), min(ys), max(ys)


# -- a cloth from its outline ---------------------------------------------------------
def inside(outline: Sequence[Vec2], p: Vec2, eps: float = 1e-7) -> bool:
    """Is ``p`` in the polygon, its edge counted in?"""
    x, y = p
    n = len(outline)
    hit = False
    for k in range(n):
        (x0, y0), (x1, y1) = outline[k], outline[(k + 1) % n]
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(cross) <= eps * max(1.0, math.hypot(x1 - x0, y1 - y0)) \
                and min(x0, x1) - eps <= x <= max(x0, x1) + eps \
                and min(y0, y1) - eps <= y <= max(y0, y1) + eps:
            return True                                    # on the edge
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            hit = not hit
    return hit


def nearest(points: Sequence[Vec2], p: Vec2) -> int:
    return min(range(len(points)),
               key=lambda n: (points[n][0] - p[0]) ** 2 + (points[n][1] - p[1]) ** 2)


def cloth_spec(outline: Sequence[Vec2], keypoints: Sequence[Tuple[str, Vec2]] = (),
               cell: float = CELL, mark_every: int = MARK_EVERY, **material) -> ClothSpec:
    """The cloth inside ``outline``: its points on a grid ``cell`` apart, two
    triangles a cell, every ``mark_every``-th point each way a mark, and each
    key point the node nearest to where it is given (a mark too)."""
    xs, ys = [p[0] for p in outline], [p[1] for p in outline]
    x0, y0 = min(xs), min(ys)
    nx, ny = round((max(xs) - x0) / cell), round((max(ys) - y0) / cell)
    nodes: List[Vec2] = []
    at: Dict[Tuple[int, int], int] = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            p = (x0 + i * cell, y0 + j * cell)
            if inside(outline, p):
                at[(i, j)] = len(nodes)
                nodes.append(p)
    triangles: List[Tuple[int, int, int]] = []
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = (at.get((i, j)), at.get((i + 1, j)), at.get((i + 1, j + 1)),
                          at.get((i, j + 1)))
            if None not in (a, b, c, d):
                triangles += [(a, b, c), (a, c, d)]
            else:                                          # a cell the outline cuts: the half in it
                corners = [n for n in (a, b, c, d) if n is not None]
                if len(corners) == 3:
                    triangles.append(tuple(corners))
    keys = tuple((name, nearest(nodes, p)) for name, p in keypoints)
    marks = sorted({n for (i, j), n in at.items() if i % mark_every == 0 and j % mark_every == 0}
                   | {n for _, n in keys})
    return ClothSpec(tuple(nodes), tuple(triangles), tuple(marks), keys, cell=cell, **material)


def cloth_body(name: str, spec: ClothSpec, pose: Pose, category: str = "cloth",
               color: Optional[str] = None, mass: float = 0.1,
               attrs: Optional[Mapping[str, object]] = None) -> BodySpec:
    """The body of a cloth lying flat, its own frame's origin at ``pose``
    (which turns it about the vertical only; its height is that of a single
    layer on what it lies on: the top of that plus ``spec.radius``)."""
    x0, x1, y0, y1 = spec.bounds()
    if abs(x0 + x1) > 1e-6 or abs(y0 + y1) > 1e-6:
        raise ValueError(f"{name}: a cloth's points are given round its own middle")
    return BodySpec(name, SHAPE, ((x1 - x0) / 2, (y1 - y0) / 2, spec.radius), pose,
                    category=category, color=color, mass=mass,
                    attrs={"footprint": "rect", **dict(attrs or {}), "cloth": spec,
                           # taken by its own rule (``pinched``), not as a box
                           "graspable": False})


def is_cloth(body) -> bool:
    """A ``BodySpec`` (or anything with a ``shape``) that is a cloth."""
    return getattr(body, "shape", None) == SHAPE


def cloth_of(body: BodySpec) -> ClothSpec:
    return body.attrs["cloth"]


def cloths(scene) -> Tuple[BodySpec, ...]:
    return tuple(b for b in scene.bodies if is_cloth(b))


def scene_problems(scene) -> List[str]:
    """What is wrong with a scene that has a cloth in it, in words.

    A cloth is simulated on the graphics card, and there a body moved by
    writing its pose -- which is how a fixed body gets to its place for the
    episode, how a mark follows what it is painted on, and how an arm's start
    pins what it holds -- moves in the picture and not for the solver: the
    cloth would lie on, or hang from, where the body was built.  So a cloth's
    scene has the table, the arms, loose bodies, and nothing else.  One cloth
    a scene, flat on the table, turned about the vertical only."""
    from ..core.geometry import yaw_of
    from .base import attached_to

    mine = cloths(scene)
    if not mine:
        return []
    out = []
    if len(mine) > 1:
        out.append(f"{len(mine)} cloths ({', '.join(b.name for b in mine)}): one a scene")
    for b in scene.bodies:
        if is_cloth(b):
            continue
        if b.fixed:
            out.append(f"{b.name} is fixed: nothing in a cloth's scene may be")
        if attached_to(b):
            out.append(f"{b.name} is attached to {attached_to(b)}: nothing in a cloth's scene may be")
    if scene.articulations:
        out.append(f"joints ({', '.join(j.name for j in scene.articulations)}): a cloth's scene has none")
    if any(s.holding for s in scene.arm_starts):
        out.append("an arm starts with something in its hand: not in a cloth's scene")
    for b in mine:
        spec = b.attrs.get("cloth")
        if not isinstance(spec, ClothSpec):
            out.append(f"{b.name}: a cloth's attrs[\"cloth\"] is its ClothSpec")
            continue
        if b.fixed:
            out.append(f"{b.name}: a cloth is not fixed")
        w, x, y, z = b.pose.quat
        if abs(x) > 1e-6 or abs(y) > 1e-6:
            out.append(f"{b.name}: a cloth starts flat (turned about the vertical only)")
        if abs(b.pose.pos[2] - (scene.table_height + spec.radius)) > 1e-4:
            out.append(f"{b.name}: a cloth starts on the table, its middle {spec.radius * 1000:g} mm "
                       f"over it (it is {(b.pose.pos[2] - scene.table_height) * 1000:.1f})")
        x0, x1, y0, y1 = spec.bounds()
        if not all(abs(v - h) < 1e-6 for v, h in zip(b.size, ((x1 - x0) / 2, (y1 - y0) / 2, spec.radius))):
            out.append(f"{b.name}: its size is not that of its points (use cloth_body)")
    return out


def flat_points(body: BodySpec, nodes: Optional[Sequence[int]] = None) -> List[Vec3]:
    """Where the cloth's points (all, or the numbered ones) are as it lies
    flat at the body's pose."""
    spec = cloth_of(body)
    picked = spec.nodes if nodes is None else [spec.nodes[n] for n in nodes]
    out = []
    for x, y in picked:
        dx, dy, dz = rotate(body.pose.quat, (x, y, 0.0))
        out.append((body.pose.pos[0] + dx, body.pose.pos[1] + dy, body.pose.pos[2] + dz))
    return out


def box_round(points: Sequence[Vec3], quat, radius: float) -> Tuple[Pose, Vec3]:
    """(pose, half extents) of the box round ``points`` in the orientation
    ``quat``, a point's own ``radius`` added below and above."""
    back = quat_conj(quat)
    local = [rotate(back, p) for p in points]
    lo = [min(p[k] for p in local) for k in range(3)]
    hi = [max(p[k] for p in local) for k in range(3)]
    middle = rotate(quat, tuple((a + b) / 2 for a, b in zip(lo, hi)))
    half = ((hi[0] - lo[0]) / 2, (hi[1] - lo[1]) / 2, (hi[2] - lo[2]) / 2 + radius)
    return Pose(middle, quat), half


def cloth_state(body: BodySpec, marks: Sequence[Vec3], held_by: Optional[str] = None,
                velocity: Vec3 = (0.0, 0.0, 0.0), color=None, visible_from=None) -> ObjectState:
    """A cloth's ``ObjectState`` from where its marks are."""
    pose, half = box_round(marks, body.pose.quat, cloth_of(body).radius)
    return ObjectState(
        name=body.name, pose=pose, size=half, category=body.category,
        color=body.color if color is None else color, held_by=held_by,
        lin_vel=tuple(float(v) for v in velocity), visible_from=visible_from,
        attrs=dict(body.attrs), footprint="rect",
        points=tuple(tuple(float(v) for v in p) for p in marks))


def key_point(obj: ObjectState, name: str) -> Vec3:
    """Where one of a cloth's named key points is now."""
    return obj.points[obj.attrs["cloth"].key(name)]


# -- the grasp ---------------------------------------------------------------------
def pinched(points: Sequence[Vec3], tcp: Pose) -> List[int]:
    """Which of ``points`` a hand closing with its tool centre point at
    ``tcp`` takes hold of: those in the patch under its fingertips, if any of
    them is near the tips (the constants above).  The numbers are places in
    ``points``."""
    back = quat_conj(tcp.quat)
    patch, nearest_below = [], math.inf
    for k, p in enumerate(points):
        x, y, z = rotate(back, tuple(a - b for a, b in zip(p, tcp.pos)))
        z -= TIP_BELOW_TCP                           # from the fingertips' ends, out of the hand
        if abs(x) <= PINCH_ACROSS and abs(y) <= PINCH_ALONG and -PINCH_ABOVE <= z <= PINCH_BELOW:
            patch.append(k)
            nearest_below = min(nearest_below, z)
    return patch if patch and nearest_below <= PINCH_NEAR else []


# -- folding, with no physics --------------------------------------------------------
def side(p: Vec2, line: Line) -> float:
    """Which side of the line a point is on: positive on one side, negative
    on the other, zero on it.  Only whether two points share a sign is used."""
    (ax, ay), (bx, by) = line
    return (bx - ax) * (p[1] - ay) - (by - ay) * (p[0] - ax)


def mirror(p: Vec2, line: Line) -> Vec2:
    (ax, ay), (bx, by) = line
    dx, dy = bx - ax, by - ay
    n = dx * dx + dy * dy
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / n
    fx, fy = ax + t * dx, ay + t * dy
    return (2 * fx - p[0], 2 * fy - p[1])


def fold_flat(points: Sequence[Vec2], layers: Sequence[int], crease: Line, taken: Vec2,
              eps: float = 1e-6) -> Tuple[List[Vec2], List[int]]:
    """The cloth folded along ``crease``: every point on ``taken``'s side of
    it goes over, mirrored, and comes to lie on top -- what was uppermost
    undermost.  (A layer's number is its place in the order things were laid
    down: what a later fold brings over is above everything before it,
    wherever it lands.)"""
    way = side(taken, crease)
    if abs(way) <= eps:
        return list(points), list(layers)
    over = [k for k, p in enumerate(points) if side(p, crease) * way > eps]
    if not over:
        return list(points), list(layers)
    going = set(over)
    top = max((layers[k] for k in range(len(points)) if k not in going), default=-1)
    high = max(layers[k] for k in over)
    out_p, out_l = list(points), list(layers)
    for k in over:
        out_p[k] = mirror(points[k], crease)
        out_l[k] = top + 1 + (high - layers[k])
    return out_p, out_l


def crease_of(picks: Sequence[Tuple[Vec2, Vec2]], apart: float = 0.02) -> Optional[Line]:
    """The line a cloth folds along when what was taken at each pick's first
    point is let go at its second: one hand, the perpendicular bisector of
    the way it went; two hands, the line through the middles of their two
    ways -- or, where those middles are less than ``apart`` from one another,
    the bisector of the way their mean went."""
    picks = [(a, b) for a, b in picks if math.dist(a, b) > 1e-4]
    if not picks:
        return None
    mids = [((a[0] + b[0]) / 2, (a[1] + b[1]) / 2) for a, b in picks]
    if len(picks) >= 2 and math.dist(mids[0], mids[-1]) >= apart:
        return (mids[0], mids[-1])
    ax = sum(a[0] for a, _ in picks) / len(picks)
    ay = sum(a[1] for a, _ in picks) / len(picks)
    bx = sum(b[0] for _, b in picks) / len(picks)
    by = sum(b[1] for _, b in picks) / len(picks)
    mx, my = (ax + bx) / 2, (ay + by) / 2
    dx, dy = bx - ax, by - ay
    return ((mx, my), (mx - dy, my + dx))


def fold_by(points: Sequence[Vec2], layers: Sequence[int],
            picks: Sequence[Tuple[Vec2, Vec2]]) -> Tuple[List[Vec2], List[int]]:
    """What letting go does where there is no physics: the cloth folded along
    :func:`crease_of` the picks -- or, when all of it lies on the side that
    was taken (it was pulled by its edge, away from itself), moved along with
    the hands, whole and flat."""
    picks = [(a, b) for a, b in picks if math.dist(a, b) > 1e-4]
    crease = crease_of(picks)
    if crease is None:
        return list(points), list(layers)
    way = side(picks[0][0], crease)
    if all(side(p, crease) * way > 0 for p in points):
        dx = sum(b[0] - a[0] for a, b in picks) / len(picks)
        dy = sum(b[1] - a[1] for a, b in picks) / len(picks)
        return [(x + dx, y + dy) for x, y in points], list(layers)
    return fold_flat(points, layers, crease, picks[0][0])


class FlatCloth:
    """A cloth where there is no physics: flat on the table in layers, a
    point's height its layer's.  A closing hand takes the patch under its
    fingertips (:func:`pinched`), which then goes where the hand goes while
    the rest stays; when no hand holds the cloth any more it is folded by
    where each patch was taken and let go (:func:`fold_by`)."""

    def __init__(self, body: BodySpec, floor: Optional[float] = None) -> None:
        self.body = body
        self.spec = cloth_of(body)
        start = flat_points(body)
        self.xy: List[Vec2] = [(p[0], p[1]) for p in start]
        self.layers: List[int] = [0] * len(start)
        #: the top of what it lies on
        self.floor = body.pose.pos[2] - self.spec.radius if floor is None else floor
        #: hand -> (the nodes it holds, each where it is in the hand's frame)
        self.grips: Dict[str, Dict[int, Vec3]] = {}
        #: hand -> (where its patch was taken, where it was let go or is now)
        self.picks: Dict[str, List[Vec2]] = {}
        self._tcp: Dict[str, Pose] = {}
        self.velocity: Vec3 = (0.0, 0.0, 0.0)

    # -- where it is ---------------------------------------------------------------
    def height(self, node: int) -> float:
        return self.floor + self.spec.radius * (1 + 2 * self.layers[node])

    def point(self, node: int) -> Vec3:
        for arm, patch in self.grips.items():
            if node in patch:
                tcp = self._tcp[arm]
                return tuple(a + b for a, b in zip(tcp.pos, rotate(tcp.quat, patch[node])))
        x, y = self.xy[node]
        return (x, y, self.height(node))

    def points(self, nodes: Optional[Sequence[int]] = None) -> List[Vec3]:
        return [self.point(n) for n in (range(len(self.xy)) if nodes is None else nodes)]

    def marks(self) -> List[Vec3]:
        return self.points(self.spec.marks)

    @property
    def held_by(self) -> Optional[str]:
        return next(iter(self.grips), None)

    def state(self, visible_from=None) -> ObjectState:
        return cloth_state(self.body, self.marks(), self.held_by,
                           self.velocity if self.grips else (0.0, 0.0, 0.0),
                           visible_from=visible_from)

    # -- the hands -----------------------------------------------------------------
    def take(self, arm: str, tcp: Pose) -> bool:
        """A hand closes here: does it take hold?"""
        patch = pinched(self.points(), tcp)
        if not patch:
            return False
        back = quat_conj(tcp.quat)
        self.grips[arm] = {n: rotate(back, tuple(a - b for a, b in zip(self.point(n), tcp.pos)))
                           for n in patch}
        self._tcp[arm] = tcp
        ax = sum(self.xy[n][0] for n in patch) / len(patch)
        ay = sum(self.xy[n][1] for n in patch) / len(patch)
        self.picks[arm] = [(ax, ay), (ax, ay)]
        return True

    def carry(self, arm: str, tcp: Pose, velocity: Vec3 = (0.0, 0.0, 0.0)) -> None:
        """The hand has moved: its patch with it."""
        if arm in self.grips:
            self._tcp[arm] = tcp
            self.velocity = velocity

    def let_go(self, arm: str) -> None:
        """The hand opens.  With the last hand off it, the cloth is folded."""
        patch = self.grips.get(arm)
        if patch is None:
            return
        now = [self.point(n) for n in patch]
        self.picks[arm][1] = (sum(p[0] for p in now) / len(now), sum(p[1] for p in now) / len(now))
        del self.grips[arm]
        if not self.grips:
            self.xy, self.layers = fold_by(self.xy, self.layers,
                                           [tuple(pick) for pick in self.picks.values()])
            self.picks = {}
            self.velocity = (0.0, 0.0, 0.0)


__all__ = [
    "SHAPE", "CELL", "MARK_EVERY", "TIP_BELOW_TCP", "PINCH_ACROSS", "PINCH_ALONG", "PINCH_ABOVE",
    "PINCH_BELOW", "PINCH_NEAR", "ClothSpec", "cloth_spec", "cloth_body", "is_cloth", "cloth_of",
    "cloths", "scene_problems", "flat_points", "box_round", "cloth_state", "key_point", "pinched", "side", "mirror",
    "fold_flat", "crease_of", "fold_by", "FlatCloth", "inside", "nearest",
]
