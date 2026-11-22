"""Planning: a path round what was seen, lifting over it or found by RRT-Connect."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component, Implementation


@dataclass(frozen=True)
class MotionParams:
    xyz_m: Tuple[float, float, float] = param("goal gripper position, station frame, metres")
    rpy_deg: Optional[Tuple[float, float, float]] = param("goal orientation", default=None)
    arm: str = param("which arm", default="right")


@tool(name="plan_motion",
      data="waypoints: [{xyz_m, rpy_deg}] in order, length_m, clearance_m, points_seen, frames, planner",
      family="compute", params=MotionParams, reads=("frames", "depth"), source="vision",
      cost=Cost("declared", 0.2), tracks=("controlled", "open"),
      examples=[{"xyz_m": [0.5, 0.1, 0.1]}],
      description="""Plan a path for one arm's gripper to a goal position round what the
      cameras have seen: every point the latest picture of each camera measured standing on
      the table (its depth, never the scene's true shapes), less the robot itself, a few
      centimetres round the start and round the goal (what the hand holds, what it goes for).
      Returns up to 12 waypoints to send with robot_waypoints; it does not move the robot. Only
      the gripper is kept clear (a column round the hand), not the rest of the arm; what no
      camera saw is taken as free. The path stays as low as the arm reaches there (the ARX X5
      reaches lower the farther out); a goal, or a way over what was seen, higher than that
      is refused. Takes 0.2 s of simulated time.""")
def plan_motion(ctx, p: MotionParams) -> ToolResult:
    import numpy as np

    impl = ctx.params.get("impl", "lift")
    if impl not in ("lift", "rrt"):
        raise ToolError("invalid_args", f"no path planner {impl!r} (lift, rrt)")
    info = ctx.station.arm(p.arm)
    start = np.asarray(ctx.station.to_station(ctx.tcp(info.name).pos), dtype=np.float64)
    goal = np.asarray(p.xyz_m, dtype=np.float64)
    origin = np.asarray(ctx.station.origin)
    clouds, used = [], []
    seen = {}
    for frame in ctx.recent_frames(12):                 # the newest of each camera
        if frame.camera not in seen:
            seen[frame.camera] = frame
    for frame in seen.values():
        try:
            pts, _ = ctx.perception.scene_points(frame)
        except ToolError:
            continue
        clouds.append(pts - origin)
        used.append(frame.id)
    if not used:
        raise ToolError("not_visible", "no picture to plan from", hint="camera_look first")
    cloud = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    for end in (start, goal):                           # what the hand holds, what it goes for
        near = (np.hypot(cloud[:, 0] - end[0], cloud[:, 1] - end[1]) < PATH_END_FREE_M) \
            & (cloud[:, 2] < end[2] + PATH_END_FREE_M)
        cloud = cloud[~near]
    rpy = list(p.rpy_deg) if p.rpy_deg is not None else [0.0, 0.0, 0.0]
    high = float(ctx.station.model.reach[3])
    ceiling = None                                      # how high the arm reaches over a spot
    if math.isfinite(ctx.station.ceiling(info.name, tuple(goal[:2]))):
        ceiling = lambda xy: ctx.station.ceiling(info.name, (float(xy[0]), float(xy[1])))  # noqa: E731
        top = ceiling(goal[:2])
        if goal[2] > top:
            raise ToolError("unreachable", "not reachable: " + ctx.station.too_high(top),
                            hint=f"a goal at most {math.floor(top * 1000) / 1000:.3f} m high here",
                            data={"ceiling_m": round(top, 3)})
    if impl == "lift":
        path = _lift_path(start, goal, cloud, high, ceiling)
        if isinstance(path, _TooHigh):
            raise ToolError("unreachable", f"not reachable: clearing what was seen needs the hand "
                            f"{path.need:.3f} m above the table, "
                            + ctx.station.too_high(path.ceiling),
                            hint="go round what is in the way, or nearer the arm's base",
                            data={"ceiling_m": round(path.ceiling, 3)})
    else:
        rng = np.random.default_rng(abs(hash((ctx.episode_seed, round(float(ctx.sim_time), 2),
                                              tuple(np.round(goal, 3))))) % (2 ** 32))
        path = _rrt_path(start, goal, cloud, high, rng, ceiling)
        if path is None:
            raise ToolError("tool_failed", "no free path found round what was seen"
                            + (" under how high the arm reaches" if ceiling else ""),
                            hint="look again from another camera, or try the lift planner")
    clearance = _clearance(path, cloud)
    length = float(sum(np.linalg.norm(np.asarray(b) - np.asarray(a)) for a, b in zip(path, path[1:])))
    points = [{"xyz_m": [round(float(v), 4) for v in q], "rpy_deg": rpy} for q in path[1:]]
    return ToolResult(data={"waypoints": points, "length_m": round(length, 3),
                            "clearance_m": None if clearance is None else round(clearance, 3),
                            "points_seen": int(len(cloud)), "frames": used, "planner": impl},
                      text=f"{len(points)} waypoint(s), {length:.2f} m ({impl}), "
                           + (f"{clearance * 100:.1f} cm clear of what was seen"
                              if clearance is not None else "nothing seen on the way"))


#: the hand as the planners keep it clear: a column this wide round the tool
#: point, from a little under the fingertips to this far above them
PATH_HAND_RADIUS_M = 0.045


PATH_HAND_BELOW_M, PATH_HAND_ABOVE_M = 0.01, 0.15


#: how far over the highest thing on the way the lift planner goes
PATH_LIFT_CLEAR_M = 0.05


#: the lift planner reads the arm's ceiling this often along the way
PATH_CEILING_STEP_M = 0.01


#: round the start and the goal, what is seen is the hand's own business
PATH_END_FREE_M = 0.035


#: RRT-Connect: step, tries, and the edge check's resolution
RRT_STEP_M, RRT_TRIES, RRT_CHECK_M = 0.03, 3000, 0.01


def _column(cloud, q):
    """The seen points inside the hand's column at tool point ``q``."""
    import numpy as np

    if not len(cloud):
        return np.zeros(0, dtype=bool)
    return ((np.hypot(cloud[:, 0] - q[0], cloud[:, 1] - q[1]) < PATH_HAND_RADIUS_M)
            & (cloud[:, 2] > q[2] - PATH_HAND_BELOW_M) & (cloud[:, 2] < q[2] + PATH_HAND_ABOVE_M))


@dataclass(frozen=True)
class _TooHigh:
    """The lift planner's answer when clearing what was seen needs the hand
    higher than the arm reaches somewhere on the way."""
    need: float
    ceiling: float


def _lift_path(start, goal, cloud, high, ceiling=None):
    """Up over the highest thing seen along the way, across, down: the way a
    person moves a held thing over others -- no higher than the arm reaches
    anywhere on the way (``ceiling(xy)``), or ``_TooHigh`` when that is too low
    to clear what was seen.  Without a ceiling, only ``high`` caps the lift."""
    import numpy as np

    d = goal[:2] - start[:2]
    span = float(np.linalg.norm(d)) or 1e-9
    if len(cloud):
        rel = cloud[:, :2] - start[:2]
        along = np.clip(rel @ d / span, 0.0, span)
        off = np.linalg.norm(rel - np.outer(along / span, d), axis=1)
        on_way = cloud[off < PATH_HAND_RADIUS_M + 0.01]
        tallest = float(on_way[:, 2].max()) if len(on_way) else 0.0
    else:
        tallest = 0.0
    need = tallest + PATH_LIFT_CLEAR_M
    z = min(high, max(start[2], goal[2], need))
    if ceiling is not None:
        n = max(2, int(np.ceil(span / PATH_CEILING_STEP_M)) + 1)
        lim = min(ceiling(start[:2] + d * t) for t in np.linspace(0.0, 1.0, n))
        if z > lim:
            if need > lim:
                return _TooHigh(float(need), float(lim))
            z = lim
    path = [tuple(start)]
    if z > start[2] + 1e-3:
        path.append((start[0], start[1], z))
    if z > goal[2] + 1e-3:
        path.append((goal[0], goal[1], z))
    path.append(tuple(goal))
    return path


def _rrt_path(start, goal, cloud, high, rng, ceiling=None):
    """RRT-Connect over the tool point, the hand's column kept clear of the
    seen points, then shortened: a path round things, not only over them.
    Every sample is kept under the arm's ceiling at its spot (``ceiling(xy)``);
    the ceiling falls ever faster outward, so the straight lines between such
    points stay under it too."""
    import numpy as np

    lo = np.minimum(start, goal) - np.array([0.25, 0.25, 0.0])
    hi = np.maximum(start, goal) + np.array([0.25, 0.25, 0.0])
    lo[2], hi[2] = max(0.01, min(start[2], goal[2]) - 0.02), max(high, start[2], goal[2])

    def free(q):
        return not _column(cloud, q).any()

    def edge_free(a, b):
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / RRT_CHECK_M)) + 1)
        return all(free(a + (b - a) * t) for t in np.linspace(0.0, 1.0, n)[1:-1])

    if edge_free(start, goal):
        return [tuple(start), tuple(goal)]
    trees = ([start], [None]), ([goal], [None])

    def extend(tree, q):
        nodes, parents = tree
        k = int(np.argmin([np.linalg.norm(n - q) for n in nodes]))
        step = q - nodes[k]
        dist = float(np.linalg.norm(step))
        new = q if dist <= RRT_STEP_M else nodes[k] + step / dist * RRT_STEP_M
        if not free(new) or not edge_free(nodes[k], new):
            return None
        nodes.append(new)
        parents.append(k)
        return len(nodes) - 1

    def walk(tree, k):
        nodes, parents = tree
        out = []
        while k is not None:
            out.append(nodes[k])
            k = parents[k]
        return out

    a, b = trees
    for _ in range(RRT_TRIES):
        q = lo + rng.random(3) * (hi - lo)
        if ceiling is not None:
            q[2] = max(0.01, min(q[2], ceiling(q[:2])))
        k = extend(a, q)
        if k is not None:
            target = a[0][k]
            while True:                                  # connect: the other tree reaches toward it
                j = extend(b, target)
                if j is None:
                    break
                if np.linalg.norm(b[0][j] - target) < 1e-6:
                    first, second = walk(a, k), walk(b, j)
                    path = first[::-1] + second[1:]
                    if a is trees[1]:
                        path = path[::-1]
                    return _shorten([np.asarray(x) for x in path], edge_free)
        a, b = b, a
    return None


def _shorten(path, edge_free, rounds: int = 60):
    """Skip waypoints while the straight line between stays free; at most 12."""
    out = list(path)
    i = 0
    while i < len(out) - 2:
        j = len(out) - 1
        while j > i + 1 and not edge_free(out[i], out[j]):
            j -= 1
        out = out[:i + 1] + out[j:]
        i += 1
    while len(out) > 13:                                 # robot_waypoints takes 12 after the start
        out.pop(len(out) // 2)
    return [tuple(float(v) for v in q) for q in out]


def _clearance(path, cloud):
    """The least distance from the path's tool point to a seen point (None:
    nothing seen)."""
    import numpy as np

    if not len(cloud):
        return None
    best = None
    for a, b in zip(path, path[1:]):
        a, b = np.asarray(a), np.asarray(b)
        n = max(2, int(np.ceil(np.linalg.norm(b - a) / 0.01)) + 1)
        for t in np.linspace(0.0, 1.0, n):
            d = float(np.min(np.linalg.norm(cloud - (a + (b - a) * t), axis=1)))
            best = d if best is None else min(best, d)
    return best


COMPONENT = Component(
    "paths", "路径规划", "a way round what the cameras saw (the moves that follow it "
    "are the interface's own; robot_waypoints is the motion extra's)",
    ("plan_motion",),
    (Implementation("lift", "抬高越过看到的最高处再落下", "up over the highest thing seen on the "
                    "way, across, down", params={"plan_motion": {"impl": "lift"}}),
     Implementation("rrt", "在深度建的占据栅格里做 RRT-Connect", "RRT-Connect in an occupancy grid "
                    "built from depth", params={"plan_motion": {"impl": "rrt"}})),
    default="lift")
