"""Planning: grasp candidates from what the cameras measured of a thing."""

from __future__ import annotations

from dataclasses import dataclass

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component, Implementation
from ruikit.perception.targets import Target, _resolve


@dataclass(frozen=True)
class GraspParams:
    target: Target = param("the object: a frame-local id (best), a pixel, or a point")
    arm: str = param("which arm", default="right")
    top_k: int = param("how many candidates", default=3, minimum=1, maximum=8)


@tool(name="plan_grasps",
      data="grasps: [{xyz_m, yaw_deg, opening_m, score, method}], best first",
      family="compute", params=GraspParams, reads=("frames", "depth"),
      source="vision", cost=Cost("declared", 0.1),
      examples=[{"target": {"frame_id": "f3.front", "id": "f3.front#2"}}],
      description="""Propose top-down grasps for a visible object: gripper positions (station
      frame, metres) a little below its top, with the hand's yaw_deg (the fingers close along y
      at yaw 0), the finger opening needed and a score (higher is better; 0 when the object is
      wider than the fingers open). Candidates come from that frame's measurement only. It
      plans no path and does not grasp -- execute a candidate with robot_move_to and
      robot_gripper.""")
def plan_grasps(ctx, p: GraspParams) -> ToolResult:
    impl = ctx.params.get("impl", "box")
    if impl == "box":
        return _grasps_from_box(ctx, p)
    if impl not in ("antipodal", "pca"):
        raise ToolError("invalid_args", f"no grasp planner {impl!r} (box, antipodal, pca)")
    if p.target.id is None or not p.target.frame_id:
        raise ToolError("invalid_args", f"the {impl} planner works on a thing's own points: give "
                        "target.frame_id and target.id from vision_detect")
    import numpy as np

    frame = ctx.frame(p.target.frame_id)
    pts, heights = ctx.perception.points_of(frame, p.target.id)
    if len(pts) < 12:
        raise ToolError("not_visible", f"too little of {p.target.id} is measured in {frame.id} "
                        "to plan a grasp on", hint="look from closer, or use the box planner")
    origin = np.asarray(ctx.station.origin)
    st = pts - origin                                   # station frame
    top = float(np.percentile(heights, 92.0))
    low = float(np.percentile(heights, 4.0))
    depth = min(GRASP_DEPTH_M, max(0.004, (top - low) / 2))
    z = max(top - depth, FINGER_CLEAR_M)
    band = st[heights >= z - 0.004]                     # what the fingers close on
    if len(band) < 6:
        band = st
    xy = band[:, :2]
    others, _ = ctx.perception.scene_points(frame)
    others = others - origin
    if len(others):
        # not this thing: further than a centimetre from every point of it
        from scipy.spatial import cKDTree

        far = cKDTree(st[:, :2]).query(others[:, :2], k=1)[0] > 0.01
        others = others[far & (others[:, 2] > z)]
    opening = ctx.station.arm(p.arm).max_opening
    if impl == "pca":
        centred = xy - xy.mean(axis=0)
        _, vecs = np.linalg.eigh(np.cov(centred.T) if len(xy) > 2 else np.eye(2))
        minor = vecs[:, 0]
        angles = [float(np.degrees(np.arctan2(minor[1], minor[0])))]
        angles.append(angles[0] + 90.0)                 # and along the longer side
    else:
        angles = [float(a) for a in range(0, 180, 15)]
    cands = []
    for theta in angles:
        t = np.radians(theta)
        c, n = np.array([np.cos(t), np.sin(t)]), np.array([-np.sin(t), np.cos(t)])
        along, across = xy @ c, xy @ n
        lo, hi = np.percentile(along, 3.0), np.percentile(along, 97.0)
        width = float(hi - lo) + GRASP_MARGIN_M
        mid = (lo + hi) / 2 * c + float(np.median(across)) * n
        score = 0.0 if width > opening else 1.0 - 0.5 * width / opening
        # each finger lands just outside the thing, across the closing line: is
        # anything else standing there, higher than where the fingers stop?
        blocked = 0
        if len(others) and score > 0:
            for side in (-1.0, 1.0):
                at = mid + side * (width / 2 + FINGER_HALF_M[0]) * c
                d = others[:, :2] - at
                blocked += int(((np.abs(d @ c) < FINGER_HALF_M[0])
                                & (np.abs(d @ n) < FINGER_HALF_M[1])).sum())
            if blocked:
                score *= 0.2
        yaw = (theta - 90.0 + 90.0) % 180.0 - 90.0      # fingers close along y at yaw 0
        cands.append({"xyz_m": [round(float(mid[0]), 4), round(float(mid[1]), 4), round(z, 4)],
                      "yaw_deg": round(yaw, 1), "opening_m": round(width, 4),
                      "score": round(score, 3), "method": impl,
                      **({"blocked_points": blocked} if blocked else {})})
    cands.sort(key=lambda c: -c["score"])
    cands = cands[:p.top_k]
    return ToolResult(data={"grasps": cands},
                      text=f"{len(cands)} grasp candidate(s) ({impl}); best yaw "
                           f"{cands[0]['yaw_deg']} deg, opening {cands[0]['opening_m'] * 100:.1f} cm"
                      if cands else f"no grasp candidate ({impl})")


#: how far below its top a thing is gripped (at most), the air left under the
#: fingertips, the room each finger needs across the closing line, and the
#: margin added to a measured width
GRASP_DEPTH_M = 0.025


FINGER_CLEAR_M = 0.004


FINGER_HALF_M = (0.01, 0.012)


GRASP_MARGIN_M = 0.01


def _grasps_from_box(ctx, p: GraspParams) -> ToolResult:
    """The first implementation: from the pose estimate's box, four headings."""
    top, yaw, ext = _resolve(ctx, p.target)
    opening = ctx.station.arm(p.arm).max_opening
    cands = []
    base_yaw = yaw or 0.0
    for turn in (0.0, 90.0, 45.0, -45.0):
        if ext is not None:
            across = ext[1] if turn % 180 == 0 else ext[0]
            if abs(turn) == 45.0:
                across = (ext[0] ** 2 + ext[1] ** 2) ** 0.5
            height = ext[2]
        else:
            across, height = 0.04, 0.04
        width = across + 0.01
        score = 0.0 if width > opening else round(1.0 - width / opening * 0.5, 3)
        z = top[2] - min(0.025, height / 2)
        cands.append({"xyz_m": [round(top[0], 4), round(top[1], 4), round(z, 4)],
                      "yaw_deg": round((base_yaw + turn + 180) % 360 - 180, 1),
                      "opening_m": round(width, 4), "score": score, "method": "box"})
    cands.sort(key=lambda c: -c["score"])
    cands = cands[:p.top_k]
    return ToolResult(data={"grasps": cands}, text=f"{len(cands)} grasp candidate(s)")


COMPONENT = Component(
    "grasps", "抓取规划", "where and how to take hold of a thing, from what was seen",
    ("plan_grasps",),
    (Implementation("box", "按估计的包围盒，自上而下", "from the estimated box, from above",
                    params={"plan_grasps": {"impl": "box"}}),
     Implementation("antipodal", "在物体的点云上找对顶的两侧", "opposite faces found in the "
                    "thing's points", params={"plan_grasps": {"impl": "antipodal"}}),
     Implementation("pca", "按点云的主方向", "across the principal axis of its points",
                    params={"plan_grasps": {"impl": "pca"}})),
    default="box")
