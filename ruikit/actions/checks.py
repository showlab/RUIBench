"""Checks after a skill's pick or place (v2.5.2), for the ablation of the
completion check.

Attached as ``+check`` (report: a failed check is said in the skill's result)
or ``+check:correct`` (the skill also corrects itself once, with no new
decision of the agent's: it picks again, or picks up what it dropped and
places it again) -- ``ruikit.harness.components.check``.  Without either, no
skill does anything it did not do before.

A check reads only what the robot itself has: the depth of the fixed camera
and of the arm's wrist camera (``scene_points``: what stands on the table, not
the table, not the robot) and the fingers.  After a pick: the fingers hold
something, and it went with the hand -- the cameras see something hanging under
the lifted hand, or, seeing nothing there, what stood where the hand took it no
longer stands as tall (off a stack or a peg, what is left keeps the spot tall,
so that alone is no evidence).  After a place: the fingers hold nothing, and the
target stands higher than before -- something now lies on it.  A spot no
camera sees counts as giving no evidence either way.

A checked skill runs to its end before the check, even inside a reply that
calls several tools (elsewhere a reply's motions run together).  Every check
is written to ``checks.jsonl`` in the episode's folder, with its pictures and,
beside what the robot measured, what the simulator knows: for judging the
checks afterwards, never for the agent."""

from __future__ import annotations

import json
import math
from dataclasses import replace
from typing import List, Optional, Tuple

from ruikit.core.result import ToolError, ToolResult
from ruikit.env.handles import BLOCKING_CAP_SECONDS

#: the spot a hand took a thing from, and a place target given as a pixel or a point
SPOT_R_M = 0.02
TARGET_R_M = 0.03
#: how tall what stands at a spot is: this percentile of its points' heights
TOP_PCT = 90
#: a spot with fewer points than this tells nothing (a camera's view is blocked)
MIN_POINTS = 12
#: a thing still standing where it was taken from: within this of its height then
STAYED_M = 0.012
#: thinner than this, a thing cannot be told from the table it lies on
THIN_M = 0.012
#: something new on a place target, cell by cell: cells this big, each from this many
#: points, at least this many of them standing this much higher than before
CELL_M = 0.01
CELL_POINTS = 3
RISEN_M = 0.005
RISEN_CELLS = 3
#: the fixed camera, which does not move, compared pixel by pixel: a surface this much higher
#: in a pixel's line of sight than before is something new there (its depth noise is the same
#: for the same viewpoint); something new on the target fills at least this many cells, and
#: the target is judged so when at least this many of its cells compare
NEW_M = 0.004
NEW_CELLS = 5
MIN_COMPARED = 5
#: what the lifted hand carries: points within this of it across, down to this under it
HELD_R_M = 0.04
HELD_BELOW_M = 0.10
#: a dropped thing is looked for within this of the target, on a grid this fine
SEARCH_R_M = 0.12
GRID_M = 0.02
#: ... and what lies there is taken as what is new within this of its middle
THING_R_M = 0.025
LOG = "checks.jsonl"


# -- what the robot measures -----------------------------------------------------
def _cameras(ctx, arm: str) -> List[str]:
    station = ctx.station
    fixed = "head" if station.two_arms else "front"
    wrist = f"wrist_{arm}" if station.two_arms else "wrist"
    return [c for c in (fixed, wrist) if c in station.cameras]


def _look(ctx, arm: str) -> list:
    frames = []
    for cam in _cameras(ctx, arm):
        try:
            frames.append(ctx.capture(cam))
        except Exception:                     # a camera that cannot render: one view less
            pass
    return frames


def _cloud(ctx, frames):
    """(N x 3): station x, y and the height over the table of what the frames saw standing."""
    import numpy as np

    origin = np.asarray(ctx.station.origin, dtype=float)
    parts = []
    for f in frames:
        try:
            pts, heights = ctx.perception.scene_points(f)
        except (ToolError, Exception):
            continue
        pts = np.asarray(pts, dtype=float)
        if len(pts):
            parts.append(np.column_stack([pts[:, 0] - origin[0], pts[:, 1] - origin[1],
                                          np.asarray(heights, dtype=float)]))
    return np.concatenate(parts) if parts else np.zeros((0, 3))


def _height(cloud, xy, r: float, below: Optional[float] = None) -> Tuple[Optional[float], int]:
    """How tall what stands within ``r`` of ``xy`` is (lower than ``below``), and
    from how many points; None when no point is there."""
    import numpy as np

    if not len(cloud):
        return None, 0
    near = np.hypot(cloud[:, 0] - xy[0], cloud[:, 1] - xy[1]) < r
    h = cloud[near, 2]
    if below is not None:
        h = h[h < below]
    if not len(h):
        return None, 0
    return float(np.percentile(h, TOP_PCT)), int(len(h))


def _holding(ctx, arm: str) -> bool:
    from ruikit.perception.proprioception import holding_something

    info = ctx.station.arm(arm)
    return bool(holding_something(ctx._ep.engine.state().arm(info.arm_id)))


def _run(ctx, handle) -> ToolResult:
    """The skill's motion to its end, now, and its report."""
    ctx._ep._run_until(lambda: handle.done, BLOCKING_CAP_SECONDS)
    return ctx._report(handle)


# -- the verdicts ------------------------------------------------------------------
def _carried(ctx, arm, after) -> int:
    """How many points (not the robot's) the cameras see hanging under the lifted hand:
    within ``HELD_R_M`` of it across, from ``HELD_BELOW_M`` under it to just above it --
    what the fingers carry."""
    import numpy as np

    if not len(after):
        return 0
    x, y, z = ctx.station.describe_pose(ctx.tcp(arm))["xyz_m"]
    near = np.hypot(after[:, 0] - x, after[:, 1] - y) < HELD_R_M
    band = (after[:, 2] > z - HELD_BELOW_M) & (after[:, 2] < z + 0.02)
    return int((near & band).sum())


def _pick_verdict(ctx, arm, xy, top_z, before, after) -> dict:
    holding = _holding(ctx, arm)
    h0, n0 = _height(before, xy, SPOT_R_M, below=top_z + 0.03)
    h1, n1 = _height(after, xy, SPOT_R_M, below=(h0 if h0 is not None else top_z) + 0.03)
    carried = _carried(ctx, arm, after)
    # what stood at the spot stands there as tall as before -- evidence only when nothing
    # is seen under the hand: taken off a stack or a peg, what is left keeps the spot tall
    stayed = (carried < MIN_POINTS and h0 is not None and h0 >= THIN_M and n0 >= MIN_POINTS
              and h1 is not None and n1 >= MIN_POINTS and h1 >= h0 - STAYED_M)
    if not holding:
        ok, why = False, "the fingers closed on nothing"
    elif stayed:
        ok, why = False, "what stood there is still standing there: it did not come with the gripper"
    else:
        ok, why = True, ""
    return {"ok": ok, "why": why, "holding": holding, "points_under_hand": carried,
            "spot_height_before_m": None if h0 is None else round(h0, 4), "points_before": n0,
            "spot_height_after_m": None if h1 is None else round(h1, 4), "points_after": n1,
            "basis": ("fingers and depth" if carried >= MIN_POINTS or (n0 >= MIN_POINTS and n1 >= MIN_POINTS)
                      else "fingers")}


def _risen(before, after, xy, r, cap: float) -> Tuple[int, int]:
    """Over the target, cell by cell (``CELL_M`` square), from every camera: of the cells seen
    both before and after, how many stand at least ``RISEN_M`` higher after, and how many
    there are.  A cell seen only after tells nothing: before, the hand was elsewhere and the
    wrist camera saw other things."""
    import numpy as np

    def cells(cloud):
        if not len(cloud):
            return {}
        near = (np.hypot(cloud[:, 0] - xy[0], cloud[:, 1] - xy[1]) < r) & (cloud[:, 2] < cap)
        pts = cloud[near]
        keys = np.floor(pts[:, :2] / CELL_M).astype(int)
        out = {}
        for key in {tuple(k) for k in keys}:
            h = pts[(keys[:, 0] == key[0]) & (keys[:, 1] == key[1]), 2]
            if len(h) >= CELL_POINTS:
                out[key] = float(np.percentile(h, TOP_PCT))
        return out

    b, a = cells(before), cells(after)
    both = [k for k in a if k in b]
    return sum(a[k] - b[k] >= RISEN_M for k in both), len(both)


def _fixed_pair(ctx, first, second):
    """The fixed camera's two views, before and after, pixel by pixel: (height before, height
    after, comparable, station x, station y of what is seen after); None without them."""
    import numpy as np

    fixed = "head" if ctx.station.two_arms else "front"
    fb = next((f for f in first if getattr(f, "camera", None) == fixed), None)
    fa = next((f for f in second if getattr(f, "camera", None) == fixed), None)
    view = getattr(ctx.perception, "view", None)
    if fb is None or fa is None or view is None:
        return None
    try:
        vb, va = view(fb), view(fa)
        hb, ha = np.asarray(vb.height, dtype=float), np.asarray(va.height, dtype=float)
        pts = np.asarray(va.points, dtype=float)
        robot = np.asarray(vb.robot, dtype=bool) | np.asarray(va.robot, dtype=bool)
    except Exception:
        return None
    if hb.shape != ha.shape or pts.shape[:2] != ha.shape:
        return None
    origin = np.asarray(ctx.station.origin, dtype=float)
    ok = np.isfinite(hb) & np.isfinite(ha) & ~robot
    return hb, ha, ok, pts[..., 0] - origin[0], pts[..., 1] - origin[1]


def _n_cells(sx, sy, mask) -> int:
    import numpy as np

    if not mask.any():
        return 0
    keys = np.floor(np.column_stack([sx[mask], sy[mask]]) / CELL_M).astype(np.int64)
    _, counts = np.unique(keys, axis=0, return_counts=True)
    return int((counts >= CELL_POINTS).sum())


def _appeared(ctx, first, second, xy, r, below: float) -> Tuple[int, int]:
    """From the fixed camera, pixel by pixel (it does not move): how many ``CELL_M`` cells of
    the target now show a surface at least ``NEW_M`` higher in a pixel's line of sight than
    before -- something came there, on the target or into it (a block sunk into a box is
    seen where the box's far wall was) -- and how many cells could be compared (seen before
    and after, not the robot).  Only what lies lower than ``below``, the hand: above it is
    the arm, which the robot's own mask leaves out only from 0.35 m up.  (0, 0) without such
    a camera's depth."""
    import numpy as np

    pair = _fixed_pair(ctx, first, second)
    if pair is None:
        return 0, 0
    hb, ha, ok, sx, sy = pair
    with np.errstate(invalid="ignore"):
        near = ok & (np.hypot(sx - xy[0], sy - xy[1]) < r) & (ha < below)
        new = near & (ha >= hb + NEW_M)
    return _n_cells(sx, sy, new), _n_cells(sx, sy, near)


def _hand_z(ctx, arm) -> float:
    """How high the hand (its tool point) is over the table now."""
    try:
        return float(ctx.station.describe_pose(ctx.tcp(arm))["xyz_m"][2])
    except Exception:
        return float("inf")


def _place_verdict(ctx, arm, xy, top_z, r, before, after, first, second, held_before) -> dict:
    """The fingers held something and let it go, and something new is on the target: by the
    fixed camera's pixels, or, the target hidden from it, by the cells every camera saw both
    times.  The target hidden from every camera, nothing is claimed."""
    holding = _holding(ctx, arm)
    hand = _hand_z(ctx, arm)
    new, compared = _appeared(ctx, first, second, xy, r + CELL_M, hand)
    basis = "fixed camera"
    if compared < MIN_COMPARED:
        new, compared = _risen(before, after, xy, r + CELL_M, min(top_z + 0.25, hand))
        basis = "every camera"
    h0, n0 = _height(before, xy, r, below=top_z + 0.02)
    h1, n1 = _height(after, xy, r, below=top_z + 0.25)
    if not held_before:
        ok, why = False, "the fingers held nothing to place"
    elif holding:
        ok, why = False, "the fingers still hold something"
    elif compared < MIN_COMPARED:
        ok, why, basis = True, "", "target unseen"
    elif new < (NEW_CELLS if basis == "fixed camera" else RISEN_CELLS):
        ok, why = False, "nothing new lies on the target"
    else:
        ok, why = True, ""
    return {"ok": ok, "why": why, "held_before": held_before, "holding": holding,
            "cells_new": new, "cells_compared": compared, "basis": basis,
            "target_height_before_m": None if h0 is None else round(h0, 4), "points_before": n0,
            "target_height_after_m": None if h1 is None else round(h1, 4), "points_after": n1}


def _dropped(ctx, arm, first, second, xy, r) -> Optional[Tuple[float, float, float]]:
    """Where something came to lie near the target, outside it, by the fixed camera's pixels:
    of the ``GRID_M`` squares within ``SEARCH_R_M``, the one where most pixels show something
    new (station x, y of what is new there, its top); None when nothing new is seen."""
    import numpy as np

    pair = _fixed_pair(ctx, first, second)
    if pair is None:
        return None
    hb, ha, ok, sx, sy = pair
    with np.errstate(invalid="ignore"):
        d = np.hypot(sx - xy[0], sy - xy[1])
        new = (ok & (ha >= hb + NEW_M) & (d >= r) & (d <= SEARCH_R_M)
               & (ha < _hand_z(ctx, arm)))
    if _n_cells(sx, sy, new) < NEW_CELLS:
        return None
    keys = np.floor(np.column_stack([sx[new], sy[new]]) / GRID_M).astype(np.int64)
    uniq, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    best = inverse.reshape(-1) == int(np.argmax(counts))
    xs, ys, hs = sx[new], sy[new], ha[new]
    cx, cy = float(np.mean(xs[best])), float(np.mean(ys[best]))
    for _ in range(2):                       # from that square to the middle of what is there
        there = np.hypot(xs - cx, ys - cy) < THING_R_M
        cx, cy = float(np.mean(xs[there])), float(np.mean(ys[there]))
    there = np.hypot(xs - cx, ys - cy) < THING_R_M
    return cx, cy, float(np.percentile(hs[there], TOP_PCT))


# -- what the simulator knows (the log only) ------------------------------------------
def _truth_pick(ctx, arm) -> dict:
    try:
        st = ctx._ep.engine.state().arm(ctx.station.arm(arm).arm_id)
        held = getattr(st, "holding", None)
        _held(ctx)[arm] = held
        return {"holding": held, "ok": held is not None}
    except Exception as exc:
        return {"error": repr(exc)[:200]}


def _truth_place(ctx, arm, xy, top_z, r) -> dict:
    try:
        world = ctx._ep.engine.state()
        body = _held(ctx).get(arm)
        still = getattr(world.arm(ctx.station.arm(arm).arm_id), "holding", None)
        out = {"placed": body, "still_holding": still}
        if body and body in world.objects:
            x, y, z = ctx.station.to_station(world.objects[body].pose.pos)
            d = math.hypot(x - xy[0], y - xy[1])
            out.update({"xyz_m": [round(x, 4), round(y, 4), round(z, 4)],
                        "from_target_m": round(d, 4),
                        "ok": still is None and d <= max(r, 0.05)})
        return out
    except Exception as exc:
        return {"error": repr(exc)[:200]}


def _note_held(ctx, arm) -> None:
    """The body the simulator has in the hand before a place (the log only): an agent may
    have taken it with robot_gripper rather than a checked pick."""
    try:
        _held(ctx)[arm] = getattr(ctx._ep.engine.state().arm(ctx.station.arm(arm).arm_id),
                                  "holding", None)
    except Exception:
        pass


def _held(ctx) -> dict:
    ep = ctx._ep
    if not hasattr(ep, "_check_held"):
        ep._check_held = {}
    return ep._check_held


# -- the log --------------------------------------------------------------------------
def _save(ctx, frames, n: int, when: str) -> dict:
    folder = getattr(ctx._ep, "log_dir", None)
    out = {}
    for f in frames:
        out[f.camera] = {"frame_id": f.id}
        if folder is None or f.image is None:
            continue
        try:
            import numpy as np
            from PIL import Image

            arr = np.asarray(f.image)
            if arr.dtype != np.uint8:
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
            path = folder / "checks" / f"{n:03d}_{when}_{f.camera}.png"
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(arr[..., :3]).save(path)
            out[f.camera]["image"] = str(path.relative_to(folder))
        except Exception as exc:
            out[f.camera]["image_error"] = repr(exc)[:120]
    return out


def _log(ctx, record: dict, shots) -> None:
    folder = getattr(ctx._ep, "log_dir", None)
    n = getattr(ctx._ep, "_check_n", 0) + 1
    ctx._ep._check_n = n
    ep = ctx._ep
    try:
        episode = f"{ep.inst.task_id}_{ep.inst.level}_s{ep.inst.seed}_{ep.profile.id.replace('/', '-')}"
    except Exception:
        episode = ""
    record = {"n": n, "episode": episode,
              "sim_time": round(float(getattr(ctx, "sim_time", 0.0) or 0.0), 3), **record}
    record["views"] = {when: _save(ctx, frames, n, when) for when, frames in shots}
    if folder is None:
        return
    try:
        folder.mkdir(parents=True, exist_ok=True)       # the episode makes it only at its end
        with open(folder / LOG, "a") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def _said(res: ToolResult, after: str, verdict: dict, extra: dict) -> ToolResult:
    """The skill's result with a failed check said in it (a passed one adds nothing)."""
    note = {"after": after, "ok": verdict["ok"], **extra}
    if verdict["ok"] and not extra:
        return res
    if not verdict["ok"]:
        note["why"] = verdict["why"]
    text = res.text + f"\ncheck after the {after}: " + (
        verdict["why"] if not verdict["ok"] else "") + extra.get("then", "")
    return replace(res, text=text.rstrip(), data={**res.data, "check": note})


# -- the checked skills ------------------------------------------------------------------
def checked_pick(ctx, arm: str, p, estimate, mode: str) -> ToolResult:
    from ruikit.actions.skill import _pick_handle

    top, yaw, ext = estimate
    xy, top_z = (top[0], top[1]), top[2]
    first = _look(ctx, arm)
    before = _cloud(ctx, first)
    res = _run(ctx, _pick_handle(ctx, arm, p, top, yaw, ext))
    if res.is_error or res.data.get("status") in (None, "running"):     # not over: no check
        return res
    second = _look(ctx, arm)
    v = _pick_verdict(ctx, arm, xy, top_z, before, _cloud(ctx, second))
    record = {"kind": "pick", "mode": mode, "arm": arm, "target_xyz_m": [round(c, 4) for c in top],
              "approach": getattr(p, "approach", "above"), "status": res.data.get("status"),
              "verdict": v, "truth": _truth_pick(ctx, arm)}
    shots = [("before", first), ("after", second)]
    extra = {}
    # picked again only with nothing in the fingers: a pick opens them first, and would
    # drop what a check took for the wrong thing
    if not v["ok"] and mode == "correct" and not v["holding"]:
        res = _run(ctx, _pick_handle(ctx, arm, p, top, yaw, ext))
        third = _look(ctx, arm)
        v2 = _pick_verdict(ctx, arm, xy, top_z, before, _cloud(ctx, third))
        record["corrected"] = {"verdict": v2, "truth": _truth_pick(ctx, arm)}
        shots.append(("corrected", third))
        extra = {"corrected": True, "ok_after_correcting": v2["ok"],
                 "then": "; the skill picked once more: " + ("now the fingers hold it" if v2["ok"]
                                                             else f"still not -- {v2['why']}")}
    _log(ctx, record, shots)
    return _said(res, "pick", v, extra)


def checked_place(ctx, arm: str, p, top, r: float, mode: str) -> ToolResult:
    from ruikit.actions.skill import PickParams, _pick_handle, _place_handle

    xy, top_z = (top[0], top[1]), top[2]
    _note_held(ctx, arm)                     # what the hand holds now, however it took it
    held_before = _holding(ctx, arm)
    first = _look(ctx, arm)
    before = _cloud(ctx, first)
    res = _run(ctx, _place_handle(ctx, arm, p, top))
    if res.is_error or res.data.get("status") in (None, "running"):     # not over: no check
        return res
    second = _look(ctx, arm)
    after = _cloud(ctx, second)
    v = _place_verdict(ctx, arm, xy, top_z, r, before, after, first, second, held_before)
    record = {"kind": "place", "mode": mode, "arm": arm, "target_xyz_m": [round(c, 4) for c in top],
              "target_r_m": r, "status": res.data.get("status"), "verdict": v,
              "truth": _truth_place(ctx, arm, xy, top_z, r)}
    shots = [("before", first), ("after", second)]
    extra = {}
    if not v["ok"] and mode == "correct":
        did, placed = "", False
        if not held_before:
            did = "had nothing to place again"
        elif v["holding"]:
            res = _run(ctx, _place_handle(ctx, arm, p, top))
            did, placed = "placed once more", True
        else:
            spot = _dropped(ctx, arm, first, second, xy, r)
            if spot is not None:
                pick = PickParams(target={"xyz_m": list(spot)}, arm=arm)
                got = _run(ctx, _pick_handle(ctx, arm, pick, spot, None, None))
                if not got.is_error and _holding(ctx, arm):
                    _truth_pick(ctx, arm)
                    res = _run(ctx, _place_handle(ctx, arm, p, top))
                    did = f"picked it up {math.hypot(spot[0] - xy[0], spot[1] - xy[1]) * 100:.0f} cm away and placed it again"
                    placed = True
                else:
                    did = "tried to pick it up where it fell and missed"
            else:
                did = "found nothing fallen near the target to pick up"
        if placed:                           # placed again: checked again
            third = _look(ctx, arm)
            v2 = _place_verdict(ctx, arm, xy, top_z, r, before, _cloud(ctx, third), first, third,
                                held_before)
            shots.append(("corrected", third))
            then = f"; the skill {did}: " + ("now it lies on the target" if v2["ok"]
                                            else f"still not -- {v2['why']}")
        else:
            v2, then = v, f"; the skill {did}"
        record["corrected"] = {"did": did, "placed_again": placed, "verdict": v2,
                               "truth": _truth_place(ctx, arm, xy, top_z, r)}
        extra = {"corrected": True, "ok_after_correcting": v2["ok"], "then": then}
    _log(ctx, record, shots)
    return _said(res, "place", v, extra)
