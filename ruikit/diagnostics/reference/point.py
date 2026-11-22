"""The reference through the image point: robot_move_to_pixel."""

from __future__ import annotations

import math
from typing import Dict, Optional

from ruibench.core.types import Pose
from ruibench.engines.base import project
from ruibench.interfaces.commands import (
    Cancel,
    MoveDelta,
    MoveTo,
    SetGripper,
)
from ruikit.diagnostics.reference.base import Expressor, _arm_name, _grip_args
from ruikit.diagnostics.reference.grounding import SURFACE_TOL, Grounded, _with_notes
from ruikit.env.station import rpy_deg_from_quat


class PointExpressor(Expressor):
    """Say each commanded pose as a pixel: find a pixel whose lifted surface
    point lies under the target (within ``xy_tol``), then the height above it."""

    style = "point"
    xy_tol = 0.003
    #: from the first pixel that will do, the search goes on this finely over
    #: this much of the line above the target: a perfect pointer points at the
    #: best pixel, and a pixel is given to a thousandth (CR1's 2 mm holes and
    #: CR2's socket were missed by the 3 mm the first one may be off, on Isaac)
    refine_step, refine_span = 0.00025, 0.003

    def express(self, ep, acts):
        for row in self.rounds(acts):
            calls = []
            for c in row:
                arm = _arm_name(getattr(c, "arm", "") or "right")
                if isinstance(c, Cancel):
                    calls.append(("robot_stop", {"arm": arm} if c.arm else {}))
                elif isinstance(c, SetGripper):
                    calls.append(("robot_gripper", _grip_args(ep, c)))
                else:
                    target = self.target_of(ep, c)
                    args = self._pixel_for(ep, arm, target)
                    if args is None:
                        self.unexpressed.append(
                            f"no pixel lands under {ep.station.describe_pose(target)['xyz_m']}")
                        continue
                    calls.append(("robot_move_to_pixel", args))
            if calls:
                ep.call_batch(calls)
            if ep.over:
                return

    @staticmethod
    def _lifted(ep, frame, point):
        """(how far off the target's place the surface seen at ``point``'s
        pixel is, that surface point, the pixel) -- or None where the point is
        out of the picture or nothing is seen there."""
        uv = project(frame.pose, point, frame.width, frame.height, frame.focal_mm, frame.apertures)
        if uv is None or not (0 <= uv[0] <= frame.width and 0 <= uv[1] <= frame.height):
            return None
        try:
            lifted = ep.perception.lift(frame, uv[0], uv[1])
        except Exception:
            return None
        hit = ep.station.to_world(lifted["xyz_m"])
        return math.hypot(hit[0] - point[0], hit[1] - point[1]), hit, uv

    def _pixel_for(self, ep, arm, target: Pose) -> Optional[dict]:
        """Take fresh pictures (a real tool call), then search -- with the
        server's own ray cast, as a perfect pointer would -- for a pixel whose
        first surface lies right under the target.  Only the final
        robot_move_to_pixel is what the interface has to express."""
        look = ep.call("camera_look", {"cameras": None})
        if look.is_error:
            return None
        x, y, z_top = target.pos
        best = None
        for info in look.data["frames"]:
            frame = ep.frames.get(info["frame_id"])
            z = z_top + 0.25                  # the surface may stand above the target
            while z >= ep.station.table_height - 1e-6:
                at = z
                z -= 0.003
                found = self._lifted(ep, frame, (x, y, at))
                if found is None:
                    continue
                miss, hit, uv = found
                if miss <= self.xy_tol and abs(hit[2] - at) <= 0.004:
                    steps = int(round(self.refine_span / self.refine_step))
                    for k in range(-steps, steps + 1):
                        near = at + k * self.refine_step
                        other = self._lifted(ep, frame, (x, y, near))
                        if other is not None and other[0] < miss and abs(other[1][2] - near) <= 0.004:
                            miss, hit, uv = other
                    height = z_top - hit[2]
                    if -0.05 <= height <= 0.5 and (best is None or miss < best[0]):
                        best = (miss, {"arm": arm, "frame_id": frame.id, "u": round(uv[0], 3),
                                       "v": round(uv[1], 3), "height_m": round(height, 4),
                                       "rpy_deg": list(rpy_deg_from_quat(target.quat))})
                    break
        if best is not None:
            return best[1]
        # no surface lies right under the target (the middle of an opening, a
        # point in the air): say it as the ray crossing the target's height --
        # the ray of the camera the robot believes it has, which is the one
        # the tool takes the pixel's ray from (through the camera's true pose
        # the hand went as far off as the calibration is: millimetres)
        believed = getattr(ep.perception, "believed", None)
        for info in look.data["frames"]:
            frame = ep.frames.get(info["frame_id"])
            uv = project(believed(frame) if believed else frame.pose, target.pos, frame.width,
                         frame.height, frame.focal_mm, frame.apertures)
            if uv is not None and 0 <= uv[0] <= frame.width and 0 <= uv[1] <= frame.height:
                height = target.pos[2] - ep.station.table_height
                if -0.05 <= height <= 0.5:
                    return {"arm": arm, "frame_id": frame.id, "u": round(uv[0], 3),
                            "v": round(uv[1], 3), "height_m": round(height, 4), "mode": "plane",
                            "rpy_deg": list(rpy_deg_from_quat(target.quat))}
        return None


class GroundedPoint(_with_notes(PointExpressor)):
    """Points in the frame the thing was seen in: at the surface right under the
    target when the target is over the thing, else along the ray that crosses
    the target's height."""

    def on_commands(self, ep, cmds):
        """Estimate what the turn needs, then measure every surface it will
        point at in one reply of vision_lift calls."""
        g = self.grounder
        moves = [c for c in cmds if isinstance(c, (MoveTo, MoveDelta))]
        self._lifted: Dict[tuple, dict] = {}
        if not moves:
            return
        g.prepare([self.target_of(ep, c) for c in moves])
        g.look()                # a move is said in a picture: it needs one, even by nothing
        asks = []
        for c in moves:
            got = g.ground(self.target_of(ep, c))
            spot = self._surface_spot(ep, got) if got is not None else None
            if spot is not None and spot not in asks:
                asks.append(spot)
        if asks:
            answers = ep.call_batch([("vision_lift", {"frame_id": fid, "u": u, "v": v})
                                     for fid, u, v in asks])
            for spot, r in zip(asks, answers):
                if not r.is_error:
                    self._lifted[spot] = r.data

    def _surface_spot(self, ep, got: "Grounded"):
        """The pixel to measure when the target is right over the thing: its top."""
        if got.seen is None or math.hypot(got.pose.pos[0] - got.seen.xyz[0],
                                          got.pose.pos[1] - got.seen.xyz[1]) >= 0.012:
            return None
        fr = ep.frames.get(got.seen.frame_id)
        top = (got.pose.pos[0], got.pose.pos[1], got.seen.xyz[2] + got.seen.size[2])
        uv = project(fr.pose, top, fr.width, fr.height, fr.focal_mm, fr.apertures)
        if uv and 0 <= uv[0] <= fr.width and 0 <= uv[1] <= fr.height:
            return (fr.id, round(uv[0], 1), round(uv[1], 1))
        return None

    def express(self, ep, acts):
        g = self.grounder
        for row in self.rounds(acts):
            calls, notes = [], []
            for c in row:
                arm = _arm_name(getattr(c, "arm", "") or "right")
                if isinstance(c, Cancel):
                    calls.append(("robot_stop", {"arm": arm} if c.arm else {}))
                elif isinstance(c, SetGripper):
                    calls.append(("robot_gripper", _grip_args(ep, c)))
                else:
                    got = g.ground(self.target_of(ep, c))
                    if got is None:
                        self.unexpressed.append("nothing estimated to point by")
                        continue
                    args, how = self._point(ep, arm, got)
                    if args is None:
                        self.unexpressed.append(f"no frame shows {ep.station.describe_pose(got.pose)['xyz_m']}")
                        continue
                    notes.append((len(calls), f"{got.note}；{how}"))
                    calls.append(("robot_move_to_pixel", args))
            for k, text in notes:
                self.note(ep, text, k)
            if calls:
                ep.call_batch(calls)
            if ep.over:
                return

    def _point(self, ep, arm, got: Grounded):
        target = got.pose
        rpy = list(rpy_deg_from_quat(target.quat))
        spot = self._surface_spot(ep, got)
        measured = getattr(self, "_lifted", {}).get(spot) if spot is not None else None
        if spot is not None and measured is None and spot not in getattr(self, "_lifted", {}):
            # not measured with the turn (a pose the turn did not plan): measure it now
            r = ep.call("vision_lift", {"frame_id": spot[0], "u": spot[1], "v": spot[2]})
            measured = None if r.is_error else r.data
        if measured is not None:
            hit = ep.station.to_world(measured["xyz_m"])
            top = (target.pos[0], target.pos[1], got.seen.xyz[2] + got.seen.size[2])
            # the reading must land where it was aimed, within the estimate's own
            # error: from the slanted front camera a ray aimed at a thin thing's top
            # (a peg, a rod) meets its side or what is behind it, a few millimetres
            # off, and a move by that surface lands off by as much (WK1's discs
            # went down 5-8 mm beside the peg, again and again); else point by height
            if math.hypot(hit[0] - top[0], hit[1] - top[1]) < SURFACE_TOL:
                height = round(target.pos[2] - hit[2], 4)
                if -0.05 <= height <= 0.5:
                    return ({"arm": arm, "frame_id": spot[0], "u": spot[1], "v": spot[2],
                             "height_m": height, "rpy_deg": rpy}, f"指它的顶面，高出 {height} m")
        frames = [ep.frames.get(got.seen.frame_id)] if got.seen else []
        frames += [f for f in self.grounder._frames() if not frames or f.id != frames[0].id]
        table = ep.station.table_height
        for fr in frames:
            uv = project(fr.pose, target.pos, fr.width, fr.height, fr.focal_mm, fr.apertures)
            height = round(target.pos[2] - table, 4)
            if uv and 0 <= uv[0] <= fr.width and 0 <= uv[1] <= fr.height and -0.05 <= height <= 0.5:
                return ({"arm": arm, "frame_id": fr.id, "u": round(uv[0], 1), "v": round(uv[1], 1),
                         "height_m": height, "mode": "plane", "rpy_deg": rpy},
                        f"指那一点、离桌面 {height} m")
        return None, ""
