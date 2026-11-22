"""Follow a thing over time: re-detected each sample, or matched by how it looks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component, Implementation
from ruikit.perception.vision import PERCEPTION_SECONDS


class TrackSession:
    """Tracking by detection (``redetect``): every 1/hz seconds, a fresh frame
    from the same camera; the object is the same-label detection nearest the
    last position, if it moved less than ``gate_m`` since then.  Hidden, gone
    or jumped: the track is lost -- it never follows through an occlusion."""

    def __init__(self, ctx, camera: str, label: str, xyz, hz: float, gate_m: float) -> None:
        self.camera, self.label = camera, label
        self.hz, self.gate = hz, gate_m
        self.last = tuple(xyz)
        self.last_t = ctx.sim_time
        self.velocity = (0.0, 0.0, 0.0)
        self.lost = False
        self.next_t = ctx.sim_time + 1.0 / hz
        self.samples = 1
        self.stopped = False

    def tick(self, episode, dt: float) -> None:
        from ruikit.env.context import ToolContext

        if self.stopped or self.lost or episode.sim_time + 1e-9 < self.next_t:
            return
        self.next_t = episode.sim_time + 1.0 / self.hz
        ctx = ToolContext(episode, "vision_track")
        frame = ctx.capture(self.camera)
        best, best_d = None, None
        for obj in ctx.perception.detect(frame, self.label):
            if obj["label"] != self.label:
                continue
            xyz = ctx.perception.pose(frame, obj["id"])["xyz_m"]
            d = sum((a - b) ** 2 for a, b in zip(xyz, self.last)) ** 0.5
            if best_d is None or d < best_d:
                best, best_d = xyz, d
        if best is None or best_d > self.gate:
            self.lost = True
            return
        span = max(1e-6, episode.sim_time - self.last_t)
        self.velocity = tuple((a - b) / span for a, b in zip(best, self.last))
        self.last, self.last_t = tuple(best), episode.sim_time
        self.samples += 1

    def request_stop(self, episode) -> None:
        self.stopped = True

    def close(self, episode) -> None:
        self.stopped = True
        if self in episode.sessions:
            episode.sessions.remove(self)


#: template tracking: a match counts from this normalised correlation up
TEMPLATE_MIN_SCORE = 0.5


#: ... and is looked for this many template sizes round the last place
TEMPLATE_SEARCH = 1.5


class TemplateTrack(TrackSession):
    """Tracking by appearance (``template``): no detector after the start.
    Each sample looks for how the thing looked at the start (and at the last
    sample) round where it was last, by normalised correlation of the grey
    picture, and lifts the best match's middle with the depth.  A poor match
    (an occlusion, a turn that changes how it looks) loses the track."""

    def __init__(self, ctx, frame, box, xyz, label, hz: float, gate_m: float) -> None:
        super().__init__(ctx, frame.camera, label, xyz, hz, gate_m)
        grey = _grey(frame.image)
        u0, v0, u1, v1 = (int(round(c)) for c in box)
        self.first = grey[v0:v1, u0:u1].copy()
        self.latest = self.first
        self.uv = ((u0 + u1) / 2.0, (v0 + v1) / 2.0)
        self.score = 1.0

    def tick(self, episode, dt: float) -> None:
        import cv2

        from ruikit.env.context import ToolContext

        if self.stopped or self.lost or episode.sim_time + 1e-9 < self.next_t:
            return
        self.next_t = episode.sim_time + 1.0 / self.hz
        ctx = ToolContext(episode, "vision_track")
        frame = ctx.capture(self.camera)
        grey = _grey(frame.image)
        th, tw = self.first.shape
        reach_u, reach_v = int(TEMPLATE_SEARCH * tw) + 4, int(TEMPLATE_SEARCH * th) + 4
        u0 = max(0, int(self.uv[0] - tw / 2 - reach_u))
        v0 = max(0, int(self.uv[1] - th / 2 - reach_v))
        u1 = min(grey.shape[1], int(self.uv[0] + tw / 2 + reach_u))
        v1 = min(grey.shape[0], int(self.uv[1] + th / 2 + reach_v))
        window = grey[v0:v1, u0:u1]
        best = None
        for template in (self.first, self.latest):
            if window.shape[0] < template.shape[0] or window.shape[1] < template.shape[1]:
                continue
            res = cv2.matchTemplate(window, template, cv2.TM_CCOEFF_NORMED)
            _, score, _, at = cv2.minMaxLoc(res)
            if best is None or score > best[0]:
                best = (score, at, template.shape)
        if best is None or best[0] < TEMPLATE_MIN_SCORE:
            self.lost = True
            return
        score, (bu, bv), (h, w) = best
        uv = (u0 + bu + w / 2.0, v0 + bv + h / 2.0)
        try:
            xyz = tuple(ctx.perception.lift(frame, uv[0], uv[1])["xyz_m"])
        except ToolError:
            self.lost = True
            return
        if sum((a - b) ** 2 for a, b in zip(xyz, self.last)) ** 0.5 > self.gate:
            self.lost = True
            return
        span = max(1e-6, episode.sim_time - self.last_t)
        self.velocity = tuple((a - b) / span for a, b in zip(xyz, self.last))
        self.last, self.last_t, self.uv, self.score = xyz, episode.sim_time, uv, round(score, 3)
        top, left = int(round(uv[1] - h / 2)), int(round(uv[0] - w / 2))
        self.latest = grey[top:top + h, left:left + w].copy()
        self.samples += 1


def _grey(image):
    import numpy as np

    from ruikit.env.frames import to_array

    arr = to_array(image)
    if arr is None:
        raise ToolError("not_supported", "template tracking needs pictures (the mock engine "
                        "renders none)")
    arr = np.asarray(arr)[..., :3].astype(np.float32)
    return (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.uint8)


@dataclass(frozen=True)
class TrackParams:
    action: Literal["start", "read", "stop"] = param(
        "'start' a track on an object, 'read' its latest estimate, 'stop' it")
    frame_id: Optional[str] = param("start: the frame of the object id; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply",
                                   default=None)
    id: Optional[str] = param("start: a frame-local object id", default=None)
    track: Optional[str] = param("read / stop: the track id from start", default=None)
    hz: float = param("start: samples per second (each costs a render)", default=5.0,
                      minimum=1.0, maximum=10.0)


@tool(name="vision_track",
      data="start: track, label, xyz_m; read: track, xyz_m, velocity_mps [vx, vy, vz], age_s, lost, samples; stop: track",
      family="perceive", params=TrackParams, reads=("frames",),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"action": "start", "frame_id": "f3.front", "id": "f3.front#2"},
                {"action": "read", "track": "trk1"}],
      description="""Keep following one visible object with the same camera while you do
      other things: 'start' on a frame-local id returns a track id; 'read' returns its latest
      position (station frame, metres), velocity (m/s), how old that estimate is, and whether
      the track is lost; 'stop' ends it. The track samples fresh pictures hz times a second
      and is lost as soon as the object is hidden or jumps -- it never guesses through an
      occlusion. Start and read cost 0.1 s of simulated time; it moves nothing.""")
def vision_track(ctx, p: TrackParams) -> ToolResult:
    tracks = ctx.memory.setdefault("_tracks", {})
    if p.action == "start":
        if not p.frame_id or not p.id:
            raise ToolError("invalid_args", "start needs frame_id and id")
        frame = ctx.frame(p.frame_id)
        pose = ctx.perception.pose(frame, p.id)
        if ctx.params.get("impl", "redetect") == "template":
            box = next((d["box"] for d in ctx.perception.detect(frame, pose["label"])
                        if d["id"] == p.id), None)
            if box is None:
                box = ctx.perception.box_of(frame, p.id)
            session = TemplateTrack(ctx, frame, box, pose["xyz_m"], pose["label"], p.hz,
                                    gate_m=max(0.05, 0.6 / p.hz))
        else:
            session = TrackSession(ctx, frame.camera, pose["label"], pose["xyz_m"], p.hz,
                                   gate_m=max(0.05, 0.6 / p.hz))
        tid = f"trk{len(tracks) + 1}"
        tracks[tid] = session
        ctx.add_session(session)
        return ToolResult(data={"track": tid, "label": pose["label"], "xyz_m": pose["xyz_m"]},
                          text=f"tracking {pose['label']} as {tid}")
    session = tracks.get(p.track or "")
    if session is None:
        raise ToolError("stale_handle", f"no track {p.track!r}")
    if p.action == "stop":
        session.stopped = True
        return ToolResult(data={"track": p.track}, text=f"stopped {p.track}")
    age = round(ctx.sim_time - session.last_t, 3)
    return ToolResult(data={"track": p.track, "xyz_m": [round(v, 4) for v in session.last],
                            "velocity_mps": [round(v, 4) for v in session.velocity],
                            "age_s": age, "lost": session.lost, "samples": session.samples},
                      text=(f"{p.track} lost" if session.lost else
                            f"{p.track} at {[round(v, 3) for v in session.last]} ({age} s old)"))


COMPONENT = Component(
    "track", "跟踪", "keep following one thing while doing other things",
    ("vision_track",),
    (Implementation("redetect", "每帧重新检测，取最近的同类", "detect anew each sample, keep the "
                    "nearest of the same word", params={"vision_track": {"impl": "redetect"}}),
     Implementation("template", "按上一帧的样子在图上匹配", "match how it looked last time "
                    "(template), no detector", params={"vision_track": {"impl": "template"}})),
    default="redetect")


#: what the agent is told of vision_track when it matches by how the thing looks
TEMPLATE_DESCRIPTION = " ".join("""Keep following one visible object with the same
    camera while you do other things, by how it looks: 'start' on a frame-local id returns
    a track id; each sample finds the picture patch most like the object's (as it looked at
    the start, or at the last sample) near where it was, and lifts its middle with the
    depth; 'read' returns its latest position (station frame, metres: the surface point at
    its middle, not its centre), velocity (m/s), how old that estimate is, and whether the
    track is lost; 'stop' ends it. No detector runs after the start; a poor match -- hidden,
    turned, gone -- loses the track, which never guesses through an occlusion. Start and
    read cost 0.1 s of simulated time; it moves nothing.""".split())
