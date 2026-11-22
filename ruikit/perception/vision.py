"""The perception tools: what is where in a frame.

Every tool here answers about one frame (``frame_id``) and only about what that
frame shows.  Object ids are frame-local (``f12.front#3``); take a new picture
and the numbering starts over.  The answers come from the perception backend:
by default from the picture and its depth (``model``; settings in
``ruikit.perception.settings``); on the mock engine, which renders none, the
``synthetic`` test double stands in.  What the perception backend found is also
kept for the scene memory (``memory_scene``).

Three more vision tools belong to Harness components (``ruikit.harness.components``),
each with several implementations chosen in the interface id: ``vision_marks``
(dots, outlines or boxes), ``vision_point`` (the detector, or a vision-language
model asked for a point) and ``vision_track`` (detect again, or match how the
thing looked).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool

PERCEPTION_SECONDS = 0.1

#: The same tools under ``perception: model``: what each says of itself when
#: its answers come from the picture and the depth camera.  The parameters
#: and the shape of the answers do not change.
FROM_PIXELS = {
    "vision_detect": """Find objects in one frame with an open-vocabulary detector run on the
        picture. Give the thing's name in your own words ('coin', 'blue ring', 'red cross'): a
        plain, specific phrase works best; '' looks for the task's own kinds of thing and for
        any 'object' all at once, and is less reliable: it can find fewer, or give a thing the
        wrong one of those words. Returns each object found: its frame-local id
        (like 'f3.front#2', valid only for that frame), as label the phrase it has matched best
        so far in this frame, the detector's score for your phrase ({low} to 1: below {doubtful} is
        doubtful, and a label other than your phrase means it looks more like that), a plain
        colour when it has one, the pixel box [u0, v0, u1, v1] of its mask and a pixel on it; cut_by_frame marks a thing the
        picture's edge cuts, whose box and pose are of the part in view. It is a detector's answer,
        not the truth: it can miss a thing, merge two that touch, or report one under the wrong
        word; nothing of the robot itself is reported, nor what is flat as the table. Takes
        0.1 s of simulated time and moves nothing.""",
    "vision_segment": """Segment an object in one frame from a text, point or box prompt
        (give exactly one), with a segmentation model run on the picture. Returns the object's
        frame-local id, the phrase it matched (or 'object' when it was found by a point or a box
        alone), its pixel box and its mask's area in pixels; a point prompt returns the thing
        that pixel lies on. A point on the bare table, or on the robot, returns nothing. Takes
        0.1 s of simulated time and moves nothing.""",
    "vision_point": """Point at things matching a description in one frame: an
        open-vocabulary detector run on the picture returns one pixel [u, v] on each match, with
        the phrase it matched. Use it to get a pixel for robot_move_to_pixel or vision_lift when
        you know what you want but not where it is in the image. It can miss a thing or point at
        the wrong one: check the pixel against the picture. Takes 0.1 s of simulated time and
        moves nothing.""",
    "vision_lift": """Turn a pixel into a 3-D point with that frame's depth image: the surface
        the depth camera read at the pixel, in the station frame (metres; x forward, y left, z
        up from the table top), with the surface normal there and, as "on", 'table' for the
        table top or the detector's word for the thing the pixel lies on, when it has one. It is
        a measurement, not the truth: the depth has about 1 mm + 0.3 % of the distance of
        noise, no reading along the edges of things (not_visible there: pick a pixel inside a
        surface), and the point is placed with the camera's calibration, which is off by about
        3 mm and 0.3 degrees (5 mm more a metre away). The depth and the camera pose are those of
        the moment the frame was taken, so a pixel in an old picture gives where things were
        then. Takes 0.1 s of simulated time and moves nothing.""",
    "vision_pose": """Estimate a detected object's pose from one frame, from its mask and
        the depth image: the middle of its footprint and of its height in the station frame
        (metres), the heading yaw_deg about the vertical of the longer side of the smallest box
        around its footprint (between -90 and 90; it says nothing for a round thing, and for a
        cross or a star that box can lie along the diagonals), how far its top face is tilted and which
        way it leans (tilt_toward_xy, a horizontal unit vector) when that face is flat enough to
        tell, and its extent (length along the heading, width across it, height, in metres). It
        is measured from the side the camera sees: from the fixed camera across the table the
        far side of a thing is hidden and its middle comes out up to a centimetre or two toward
        the camera; from the wrist camera above, its height is taken from what stands around
        it. The camera's calibration error (about 3 mm and 0.3 degrees) is in every answer, so
        the two cameras can put one thing a centimetre or two apart and its height several
        millimetres: the nearer camera is the one to go by. A mask that is cut short or takes
        in a neighbour moves the middle with it. Works only
        for ids of that frame. Takes 0.1 s of simulated time and moves nothing.""",
}
FROM_PIXELS = {name: " ".join(text.split()) for name, text in FROM_PIXELS.items()}


def from_pixels_described(name: str, text: str, settings) -> str:
    """A pixel tool's description with the perception backend as configured:
    depth from the picture alone, or depth noise and calibration error made
    worse (``ruikit.perception.settings``)."""
    noise, calib = float(settings.get("noise", 1.0)), float(settings.get("calib", 1.0))
    if calib != 1.0:
        text = text.replace("about 3 mm and 0.3 degrees",
                            f"about {3 * calib:g} mm and {0.3 * calib:g} degrees")
        text = text.replace("(5 mm more a metre away)", f"({5 * calib:g} mm more a metre away)")
    if settings.get("depth") == "mono" and name in ("vision_lift", "vision_pose"):
        text = text.replace(
            "the depth has about 1 mm + 0.3 % of the distance of noise, no reading along the "
            "edges of things (not_visible there: pick a pixel inside a surface)",
            "there is no depth camera: the depth is predicted from the picture by a monocular "
            "model, its scale fitted to the table top, and is off by several per cent of the "
            "distance -- centimetres for the fixed camera")
        text = text.replace("from its mask and the depth image",
                            "from its mask and the depth a monocular model predicts from the "
                            "picture (no depth camera: positions are off by centimetres)")
        text = text.replace("frame's depth image", "frame's predicted depth")
        text = text.replace("the surface the depth camera read at the pixel",
                            "the surface the predicted depth puts at the pixel")
    elif noise != 1.0:
        text = text.replace("about 1 mm + 0.3 % of the distance of noise",
                            f"about {noise:g} mm + {0.3 * noise:g} % of the distance of noise")
    return text


#: the scene memory keeps at most this many sightings (the oldest go first)
SCENE_MAX = 400


def remember(ctx, frame, found, posed: Optional[dict] = None) -> None:
    """File what a perception call answered in the episode's scene memory --
    only what the agent was told: ids, words, colours, boxes, and positions
    from ``vision_pose``.  ``memory_scene`` reads it back."""
    scene = ctx.memory.setdefault("_scene", [])
    t = round(ctx.sim_time, 2)
    for obj in found:
        entry = {"t": t, "frame_id": frame.id, "camera": frame.camera, "id": obj.get("id"),
                 "label": obj.get("label")}
        for key in ("color", "box", "score"):
            if key in obj:
                entry[key] = obj[key]
        scene.append(entry)
    if posed is not None:
        scene.append({"t": t, "frame_id": frame.id, "camera": frame.camera,
                      "id": posed.get("id"), "label": posed.get("label"),
                      "xyz_m": posed.get("xyz_m"), "extent_m": posed.get("extent_m"),
                      "yaw_deg": posed.get("yaw_deg")})
    del scene[:-SCENE_MAX]


@dataclass(frozen=True)
class DetectParams:
    frame_id: str = param("a frame_id returned by camera_look; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    query: str = param("what to look for, in plain words ('ring', 'blue pot'); '' lists "
                       "everything visible", default="")


@tool(name="vision_detect",
      data="frame_id, objects: [{id, label, box [u0, v0, u1, v1], center [u, v], and where they apply color, score, cut_by_frame}]",
      family="perceive", params=DetectParams, reads=("frames",),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"frame_id": "f3.front", "query": "ring"}, {"frame_id": "f3.front"}],
      description="""Find objects in one frame. Returns each visible object's frame-local id
      (like 'f3.front#2', valid only for that frame), its category word, a plain colour when it
      has one, its pixel box [u0, v0, u1, v1] and its centre pixel -- the box is a detector's,
      a few pixels off around where the robot's estimate puts the object. Only objects actually
      visible in that picture are reported: something hidden under, behind or outside the view
      is not. Labels are generic categories; the tool never reads names, text or hidden
      properties. Takes 0.1 s of simulated time and moves nothing.""")
def vision_detect(ctx, p: DetectParams) -> ToolResult:
    frame = ctx.frame(p.frame_id)
    found = ctx.perception.detect(frame, p.query)
    remember(ctx, frame, found)
    what = f"'{p.query}'" if p.query else "objects"
    text = f"{len(found)} {what} in {frame.id}: " + ", ".join(
        f"{d['id']} {d.get('color', '')} {d['label']}".replace("  ", " ") for d in found[:12])
    if len(found) > 12:
        text += f", ... ({len(found) - 12} more)"
    return ToolResult(data={"frame_id": frame.id, "objects": found}, text=text)


@dataclass(frozen=True)
class SegmentPrompt:
    text: Optional[str] = param("describe the object in words", default=None)
    point: Optional[Tuple[float, float]] = param("a pixel [u, v] on the object", default=None)
    box: Optional[Tuple[int, int, int, int]] = param("a pixel box [u0, v0, u1, v1] around it",
                                                     default=None)


@dataclass(frozen=True)
class SegmentParams:
    frame_id: str = param("a frame_id returned by camera_look; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    prompt: SegmentPrompt = param("exactly one of text, point or box")


@tool(name="vision_segment",
      data="frame_id, segments: [{id, label, box [u0, v0, u1, v1], area_px}] (empty when nothing is there)",
      family="perceive", params=SegmentParams, reads=("frames",),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"frame_id": "f3.front", "prompt": {"point": [256, 300]}},
                {"frame_id": "f3.front", "prompt": {"text": "pot"}}],
      description="""Segment an object in one frame from a text, point or box prompt (give
      exactly one). Returns the object's frame-local id, category, pixel box and pixel area; a
      point prompt returns the object that pixel lands on. Only visible objects can be
      segmented, and a point on the bare table returns nothing. Takes 0.1 s of simulated time
      and moves nothing.""")
def vision_segment(ctx, p: SegmentParams) -> ToolResult:
    frame = ctx.frame(p.frame_id)
    given = [k for k in ("text", "point", "box") if getattr(p.prompt, k) is not None]
    if len(given) != 1:
        raise ToolError("invalid_args", "give exactly one of prompt.text, prompt.point, prompt.box")
    per = ctx.perception
    if p.prompt.text is not None:
        segs = [{**d, "area_px": (d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1])}
                for d in per.detect(frame, p.prompt.text)]
    elif p.prompt.point is not None:
        seg = per.segment_at(frame, *p.prompt.point)
        segs = [seg] if seg else []
    else:
        seg = per.segment_box(frame, p.prompt.box)
        segs = [seg] if seg else []
    if not segs:
        return ToolResult(data={"frame_id": frame.id, "segments": []},
                          text=f"nothing to segment there in {frame.id}")
    return ToolResult(data={"frame_id": frame.id, "segments": segs},
                      text=", ".join(f"{s['id']} {s['label']} ({s['area_px']} px)" for s in segs))


@dataclass(frozen=True)
class LiftParams:
    frame_id: str = param("a frame_id returned by camera_look; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    u: float = param("pixel column (0 at the left edge)")
    v: float = param("pixel row (0 at the top edge)")


@tool(name="vision_lift",
      data="frame_id, xyz_m [x, y, z], normal [nx, ny, nz], on",
      family="perceive", params=LiftParams, reads=("frames", "depth"),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"frame_id": "f3.front", "u": 256, "v": 300}],
      description="""Turn a pixel into a 3-D point using that frame's depth: the first surface
      the pixel's ray meets, in the station frame (metres; x forward, y left, z up from the
      table top), with the surface normal and, as "on", the label of the thing the ray hit
      ('table' for the table top; a part of something, such as a balance's pan, carries the
      whole thing's label). It is a measurement, not the truth: it meets boxes that stand in
      for each thing where the robot's estimate puts it (one box for a doll, several for a
      board with holes or a tray with walls), adds about 1 mm + 0.3 % of the distance of
      depth noise, and places the point with the camera's calibration, which is off by about
      3 mm and 0.3 degrees (5 mm more a metre away). The depth and camera pose are
      those of the moment the frame was taken, so a pixel in an old picture gives where things
      were then. Pixels outside the image or on nothing return not_visible. Takes 0.1 s of
      simulated time and moves nothing.""")
def vision_lift(ctx, p: LiftParams) -> ToolResult:
    frame = ctx.frame(p.frame_id)
    out = ctx.perception.lift(frame, p.u, p.v)
    return ToolResult(data={"frame_id": frame.id, **out},
                      text=f"({p.u:g}, {p.v:g}) in {frame.id} is at {out['xyz_m']}"
                           + (f" on the {out['on']}" if "on" in out else ""))


@dataclass(frozen=True)
class PoseParams:
    frame_id: str = param("the frame the object id belongs to; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    id: str = param("a frame-local object id from vision_detect or vision_segment")


@tool(name="vision_pose",
      data="frame_id, id, label, xyz_m [x, y, z], yaw_deg, extent_m [length, width, height], and tilt_deg with tilt_toward_xy [x, y] when the tilt can be told",
      family="perceive", params=PoseParams, reads=("frames", "depth"),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"frame_id": "f3.front", "id": "f3.front#2"}],
      description="""Estimate a visible object's pose from one frame: its centre in the station
      frame (metres), its heading yaw_deg about the vertical, how far it is tilted from upright
      and, when tilted, which way its top leans (tilt_toward_xy, a horizontal unit vector in
      the station frame: a beam's top leans toward the end that went down), and its extent
      (full width, depth, height in metres). It is the robot's estimate, not the truth: each
      object is off by its own fixed error (about 5 mm, sometimes 1-1.5 cm; 3 degrees; 5 % in
      size) that every perception tool shares, plus a couple of millimetres that change with
      the viewpoint and the camera's calibration error. Works only for ids of that frame. Takes 0.1 s of simulated time and moves
      nothing.""")
def vision_pose(ctx, p: PoseParams) -> ToolResult:
    frame = ctx.frame(p.frame_id)
    out = ctx.perception.pose(frame, p.id)
    remember(ctx, frame, (), posed={"id": p.id, **out})
    return ToolResult(data={"frame_id": frame.id, **out},
                      text=f"{p.id} {out['label']} at {out['xyz_m']}, yaw {out['yaw_deg']} deg")
