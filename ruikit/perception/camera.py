"""The cameras: take pictures, look closer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

from ruikit.core.result import Image, ToolError, ToolResult
from ruikit.core.params import param
from ruikit.core.tool import Cost, tool
from ruikit.env.frames import encode_png

#: simulated seconds a camera takes to deliver a picture
RENDER_SECONDS = 0.05


def frame_image(ctx, frame, box=None, upscale_to=None) -> Image:
    return picture(frame.id, frame.camera, frame.image if frame.has_pixels else None,
                   frame.width, frame.height, box, upscale_to)


def picture(frame_id, camera, image, width, height, box=None, upscale_to=None) -> Image:
    """A frame's picture, or an enlarged crop of it: a crop is its own file and
    says the size it is delivered at (the box says where it came from)."""
    png = encode_png(image, box, upscale_to) if image is not None else None
    note = "" if png else "this engine renders no pixels (mock); use the vision tools"
    w, h = width, height
    if png and box is not None:
        note = f"zoom {box[0]}-{box[1]}-{box[2]}-{box[3]}"
        w, h = int.from_bytes(png[16:20], "big"), int.from_bytes(png[20:24], "big")
    return Image(frame_id=frame_id, camera=camera, width=w, height=h, png=png, note=note)


def frame_info(ctx, frame, with_calibration: bool = False) -> dict:
    out = {"frame_id": frame.id, "camera": frame.camera, "t": round(frame.sim_time, 3),
           "width": frame.width, "height": frame.height}
    if with_calibration:
        k = frame.intrinsics()
        out["intrinsics"] = {key: round(v, 2) for key, v in k.items()}
        cal = ctx.calibration(frame)           # the robot's calibration, not the simulator's pose
        out["camera_xyz_m"] = [round(v, 4) for v in ctx.station.to_station(cal.pos)]
        out["camera_quat_wxyz"] = [round(v, 5) for v in cal.quat]
        out["optical_convention"] = "camera looks along +z, image x right, y down"
    return out


@dataclass(frozen=True)
class LookParams:
    cameras: Optional[List[str]] = param(
        "cameras to capture, e.g. ['front', 'wrist']; null captures every camera", default=None)
    calibration: bool = param(
        "also return each camera's intrinsics and its pose in the station frame (the robot's "
        "calibration, off by about 3 mm and 0.3 degrees, like a real one)", default=False)


@tool(name="camera_look",
      data="frames: [{frame_id, camera, t, width, height}], one per camera",
      family="observe", params=LookParams, reads=("cameras",),
      source="sensor", cost=Cost("declared", 0.0),
      examples=[{"cameras": ["front"]}, {"cameras": None, "calibration": True}],
      description="""Take a picture now with one or more cameras and return the images, each
      with a frame_id. Pass that frame_id to the tools that read pictures (the vision tools,
      and pointing where the interface has it): pixels are always read in the frame they came
      from (u right, v down). The fixed camera is 'front'
      on the one-arm station and 'head' on the two-arm station; wrist cameras ride the hands and
      only move when an arm moves. Each camera takes 0.05 s of simulated time to deliver. This
      tool does not move the robot, does not zoom, and does not report objects -- use
      vision_detect on the frame for that.""")
def camera_look(ctx, p: LookParams) -> ToolResult:
    cams = p.cameras or ctx.cameras()
    images, infos = [], []
    for cam in cams:
        frame = ctx.capture(cam)
        images.append(frame_image(ctx, frame))
        infos.append(frame_info(ctx, frame, p.calibration))
    ctx.advance(RENDER_SECONDS * len(cams))
    names = ", ".join(f"{i['camera']}={i['frame_id']}" for i in infos)
    return ToolResult(data={"frames": infos}, text=f"captured {names}", images=images)


@dataclass(frozen=True)
class ZoomParams:
    frame_id: str = param("a frame_id returned by camera_look, or a demonstration's frame "
                          "from task_demo; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    box: Tuple[int, int, int, int] = param(
        "region to enlarge, [u0, v0, u1, v1] in that frame's pixels")


@tool(name="camera_zoom",
      data="frame_id, box [u0, v0, u1, v1] (the picture comes with the result)",
      family="observe", params=ZoomParams, reads=("frames",),
      source="sensor", cost=Cost("declared", 0.0),
      examples=[{"frame_id": "f3.front", "box": [200, 180, 330, 300]}],
      description="""Crop a region of an existing frame and enlarge it for a closer look. It
      adds no information: it is the same picture at the same resolution, only bigger on
      screen, so text too small or too far to resolve stays unreadable -- move the wrist camera
      closer and take a new picture instead. The box is [u0, v0, u1, v1] in pixels of that
      frame; the crop comes back with its long side 512 pixels. A demonstration's frames can
      be zoomed too. Costs no simulated time and does not move anything.""")
def camera_zoom(ctx, p: ZoomParams) -> ToolResult:
    fid, camera, image, width, height = _zoomable(ctx, p.frame_id)
    u0, v0, u1, v1 = p.box
    u0, u1 = max(0, min(u0, u1)), min(width, max(u0, u1))
    v0, v1 = max(0, min(v0, v1)), min(height, max(v0, v1))
    if u1 - u0 < 4 or v1 - v0 < 4:
        raise ToolError("invalid_args", "the box is empty or outside the image",
                        hint=f"{fid} is {width}x{height}")
    shown = picture(fid, camera, image, width, height, (u0, v0, u1, v1), upscale_to=512)
    scale = f", enlarged {shown.width / (u1 - u0):.1f}x" if shown.png else ""
    return ToolResult(data={"frame_id": fid, "box": [u0, v0, u1, v1]},
                      text=f"zoomed into {fid} [{u0}, {v0}, {u1}, {v1}]: {u1 - u0}x{v1 - v0} "
                           f"pixels of the frame{scale}", images=[shown])


def _zoomable(ctx, frame_id: str):
    """(id, camera, image, width, height) of a frame this episode took, of a
    demonstration's frame (``demo<index>.<frame>.<camera>`` from task_demo), or
    of a marked copy of a frame (``f3.front.marks`` from vision_marks)."""
    if frame_id.endswith(".marks"):
        kept = ctx.memory.get("_marked", {}).get(frame_id)
        if kept is None:
            raise ToolError("stale_frame", f"no marked picture {frame_id!r}",
                            hint="vision_marks returns the id of the picture it marks, as "
                                 "'marked'; the last 12 are kept")
        camera, image = kept
        return frame_id, camera, image, int(image.shape[1]), int(image.shape[0])
    if not frame_id.startswith("demo"):
        frame = ctx.frame(frame_id)
        return (frame.id, frame.camera, frame.image if frame.has_pixels else None,
                frame.width, frame.height)
    try:
        index, i, cam = frame_id[len("demo"):].split(".", 2)
        t, by_cam = ctx.task.demos()[int(index)]["frames"][int(i)]
        frame = by_cam[cam]
    except (ValueError, IndexError, KeyError):
        raise ToolError("stale_frame", f"no demonstration frame {frame_id!r}",
                        hint="demonstration frames are named demo<index>.<frame>.<camera>, as "
                             "task_demo returns them") from None
    w, h = ctx.camera_size                # a mock demo has no pixels, nor a size
    return (frame_id, cam, getattr(frame, "image", None),
            int(getattr(frame, "width", 0) or w), int(getattr(frame, "height", 0) or h))
