"""Visual marks: numbered marks drawn on a copy of a frame, one per thing seen."""

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Literal, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component, Implementation
from ruikit.perception.camera import frame_image
from ruikit.perception.vision import PERCEPTION_SECONDS


@dataclass(frozen=True)
class MarksParams:
    frame_id: str = param("a frame_id returned by camera_look; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    kind: Literal["objects", "grid"] = param(
        "'objects': a numbered mark on each visible object; 'grid': a labelled 5x5 grid",
        default="objects")


@tool(name="vision_marks",
      data="frame_id, kind, marks: [{mark, id, label, u, v}], marked (the marked picture's id: camera_zoom, or show() in a program, can show it again)",
      family="perceive", params=MarksParams, reads=("frames",),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"frame_id": "f3.front", "kind": "objects"}],
      description="""Draw numbered marks on a copy of a frame (Set-of-Mark style) and return
      the annotated image with a table from each mark to its pixel. 'objects' marks every
      visible object; 'grid' overlays a 5x5 grid whose cells are named a1..e5 (columns a-e
      from the left, rows 1-5 from the bottom). Marks belong to that frame only. Takes 0.1 s of
      simulated time and moves nothing.""")
def vision_marks(ctx, p: MarksParams) -> ToolResult:
    frame = ctx.frame(p.frame_id)
    marks = []
    if p.kind == "objects":
        for i, d in enumerate(ctx.perception.detect(frame, ""), start=1):
            marks.append({"mark": str(i), "id": d["id"], "label": d["label"],
                          "u": d["center"][0], "v": d["center"][1]})
    else:
        for ci, col in enumerate("abcde"):
            for ri in range(5):
                u = int((ci + 0.5) * frame.width / 5)
                v = int((4 - ri + 0.5) * frame.height / 5)
                marks.append({"mark": f"{col}{ri + 1}", "u": u, "v": v})
    image = frame_image(ctx, frame)
    style = ctx.params.get("style", "dots") if p.kind == "objects" else "dots"
    if frame.has_pixels:
        shapes = {}
        if style in ("masks", "boxes"):
            for m in marks:
                shape = ctx.perception.mask_of(frame, m["id"]) if style == "masks" else None
                shapes[m["mark"]] = shape if shape is not None else \
                    ctx.perception.box_of(frame, m["id"])
        image.png = _draw_marks(frame, marks, style, shapes)
    data = {"frame_id": frame.id, "kind": p.kind, "marks": marks}
    if image.png:
        # kept, so a program (whose tools return no pictures) can show() it, and
        # camera_zoom can enlarge a part of it
        import io

        import numpy as np
        from PIL import Image as PILImage

        kept = ctx.memory.setdefault("_marked", {})
        # its own name, so the file a session saves is the marked picture, not the plain one
        data["marked"] = image.frame_id = f"{frame.id}{'.grid' if p.kind == 'grid' else ''}.marks"
        kept[data["marked"]] = (frame.camera, np.asarray(PILImage.open(io.BytesIO(image.png))
                                                         .convert("RGB")))
        while len(kept) > MARKED_KEPT:
            kept.pop(next(iter(kept)))
    return ToolResult(data=data, text=f"{len(marks)} marks on {frame.id}", images=[image])


#: marked pictures kept for show() and camera_zoom, the newest
MARKED_KEPT = 12


#: the marks' colours, one a mark in turn (distinct on wood, on white and on dark)
MARK_COLOURS = ((230, 25, 75), (60, 180, 75), (0, 130, 200), (245, 130, 48), (145, 30, 180),
                (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128))


def _draw_marks(frame, marks, style: str = "dots", shapes: Optional[dict] = None) -> Optional[bytes]:
    """The picture with the marks on it: a numbered dot at each mark (Set-of-
    Mark's dots), or each thing's outline (``masks``) or box (``boxes``) in a
    colour of its own, numbered."""
    import numpy as np
    from PIL import Image as PILImage
    from PIL import ImageDraw

    from ruikit.env.frames import to_array

    arr = to_array(frame.image)
    if arr is None:
        return None
    img = PILImage.fromarray(np.asarray(arr)[..., :3].astype(np.uint8))
    draw = ImageDraw.Draw(img)
    for k, m in enumerate(marks):
        colour = MARK_COLOURS[k % len(MARK_COLOURS)]
        shape = (shapes or {}).get(m["mark"])
        if style == "masks" and shape is not None and hasattr(shape, "shape"):
            mask = np.asarray(shape, dtype=bool)
            inner = mask.copy()
            inner[1:, :] &= mask[:-1, :]
            inner[:-1, :] &= mask[1:, :]
            inner[:, 1:] &= mask[:, :-1]
            inner[:, :-1] &= mask[:, 1:]
            edge = mask & ~inner
            pix = np.asarray(img).copy()
            thick = edge.copy()
            thick[1:, :] |= edge[:-1, :]
            thick[:, 1:] |= edge[:, :-1]
            pix[thick] = colour
            img = PILImage.fromarray(pix)
            draw = ImageDraw.Draw(img)
        elif style == "boxes" and shape is not None and len(shape) == 4:
            u0, v0, u1, v1 = (float(c) for c in shape)
            draw.rectangle((u0, v0, u1 - 1, v1 - 1), outline=colour, width=2)
        u, v = m["u"], m["v"]
        r = 9 if len(m["mark"]) < 2 else 11
        draw.ellipse((u - r, v - r, u + r, v + r), fill=(255, 255, 255),
                     outline=colour if style != "dots" else (0, 0, 0), width=2)
        draw.text((u - 3.5 * len(m["mark"]), v - 6), m["mark"], fill=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


COMPONENT = Component(
    "marks", "视觉标记", "numbered marks drawn on a picture (Set-of-Mark)",
    ("vision_marks",),
    (Implementation("dots", "物体中心画编号圆点", "a numbered dot on each thing",
                    params={"vision_marks": {"style": "dots"}}),
     Implementation("masks", "描出物体轮廓再标号", "each thing outlined and numbered",
                    params={"vision_marks": {"style": "masks"}}),
     Implementation("boxes", "画框再标号", "a numbered box round each thing",
                    params={"vision_marks": {"style": "boxes"}})),
    default="dots")
