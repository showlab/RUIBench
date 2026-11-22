"""Point by description: pixels of the things matching a few words, from a detector or a
vision-language model."""

from __future__ import annotations

from dataclasses import dataclass

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component, Implementation
from ruikit.perception.vision import PERCEPTION_SECONDS


@dataclass(frozen=True)
class PointParams:
    frame_id: str = param("a frame_id returned by camera_look; or 'latest.<camera>' (e.g. 'latest.wrist') for that camera's newest picture, such as one taken earlier in this same reply")
    query: str = param("what to point at, in plain words")


@tool(name="vision_point",
      data="frame_id, points: [{u, v, label}]",
      family="perceive", params=PointParams, reads=("frames",),
      source="vision", cost=Cost("declared", PERCEPTION_SECONDS),
      examples=[{"frame_id": "f3.front", "query": "basket"}],
      description="""Point at things matching a description in one frame: returns one pixel
      [u, v] (the centre of the visible part) per match, with its category. Use it to get a
      pixel for robot_move_to_pixel or vision_lift when you know what you want but not where it
      is in the image. Only visible objects are pointed at. Takes 0.1 s of simulated time and
      moves nothing.""")
def vision_point(ctx, p: PointParams) -> ToolResult:
    frame = ctx.frame(p.frame_id)
    if ctx.params.get("impl", "detector") == "vlm":
        return _point_by_vlm(ctx, frame, p.query)
    found = ctx.perception.detect(frame, p.query)
    points = [{"u": d["center"][0], "v": d["center"][1], "label": d["label"],
               **({"color": d["color"]} if "color" in d else {})} for d in found]
    return ToolResult(data={"frame_id": frame.id, "points": points},
                      text=f"{len(points)} point(s) for '{p.query}' in {frame.id}")


#: what a vision-language model is asked when it is to point (Gemini
#: Robotics-ER and Molmo answer points; every model here is asked in pixels)
POINT_PROMPT = ("Find every {query} in this picture. It is {w} pixels wide and {h} high; x runs "
                "from the left edge, y from the top edge. Reply with JSON only, "
                "{{\"points\": [[x, y], ...]}}: one point on each one you find, on the thing "
                "itself, at most 10; {{\"points\": []}} when there is none.")


def _point_by_vlm(ctx, frame, query: str) -> ToolResult:
    """The component's second implementation: a vision-language model looks
    at the picture and answers points.  Its real latency is charged."""
    import json
    import time

    from ruikit.env.frames import encode_png
    from ruikit.extras.backends import ASKERS, get_backend

    if not frame.has_pixels:
        raise ToolError("not_supported", "pointing by a vision-language model needs a picture "
                        "(the mock engine renders none)")
    backend = get_backend(ASKERS, "vision-language", ctx.params.get("backend"))
    t0 = time.time()
    answer = backend.ask([encode_png(frame.image)],
                         POINT_PROMPT.format(query=query, w=frame.width, h=frame.height))
    ctx.advance(max(0.0, time.time() - t0))
    start, end = answer.find("{"), answer.rfind("}")
    try:
        raw = json.loads(answer[start:end + 1]).get("points", []) if start >= 0 else []
    except ValueError:
        raw = []
    points = []
    for item in raw[:10]:
        try:
            u, v = float(item[0]), float(item[1])
        except (TypeError, ValueError, IndexError):
            continue
        if 0 <= u < frame.width and 0 <= v < frame.height:
            points.append({"u": int(round(u)), "v": int(round(v)), "label": query})
    return ToolResult(data={"frame_id": frame.id, "points": points},
                      text=f"{len(points)} point(s) for '{query}' in {frame.id} "
                           f"({getattr(backend, 'name', 'a model')} pointed)")


COMPONENT = Component(
    "pointer", "按描述指物", "a pixel on the thing a description names",
    ("vision_point",),
    (Implementation("detector", "用感知的检测器找，取中心",
                    "the perception backend's detector, the middle of each find",
                    params={"vision_point": {"impl": "detector"}}),
     Implementation("vlm", "问看图模型要一个点（Gemini-ER 式）", "a vision-language model asked for "
                    "a point (Gemini Robotics-ER style)",
                    params={"vision_point": {"impl": "vlm", "backend": "openai"}},
                    needs="a model route: vlm/<route>, e.g. vlm/openai (RUIKIT_ASK_MODEL)")),
    default="detector")


#: what the agent is told of vision_point when a vision-language model answers it
VLM_DESCRIPTION = " ".join("""Point at things matching a description in one frame:
    a vision-language model looks at the picture and answers a pixel [u, v] on each one it
    finds (at most 10), labelled with your words. Use it to get a pixel for
    robot_move_to_pixel or vision_lift when you know what you want but not where it is in
    the image. It is a model's answer: it can miss a thing, point beside it or at the wrong
    one -- check the pixel against the picture. Its real latency (seconds) is charged as
    simulated time; it moves nothing.""".split())
