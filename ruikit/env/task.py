"""The task tools: the instruction, the demonstrations, the events, and the end.

The public task channel.  An instruction can change mid-episode (a Simon-says
command, a change of mind), a toy can answer ("The monkey ate it."), an
opponent can move: all of it arrives as events, appended to every tool result
as well, so an agent whose client drops notifications still gets them.
``task_done`` ends the episode; it never decides success -- the physical
trajectory does.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ruikit.core.result import Image, ToolError, ToolResult
from ruikit.core.params import NoParams, param
from ruikit.core.tool import Cost, tool
from ruikit.env.frames import encode_png


@tool(name="task_info",
      data="instruction, instruction_version, hint (when the task has one), demonstrations (how many), conventions {frame, orientation, pixels, arms, cameras, gripper_max_opening_m, gripper, estimates, ...}",
      family="task", params=NoParams, source="task",
      cost=Cost("declared", 0.0),
      description="""Read the task: the current instruction and its version number (an
      instruction may change while you work), the hint (how the scene's contraptions work --
      never the answer), the arms and cameras of this station, the coordinate conventions every
      tool uses, how many demonstrations there are, and the budget left. Costs no simulated
      time and moves nothing.""")
def task_info(ctx, p: NoParams) -> ToolResult:
    demos = ctx.task.demos()
    data = {"instruction": ctx.task.instruction,
            "instruction_version": ctx.task.instruction_version,
            "hint": ctx.task.hint, "demonstrations": len(demos),
            "conventions": ctx.station.conventions(ctx.profile.perception)}
    text = f"Instruction (v{data['instruction_version']}): {data['instruction']}"
    if data["hint"]:
        text += f" Hint: {data['hint']}"
    return ToolResult(data=data, text=text)


@dataclass(frozen=True)
class DemoParams:
    index: int = param("which demonstration, from 0", default=0, minimum=0)
    frames: Optional[List[int]] = param(
        "which frames to return (indices); null returns an even sample of at most 8",
        default=None)
    camera: Optional[str] = param("which camera's view; null: the fixed camera", default=None)


@tool(name="task_demo",
      data="index, caption, seconds, picture, frames_total, frames_returned, frame_ids (for camera_zoom and show)",
      family="task", params=DemoParams, source="task",
      cost=Cost("declared", 0.0),
      examples=[{"index": 0}, {"index": 0, "frames": [0, 5, 10]}],
      description="""Watch a demonstration that was shown before the episode: returns its
      caption, how long it ran, and a sample of its frames (about one per simulated second)
      from the chosen camera. Demonstrations are pictures only -- no object states, no code --
      and may be filmed at another table. A still picture has a single frame. Costs no
      simulated time and moves nothing.""")
def task_demo(ctx, p: DemoParams) -> ToolResult:
    demos = ctx.task.demos()
    if not demos:
        raise ToolError("invalid_args", "this task has no demonstration")
    if p.index >= len(demos):
        raise ToolError("invalid_args", f"there are {len(demos)} demonstrations (0-{len(demos) - 1})")
    demo = demos[p.index]
    frames = demo["frames"]
    if p.frames is not None:
        picked = [i for i in p.frames if 0 <= i < len(frames)]
    else:
        n = len(frames)
        step = max(1, -(-n // 8))
        picked = list(range(0, n, step))
        if n and picked[-1] != n - 1:
            picked.append(n - 1)
    cam = p.camera or ("head" if ctx.station.two_arms else "front")
    images = []
    for i in picked:
        t, by_cam = frames[i]
        frame = by_cam.get(cam) or next(iter(by_cam.values()), None)
        if frame is None:
            continue
        png = encode_png(getattr(frame, "image", None))
        images.append(Image(frame_id=f"demo{p.index}.{i}.{cam}", camera=cam,
                            width=int(getattr(frame, "width", 0)),
                            height=int(getattr(frame, "height", 0)), png=png,
                            note="" if png else "this engine renders no pixels (mock)"))
    return ToolResult(data={"index": p.index, "caption": demo["caption"],
                            "seconds": demo["seconds"], "picture": demo["picture"],
                            "frames_total": len(frames), "frames_returned": picked,
                            # the ids, so a program can show() or zoom them (CL2, program interface)
                            "frame_ids": [im.frame_id for im in images]},
                      text=f"demonstration {p.index}: {demo['caption']} "
                           f"({len(images)} of {len(frames)} frames)", images=images)


@dataclass(frozen=True)
class EventsParams:
    since: int = param("return events from this index on (0: all so far)", default=0, minimum=0)


@tool(name="task_events",
      data="events: [{t, type, text}], next_since",
      family="task", params=EventsParams, source="task",
      cost=Cost("declared", 0.0), examples=[{"since": 0}],
      description="""List the task's events so far: instruction changes (with the new text and
      version), messages from the scene (a toy answering, a one-bit 'correct'/'wrong'), and
      the like, each with its simulated time. Every tool result already carries the events new
      since your previous call; this tool is for looking back. Costs no simulated time.""")
def task_events(ctx, p: EventsParams) -> ToolResult:
    events = ctx.task.events(p.since)
    return ToolResult(data={"events": events, "next_since": p.since + len(events)},
                      text=f"{len(events)} event(s)")


@dataclass(frozen=True)
class DoneParams:
    report: str = param("one sentence on what you did", default="")


@tool(name="task_done",
      data="report",
      family="task", params=DoneParams, source="task",
      cost=Cost("declared", 0.0), examples=[{"report": "The tower is on the right peg."}],
      description="""Declare the task finished and end the episode. Motions still running are
      allowed to finish and the world keeps running a few seconds before it is judged. Success
      is decided from what physically happened, never from this declaration, so call it only
      when you believe the goal is met. After it, every tool returns episode_over.""")
def task_done(ctx, p: DoneParams) -> ToolResult:
    ctx.task.done(p.report)
    return ToolResult(data={"report": p.report}, text="the episode will end now")


@dataclass(frozen=True)
class GiveUpParams:
    reason: str = param("why you cannot finish", default="")


@tool(name="task_give_up", family="task", params=GiveUpParams, source="task",
      cost=Cost("declared", 0.0), examples=[{"reason": "the target is out of reach"}],
      description="""Give up and end the episode, saying why. Use it when the goal cannot be
      reached; it is scored as not done. After it, every tool returns episode_over.""")
def task_give_up(ctx, p: GiveUpParams) -> ToolResult:
    ctx.task.give_up(p.reason)
    return ToolResult(data={"reason": p.reason}, text="the episode will end now")
