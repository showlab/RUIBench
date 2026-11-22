"""Memory: notes, past frames and a record of what was seen, for loops whose
own memory is thin.

Episode-scoped: everything here is forgotten when the episode ends.
``memory_frames`` returns only pictures the agent already took -- it cannot
film the past -- and ``memory_scene`` only what the perception tools already
told it: a record kept as it goes, not a new look.
The memory component offers them alone or together (``COMPONENT`` below): the
notes, the pictures, the scene record.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, tool
from ruikit.harness.components.base import Component, Implementation
from ruikit.perception.camera import frame_image

MAX_NOTES = 200
MAX_NOTE_CHARS = 2000


@dataclass(frozen=True)
class NoteParams:
    text: str = param("what to remember")
    key: Optional[str] = param("a name to find it by later; the same key overwrites",
                               default=None)


@tool(name="memory_note",
      data="key",
      family="memory", params=NoteParams, source="agent",
      cost=Cost("declared", 0.0),
      examples=[{"key": "plan", "text": "move small ring first"}, {"text": "coin 3 was light"}],
      description="""Write a note for later in this episode, optionally under a key (writing
      the same key again replaces it). Notes are private to this episode and forgotten when it
      ends. At most 200 notes of 2000 characters. Costs no simulated time and does not touch
      the robot.""")
def memory_note(ctx, p: NoteParams) -> ToolResult:
    notes = ctx.memory.setdefault("notes", {})
    if len(p.text) > MAX_NOTE_CHARS:
        raise ToolError("invalid_args", f"a note holds at most {MAX_NOTE_CHARS} characters")
    key = p.key or f"note{len(notes) + 1}"
    if key not in notes and len(notes) >= MAX_NOTES:
        raise ToolError("invalid_args", f"at most {MAX_NOTES} notes")
    notes[key] = {"text": p.text, "t": round(ctx.sim_time, 2)}
    return ToolResult(data={"key": key}, text=f"noted as {key!r}")


@dataclass(frozen=True)
class RecallParams:
    key: Optional[str] = param("the key of one note", default=None)
    query: Optional[str] = param("or: words to search the notes for; null lists them all",
                                 default=None)


@tool(name="memory_recall",
      data="notes {key: {text, t}}",
      family="memory", params=RecallParams, source="agent",
      cost=Cost("declared", 0.0), examples=[{"key": "plan"}, {"query": "coin"}, {}],
      description="""Read back notes written earlier in this episode: one by key, those whose
      text contains all of the query's words, or all of them. Costs no simulated time.""")
def memory_recall(ctx, p: RecallParams) -> ToolResult:
    notes = ctx.memory.get("notes", {})
    if p.key is not None:
        hits = {p.key: notes[p.key]} if p.key in notes else {}
    else:
        words = (p.query or "").lower().split()
        hits = {k: v for k, v in notes.items() if all(w in v["text"].lower() for w in words)}
    return ToolResult(data={"notes": hits}, text=f"{len(hits)} note(s)")


@dataclass(frozen=True)
class FramesParams:
    frame_ids: Optional[List[str]] = param("frames to show again", default=None)
    last_n: Optional[int] = param("or: the last n frames taken", default=None,
                                  minimum=1, maximum=8)


@tool(name="memory_frames",
      data="frames: [{frame_id, camera, t}] (the pictures come with the result)",
      family="memory", params=FramesParams, reads=("frames",),
      source="agent", cost=Cost("declared", 0.0),
      examples=[{"last_n": 2}, {"frame_ids": ["f3.front"]}],
      description="""Show again pictures you took earlier in this episode, by frame id or the
      last n (at most 8). It only returns frames you already captured with camera_look -- it
      cannot show what the cameras saw when you were not looking. Costs no simulated time.""")
def memory_frames(ctx, p: FramesParams) -> ToolResult:
    if (p.frame_ids is None) == (p.last_n is None):
        raise ToolError("invalid_args", "give exactly one of frame_ids or last_n")
    frames = [ctx.frame(f) for f in p.frame_ids] if p.frame_ids else ctx.recent_frames(p.last_n)
    images = [frame_image(ctx, f) for f in frames]
    return ToolResult(data={"frames": [{"frame_id": f.id, "camera": f.camera,
                                        "t": round(f.sim_time, 3)} for f in frames]},
                      text=f"{len(frames)} frame(s)", images=images)


@dataclass(frozen=True)
class SceneParams:
    label: Optional[str] = param("only things under this word (as the tools labelled them); "
                                 "null: everything", default=None)
    since_s: Optional[float] = param("only what was seen in the last this many seconds",
                                     default=None, minimum=0.0)


#: two placed sightings of one word this close (metres) are one thing
SCENE_SAME_M = 0.03


@tool(name="memory_scene",
      data="things: [{label, color, xyz_m (latest, if a pose was estimated), extent_m, seen_t, posed_t, sightings, frame_id, id}], unplaced: [{label, color, frame_id, id, t}]",
      family="memory", params=SceneParams, source="agent", cost=Cost("declared", 0.0),
      examples=[{}, {"label": "coin"}, {"since_s": 30}],
      description="""Read the scene record: every thing the perception tools found this
      episode, as they told you -- its word and colour, and where vision_pose last put it
      (station frame, metres) with when. Sightings of one word within 3 cm of each other count
      as one thing; a thing found but never posed is listed apart, by its frame-local id. It is
      a record of earlier answers, not a new look: a thing moved since is where it was then.
      Costs no simulated time and does not touch the robot.""")
def memory_scene(ctx, p: SceneParams) -> ToolResult:
    import math

    entries = list(ctx.memory.get("_scene", ()))
    if p.since_s is not None:
        entries = [e for e in entries if ctx.sim_time - e["t"] <= p.since_s]
    if p.label:
        word = p.label.strip().lower()
        entries = [e for e in entries if word in str(e.get("label", "")).lower()]
    things, unplaced, placed_ids = [], [], set()
    for e in entries:
        if e.get("xyz_m") is None:
            continue
        placed_ids.add((e["frame_id"], e["id"]))
        same = next((t for t in things if t["label"] == e["label"]
                     and math.dist(t["xyz_m"], e["xyz_m"]) <= SCENE_SAME_M), None)
        if same is None:
            things.append({"label": e["label"], "xyz_m": e["xyz_m"], "extent_m": e.get("extent_m"),
                           "posed_t": e["t"], "seen_t": e["t"], "sightings": 1,
                           "frame_id": e["frame_id"], "id": e["id"]})
        else:
            same.update({"xyz_m": e["xyz_m"], "extent_m": e.get("extent_m"), "posed_t": e["t"],
                         "seen_t": e["t"], "frame_id": e["frame_id"], "id": e["id"]})
            same["sightings"] += 1
    for e in entries:
        if e.get("xyz_m") is not None:
            continue
        colour = e.get("color")
        hit = next((t for t in things if (e["frame_id"], e["id"]) == (t["frame_id"], t["id"])), None)
        if hit is not None:
            hit["seen_t"] = max(hit["seen_t"], e["t"])
            if colour:
                hit["color"] = colour
        elif (e["frame_id"], e["id"]) not in placed_ids:
            unplaced.append({"label": e.get("label"), **({"color": colour} if colour else {}),
                             "frame_id": e["frame_id"], "id": e["id"], "t": e["t"]})
    text = (f"{len(things)} placed thing(s), {len(unplaced)} found but not placed"
            + (": " + "; ".join(f"{t['label']} at {t['xyz_m']} (t={t['posed_t']})"
                                for t in things[:8]) if things else ""))
    return ToolResult(data={"things": things, "unplaced": unplaced[-40:]}, text=text)


#: the memory variants of the Harness ablation: text, image, interleaved
COMPONENT = Component(
    "memory", "记忆", "remember within the episode",
    ("memory_note", "memory_recall", "memory_frames", "memory_scene"),
    (Implementation("text", "文本：自己写、自己读的笔记", "text: notes the agent writes and reads back",
                    ("memory_note", "memory_recall")),
     Implementation("image", "图像：回看自己拍过的照片", "image: the pictures it took, shown again",
                    ("memory_frames",)),
     Implementation("scene", "场景记忆：自动记下认出过的东西在哪", "an automatic record of what its "
                    "perception calls found, and where", ("memory_scene",)),
     Implementation("interleaved", "交错：笔记、照片与场景记录都给", "interleaved: the notes, the "
                    "pictures and the scene record together",
                    ("memory_note", "memory_recall", "memory_frames", "memory_scene"))),
    default="interleaved",
    aliases={"notes": "text", "frames": "image", "all": "interleaved"})
