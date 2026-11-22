"""Grounded expressors: the reference's intent, the interface's perception.

The coverage expressors (``reference``) send the reference's poses as they
are -- enough to ask whether an interface's tools can *act* a task.  An agent
has no poses to send until it has looked.  These expressors do what it must:
each turn they look, find the thing a commanded pose is relative to (its
*anchor*), estimate that thing's pose with the interface's own perception
tools, and send the estimate plus the reference's offset from it.

So *what* to do -- which thing, where relative to it -- is still the
reference's; every coordinate the robot is sent comes from a tool's answer
and carries its error.  A pose far from everything seen (up to the carrying
height, back to rest) is sent as it is; an anchor no camera sees at that
moment falls back to the reference's own number, and the note says so.

Each agent spends as few replies on looking as its interface allows -- one
reply per round of dependent calls, the independent ones of a round sent
together:

* the step interface (``rui.step/v1``) looks and reads where its hand is (one
  reply), finds everything the turn needs (one), estimates their poses (one),
  then steps;
* the pose interface (``rui.pose/v1``) looks, finds, estimates (three
  replies), then ``robot_move_to``;
* the pointing interface (``rui.point/v1``) does the same, plus one reply of
  ``vision_lift`` when it points at a surface, then ``robot_move_to_pixel`` in
  the frame the thing was seen in;
* the program interface (``rui.code/v1``) writes two programs a turn: one that
  looks, finds and estimates into ``seen`` (a program chains dependent calls in
  one reply), one that moves by those variables -- and, like the others'
  look-up of an anchor they did not plan for, a third that asks for the things
  the pictures show when none of the ones it asked for was there.

Every grounding leaves a note (``notes``) on the reply that carries it, for
the showcase page.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from ruibench.core.geometry import quat_mul, yaw_of
from ruibench.core.types import Pose
from ruibench.interfaces.commands import MoveDelta, MoveTo
from ruikit.core.result import ToolError
from ruikit.env.station import rpy_deg_from_quat

#: a pose further than this (xy distance plus a third of the height difference)
#: from everything seen is sent as it is
RADIUS = 0.25
#: a fixed thing counts as this much further, so a movable one nearby wins
FIXED_PENALTY = 0.02
#: the pointing interface points at a thing's top only when the measured surface
#: is this close to where it aimed (about the perception error); else by height
SURFACE_TOL = 0.003


def _rz(angle: float) -> Tuple[float, float, float, float]:
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def _turn(v, angle: float):
    c, s = math.cos(angle), math.sin(angle)
    return (c * v[0] - s * v[1], s * v[0] + c * v[1], v[2])


def _r(x: float, nd: int = 3) -> float:
    return round(x, nd) + 0.0


@dataclass
class Seen:
    """A thing as a perception tool reported it."""
    body: str
    label: str
    frame_id: str
    obj_id: str
    xyz: Tuple[float, float, float]           # world, as estimated
    yaw: float                                # radians, as estimated
    true_xyz: Tuple[float, float, float]      # world, at the frame's capture
    true_yaw: float
    size: Tuple[float, float, float]
    lifted: bool = False                      # the estimate put it below the table: raised


@dataclass
class Grounded:
    pose: Pose
    seen: Optional[Seen]
    note: str
    offset: Optional[Tuple[float, float, float]] = None      # station axes, from the estimate
    turn_deg: float = 0.0                                     # the hand's yaw less the thing's


class Grounder:
    """One turn's looking: the frames, and what has been estimated in them."""

    def __init__(self, ep, via_code: bool = False) -> None:
        self.ep = ep
        self.via_code = via_code
        self.frames: List[dict] = []
        self.seen: Dict[str, Optional[Seen]] = {}
        # what the agent last estimated of each thing, and when: what it goes by
        # when it cannot see the thing now (it cannot know it moved since)
        self.memory: Dict[str, Tuple[Seen, float]] = {}

    def new_turn(self) -> None:
        self.frames, self.seen = [], {}

    # -- looking --------------------------------------------------------------
    def look(self) -> None:
        if not self.frames:
            r = self.ep.call("camera_look", {})
            self.frames = [] if r.is_error else list(r.data.get("frames") or [])

    def _frames(self):
        """This turn's frames, the fixed camera's first."""
        out = [self.ep.frames.get(f["frame_id"]) for f in self.frames]
        return sorted(out, key=lambda fr: (not fr.camera.startswith(("front", "head")), fr.camera))

    def anchor(self, target: Pose) -> Optional[str]:
        ranked = self.anchors(target)
        return ranked[0] if ranked else None

    def anchors(self, target: Pose) -> List[str]:
        """Things seen this turn a pose could be relative to, nearest first."""
        found: Dict[str, float] = {}
        held = {n for fr in self._frames() for n, o in fr.objects.items() if o.is_held}
        for fr in self._frames():
            # only what detection can find in this frame (a drawn-only lane is not)
            findable = {name for _, name, _ in self.ep.perception._ensure_ids(fr)}
            for name in fr.visible:
                obj = fr.objects.get(name)
                if obj is None or name not in self.ep.perception.bodies or name not in findable:
                    continue
                dx, dy = target.pos[0] - obj.pose.pos[0], target.pos[1] - obj.pose.pos[1]
                flat = math.hypot(dx, dy)
                if name in held and flat > 0.04:
                    continue                  # a carried thing is not where it will go
                d = flat + 0.3 * abs(target.pos[2] - obj.pose.pos[2])
                if getattr(self.ep.perception.bodies[name], "fixed", False):
                    d += FIXED_PENALTY
                if d < RADIUS and d < found.get(name, math.inf):
                    found[name] = d
        return sorted(found, key=found.get)

    def wanted(self, targets: List[Pose], per_pose: Optional[int] = 2) -> List[str]:
        """The things this turn's poses are relative to -- the reference's
        intent: the nearest ``per_pose`` to each pose (None: every one in
        reach).  Which things; never where they are (only the tools say that)."""
        world = self.ep.engine.state()
        out: Dict[str, None] = {}
        for t in targets:
            near = []
            for name, obj in world.objects.items():
                if name not in self.ep.perception.bodies:
                    continue
                flat = math.hypot(t.pos[0] - obj.pose.pos[0], t.pos[1] - obj.pose.pos[1])
                if obj.is_held and flat > 0.04:
                    continue                  # a carried thing is not where it will go
                d = flat + 0.3 * abs(t.pos[2] - obj.pose.pos[2])
                if getattr(self.ep.perception.bodies[name], "fixed", False):
                    d += FIXED_PENALTY
                if d < RADIUS:
                    near.append((d, name))
            for _, name in sorted(near)[:per_pose]:
                out[name] = None
        return list(out)

    def prepare(self, targets: List[Pose], with_state: bool = False) -> None:
        """Everything this turn's poses need, in as few replies as the calls'
        dependencies allow: look (and, for a stepping agent, read the hand) in
        one reply, then find and estimate the things the poses are relative to
        in two more.  A turn whose poses are by nothing (up to the carrying
        height, back to rest) does not look."""
        want = [b for b in self.wanted(targets) if b not in self.seen]
        if not want:
            return
        if not self.frames:
            calls = [("camera_look", {})] + ([("robot_state", {})] if with_state else [])
            r = self.ep.call_batch(calls)[0]
            self.frames = [] if r.is_error else list(r.data.get("frames") or [])
        self._estimate(want)

    def perceive(self, body: str) -> Optional[Seen]:
        """One thing the turn did not plan for, found and estimated now."""
        if body not in self.seen:
            self._estimate([body])
        return self.seen.get(body)

    def _estimate(self, bodies: List[str]) -> None:
        """Ask for each wanted kind of thing in every picture (one reply: the
        agent does not know which camera shows it), then estimate each thing
        in the first picture it was found in, the fixed camera's first (one
        reply)."""
        if not bodies:
            return
        labels = list(dict.fromkeys(self.ep.perception.label(b)["label"] for b in bodies))
        asks = [(fr, label) for fr in self._frames() for label in labels]
        found = self.ep.call_batch([("vision_detect", {"frame_id": fr.id, "query": label})
                                    for fr, label in asks]) if asks else []
        ids: Dict[str, Tuple[object, str]] = {}          # body -> (frame, its id there)
        for (fr, _), r in zip(asks, found):
            if r.is_error:
                continue
            for o in r.data.get("objects") or []:
                try:
                    body = self.ep.perception.resolve(fr, o["id"])
                except ToolError:
                    continue             # a find from pixels that is no thing of the scene
                if body in bodies and body not in ids:
                    ids[body] = (fr, o["id"])
        order = [b for b in bodies if b in ids]
        poses = self.ep.call_batch([("vision_pose", {"frame_id": ids[b][0].id, "id": ids[b][1]})
                                    for b in order]) if order else []
        for body in bodies:
            self.seen.setdefault(body, None)
        for body, r in zip(order, poses):
            if not r.is_error:
                fr, oid = ids[body]
                self.seen[body] = self._seen(body, self.ep.perception.label(body)["label"], fr, oid,
                                             r.data)
                self.memory[body] = (self.seen[body], self.ep.sim_time)

    def _seen(self, body, label, fr, oid, data) -> Seen:
        """An estimate as the agent uses it, with the one thing it knows for
        sure: nothing lies below the table, so a thing's middle is at least
        half its reported thickness up (an estimate a few millimetres low sent
        the fingers into the table under WK5's weights, which slipped out)."""
        obj = fr.objects[body]
        xyz = list(data["xyz_m"])
        extent = data.get("extent_m")
        lifted = bool(extent) and xyz[2] < extent[2] / 2
        if lifted:
            xyz[2] = extent[2] / 2
        return Seen(body, label, fr.id, oid, tuple(self.ep.station.to_world(xyz)),
                    math.radians(data["yaw_deg"]), tuple(obj.pose.pos), yaw_of(obj.pose.quat),
                    tuple(self.ep.perception.bodies[body].size), lifted)

    # -- a pose from what was seen ------------------------------------------------
    def ground(self, target: Pose) -> Optional[Grounded]:
        """The target from what the agent has estimated -- never from the
        simulator: an estimate made now, else the last one it made of the
        thing; None when it has neither (the move cannot be made)."""
        ranked = self.anchors(target)
        if not ranked:
            if self._near_anything(target):
                return self._from_memory(target)
            return Grounded(target, None, "直接给的位姿：离看到的东西都远（抬到搬运高度、收臂等），不需要看")
        for body in ranked:
            if body in self.seen or self.via_code:
                seen = self.seen.get(body)
            else:
                seen = self.perceive(body)               # not planned for: look it up now
            if seen is not None:
                return self.relative(target, seen)
        return self._from_memory(target, ranked)

    def _near_anything(self, target: Pose) -> bool:
        """Is the target by something the agent has seen before (and cannot see now)?"""
        return any(math.hypot(target.pos[0] - s.true_xyz[0], target.pos[1] - s.true_xyz[1]) < RADIUS
                   for s, _ in self.memory.values())

    def _from_memory(self, target: Pose, ranked: Optional[List[str]] = None) -> Optional[Grounded]:
        """The last estimate of the nearest thing seen before: where the agent
        believes it is, which it cannot know it has moved from."""
        now = self.ep.engine.state()
        best = None
        for body, (seen, when) in self.memory.items():
            obj = now.objects.get(body)
            if obj is None or obj.is_held:
                continue                      # the hand's own load is no landmark
            d = math.hypot(target.pos[0] - obj.pose.pos[0], target.pos[1] - obj.pose.pos[1])
            if d < RADIUS and (best is None or d < best[0]):
                best = (d, body, seen, when, obj)
        if best is None:
            return None
        _, body, seen, when, obj = best
        # the reference's offset from where the thing is; the agent's belief of where that is
        now_seen = Seen(seen.body, seen.label, seen.frame_id, seen.obj_id, seen.xyz, seen.yaw,
                        tuple(obj.pose.pos), yaw_of(obj.pose.quat), seen.size)
        got = self.relative(target, now_seen)
        got.note += f"（此刻看不见，用的是 {when:.1f} s 时的估计）"
        return got

    def relative(self, target: Pose, seen: Seen) -> Grounded:
        """The target where the reference put it relative to the thing, but from
        the thing's estimated pose: the estimate's error is all that changes."""
        dyaw = seen.yaw - seen.true_yaw
        rel = _turn(tuple(t - b for t, b in zip(target.pos, seen.true_xyz)), -seen.true_yaw)
        pos = tuple(p + d for p, d in zip(seen.xyz, _turn(rel, seen.yaw)))
        quat = quat_mul(_rz(dyaw), target.quat)
        at = self.ep.station.to_station(pos)
        base = self.ep.station.to_station(seen.xyz)
        offset = tuple(_r(a - b) for a, b in zip(at, base))
        hand_yaw = rpy_deg_from_quat(quat)[2]
        turn = ((hand_yaw - math.degrees(seen.yaw) + 180.0) % 360.0) - 180.0
        note = (f"= {seen.label}（{seen.obj_id}）的估计位置 {[_r(x) for x in base]} "
                f"+ 偏移 {list(offset)}")
        if seen.lifted:
            note += "（估计比桌面还低：高度取它半个厚度）"
        return Grounded(Pose(pos, quat), seen, note, offset, _r(turn, 1))


def _with_notes(cls):
    """An expressor that grounds every pose it sends and notes where it came from."""

    class Grounded_(cls):
        grounded = True

        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.notes: List[dict] = []
            self.grounder: Optional[Grounder] = None

        def run(self, ep):
            self.grounder = Grounder(ep, via_code=getattr(self, "via_code", False))
            return super().run(ep)

        def on_turn(self, ep):
            self.grounder.new_turn()

        def note(self, ep, text: str, call: int = 0, target: Optional[Pose] = None) -> None:
            entry = {"row": len(ep.log), "call": call, "note": text}
            if target is not None:
                entry["target"] = ep.station.describe_pose(target)
            self.notes.append(entry)

    Grounded_.__name__ = f"Grounded{cls.__name__}"
    return Grounded_


def _targets(expr, ep, cmds) -> List[Pose]:
    return [expr.target_of(ep, c) for c in cmds if isinstance(c, (MoveTo, MoveDelta))]
