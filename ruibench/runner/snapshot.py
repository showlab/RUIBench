"""A world state, whole, as the log keeps it -- and back again.

An episode's log is what it is scored from after the fact (``ruibench
rescore``): in another process, from the log alone, so that an agent never
shares a process with the scorer that judges it (v15, the RUIKit
collaborator's audit, ``bench-issue.md`` §11).  Every snapshot therefore holds
everything a predicate reads -- each body's whole pose, its velocity, its
holder, its colour when a toy changed it, the cameras that see it, a cloth's
marks and the box round them; each
hand's pose, grip and load; the joints; the instruction in force; and what has
happened (the events fired since the snapshot before, repeats and all: a
mole whacked twice is two whacks, DI5) -- while what never
changes with time (sizes, categories, attributes) comes back from the scene,
which the task, level and seed rebuild.

The keys the viewer and the reports read (``pos``, ``yaw``, ``held_by``, an
arm's ``pos`` and ``gripper``, ``instruction_version``) keep their meaning.
RUIKit logs its trajectories with the same ``dump``.
"""

from __future__ import annotations

from typing import Iterable, List, Mapping, Optional, Sequence

from ..core.geometry import yaw_of
from ..core.types import ArmState, ObjectState, Pose, RobotState, WorldState

#: rounding: 0.1 mm, a millionth of a quaternion component, 0.1 mm/s
POS, QUAT, VEL = 4, 6, 4


def _colour(value):
    return list(value) if isinstance(value, (tuple, list)) else value


def dump(world: WorldState, previous: Optional[WorldState] = None, scene=None) -> dict:
    """One snapshot.  ``previous``: the snapshot before, whose events are not
    repeated; ``scene``: a body's colour is kept only when it is not the
    scene's (a lamp lit, a strip read).

    An engine only appends to ``events_fired``, and the same event may fire
    again, so what is new is what follows the events before -- not the ones
    not seen before.  Should the list not go on from the one before, the
    snapshot keeps all of it (``events_from_start``)."""
    colours = {b.name: b.color for b in scene.bodies} if scene is not None else {}
    before = tuple(previous.events_fired) if previous is not None else ()
    fired = tuple(world.events_fired)
    goes_on = fired[:len(before)] == before
    objects = {}
    for name, o in world.objects.items():
        d = {"pos": [round(v, POS) for v in o.pose.pos], "yaw": round(yaw_of(o.pose.quat), 3),
             "held_by": o.held_by, "quat": [round(v, QUAT) for v in o.pose.quat],
             "vel": [round(v, VEL) for v in o.lin_vel]}
        if o.visible_from is not None:
            d["visible_from"] = list(o.visible_from)
        if scene is None or o.color != colours.get(name):
            d["color"] = _colour(o.color)
        if o.points is not None:
            # a cloth: its marks, and the box round them
            d["points"] = [[round(v, POS) for v in p] for p in o.points]
            d["size"] = [round(v, POS) for v in o.size]
        objects[name] = d
    arms = {}
    for aid in world.arm_ids:
        a = world.arm(aid)
        arms[aid] = {"pos": [round(v, POS) for v in a.eef_pose.pos], "gripper": round(a.gripper, 3),
                     "quat": [round(v, QUAT) for v in a.eef_pose.quat], "holding": a.holding,
                     "busy": a.busy}
        if a.opening is not None:
            arms[aid]["opening"] = round(a.opening, POS)
        if a.joints is not None:              # the arm's joint angles (radians, base to wrist)
            arms[aid]["joints"] = [round(float(v), 6) for v in a.joints]
    return {"t": round(world.sim_time, 4), "objects": objects, "arms": arms,
            "joints": {n: round(float(v), 6) for n, v in world.articulations.items()},
            "instruction": world.instruction, "instruction_version": world.instruction_version,
            "events": list(fired[len(before):] if goes_on else fired),
            **({} if goes_on else {"events_from_start": True})}


def dump_all(traj: Sequence[WorldState], scene=None) -> List[dict]:
    out, prev = [], None
    for w in traj:
        out.append(dump(w, prev, scene))
        prev = w
    return out


def is_whole(snapshots: Sequence[Mapping]) -> bool:
    """Were these written by ``dump`` (and not by the positions-only logs of
    ``ruibench-log/1``)?"""
    return bool(snapshots) and "joints" in snapshots[0] and "events" in snapshots[0]


def load_all(snapshots: Iterable[Mapping], scene) -> List[WorldState]:
    """The trajectory back, bodies' fixed properties from ``scene``."""
    from ..engines.mock import footprint_of

    specs = {b.name: b for b in scene.bodies}
    events: List[str] = []
    out = []
    for d in snapshots:
        if d.get("events_from_start"):
            events = []
        events += list(d.get("events", ()))
        objects = {}
        for name, o in d["objects"].items():
            spec = specs[name]
            vis = o.get("visible_from")
            colour = o.get("color", spec.color)
            points = o.get("points")
            objects[name] = ObjectState(
                name=name, pose=Pose(tuple(o["pos"]), tuple(o["quat"])),
                size=tuple(o["size"]) if points is not None else spec.size,
                category=spec.category, color=tuple(colour) if isinstance(colour, list) else colour,
                held_by=o.get("held_by"), lin_vel=tuple(o.get("vel", (0.0, 0.0, 0.0))),
                visible_from=None if vis is None else tuple(vis), attrs=dict(spec.attrs),
                footprint=footprint_of(spec),
                points=None if points is None else tuple(tuple(p) for p in points))
        robots = {}
        for aid, a in d["arms"].items():
            robot, _, part = aid.partition("/")
            robots.setdefault(robot, {})[part] = ArmState(
                name=aid, eef_pose=Pose(tuple(a["pos"]), tuple(a["quat"])), gripper=a["gripper"],
                holding=a.get("holding"), busy=bool(a.get("busy", False)), opening=a.get("opening"))
        out.append(WorldState(
            sim_time=d["t"], objects=objects,
            robots={r: RobotState(r, arms) for r, arms in robots.items()},
            articulations=dict(d.get("joints", {})), instruction=d.get("instruction", ""),
            instruction_version=d.get("instruction_version", 0), events_fired=tuple(events)))
    return out


def rescore(log: Mapping):
    """Score an episode again from its log alone: the task rebuilt from its
    id, level, seed and config, the trajectory from the snapshots.  Returns
    (the new score, what differs from the stored one)."""
    import dataclasses

    from ..core.config import RunConfig
    from ..tasks.instance import build

    names = {f.name for f in dataclasses.fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in dict(log.get("config") or {}).items()
                       if k in names and k != "budget"})
    inst = build(log["task_id"], log["level"], int(log["seed"]), cfg)
    traj = load_all(log["trajectory"], inst.scene)
    meta = dict(log.get("scoring") or {})
    meta["metrics"] = dict((log.get("score") or {}).get("metrics", {}))
    new = inst.scorer.score(traj, meta)
    old = dict(log.get("score") or {})
    fresh = new.as_dict()
    diffs = []
    for key in ("success", "failure", "violations", "flags"):
        if key in old and old[key] != fresh[key]:
            diffs.append(f"{key}: stored {old[key]!r}, now {fresh[key]!r}")
    stored_ms, now_ms = dict(old.get("milestones_hit", {})), fresh["milestones_hit"]
    if set(stored_ms) != set(now_ms):
        diffs.append(f"milestones: stored {sorted(stored_ms)}, now {sorted(now_ms)}")
    return new, diffs
