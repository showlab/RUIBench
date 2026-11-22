"""How precisely the agents really grasped: the grasp's offset from the object's
centre, beside the estimate's error and the gripper's room."""

from __future__ import annotations

import glob
import json
import math
import os
import statistics
from typing import List


def _runs(clips: str):
    for folder in sorted(glob.glob(os.path.join(clips, "*"))):
        ep = glob.glob(os.path.join(folder, "*.episode.jsonl"))
        if ep:
            yield os.path.basename(folder), folder, json.loads(open(ep[0]).readline())


def _closes(traj):
    """(index, arm, held name) for every close of a gripper that caught something."""
    for arm in traj[0]["arms"]:
        for i in range(1, len(traj)):
            if traj[i - 1]["arms"][arm]["gripper"] < 0.5 <= traj[i]["arms"][arm]["gripper"]:
                held = next((traj[j]["arms"][arm]["holding"] for j in range(i, min(len(traj), i + 30))
                             if traj[j]["arms"][arm]["holding"]), None)
                yield i, arm, (held.split("/")[-1] if held else None)


def audit_grasps(clips: str) -> List[dict]:
    """At every close: the hand's horizontal distance from the true centre of
    what it caught (mm), or a miss.  Under the privileged condition the
    perception it grasped by had no error, so the offset it owes to perception
    is zero (``expected``)."""
    out = []
    for run, _, ep in _runs(clips):
        traj = ep["trajectory"]
        privileged = ep.get("condition") == "privileged"
        for i, arm, held in _closes(traj):
            tcp = traj[i]["arms"][arm]["pos"]
            row = {"run": run, "t": traj[i]["t"]}
            if privileged:
                row["expected"] = "zero: privileged perception"
            if held is None or held not in traj[i]["objects"]:
                out.append({**row, "held": None})
                continue
            obj = traj[i]["objects"][held]["pos"]
            out.append({**row, "held": held,
                        "xy_mm": round(1000 * math.dist(tcp[:2], obj[:2]), 1)})
    return out


def audit_where(clips: str) -> List[dict]:
    """For each grasp: vision_pose's error on that object, the grasp's error,
    and how much room the gripper had ((opening - width) / 2), in mm."""
    from ruikit.env import Episode

    stations, out = {}, []
    for run, folder, ep in _runs(clips):
        task = run.split("-")[0]
        if task not in stations:
            e = Episode(task, level="easy", seed=0, profile="rui.pose/v1")
            stations[task] = (e.station, dict(e.perception.bodies),
                              {a.split("/")[-1]: e.station.arm(a.split("/")[-1]).max_opening
                               for a in e.scene.arm_ids})
            e.close()
        st, bodies, openings = stations[task]
        traj = ep["trajectory"]
        logs = [f for f in glob.glob(os.path.join(folder, "*.jsonl")) if not f.endswith(".episode.jsonl")]
        rows = [json.loads(l) for l in open(logs[0])][1:] if logs else []
        truth = lambda s, n: st.to_station(tuple(s["objects"][n]["pos"]))
        snap = lambda t: min(traj, key=lambda s: abs(s["t"] - t))
        est = {}
        for r in rows:
            answers = []
            if r.get("tool") == "vision_pose":
                answers = [(r.get("results") or [{}])[0]]
            elif r.get("tool") == "batch":
                answers = [res for c, res in zip(r["args"].get("calls", []), r.get("results") or [])
                           if c.get("tool") == "vision_pose"]
            answers += [i.get("result") or {} for i in r.get("inner") or []
                        if isinstance(i, dict) and i.get("tool") in ("vision_pose", "vision.pose")]
            for res in answers:
                xyz = (res.get("data") or {}).get("xyz_m") if isinstance(res, dict) else None
                if not xyz:
                    continue
                s = snap(r.get("t0", 0.0))
                name, d = min(((n, math.dist(truth(s, n)[:2], xyz[:2])) for n in s["objects"]),
                              key=lambda x: x[1])
                if d < 0.04:
                    est.setdefault(name, []).append(1000 * d)
        for i, arm, held in _closes(traj):
            if held is None or held not in bodies:
                continue
            tcp = st.to_station(tuple(traj[i]["arms"][arm]["pos"]))
            width = 2 * min(bodies[held].size[0], bodies[held].size[1])
            room = max(0.0, (openings.get(arm.split("/")[-1], 0.08) - width) / 2)
            e = est.get(held)
            out.append({"run": run, "object": held,
                        "estimate_mm": round(statistics.median(e), 1) if e else None,
                        "grasp_mm": round(1000 * math.dist(tcp[:2], truth(traj[i], held)[:2]), 1),
                        "room_mm": round(1000 * room, 1)})
    return out
