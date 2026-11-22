"""No tool, and no mix of tools, tells an agent where things really are
(the advisor, 9/30: like a real robot, the agent does not know positions).

The Claude agents grasped within 1.3 mm of the true centres because depth had
1 mm of noise and detection boxes none, so either one checked vision_pose back
to the truth.  Here an adversary uses the tools as an agent could, for every
object it can see, keeps whichever answer came out best (the truth picks it --
a lower bound on what any agent could do), and must still be off like a real
robot.  A new perception tool, or one an interface adds, belongs in here.
"""

import math
import statistics


from ruikit.env import Episode

TASKS = ("DS1", "DS4", "WK5", "CL2", "WK6")


def _truth(ep, fid, oid):
    name = ep.perception.resolve(ep.frames.get(fid), oid)            # scoring only
    return ep.station.to_station(ep.engine.state().objects[name].pose.pos)


def _best_errors(task, per_task=4, grid=9):
    ep = Episode(task, level="easy", seed=0, profile="rui.pose/v1")
    errors = []
    try:
        cam = "head" if ep.station.two_arms else "front"
        fid = ep.call("camera_look", {"cameras": [cam]}).data["frames"][0]["frame_id"]
        for obj in ep.call("vision_detect", {"frame_id": fid, "query": ""}).data["objects"][:per_task]:
            truth = _truth(ep, fid, obj["id"])
            guesses = []
            pose = ep.call("vision_pose", {"frame_id": fid, "id": obj["id"]})
            if not pose.is_error:
                guesses.append(pose.data["xyz_m"][:2])
            u0, v0, u1, v1 = obj["box"]
            mid = ep.call("vision_lift", {"frame_id": fid, "u": (u0 + u1) / 2, "v": (v0 + v1) / 2})
            if not mid.is_error:
                guesses.append(mid.data["xyz_m"][:2])
            top = []
            for i in range(grid):
                for j in range(grid):
                    r = ep.call("vision_lift", {"frame_id": fid, "u": u0 + (u1 - u0) * (i + .5) / grid,
                                                "v": v0 + (v1 - v0) * (j + .5) / grid})
                    if not r.is_error and r.data.get("on") == obj["label"] and r.data["normal"][2] > .9:
                        top.append(r.data["xyz_m"])
            if len(top) >= 4:
                xs, ys = [p[0] for p in top], [p[1] for p in top]
                guesses.append([(min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2])
            if len(guesses) >= 2:
                guesses.append([statistics.mean(g[k] for g in guesses) for k in (0, 1)])
            if guesses:
                errors.append(min(1000 * math.dist(g, truth[:2]) for g in guesses))
    finally:
        ep.close()
    return errors


def test_the_best_an_agent_can_do_is_a_real_robots_best():
    errors = sorted(e for t in TASKS for e in _best_errors(t))
    assert len(errors) >= 12
    # before 9/30: the best of these was within 1 mm for a third of the objects
    assert statistics.median(errors) >= 3.0, errors
    assert sum(e < 1.5 for e in errors) <= 0.15 * len(errors), errors


def test_the_calibration_the_agent_reads_is_not_the_simulators():
    ep = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        look = ep.call("camera_look", {"calibration": True})
        off = []
        for f in look.data["frames"]:
            true = ep.station.to_station(ep.frames.get(f["frame_id"]).pose.pos)
            off.append(1000 * math.dist(true, f["camera_xyz_m"]))
        assert max(off) > 1.0, off
    finally:
        ep.close()


def test_an_interface_cannot_make_perception_exact():
    import dataclasses

    from ruikit.interfaces import load_profile

    exact = dataclasses.replace(load_profile("rui.pose/v1"),
                                params={"vision_pose": {"error_scale": 0.0}})
    ep = Episode("DS1", level="easy", seed=0, profile=exact)
    try:
        assert ep.perception.scale == 1.0
    finally:
        ep.close()
