"""The task's secrets never reach the agent (bench-issue #12, approach-v2 §8.2).

The episode scans every result on its way out -- a reply, or a call's answer
inside a program -- for the task's secrets (answer-bearing names, hidden
mappings, the canary); a hit fails the run.  The probe asks every information
tool everything it can at the start of each answer-bearing task.
"""

from dataclasses import replace

import pytest

from ruikit import Episode
from ruikit.interfaces import load_profile


def test_a_secret_in_a_reply_voids_the_run(tmp_path):
    import json

    from ruibench.runner.snapshot import rescore

    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1", log_dir=str(tmp_path))
    try:
        f = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        ep.secrets = (f,)                 # a stand-in: every perception reply names its frame
        ep.call("vision_detect", {"frame_id": f, "query": ""})
        assert [leak["tool"] for leak in ep.leaks] == ["vision_detect"]
        ep.call("task_done", {"report": "done"})
        assert not ep.outcome.success
        assert ep.outcome.score["failure"] == "secret_leak"
        assert ep.outcome.metrics["leaks"] >= 1
    finally:
        ep.close()
    # the scorer judged it, so the log alone gives the same verdict elsewhere
    log = json.loads(next(tmp_path.glob("*.episode.jsonl")).read_text().splitlines()[0])
    new, diffs = rescore(log)
    assert not new.success and not diffs, diffs


def test_a_secret_a_program_is_handed_counts_even_unprinted():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.code/v1")
    try:
        ep.secrets = ("holding_something",)      # a key of every robot_state answer
        r = ep.call("code_run", {"source": "s = robot.state()\nresult = 1"})
        assert not r.is_error and r.data["result"] == 1
        assert [leak["tool"] for leak in ep.leaks] == ["robot_state"]
    finally:
        ep.close()


#: the tasks whose answer a leak would give away
ANSWER_BEARING = [("DS1", "easy"), ("DS1", "hard"), ("DS2", "easy"), ("DS3", "easy"),
                  ("DS4", "easy"), ("DS5", "hard"), ("CL2", "easy"), ("DI6", "easy")]


def _probe_profile():
    """The open track's tools, with every perception tool offered directly."""
    base = load_profile("open/v1")
    extra = ("vision_segment", "vision_pose", "vision_marks", "memory_frames")
    return replace(base, id="probe/v1", tools=tuple(base.tools) + extra)


@pytest.mark.parametrize("task,level", ANSWER_BEARING)
def test_no_information_tool_gives_a_secret_away(task, level):
    ep = Episode(task, level=level, seed=0, profile=_probe_profile())
    try:
        if not ep.secrets:
            pytest.skip(f"{task} registers no secrets yet (bench-issue #12)")
        demos = ep.call("task_info", {}).data.get("demonstrations") or 0
        for i in range(demos if isinstance(demos, int) else len(demos)):
            ep.call("task_demo", {"index": i})
        ep.call("task_events", {})
        ep.call("robot_state", {})
        for fr in ep.call("camera_look", {"calibration": True}).data["frames"]:
            f = fr["frame_id"]
            ep.call("vision_marks", {"frame_id": f, "kind": "objects"})
            ep.call("vision_point", {"frame_id": f, "query": ""})
            for obj in ep.call("vision_detect", {"frame_id": f, "query": ""}).data["objects"]:
                u, v = obj["center"]
                ep.call("vision_pose", {"frame_id": f, "id": obj["id"]})
                ep.call("vision_segment", {"frame_id": f, "prompt": {"point": [u, v]}})
                ep.call("vision_lift", {"frame_id": f, "u": u, "v": v})
        ep.call("memory_frames", {"last_n": 4})
        assert not ep.leaks, ep.leaks[:3]
    finally:
        ep.close()
