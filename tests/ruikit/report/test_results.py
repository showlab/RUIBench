"""The report after a run (v2.2): each agent and interface's success rate,
tokens, time and videos, as a table and as a page that plays every episode."""

import json
import shutil
from pathlib import Path

import pytest

from ruikit.experiments import provenance
from ruikit.report import clips, results_page, tables

#: a result's record of where it comes from, as a fresh run on this version writes it
FRESH = {"ruikit": "2.3.6", "frozen": provenance.expected_version(), "freeze_clean": True,
         "commit": "c", "process_episode": 1, "per_worker": 1, "python": "python",
         "python_version": "3.10", "numpy": "1.26"}


def _row(tmp_path, agent, task, level, ok, tin, tout, minutes, video=True, filmed=True,
         column="pose", harness="claude-code", model="claude-opus-5-5", seconds_key="agent_seconds"):
    folder = tmp_path / agent / f"{task}_{level}_s15_rui.pose-v1+aids"
    folder.mkdir(parents=True)
    if video:
        (folder / "video.mp4").write_bytes(b"\0")
    return {"agent": agent, "harness": harness, "model": model, "clock": "8s", "task": task,
            "level": level, "seed": 15, "profile": "rui.pose/v1+aids", "column": column,
            "aids": "aids", "suite": "learning", "success": ok, "reason": "agent_done",
            "replies": 9, "sim_seconds": 80.0, seconds_key: minutes * 60,
            "usage": {"input_tokens": tin, "output_tokens": tout}, "folder": str(folder),
            "video": "video.mp4" if video else None, "filmed": filmed, "provenance": FRESH}


def test_the_summary_gives_success_tokens_time_and_videos(tmp_path):
    rows = [_row(tmp_path, "claude-code-claude-opus-5-5", "CL2", "easy", True, 200_000, 2_000, 4.0),
            _row(tmp_path, "claude-code-claude-opus-5-5", "CL2", "hard", False, 300_000, 3_000, 6.0),
            _row(tmp_path, "claude-code-claude-opus-5-5", "WK1", "easy", True, 100_000, 1_000, 2.0,
                 video=False)]
    pose, total = tables.summary_rows(rows)
    assert pose["interface"] == "pose" and total["interface"] == "all"
    assert (pose["episodes"], pose["successes"]) == (3, 2) and pose["videos"] == 2
    assert pose["tokens_in_median"] == 200_000 and pose["tokens_in"] == 600_000
    assert pose["tokens_out"] == 6_000 and pose["minutes_median"] == 4.0
    assert round(pose["hours"], 3) == 0.2 and 0 < pose["lo"] < 2 / 3 < pose["hi"] < 1
    table = "\n".join(tables.summary(rows))
    assert "2/3 67%" in table and "600.0k / 6.0k" in table and "2/3 |" in table
    # the simple loop's time is its episode's (wall_seconds): no simulator start in it
    loop = _row(tmp_path, "loop-openai-m", "CL2", "easy", True, 1, 1, 3.0, harness="loop",
                model="openai:m", seconds_key="wall_seconds")
    assert tables.episode_minutes(loop) == 3.0


def test_filmed_and_unfilmed_episodes_do_not_compare(tmp_path):
    rows = [_row(tmp_path, "a", t, "easy", True, 1, 1, 1.0) for t in ("CL1", "CL2", "CL3")]
    rows.append(_row(tmp_path, "a", "CL4", "easy", True, 1, 1, 1.0, video=False, filmed=False))
    odd = tables.not_comparable(rows)
    assert [r["task"] for r, _ in odd] == ["CL4"] and "not filmed" in odd[0][1][-1]
    assert "Filmed and unfilmed episodes are mixed" in tables.report("x", rows)


def test_the_page_plays_each_episode_from_its_folder(tmp_path):
    rows = [_row(tmp_path, "claude-code-claude-opus-5-5", "CL2", "easy", True, 200_000, 2_000, 4.0),
            _row(tmp_path, "loop-openai-m@paused", "CL2", "easy", False, 9, 9, 1.0,
                 harness="loop", model="openai:m", video=False, seconds_key="wall_seconds")]
    page = results_page.render(rows)
    data = json.loads(page.split('<script id="data" type="application/json">')[1].split("</script>")[0])
    (exp,) = data["experiments"]
    videos = {e["a"]: e["v"] for e in exp["episodes"]}
    assert videos["claude-code-claude-opus-5-5"] == (
        "claude-code-claude-opus-5-5/CL2_easy_s15_rui.pose-v1+aids/video.mp4")
    assert videos["loop-openai-m@paused"] == ""
    names = {s["agent"]: s["agent_name"] for s in exp["summary"]}
    assert names["claude-code-claude-opus-5-5"] == "Claude Code + claude-opus-5-5"
    assert names["loop-openai-m@paused"] == "minimal loop + openai:m (clock paused)"
    assert {s["interface_name"] for s in exp["summary"]} == {"Pose", "all interfaces"}
    assert results_page.column_name("code-skill") == "Program-Skill"       # an older name
    assert results_page.column_name("program-skill") == "Program-Skill"
    assert results_page.column_name("pose@privileged") == "Pose (privileged)"
    # an agent's own words cannot close the page's script
    rows[0]["summary"] = "done </script><script>alert(1)</script>"
    assert "</script><script>alert" not in results_page.render(rows)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="no ffmpeg")
def test_a_filmed_episode_becomes_a_video_and_its_frames_go(tmp_path):
    from PIL import Image

    frames = tmp_path / "frames" / "run"
    frames.mkdir(parents=True)
    records = []
    for k, t in enumerate((0.0, 0.2, 0.4)):
        name = f"run/f{k}-front.jpg"
        Image.new("RGB", (64, 48), (40 * k, 80, 120)).save(tmp_path / "frames" / name)
        records.append({"t": t, "camera": "front", "path": name})
    log = {"frames": records, "demonstrations": [], "instruction": "Put Nono to bed.",
           "task_id": "CL2", "level": "easy", "seed": 15, "score": {"success": True},
           "config": {"record_dir": "/nowhere/else"}}           # moved since: frames beside it
    (tmp_path / "CL2_easy_s15_rui.pose-v1.episode.jsonl").write_text(json.dumps(log) + "\n")
    assert clips.film(tmp_path) == "video.mp4"
    assert (tmp_path / "video.mp4").stat().st_size > 0 and not (tmp_path / "frames").exists()
    assert clips.film(tmp_path / "nothing-here") is None


def test_a_plan_films_every_episode_unless_told_not_to(tmp_path, capsys):
    from ruikit.experiments import batch
    from ruikit.experiments.plans import load_plan

    assert all(e.record for e in load_plan("paper"))
    base = ["--plan", "paper", "--out", str(tmp_path), "--experiment", "rui",
            "--agent", "claude-code", "--task", "CL2", "--level", "easy", "--dry-run"]
    assert batch.main(base) == 0
    assert capsys.readouterr().out.strip().endswith("13 episodes, each filmed")
    assert batch.main(base + ["--no-record"]) == 0
    assert capsys.readouterr().out.strip().endswith("13 episodes, not filmed")


def test_a_success_that_skipped_a_milestone_is_one_to_review(tmp_path):
    """A doll in bed that was never in hand: the success is listed for a look at
    its video (pushed there, or the task hacked); one that went step by step is not."""
    from ruikit.experiments.results import milestone_order

    order = milestone_order("CL2", "easy", 15)
    assert "named_doll_picked" in order
    straight = _row(tmp_path / "a", "claude-code-claude-opus-5-5", "CL2", "easy", True, 1, 1, 1.0)
    straight["milestones"] = list(order)
    pushed = _row(tmp_path / "b", "claude-code-claude-opus-5-5", "CL2", "easy", True, 1, 1, 1.0)
    pushed["milestones"] = [m for m in order if m not in ("a_doll_picked", "named_doll_picked")]
    failed = dict(pushed, success=False)
    assert tables.skipped(straight) == [] and tables.skipped(failed) == []
    assert tables.skipped(pushed) == ["a_doll_picked", "named_doll_picked"]
    (row,) = [s for s in tables.summary_rows([straight, pushed]) if s["interface"] == "pose"]
    assert row["to_review"] == 1
    text = tables.report("x", [straight, pushed])
    assert "## Successes to review" in text and "milestone not reached: named_doll_picked" in text
    page = results_page.render([straight, pushed])
    assert '"skip": ["a_doll_picked", "named_doll_picked"]' in page


def test_an_object_carried_along_with_a_held_one_is_seen():
    """The WK1 rollout: the top ring held, the ring under it lifted with it."""
    from ruikit.experiments.results import carried_along

    def w(t, held_z, other_z, held=True):
        return {"t": t, "objects": {
            "ring_small": {"pos": [0.3, 0.0, held_z], "held_by": "robot0/right" if held else None},
            "ring_medium": {"pos": [0.3, 0.0, other_z], "held_by": None}}}

    base = 0.82
    lifted = [w(0.0, base + 0.01, base)] + [w(0.1 * k, base + 0.06, base + 0.05) for k in range(1, 8)]
    assert carried_along(lifted) == ["ring_medium"]
    assert carried_along([w(0.0, base + 0.01, base)] + [w(0.1 * k, base + 0.06, base + 0.02)
                                                         for k in range(1, 8)]) == []
    # nothing held: a lever or a launcher lifts things on its own
    assert carried_along([w(0.0, base + 0.01, base, held=False)]
                         + [w(0.1 * k, base + 0.06, base + 0.05, held=False) for k in range(1, 8)]) == []



def test_what_was_stacked_before_a_hold_is_not_carried_by_it():
    """v2.3.2: heights from where each hold begins.  A brick put on the tower
    before the next is taken is above its start, and is not carried by it; nor
    is a part fastened to the held thing."""
    from ruikit.experiments.results import carried_along

    def w(t, held, z_held, z_stacked, z_card):
        return {"t": t, "objects": {
            "brick2": {"pos": [0.3, 0.0, z_held], "held_by": "robot0/right" if held else None},
            "brick1": {"pos": [0.3, 0.1, z_stacked], "held_by": None},
            "card": {"pos": [0.3, 0.0, z_card], "held_by": None}}}

    base = 0.82
    # brick1 already stacked 5 cm up when brick2 is taken and lifted; the card rides on brick2
    traj = [w(0.0, False, base, base + 0.05, base)] + [
        w(0.1 * k, True, base + 0.01 * k, base + 0.05, base + 0.01 * k) for k in range(1, 10)]
    assert carried_along(traj, fastened=["card"]) == []
    assert carried_along(traj) == ["card"]                       # unless known to be fastened

def test_a_success_that_carried_something_along_is_one_to_review(tmp_path):
    row = _row(tmp_path, "codex-gpt-5.6-sol", "WK1", "easy", True, 1, 1, 1.0)
    from ruikit.experiments.results import milestone_order

    row["milestones"] = list(milestone_order("WK1", "easy", 15))
    assert tables.review(row) == []
    row["carried"] = ["ring_medium"]
    assert tables.review(row) == ["carried along without a grasp: ring_medium"]
    assert tables.summary_rows([row])[0]["to_review"] == 1
    assert '"carry": ["ring_medium"]' in results_page.render([row])
    assert tables.review(dict(row, success=False)) == []


def test_a_flagged_episode_counts_as_its_review_decides(tmp_path):
    """The physics audit fails an episode until someone has looked at it: a review
    in OUT/reviews.json clears it (it counts as the rest of its score says) or
    upholds it; one nobody has reviewed stays a failure."""
    from ruikit.experiments import results

    for cell in ("DS3_easy_s15_a", "DS3_easy_s15_b", "DS3_easy_s15_c"):
        folder = tmp_path / "claude-code-claude-opus-5-5" / cell
        folder.mkdir(parents=True)
        (folder / "result.json").write_text(json.dumps(
            {"task": "DS3", "success": False, "failure": "physics_flag"}))
    (tmp_path / "reviews.json").write_text(json.dumps({
        "claude-code-claude-opus-5-5/DS3_easy_s15_a": {"decision": "cleared", "success": True,
                                                       "by": "lead", "why": "dropped after the coin"},
        "claude-code-claude-opus-5-5/DS3_easy_s15_b": {"decision": "upheld", "success": False,
                                                       "by": "lead", "why": "a second shell moved"}}))
    rows = {Path(r["folder"]).name: r for r in results.load_results(tmp_path)}
    assert rows["DS3_easy_s15_a"]["success"] and rows["DS3_easy_s15_a"]["review"]["by"] == "lead"
    assert not rows["DS3_easy_s15_b"]["success"] and rows["DS3_easy_s15_b"]["review"]["decision"] == "upheld"
    assert not rows["DS3_easy_s15_c"]["success"] and "review" not in rows["DS3_easy_s15_c"]
