"""The tools for changing the benchmark while it runs (v2.3.2): what a change does
to the rollouts already run (recheck), the reviews of what someone would otherwise
watch (review)."""

import json
from pathlib import Path

from ruikit.experiments import recheck, review
from ruikit.experiments import results as cells


def _result(out, cell, **fields):
    folder = Path(out) / "claude-code-claude-opus-5-5" / cell
    folder.mkdir(parents=True)
    row = {"harness": "claude-code", "model": "claude-opus-5-5", "clock": "8s", "aids": "aids",
           "task": cell.split("_")[0], "level": cell.split("_")[1], "seed": 15, **fields}
    (folder / "result.json").write_text(json.dumps(row))
    return folder


def test_recheck_reads_results_but_not_what_was_put_aside(tmp_path):
    for sub in ("claude-code-x/WK1_easy_s15_a", "discarded/WK1_easy_s15_b", "superseded/v2.3/WK1_easy_s15_c"):
        (tmp_path / sub).mkdir(parents=True)
        (tmp_path / sub / "WK1_easy_s15.episode.jsonl").write_text("{}\n")
    assert [p.parent.name for p in recheck.episode_logs([str(tmp_path)])] == ["WK1_easy_s15_a"]
    # named itself, a folder put aside is read
    assert len(recheck.episode_logs([str(tmp_path / "superseded")])) == 1


def test_the_recheck_report_says_what_flips_and_what_runs_again():
    rows = [{"log": "/x/a", "task": "CL3", "level": "easy", "seed": 15, "was": False, "now": False,
             "violations_was": [], "violations_now": [], "milestones_was": [], "milestones_now": [],
             "told": False, "shown": False, "shown_parts": True},
            {"log": "/x/b", "task": "WK1", "level": "easy", "seed": 15, "was": True, "now": False,
             "violations_was": [], "violations_now": ["one_ring_at_a_time"], "milestones_was": [],
             "milestones_now": [], "told": False, "shown": False}]
    text = recheck.report(rows, base="old")
    assert "| CL3 | 1 | 0 | 0 | 0 | 0 | 1 | 1 |" in text
    assert "| WK1 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |" in text and "one_ring_at_a_time" in text


def test_reviews_are_kept_and_what_was_reviewed_is_not_listed_again(tmp_path):
    _result(tmp_path, "DS3_easy_s15_rui.pose-v1+aids", success=False, failure="physics_flag")
    _result(tmp_path, "CL2_easy_s15_rui.pose-v1+aids", success=True, carried=["doll5"])
    _result(tmp_path, "CR3_easy_s15_rui.pose-v1+aids", success=True,      # every step, nothing carried
            milestones=list(cells.milestone_order("CR3", "easy", 15)))
    todo = {key.split("/")[-1]: kind for key, _, kind in review.candidates(tmp_path)}
    assert todo == {"DS3_easy_s15_rui.pose-v1+aids": "physics", "CL2_easy_s15_rui.pose-v1+aids": "success"}
    review.write(tmp_path, "claude-code-claude-opus-5-5/DS3_easy_s15_rui.pose-v1+aids",
                 {"kind": "physics", "decision": "cleared", "success": True, "by": "x", "why": "y"})
    review.write(tmp_path, "claude-code-claude-opus-5-5/CL2_easy_s15_rui.pose-v1+aids",
                 {"kind": "success", "decision": "hack", "success": False, "by": "x", "why": "two at once"})
    assert review.candidates(tmp_path) == []
    rows = {Path(r["folder"]).name: r for r in cells.load_results(tmp_path)}
    assert rows["DS3_easy_s15_rui.pose-v1+aids"]["success"] is True
    assert rows["CL2_easy_s15_rui.pose-v1+aids"]["success"] is False
    assert rows["CL2_easy_s15_rui.pose-v1+aids"]["failure"] == "hack"


def test_the_times_to_look_at_are_the_flags_and_the_milestones():
    log = {"score": {"flags": ["too_fast:shell1@244.0", "too_fast:detector@121.23"],
                     "milestones_hit": {"detector_up": 30.2, "coin_in_bucket": 200.0}}}
    assert review.moments({"sim_seconds": 245.9}, log, "physics") == [121.2, 244.0, 245.9]
    # a success: its milestones, and the whole episode evenly (a step the milestones missed)
    assert review.moments({"sim_seconds": 245.9}, log, "success") == [
        30.2, 49.2, 98.4, 147.5, 196.7, 200.0, 245.9]

