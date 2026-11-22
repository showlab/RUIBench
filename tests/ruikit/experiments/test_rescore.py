"""Results scored again after a change to how a task is scored (v2.3.4)."""

import json

from ruibench.core.config import RunConfig
from ruibench.runner import RunPlan, run_plan
from ruikit.experiments import provenance
from ruikit.experiments.rescore import mark_episodes, rescore_results

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")


def _episode(out, task="WK6", frozen="v2.3.3", **result):
    """One stored episode of the reference, its result as ``result`` says."""
    [run] = run_plan(RunPlan([task], levels=("easy",), seeds=(0,)),
                     agent_factory=lambda i: i.reference_factory(i), config=CFG)
    folder = out / "claude-code-claude-opus-5-5" / f"{task}_easy_s0_rui.pose-v1+aids"
    folder.mkdir(parents=True)
    log = json.loads(run.log.to_json())
    (folder / f"{folder.name}.episode.jsonl").write_text(json.dumps(log) + "\n")
    row = {"task": task, "level": "easy", "seed": 0, "column": "pose",
           "provenance": {**provenance.stamp(1, 1), "frozen": frozen, "freeze_clean": True},
           "success": False, "failure": "execution_failure", "progress": 0.0, "milestones": [],
           **result}
    (folder / "result.json").write_text(json.dumps(row))
    return folder


def test_a_scoring_change_is_applied_to_the_results_already_run(tmp_path, monkeypatch):
    """A WK6 episode of v2.3.3, failed by its scorer then, takes v2.3.4's
    verdict, keeps the old one beside it, and compares again; done twice, the
    second time changes nothing."""
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.4")
    folder = _episode(tmp_path)
    [row] = rescore_results(tmp_path, ["WK6"])
    assert (row["was"], row["now"]) == (False, True)
    result = json.loads((folder / "result.json").read_text())
    assert result["success"] and result["failure"] == "none"
    assert result["scored_again"]["frozen"] == "v2.3.4"
    assert result["scored_again"]["before"]["success"] is False
    assert result["furthest_milestone"] == "x_won"
    assert provenance.not_comparable(result, "v2.3.4") == []
    [again] = rescore_results(tmp_path, ["WK6"])
    assert again["refused"] == "already scored again with v2.3.4"


def test_a_dry_run_writes_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.4")
    folder = _episode(tmp_path)
    before = (folder / "result.json").read_text()
    [row] = rescore_results(tmp_path, ["WK6"], dry_run=True)
    assert row["now"] is True and (folder / "result.json").read_text() == before


def test_an_episode_told_something_else_or_changed_otherwise_is_refused(tmp_path, monkeypatch):
    """Scoring again is for a change to the scorer alone: an episode whose
    instruction is not this code's, or a task its version changed otherwise
    (CR2 in v2.3.3, WK4 in v2.3.4), is refused and its result left as it is."""
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.4")
    folder = _episode(tmp_path)
    log_path = next(folder.glob("*.episode.jsonl"))
    log = json.loads(log_path.read_text())
    log["instruction"] = "You are O. Lose the game."
    log_path.write_text(json.dumps(log) + "\n")
    [row] = rescore_results(tmp_path, ["WK6"])
    assert row["refused"] == "told something else"
    assert json.loads((folder / "result.json").read_text())["success"] is False
    other = tmp_path / "other"
    _episode(other, task="CR2", frozen="v2.3.2")
    [row] = rescore_results(other, ["CR2"])
    assert row["refused"] == "CR2 changed from v2.3.2 to v2.3.4 in more than its scoring"
    wk4 = tmp_path / "wk4"
    _episode(wk4, task="WK4")
    [row] = rescore_results(wk4, ["WK4"])
    assert row["refused"] == "WK4 changed from v2.3.3 to v2.3.4 in more than its scoring"


def test_only_a_result_scored_again_compares_across_a_scoring_change():
    """v2.3.4 changed CL5, WK6 and CR3 in their scoring only: a result of v2.3.3
    compares once scored again with v2.3.4.  WK4 changed in more (the X5's hands
    read otherwise), and so did CR2 in v2.3.3: those compare scored again or not."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3.3", "freeze_clean": True}
    assert provenance.scored_again_between("v2.3.3", "v2.3.4") == {"CL5", "WK6", "CR3"}
    assert provenance.scored_again_between("v2.3.2", "v2.3.4") == {"CL5", "WK6", "CR3"}
    assert provenance.not_comparable({"task": "WK6", "provenance": old}, "v2.3.4") == [
        "frozen version v2.3.3, not v2.3.4"]
    assert provenance.not_comparable({"task": "WK6", "provenance": old,
                                      "scored_again": {"frozen": "v2.3.4"}}, "v2.3.4") == []
    for task in ("WK4", "WK5", "DI5", "DI6", "CL6", "CR6", "DI1"):
        assert provenance.not_comparable({"task": task, "provenance": old,
                                          "scored_again": {"frozen": "v2.3.4"}}, "v2.3.4") == [
            "frozen version v2.3.3, not v2.3.4"]
    v232 = {**old, "frozen": "v2.3.2"}
    assert provenance.not_comparable({"task": "CR2", "provenance": v232,
                                      "scored_again": {"frozen": "v2.3.4"}}, "v2.3.4") == [
        "frozen version v2.3.2, not v2.3.4"]
    assert provenance.not_comparable({"task": "CR1", "provenance": old}, "v2.3.4") == []
    # v2.3.5 changed no task: every result of v2.3.4 compares, but for an episode it touches
    assert provenance.changed_between("v2.3.4", "v2.3.5") == frozenset()
    assert provenance.scored_again_between("v2.3.3", "v2.3.5") == {"CL5", "WK6", "CR3"}
    v234 = {**old, "frozen": "v2.3.4"}
    for task in ("DS1", "CR1", "WK4"):
        assert provenance.not_comparable({"task": task, "provenance": v234}, "v2.3.5") == []


def test_a_result_whose_hand_was_read_at_pitch_90_is_marked(tmp_path, monkeypatch):
    """From v2.3.4 a hand within 0.6 degrees of pitch +-90 reads roll 0: an
    earlier episode whose hand was read there would have gone otherwise, in any
    task, and is marked so it does not compare; one never read there is left."""
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.4")
    old = {**provenance.stamp(1, 1), "frozen": "v2.3.3", "freeze_clean": True}
    folders = {}
    for name, pitch in (("CR2_hard_s15_rui.delta-v1+aids", 89.8), ("CR1_easy_s15_rui.delta-v1+aids", 45.0)):
        f = tmp_path / "claude-code-claude-opus-5-5" / name
        f.mkdir(parents=True)
        (f / "result.json").write_text(json.dumps({"task": name[:3], "column": "delta",
                                                   "provenance": old}))
        (f / f"{name}.jsonl").write_text(json.dumps(
            {"i": 1, "tool": "robot_move_delta", "results": [{"ok": True, "data": {
                "tcp": {"xyz_m": [0.4, 0.0, 0.2], "rpy_deg": [-180.0, pitch, 180.0]}}}]}) + "\n")
        folders[name[:3]] = f
    assert [r["task"] for r in mark_episodes(tmp_path, dry_run=True)] == ["CR2"]
    assert "read_otherwise" not in json.loads((folders["CR2"] / "result.json").read_text())
    [row] = mark_episodes(tmp_path)
    marked = json.loads((folders["CR2"] / "result.json").read_text())
    assert marked["read_otherwise"] == "v2.3.4"
    assert provenance.not_comparable(marked, "v2.3.4") == [
        "its hand was read at pitch +-90 degrees, which v2.3.4 reads otherwise"]
    assert provenance.not_comparable(json.loads((folders["CR1"] / "result.json").read_text()),
                                     "v2.3.4") == []
    assert mark_episodes(tmp_path) == []              # marked once
    # still so once a later version is the expected one, and one not yet marked is marked then
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.5")
    assert provenance.not_comparable(marked, "v2.3.5") == [
        "its hand was read at pitch +-90 degrees, which v2.3.4 reads otherwise"]
    unmarked = json.loads((folders["CR2"] / "result.json").read_text())
    del unmarked["read_otherwise"]
    (folders["CR2"] / "result.json").write_text(json.dumps(unmarked))
    assert [r["changes"] for r in mark_episodes(tmp_path)] == [["read_otherwise"]]


def test_a_minimal_loop_reply_without_a_call_is_marked(tmp_path, monkeypatch):
    """From v2.3.5 a minimal-loop reply that called no tool costs its think time
    on the fixed clock: an earlier such episode, in any task, does not compare.
    One whose every reply called a tool does; so does any other harness's, and a
    minimal-loop episode on the measured clock."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3.4", "freeze_clean": True}
    replies = {"talked": [{"calls": [{"name": "task_info"}]}, {"calls": [], "text": "I think..."}],
               "busy": [{"calls": [{"name": "task_info"}]}, {"calls": [{"name": "camera_look"}]}]}
    cases = {"loop_8s_talked": ("loop", "8s", "talked"), "loop_8s_busy": ("loop", "8s", "busy"),
             "loop_measured_talked": ("loop", "measured", "talked"),
             "claude_8s_talked": ("claude-code", "8s", "talked")}
    folders = {}
    for name, (harness, clock, kind) in cases.items():
        f = tmp_path / "agent" / f"CR1_easy_s15_{name}"
        f.mkdir(parents=True)
        (f / "result.json").write_text(json.dumps({"task": "CR1", "harness": harness,
                                                   "clock": clock, "provenance": old}))
        (f / f"{f.name}.model.jsonl").write_text("\n".join(
            json.dumps(r) for r in [{"model": "m", "usage": {}}, *replies[kind]]) + "\n")
        folders[name] = f
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.4")
    assert mark_episodes(tmp_path) == []              # not a change of v2.3.4
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.5")
    [row] = mark_episodes(tmp_path)
    assert row["folder"].endswith("loop_8s_talked") and row["changes"] == ["replied_without_a_call"]
    marked = json.loads((folders["loop_8s_talked"] / "result.json").read_text())
    assert provenance.not_comparable(marked, "v2.3.5") == [
        "a reply without a tool call, which costs its think time from v2.3.5"]
    for name in ("loop_8s_busy", "loop_measured_talked", "claude_8s_talked"):
        row = json.loads((folders[name] / "result.json").read_text())
        assert provenance.not_comparable(row, "v2.3.5") == []


def test_a_result_of_a_harness_changed_since_does_not_compare():
    """v2.3.4 changed how Claude Code runs (its tools' descriptions whole, the
    pictures of a failed call kept): every Claude Code result of v2.3.3 is run
    again, whatever its task, scored again or not; other harnesses' still compare."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3.3", "freeze_clean": True}
    assert provenance.harnesses_changed("v2.3.3", "v2.3.6") == {"claude-code"}
    assert provenance.harnesses_changed("v2.3.4", "v2.3.6") == frozenset()
    assert provenance.harnesses_changed("v2.3.9", "v2.3.12") == frozenset()      # effort: per result
    assert provenance.not_comparable({"task": "CR1", "harness": "claude-code", "provenance": old},
                                     "v2.3.6") == [
        "frozen version v2.3.3: its harness (claude-code) changed since"]
    assert provenance.not_comparable({"task": "CR1", "harness": "codex", "provenance": old},
                                     "v2.3.6") == []
    v234 = {**old, "frozen": "v2.3.4"}
    assert provenance.not_comparable({"task": "CR1", "harness": "claude-code", "provenance": v234},
                                     "v2.3.6") == []


def test_a_ds3_episode_the_physics_audit_flagged_is_marked(tmp_path, monkeypatch):
    """Until v2.3.8 the lights riding on DS3's detector gathered free-fall speed
    and shot away any shell they touched -- an episode the physics audit flagged.
    Such a result of an earlier version does not compare with v2.3.8's; an
    unflagged DS3 result does, and so does a flagged result of another task."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3.7", "freeze_clean": True}
    cases = {"DS3_shot": ("DS3", "physics_flag"), "DS3_fine": ("DS3", "violation"),
             "DS3_won": ("DS3", None), "CR1_flagged": ("CR1", "physics_flag")}
    folders = {}
    for name, (task, failure) in cases.items():
        f = tmp_path / "agent" / f"{name}_easy_s15_rui.pose-v1+aids"
        f.mkdir(parents=True)
        (f / "result.json").write_text(json.dumps(
            {"task": task, "harness": "claude-code", "failure": failure, "provenance": old}))
        folders[name] = f
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.8")
    [row] = mark_episodes(tmp_path)
    assert row["folder"].endswith("DS3_shot_easy_s15_rui.pose-v1+aids")
    assert row["changes"] == ["shot_by_a_mark"]
    marked = json.loads((folders["DS3_shot"] / "result.json").read_text())
    assert provenance.not_comparable(marked, "v2.3.8") == [
        "a body shot away by a mark riding on another, which v2.3.8's marks cannot do"]
    for name in ("DS3_fine", "DS3_won", "CR1_flagged"):
        row = json.loads((folders[name] / "result.json").read_text())
        assert provenance.not_comparable(row, "v2.3.8") == []


def test_a_result_at_another_reasoning_effort_does_not_compare(monkeypatch):
    """Each result records its reasoning effort: one run at another effort than
    its harness's now does not compare (v2.3.12: Claude Code and Codex at high);
    one at the same effort does, whatever the versions between."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.3.9", "freeze_clean": True}
    monkeypatch.setattr(provenance, "effort_now", lambda harness, model: "high")
    row = {"task": "CR1", "harness": "codex", "model": "gpt-5.6-sol", "provenance": old}
    assert provenance.not_comparable({**row, "effort": "medium"}, "v2.3.12") == [
        "reasoning effort medium, not high"]
    assert provenance.not_comparable({**row, "effort": "high"}, "v2.3.12") == []
    assert provenance.not_comparable({**row, "harness": "claude-code", "effort": "high"}, "v2.3.12") == []


def test_a_qwencloud_call_refused_for_text_it_now_reads_is_marked(tmp_path, monkeypatch):
    """From v2.4.1 a QwenCloud argument that came as text where its parameter
    takes a list or a number is read as the JSON it spells: an earlier episode
    with such a refusal does not compare.  A refusal of text that spells no such
    value, another refusal, and another channel's episode still compare."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.4.0", "freeze_clean": True}
    texts = {"listed": "rpy_deg: expected a list, got '[180, 0, 0]'",
             "named": "cameras: expected a list, got " + repr('["front"]'),
             "worded": "approach_m: expected a number, got 'close'",
             "ranged": "n: must be <= 5, got 9"}
    cases = {"qwen_listed": ("qwencloud", "listed"), "qwen_named": ("qwencloud", "named"),
             "qwen_worded": ("qwencloud", "worded"), "qwen_ranged": ("qwencloud", "ranged")}
    folders = {}
    for name, (channel, kind) in cases.items():
        f = tmp_path / "agent" / f"CR1_easy_s15_{name}"
        f.mkdir(parents=True)
        (f / "result.json").write_text(json.dumps(
            {"task": "CR1", "harness": "loop", "model_settings": {"channel": channel},
             "provenance": old}))
        (f / f"{f.name}.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"header": True},
            {"i": 1, "tool": "robot_move_to", "args": {}, "results": [
                {"ok": False, "error": "invalid_args", "text": texts[kind]}]}]) + "\n")
        folders[name] = f
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.4.0")
    assert mark_episodes(tmp_path) == []              # not a change of v2.4.0
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.4.1")
    rows = mark_episodes(tmp_path)
    assert sorted(r["folder"].split("CR1_easy_s15_")[1] for r in rows) == ["qwen_listed",
                                                                         "qwen_named"]
    assert {c for r in rows for c in r["changes"]} == {"typed_as_text"}
    marked = json.loads((folders["qwen_listed"] / "result.json").read_text())
    assert provenance.not_comparable(marked, "v2.4.1") == [
        "a QwenCloud call refused for a list or number that came as text, which v2.4.1 reads"]
    for name in ("qwen_worded", "qwen_ranged"):
        row = json.loads((folders[name] / "result.json").read_text())
        assert provenance.not_comparable(row, "v2.4.1") == []


def test_a_minimal_loop_reply_with_more_pictures_than_kept_is_marked(tmp_path, monkeypatch):
    """Until v2.4.2 a minimal-loop reply that brought more pictures than the
    history keeps had the first of them dropped before the model saw them: such
    an earlier episode does not compare.  One whose replies stayed within it
    does, so does one with a wider window, one whose pictures had no pixels,
    and another harness's."""
    old = {**provenance.stamp(1, 1), "frozen": "v2.4.1", "freeze_clean": True}
    picture = {"frame_id": "demo0.0.front", "camera": "front", "width": 64, "height": 64,
               "path": "demo0.0.front.jpg"}
    blank = {**picture, "note": "this engine renders no pixels (mock)"}
    cases = {"loop_eight": ("loop", 6, [picture] * 8), "loop_six": ("loop", 6, [picture] * 6),
             "loop_wide": ("loop", 10, [picture] * 8), "loop_blank": ("loop", 6, [blank] * 8),
             "claude_eight": ("claude-code", 6, [picture] * 8)}
    folders = {}
    for name, (harness, keep, images) in cases.items():
        f = tmp_path / "agent" / f"CL1_easy_s15_{name}"
        f.mkdir(parents=True)
        (f / "result.json").write_text(json.dumps({"task": "CL1", "harness": harness,
                                                   "provenance": old}))
        (f / f"{f.name}.model.jsonl").write_text(
            json.dumps({"model": "m", "keep_images": keep}) + "\n")
        (f / f"{f.name}.jsonl").write_text("\n".join(json.dumps(r) for r in [
            {"header": True},
            {"i": 0, "turn": 1, "tool": "batch", "args": {"calls": []},
             "results": [{"ok": True, "text": "demonstration 0", "images": images[:4]},
                         {"ok": True, "text": "demonstration 1", "images": images[4:]}]}]) + "\n")
        folders[name] = f
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.4.1")
    assert mark_episodes(tmp_path) == []              # not a change of v2.4.1
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.4.2")
    [row] = mark_episodes(tmp_path)
    assert row["folder"].endswith("loop_eight") and row["changes"] == ["pictures_unseen"]
    marked = json.loads((folders["loop_eight"] / "result.json").read_text())
    assert provenance.not_comparable(marked, "v2.4.2") == [
        "a minimal-loop reply's pictures dropped before the model saw them, which v2.4.2 shows"]
    for name in ("loop_six", "loop_wide", "loop_blank", "claude_eight"):
        row = json.loads((folders[name] / "result.json").read_text())
        assert provenance.not_comparable(row, "v2.4.2") == []
