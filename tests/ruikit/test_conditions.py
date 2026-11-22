"""v1.17: the evaluation's conditions and the Harness components it ablates --
the privileged control, the clocks a row may run on, what an episode's end
says of a claimed finish, the plan of the experiments, the operation guide."""
import json

import pytest

from ruikit import Episode
from ruikit.conditions import Condition, clock_settings
from ruikit.experiments import plans
from ruikit.experiments import results as cells
from ruikit.harness.components.guides import guide_text
from ruikit.interfaces import load_profile
from ruikit.report import tables


def test_the_privileged_condition_answers_with_the_simulators_values():
    """``@privileged`` -- by name only, never a default -- puts what a tool
    estimates where the thing is, and the log says which condition it was."""
    prof = load_profile("rui.pose/v1+aids@privileged")
    assert prof.perception == "privileged" and prof.id.endswith("@privileged")
    assert prof.perception_settings() == {"perception": "privileged"}
    assert load_profile("rui.pose/v1+aids").perception == "model"
    ep = Episode("DS1", level="easy", seed=0, profile=prof, engine="mock")
    try:
        assert ep.perception_info == {"perception": "privileged"}
        fid = ep.call("camera_look", {"cameras": ["front"]}).data["frames"][0]["frame_id"]
        for det in ep.call("vision_detect", {"frame_id": fid, "query": "present"}).data["objects"]:
            est = ep.call("vision_pose", {"frame_id": fid, "id": det["id"]}).data
            name = ep.perception.resolve(ep.frames.get(fid), det["id"])
            true = ep.station.to_station(ep.engine.state().objects[name].pose.pos)
            assert est["xyz_m"] == pytest.approx(true, abs=1e-4)
    finally:
        ep.close()


def test_under_the_privileged_condition_the_tools_say_they_are_exact():
    """A control changes only what perception answers -- and the tools say what
    they do: no talk of errors where there are none, in a program's functions too."""
    from ruikit.serve.schemas import PRIVILEGED, code_reference, tool_entries

    for pid in ("rui.pose/v1+aids@privileged", "rui.code/v1+aids@privileged"):
        prof = load_profile(pid)
        text = " ".join(e["description"] for e in tool_entries(prof)) + code_reference(prof)
        assert all(said not in text for _, said, _ in PRIVILEGED), pid
        assert "perception is privileged" in text, pid
    double = " ".join(e["description"] for e in tool_entries(load_profile("rui.pose/v1@synthetic")))
    assert all(said in double for _, said, _ in PRIVILEGED)       # what the test double says


def test_the_privileged_condition_is_written_in_the_episodes_log(tmp_path):
    ep = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1@privileged", engine="mock",
                 log_dir=str(tmp_path))
    ep.call("task_done", {"report": "-"})
    ep.close()
    log = json.loads(next(tmp_path.glob("*.episode.jsonl")).read_text().splitlines()[0])
    assert log["condition"] == "privileged"


@pytest.mark.parametrize("clock,episode", [
    ("8s", {"clock": "fixed", "think_seconds": 8.0, "think_per": "reply"}),
    ("2.5s", {"clock": "fixed", "think_seconds": 2.5, "think_per": "reply"}),
    ("paused", {"clock": "zero"}),
    ("measured", {"clock": "measured"})])
def test_a_rows_clock_is_the_episodes(clock, episode):
    assert clock_settings(clock) == episode == Condition(clock).episode_clock()
    assert cells.agent_dir("loop", "openai:m", clock) == (
        "loop-openai-m" if clock == "8s" else f"loop-openai-m@{clock}")


def test_a_clock_that_is_none_is_refused():
    with pytest.raises(ValueError):
        clock_settings("8")
    with pytest.raises(ValueError):
        Condition("8s", information="exact")
    with pytest.raises(ValueError):
        plans.plan(["m"], ["WK1"], ["easy"], "0", ["rui.pose/v1"], clock="slow")


def test_the_privileged_control_is_said_in_the_interface_id():
    assert Condition(information="privileged").interface("rui.pose/v1+aids") == \
        "rui.pose/v1+aids@privileged"
    assert Condition().interface("rui.pose/v1+aids") == "rui.pose/v1+aids"
    with pytest.raises(ValueError):
        Condition(information="privileged").interface("rui.pose/v1@sam3")


def test_the_plan_is_the_papers_experiments():
    """Rows are agents under a condition, columns the 12 fixed RUIs and RUI-Free,
    the easy level on the first test seed; the ablations pair with them."""
    from ruikit import naming

    exps = {e.name: e for e in plans.load_plan("ruikit/configs/experiments/paper.yaml")}
    assert list(exps) == ["rui", "rui-seed2", "rui-seq", "rui-seq-control", "rui-seq-seeds",
                          "check-report", "check-correct"]
    twelve = [p for p in exps["rui"].profiles if p != "rui.free/v1"]
    assert len(twelve) == 12 and all(p.endswith("+aids") for p in twelve)
    assert {cells.column_of(p) for p in exps["rui"].profiles} == set(naming.RUIS)
    assert {(r.harness, r.model) for r in exps["rui"].agents} == {
        ("claude-code", "claude-opus-5-5"), ("codex", "gpt-6-astra")}
    assert {cells.column_of(p) for p in exps["rui-seq"].profiles} == {
        "sequence-delta", "sequence-step", "sequence-point"}
    assert all(p.endswith("+check") for p in exps["check-report"].profiles)
    jobs = plans.select(exps.values())
    assert {(j.task, j.level, j.seed) for j in jobs if j.task == "CL2"} == {
        ("CL2", "easy", s) for s in (15, 16, 17, 18, 19)}
    assert {j.seed for j in plans.select([exps["rui"]])} == {15}
    assert all(not e.owners for e in exps.values())                   # no one's name in it


def test_a_result_says_whether_the_agent_claimed_it_was_done(tmp_path):
    """The verdict beside the agent's own word: an episode it said was done and
    failed is a false claim -- the cost of having no success check."""
    ep = Episode("CL2", level="easy", seed=0, profile="rui.pose/v1", engine="mock",
                 log_dir=str(tmp_path))
    ep.call("task_done", {"report": "I think so"})
    ep.close()
    row = cells.result_row(harness="loop", model="m", task="CL2", level="easy", seed=0,
                           profile="rui.pose/v1", folder=tmp_path, engine="mock",
                           reason=ep.outcome.reason, success=ep.outcome.success)
    assert row["declared_done"] is True and row["success"] is False
    assert row["furthest_milestone"] == "film_watched"      # the first of four, and the furthest
    rows = [row, {**row, "declared_done": False, "seed": 1},
            {**row, "success": True, "seed": 2}]
    claims = "\n".join(tables.claims(rows))
    assert "| 3 | 2 | 1/2 = 50%" in claims


def test_an_operation_guide_is_added_to_what_the_agent_is_told():
    from ruikit.harness.minimal_loop import instructions

    prof = load_profile("rui.pose/v1+aids+guide")
    assert prof.guide == "manipulation" and prof.id == "rui.pose/v1+aids+guide"
    ep = Episode("DS1", level="easy", seed=0, profile=prof, engine="mock")
    try:
        assert instructions(ep).endswith(guide_text("manipulation"))
    finally:
        ep.close()
    ep = Episode("DS1", level="easy", seed=0, profile="rui.pose/v1+aids", engine="mock")
    try:
        assert "Operating the robot" not in instructions(ep)
    finally:
        ep.close()
    with pytest.raises(ValueError, match="no operation guide"):
        load_profile("rui.pose/v1+guide:none-such")
