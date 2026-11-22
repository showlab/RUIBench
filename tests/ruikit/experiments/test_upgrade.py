"""A plan's results brought up to the version frozen now (v2.3.6): what compares
stays, what can be scored again is, and only the rest is moved aside to be run."""

import json

from ruikit.experiments import provenance
from ruikit.experiments.upgrade import upgrade

from test_rescore import _episode

AGENT = "claude-code-claude-opus-5-5"


def _result(out, name, task, frozen, **extra):
    f = out / AGENT / name
    f.mkdir(parents=True)
    old = {**provenance.stamp(1, 1), "frozen": frozen, "freeze_clean": True}
    (f / "result.json").write_text(json.dumps({"task": task, "provenance": old, "success": False,
                                               "usage": {"input_tokens": 1000, "output_tokens": 10},
                                               **extra}))
    return f


def test_only_what_does_not_compare_is_moved_and_scored_again_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "expected_version", lambda: "v2.3.6")
    stays = _result(tmp_path, "CR1_easy_s15_rui.pose-v1+aids", "CR1", "v2.3.5")
    old_wk4 = _result(tmp_path, "WK4_easy_s15_rui.pose-v1+aids", "WK4", "v2.3.3")
    wk6 = _episode(tmp_path, task="WK6", frozen="v2.3.3")          # scored again, then compares
    lock = _result(tmp_path, "CR2_hard_s15_rui.delta-v1+aids", "CR2", "v2.3.3")
    (lock / f"{lock.name}.jsonl").write_text(json.dumps(
        {"results": [{"data": {"tcp": {"rpy_deg": [0.0, 89.8, 0.0]}}}]}) + "\n")
    (tmp_path / "reviews.json").write_text(json.dumps({
        "_about": "reviews", f"{AGENT}/{old_wk4.name}": {"decision": "hack"},
        f"{AGENT}/{stays.name}": {"decision": "confirmed"}}))
    (tmp_path / "review" / AGENT / old_wk4.name).mkdir(parents=True)

    before = {p: p.read_text() for p in tmp_path.rglob("*.json")}
    plan = upgrade(tmp_path)                                       # what it would do: nothing written
    assert {p: p.read_text() for p in tmp_path.rglob("*.json")} == before
    assert sorted(plan["moved"]) == sorted([str(old_wk4), str(lock)])
    assert plan["kept"] == 2 and plan["cost"]["input_tokens"] == 2000

    done = upgrade(tmp_path, go=True)
    assert sorted(done["moved"]) == sorted(plan["moved"])
    assert stays.exists() and wk6.exists() and not old_wk4.exists() and not lock.exists()
    assert (tmp_path / "superseded" / "v2.3.3" / AGENT / old_wk4.name / "result.json").exists()
    assert (tmp_path / "superseded" / "v2.3.3" / "review" / AGENT / old_wk4.name).exists()
    reviews = json.loads((tmp_path / "reviews.json").read_text())
    assert f"{AGENT}/{old_wk4.name}" not in reviews and f"{AGENT}/{stays.name}" in reviews
    moved = json.loads((tmp_path / "superseded" / "v2.3.3" / "reviews.json").read_text())
    assert moved[f"{AGENT}/{old_wk4.name}"] == {"decision": "hack"}
    wk6_row = json.loads((wk6 / "result.json").read_text())
    assert wk6_row["scored_again"]["frozen"] == "v2.3.6"
    assert provenance.not_comparable(wk6_row, "v2.3.6") == []
    again = upgrade(tmp_path, go=True)                             # nothing left to do
    assert again["moved"] == [] and again["marked"] == [] and again["kept"] == 2
