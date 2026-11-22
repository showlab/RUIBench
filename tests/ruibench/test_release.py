"""What someone installing RUIBench meets first: the licence, the command
line, the seed splits, where fetched data goes, and a prepared asset whose
scan went missing (v13)."""

from pathlib import Path

import pytest

from ruibench.cli import main

ROOT = Path(__file__).resolve().parents[2]


def test_the_licence_and_credits_are_there():
    assert (ROOT / "LICENSE").read_text().lstrip().startswith("Apache License")
    assert "Version 2.0" in (ROOT / "LICENSE").read_text()
    assert "Apache License" in (ROOT / "NOTICE").read_text()
    credits = (ROOT / "CREDITS.md").read_text()
    assert "CC BY 4.0" in credits and "ARX" in credits and "Isaac" in credits
    assert 'license = "Apache-2.0"' in (ROOT / "pyproject.toml").read_text()


def test_tasks_take_a_suite_by_its_prefix_and_refuse_an_unknown_one(capsys):
    assert main(["tasks", "WK"]) == 0
    out = capsys.readouterr().out
    assert "WK1" in out and "CL1" not in out
    assert main(["tasks", "nonsense"]) == 2
    assert "unknown suite" in capsys.readouterr().err


def test_an_interface_that_is_only_a_contract_is_refused_before_any_episode(capsys):
    assert main(["run", "DI4", "--interface", "I2", "--seeds", "0"]) == 2
    err = capsys.readouterr().err
    assert "I2 is not implemented yet" in err and "Traceback" not in err


def test_rescore_says_what_it_checks(tmp_path, capsys):
    import json

    from ruibench.core.config import RunConfig
    from ruibench.runner import RunPlan, run_plan
    out = tmp_path / "x.jsonl"
    run_plan(RunPlan(["WK1"], levels=("easy",), seeds=(0,)),
             agent_factory=lambda i: i.reference_factory(i),
             config=RunConfig(engine="mock"), out=str(out))
    assert main(["rescore", str(out)]) == 0
    assert "rescored 1 episodes from their trajectories" in capsys.readouterr().out
    bad = json.loads(out.read_text().splitlines()[0])
    bad["score"]["failure"] = "timeout"                   # a stored label the scorer disagrees with
    out.write_text(json.dumps(bad) + "\n")
    assert main(["rescore", str(out)]) == 1
    # a log from before v15 kept positions only: its stored score is checked
    old = json.loads(out.read_text().splitlines()[0])
    old["trajectory"] = [{"t": s["t"], "objects": {}, "arms": {}} for s in old["trajectory"]]
    out.write_text(json.dumps(old) + "\n")
    assert main(["rescore", str(out)]) == 1
    assert "checked 1 older" in capsys.readouterr().out


def test_splits_deal_good_missions_out_dev_test_spare():
    from ruibench.tasks import splits

    data = splits.make(tasks=["WK1", "DI4"], solved={("WK1", "hard", 2): False})
    for task in ("WK1", "DI4"):
        for level in ("easy", "hard"):
            cell = data["tasks"][task][level]
            dealt = cell["dev"] + cell["test"] + cell["spare"]
            assert [len(cell[s]) for s in ("dev", "test", "spare")] == [15, 20, 5]
            assert len(set(dealt)) == 40
    # a mission the reference failed on Isaac is left out
    assert 2 not in data["tasks"]["WK1"]["hard"]["dev"]
    assert "failed" in data["tasks"]["WK1"]["hard"]["left_out"]["2"]
    assert data["frozen"] is False


def test_the_seed_splits_deal_every_task_and_level_without_overlap():
    """The committed splits: every main-board task that is built, at both
    levels, the sizes of the design, no seed in two splits, none that was left
    out.  A task still planned has no missions to deal (the contact suite's,
    while it is being built)."""
    from ruibench.tasks import freeze, splits
    from ruibench.tasks.loader import main_board
    from ruibench.tasks.spec import Status

    data = splits.load()
    # False while a draft; once frozen, the name of the version that froze it
    record = freeze.load()
    assert data["frozen"] == (record["name"] if record else False)
    for spec in main_board():
        if spec.status is not Status.IMPLEMENTED:
            assert spec.id not in data["tasks"], spec.id
            continue
        for level in spec.knob.levels:
            cell = data["tasks"][spec.id][level]
            dealt = [seed for name in splits.SPLITS for seed in cell[name]]
            assert [len(cell[name]) for name in splits.SPLITS] == [n for _, n in splits.SIZES]
            assert len(set(dealt)) == len(dealt)
            assert not set(dealt) & {int(seed) for seed in cell["left_out"]}


def test_run_takes_a_split(tmp_path, monkeypatch):
    import json

    from ruibench.tasks import splits

    fake = tmp_path / "splits.json"
    fake.write_text(json.dumps({"frozen": False, "tasks": {"WK1": {
        "easy": {"dev": [3], "test": [], "spare": []}}}}))
    monkeypatch.setattr(splits, "PATH", fake)
    monkeypatch.setattr(splits, "load", lambda path=fake: json.loads(fake.read_text()))
    out = tmp_path / "run.jsonl"
    assert main(["run", "WK1", "--levels", "easy", "--split", "dev", "-o", str(out)]) == 0
    [log] = [json.loads(line) for line in out.read_text().splitlines()]
    assert log["seed"] == 3 and log["config"]["split"] == "dev"


def test_fetched_data_stays_in_the_checkout_here():
    from ruibench.core import paths

    assert paths.checkout() == ROOT
    assert paths.data_dir("third_party", "third_party") == ROOT / "third_party"


def test_a_prepared_asset_whose_scan_is_gone_is_found(tmp_path):
    from ruibench.engines.isaac.prepare import missing_sources

    (tmp_path / "scan.usd").write_text("#usda 1.0\n")
    wrapper = tmp_path / "thing.usda"
    wrapper.write_text('#usda 1.0\ndef Xform "a" (prepend references = @./scan.usd@) {}\n'
                       'def Xform "b" (prepend references = @./gone.usd@) {}\n'
                       'def Material "m" { asset info:mdl:sourceAsset = @OmniPBR.mdl@ }\n')
    assert [p.name for p in missing_sources(wrapper)] == ["gone.usd"]
