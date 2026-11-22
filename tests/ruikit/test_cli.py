"""The command line: one spelling of every option, the interfaces by the paper's names."""

import pytest

from ruikit import cli


def test_interfaces_lists_the_thirteen_by_name_and_id(capsys):
    assert cli.main(["interfaces"]) == 0
    out = capsys.readouterr().out
    for name, pid in (("program-skill", "rui.code-skill/v1+aids"), ("program-pose", "rui.code/v1+aids"),
                      ("rui-free", "rui.free/v1"), ("sequence-delta", "rui.seq-delta/v1+aids")):
        assert name in out and pid in out
    assert "Beyond the paper" in out and "open/v1" in out


def test_one_interface_by_name(capsys):
    assert cli.main(["interface", "program-skill"]) == 0
    assert "id: rui.code-skill/v1+aids" in capsys.readouterr().out    # YAML, as the option says


def test_run_takes_the_papers_names_and_the_older_spellings(capsys):
    args = ["--plan", "paper", "--experiment", "rui", "--agent", "claude-code", "--task", "WK1",
            "--dry-run"]
    assert cli.main(["run"] + args + ["--interface", "program-pose", "rui-free"]) == 0
    new = capsys.readouterr().out
    assert "s15 rui.code/v1+aids" in new and "s15 rui.free/v1" in new and "2 episodes" in new
    assert cli.main(["run", "--plan", "paper", "--experiments", "rui", "--harness", "claude-code",
                     "--tasks", "WK1", "--columns", "program", "free", "--dry-run"]) == 0
    assert capsys.readouterr().out == new


def test_run_without_a_plan_needs_a_model(capsys):
    with pytest.raises(SystemExit):
        cli.main(["run", "--task", "WK1", "--dry-run"])
    assert "--model" in capsys.readouterr().err
    assert cli.main(["run", "--agent", "minimal-loop", "--model", "openai:m", "--task", "WK1",
                     "--level", "easy", "--seeds", "test:1", "--interface", "skill", "--dry-run"]) == 0
    assert "s15 rui.skill/v1+aids" in capsys.readouterr().out


def test_episode_needs_one_of_each(capsys):
    with pytest.raises(SystemExit):
        cli.main(["episode", "--task", "WK1", "--engine", "mock"])          # no model
    with pytest.raises(SystemExit):
        cli.main(["episode", "--task", "WK", "--model", "openai:m"])        # a suite is six tasks


def test_serve_mcp_speaks_the_shared_options(monkeypatch):
    seen = {}
    import ruikit.serve.mcp_stdio as mcp

    monkeypatch.setattr(mcp, "main", lambda argv: seen.setdefault("argv", argv) and 0)
    cli.main(["serve-mcp", "--task", "WK1", "--seeds", "test:1", "--interface", "program-skill",
              "--clock", "8s", "--engine", "mock", "--out", "logs"])
    argv = seen["argv"]
    assert argv[argv.index("--profile") + 1] == "rui.code-skill/v1+aids"
    assert argv[argv.index("--seed") + 1] == "15"
    assert argv[argv.index("--clock") + 1] == "fixed" and argv[argv.index("--think-seconds") + 1] == "8"
    assert argv[argv.index("--log-dir") + 1] == "logs"
