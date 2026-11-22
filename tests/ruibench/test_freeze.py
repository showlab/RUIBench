"""Freezing for the test split: the fingerprint, what drifts, what is refused."""

import json

import pytest


@pytest.fixture
def frozen(tmp_path, monkeypatch):
    """The real splits, copied; frozen into files under tmp_path."""
    from ruibench.tasks import freeze, splits

    split_copy = tmp_path / "splits.json"
    split_copy.write_text(splits.PATH.read_text())
    record_path = tmp_path / "frozen.json"
    monkeypatch.setattr(freeze, "PATH", record_path)
    monkeypatch.setattr(splits, "PATH", split_copy)
    record = freeze.freeze("v-test", path=record_path, splits_path=split_copy)
    return record, record_path, split_copy


def test_a_fresh_freeze_has_no_drift_and_marks_the_splits(frozen):
    from ruibench.tasks import freeze

    record, record_path, split_copy = frozen
    assert len(record["files"]) > 50
    assert "ruikit/configs/interfaces.yaml" in record["files"]
    assert any(p.startswith("ruibench/tasks/cards/") for p in record["files"])
    assert json.loads(split_copy.read_text())["frozen"] == "v-test"
    assert freeze.drift(record_path, now=record["files"]) == []


def test_a_changed_file_or_t_ref_is_drift(frozen, monkeypatch):
    import ruibench.core.config as config
    from ruibench.tasks import freeze

    record, record_path, _ = frozen
    now = dict(record["files"])
    first = sorted(now)[0]
    now[first] = "0" * 64
    now["ruibench/tasks/impl/new_task.py"] = "1" * 64
    lines = freeze.drift(record_path, now=now)
    assert f"changed: {first}" in lines and "added: ruibench/tasks/impl/new_task.py" in lines
    monkeypatch.setattr(config, "DEFAULT_T_REF_SECONDS", record["t_ref_seconds"] + 1)
    assert any(line.startswith("T_ref:") for line in freeze.drift(record_path, now=record["files"]))


def test_frozen_splits_are_not_regenerated(frozen):
    from ruibench.core.errors import FrozenAssetError
    from ruibench.tasks import splits

    with pytest.raises(FrozenAssetError):
        splits.main([])


def test_validate_reports_drift_after_a_freeze(frozen, monkeypatch):
    import importlib

    from ruibench.tasks import freeze

    V = importlib.import_module("ruibench.tasks.validate")      # the module, not the function

    record, _, _ = frozen
    changed = dict(record["files"])
    changed[sorted(changed)[0]] = "0" * 64
    monkeypatch.setattr(freeze, "fingerprint", lambda root=None: changed)
    findings = V._freeze_findings()
    assert findings and all(f.level == "warning" for f in findings)
    assert "changed:" in findings[0].message


def test_the_mock_is_not_fingerprinted_but_what_isaac_shares_with_it_is():
    """v1.2: the mock never produces a result, so fixing it voids nothing; the
    grasp test the Isaac engine also uses moved to engines/base.py."""
    from ruibench.tasks import freeze

    prints = freeze.fingerprint()
    assert "ruibench/engines/mock.py" not in prints
    assert "ruibench/engines/base.py" in prints
    # perception from pixels is part of what an episode sees (v1.3)
    for path in ("ruikit/perception/backends/pixels.py", "ruikit/perception/service/client.py",
                 "ruikit/perception/service/server.py", "ruikit/serve/schemas.py"):
        assert path in prints, path
    assert "ruikit/extras/backends/__init__.py" in prints      # which tools are offered (v1.4)
    from ruibench.engines import base, mock
    assert mock.in_grasp is base.in_grasp


def test_a_ready_made_agents_set_up_is_fingerprinted():
    """v2.1: a ready-made agent's prompt, what it is given, its self-test, the
    session it reaches and the audit that decides which of its runs count shape
    its episodes, as the simple loop shapes its own."""
    from ruibench.tasks import freeze

    prints = freeze.fingerprint()
    for path in ("ruikit/harness/session_agent.py", "ruikit/harness/claude_code.py",
                 "ruikit/harness/codex.py", "ruikit/harness/mcp.py", "ruikit/harness/selftest.py",
                 "ruikit/harness/minimal_loop.py", "ruikit/serve/mcp_stdio.py",
                 "ruikit/serve/session_server.py", "ruikit/experiments/rules_audit.py"):
        assert path in prints, path
