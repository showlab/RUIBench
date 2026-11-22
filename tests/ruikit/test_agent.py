"""The public API: an Agent -- a model, a Harness, an interface and its
components -- run on a task under a condition writes one cell of the shared
results layout."""

import json

import pytest

import ruikit
from ruikit.agent import as_condition

from fake_model import FakeGateway, chat_calls


def test_an_agent_names_its_interface_by_form_and_abstraction():
    agent = ruikit.Agent(model="openai:m", interface=("program", "pose"), components=["aids"])
    assert agent.interface_id() == "rui.code/v1+aids" and agent.harness_label == "loop"
    assert agent.interface_id(as_condition("privileged")) == "rui.code/v1+aids@privileged"
    assert ruikit.Agent("m", interface="rui.step/v1").interface_id() == "rui.step/v1"
    assert as_condition("paused") == ruikit.Condition(clock="paused")
    with pytest.raises(ValueError):
        ruikit.Agent("m", harness="nope").harness_label


def test_running_an_agent_writes_one_cell(tmp_path):
    fake = FakeGateway([chat_calls(("camera_look", {})),
                        chat_calls(("task_done", {"report": "done"}))])
    try:
        agent = ruikit.Agent(model="opus5", interface=("per_action", "pose"), components=["aids"])
        row = ruikit.run(agent, task="WK1", level="easy", seed=0, condition="paused",
                         engine="mock", out=tmp_path, base_url=fake.url)
    finally:
        fake.close()
    folder = tmp_path / "loop-opus5@paused" / "WK1_easy_s0_rui.pose-v1+aids"
    assert json.loads((folder / "result.json").read_text()) == row
    assert row["profile"] == "rui.pose/v1+aids" and row["harness"] == "loop"
    assert row["think_seconds"] == 0.0 and row["replies"] == 2
    assert [path for path, _ in fake.got] == ["/v1/chat/completions"] * 2
