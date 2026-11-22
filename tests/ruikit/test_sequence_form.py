"""The sequence form: a program that only acts -- a series of actions with
nothing seen in between -- and the looking left to the replies."""

import pytest

from ruikit.core.tool import load_builtin_tools
from ruikit.env.episode import Episode
from ruikit.interfaces import load_profile
from ruikit.serve.schemas import tool_entries

load_builtin_tools()


@pytest.fixture
def ep():
    return Episode("DS1", level="easy", seed=0, profile="rui.seq-delta/v1+aids", engine="mock")


def test_a_program_acts_and_the_looking_stays_outside():
    prof = load_profile("rui.seq-delta/v1+aids")
    assert prof.code_tools == ("robot_move_delta", "robot_gripper", "robot_wait", "robot_stop")
    assert {"camera_look", "vision_detect", "vision_point", "plan_grasps", "code_run"} <= set(prof.tools)
    assert not set(prof.code_tools) & {"camera_look", "robot_state", "vision_detect"}
    assert prof.tool_params("code_run") == {"blind": True}
    # an abstraction's tool that does not act (inverse kinematics) is called as a tool
    joint = load_profile("rui.seq-joint/v1")
    assert "robot_solve_ik" in joint.tools and "robot_solve_ik" not in joint.code_tools
    code_run = [e for e in tool_entries(prof) if e["name"] == "code_run"][0]["description"]
    assert "return nothing" in code_run and "-> data" not in code_run


def test_each_call_hands_the_program_nothing_and_the_reports_come_after(ep):
    r = ep.call("code_run", {"source": "a = robot.move_delta(dxyz_m=[0.01, 0, 0])\n"
                                       "b = robot.gripper('open')\n"
                                       "result = [a, b]"})
    out = r.as_json()
    assert out["ok"], out
    assert out["data"]["result"] == [{}, {}]
    assert [a["call"] for a in out["data"]["actions"]] == ["robot_move_delta", "robot_gripper"]
    assert all(a["ok"] for a in out["data"]["actions"])
    assert "2 action(s)" in out["text"]


def test_a_failed_action_does_not_stop_the_series(ep):
    r = ep.call("code_run", {"source": "robot.move_delta(dxyz_m=[1, 0])\n"
                                       "robot.move_delta(dxyz_m=[0, 0.01, 0])"})
    out = r.as_json()
    assert out["ok"], out
    first, second = out["data"]["actions"]
    assert not first["ok"] and first["error"] == "invalid_args"
    assert second["ok"]
    assert "1 of them reported a failure" in out["text"]


def test_a_program_cannot_look(ep):
    out = ep.call("code_run", {"source": "f = camera.look(['front'])"}).as_json()
    assert not out["ok"] and out["error"] == "tool_failed"
    assert ep.call("camera_look", {}).as_json()["ok"]            # as a tool it can


def test_there_is_no_controller_mode(ep):
    out = ep.call("code_run", {"source": "def step(state):\n    pass",
                               "mode": "controller", "seconds": 1.0}).as_json()
    assert not out["ok"] and out["error"] == "invalid_args"


def test_a_program_of_the_program_form_still_sees_its_results():
    ep = Episode("DS1", level="easy", seed=0, profile="rui.code-delta/v1+aids", engine="mock")
    out = ep.call("code_run", {"source": "result = robot.move_delta(dxyz_m=[0.01, 0, 0])['ok']"}).as_json()
    assert out["ok"] and out["data"]["result"] is True and "actions" not in out["data"]
