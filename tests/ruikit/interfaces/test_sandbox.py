"""code_run and agent-defined tools: online, confined, charged (approach-v2 §7.2, §8.2 rule 9)."""

import json
import shutil
from pathlib import Path

import pytest

from ruikit import Episode


@pytest.fixture
def code_ep():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.code/v1")
    yield ep
    ep.close()


def run(ep, source):
    return ep.call("code_run", {"source": source})


def test_a_program_acts_online(code_ep):
    r = run(code_ep, "a = robot.state()['data']['arms'][0]['tcp']['xyz_m']\n"
                     "robot.move_to(xyz_m=[a[0] + 0.1, a[1], a[2]])\n"
                     "b = robot.state()['data']['arms'][0]['tcp']['xyz_m']\n"
                     "result = [a, b]")
    assert not r.is_error, r.data
    a, b = r.data["result"]
    assert b[0] == pytest.approx(a[0] + 0.1, abs=0.005)     # the second read saw the move
    assert r.data["tool_calls"] == 3


def test_a_program_in_a_reply_with_other_calls_still_waits_for_its_moves(code_ep):
    """Parallel calls in one reply run their motions together -- but a
    program inside that reply gets each of its own calls' answers in turn."""
    src = ("a = robot.state()['data']['arms'][0]['tcp']['xyz_m']\n"
           "robot.move_to(xyz_m=[a[0] + 0.1, a[1], a[2]])\n"
           "result = robot.state()['data']['arms'][0]['tcp']['xyz_m'][0] - a[0]")
    run_it, events = code_ep.call_batch([("code_run", {"source": src}), ("task_events", {})])
    assert not run_it.is_error, run_it.data
    assert run_it.data["result"] == pytest.approx(0.1, abs=0.005)
    assert not events.is_error


def test_tool_errors_raise_inside_the_program(code_ep):
    r = run(code_ep, "try:\n    vision.pose(frame_id='f9.front', id='f9.front#1')\n"
                     "except ToolCallError as e:\n    result = e.code")
    assert r.data["result"] == "stale_frame"


def test_the_namespace_persists_between_runs(code_ep):
    run(code_ep, "memo = 41")
    assert run(code_ep, "result = memo + 1").data["result"] == 42


def test_a_run_answers_with_its_own_result_only(code_ep):
    """A run that sets no `result` answers None, not the last run's (v2.3.4)."""
    assert run(code_ep, "result = 'first'").data["result"] == "first"
    assert run(code_ep, "memo = 1").data["result"] is None


def test_the_program_tool_names_only_what_its_interface_offers():
    """code_run's description gave robot.move_to as its example, which five of
    the six program interfaces lack, and named the imports only by example (v2.3.4)."""
    import ast
    import re
    from pathlib import Path

    from ruikit.interfaces import sandbox
    from ruikit.serve.schemas import tool_entries
    from ruikit.interfaces import load_profile

    child = (Path(sandbox.__file__).parent / "child.py").read_text()
    allowed = next(ast.literal_eval(n.value) for n in ast.parse(child).body
                   if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ALLOWED_MODULES")
    for interface in ("code-delta", "code-step", "code-joint", "code-point", "code-skill", "code"):
        profile = load_profile(f"rui.{interface}/v1+aids")
        text = next(e["description"] for e in tool_entries(profile) if e["name"] == "code_run")
        head, _, listed = text.partition("The functions a program can call.")
        for name in re.findall(r"\b(robot|skill|camera|vision|plan)\.(\w+)\(", head):
            assert f"{name[0]}_{name[1]}" in profile.code_tools, (interface, name)
        assert "camera_zoom" not in head
        imports = re.search(r"may import only (.*?); there is", " ".join(head.split())).group(1)
        assert set(re.split(r", | and ", imports)) == allowed


@pytest.mark.parametrize("source", [
    "import os\nresult = os.listdir('/')",
    "import subprocess",
    "result = open('/etc/passwd').read()",
    "import ruibench",
])
def test_no_way_out(code_ep, source):
    r = run(code_ep, source)
    assert r.is_error


def test_programs_cannot_see_the_repository(code_ep):
    # even numpy's file readers find nothing: the repo is not mounted
    repo_file = str(Path(__file__).resolve().parents[3] / "pyproject.toml")
    r = run(code_ep, "import numpy as np\n"
                     f"try:\n    np.loadtxt({repo_file!r}, dtype=str)\n"
                     "    result = 'read'\nexcept Exception as e:\n    result = type(e).__name__")
    if code_ep.sandbox.isolation == "bwrap":
        assert r.data["result"] != "read"


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap not installed")
def test_bubblewrap_is_used_when_available(code_ep):
    assert run(code_ep, "result = 1").data["isolation"] == "bwrap"


def test_only_the_profile_tools_are_callable(code_ep):
    r = run(code_ep, "result = sorted(k for k in dir() if not k.startswith('_'))")
    names = r.data["result"]
    assert "skill_pick" not in names and "policy_run" not in names and "code_run" not in names


def test_compute_time_is_charged(code_ep):
    t0 = code_ep.sim_time
    run(code_ep, "x = 0\nfor i in range(3_000_000):\n    x += i\nresult = x")
    assert code_ep.sim_time > t0


def test_agents_can_build_tools_from_tools():
    ep = Episode("WK1", level="easy", seed=0, profile="open/v1")
    try:
        schema = json.dumps({"type": "object", "properties": {"dz": {"type": "number"}},
                             "required": ["dz"]})
        code = ("def run(dz):\n"
                "    p = robot.state()['data']['arms'][0]['tcp']['xyz_m']\n"
                "    robot.move_to(xyz_m=[p[0], p[1], p[2] + dz])\n"
                "    return robot.state()['data']['arms'][0]['tcp']['xyz_m'][2]")
        d = ep.call("toolkit_define", {"name": "my_lift", "description": "Raise the hand by dz.",
                                       "input_schema": schema, "code": code})
        assert not d.is_error, d.text
        z0 = ep.call("robot_state", {}).data["arms"][0]["tcp"]["xyz_m"][2]
        r = ep.call("my_lift", {"dz": 0.05})
        assert not r.is_error, r.data
        assert r.data["result"] == pytest.approx(z0 + 0.05, abs=0.005)
        assert r.data["tool_calls"] == 3
        # no new powers: a custom tool cannot run programs or define tools
        bad = ep.call("toolkit_define", {"name": "my_escape", "description": "Tries code_run inside.",
                                         "input_schema": json.dumps({"type": "object", "properties": {}}),
                                         "code": "def run():\n    return code_run(source='1')"})
        assert not bad.is_error
        assert ep.call("my_escape", {}).is_error
        assert ep.call("toolkit_define", {"name": "robot_move_to", "description": "shadowing a builtin",
                                          "input_schema": "{}", "code": "def run(): pass"}).error == "invalid_args"
    finally:
        ep.close()


def test_a_sandbox_closed_while_a_program_waits_fails_that_call_only():
    """When the sandbox was closed while its program waited on a tool (the
    call ended the episode), the host wrote to the dead process and the whole
    run died with AttributeError (RUI-Zero, 10/2).  Now the program fails."""
    from ruikit.interfaces.sandbox.host import Sandbox

    box = Sandbox(isolation="process")
    try:
        def on_call(name, args):
            box.close()                        # what an episode ending in the call does
            return {"ok": True, "data": {}}

        out = box.run("x = robot.state()\nresult = 1", {"robot_state": ["arm"]}, on_call)
        assert out["error"] and "stopped while it ran" in out["error"]
        # and the next program gets a fresh process
        again = box.run("result = 2", {}, lambda name, args: {})
        assert again["value"] == 2 and not again["error"]
    finally:
        box.close()
