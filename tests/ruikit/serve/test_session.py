"""An episode an outside agent drives from the shell, one reply a command, and
the tools described as the interface actually has them."""

import json
import re

from ruikit.interfaces import load_profile
from ruikit.serve import session
from ruikit.serve.schemas import tool_entries


def _run(capsys, *argv):
    session.main(list(argv))
    return capsys.readouterr().out


def test_an_agent_drives_an_episode_from_the_shell(tmp_path, capsys):
    d = str(tmp_path / "s")
    out = _run(capsys, "start", "--dir", d, "--task", "DS1", "--profile", "rui.code/v1",
               "--engine", "mock")
    assert "ready after" in out and "Put Tom's present on the red mat." in out
    info = json.loads(_run(capsys, "info", "--dir", d))
    assert info["limits"] == {"replies": 600, "simulated_seconds": 480.0,
                              "clock": "the world waits while you think; your moves take their time"}
    # a function inside code_run is documented by its program name
    doc = json.loads(_run(capsys, "tool", "--dir", d, "robot.move_to"))
    assert doc["called"] == "inside a code_run program" and doc["parameters"]
    out = _run(capsys, "call", "--dir", d, "code_run",
               json.dumps({"source": "result = [f['frame_id'] for f in camera.look()['data']['frames']]"}),
               "--why", "see the table")
    assert "[1] ok" in out and "replies 1 of 600" in out
    # a program as it is written, from a file: no escaping into one line of JSON
    (tmp_path / "prog.py").write_text('frames = camera.look()["data"]["frames"]\n'
                                      'result = "seen: %d" % len(frames)\n')
    out = _run(capsys, "call", "--dir", d, "code_run", "--program", str(tmp_path / "prog.py"))
    assert "[1] ok" in out and "seen: " in out and "replies 2 of 600" in out
    assert json.loads(_run(capsys, "tool", "--dir", d, "vision.pose"))["data"].startswith("frame_id")
    out = _run(capsys, "call", "--dir", d, "--batch",
               json.dumps([["task_info", {}], ["task_info", {}]]))
    assert "[2] ok" in out and "replies 3 of 600" in out          # a batch is one reply
    out = _run(capsys, "call", "--dir", d, "task_done", json.dumps({"report": "stopping"}))
    assert "THE EPISODE HAS ENDED (agent_done)" in out
    for _ in range(100):                                          # the server finalizes and exits
        state = json.loads(_run(capsys, "status", "--dir", d))
        if state["status"] == "over":
            break
        import time
        time.sleep(0.1)
    assert state["status"] == "over" and state["replies"] == 4 and state["reason"] == "agent_done"
    notes = json.loads((tmp_path / "s" / "notes.json").read_text())
    assert notes == [{"row": 0, "call": 0, "note": "see the table"}]


def test_each_interface_is_told_its_own_settings():
    step = next(e for e in tool_entries(load_profile("rui.step/v1")) if e["name"] == "robot_step")
    sizes = re.search(r"\(in this interface ([^)]*)\)", step["description"]).group(1)
    assert sizes.startswith("0.25 / 1 / 4 cm") and "2 / 5 / 15 degrees" in sizes
    code = next(e for e in tool_entries(load_profile("rui.code/v1")) if e["name"] == "code_run")
    for fn in ("camera.look(", "vision.pose(", "robot.move_to(", "robot.gripper(", "task.done("):
        assert fn in code["description"], fn
    point = next(e for e in tool_entries(load_profile("rui.point/v1")) if e["name"] == "robot_move_to_pixel")
    assert "robot_move_to;" not in point["description"]          # a tool I3 does not have


def test_a_reply_is_one_action_per_arm_on_the_tool_interfaces():
    """[step, wait, step] would chain steps into one reply (an I1 agent did, ten a
    reply, on WK5): refused whole; two arms acting together stay one reply."""
    from ruikit.env import Episode

    ep = Episode("DS1", level="easy", seed=0, profile="rui.step/v1", engine="mock")
    try:
        before = ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose.pos
        out = ep.call_batch([("robot_step", {"move": "forward", "size": "coarse"}),
                             ("robot_wait", {"seconds": 0.5}),
                             ("robot_step", {"move": "forward", "size": "coarse"})])
        assert all(r.is_error for r in out) and "one action per arm" in out[0].hint
        assert ep.engine.state().arm(ep.scene.arm_ids[0]).eef_pose.pos == before
        assert ep.replies == 1                                    # still a reply
        ok = ep.call_batch([("robot_step", {"move": "forward", "size": "coarse"}),
                            ("camera_look", {})])
        assert not any(r.is_error for r in ok)
    finally:
        ep.close()
    two = Episode("WK4", level="easy", seed=0, profile="rui.pose/v1", engine="mock")
    try:
        arms = [a.split("/")[-1] for a in two.scene.arm_ids]
        both = two.call_batch([("robot_gripper", {"arm": arms[0], "command": "open"}),
                               ("robot_gripper", {"arm": arms[1], "command": "open"})])
        assert not any(r.is_error for r in both)
    finally:
        two.close()


def test_the_program_description_names_arguments_in_their_real_order():
    """code_run's own example must match each function's positional order
    (WK6 via I4 found robot.move_to(arm, xyz_m, ...) against (xyz_m, arm, ...))."""
    import dataclasses

    from ruikit.core.tool import get_tool, load_builtin_tools

    load_builtin_tools()
    desc = " ".join(get_tool("code_run").spec.description.split())
    for call in re.findall(r"(\w+)\.(\w+)\(([a-z_, ]+)\)", desc):
        family, rest, args = call
        tool = get_tool(f"{family}_{rest}")
        names = [f.name for f in dataclasses.fields(tool.spec.params)]
        given = [a.strip() for a in args.split(",")]
        assert names[:len(given)] == given, (family, rest, given, names)


def test_a_caller_gone_before_its_answer_does_not_end_the_session(tmp_path, capsys):
    """An agent stopped while its call ran is gone before the answer: the session
    goes on, and a stop ends it (v2.3.7: the broken pipe ended the server, and
    Isaac hung until its worker gave up on it)."""
    import socket
    import struct
    import time

    d = tmp_path / "s"
    _run(capsys, "start", "--dir", str(d), "--task", "DS1", "--profile", "rui.code/v1",
         "--engine", "mock")
    state = json.loads((d / "session.json").read_text())
    for _ in range(3):
        c = socket.create_connection(("127.0.0.1", state["port"]))
        c.sendall((json.dumps({"token": state["token"], "op": "call",
                               "calls": [["task_info", {}]]}) + "\n").encode("utf-8"))
        c.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
        c.close()                                                 # gone, with a reset
    time.sleep(0.5)
    assert "ok" in _run(capsys, "call", "--dir", str(d), "task_info", "{}")
    session._ask(d, {"op": "stop"})
    for _ in range(100):
        if json.loads(_run(capsys, "status", "--dir", str(d)))["status"] == "over":
            break
        time.sleep(0.1)
    assert json.loads(_run(capsys, "status", "--dir", str(d)))["reason"] == "agent_stopped"
