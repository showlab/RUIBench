"""The fair hook-up of a ready-made agent (v1.14): MCP attached to a session
that is already running, ``batch`` for several calls as one reply, pictures
inline -- and, for an agent that still uses the shell, one call a command."""

import json
import os
import subprocess
import sys
import time
from pathlib import Path

from ruikit.serve import session, session_server

ENV = dict(os.environ)


def _start(tmp_path, capsys, profile="rui.pose/v1+aids"):
    d = str(tmp_path / "s")
    session.main(["start", "--dir", d, "--task", "DS1", "--profile", profile, "--engine", "mock"])
    assert "ready after" in capsys.readouterr().out
    return d


def _mcp(d, *messages):
    """Send JSON-RPC messages to the bridge, one a line; its answers, by id."""
    lines = "".join(json.dumps(m) + "\n" for m in messages)
    out = subprocess.run([sys.executable, "-m", "ruikit.serve.mcp_stdio", "--session", d],
                         input=lines, capture_output=True, text=True, env=ENV, timeout=120)
    return {r["id"]: r for r in map(json.loads, out.stdout.splitlines()) if "id" in r}


def test_mcp_drives_a_running_session_one_reply_a_call(tmp_path, capsys):
    d = _start(tmp_path, capsys)
    got = _mcp(d,
               {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
               {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
               {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                "params": {"name": "task_info", "arguments": {}}},
               {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                "params": {"name": "batch", "arguments": {"calls": [
                    {"name": "camera_look", "arguments": {}},
                    {"name": "robot_state", "arguments": {}}]}}})
    assert "DS1" in got[1]["result"]["instructions"]
    names = [t["name"] for t in got[2]["result"]["tools"]]
    assert "batch" in names and "plan_grasps" in names and "robot_move_to" in names
    assert "robot_waypoints" not in names
    batch = next(t for t in got[2]["result"]["tools"] if t["name"] == "batch")
    assert set(batch["inputSchema"]["properties"]["calls"]["items"]["properties"]["name"]["enum"]) \
        == set(names) - {"batch"}
    text = [c["text"] for c in got[3]["result"]["content"] if c["type"] == "text"]
    assert "replies 1 of" in text[-1]
    text = [c["text"] for c in got[4]["result"]["content"] if c["type"] == "text"]
    assert text[-1].startswith("simulated time") and "replies 2 of" in text[-1]   # one reply
    assert any(t.startswith("[1] camera_look") for t in text)
    assert any(t.startswith("[2] robot_state") for t in text)
    assert not any("path" in t for t in text if t.startswith("[1]"))   # no file paths to open
    session.main(["stop", "--dir", d])


def test_a_picture_comes_inline_for_an_mcp_client(tmp_path):
    from ruikit.core.result import Image, ToolResult

    res = ToolResult(data={}, images=[Image(frame_id="f1.front", camera="front", width=2,
                                            height=2, png=b"\x89PNG-bytes")])
    plain = session_server._result(res, tmp_path)
    inline = session_server._result(res, tmp_path, inline=True)
    assert "png_b64" not in plain["images"][0]
    assert inline["images"][0]["png_b64"] == "iVBORy1ieXRlcw=="


def test_one_session_call_per_shell_command(tmp_path, capsys):
    """A loop or a chain of calls in one shell command is a program in disguise:
    the second call of the command is refused; the next command may call."""
    d = _start(tmp_path, capsys, profile="rui.step/v1")
    call = f"{sys.executable} -m ruikit.serve.session call --dir {d} task_info '{{}}'"
    out = subprocess.run(["bash", "-c", f"{call} && {call}"], capture_output=True, text=True,
                         env=ENV, timeout=120).stdout
    assert out.count("[1] ok") == 1 and "one session call per shell command" in out
    out = subprocess.run(["bash", "-c", f"for i in 1 2; do {call}; done"], capture_output=True,
                         text=True, env=ENV, timeout=120).stdout
    assert out.count("[1] ok") == 1 and "one session call per shell command" in out
    out = subprocess.run(["bash", "-c", call], capture_output=True, text=True, env=ENV,
                         timeout=120).stdout
    assert "[1] ok" in out and "replies 3 of" in out        # a command of its own: fine
    session.main(["stop", "--dir", d])
    for _ in range(100):
        if session._state(tmp_path / "s").get("status") == "over":
            break
        time.sleep(0.1)


def test_a_failed_call_keeps_its_pictures_for_claude_code():
    """Claude Code shows the model only the text of a result flagged isError: a
    move that ended blocked, or a program that raised, lost its pictures.  The
    bridge it starts (``--keep-pictures``) does not flag a reply that carries
    pictures; the failure is in the reply's text (v2.3.4)."""
    from ruikit.harness import claude_code
    from ruikit.harness.mcp import _mcp_server
    from ruikit.serve.mcp_stdio import SessionMcp

    picture = {"frame_id": "f1.front", "camera": "front", "width": 2, "height": 2,
               "png_b64": "iVBORy1ieXRlcw=="}
    blocked = {"results": [{"ok": False, "error": "blocked", "text": "blocked",
                            "images": [picture]}], "sim_time": 9.0, "replies": 1, "limits": {}}
    for keep, flagged in ((False, True), (True, False)):
        bridge = SessionMcp.__new__(SessionMcp)
        bridge.keep_pictures = keep
        bridge._ask = lambda req: json.loads(json.dumps(blocked))
        out = bridge.call("robot_move_to", {})
        assert out["isError"] is flagged
        assert any(c["type"] == "image" for c in out["content"])
        assert any('"error": "blocked"' in c.get("text", "") for c in out["content"])
    bridge._ask = lambda req: {"results": [{"ok": False, "error": "invalid_args"}], "limits": {}}
    assert bridge.call("robot_move_to", {})["isError"] is True     # no picture: still an error
    argv, _ = claude_code.claude_command("claude-opus-5-5", "go", Path("/tmp/s"))
    config = json.loads(argv[argv.index("--mcp-config") + 1])
    assert "--keep-pictures" in config["mcpServers"]["ruibench"]["args"]
    assert "--keep-pictures" not in _mcp_server(Path("/tmp/s"))["args"]   # Codex sees them anyway
