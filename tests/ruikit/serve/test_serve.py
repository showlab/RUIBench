"""The three doors: MCP, the SDK, and the reference solvability on WK1."""

import io
import json

import pytest

from ruikit import Episode
from ruikit.diagnostics.reference import EXPRESSORS, style_of
from ruikit.interfaces import list_profiles
from ruikit.interfaces.catalog import form_of
from ruikit.serve.mcp_stdio import McpServer
from ruikit.serve.sdk import Client


def _rpc(server, method, params=None, mid=1):
    return server.handle({"jsonrpc": "2.0", "id": mid, "method": method, "params": params or {}})


def test_mcp_lists_and_calls_tools():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        srv = McpServer(ep, out=io.StringIO())
        init = _rpc(srv, "initialize")["result"]
        assert init["capabilities"]["tools"]["listChanged"] is True
        assert "Tower of Hanoi" in init["instructions"]
        names = [t["name"] for t in _rpc(srv, "tools/list")["result"]["tools"]]
        assert names == list(ep.profile.tools)
        res = _rpc(srv, "tools/call", {"name": "camera_look", "arguments": {"cameras": ["front"]}})
        body = res["result"]
        assert body["isError"] is False
        assert "structuredContent" not in body          # clients drop images next to it
        payload = json.loads(body["content"][-1]["text"])
        assert payload["data"]["frames"][0]["frame_id"].endswith(".front")
        bad = _rpc(srv, "tools/call", {"name": "robot_step", "arguments": {"move": "up"}})
        assert bad["result"]["isError"] is True
        assert _rpc(srv, "no/such")["error"]["code"] == -32601
    finally:
        ep.close()


def test_mcp_announces_custom_tools():
    ep = Episode("WK1", level="easy", seed=0, profile="open/v1")
    try:
        out = io.StringIO()
        srv = McpServer(ep, out=out)
        msg = {"jsonrpc": "2.0", "id": 1, "method": "tools/call",
               "params": {"name": "toolkit_define", "arguments": {
                   "name": "my_noop", "description": "Does nothing at all, for testing.",
                   "input_schema": json.dumps({"type": "object", "properties": {}}),
                   "code": "def run():\n    return 1"}}}
        srv.serve(io.StringIO(json.dumps(msg) + "\n"))
        lines = [json.loads(l) for l in out.getvalue().splitlines()]
        assert any(l.get("method") == "notifications/tools/list_changed" for l in lines)
        assert "my_noop" in [t["name"] for t in srv.tool_list()]
    finally:
        ep.close()


def test_the_sdk_mirrors_the_profile():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.pose/v1")
    try:
        robot = Client(ep)
        assert hasattr(robot.robot, "move_to") and not hasattr(robot.robot, "step")
        r = robot.robot.move_to(xyz_m=[0.4, 0.0, 0.2])
        assert not r.is_error
    finally:
        ep.close()


@pytest.mark.parametrize("profile", [i for i in list_profiles()
                                     if form_of(i)[0] in ("per_action", "program")])
def test_every_base_profile_can_express_wk1(profile):
    """Layer 2 of solvability on the easiest task: the reference decides, the
    profile's tools act -- and the task is done."""
    from ruikit import Budget

    ep = Episode("WK1", level="easy", seed=0, profile=profile, budget=Budget(max_tool_calls=4000))
    try:
        out = EXPRESSORS[style_of(profile)]().run(ep)
        assert out.success, (profile, out.reason)
    finally:
        ep.close()
