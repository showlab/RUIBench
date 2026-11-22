"""Tools from outside the library: a plugin file and a profile YAML anywhere,
through ``load_profile``, the session server and the batch runner."""

import json
import uuid

import pytest
import yaml

from ruikit.core import plugins
from ruikit.core.tool import REGISTRY
from ruikit.interfaces import load_profile


@pytest.fixture(autouse=True)
def _registry_as_it_was():
    """The registry is per process: a plugin's tools leave with its test, so the
    library's own checks never see them."""
    before = set(REGISTRY)
    yield
    for name in set(REGISTRY) - before:
        del REGISTRY[name]


PLUGIN = '''
from dataclasses import dataclass

from ruikit.core.result import ToolResult
from ruikit.core.params import param
from ruikit.core.tool import Cost, tool


@dataclass(frozen=True)
class EchoParams:
    text: str = param("what to say back")


@tool(name="{name}", family="compute", params=EchoParams, source="agent",
      cost=Cost("declared", 0.0), examples=[{{"text": "hello"}}],
      description="""Say the text back, unchanged. A plugin tool for the tests: it does not
      touch the robot, reads no camera and costs no simulated time.""")
def echo(ctx, p: EchoParams) -> ToolResult:
    return ToolResult(data={{"echo": p.text}}, text=p.text)
'''


def _plugin(tmp_path):
    """A plugin file with one tool, and a profile outside the library that adds
    it to I2's tools.  Names are new each time: the registry is per process."""
    uid = uuid.uuid4().hex[:8]
    name = f"echo_{uid}"
    path = tmp_path / f"plug_{uid}.py"
    path.write_text(PLUGIN.format(name=name))
    prof = tmp_path / f"iface_{uid}.yaml"
    prof.write_text(yaml.safe_dump({
        "id": f"test.plugin{uid}/v1", "extends": "rui.pose/v1", "title": "I2 and an echo",
        "tools": list(load_profile("rui.pose/v1").tools) + [name]}))
    return name, path, prof


def test_a_plugin_file_adds_a_tool_a_profile_outside_the_library_can_list(tmp_path):
    name, path, prof = _plugin(tmp_path)
    with pytest.raises(ValueError):
        load_profile(str(prof))                       # its tool is not there yet
    assert plugins.load([str(path)]) == [path.stem] and name in REGISTRY
    assert plugins.load([str(path)]) == [path.stem]   # twice: nothing more
    p = load_profile(str(prof))
    assert name in p.tools and p.channel == "tools" and p.id.startswith("test.plugin")
    with pytest.raises(FileNotFoundError):
        plugins.load([str(tmp_path / "missing.py")])


def test_the_session_server_takes_plugins(tmp_path, capsys):
    from ruikit.serve import session

    name, path, prof = _plugin(tmp_path)
    d = str(tmp_path / "s")
    session.main(["start", "--dir", d, "--task", "DS1", "--profile", str(prof), "--engine", "mock",
                  "--plugins", str(path)])
    assert "ready after" in capsys.readouterr().out
    session.main(["call", "--dir", d, name, json.dumps({"text": "hi from a plugin"})])
    out = capsys.readouterr().out
    assert "[1] ok" in out and "hi from a plugin" in out
    session.main(["call", "--dir", d, "task_done", json.dumps({"report": "done"})])
    assert "THE EPISODE HAS ENDED" in capsys.readouterr().out


def test_the_batch_runner_takes_plugins(tmp_path):
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ruikit.experiments import batch

    name, path, prof = _plugin(tmp_path)
    reply = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "type": "function",
             "function": {"name": name, "arguments": json.dumps({"text": "from the batch"})}},
            {"id": "c1", "type": "function",
             "function": {"name": "task_done", "arguments": json.dumps({"report": "done"})}}]}}]}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            data = json.dumps(reply).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_address[1]}/v1"
    said = []
    try:
        # without the plugin, the runner says so before starting anything
        code = batch.run(["opus5"], ["WK1"], ["easy"], "0", [str(prof)], str(tmp_path / "a"),
                         engine="mock", base_url=url, say=said.append)
        assert code == 2 and "cannot use the profiles" in said[-1]
        code = batch.run(["opus5"], ["WK1"], ["easy"], "0", [str(prof)], str(tmp_path / "b"),
                         engine="mock", base_url=url, plugins=[str(path)], say=said.append)
    finally:
        server.shutdown()
    assert code == 0
    folder = next((tmp_path / "b" / "loop-opus5").iterdir())
    assert folder.name == f"WK1_easy_s0_{load_profile(str(prof)).id.replace('/', '-')}"
    rows = [json.loads(l) for l in next(f for f in folder.glob("*.jsonl")
                                         if not f.name.endswith((".episode.jsonl", ".model.jsonl")))
            .read_text().splitlines()]
    calls = [c for r in rows if r.get("tool") == "batch" for c in r["args"]["calls"]]
    assert any(c["tool"] == name for c in calls)
    assert json.loads((folder / "result.json").read_text())["reason"] == "agent_done"
