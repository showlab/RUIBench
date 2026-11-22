"""The QwenCloud channel: its key, the streamed answer put back together, the
thinking put back into the history, every request kept, a full context told."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ruikit.core.result import ToolError
from ruikit.models.chat_completions import ChatCompletions
from ruikit.models.conversation import _full_context
from ruikit.models.gateway import Gateway

PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 32


class FakeStream:
    """Answers each POST with the next scripted list of chunks, as server-sent
    events; or with ``(status, text)``.  Keeps every request and its headers."""

    def __init__(self, script):
        self.script = list(script)
        self.got = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.got.append((dict(self.headers), body))
                reply = outer.script.pop(0)
                if isinstance(reply, tuple):
                    status, text = reply
                    data = text.encode()
                    self.send_response(status)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("x-request-id", f"req-{len(outer.got)}")
                self.end_headers()
                for chunk in reply:
                    self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.write(b"data: [DONE]\n\n")

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def chunks(thinking, call_id, name, args):
    """An answer thinking, then calling one tool, its parts spread over chunks."""
    half = len(thinking) // 2
    return [
        {"id": "x1", "model": "qwen3.8-max", "choices": [{"delta": {"reasoning_content": thinking[:half]}}]},
        {"choices": [{"delta": {"reasoning_content": thinking[half:]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": call_id, "type": "function",
                                                "function": {"name": name, "arguments": ""}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(args)[:5]}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": json.dumps(args)[5:]}}]},
                      "finish_reason": "tool_calls"}]},
        {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 9,
                                  "completion_tokens_details": {"reasoning_tokens": 7}}},
    ]


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    # the keys in a secrets file of the test's own: the settings take a variable of the
    # environment only for a key the secrets file has, and this machine's may have none
    from ruikit.models import settings

    secrets = tmp_path / "models.env"
    secrets.write_text("QWENCLOUD_API_KEY=sk-ws-test\nQWENTOKENPLAN_API_KEY=sk-sp-test\n")
    monkeypatch.setattr(settings, "SECRETS", secrets)
    monkeypatch.setenv("QWENCLOUD_API_KEY", "sk-ws-test")
    monkeypatch.setenv("RUIKIT_QWENCLOUD_SAMPLING", json.dumps({"reasoning_effort": "medium",
                                                                "temperature": 1.0}))
    monkeypatch.setenv("RUIKIT_PASSBACK_THINKING", "1")
    servers = []

    def make(script):
        fake = FakeStream(script)
        servers.append(fake)
        gw = Gateway("qwencloud:qwen3.8-max", base_url=fake.url)
        return fake, gw

    yield make
    for fake in servers:
        fake.close()


TOOLS = [{"name": "robot_move", "description": "move", "schema": {"type": "object"}}]


def test_the_streamed_answer_comes_back_whole_and_its_thinking_goes_back(cloud, tmp_path):
    fake, gw = cloud([chunks("first, look at the cup", "call_a", "robot_move", {"x": 0.1}),
                      chunks("then lift it", "call_b", "robot_move", {"z": 0.2})])
    gw.requests_log = tmp_path / "ep.requests.jsonl"
    conv = ChatCompletions(gw, "be a robot", TOOLS, max_output_tokens=32768)
    conv.user("go", [PNG])
    reply = conv.ask()
    assert [c.name for c in reply.calls] == ["robot_move"]
    assert reply.calls[0].arguments == {"x": 0.1}
    assert reply.reasoning == "first, look at the cup"
    conv.results(reply.calls, ["moved"])
    conv.ask()

    (headers, first), (_, second) = fake.got
    assert headers["Authorization"] == "Bearer sk-ws-test"
    assert first["stream"] is True and first["stream_options"] == {"include_usage": True}
    assert first["enable_thinking"] is True and first["reasoning_effort"] == "medium"
    assert first["preserve_thinking"] is True and first["max_tokens"] == 32768
    assistant = [m for m in second["messages"] if m["role"] == "assistant"]
    assert assistant[0]["reasoning_content"] == "first, look at the cup"
    assert "first, look" not in assistant[0]["content"]

    rows = [json.loads(line) for line in gw.requests_log.read_text().splitlines()]
    assert len(rows) == 2
    assert rows[0]["headers"]["x-request-id"] == "req-1"
    assert rows[0]["request"]["tools"] and "tools" not in rows[1]["request"]
    assert rows[0]["request"]["tools_sha"] == rows[1]["request"]["tools_sha"]
    image = [p for m in rows[0]["request"]["messages"] if isinstance(m["content"], list)
             for p in m["content"] if p["type"] == "image_url"][0]
    assert image["image_url"]["url"].startswith("inputs/")
    assert (tmp_path / image["image_url"]["url"]).read_bytes() == PNG
    assert rows[1]["response"]["choices"][0]["message"]["reasoning_content"] == "then lift it"
    assert rows[1]["response"]["usage"]["completion_tokens_details"]["reasoning_tokens"] == 7
    assert "sk-ws-test" not in gw.requests_log.read_text()


def test_without_passback_the_history_carries_no_thinking(cloud, monkeypatch):
    monkeypatch.setenv("RUIKIT_PASSBACK_THINKING", "0")
    fake, gw = cloud([chunks("plan A", "call_a", "robot_move", {}),
                      chunks("plan B", "call_b", "robot_move", {})])
    conv = ChatCompletions(gw, "be a robot", TOOLS)
    conv.user("go")
    reply = conv.ask()
    conv.results(reply.calls, ["moved"])
    conv.ask()
    second = fake.got[1][1]
    assert second["preserve_thinking"] is False
    assert all("reasoning_content" not in m for m in second["messages"])


def test_a_full_context_is_told_in_the_words_the_window_knows(cloud):
    fake, gw = cloud([(400, '{"error": {"message": "Range of input length should be [1, 991808]"}}')])
    with pytest.raises(ToolError) as exc:
        gw.post("/chat/completions", {"model": gw.model, "messages": []})
    assert _full_context(exc.value)


def test_no_key_no_channel(monkeypatch, tmp_path):
    monkeypatch.setenv("RUIKIT_SECRETS", str(tmp_path / "none.env"))
    monkeypatch.delenv("QWENCLOUD_API_KEY", raising=False)
    import ruikit.models.settings as settings
    monkeypatch.setattr(settings, "SECRETS", tmp_path / "none.env")
    with pytest.raises(ToolError):
        Gateway("qwencloud:qwen3.8-max", base_url="http://127.0.0.1:9/v1")


def test_an_answer_the_content_filter_stopped_is_asked_again(cloud, tmp_path):
    flagged = [{"error": {"code": "data_inspection_failed",
                          "message": "Output data may contain inappropriate content."}}]
    fake, gw = cloud([flagged, chunks("fine", "call_a", "robot_move", {})])
    gw.requests_log = tmp_path / "ep.requests.jsonl"
    body = gw.post("/chat/completions", {"model": gw.model, "messages": [{"role": "user", "content": "go"}]})
    assert body["choices"][0]["message"]["tool_calls"][0]["id"] == "call_a"
    rows = [json.loads(line) for line in gw.requests_log.read_text().splitlines()]
    assert rows[0]["asked_again"] == 1 and "inappropriate" in rows[0]["filtered"]
    assert len(fake.got) == 2


def test_a_filtered_input_is_not_asked_again(cloud):
    fake, gw = cloud([(400, '{"error": {"message": "Input data may contain inappropriate content."}}')])
    with pytest.raises(ToolError):
        gw.post("/chat/completions", {"model": gw.model, "messages": []})
    assert len(fake.got) == 1


TIMED_OUT = (400, 'data: {"error": {"code": "invalid_parameter_error", "param": null, '
                  '"message": "Download multimodal file timed out", "type": "invalid_request_error"}}')


def test_a_picture_fetch_that_timed_out_is_sent_again(cloud):
    """QwenCloud's own fetch of the pictures timed out: a 400, though the request is
    sound -- sent again, as to a busy server, and the episode goes on."""
    fake, gw = cloud([TIMED_OUT, chunks("fine", "call_a", "robot_move", {})])
    gw.backoff_s = 0.01
    body = gw.post("/chat/completions", {"model": gw.model, "messages": [{"role": "user", "content": "go"}]})
    assert body["choices"][0]["message"]["tool_calls"][0]["id"] == "call_a"
    assert len(fake.got) == 2 and fake.got[0][1] == fake.got[1][1]


def test_a_picture_fetch_that_keeps_timing_out_ends_after_the_retries(cloud):
    fake, gw = cloud([TIMED_OUT] * 3)
    gw.retries, gw.backoff_s = 2, 0.01
    with pytest.raises(ToolError) as exc:
        gw.post("/chat/completions", {"model": gw.model, "messages": []})
    assert "after 3 tries" in str(exc.value) and len(fake.got) == 3


# -- the local vLLM: the same passback, in the field its chat template reads -------------
def test_a_local_model_gets_its_thinking_back_as_reasoning(monkeypatch):
    from tests.ruikit.fake_model import FakeGateway

    monkeypatch.setenv("RUIKIT_LOCAL_THINKING", "1")
    monkeypatch.setenv("RUIKIT_PASSBACK_THINKING", "1")
    answer = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "", "reasoning": "grasp the cup first",
        "tool_calls": [{"id": "c0", "type": "function",
                        "function": {"name": "robot_move", "arguments": "{}"}}]}}]}
    fake = FakeGateway([answer, answer])
    try:
        gw = Gateway("local:qwen3.8-27b", base_url=fake.url)
        conv = ChatCompletions(gw, "be a robot", TOOLS)
        conv.user("go")
        reply = conv.ask()
        conv.results(reply.calls, ["moved"])
        conv.ask()
    finally:
        fake.close()
    second = fake.got[1][1]
    assistant = [m for m in second["messages"] if m["role"] == "assistant"]
    assert assistant[0]["reasoning"] == "grasp the cup first"
    assert "reasoning_content" not in assistant[0]
    assert second["chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True}


def test_without_the_switch_a_local_model_gets_none(monkeypatch):
    monkeypatch.setenv("RUIKIT_PASSBACK_THINKING", "0")
    gw = Gateway("local:qwen3.8-27b", base_url="http://127.0.0.1:9/v1")
    assert gw.passback is False


def test_once_the_window_is_on_only_the_latest_answers_keep_their_thinking(monkeypatch):
    from ruikit.models.conversation import FULL_CONTEXT_NOTE

    monkeypatch.setenv("RUIKIT_PASSBACK_THINKING", "1")
    monkeypatch.setenv("RUIKIT_PASSBACK_WINDOW", "2")
    gw = Gateway("local:qwen3.8-27b", base_url="http://127.0.0.1:9/v1")
    history = []
    for k in range(5):
        gw._thoughts[f"c{k}"] = f"thought {k}"
        history.append({"role": "assistant", "content": "",
                        "tool_calls": [{"id": f"c{k}", "type": "function",
                                        "function": {"name": "m", "arguments": "{}"}}]})
        history.append({"role": "tool", "tool_call_id": f"c{k}", "content": "ok"})
    whole = gw._with_thoughts({"messages": history})
    assert [m.get("reasoning") for m in whole["messages"] if m["role"] == "assistant"] == \
        [f"thought {k}" for k in range(5)]
    history.insert(6, {"role": "user", "content": FULL_CONTEXT_NOTE.format(keep=8)})
    windowed = gw._with_thoughts({"messages": history})
    assert [m.get("reasoning") for m in windowed["messages"] if m["role"] == "assistant"] == \
        [None, None, None, "thought 3", "thought 4"]


# -- the Token Plan first, the pay-as-you-go endpoint when it refuses ----------------------
@pytest.fixture
def plan(cloud, monkeypatch, tmp_path):
    monkeypatch.setenv("RUIKIT_QWENCLOUD_PLAN", "1")
    monkeypatch.setenv("QWENTOKENPLAN_API_KEY", "sk-sp-test")
    monkeypatch.setenv("RUIKIT_QWENCLOUD_PLAN_STATE", str(tmp_path / "plan.json"))

    def make(plan_script, payg_script):
        planned = FakeStream(plan_script)
        monkeypatch.setenv("RUIKIT_QWENTOKENPLAN_URL", planned.url)
        payg, gw = cloud(payg_script)
        gw.requests_log = tmp_path / "ep.requests.jsonl"
        return planned, payg, gw

    return make


def ask(gw):
    return gw.post("/chat/completions", {"model": gw.model, "messages": [{"role": "user", "content": "go"}]})


def test_the_token_plan_answers_first(plan):
    planned, payg, gw = plan([chunks("t", "c1", "m", {})], [])
    ask(gw)
    assert planned.got[0][0]["Authorization"] == "Bearer sk-sp-test" and not payg.got
    assert gw.endpoints == {"tokenplan": 1, "payg": 0}
    row = json.loads(gw.requests_log.read_text().splitlines()[0])
    assert row["endpoint"] == "tokenplan"
    planned.close()


def test_a_spent_plan_hands_over_to_pay_as_you_go_for_every_worker(plan, tmp_path):
    from ruikit.models.gateway import Gateway

    planned, payg, gw = plan([(429, '{"error": {"message": "Allocated quota exceeded"}}')],
                             [chunks("t", "c1", "m", {}), chunks("t", "c2", "m", {})])
    ask(gw)
    assert len(planned.got) == 1 and payg.got[0][0]["Authorization"] == "Bearer sk-ws-test"
    assert gw.endpoints == {"tokenplan": 0, "payg": 1}
    state = json.loads((tmp_path / "plan.json").read_text())
    assert "quota" in state["why"] and state["rests_until"] > 0
    other = Gateway("qwencloud:qwen3.8-max", base_url=payg.url)     # another worker
    ask(other)
    assert len(planned.got) == 1 and len(payg.got) == 2
    planned.close()


def test_a_too_long_request_is_not_the_plans_fault(plan):
    planned, payg, gw = plan([(400, '{"error": {"message": "Range of input length should be [1, 983616]"}}')], [])
    with pytest.raises(ToolError) as exc:
        ask(gw)
    assert _full_context(exc.value) and not payg.got
    planned.close()


def test_how_long_the_plan_rests():
    from ruikit.models.gateway import PLAN_KEY_REST_S, PLAN_RATE_REST_S, PLAN_REST_S, _plan_rest

    assert _plan_rest(429, "insufficient_quota: Your token-plan quota has been exhausted.") == PLAN_KEY_REST_S
    assert _plan_rest(429, "Requests rate limit exceeded") == PLAN_RATE_REST_S
    assert _plan_rest(403, "") == PLAN_KEY_REST_S
    assert _plan_rest(503, "busy") == PLAN_REST_S


MOVE = [{"name": "robot_move_to", "description": "move", "schema": {
    "type": "object", "properties": {
        "xyz_m": {"type": "array", "items": {"type": "number"}},
        "rpy_deg": {"type": ["array", "null"], "items": {"type": "number"}},
        "approach_m": {"type": ["number", "null"]},
        "note": {"type": ["string", "null"]}}}}]


def test_a_list_or_number_that_came_as_text_is_read_by_the_schema(cloud):
    """QwenCloud hands on an optional list or number (a union type) as the text
    the model wrote: it is read as the JSON it spells, as an open Qwen server
    reads it; a parameter that takes text keeps it."""
    fake, gw = cloud([chunks("reach", "call_a", "robot_move_to",
                             {"xyz_m": [0.5, 0, 0.2], "rpy_deg": "[180, 0, 0]",
                              "approach_m": "0.05", "note": "[1]"})])
    conv = ChatCompletions(gw, "be a robot", MOVE)
    conv.user("go", [PNG])
    reply = conv.ask()
    assert reply.calls[0].arguments == {"xyz_m": [0.5, 0, 0.2], "rpy_deg": [180, 0, 0],
                                        "approach_m": 0.05, "note": "[1]"}
    kept = [m for m in conv.history if m["role"] == "assistant"][0]
    assert json.loads(kept["tool_calls"][0]["function"]["arguments"])["rpy_deg"] == [180, 0, 0]


def test_text_that_spells_no_value_the_parameter_takes_stays_text():
    from ruikit.models.chat_completions import typed_arguments

    schema = MOVE[0]["schema"]
    assert typed_arguments({"rpy_deg": "up", "approach_m": "[1]", "other": "2"}, schema) == \
        {"rpy_deg": "up", "approach_m": "[1]", "other": "2"}
    assert typed_arguments({"approach_m": "true"}, schema) == {"approach_m": "true"}
    assert typed_arguments({"rpy_deg": "null"}, schema) == {"rpy_deg": None}


def test_the_tools_take_what_is_read():
    """The tools' own schemas: what is read passes their validation."""
    from ruikit.core.params import parse_args
    from ruikit.core.tool import get_tool, load_builtin_tools
    from ruikit.models.chat_completions import typed_arguments

    load_builtin_tools()
    for name, raw in (("robot_move_to", {"xyz_m": [0.5, 0, 0.2], "rpy_deg": "[180, 0, 0]",
                                         "approach_m": "0.05"}),
                      ("task_demo", {"frames": "[0, 3]"}),
                      ("robot_gripper", {"width_m": "0.04"}),
                      ("camera_look", {"cameras": '["front"]'})):
        spec = get_tool(name).spec
        parse_args(spec.params, typed_arguments(raw, spec.input_schema()))
