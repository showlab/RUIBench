"""A real model's loop, against a stand-in for a model channel."""

import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ruikit.harness.minimal_loop import NUDGES, run_model
from ruikit.models.base import GatewayDown
from ruikit.models.conversation import conversation
from ruikit.models.gateway import Gateway

from fake_model import FakeGateway, chat_calls, chat_text


def _rows(folder):
    path = next(p for p in folder.glob("*.jsonl")
                if not p.name.endswith((".episode.jsonl", ".model.jsonl")))
    return [r for r in map(json.loads, path.read_text().splitlines()) if "tool" in r]


def test_a_model_drives_the_episode_through_chat_completions(tmp_path):
    fake = FakeGateway([chat_calls(("task_info", {}), ("camera_look", {})),
                        chat_calls(("task_done", {"report": "all set"}))])
    try:
        row = run_model("DS1", "easy", 0, "rui.pose/v1", "claude-opus-5", log_dir=str(tmp_path),
                        base_url=fake.url)
    finally:
        fake.close()
    (path0, first), (path1, second) = fake.got
    assert path0 == path1 == "/v1/chat/completions" and first["model"] == "claude-opus-5"
    assert first["messages"][0]["role"] == "system" and "DS1" in first["messages"][0]["content"]
    assert {t["function"]["name"] for t in first["tools"]} >= {"robot_move_to", "task_done"}
    # each call's result goes back under its own id
    answered = {m["tool_call_id"]: json.loads(m["content"]) for m in second["messages"]
                if m["role"] == "tool"}
    assert set(answered) == {"c0", "c1"} and answered["c1"]["ok"]
    # its limits are the task's, as in a batch and for the ready-made agents (v1.15)
    log = next(p for p in tmp_path.glob("*.jsonl")
               if not p.name.endswith((".episode.jsonl", ".model.jsonl")))
    budget = json.loads(log.read_text().splitlines()[0])["header"]["budget"]
    assert budget["max_wall_seconds"] == 86400.0 and budget["max_tool_calls"] == 6000
    # the two calls of one reply were one reply of the episode
    rows = _rows(tmp_path)
    assert rows[0]["tool"] == "batch" and [c["tool"] for c in rows[0]["args"]["calls"]] == \
        ["task_info", "camera_look"]
    assert row["reason"] == "agent_done" and row["replies"] == 2
    assert row["usage"] == {"input_tokens": 200, "output_tokens": 20}
    model_rows = [json.loads(l) for l in
                  (tmp_path / "DS1_easy_s0_rui.pose-v1.model.jsonl").read_text().splitlines()][1:]
    assert [(r["waited_s"], r["retries"]) for r in model_rows] == [(0.0, 0), (0.0, 0)]


def test_a_rate_limit_wait_is_counted_apart_from_the_answer():
    # a stand-in that turns the first try away (429), then answers
    tries = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            tries.append(1)
            data = json.dumps(chat_text("OK")).encode() if len(tries) > 1 else b"{}"
            self.send_response(200 if len(tries) > 1 else 429)
            if len(tries) == 1:
                self.send_header("Retry-After", "0.2")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        conv = conversation(Gateway("opus5", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1"),
                            system="s", tools=[])
        conv.user("hi")
        reply = conv.ask()
    finally:
        server.shutdown()
    assert reply.text == "OK" and reply.retries == 1 and len(tries) == 2
    assert 0.2 <= reply.waited_s <= reply.latency_s


def test_gpt_models_go_through_responses(tmp_path):
    fake = FakeGateway([
        {"output": [{"type": "function_call", "call_id": "r1", "name": "camera_look",
                     "arguments": "{}"}], "status": "completed"},
        {"output": [{"type": "function_call", "call_id": "r2", "name": "task_done",
                     "arguments": json.dumps({"report": "ok"})}], "status": "completed"}])
    try:
        run_model("DS1", "easy", 0, "rui.code/v1", "gpt-6-astra", base_url=fake.url)
    finally:
        fake.close()
    (path0, first), (_, second) = fake.got
    assert path0 == "/v1/responses" and first["model"] == "gpt-6-astra"
    assert "DS1" in first["instructions"] and first["tools"][0]["type"] == "function"
    outs = [i for i in second["input"] if i.get("type") == "function_call_output"]
    assert outs and outs[0]["call_id"] == "r1"


def test_a_model_that_stops_calling_tools_is_nudged_then_stopped():
    fake = FakeGateway([chat_text("I think I am done.")] * (NUDGES + 1))
    try:
        row = run_model("DS1", "easy", 0, "rui.pose/v1", "flash", base_url=fake.url)
    finally:
        fake.close()
    assert len(fake.got) == NUDGES + 1 and row["reason"] == "agent_stopped"


def test_a_reply_without_a_call_is_on_the_clock(tmp_path):
    # 8 s a reply: one that only writes costs it too, as one that calls a tool does
    fake = FakeGateway([chat_text("Let me think about the rings first."),
                        chat_calls(("task_info", {})),
                        chat_calls(("task_done", {"report": "done"}))])
    try:
        run_model("DS1", "easy", 0, "rui.pose/v1", "flash", think_seconds=8.0,
                  log_dir=str(tmp_path), base_url=fake.url)
    finally:
        fake.close()
    rows = [json.loads(l) for l in
            (tmp_path / "DS1_easy_s0_rui.pose-v1.model.jsonl").read_text().splitlines()][1:]
    assert [r["t"] for r in rows] == pytest.approx([0.0, 8.0, 16.0], abs=0.2)


def test_a_down_gateway_says_how_to_bring_it_up():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(GatewayDown) as err:
        Gateway("opus5", base_url=f"http://127.0.0.1:{port}/v1").ask("hi")
    assert "OPENAI_BASE_URL" in err.value.hint and "OpenAI" in str(err.value)


def _images(request):
    return [p for m in request["messages"] if isinstance(m["content"], list)
            for p in m["content"] if p["type"] == "image_url"]


def test_only_the_latest_pictures_are_kept():
    fake = FakeGateway([])
    try:
        conv = conversation(Gateway("opus5", base_url=fake.url), system="s", tools=[],
                            keep_images=2)
        for k in range(3):
            conv.user(f"look {k}", [b"\x89PNG fake"], [f"[f{k}.front] front 64x64"])
            conv.ask()
    finally:
        fake.close()
    sent = fake.got[-1][1]
    assert len(_images(sent)) == 2 and all("_label" not in p for m in sent["messages"]
                                           if isinstance(m["content"], list)
                                           for p in m["content"])
    assert "[f0.front] front 64x64: dropped" in json.dumps(sent)


def test_a_reply_s_pictures_are_each_seen_once():
    """A reply that brings more pictures than the history keeps (a demonstration's
    sample of up to nine) has every one sent; only once the model has seen them
    are the older ones dropped (v2.4.2)."""
    fake = FakeGateway([])
    try:
        conv = conversation(Gateway("opus5", base_url=fake.url), system="s", tools=[],
                            keep_images=2)
        conv.user("go")
        conv.ask()
        conv.user("The pictures those calls returned:", [b"\x89PNG fake"] * 3,
                  [f"[demo0.{k}.front] front 64x64" for k in range(3)])
        conv.ask()
        conv.user("go on")
        conv.ask()
    finally:
        fake.close()
    whole, after = fake.got[1][1], fake.got[2][1]
    assert len(_images(whole)) == 3 and "dropped" not in json.dumps(whole)
    assert len(_images(after)) == 2
    assert "[demo0.0.front] front 64x64: dropped" in json.dumps(after)


PROGRAM = "\n".join(f"print({k})" for k in range(30))
CONVENTIONS = {"ok": True, "text": "Instruction (v0): hand Tom his present.",
               "data": {"conventions": {"frame": "station: x forward, y left, z up; metres"}}}
RAN = {"ok": True, "text": "program finished after 3 tool call(s)",
       "data": {"stdout": "x=0.41 y=-0.12\n" * 40, "tool_calls": 3}, "sim_time": 40.0}


def _a_conversation(keep_calls, calls, kind=None):
    """A chat-completions conversation (or ``kind``'s) holding ``calls``:
    ``(name, arguments, result)``, one reply each, as ``_ask`` and ``results`` keep them."""
    from ruikit.models.base import Call

    gw = Gateway("opus5", base_url="http://127.0.0.1:9/v1")
    conv = (kind(gw, "s", [], keep_calls=keep_calls) if kind else
            conversation(gw, system="s", tools=[], keep_calls=keep_calls))
    for k, (name, args, out) in enumerate(calls):
        call = Call(f"c{k}", name, args, "")
        if kind is None:
            conv.history.append({"role": "assistant", "content": f"reply {k}", "tool_calls": [
                {"id": call.id, "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]})
        else:
            conv.history.append({"type": "function_call", "call_id": call.id, "name": name,
                                 "arguments": json.dumps(args)})
        conv.results([call], [json.dumps(out)])
    return conv


def _sent_calls(conv):
    """``(name, arguments, result)`` of each call as the next request sends it."""
    sent = conv.outgoing()
    calls = [(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
             for m in sent for tc in m.get("tool_calls") or ()]
    calls += [(m["name"], json.loads(m["arguments"])) for m in sent
              if m.get("type") == "function_call"]
    outs = [json.loads(m.get("content", m.get("output"))) for m in sent
            if m.get("role") == "tool" or m.get("type") == "function_call_output"]
    return [(n, a, o) for (n, a), o in zip(calls, outs)]


def test_only_the_latest_calls_stay_whole():
    note = {"key": "bins", "text": "red bin at 0.41 -0.12; blue bin at 0.38 0.15. " * 8}
    conv = _a_conversation(2, [("task_info", {}, CONVENTIONS),
                               ("code_run", {"source": PROGRAM}, RAN),
                               ("memory_note", note, {"ok": True, "text": "noted as 'bins'"}),
                               ("code_run", {"source": PROGRAM}, RAN),
                               ("code_run", {"source": PROGRAM}, RAN)])
    conv._prune_calls()
    (n0, a0, o0), (n1, a1, o1), (n2, a2, o2), last, latest = _sent_calls(conv)
    # the first task_info stays whole: the station's conventions
    assert o0 == CONVENTIONS
    # an older call: its result's one line, its program cut to a line
    assert set(o1) == {"ok", "text", "cut"} and o1["text"] == RAN["text"]
    assert a1 == {"source": "[30 lines, sent earlier: dropped from the history to save space]"}
    # a long note too (memory_recall reads it back); its key, and a result already as
    # short as its line, stay as they were
    assert a2 == {"key": "bins", "text": f"[{len(note['text'])} characters, sent earlier: "
                                         "dropped from the history to save space]"}
    assert o2 == {"ok": True, "text": "noted as 'bins'"}
    # the latest two calls, whole
    assert last == latest == ("code_run", {"source": PROGRAM}, RAN)
    # each result still answers its call; the model's words stay
    sent = conv.outgoing()
    assert [m["tool_call_id"] for m in sent if m["role"] == "tool"] == ["c0", "c1", "c2", "c3", "c4"]
    assert [m["content"] for m in sent if m["role"] == "assistant"] == \
        [f"reply {k}" for k in range(5)]


def test_a_call_once_cut_stays_as_cut():
    # the history before the latest calls does not change from one request to the next (an
    # endpoint's prefix cache reads it again) but where a call leaves the window
    conv = _a_conversation(2, [("code_run", {"source": PROGRAM}, RAN)] * 4)
    conv._prune_calls()
    before = json.dumps(conv.outgoing())
    conv._prune_calls()
    assert json.dumps(conv.outgoing()) == before
    more = _a_conversation(2, [("code_run", {"source": PROGRAM}, RAN)] * 5)
    more._prune_calls()
    # the system message and the two calls cut already: as they were
    assert more.outgoing()[:1 + 2 * 2] == json.loads(before)[:1 + 2 * 2]


def test_every_word_stays_without_a_window():
    conv = _a_conversation(None, [("code_run", {"source": PROGRAM}, RAN)] * 12)
    before = json.dumps(conv.history)
    conv._prune_calls()
    assert json.dumps(conv.history) == before


def test_the_window_holds_on_the_responses_endpoint():
    from ruikit.models.responses import Responses

    conv = _a_conversation(1, [("code_run", {"source": PROGRAM}, RAN)] * 3, kind=Responses)
    conv._prune_calls()
    (_, a0, o0), (_, a1, o1), (_, a2, o2) = _sent_calls(conv)
    assert "cut" in o0 and "cut" in o1 and o2 == RAN
    assert a0["source"].startswith("[30 lines") and a2 == {"source": PROGRAM}


FULL = ("Qwen/Qwen3.8-27B-FP8: HTTP 400: {\"error\":{\"message\":\"This model's maximum context "
        "length is 262144 tokens. However, you requested 4096 output tokens and your prompt "
        "contains at least 258049 input tokens\",\"code\":400}}")


class _ContextOf:
    """A stand-in endpoint with a context window of ``limit`` characters of messages:
    it refuses a longer request as vLLM does, and answers the rest from ``script``."""

    model, openai_dialect, provider, thinking, responses = "qwen", False, "local", False, False

    def __init__(self, limit, script, refusal=FULL):
        self.limit, self.script, self.refusal = limit, list(script), refusal
        self.sent, self.refused = [], 0

    def post(self, path, payload, tries=None):
        from ruikit.core.result import ToolError

        if len(json.dumps(payload["messages"])) > self.limit:
            self.refused += 1
            raise ToolError("tool_failed", self.refusal)
        self.sent.append(payload)
        return self.script.pop(0) if self.script else chat_text("nothing more")


def _a_long_conversation(gw, calls, on_overflow=8):
    from ruikit.models.base import Call
    from ruikit.models.chat_completions import ChatCompletions

    conv = ChatCompletions(gw, "s", [], on_overflow=on_overflow)
    for k in range(calls):
        conv.history.append({"role": "assistant", "content": "", "tool_calls": [
            {"id": f"c{k}", "type": "function",
             "function": {"name": "code_run", "arguments": json.dumps({"source": PROGRAM})}}]})
        conv.results([Call(f"c{k}", "code_run", {"source": PROGRAM}, "")], [json.dumps(RAN)])
    return conv


def test_every_word_stays_until_the_context_is_full(tmp_path):
    # an episode that never fills the context is sent word for word as without the window
    from ruikit.harness.minimal_loop import KEEP_CALLS

    script = [chat_calls(("task_info", {}))] + \
        [chat_calls(("robot_state", {}))] * (KEEP_CALLS + 3) + \
        [chat_calls(("task_done", {"report": "done"}))]
    fake = FakeGateway(script)
    try:
        run_model("DS1", "easy", 0, "rui.pose/v1", "opus5", log_dir=str(tmp_path),
                  base_url=fake.url)
    finally:
        fake.close()
    first, last = fake.got[0][1], fake.got[-1][1]
    assert "stay whole" not in first["messages"][0]["content"]
    outs = [json.loads(m["content"]) for m in last["messages"] if m["role"] == "tool"]
    assert len(outs) == KEEP_CALLS + 4 and not any("cut" in o for o in outs)
    assert not any("context window" in json.dumps(m) for m in last["messages"])
    header = json.loads((tmp_path / "DS1_easy_s0_rui.pose-v1.model.jsonl")
                        .read_text().splitlines()[0])
    assert header["on_overflow"] == KEEP_CALLS and header["window_from"] is None
    assert header["keep_calls"] is None and header["keep_images"] == 6


def test_a_full_context_turns_the_window_on():
    whole = _a_long_conversation(_ContextOf(10 ** 9, []), 12)
    size = len(json.dumps(whole.outgoing()))
    gw = _ContextOf(size - 1, [chat_text("go on")])          # one character too long
    conv = _a_long_conversation(gw, 12)
    conv.requests = 5
    reply = conv.ask()
    # refused once, then sent again with the window on, and the model told why
    assert gw.refused == 1 and reply.text == "go on"
    assert conv.keep_calls == 8 and conv.window_from == 5 and conv.requests == 6
    sent = gw.sent[0]["messages"]
    assert "reached the model's context window" in sent[-1]["content"]
    outs = [json.loads(m["content"]) for m in sent if m["role"] == "tool"]
    assert [("cut" in o) for o in outs] == [True] * 4 + [False] * 8
    # the next request keeps the window: no refusal, nothing more to tell
    gw.script.append(chat_text("again"))
    conv.ask()
    assert gw.refused == 1 and json.dumps(gw.sent[1]["messages"]).count("context window") == 1


def test_the_window_narrows_while_the_request_is_still_too_long():
    cut8 = _a_long_conversation(_ContextOf(10 ** 9, []), 12)
    cut8.keep_calls = 8
    cut8._prune_calls()
    gw = _ContextOf(len(json.dumps(cut8.outgoing())) - 1, [chat_text("ok")])
    conv = _a_long_conversation(gw, 12)
    conv.ask()
    assert gw.refused == 2 and conv.keep_calls == 4
    # nothing left to narrow: the refusal ends the episode as before
    tiny = _ContextOf(10, [])
    conv = _a_long_conversation(tiny, 3)
    with pytest.raises(Exception) as err:
        conv.ask()
    assert "maximum context length" in str(err.value) and conv.keep_calls == 1


def test_only_a_full_context_turns_the_window_on():
    # another refusal, or no window asked for: the episode ends as before
    other = _ContextOf(10, [], refusal="qwen: HTTP 400: {\"error\":{\"message\":\"bad tool\"}}")
    conv = _a_long_conversation(other, 3)
    with pytest.raises(Exception):
        conv.ask()
    assert other.refused == 1 and conv.keep_calls is None
    conv = _a_long_conversation(_ContextOf(10, []), 3, on_overflow=None)
    with pytest.raises(Exception):
        conv.ask()
    assert conv.keep_calls is None and conv.window_from is None


def test_a_spent_quota_is_named_not_retried():
    fake_calls = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            fake_calls.append(1)
            data = b'{"error":{"message":"You have exceeded your monthly quota","code":"quota_exceeded"}}'
            self.send_response(402)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with pytest.raises(Exception) as err:
            Gateway("opus5", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1").ask("hi")
    finally:
        server.shutdown()
    assert "quota is used up" in str(err.value) and len(fake_calls) == 1


def _refusing_server(codes, retry_after):
    """Answers each POST with the next status (and Retry-After) from the lists,
    then 200 OK; remembers when each request came."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            import time
            self.rfile.read(int(self.headers["Content-Length"]))
            seen.append(time.time())
            k = len(seen) - 1
            code = codes[k] if k < len(codes) else 200
            data = json.dumps(chat_text("OK")).encode() if code == 200 else b"{}"
            self.send_response(code)
            if code != 200 and k < len(retry_after) and retry_after[k] is not None:
                self.send_header("Retry-After", retry_after[k])
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def test_rate_limits_back_off_whatever_retry_after_says():
    # an endpoint may ask for a second every time while its limit is per minute: the pauses
    # must keep doubling -- and never be shorter than asked
    server, seen = _refusing_server([429, 429, 429], ["0.3", "0.01", "0.01"])
    try:
        gw = Gateway("opus5", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                     backoff_s=0.05)
        conv = conversation(gw, system="s", tools=[])
        conv.user("hi")
        reply = conv.ask()
    finally:
        server.shutdown()
    gaps = [b - a for a, b in zip(seen, seen[1:])]
    assert reply.text == "OK" and reply.retries == 3 and len(seen) == 4
    assert gaps[0] >= 0.3                                  # as long as asked (> 0.05)
    assert gaps[1] >= 0.1 and gaps[2] >= 1.5 * gaps[1]     # doubling, though asked 0.01


def test_a_rate_limit_gets_more_tries_than_a_busy_server():
    from ruikit.core.result import ToolError

    for code, tries, kind in ((429, 4, "rate_limited"), (503, 2, "tool_failed")):
        server, seen = _refusing_server([code] * 10, [])
        try:
            gw = Gateway("opus5", base_url=f"http://127.0.0.1:{server.server_address[1]}/v1",
                         retries=1, rate_retries=3, backoff_s=0.01)
            with pytest.raises(ToolError) as err:
                gw.ask("hi")
        finally:
            server.shutdown()
        assert len(seen) == tries and err.value.code == kind, (code, len(seen))


def test_retry_after_is_read_in_every_form():
    import email.utils
    import time
    from email.message import Message

    from ruikit.models.gateway import _retry_after

    when = Message()
    when["Retry-After"] = email.utils.formatdate(time.time() + 5, usegmt=True)
    ms = Message()
    ms["retry-after-ms"] = "1500"
    ms["Retry-After"] = "1"
    odd = Message()
    odd["Retry-After"] = "soon"
    assert 3 < _retry_after(when) <= 5 and _retry_after(ms) == 1.5
    assert _retry_after(odd) == 0.0 and _retry_after(Message()) == 0.0


def test_a_local_model_thinks_apart_and_is_told_whether_to(monkeypatch):
    """local:<model> (v1.14): this machine's vLLM; its thinking comes back apart
    from its answer, is kept for the transcript and not sent back, and is off
    unless asked for."""
    import json
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from ruikit.models.conversation import conversation
    from ruikit.models.gateway import Gateway

    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            data = json.dumps({"choices": [{"finish_reason": "stop", "message": {
                "role": "assistant", "content": "\n\nOK", "reasoning": "the user wants OK"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 2}}).encode()
            self.send_response(200)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        monkeypatch.setenv("RUIKIT_LOCAL_BASE_URL", f"http://127.0.0.1:{srv.server_address[1]}/v1")
        gw = Gateway("local:Qwen/Qwen3.5-4B")
        assert gw.model == "Qwen/Qwen3.5-4B" and not gw.thinking
        conv = conversation(gw, system="be brief", tools=[])
        conv.user("say OK")
        reply = conv.ask()
        assert reply.text == "OK" and reply.reasoning == "the user wants OK"
        assert seen[0]["chat_template_kwargs"] == {"enable_thinking": False}
        assert "max_tokens" in seen[0]
        assert conv.history[-1] == {"role": "assistant", "content": "OK"}   # no thinking sent back
        monkeypatch.setenv("RUIKIT_LOCAL_THINKING", "1")
        assert Gateway("local:Qwen/Qwen3.5-4B").thinking
    finally:
        srv.shutdown()
