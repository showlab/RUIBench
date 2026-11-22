"""A stand-in for a model channel: answers each request with the next scripted body
and keeps every request it got."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class FakeGateway:
    """Answers each POST with the next scripted body; remembers what it got."""

    def __init__(self, script):
        self.script = list(script)
        self.got = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                outer.got.append((self.path, body))
                reply = outer.script.pop(0) if outer.script else chat_text("nothing more")
                data = json.dumps(reply).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def chat_calls(*calls):
    return {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": "",
        "tool_calls": [{"id": f"c{k}", "type": "function",
                        "function": {"name": n, "arguments": json.dumps(a)}}
                       for k, (n, a) in enumerate(calls)]}}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 10}}


def chat_text(text):
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": text}}]}

