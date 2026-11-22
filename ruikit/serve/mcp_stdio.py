"""A minimal MCP server over stdio for one episode.

Lets any MCP client -- Claude Code, Codex, Gemini CLI, OpenHands -- drive a
RUIBench episode through a profile's tools::

    python -m ruikit.serve.mcp_stdio --task WK1 --level easy --seed 0 \\
        --profile open/v1 --engine mock --log-dir runs/ruikit
    python -m ruikit.serve.mcp_stdio --session D     # an episode ruikit.serve.session started

``--session`` attaches to a session that is already running -- Isaac started
on its own card, its clock and limits set, the video filming -- so an agent's
MCP client does not wait minutes for a simulator to start.  Its tools are the
profile's plus ``batch``: several calls as one reply, as a tool-calling loop
sends them, since every other call is a reply of its own.  Pictures come back
as image content, so the agent needs no file tools.

Standard library only (JSON-RPC 2.0, one message per line).  It answers the
2026-07-28 stateless shape (``server/discover``, per-request ``_meta``) and the
older handshake (``initialize``) alike, because clients in the field still
send it.  State that spans calls lives in the episode and in server-minted
handles passed as ordinary tool arguments, as the current spec asks -- no
protocol sessions, no dependency on the tasks extension.

Image results carry the picture as ``image`` content plus a text line, and no
``structuredContent`` (a structured result makes some clients drop images);
the JSON of the result follows as text.  When the tool list changes (a custom
tool defined with ``toolkit_define``) the server says so with
``notifications/tools/list_changed``.
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

PROTOCOL = "2026-07-28"
SERVER = {"name": "ruikit", "version": "0.1.0"}


class McpServer:
    def __init__(self, episode, out=None) -> None:
        self.ep = episode
        self.out = out or sys.stdout
        self._custom_seen = 0

    # -- plumbing --------------------------------------------------------------
    def send(self, msg: dict) -> None:
        self.out.write(json.dumps(msg) + "\n")
        self.out.flush()

    def serve(self, inp=None) -> None:
        for line in (inp or sys.stdin):
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                self.send({"jsonrpc": "2.0", "id": None,
                           "error": {"code": -32700, "message": "parse error"}})
                continue
            reply = self.handle(msg)
            if reply is not None:
                self.send(reply)
            if self.tools_changed():
                self.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})

    def tools_changed(self) -> bool:
        if len(self.ep.custom_tools) != self._custom_seen:
            self._custom_seen = len(self.ep.custom_tools)
            return True
        return False

    # -- methods ------------------------------------------------------------------
    def handle(self, msg: dict) -> Optional[dict]:
        method, mid = msg.get("method"), msg.get("id")
        params = msg.get("params") or {}
        if mid is None:                      # a notification: nothing to answer
            return None
        try:
            result = self.dispatch(method, params)
        except KeyError:
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32601, "message": f"method not found: {method}"}}
        except Exception as exc:              # pragma: no cover - defensive
            return {"jsonrpc": "2.0", "id": mid,
                    "error": {"code": -32603, "message": f"{type(exc).__name__}: {exc}"}}
        return {"jsonrpc": "2.0", "id": mid, "result": result}

    def dispatch(self, method: str, params: dict) -> dict:
        if method in ("initialize", "server/discover"):
            # the client's own version: it may not speak ours (Claude Code 2.1 refuses
            # 2026-07-28), and nothing this server says differs between versions
            asked = params.get("protocolVersion") or (params.get("protocolVersions") or [None])[0]
            return {"protocolVersion": asked or PROTOCOL, "serverInfo": SERVER,
                    "capabilities": {"tools": {"listChanged": True}},
                    "instructions": self.instructions(), "resultType": "complete"}
        if method == "ping":
            return {"resultType": "complete"}
        if method == "tools/list":
            return {"tools": self.tool_list(), "resultType": "complete",
                    "ttlMs": 0, "cacheScope": "private"}
        if method == "tools/call":
            return self.call(params.get("name", ""), params.get("arguments") or {})
        raise KeyError(method)

    def instructions(self) -> str:
        ep = self.ep
        return (f"RUIBench task {ep.inst.task_id} ({ep.inst.level}). Instruction: "
                f"{ep.instruction} Hint: {ep.inst.hint or '-'} Act only through these tools; "
                "the world keeps moving while you think. Call task_info for the conventions "
                "and task_done when finished.")

    def tool_list(self) -> list:
        from ruikit.serve.schemas import mcp_tools

        extra = [{"name": c.name, "description": c.description, "inputSchema": c.input_schema(),
                  "annotations": {"readOnlyHint": False}}
                 for c in self.ep.custom_tools.values()]
        return mcp_tools(self.ep.profile, extra)

    def call(self, name: str, args: Dict[str, Any]) -> dict:
        result = self.ep.call(name, args)
        return to_mcp_content(result)


class SessionMcp(McpServer):
    """The MCP face of a session ``ruikit.serve.session`` runs: every call goes
    to it as one reply, ``batch`` as one reply of several calls."""

    def __init__(self, folder, out=None, wait_s: float = 1800.0, keep_pictures: bool = False) -> None:
        from ruikit.serve.session import _state

        self.folder = Path(folder)
        self.out = out or sys.stdout
        #: never flag a reply that carries pictures as an error: Claude Code shows
        #: the model only the text of a result flagged isError (v2.3.4: a move
        #: that ended blocked, a program that raised, lost their pictures)
        self.keep_pictures = keep_pictures
        t0 = time.time()
        while _state(self.folder).get("status") not in ("ready", "failed", "over") \
                and time.time() - t0 < wait_s:
            time.sleep(1.0)
        self._listed = None

    def _ask(self, req: dict) -> dict:
        from ruikit.serve.session import _ask

        return _ask(self.folder, req)

    def tools_changed(self) -> bool:
        return False

    def instructions(self) -> str:
        info = (self._list().get("instructions") or {})
        return (f"RUIBench task {info.get('task')} ({info.get('level')}) through "
                f"{info.get('interface')}. Instruction: {info.get('instruction')} Hint: "
                f"{info.get('hint') or '-'} Act only through these tools; every tool call is one "
                "reply (batch sends several as one). Call task_info for the conventions and "
                "task_done when finished.")

    def _list(self) -> dict:
        if self._listed is None:
            got = self._ask({"op": "mcp_tools"})
            if got.get("error"):
                return {"tools": [], "error": got["error"]}
            self._listed = got
        return self._listed

    def tool_list(self) -> list:
        return list(self._list().get("tools") or [])

    def call(self, name: str, args: Dict[str, Any]) -> dict:
        if name == "batch":
            calls = [[c.get("name", ""), c.get("arguments") or {}]
                     for c in (args or {}).get("calls") or [] if isinstance(c, dict)]
        else:
            calls = [[name, args or {}]]
        reply = self._ask({"op": "call", "calls": calls, "inline": True, "via": "mcp"})
        if reply.get("error"):
            return {"content": [{"type": "text", "text": f"ERROR {reply['error']}"}],
                    "isError": True, "resultType": "complete"}
        content, bad = [], False
        results = reply.get("results") or []
        for k, r in enumerate(results, 1):
            for im in r.get("images") or []:
                if im.get("png_b64"):
                    content.append({"type": "image", "mimeType": "image/png",
                                    "data": im.pop("png_b64")})
                content.append({"type": "text", "text": f"[{im.get('frame_id')}] {im.get('camera')} "
                                                        f"{im.get('width')}x{im.get('height')}"
                                                        + (f" ({im['note']})" if im.get("note") else "")})
            shown = {k_: v for k_, v in r.items() if k_ != "images"}
            shown["images"] = [{k_: v for k_, v in im.items() if k_ != "path"}
                               for im in r.get("images") or []]
            head = f"[{k}] {calls[k - 1][0]}: " if len(results) > 1 else ""
            content.append({"type": "text", "text": head + json.dumps(shown, ensure_ascii=False)})
            bad = bad or not r.get("ok", True)
        lim = reply.get("limits") or {}
        status = (f"simulated time {reply.get('sim_time')} s of {lim.get('simulated_seconds')}; "
                  f"replies {reply.get('replies')} of {lim.get('replies')}")
        if reply.get("ended"):
            status += f". THE EPISODE HAS ENDED ({reply['ended']})"
        content.append({"type": "text", "text": status})
        pictures = any(c["type"] == "image" for c in content)
        return {"content": content,
                "isError": bad and len(results) == 1 and not (self.keep_pictures and pictures),
                "resultType": "complete"}


def to_mcp_content(result) -> dict:
    content = []
    for im in result.images:
        if im.png:
            content.append({"type": "image", "mimeType": "image/png",
                            "data": base64.b64encode(im.png).decode("ascii")})
        content.append({"type": "text", "text": f"[{im.frame_id}] {im.camera} "
                                                f"{im.width}x{im.height}"
                                                + (f" ({im.note})" if im.note else "")})
    content.append({"type": "text", "text": json.dumps(result.as_json())})
    return {"content": content, "isError": result.is_error, "resultType": "complete"}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Serve one RUIBench episode over MCP (stdio).")
    ap.add_argument("--session", default=None,
                    help="attach to the session ruikit.serve.session runs in this folder")
    ap.add_argument("--task", default=None)
    ap.add_argument("--level", default="easy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="open/v1")
    ap.add_argument("--engine", default="mock")
    ap.add_argument("--clock", default="measured", choices=("zero", "fixed", "measured"))
    ap.add_argument("--think-seconds", type=float, default=0.0)
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--keep-pictures", action="store_true",
                    help="never flag a reply that carries pictures as an error (Claude Code drops "
                         "the pictures of an error)")
    args = ap.parse_args(argv)
    if args.session:
        SessionMcp(args.session, keep_pictures=args.keep_pictures).serve()
        return 0
    if not args.task:
        ap.error("--task (or --session) is required")
    from ruikit.env import Episode

    ep = Episode(args.task, level=args.level, seed=args.seed, profile=args.profile,
                 engine=args.engine, clock=args.clock, think_seconds=args.think_seconds,
                 log_dir=args.log_dir)
    try:
        McpServer(ep).serve()
    finally:
        ep.close()
        if ep.outcome:
            print(json.dumps({"outcome": ep.outcome.__dict__}), file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
