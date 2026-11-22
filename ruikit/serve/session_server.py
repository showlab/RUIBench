"""The session server: one episode served to an agent that runs as its own process, one
reply per shell command or MCP request."""

from __future__ import annotations

import json
import os
import secrets
import socket
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from ruikit.serve.session import IDLE_SECONDS, SHELLS, _state, _write_state


# ----------------------------------------------------------------------------
# the server
# ----------------------------------------------------------------------------
def _result(res, folder: Path, inline: bool = False) -> dict:
    """A result as the client gets it: pictures as the files they were saved to,
    and with ``inline`` their PNG too (an MCP client hands it to the model)."""
    import base64

    out = res.as_json() if hasattr(res, "as_json") else {"ok": False, "error": str(res)}
    for im, raw in zip(out.get("images") or [], getattr(res, "images", None) or []):
        if im.get("path"):
            im["path"] = str((folder / "images" / im["path"]).resolve())
        if inline and getattr(raw, "png", None):
            im["png_b64"] = base64.b64encode(raw.png).decode("ascii")
    return out


def batch_tool(names) -> dict:
    """The MCP form of a reply with several calls: the ``batch`` tool, which a
    tool-calling loop does not need (its model sends several calls in one
    reply) and a client that sends each call by itself does."""
    return {"name": "batch", "title": "batch",
            "description": ("Send several tool calls of this interface as ONE reply. Every other "
                            "tool call is a reply of its own; calls batched here count as one, "
                            "start in order and run together where they can (moves of different "
                            "arms at the same time), and the reply comes back when every blocking "
                            "call is done -- task_info's one_reply says what a reply may hold. Use "
                            "it for calls that do not depend on each other's results. Returns each "
                            "call's result in order."),
            "inputSchema": {"type": "object", "additionalProperties": False, "required": ["calls"],
                            "properties": {"calls": {
                                "type": "array", "minItems": 1, "maxItems": 12,
                                "items": {"type": "object", "additionalProperties": False,
                                          "required": ["name", "arguments"],
                                          "properties": {
                                              "name": {"type": "string", "enum": sorted(names)},
                                              "arguments": {"type": "object"}}}}}},
            "annotations": {"readOnlyHint": False, "destructiveHint": False,
                            "openWorldHint": False}}


def serve(args) -> None:
    from ruikit.core import plugins
    from ruikit.env import Budget, Episode
    from ruikit.report.clips import RECORD_EVERY, RECORD_SIZE

    folder = Path(args.dir).resolve()
    if args.gpu is not None:
        os.environ["RUIBENCH_ISAAC_GPU"] = str(args.gpu)
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from ruikit.conditions import clock_seconds, clock_settings

    if args.clock:                            # 8s, paused or measured
        clock = clock_settings(args.clock)
        args.think_seconds = clock_seconds(args.clock)
    else:
        clock = (dict(clock="fixed", think_seconds=args.think_seconds, think_per="reply")
                 if args.think_seconds else {})
    # filmed, the episode ends as an unfilmed one does: no play-out after a success
    # (filming still changes the pictures the agent is shown: ruikit.experiments.worker)
    film = (dict(record_dir=str(folder / "frames"), record_every=RECORD_EVERY,
                 record_size=RECORD_SIZE) if args.record else {})
    state = {"status": "starting", "pid": os.getpid(), "task": args.task, "level": args.level,
             "seed": args.seed, "profile": args.profile, "engine": args.engine,
             "think_seconds": args.think_seconds, "clock": args.clock or None,
             "token": secrets.token_hex(16)}
    _write_state(folder, state)
    try:
        plugins.load(args.plugins)            # their tools, before the profile that lists them
        # the task's own limits (replies, simulated time) stand; the wall clock does
        # not end an episode -- an agent that thinks for hours is charged by --think-seconds
        ep = Episode(args.task, level=args.level, seed=args.seed, profile=args.profile,
                     engine=args.engine, log_dir=str(folder), image_dir=str(folder / "images"),
                     agent=args.agent, budget=Budget(max_tool_calls=6000, max_wall_seconds=86400.0),
                     **clock, **film)
    except Exception as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        _write_state(folder, state)
        raise
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    srv.settimeout(IDLE_SECONDS)
    notes: List[dict] = []
    shells: Dict[tuple, int] = {}             # a shell command -> the process that made its call
    t_start = time.time()
    state.update(status="ready", port=srv.getsockname()[1], instruction=ep.instruction,
                 hint=ep.inst.hint or "", budget=ep.budget.as_dict())
    _write_state(folder, state)

    def info() -> dict:
        from ruikit.harness.components.guides import guide_text
        from ruikit.serve.schemas import tool_entries

        return {"task": args.task, "level": args.level, "interface": ep.profile.id,
                "guide": guide_text(ep.profile.guide) if ep.profile.guide else "",
                "title": ep.profile.title, "instruction": ep.instruction, "hint": ep.inst.hint or "",
                "arms": [a.split("/")[-1] for a in ep.scene.arm_ids], "real": ep.real,
                "limits": {"replies": ep.budget.max_replies,
                           "simulated_seconds": ep.budget.max_sim_seconds,
                           "clock": ("this is a real robot: the world keeps going while you think, "
                                     "and the time you take counts, on top of what your moves take"
                                     if ep.real else
                                     f"as on a real robot the world keeps going while you think: "
                                     f"every reply costs {args.think_seconds:g} s of simulated time, "
                                     f"on top of what your moves take"
                                     if args.think_seconds else
                                     "as on a real robot the world keeps going while you think: "
                                     "the time you really take, on top of what your moves take"
                                     if args.clock == "measured" else
                                     "the world waits while you think; your moves take their time")},
                "tools": [{"name": e["name"], "description": e["description"].split("\n")[0][:200]}
                          for e in tool_entries(ep.profile)]}

    while not ep.over:
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            ep.finish("agent_stopped")
            break
        with conn:
            raw = b""
            try:
                while not raw.endswith(b"\n"):
                    chunk = conn.recv(65536)
                    if not chunk:
                        break
                    raw += chunk
            except OSError:
                continue                            # the caller has gone
            try:
                req = json.loads(raw.decode("utf-8"))
            except ValueError:
                _send(conn, b'{"error": "bad request"}\n')
                continue
            if req.get("token") != state["token"]:
                _send(conn, b'{"error": "wrong token"}\n')
                continue
            op = req.get("op")
            if op == "info":
                reply: Dict[str, Any] = info()
            elif op == "tool":
                from ruikit.serve.schemas import tool_entries

                want = str(req.get("name") or "")
                hits = [e for e in tool_entries(ep.profile) if e["name"] == want]
                flat = want.replace(".", "_", 1)
                if not hits and flat in ep.profile.code_tools:     # a function inside code_run
                    from ruikit.core.tool import get_tool
                    from ruikit.serve.schemas import _as_configured

                    spec = get_tool(flat).spec
                    hits = [{"name": want if "." in want else flat.replace("_", ".", 1),
                             "called": "inside a code_run program",
                             "description": _as_configured(flat, spec.description, ep.profile),
                             "parameters": spec.signature_doc().splitlines(),
                             "data": spec.data}]
                reply = hits[0] if hits else {"error": f"no tool {want!r} in this interface"}
            elif op == "mcp_tools":
                from ruikit.serve.schemas import mcp_tools

                extra = [{"name": c.name, "description": c.description,
                          "inputSchema": c.input_schema(), "annotations": {"readOnlyHint": False}}
                         for c in ep.custom_tools.values()]
                tools = mcp_tools(ep.profile, extra)
                reply = {"tools": tools + [batch_tool([t["name"] for t in tools])],
                         "instructions": info()}
            elif op == "call":
                calls = [(c[0], c[1] if len(c) > 1 else {}) for c in req.get("calls") or []]
                caller = req.get("caller") or {}
                shell = (caller.get("pid"), caller.get("start"))
                me = (caller.get("self"), caller.get("self_start"))
                if not calls:
                    reply = {"error": "a call needs a tool"}
                elif (caller.get("comm") in SHELLS and shells.get(shell, caller.get("self"))
                      != caller.get("self")) or (me[0] is not None and me in shells):
                    # a second process of one shell command -- or the shell itself, which
                    # became the last command of its chain ('A && B' runs B in its place)
                    reply = {"error": "one session call per shell command: this command already "
                                      "made its call (a loop, '&&', ';' or a pipe of calls is "
                                      "refused); send the next call as a command of its own"}
                else:
                    if caller.get("comm") in SHELLS:
                        shells[shell] = caller.get("self")
                    if req.get("why"):
                        notes.append({"row": len(ep.log), "call": 0, "note": str(req["why"])[:500]})
                    results = ep.call_batch(calls) if len(calls) > 1 else [ep.call(*calls[0])]
                    inline = bool(req.get("inline"))
                    reply = {"results": [_result(r, folder, inline) for r in results],
                             "sim_time": round(ep.sim_time, 2), "replies": ep.replies,
                             "limits": {"replies": ep.budget.max_replies,
                                        "simulated_seconds": ep.budget.max_sim_seconds}}
                    if ep.over:
                        reply["ended"] = ep.outcome.reason
            elif op == "stop":
                ep.finish("agent_stopped")
                reply = {"ended": ep.outcome.reason}
            else:
                reply = {"error": f"unknown op {op!r}"}
            _send(conn, (json.dumps(reply, ensure_ascii=False, default=str) + "\n").encode("utf-8"))
    (folder / "notes.json").write_text(json.dumps(notes, ensure_ascii=False))
    # the episode wrote its logs when it finished; the engine is not closed -- Isaac's
    # shutdown can hang for minutes (the coverage workers leave the same way)
    finalize(folder, film=args.record, wall_seconds=time.time() - t_start)
    os._exit(0)


def _outcome_row(folder: Path) -> dict:
    """The tool log's last row, which holds the outcome -- found by what it holds,
    not by name: an agent's own record (``agent.jsonl``) can sit beside it."""
    for p in sorted(folder.glob("*.jsonl")):
        if p.name.endswith((".episode.jsonl", ".model.jsonl")):
            continue
        try:
            row = json.loads(p.read_text().splitlines()[-1])
        except (OSError, ValueError, IndexError):
            continue
        if isinstance(row, dict) and "outcome" in row:
            return row
    return {}


def _send(conn: socket.socket, data: bytes) -> None:
    """The answer to a caller, who may have gone before it (its agent stopped
    while the call ran): the episode goes on, and a stop ends it (v2.3.7: the
    broken pipe ended the server, and Isaac hung until its worker gave up)."""
    try:
        conn.sendall(data)
    except OSError:
        pass


def finalize(folder: Path, film: bool = True, wall_seconds: Optional[float] = None) -> dict:
    """The session's outcome from the episode's own logs, and (``film``) its
    video and ``clip.json`` for the showcase -- also for a server that was
    stopped before it got that far."""
    from ruikit.diagnostics.coverage import Cell
    from ruikit.report.clips import _film

    state = _state(folder)
    last = _outcome_row(folder)
    out = last.get("outcome") or {}
    metrics = out.get("metrics") or {}
    state.update(status="over", reason=out.get("reason"), success=out.get("success"),
                 replies=metrics.get("replies"), sim_seconds=metrics.get("sim_seconds"))
    if wall_seconds is not None:
        state["wall_seconds"] = round(wall_seconds, 1)
    if film and (folder / "frames").exists():
        notes = json.loads((folder / "notes.json").read_text()) if (folder / "notes.json").exists() else []
        cell = Cell(state["task"], state["level"], state["seed"], state["profile"],
                    bool(out.get("success")), float(out.get("progress") or 0.0), str(out.get("reason")),
                    int(metrics.get("tool_calls", 0)), float(metrics.get("sim_seconds", 0.0)),
                    float(metrics.get("wall_seconds", 0.0)), [], replies=int(metrics.get("replies", 0)),
                    think_seconds=float(state.get("think_seconds") or 0.0), engine=state.get("engine", ""))
        try:
            _film(folder, cell, notes, state.get("budget") or {})
        except Exception as exc:                                  # pragma: no cover
            state["film_error"] = f"{type(exc).__name__}: {exc}"
    _write_state(folder, state)
    return state
