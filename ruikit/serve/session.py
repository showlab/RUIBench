"""One task episode that an outside agent drives from the shell, one reply a command.

For agents that act through a shell (Claude Code, a Claude Agent SDK agent):
the episode lives in a server process, and every ``call`` is one reply --
one tool call, or several independent ones sent together (``--batch``).  The
pictures a reply returns are saved as files the agent can open; nothing else
about the world reaches it.

    python -m ruikit.serve.session start --dir D --task DS1 --profile rui.pose/v1 --engine isaac
    python -m ruikit.serve.session info --dir D            # the task, the limits, the tools
    python -m ruikit.serve.session tool --dir D robot_move_to   # one tool's parameters
    python -m ruikit.serve.session call --dir D camera_look '{}' --why "see the table"
    python -m ruikit.serve.session call --dir D \\
        --batch '[["vision_detect", {...}], ["vision_pose", {...}]]'
    python -m ruikit.serve.session call --dir D code_run --program prog.py  # or --program - (stdin)
    python -m ruikit.serve.session stop --dir D
    python -m ruikit.serve.session start ... --plugins my/tools.py --profile my/interface.yaml

The episode's limits and clock are the Episode's (replies <= 10N, simulated
time <= min(N x T_ref, 1800 s)); ``--clock`` (``8s``, ``paused``, ``measured``)
or ``--think-seconds S`` sets what a reply costs, and with neither the world
waits while the agent thinks.  When the episode ends (``task_done``, a limit)
the server writes its logs into D -- the same clip folder the showcase reads
(``--record`` films it, and the episode still ends at a success) -- and exits.

One shell command makes at most one ``call``: a second call from the same
shell -- a loop, ``&&``, a pipe -- is refused, so an agent of a per-action
interface cannot write a program in the shell instead.  An agent can also
reach the session over MCP, with no shell at all
(``python -m ruikit.serve.mcp_stdio --session D``; ``docs/AGENT_TESTING.md``).
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

IDLE_SECONDS = 3600.0
STATE = "session.json"
#: a ``call`` whose parent process is one of these came from a shell command
#: (or a command runner): a second one from the same process is refused
SHELLS = ("bash", "sh", "dash", "zsh", "fish", "ksh", "tcsh", "csh", "xargs", "parallel",
          "env", "timeout", "nice", "nohup", "watch", "find", "time", "stdbuf", "script")


def _state(folder: Path) -> dict:
    try:
        return json.loads((folder / STATE).read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _write_state(folder: Path, state: dict) -> None:
    tmp = folder / (STATE + ".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1))
    os.chmod(tmp, 0o600)
    tmp.replace(folder / STATE)


# ----------------------------------------------------------------------------
# the client
# ----------------------------------------------------------------------------
def _caller() -> dict:
    """The process this command was started by: a shell's pid and start time
    tell one shell command from the next; ``self``, which process of it calls
    (one process may call again -- a second process of the same shell may not)."""
    def started(pid) -> str:
        try:
            return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
        except (OSError, IndexError):
            return ""

    pid = os.getppid()
    try:
        comm = Path(f"/proc/{pid}/comm").read_text().strip()
    except OSError:
        comm = ""
    return {"pid": pid, "start": started(pid), "comm": comm, "self": os.getpid(),
            "self_start": started(os.getpid())}


def _ask(folder: Path, req: dict, timeout: float = 1800.0) -> dict:
    state = _state(folder)
    if state.get("status") != "ready":
        return {"error": f"the session is {state.get('status', 'not started')}"
                         + (f" ({state.get('reason')})" if state.get("reason") else "")}
    req = dict(req, token=state["token"])
    with socket.create_connection(("127.0.0.1", state["port"]), timeout=timeout) as s:
        s.sendall((json.dumps(req, ensure_ascii=False) + "\n").encode("utf-8"))
        raw = b""
        while not raw.endswith(b"\n"):
            chunk = s.recv(1 << 20)
            if not chunk:
                break
            raw += chunk
    return json.loads(raw.decode("utf-8")) if raw else {"error": "no answer"}


def _print_results(reply: dict, width: int) -> None:
    if reply.get("error"):
        print(f"ERROR {reply['error']}")
        return
    for k, r in enumerate(reply.get("results") or [], 1):
        head = "ok" if r.get("ok") else f"ERROR {r.get('error')}"
        print(f"[{k}] {head}")
        for key in ("text", "hint"):
            if r.get(key):
                print(f"    {key}: {r[key]}")
        if r.get("data"):
            data = json.dumps(r["data"], ensure_ascii=False)
            more = len(data) - width
            print(f"    data: {data[:width]}" + (f" … [{more} more characters not shown; "
                                                 f"--width {len(data)} shows all]" if more > 0 else ""))
        for im in r.get("images") or []:
            where = im.get("path") or "(no pixels on this engine)"
            what = f"{im.get('camera')}, {im.get('width')}x{im.get('height')}"
            if (im.get("note") or "").startswith("zoom"):
                what += f", {im['note']}"
            print(f"    picture {im.get('frame_id')} ({what}): {where}")
        for ev in r.get("events") or []:
            print(f"    event: {json.dumps(ev, ensure_ascii=False)[:300]}")
    lim = reply.get("limits") or {}
    print(f"-- simulated time {reply.get('sim_time')} s of {lim.get('simulated_seconds')}; "
          f"replies {reply.get('replies')} of {lim.get('replies')}")
    if reply.get("ended"):
        print(f"-- THE EPISODE HAS ENDED ({reply['ended']})")


def main(argv: Optional[List[str]] = None) -> int:
    from ruikit.serve.session_server import finalize, serve

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("start")
    st.add_argument("--dir", required=True)
    st.add_argument("--task", required=True)
    st.add_argument("--level", default="easy")
    st.add_argument("--seed", type=int, default=0)
    st.add_argument("--profile", required=True)
    st.add_argument("--engine", default="mock")
    st.add_argument("--gpu", type=int, default=None)
    st.add_argument("--think-seconds", type=float, default=0.0)
    st.add_argument("--clock", default=None,
                    help="8s (any <N>s: every reply costs N s), paused or measured; "
                         "instead of --think-seconds")
    st.add_argument("--record", action="store_true")
    st.add_argument("--agent", default="an outside agent through ruikit.serve.session")
    st.add_argument("--plugins", nargs="+", default=[],
                    help="files or modules with tools the profile lists (outside the library)")
    st.add_argument("--serve", action="store_true", help=argparse.SUPPRESS)
    for name in ("info", "stop", "status", "finalize"):
        p = sub.add_parser(name)
        p.add_argument("--dir", required=True)
    tl = sub.add_parser("tool")
    tl.add_argument("--dir", required=True)
    tl.add_argument("name")
    cl = sub.add_parser("call")
    cl.add_argument("--dir", required=True)
    cl.add_argument("tool", nargs="?")
    cl.add_argument("args", nargs="?", default="{}")
    cl.add_argument("--batch", default=None, help='[["tool", {args}], ...]: independent calls, one reply')
    cl.add_argument("--why", default="", help="what this reply is for (shown on the replay page)")
    cl.add_argument("--program", default=None, metavar="FILE",
                    help="for code_run: the program, read from FILE or from standard input ('-') "
                         "as it is written, instead of escaped into the JSON (its other "
                         "arguments still go there)")
    # task_info alone is 2.6k characters: a width of 2500 cut off its last conventions
    cl.add_argument("--width", type=int, default=20000, help="print at most this much of each result")
    args = ap.parse_args(argv)
    folder = Path(args.dir)

    if args.cmd == "start":
        folder.mkdir(parents=True, exist_ok=True)
        if args.serve:
            serve(args)
            return 0
        cmd = [sys.executable, "-m", "ruikit.serve.session"] + (argv if argv is not None else sys.argv[1:]) + ["--serve"]
        with open(folder / "server.log", "w") as log:
            subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                             start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 1800:
            state = _state(folder)
            if state.get("status") in ("ready", "failed"):
                break
            time.sleep(2)
        state = _state(folder)
        if state.get("status") != "ready":
            print(f"the session did not start: {state.get('error') or state.get('status')}; "
                  f"see {folder / 'server.log'}")
            return 1
        print(f"ready after {time.time() - t0:.0f} s")
        args.cmd = "info"
    if args.cmd == "info":
        reply = _ask(folder, {"op": "info"})
        print(json.dumps(reply, ensure_ascii=False, indent=1))
        return 0 if "error" not in reply else 1
    if args.cmd == "tool":
        print(json.dumps(_ask(folder, {"op": "tool", "name": args.name}), ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "status":
        state = _state(folder)
        print(json.dumps({k: v for k, v in state.items() if k != "token"}, ensure_ascii=False, indent=1))
        return 0
    if args.cmd == "finalize":
        state = finalize(folder)
        print(json.dumps({k: v for k, v in state.items() if k != "token"}, ensure_ascii=False))
        return 0
    if args.cmd == "stop":
        print(json.dumps(_ask(folder, {"op": "stop"}), ensure_ascii=False))
        return 0
    if args.cmd == "call":
        if args.batch:
            calls = json.loads(args.batch)
        elif args.tool:
            given = json.loads(args.args)
            if args.program is not None:
                # a program escaped by hand into one line of JSON cost every agent
                # replies, and some wrote files of their own to get round it
                given["source"] = (sys.stdin.read() if args.program == "-"
                                   else Path(args.program).read_text(encoding="utf-8"))
            calls = [[args.tool, given]]
        else:
            ap.error("call needs a tool or --batch")
        _print_results(_ask(folder, {"op": "call", "calls": calls, "why": args.why,
                                     "caller": _caller()}), args.width)
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
