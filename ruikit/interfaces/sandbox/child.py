"""The program side of ``code_run``: runs *inside* the sandbox.

Standard library only (it runs with no access to the repository, the tasks, the
scorer or the network -- see ``host.py``).  It executes the agent's code with a
restricted import list and exposes the profile's tools as plain functions --
``robot.move_to(xyz_m=[...])``, ``vision.detect(frame_id, query)`` -- each of
which is one RPC to the tool server, where the call is validated, run on the
shared world, timed and logged.  Nothing the program does reaches the world
any other way.

Protocol: one JSON object per line.  host -> child: ``run`` (code, tools),
``result`` (the answer to a call), ``invoke`` (call a custom tool's ``run``).
child -> host: ``call`` (a tool call), ``done`` (stdout, value, error, shown).
"""

import builtins
import io
import json
import math
import os
import sys
import traceback

# the numpy mounted by the host lives here; nothing else is importable
if os.path.isdir("/pkgs"):
    sys.path.insert(0, "/pkgs")
elif os.environ.get("RUIKIT_NUMPY"):       # the unsandboxed fallback
    sys.path.insert(0, os.environ["RUIKIT_NUMPY"])

ALLOWED_MODULES = {
    "math", "cmath", "statistics", "random", "itertools", "functools", "collections",
    "heapq", "bisect", "json", "re", "string", "dataclasses", "typing", "copy",
    "operator", "numbers", "decimal", "fractions", "enum", "numpy",
}
BLOCKED_BUILTINS = ("open", "input", "breakpoint", "help", "exit", "quit")
MAX_STDOUT = 8000

_rpc_out = os.fdopen(os.dup(1), "w", buffering=1)
_rpc_in = sys.stdin
_real_import = builtins.__import__


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    root = name.split(".")[0]
    if level != 0 or root not in ALLOWED_MODULES:
        raise ImportError(f"import of {name!r} is not allowed in code_run "
                          f"(allowed: {sorted(ALLOWED_MODULES)})")
    return _real_import(name, globals, locals, fromlist, level)


class ToolCallError(Exception):
    """A tool returned an error.  ``code`` is one of the documented error codes."""

    def __init__(self, code, message, result):
        super().__init__(f"[{code}] {message}")
        self.code = code
        self.result = result


_seq = [0]
_shown = []
#: the newest frame id per camera this program has seen, so show('latest.wrist')
#: pins the picture of that moment, not whichever is newest when the program ends
_latest = {}


def _rpc(tool, args):
    _seq[0] += 1
    _rpc_out.write(json.dumps({"op": "call", "id": _seq[0], "tool": tool, "args": args}) + "\n")
    while True:
        line = _rpc_in.readline()
        if not line:
            raise SystemExit("tool server went away")
        msg = json.loads(line)
        if msg.get("op") == "result" and msg.get("id") == _seq[0]:
            return msg["result"]


def _make_tool(name, fields):
    def call(*args, **kwargs):
        if len(args) > len(fields):
            raise TypeError(f"{name} takes at most {len(fields)} positional arguments")
        for field, value in zip(fields, args):
            if field in kwargs:
                raise TypeError(f"{name} got {field!r} twice")
            kwargs[field] = value
        if hasattr(kwargs.get("xyz_m"), "tolist"):
            kwargs["xyz_m"] = kwargs["xyz_m"].tolist()
        kwargs = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in kwargs.items()}
        result = _rpc(name, kwargs)
        for f in ((result.get("data") or {}).get("frames") or []) if isinstance(result, dict) else []:
            if isinstance(f, dict) and f.get("frame_id"):
                _latest[f.get("camera")] = _latest[None] = f["frame_id"]
        if not result.get("ok", True):
            raise ToolCallError(result.get("error", "tool_failed"), result.get("text", ""), result)
        return result

    call.__name__ = name
    call.__doc__ = f"{name}({', '.join(fields)}) -- tool call"
    return call


class _Family:
    def __init__(self, name):
        self._name = name

    def __repr__(self):
        return f"<tools {self._name}: {', '.join(k for k in vars(self) if not k.startswith('_'))}>"


def _namespace(tools):
    ns = {}
    for name, fields in tools.items():
        family, _, rest = name.partition("_")
        fam = ns.setdefault(family, _Family(family))
        setattr(fam, rest or name, _make_tool(name, fields))
        ns[name] = getattr(fam, rest or name)

    def show(frame_id, box=None):
        """Attach a frame's picture to what the model sees after this program;
        box=[u0, v0, u1, v1] attaches that part of it enlarged, as camera_zoom does.
        'latest.<camera>' means that camera's newest picture at the moment of the call."""
        fid = str(frame_id)
        if fid == "latest" or fid.startswith("latest."):
            frame_id = _latest.get(fid[len("latest."):] if "." in fid else None, fid)
        if box is not None:
            u0, v0, u1, v1 = (int(round(float(x))) for x in box)
            _shown.append(f"{frame_id}@{u0},{v0},{u1},{v1}")
        else:
            _shown.append(str(frame_id))

    ns["show"] = show
    ns["ToolCallError"] = ToolCallError
    ns["json"], ns["math"] = json, math          # the usual two, without an import
    safe = {k: getattr(builtins, k) for k in dir(builtins) if k not in BLOCKED_BUILTINS}
    safe["__import__"] = _safe_import
    ns["__builtins__"] = safe
    ns["__name__"] = "__program__"
    return ns


_globals = {}


def _run(code, tools, call_custom=None):
    """Run a program (persistent namespace) or one call of a custom tool (a
    fresh namespace each time, so a tool cannot touch the program's state)."""
    global _globals
    if call_custom is not None:
        ns = _namespace(tools)
    else:
        if not _globals:
            _globals = _namespace(tools)
        else:
            _globals.update({k: v for k, v in _namespace(tools).items() if k != "__builtins__"})
        ns = _globals
    buf = io.StringIO()
    old = sys.stdout
    sys.stdout = buf
    _shown.clear()
    value, error = None, None
    try:
        if call_custom is not None:
            exec(compile(code, "<tool>", "exec"), ns, ns)
            fn = ns.get("run")
            if not callable(fn):
                raise NameError("a custom tool's code must define run(**arguments)")
            value = fn(**call_custom)
        else:
            tree = compile(code, "<program>", "exec")
            # each run answers with its own `result`, not one an earlier run left (v2.3.4)
            ns.pop("result", None)
            exec(tree, ns, ns)
            value = ns.get("result")
    except SystemExit:
        raise
    except BaseException:
        error = "".join(traceback.format_exception(*sys.exc_info())[-6:])
    finally:
        sys.stdout = old
    out = buf.getvalue()
    if len(out) > MAX_STDOUT:
        out = out[:MAX_STDOUT] + f"\n... ({len(out) - MAX_STDOUT} more characters cut)"
    try:
        json.dumps(value)
    except (TypeError, ValueError):
        value = repr(value)
    _rpc_out.write(json.dumps({"op": "done", "stdout": out, "value": value, "error": error,
                               "shown": list(_shown)}) + "\n")


def main():
    for line in _rpc_in:
        msg = json.loads(line)
        if msg.get("op") == "run":
            _run(msg["code"], msg["tools"])
        elif msg.get("op") == "invoke":
            _run(msg["code"], msg["tools"], call_custom=msg.get("args") or {})
        elif msg.get("op") == "exit":
            break


if __name__ == "__main__":
    main()
