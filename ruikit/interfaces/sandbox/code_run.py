"""``code_run``: a program over the interface's tools, run in the sandbox.

The program runs in a sandboxed process (``ruikit.interfaces.sandbox.host``)
where the interface's ``code_tools`` are plain functions --
``camera.look(...)``, ``vision.detect(frame_id, "ring")`` -- each one a real
tool call answered by the episode *while the program runs*.  So a program that
moves and then looks sees the world after the move.

Two modes: ``script`` runs the program once and hands back; ``controller``
runs a ``step(state)`` function every control tick for a given time, each tick
seeing the current proprioception and new events.

Blind (``code_run``'s ``blind`` setting, the sequence form): each call starts
its action as the tool does and hands the program nothing back, so a program is
a series of actions with nothing seen in between; what each action reported
comes with the result once the program ends.  Only ``script`` runs.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Literal, Optional

from ruikit.core.params import param
from ruikit.core.result import ToolError, ToolResult
from ruikit.core.tool import Cost, get_tool, tool
from ruikit.perception.camera import _zoomable, picture

MAX_SOURCE_CHARS = 20000
#: what an action's report keeps in a blind program's result, in characters
REPORT_CHARS = 200


def tool_fields(names) -> dict:
    """``{tool name: [parameter names in order]}`` for the program's namespace."""
    out = {}
    for name in names:
        spec = get_tool(name).spec
        out[name] = [f.name for f in fields(spec.params) if not f.metadata.get("hidden")]
    return out


def api_docs(names) -> str:
    """The code-mode API reference: one block per callable tool."""
    blocks = []
    for name in names:
        spec = get_tool(name).spec
        args = ", ".join(f.name for f in fields(spec.params) if not f.metadata.get("hidden"))
        blocks.append(f"{spec.code_name}({args})\n    {spec.description}\n"
                      + "\n".join("    " + line for line in spec.signature_doc().splitlines()))
    return "\n\n".join(blocks)


def run_program(ctx, source: str, callable_names, custom_args=None, blind: bool = False) -> dict:
    """Run ``source`` in the sandbox, answering its tool calls on the episode;
    ``blind``: each call's result is kept from the program (it gets ``{}``) and
    listed in ``reports``."""
    allowed = set(callable_names)
    calls = [0]
    reports = []

    def on_call(name: str, args: dict) -> dict:
        calls[0] += 1
        result = ctx.call(name, args, allowed=allowed).as_json()
        if not blind:
            return result
        report = {"call": name, "ok": bool(result.get("ok", True)),
                  "text": str(result.get("text") or "")[:REPORT_CHARS]}
        if result.get("error"):
            report["error"] = result["error"]
        reports.append(report)
        return {}

    fields_by_tool = tool_fields([n for n in callable_names if not n.startswith("my_")])
    for name, custom in ctx.custom_tools().items():
        if name in allowed:
            fields_by_tool[name] = list(custom.fields)
    out = ctx.sandbox().run(source, fields_by_tool, on_call, custom_args=custom_args)
    out["calls"] = calls[0]
    if blind:
        out["reports"] = reports
    ctx.advance(float(out.get("compute_s", 0.0)))
    return out


#: what code_run says of itself where its programs run blind (the sequence form)
BLIND_DESCRIPTION = """Send the robot a series of actions, written as a Python program. In this
interface a program only acts: its functions (listed below: this configuration's moves, the
gripper, waiting and stopping) each run their action as the tool of the same name does, one
after another, but return nothing (an empty dict) and raise nothing when an action fails or the
world stops it, so a program cannot look, read the robot or decide on what happened. Look with
the perception tools between programs. When the program ends, the result
lists what each action reported (data.actions), and pictures of the world after it ride on the
result as after any move. A program may import only bisect, cmath, collections, copy,
dataclasses, decimal, enum, fractions, functools, heapq, itertools, json, math, numbers, numpy,
operator, random, re, statistics, string and typing; there is no file system or network.
print() output is returned too (8000 characters at most). The program's own compute time is
charged to the simulated clock (30 s at most per run), and its variables persist between runs
in this episode."""


@dataclass(frozen=True)
class CodeParams:
    source: str = param("Python source. Tools are functions grouped by family (camera.look, "
                        "vision.detect, ...); set `result` to return a value, call "
                        "show(frame_id) to see a picture afterwards")
    mode: Literal["script", "controller"] = param(
        "'script': run once; 'controller': call step(state) every tick for `seconds`",
        default="script")
    seconds: Optional[float] = param("controller mode: how long to run", default=None,
                                     minimum=0.1, maximum=60.0)
    hz: float = param("controller mode: ticks per second", default=10.0, minimum=1.0,
                      maximum=20.0)


@tool(name="code_run", family="code", params=CodeParams, source="agent",
      returns="result", cost=Cost("physical"),
      examples=[{"source": "f = camera.look(['front'])['data']['frames'][0]['frame_id']\n"
                           "rings = vision.detect(f, 'ring')['data']['objects']\n"
                           "result = [r['label'] for r in rings]"}],
      description="""Run a Python program against the robot. The tools of this configuration
      are functions grouped by family -- e.g. camera.look(cameras) and vision.detect(frame_id,
      query); this configuration's moves and the rest are listed below -- and each returns the
      tool's result as a dict ({'ok', 'text', 'data', ...}) or raises ToolCallError. Calls run
      on the live world while the program runs, so a program can move, look again and decide.
      A program may import only bisect, cmath, collections, copy, dataclasses, decimal, enum,
      fractions, functools, heapq, itertools, json, math, numbers, numpy, operator, random, re,
      statistics, string and typing; there is no file system or network. Set `result` to
      return a value (each run starts without one); print() output is returned too (8000
      characters at most); show(frame_id) attaches that picture to this tool's result, and
      show(frame_id, box=[u0, v0, u1, v1]) that part of it enlarged (4 pictures a run at most:
      the rest are named in data.not_shown). What each function's result holds is listed with
      it below ("-> data").
      The program's own compute time is charged to the simulated clock (30 s at most per run),
      and its variables but `result` persist between runs in this episode.""")
def code_run(ctx, p: CodeParams) -> ToolResult:
    if len(p.source) > MAX_SOURCE_CHARS:
        raise ToolError("invalid_args", f"the program is longer than {MAX_SOURCE_CHARS} characters")
    try:
        compile(p.source, "<program>", "exec")
    except SyntaxError as exc:
        raise ToolError("invalid_args", f"syntax error on line {exc.lineno}: {exc.msg}")
    names = list(ctx.profile.code_tools) + [n for n in ctx.custom_tools()]
    blind = bool(ctx.profile.tool_params("code_run").get("blind"))
    if p.mode == "script":
        out = run_program(ctx, p.source, names, blind=blind)
    elif blind:
        raise ToolError("invalid_args", "this interface runs a program once, as a series of "
                        "actions with nothing seen in between: there is no controller mode")
    else:
        if not p.seconds:
            raise ToolError("invalid_args", "controller mode needs seconds")
        out = _controller(ctx, p, names)
    images, unshown = [], []
    wanted = out.get("shown", [])
    for shown in wanted[4:]:                  # said, not silently dropped (CL2, rui.code/v1)
        unshown.append(f"{shown} (at most 4 pictures per program)")
    for shown in wanted[:4]:
        fid, _, box = str(shown).partition("@")
        try:
            fid, camera, image, width, height = _zoomable(ctx, fid)
            if box:
                u0, v0, u1, v1 = (int(x) for x in box.split(","))
                u0, u1 = max(0, min(u0, u1)), min(width, max(u0, u1))
                v0, v1 = max(0, min(v0, v1)), min(height, max(v0, v1))
                if u1 - u0 < 4 or v1 - v0 < 4:
                    raise ValueError("empty box")
                images.append(picture(fid, camera, image, width, height, (u0, v0, u1, v1),
                                      upscale_to=512))
            else:
                images.append(picture(fid, camera, image, width, height))
        except (ToolError, ValueError):
            unshown.append(str(shown))           # a typo, an empty box: said, not dropped
    data = {"stdout": out.get("stdout", ""), "result": out.get("value"),
            "error": out.get("error"), "tool_calls": out.get("calls", 0),
            "compute_s": out.get("compute_s", 0.0), "isolation": out.get("isolation")}
    if blind:
        data["actions"] = out.get("reports", [])
    if unshown:
        data["not_shown"] = {"frames": unshown, "why": "show() takes a frame_id this episode "
                                                       "returned (a camera's or a demonstration's) "
                                                       "and a box that is not empty"}
    if out.get("error"):
        return ToolResult(data=data, text="the program raised an error", images=images,
                          error="tool_failed", hint=out["error"].strip().splitlines()[-1][:300])
    if blind:
        failed = sum(1 for r in data["actions"] if not r["ok"])
        return ToolResult(data=data, images=images,
                          text=f"program finished after {data['tool_calls']} action(s)"
                               + (f", {failed} of them reported a failure" if failed else "")
                               + "; each action's report is in data.actions")
    return ToolResult(data=data, text=f"program finished after {data['tool_calls']} tool call(s)",
                      images=images)


def _controller(ctx, p: CodeParams, names) -> dict:
    """Define the program, then call its step(state) every 1/hz seconds."""
    setup = run_program(ctx, p.source + "\nassert callable(step), 'define step(state)'", names)
    if setup.get("error"):
        return setup
    ticks = int(round(p.seconds * p.hz))
    total = {"stdout": setup.get("stdout", ""), "calls": setup.get("calls", 0),
             "compute_s": setup.get("compute_s", 0.0), "shown": list(setup.get("shown", [])),
             "isolation": setup.get("isolation"), "value": None, "error": None}
    seen = len(ctx.task.events())
    for _ in range(ticks):
        state = {"t": round(ctx.sim_time, 3),
                 "arms": [ctx.arm_state(a) for a in sorted(ctx.station.arms)],
                 "events": ctx.task.events(seen)}
        seen += len(state["events"])
        out = run_program(ctx, f"result = step({state!r})", names)
        total["calls"] += out.get("calls", 0)
        total["compute_s"] += out.get("compute_s", 0.0)
        total["stdout"] = (total["stdout"] + out.get("stdout", ""))[-8000:]
        total["shown"] += out.get("shown", [])
        total["value"] = out.get("value")
        if out.get("error"):
            total["error"] = out["error"]
            break
        ctx.advance(1.0 / p.hz)
    return total
