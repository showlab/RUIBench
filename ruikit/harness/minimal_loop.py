"""A real model drives a task through a profile's tools.

A thin loop: the model sees the task and the profile's tools, and every reply
it sends -- one or more tool calls, the independent ones run together -- is
one reply of the episode.  Results go back as text (the result's JSON) and
pictures, each picture shown at least once.  The history keeps every word and
the latest six pictures until the endpoint refuses a request as longer than the
model's context window; from then on it keeps the latest ``KEEP_CALLS`` tool
calls whole, an older call cut to its result's one line and its arguments'
short values, and the model is told so (``--on-overflow 0``: the episode ends
there instead; ``--keep-calls N``:
a window from the start).  The model is reached over a model channel (``ruikit.models``,
``ruikit/configs/models.yaml``).  ``ruikit episode`` runs it with the shared options; by hand:

    python -m ruikit.harness.minimal_loop --ping --model openai:<model>   # a hello
    python -m ruikit.harness.minimal_loop --task DS1 --profile rui.code/v1+aids --model openai:<model> \\
        --engine isaac --think-seconds 8 --log-dir runs/x

The clock: ``measured`` by default -- the world goes on for as long as the
model really took to answer, the gap between a result and the next call,
less the time spent waiting out the endpoint's retries (``waited_s`` in the
``.model.jsonl`` rows, of ``latency_s``);
``--think-seconds S`` charges S seconds a reply instead, the same for every
model (a fair, repeatable comparison).  The episode's limits apply as to any
agent (``Budget``: replies, simulated time).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional

from ruikit.harness.base import Harness
from ruikit.models.conversation import conversation
from ruikit.models.gateway import Gateway
from ruikit.serve.schemas import tool_entries

#: a reply without tool calls is answered this many times before the run ends
NUDGES = 2
#: the tool calls kept whole once the model's context window is full; older ones are
#: then cut to a line (``ruikit.models.conversation.ModelClient.on_overflow``).  The
#: request grows with each reply, and an episode repeating one call can pass the
#: context window before its time is up: without this, the endpoint refuses the
#: request and the episode has no result.  Until then every word stays, so an
#: episode that never fills the context runs word for word as without it.
KEEP_CALLS = 8


def instructions(ep, keep_calls: Optional[int] = None) -> str:
    """What the model is told once, before anything else."""
    inst = ep.inst
    arms = ", ".join(a.split("/")[-1] for a in ep.scene.arm_ids)
    return (f"You control a robot at a table in RUIBench task {inst.task_id} ({inst.level}), "
            f"through the tools of the interface {ep.profile.id} ({ep.profile.title}).\n"
            f"Task: {ep.instruction}\n"
            f"Hint: {inst.hint or '-'}\n"
            f"Arms: {arms}.\n"
            "Act only through the tools. One reply may hold several tool calls; independent "
            "ones run together (moves on different arms at the same time), and the reply comes "
            "back when every blocking one is done. The world keeps moving while you think, and "
            "each episode has limits on replies and on simulated time. Call task_info first for "
            "the station's conventions (frames, units, cameras), look before you move, and "
            "call task_done when the task is finished. Reaching the goal does not end the "
            "episode: task_done does, or a limit, and the episode is judged on the world as "
            "it is then." + _window(keep_calls) + _guide(ep))


def _window(keep_calls: Optional[int]) -> str:
    """What the model is told of a window kept from the start (``--keep-calls``)."""
    if keep_calls is None:
        return ""
    return (f" Only your latest {keep_calls} tool calls stay whole in this conversation: an "
            "older result keeps only ok, error, text and hint, and an older call's long "
            "arguments (a program, a note) are cut to a line. The text of your replies stays "
            "whole: write there what you will need again (positions, readings).")


def _guide(ep) -> str:
    """The interface's operation guide, if it carries one (``+guide``)."""
    if not ep.profile.guide:
        return ""
    from ruikit.harness.components.guides import guide_text

    return "\n\n" + guide_text(ep.profile.guide)


def _tools(ep) -> List[dict]:
    tools = [{"name": e["name"], "description": e["description"], "schema": e["schema"]}
             for e in tool_entries(ep.profile)]
    tools += [{"name": c.name, "description": c.description, "schema": c.input_schema()}
              for c in ep.custom_tools.values()]
    return tools


def _output(result) -> str:
    return json.dumps(result.as_json(), ensure_ascii=False, default=str)


def _pictures(result) -> List[tuple]:
    return [(f"[{im.frame_id}] {im.camera} {im.width}x{im.height}", im.png)
            for im in result.images if im.png]


class ModelAgent:
    """Run one episode with a model deciding every reply."""

    def __init__(self, model: str, base_url: Optional[str] = None, keep_images: int = 6,
                 max_output_tokens: int = 4096, transcript: Optional[str] = None,
                 keep_calls: Optional[int] = None,
                 on_overflow: Optional[int] = KEEP_CALLS) -> None:
        self.gateway = Gateway(model, base_url=base_url)
        self.keep_images = keep_images
        self.keep_calls = keep_calls
        self.on_overflow = on_overflow
        self.max_output_tokens = max_output_tokens
        self.transcript_path = Path(transcript) if transcript else None
        self.transcript: List[dict] = []
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.conv = None

    def run(self, ep):
        from ruibench.runner import heartbeat

        conv = self.conv = conversation(
            self.gateway, system=instructions(ep, self.keep_calls), tools=_tools(ep),
            keep_images=self.keep_images, max_output_tokens=self.max_output_tokens,
            keep_calls=self.keep_calls, on_overflow=self.on_overflow)
        conv.user("Begin.")
        quiet = 0
        try:
            while not ep.over:
                conv.tools = _tools(ep)          # a tool the model made joins the list
                # waiting on the model, not the simulator: a worker's watchdog is more patient
                heartbeat.beat(heartbeat.AGENT, replies=ep.replies)
                reply = conv.ask()
                ep.endpoint_wait(reply.waited_s)
                self._note(ep, reply)
                if not reply.calls:
                    ep.idle_reply()              # the world goes on while it writes, as for any reply
                    if ep.over:
                        break
                    if quiet >= NUDGES:
                        ep.finish("agent_stopped")
                        break
                    quiet += 1
                    conv.user("Keep going through the tools, or call task_done if the task "
                              "is finished.")
                    continue
                quiet = 0
                good = [c for c in reply.calls if not c.bad_arguments]
                results = ep.call_batch([(c.name, c.arguments) for c in good]) if good else []
                by_id = {c.id: r for c, r in zip(good, results)}
                outputs, pictures = [], []
                for c in reply.calls:
                    r = by_id.get(c.id)
                    if r is None:
                        outputs.append(json.dumps({"ok": False, "error": "bad_arguments",
                                                   "text": "the arguments were not a JSON object",
                                                   "got": c.bad_arguments[:300]}))
                        continue
                    outputs.append(_output(r))
                    pictures += _pictures(r)
                conv.results(reply.calls, outputs, pictures)
        finally:
            self._save()
        if not ep.over:
            ep.finish("agent_stopped")
        return ep.outcome

    def _note(self, ep, reply) -> None:
        for k in self.usage:
            alt = {"input_tokens": "prompt_tokens", "output_tokens": "completion_tokens"}[k]
            self.usage[k] += int(reply.usage.get(k) or reply.usage.get(alt) or 0)
        self.transcript.append({"t": round(ep.sim_time, 3), "latency_s": reply.latency_s,
                                "waited_s": reply.waited_s, "retries": reply.retries,
                                "text": reply.text,
                                **({"reasoning": reply.reasoning} if reply.reasoning else {}),
                                "calls": [{"name": c.name, "arguments": c.arguments} for c in reply.calls],
                                "usage": reply.usage, "stop": reply.stop})

    def _save(self) -> None:
        if self.transcript_path is None:
            return
        self.transcript_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.transcript_path, "w") as fh:
            # the history kept: a window from the start, or one turned on when the context
            # filled (at which reply, and with how many calls in the end)
            window = {"keep_images": self.keep_images, "keep_calls": self.keep_calls,
                      "on_overflow": self.on_overflow,
                      "window_from": self.conv.window_from if self.conv else None,
                      "window_keep": self.conv.keep_calls if self.conv else self.keep_calls}
            fh.write(json.dumps({"model": self.gateway.model, "usage": self.usage,
                                 **window}) + "\n")
            for row in self.transcript:
                fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


class MinimalLoop(Harness):
    """Our simple loop: the model's tool calls, run in this process, their
    results and pictures handed back, until the episode ends."""

    name = "loop"
    worker_module = "ruikit.experiments.batch"

    def run(self, job, out: Path, engine: str = "isaac", base_url: Optional[str] = None,
            **options) -> dict:
        from ruibench.engines import make_engine
        from ruikit.experiments.worker import run_job

        sim = make_engine(engine)
        try:
            return run_job(job, Path(out), sim, {"engine": engine, "base_url": base_url})
        finally:
            if engine != "isaac":          # Isaac's app is left to the process's exit
                sim.close()


def run_model(task: str, level: str, seed: int, profile: str, model: str, engine="mock",
              think_seconds: float = 0.0, log_dir: Optional[str] = None,
              base_url: Optional[str] = None, keep_calls: Optional[int] = None,
              on_overflow: Optional[int] = KEEP_CALLS) -> dict:
    from ruikit.env import Budget, Episode
    from ruikit.experiments.worker import EPISODE_BUDGET

    clock = (dict(clock="fixed", think_seconds=think_seconds, think_per="reply")
             if think_seconds else dict(clock="measured"))
    # the task's limits end it -- as in a batch, and as for the ready-made agents
    ep = Episode(task, level=level, seed=seed, profile=profile, engine=engine, log_dir=log_dir,
                 agent=f"{model} via the minimal loop", budget=Budget(**EPISODE_BUDGET),
                 **clock)
    transcript = None
    if log_dir:
        transcript = str(Path(log_dir) / f"{task}_{level}_s{seed}_{profile.replace('/', '-')}"
                                         f".model.jsonl")
    agent = ModelAgent(model, base_url=base_url, transcript=transcript, keep_calls=keep_calls,
                       on_overflow=on_overflow)
    t0 = time.time()
    try:
        out = agent.run(ep)
    finally:
        if engine == "isaac":
            # Isaac's shutdown can hold ~20 cores for half an hour: finish (that writes the
            # logs) and leave the app to the process's exit (``os._exit`` below)
            if not ep.over:
                ep.finish("closed")
        else:
            ep.close()
    return {"task": task, "level": level, "seed": seed, "profile": profile,
            "model": agent.gateway.model, "success": out.success, "reason": out.reason,
            "replies": out.metrics.get("replies"), "sim_seconds": out.metrics.get("sim_seconds"),
            "wall_seconds": round(time.time() - t0, 1), "usage": agent.usage,
            "think_seconds": think_seconds}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ping", action="store_true", help="list the channel's models, ask a hello, exit")
    ap.add_argument("--task", default="DS1")
    ap.add_argument("--level", default="easy")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="rui.pose/v1")
    ap.add_argument("--model", required=True,
                    help="<channel>:<name> of ruikit/configs/models.yaml: openai:..., anthropic:..., "
                         "local:... (a vLLM server)")
    ap.add_argument("--engine", default="mock")
    ap.add_argument("--think-seconds", type=float, default=0.0,
                    help="charge this much simulated time a reply (default: what the model took)")
    ap.add_argument("--log-dir", default=None)
    ap.add_argument("--on-overflow", type=int, default=KEEP_CALLS,
                    help=f"once the context window is full, the tool calls kept whole (default "
                         f"{KEEP_CALLS}); 0 ends the episode there instead")
    ap.add_argument("--keep-calls", type=int, default=0,
                    help="keep only this many tool calls whole from the start, and say so "
                         "(default 0: every word, until the context is full)")
    ap.add_argument("--base-url", default=None, help="the channel's endpoint, if not its own")
    args = ap.parse_args(argv)
    if args.ping:
        gw = Gateway(args.model, base_url=args.base_url)
        print(f"{gw.base_url}: " + ", ".join(gw.models()))
        print(f"{gw.model} says: {gw.ask('Reply with exactly: OK')}")
        return 0
    row = run_model(args.task, args.level, args.seed, args.profile, args.model, args.engine,
                    args.think_seconds, args.log_dir, args.base_url, args.keep_calls or None,
                    args.on_overflow or None)
    print(json.dumps(row, ensure_ascii=False))
    return 0 if row["success"] else 1


if __name__ == "__main__":
    import os
    import traceback

    from ruikit.core.result import ToolError

    code = 2
    try:
        code = main()
    except ToolError as exc:          # the channel, a key, a quota: said plainly
        print(f"{exc}" + (f"\n  -> {exc.hint}" if exc.hint else ""), file=sys.stderr)
    except Exception:
        traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    # the logs are written; an Isaac app left to shut down by itself can take many
    # minutes (the coverage workers and the session server leave the same way)
    os._exit(code)
