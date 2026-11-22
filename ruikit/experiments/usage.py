"""What a ready-made agent spent -- model calls and tokens -- from its own records.

    python -m ruikit.experiments.usage claude AGENT.jsonl       # a Claude Code transcript or -p output
    python -m ruikit.experiments.usage codex ROLLOUT.jsonl      # a Codex session record
    python -m ruikit.experiments.usage session ROUND/clips/RUN  # the hand-run rounds: find the
                                                           # subagent whose prompt names it

Claude Code: the stream-json it prints with ``-p`` (``ruikit.harness.session_agent``
keeps it as ``agent.jsonl``) ends with a ``result`` record of the run's
totals; a transcript (``~/.claude/projects/...``) logs each response once per
content block, so a response is counted once, by its id.  Codex: its session
record (``CODEX_HOME/sessions/.../rollout-*.jsonl``) keeps the thread's
running totals.  Input tokens count every token sent -- new, cache writes and
cache reads (the reads also apart).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional

from ruikit.core.paths import ROOT

#: where Claude Code keeps this repository's transcripts; a subagent's own is
#: <session>/subagents/[...]/agent-*.jsonl
TRANSCRIPTS = Path(os.environ.get("RUIKIT_CLAUDE_PROJECTS")
                   or Path.home() / ".claude" / "projects" / str(ROOT).replace("/", "-"))


def _lines(paths: Iterable[Path]):
    for p in paths:
        try:
            fh = open(p, errors="ignore")
        except OSError:
            continue
        with fh:
            for line in fh:
                try:
                    yield json.loads(line)
                except ValueError:
                    continue


def claude_usage(paths) -> Optional[dict]:
    """Model calls and tokens of a Claude Code run, from its transcript(s) or its
    ``-p --output-format stream-json`` output.  None when there is nothing."""
    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    responses, out, final, models = {}, {}, set(), {}
    result = None
    for d in _lines(paths):
        if d.get("type") == "result" and isinstance(d.get("usage"), dict):
            result = d
            continue
        m = d.get("message")
        if d.get("type") != "assistant" or not isinstance(m, dict) or not m.get("usage"):
            continue
        key = m.get("id") or d.get("requestId") or d.get("uuid")
        if key not in responses:
            models[m.get("model")] = models.get(m.get("model"), 0) + 1
        responses[key] = m["usage"]
        out[key] = max(out.get(key, 0), int(m["usage"].get("output_tokens") or 0))
        if m.get("stop_reason"):
            final.add(key)
    if not responses and result is None:
        return None
    fresh = sum(int(u.get("input_tokens") or 0) for u in responses.values())
    written = sum(int(u.get("cache_creation_input_tokens") or 0) for u in responses.values())
    read = sum(int(u.get("cache_read_input_tokens") or 0) for u in responses.values())
    row = {"model_calls": len(responses), "input_tokens": fresh + written + read,
           "cache_read_tokens": read, "output_tokens": sum(out.values()),
           "model": max(models, key=models.get) if models else None, "source": "claude"}
    if result is not None:                     # the run's own totals, when it printed them
        u = result["usage"]
        read = int(u.get("cache_read_input_tokens") or 0)
        row.update(input_tokens=int(u.get("input_tokens") or 0)
                   + int(u.get("cache_creation_input_tokens") or 0) + read,
                   cache_read_tokens=read, output_tokens=int(u.get("output_tokens") or 0),
                   usd=result.get("total_cost_usd"), turns=result.get("num_turns"))
        mu = result.get("modelUsage") or {}
        if mu and not row["model"]:
            row["model"] = max(mu, key=lambda k: (mu[k] or {}).get("outputTokens", 0))
    else:
        row["output_final"] = len(final)       # most blocks log the count from the stream's start
    return row


def claude_subagent_usage(folder, root: Path = TRANSCRIPTS) -> Optional[dict]:
    """A hand-run round's agent: the subagent whose
    prompt names this session folder, read as ``claude_usage`` reads any."""
    needle = json.dumps(f"D={Path(folder).resolve()}\n")[1:-1]
    found = []
    for f in sorted(Path(root).glob("*/subagents/**/agent-*.jsonl")):
        try:
            with open(f, errors="ignore") as fh:
                head = "".join(next(fh, "") for _ in range(8))
        except OSError:
            continue
        if needle in head:
            found.append(f)
    if not found:
        return None
    row = claude_usage(found)
    if row is not None:
        row["transcripts"] = [f.name for f in found]
    return row


def codex_usage(paths) -> Optional[dict]:
    """Model calls and tokens of a Codex run, from its session record: the
    thread's totals as of its last response."""
    paths = [Path(paths)] if isinstance(paths, (str, Path)) else [Path(p) for p in paths]
    calls, total, model, limits = 0, None, None, None
    for d in _lines(paths):
        p = d.get("payload") if isinstance(d.get("payload"), dict) else {}
        if d.get("type") == "token_usage_record":
            calls += 1
            total = p.get("thread_token_usage") or total
        elif d.get("type") == "turn_context" and p.get("model"):
            model = p["model"]
        elif d.get("type") == "event_msg" and p.get("type") == "token_count":
            limits = p.get("rate_limits") or limits
            if total is None and (p.get("info") or {}).get("total_token_usage"):
                total = p["info"]["total_token_usage"]
    if total is None:
        return None
    row = {"model_calls": calls, "input_tokens": int(total.get("input_tokens") or 0),
           "cache_read_tokens": int(total.get("cached_input_tokens") or 0),
           "output_tokens": int(total.get("output_tokens") or 0),
           "reasoning_tokens": int(total.get("reasoning_output_tokens") or 0),
           "model": model, "source": "codex"}
    if limits and limits.get("rate_limit_reached_type"):
        row["rate_limit_reached"] = limits["rate_limit_reached_type"]
    return row


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("kind", choices=["claude", "codex", "session"])
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args(argv)
    for p in args.paths:
        row = {"claude": claude_usage, "codex": codex_usage,
               "session": claude_subagent_usage}[args.kind](p)
        print(json.dumps({"path": p, **(row or {"usage": None})}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
