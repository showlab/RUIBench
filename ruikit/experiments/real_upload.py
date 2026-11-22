"""A day's real-robot episodes, to the lab's gated record on the Hugging Face Hub.

    python -m ruikit.experiments.real_upload 1102-PickAndPlace --dry-run   # what would go
    python -m ruikit.experiments.real_upload 1102-PickAndPlace

The name says the day: ``MMDD-<what>``.  Every episode an operator judged that day goes to
``<name>/<agent>/<cell>/`` of the repository: its video, its result, the logs of its calls
and of its agent, the pictures the agent saw and the arm's flight.  Beside them is a
``details.md`` to read: the times, the tokens, the tools chosen, each call and its answer,
the code written and what the agent said.  The day's ``README.md`` is a table of its
episodes.  Uploading again replaces what is there.  The repository is gated: anyone sees its
card and its files' names, and only those let in by hand can read them.  One that is not
there yet is created private, and nothing goes to one open to anyone.
"""

from __future__ import annotations

import os

import argparse
import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

from ruikit.experiments.real import DEFAULT_OUT, agent_says

#: the Hugging Face dataset the rollouts go to (``RUIBENCH_REAL_DATASET``: <owner>/<name>)
REPO = os.environ.get("RUIBENCH_REAL_DATASET", "")
#: what an episode's folder holds that is no record of it: the session's token and port
LEFT_OUT = ("session.json", "running.json")
CARD = """---
viewer: false
---

# RUIBench on the real robot

Rollouts on the lab's Franka, one folder a day (`MMDD-<what>`). Each folder's `README.md` is a
table of its episodes; each episode has its `video.mp4`, a `details.md` to read, and its raw logs.
"""


def judged_on(row: Mapping) -> str:
    """The day an episode was judged, on this machine's clock: ``MMDD``; "" if it was not."""
    at = (row.get("verdict") or {}).get("at")
    return datetime.fromisoformat(at).astimezone().strftime("%m%d") if at else ""


def _result(folder: Path) -> dict:
    return json.loads((folder / "result.json").read_text(encoding="utf-8"))


def episodes(out: Path, day: str, tasks: Sequence[str] = ()) -> List[Path]:
    """The folders of the episodes judged on ``day``, of these tasks if any are named, in
    the order they were."""
    found = [p.parent for p in Path(out).glob("*/*/result.json") if judged_on(_result(p.parent)) == day
             and (not tasks or _result(p.parent)["task"] in tasks)]
    return sorted(found, key=lambda f: _result(f)["verdict"]["at"])


def files_of(folder: Path, out: Path) -> Dict[str, Path]:
    """What goes up of an episode, by its path there: all of its folder, and the flight
    of the session that ran it (named by its process)."""
    files = {p.relative_to(folder).as_posix(): p for p in sorted(folder.rglob("*"))
             if p.is_file() and p.name not in LEFT_OUT}
    session = json.loads((folder / "session.json").read_text(encoding="utf-8"))
    for flight in Path(out, "flight").glob(f"*-{session.get('pid')}.jsonl"):
        files["flight.jsonl"] = flight
    return files


def _answer(result: Mapping) -> str:
    text = " ".join(str(result.get("text") or "").split())
    return f"!! {result['error']}: {text}" if result.get("error") else f"-> {text}"


def _shown(args: Mapping) -> str:
    return json.dumps({k: v for k, v in args.items() if k != "source"}, ensure_ascii=False)


def _call(number: str, row: Mapping, tool: str, args: Mapping, result: Mapping) -> List[str]:
    """A call of the agent's, with the code it wrote, what that code printed and its steps."""
    out = [f"### {number} `{tool}`", "",
           f"{row['t0']:.1f}-{row['t1']:.1f} s of the episode, after {row['think_s']:.1f} s of "
           f"thinking. Arguments: `{_shown(args)}`", ""]
    if args.get("source"):
        out += ["```python", str(args["source"]).rstrip(), "```", ""]
    out += [_answer(result), ""]
    data = result.get("data") or {}
    for field in ("stdout", "result"):
        if data.get(field) and args.get("source"):
            out += [f"It {'printed' if field == 'stdout' else 'returned'}:", "", "```",
                    str(data[field]).rstrip(), "```", ""]
    for n, step in enumerate(row.get("inner") or (), 1):
        out.append(f"- {number}.{n} `{step['tool']}` `{_shown(step.get('args') or {})}` "
                   f"({step['t0']:.1f}-{step['t1']:.1f} s) {_answer(step.get('result') or {})}")
    return out + ([""] if row.get("inner") else [])


def _tools_chosen(rows: List[dict]) -> Counter:
    return Counter(step["tool"] for row in rows
                   for step in [{"tool": row["tool"]}, *(row.get("inner") or ())])


def details(folder: Path) -> str:
    """An episode, to read."""
    row = _result(folder)
    calls = [json.loads(line) for line in
             (folder / f"{folder.name}.jsonl").read_text(encoding="utf-8").splitlines()]
    calls = [c for c in calls if "tool" in c]
    usage, verdict = row.get("usage") or {}, row.get("verdict") or {}
    began = datetime.fromtimestamp((folder / "prompt.txt").stat().st_mtime)
    tools = ", ".join(f"`{name}` x{n}" for name, n in _tools_chosen(calls).most_common())
    out = [f"# {row['task']} seed {row.get('seed', 0)} through {row['profile']}", "",
           f"**{'Success' if row.get('success') else 'Failure'}**, judged by the operator: "
           f"{verdict.get('criterion', '')}.", "",
           f"Operator's note: {verdict.get('note') or '(none)'}", "",
           f"Laid out: {row.get('layout') or '(as the card says)'}", "",
           "[The video](video.mp4)", "",
           "| | |", "|---|---|",
           f"| Agent | {row['harness']} {row['agent_cli']}, {row['model']}, {row['effort']} effort |",
           f"| Started | {began:%Y-%m-%d %H:%M:%S} |",
           f"| Episode's clock | {row['sim_seconds']:.1f} s |",
           f"| Wall clock, session and agent | {row['wall_seconds']:.0f} s, {row['agent_seconds']:.0f} s |",
           f"| Replies and tool calls | {row['replies']}, {row['tool_calls']} |",
           f"| Input tokens (of them read from the cache) | {usage.get('input_tokens', 0):,} "
           f"({usage.get('cache_read_tokens', 0):,}) |",
           f"| Output tokens | {usage.get('output_tokens', 0):,} |",
           f"| Cost | ${usage.get('usd', 0):.2f} |",
           f"| Ended by | {row['reason']} |",
           f"| Code | {row['provenance']['commit'][:7]}, {row['provenance']['frozen']} |",
           f"| Tools chosen | {tools} |", "",
           "## The agent's summary", "", str(row.get("summary") or "(none)"), "",
           "## The calls", ""]
    for n, call in enumerate(calls, 1):
        results = call.get("results") or [{}]
        if call["tool"] == "batch":
            for m, (inside, result) in enumerate(zip(call["args"]["calls"], results), 1):
                out += _call(f"#{n}-{m}", call, inside["tool"], inside.get("args") or {}, result)
        else:
            out += _call(f"#{n}", call, call["tool"], call.get("args") or {}, results[0])
    out += ["## What the agent said", ""]
    record = folder / "agent.jsonl"
    said = [agent_says(line) for line in record.read_text(errors="replace").splitlines()]
    out += [f"- {' '.join(words.split())}" for words in said if words.strip()]
    return "\n".join(out) + "\n"


def day_page(name: str, folders: List[Path]) -> str:
    """A day's episodes, as a table."""
    from ruibench.tasks.real import get

    rows = [_result(f) for f in folders]
    tasks = {r["task"] for r in rows}
    out = [f"# {name}", ""]
    for task in sorted(tasks):
        card = get(task)
        out += [f"**{task}, {card.name}.** The agent is told: {card.instruction}", "",
                f"Success: {card.success}", ""]
    tables = {(r["task"], r.get("seed", 0)): r.get("layout") or get(r["task"]).setup for r in rows}
    for (task, seed), layout in sorted(tables.items()):
        out += [f"Laid out, {task} seed {seed}: {layout}", ""]
    wins = sum(1 for r in rows if r.get("success"))
    out += [f"{wins} of {len(rows)} episodes succeeded, as judged by the operator.", "",
            "| Agent | Seed | Interface | Result | Clock (s) | Replies | Calls | Input tokens"
            " | Output tokens | Cost | Video | Details | Note |",
            "|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|---|---|"]
    for folder, r in zip(folders, rows):
        usage, where = r.get("usage") or {}, f"{folder.parent.name}/{folder.name}".replace("+", "%2B")
        where = where.replace("@", "%40")
        out.append(f"| {r['harness']} {r['model']} | {r.get('seed', 0)} | {r['profile']} "
                   f"| {'success' if r.get('success') else 'failure'} | {r['sim_seconds']:.1f} "
                   f"| {r['replies']} | {r['tool_calls']} | {usage.get('input_tokens', 0):,} "
                   f"| {usage.get('output_tokens', 0):,} | ${usage.get('usd', 0):.2f} "
                   f"| [video]({where}/video.mp4) | [details]({where}/details.md) "
                   f"| {(r.get('verdict') or {}).get('note') or ''} |")
    return "\n".join(out) + "\n"


def to_upload(name: str, out: Path, folders: List[Path]) -> Dict[str, Union[Path, bytes]]:
    """What goes to the day's folder, by its path there."""
    uploads: Dict[str, Union[Path, bytes]] = {f"{name}/README.md": day_page(name, folders).encode()}
    for folder in folders:
        at = f"{name}/{folder.parent.name}/{folder.name}"
        uploads[f"{at}/details.md"] = details(folder).encode()
        uploads.update({f"{at}/{rel}": path for rel, path in files_of(folder, out).items()})
    return uploads


def upload(uploads: Mapping[str, Union[Path, bytes]], message: str, repo: str = REPO, api=None) -> str:
    """Into the repository, created private if it is not there; its address."""
    if not repo:
        raise ValueError("which Hugging Face dataset? set RUIBENCH_REAL_DATASET=<owner>/<name>")
    from huggingface_hub import CommitOperationAdd, HfApi

    api = api or HfApi()
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    info = api.repo_info(repo, repo_type="dataset")
    if not (info.private or info.gated):
        raise SystemExit(f"{repo} is open to anyone: nothing goes there")
    operations = [CommitOperationAdd(path, source if isinstance(source, bytes) else str(source))
                  for path, source in uploads.items()]
    if not api.file_exists(repo, "README.md", repo_type="dataset"):
        operations.append(CommitOperationAdd("README.md", CARD.encode()))
    api.create_commit(repo, operations, commit_message=message, repo_type="dataset")
    return f"https://huggingface.co/datasets/{repo}"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("name", help="the day's folder: MMDD-<what>, as 1102-PickAndPlace")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help="where the episodes are")
    ap.add_argument("--task", "--tasks", dest="tasks", nargs="*", default=[],
                    help="only these tasks: a suite's own folder, when a day holds more than one")
    ap.add_argument("--dry-run", action="store_true", help="say what would go, and stop")
    args = ap.parse_args(argv)
    if not re.fullmatch(r"\d{4}-[\w.-]+", args.name):
        ap.error(f"{args.name!r} is no MMDD-<what>")
    out = Path(args.out)
    folders = episodes(out, args.name[:4], args.tasks)
    if not folders:
        print(f"no episode in {out} was judged on {args.name[:4]}")
        return 1
    uploads = to_upload(args.name, out, folders)
    size = sum(len(s) if isinstance(s, bytes) else s.stat().st_size for s in uploads.values())
    print(f"{args.name}: {len(folders)} episodes, {len(uploads)} files, {size / 1e6:.0f} MB, to {REPO}")
    if args.dry_run:
        return 0
    print(upload(uploads, f"{args.name}: {len(folders)} episodes"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
