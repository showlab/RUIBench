"""What a change to the benchmark does to the rollouts already run (v2.3.2).

Run it in a working copy of the change, on the folders of results it would
touch.  It reads every stored episode log (``*.episode.jsonl``) and reports, for
each episode:

* **judged**: the episode scored again with this code -- success, rules broken,
  milestones.  A hack the change closes now fails; an honest episode still
  succeeds.
* **told**: the instruction or hint at the episode's task, level and seed is not
  what its agent was told.
* **shown**: its scene is not what its agent saw -- by the scene kept in the log,
  and with ``--base`` (the code the logs were run on) body by body, part by part:
  a card redrawn moves no body and recolours none.

An episode told or shown something else cannot be scored again into a result:
it is run again.  A change is ready to freeze when what it flips is what it was
meant to flip, and what it makes run again is known.  The reactors' events are
the ones logged: a change to what a reactor does shows only in new episodes.

    python -m ruikit.experiments.recheck runs/paper runs/other/logs \\
        [--base runs/kitNN/BASE] [--tasks CL3 WK1] [--out recheck.md]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ruikit.core.paths import ROOT

#: the folders whose logs are not results: kept for the record only
SKIP_DIRS = ("discarded", "superseded", "selftest", ".workers", "frames", "images")

#: a scene, body by body and part by part, as one digest -- the same code run
#: in this checkout and in ``--base``
_DIGEST = r'''
import hashlib, json, sys
from ruibench.tasks.instance import build
out = {}
for task, level, seed in json.load(sys.stdin):
    try:
        s = build(task, level, int(seed)).scene
    except Exception as exc:                       # a task the other code does not have
        out[f"{task}|{level}|{seed}"] = f"error: {type(exc).__name__}"
        continue
    text = repr((sorted(repr(b) for b in s.bodies), sorted(repr(m) for m in s.markers),
                 sorted(repr(a) for a in s.articulations)))
    out[f"{task}|{level}|{seed}"] = hashlib.sha256(text.encode()).hexdigest()
print(json.dumps(out))
'''


def episode_logs(paths: Sequence[str]) -> List[Path]:
    out = []
    for p in paths:
        p = Path(p)
        found = [p] if p.is_file() else sorted(p.rglob("*.episode.jsonl"))
        out += [f for f in found if f == p or not any(part in SKIP_DIRS for part in f.relative_to(p).parts)]
    return out


def _header(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.loads(fh.readline())


def _instance(log: dict):
    from ruibench.core.config import RunConfig
    from ruibench.tasks.instance import build

    names = {f.name for f in dataclasses.fields(RunConfig)}
    cfg = RunConfig(**{k: v for k, v in dict(log.get("config") or {}).items()
                       if k in names and k != "budget"})
    return build(log["task_id"], log["level"], int(log["seed"]), cfg)


def _bodies(summary: dict) -> str:
    return json.dumps(summary.get("bodies") or [], sort_keys=True)


def digests(cells: Iterable[Tuple[str, str, int]], code: Path) -> Dict[str, str]:
    """The scene digest of each (task, level, seed) with the code at ``code``."""
    env = dict(os.environ, PYTHONPATH=str(code))
    proc = subprocess.run([sys.executable, "-c", _DIGEST], input=json.dumps(sorted(set(cells))),
                          capture_output=True, text=True, env=env, cwd=str(code), timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(f"digests with {code}: {proc.stderr.strip()[-400:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def recheck(paths: Sequence[str], tasks: Sequence[str] = (), base: Optional[str] = None,
            say=print) -> List[dict]:
    """One row per episode log: what this code changes about it."""
    from ruibench.runner.snapshot import rescore

    rows, built = [], {}
    logs = episode_logs(paths)
    say(f"{len(logs)} episode logs")
    for i, path in enumerate(logs, 1):
        try:
            log = _header(path)
        except (OSError, ValueError):
            continue
        task, level, seed = log.get("task_id"), log.get("level"), log.get("seed")
        if not task or (tasks and task not in tasks):
            continue
        key = (task, level, int(seed), json.dumps(log.get("config") or {}, sort_keys=True, default=str))
        if key not in built:
            built[key] = _instance(log)
        inst = built[key]
        told = (log.get("instruction") or "") != (inst.instruction or "") \
            or (log.get("hint") or "") != (inst.hint or "")
        shown = _bodies(log.get("scene") or {}) != _bodies(inst.scene.summary())
        old = log.get("score") or {}
        try:
            new, _ = rescore(log)
            now = new.as_dict()
        except Exception as exc:                  # noqa: BLE001 -- a log this code cannot read
            now = {"error": f"{type(exc).__name__}: {str(exc)[:120]}"}
        rows.append({"log": str(path), "task": task, "level": level, "seed": int(seed),
                     "was": old.get("success"), "now": now.get("success"),
                     "violations_was": old.get("violations") or [],
                     "violations_now": now.get("violations") or [],
                     "milestones_was": sorted((old.get("milestones_hit") or {})),
                     "milestones_now": sorted((now.get("milestones_hit") or {})),
                     "told": told, "shown": shown, "error": now.get("error")})
        if i % 50 == 0:
            say(f"  {i} of {len(logs)}")
    if base:
        cells = [(r["task"], r["level"], r["seed"]) for r in rows]
        before, after = digests(cells, Path(base)), digests(cells, ROOT)
        for r in rows:
            k = f"{r['task']}|{r['level']}|{r['seed']}"
            r["shown_parts"] = before.get(k) != after.get(k)
    return rows


def report(rows: Sequence[dict], base: Optional[str] = None) -> str:
    """The rows as Markdown: per task, then each episode the change touches."""
    by = defaultdict(list)
    for r in rows:
        by[r["task"]].append(r)
    lines = ["# What the change does to the rollouts already run", "",
             f"{len(rows)} episodes. *judged*: scored again with this code; *told*: instruction or "
             "hint changed; *shown*: scene changed" + (f" (parts compared with `{base}`)" if base else
                                                       " (by the scene kept in each log)") +
             ". An episode told or shown something else is run again, not scored again.", "",
             "| task | episodes | success → failure | failure → success | rules or milestones changed "
             "| told | shown | run again |", "|---|---|---|---|---|---|---|---|"]
    touched = []
    for task in sorted(by):
        rs = by[task]
        sf = [r for r in rs if r["was"] and r["now"] is False]
        fs = [r for r in rs if r["was"] is False and r["now"]]
        other = [r for r in rs if r not in sf and r not in fs and (
            r["violations_was"] != r["violations_now"] or r["milestones_was"] != r["milestones_now"])]
        told = [r for r in rs if r["told"]]
        shown = [r for r in rs if r["shown"] or r.get("shown_parts")]
        again = {r["log"] for r in told + shown}
        lines.append(f"| {task} | {len(rs)} | {len(sf)} | {len(fs)} | {len(other)} | {len(told)} | "
                     f"{len(shown)} | {len(again)} |")
        touched += [(task, "success → failure", r) for r in sf] + [(task, "failure → success", r) for r in fs]
        touched += [(task, "rules or milestones", r) for r in other]
    errors = [r for r in rows if r.get("error")]
    if touched:
        lines += ["", "## Episodes whose score changes", "", "| task | change | episode | rules (was → now) |",
                  "|---|---|---|---|"]
        for task, change, r in touched:
            lines.append(f"| {task} | {change} | `{_short(r['log'])}` | "
                         f"{', '.join(r['violations_was']) or '-'} → {', '.join(r['violations_now']) or '-'} |")
    if errors:
        lines += ["", f"## Logs this code could not score ({len(errors)})", ""]
        lines += [f"- `{_short(r['log'])}`: {r['error']}" for r in errors[:20]]
    return "\n".join(lines) + "\n"


def _short(path: str) -> str:
    p = Path(path)
    return str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("paths", nargs="+", help="folders of results or of episode logs")
    ap.add_argument("--tasks", nargs="*", default=(), help="only these tasks")
    ap.add_argument("--base", help="the code the logs were run on (a checkout), to compare scenes "
                                   "part by part")
    ap.add_argument("--out", help="write the report here (default: print it)")
    args = ap.parse_args(argv)
    rows = recheck(args.paths, args.tasks, args.base, say=lambda s: print(s, file=sys.stderr))
    text = report(rows, args.base)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(args.out)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
