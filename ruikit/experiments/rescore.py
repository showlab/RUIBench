"""Results already run, scored again with this code: a change to how tasks are scored (v2.3.4).

A change to a task's scorer alone -- not to what its agents are told or shown,
nor to what its reactors do -- needs no new episodes: the scorer reads only
the trajectory, and no agent sees its verdict.  So each stored episode of the
task is scored again from its log, and its ``result.json`` takes the new
verdict, the old one kept beside it (``scored_again``).  A result so scored
compares with the new version's (``provenance.SCORED_AGAIN``).

Run it in the checkout of the new version.  An episode is refused when its
task changed since its version in more than its scoring, or when its
instruction, hint or scene is not this code's (``recheck``'s *told* and
*shown*).

It also marks every earlier result, of any task, that a later version's change
touches though its task did not change (``provenance.EPISODE_CHANGES``): a hand
read within 0.6 degrees of pitch +-90, which v2.3.4 reads roll 0
(``read_otherwise``); a minimal-loop reply without a tool call on the fixed
clock, which costs its think time from v2.3.5 (``replied_without_a_call``); a
QwenCloud call refused for a list or number that came as text, which v2.4.1
reads (``typed_as_text``); a minimal-loop reply that brought more pictures than
the history keeps, the first dropped before the model saw them, which v2.4.2
shows (``pictures_unseen``).  Such
an episode would have gone otherwise and does not compare: run it again.

    ruikit rescore runs/paper --task CL5 WK6 CR3 DS1 [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import List, Optional, Sequence

from ruikit.experiments import provenance
from ruikit.experiments.recheck import SKIP_DIRS, _bodies, _header, _instance
from ruikit.experiments.results import milestone_order

#: what a result takes from its score
SCORED = ("success", "failure", "progress", "milestones", "last_milestone", "furthest_milestone")
#: a reading of the hand in a tool's result, and the pitch band v2.3.4 reads otherwise
READING = re.compile(r'"rpy_deg": \[\s*(-?[0-9.]+),\s*(-?[0-9.]+),\s*(-?[0-9.]+)\]')
LOCK_DEG = 90.0 - 89.427                   # |cos pitch| < GIMBAL_LOCK (0.01)


def result_folders(out: Path) -> List[Path]:
    """The folders under ``out`` that hold a result: ``OUT/<agent>/<episode>``."""
    return sorted(p.parent for p in Path(out).glob("*/*/result.json")
                  if p.parent.parent.name not in SKIP_DIRS)


def verdict(log: dict, task: str, level: str, seed: int) -> dict:
    """The fields a result takes from an episode's score, scored with this code."""
    from ruibench.runner.snapshot import rescore

    now = rescore(log)[0].as_dict()
    hit = now.get("milestones_hit") or {}
    order = [name for name, _ in sorted(hit.items(), key=lambda kv: kv[1])]
    steps = milestone_order(task, level, seed)
    last = order[-1] if order else None
    return {"success": bool(now.get("success")), "failure": now.get("failure"),
            "progress": now.get("progress"), "milestones": order, "last_milestone": last,
            "furthest_milestone": (next((m for m in reversed(steps) if m in hit), None)
                                   if steps else last)}


def refusal(row: dict, log: dict, expected: str) -> Optional[str]:
    """Why this episode cannot be scored again into a result of ``expected``."""
    frozen = (row.get("provenance") or {}).get("frozen")
    if frozen == expected:
        return f"already scored by {expected}"
    if (row.get("scored_again") or {}).get("frozen") == expected:
        return f"already scored again with {expected}"
    if row.get("task") not in provenance.scored_again_between(frozen, expected):
        return f"{row.get('task')} changed from {frozen} to {expected} in more than its scoring"
    inst = _instance(log)
    if ((log.get("instruction") or "") != (inst.instruction or "")
            or (log.get("hint") or "") != (inst.hint or "")):
        return "told something else"
    if _bodies(log.get("scene") or {}) != _bodies(inst.scene.summary()):
        return "shown something else"
    return None


def rescore_results(out, tasks: Sequence[str], dry_run: bool = False, say=print) -> List[dict]:
    """Score again every result of ``tasks`` under ``out``: one row per episode."""
    expected = provenance.expected_version()
    rows = []
    for folder in result_folders(Path(out)):
        path = folder / "result.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if row.get("task") not in tasks:
            continue
        logs = sorted(folder.glob("*.episode.jsonl"))
        if not logs:
            rows.append({"folder": str(folder), "task": row["task"], "refused": "no episode log"})
            continue
        log = _header(logs[0])
        why = refusal(row, log, expected)
        if why:
            rows.append({"folder": str(folder), "task": row["task"], "refused": why})
            continue
        before = {k: row.get(k) for k in SCORED}
        after = verdict(log, row["task"], row["level"], int(row["seed"]))
        rows.append({"folder": str(folder), "task": row["task"], "level": row["level"],
                     "column": row.get("column"), "was": before["success"], "now": after["success"]})
        if dry_run:
            continue
        row.update(after)
        row["scored_again"] = {"frozen": expected,
                               "from": (row.get("provenance") or {}).get("frozen"), "before": before}
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(row, ensure_ascii=False, default=str, indent=1))
        os.replace(tmp, path)
    return rows


def read_at_the_lock(folder: Path) -> bool:
    """Was the hand read within ``LOCK_DEG`` of pitch +-90 in this episode's
    tool results (the session's log, ``<episode>.jsonl``)?"""
    log = Path(folder) / f"{Path(folder).name}.jsonl"
    try:
        text = log.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False
    return any(abs(90.0 - abs(float(m.group(2)))) < LOCK_DEG for m in READING.finditer(text))


def replied_without_a_call(folder: Path, row: dict) -> bool:
    """A minimal-loop episode on the fixed clock with a reply that called no
    tool: a row of its ``.model.jsonl`` with no ``calls``."""
    if row.get("harness") != "loop" or not re.fullmatch(r"\d+(\.\d+)?s", str(row.get("clock"))):
        return False
    for log in Path(folder).glob("*.model.jsonl"):
        with open(log, encoding="utf-8", errors="replace") as fh:
            next(fh, None)                                  # the header: model and usage
            for line in fh:
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if isinstance(rec, dict) and "calls" in rec and not rec["calls"]:
                    return True
    return False


#: the tasks whose riding marks had colliders before v2.3.8: DS3's detector
#: lights, which gathered free-fall speed and shot away any shell they touched
#: -- an episode the physics audit flagged
MARKS_COLLIDED = frozenset({"DS3"})


def shot_by_a_mark(folder: Path, row: dict) -> bool:
    return row.get("task") in MARKS_COLLIDED and row.get("failure") == "physics_flag"


#: a tool's refusal of a value of the wrong kind (``core.params``), with the text it got
REFUSED_TEXT = re.compile(r"expected (a list|an object|a number|an integer|true or false), "
                          r"got '((?:[^'\\]|\\.)*)'")
#: what each refusal's kind would have taken, read from the text
KINDS = {"a list": lambda v: isinstance(v, list), "an object": lambda v: isinstance(v, dict),
         "a number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
         "an integer": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
         "true or false": lambda v: isinstance(v, bool)}


def typed_as_text(folder: Path, row: dict) -> bool:
    """A QwenCloud episode with a call refused for a value that came as text
    and spells one the parameter takes: v2.4.1 reads it, and the call goes on
    (``chat_completions.typed_arguments``)."""
    if (row.get("model_settings") or {}).get("channel") != "qwencloud":
        return False
    log = Path(folder) / f"{Path(folder).name}.jsonl"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in lines[1:]:                                  # the first is the header
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        for res in (rec.get("results") or []) if isinstance(rec, dict) else ():
            if not isinstance(res, dict) or res.get("ok") is not False:
                continue
            for m in REFUSED_TEXT.finditer(str(res.get("text") or "")):
                try:
                    value = json.loads(m.group(2).encode().decode("unicode_escape"))
                except ValueError:
                    continue
                if KINDS[m.group(1)](value):
                    return True
    return False


#: the pictures a minimal-loop history keeps, unless its ``.model.jsonl`` says otherwise
KEEP_IMAGES = 6


def pictures_unseen(folder: Path, row: dict) -> bool:
    """A minimal-loop episode with a reply that brought more pictures than its
    history keeps (``keep_images``, in its ``.model.jsonl``): before v2.4.2 the
    first of them were dropped before the model saw them.  A reply is one row
    of the session's log (``Episode.call_batch``), its pictures those its
    results carry with pixels."""
    if row.get("harness") != "loop":
        return False
    keep = KEEP_IMAGES
    for log in Path(folder).glob("*.model.jsonl"):
        with open(log, encoding="utf-8", errors="replace") as fh:
            try:
                keep = int(json.loads(next(fh, "{}")).get("keep_images") or keep)
            except (ValueError, TypeError, AttributeError):
                pass
    log = Path(folder) / f"{Path(folder).name}.jsonl"
    try:
        lines = log.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return False
    for line in lines[1:]:                                  # the first is the header
        try:
            rec = json.loads(line)
        except ValueError:
            continue
        if not isinstance(rec, dict):
            continue
        shown = sum(1 for res in rec.get("results") or () if isinstance(res, dict)
                    for im in res.get("images") or ()
                    if isinstance(im, dict) and "no pixels" not in str(im.get("note") or ""))
        if shown > keep:
            return True
    return False


#: how a change of ``provenance.EPISODE_CHANGES`` is told in an episode's folder
TOUCHED = {"read_otherwise": lambda folder, row: read_at_the_lock(folder),
           "replied_without_a_call": replied_without_a_call,
           "shot_by_a_mark": shot_by_a_mark,
           "typed_as_text": typed_as_text,
           "pictures_unseen": pictures_unseen}


def mark_episodes(out, dry_run: bool = False) -> List[dict]:
    """Mark each earlier result that a later version's change touches
    (``provenance.EPISODE_CHANGES``), in a task that compares otherwise -- not
    changed since, or changed in its scoring alone: it does not compare with
    the expected version's.  One row per result marked, with the keys marked."""
    expected = provenance.expected_version()
    rows = []
    for folder in result_folders(Path(out)):
        path = folder / "result.json"
        try:
            row = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        frozen = (row.get("provenance") or {}).get("frozen")
        if not frozen or frozen == expected:
            continue
        after = provenance.versions_after(frozen, expected)
        changed = provenance.changed_between(frozen, expected)
        if after is None or changed is None or (
                row.get("task") in changed
                and row.get("task") not in provenance.scored_again_between(frozen, expected)):
            continue                                        # it does not compare anyway
        keys = {key: version for version, key, _ in provenance.EPISODE_CHANGES
                if version in after and not row.get(key) and TOUCHED[key](folder, row)}
        if not keys:
            continue
        rows.append({"folder": str(folder), "task": row.get("task"), "column": row.get("column"),
                     "changes": sorted(keys)})
        if dry_run:
            continue
        row.update(keys)
        tmp = path.with_name(f".{path.name}.{os.getpid()}")
        tmp.write_text(json.dumps(row, ensure_ascii=False, default=str, indent=1))
        os.replace(tmp, path)
    return rows


def report(rows: Sequence[dict], dry_run: bool) -> str:
    done = [r for r in rows if "refused" not in r]
    flips = Counter((r["task"], r["was"], r["now"]) for r in done)
    lines = [f"{len(done)} results {'would be' if dry_run else 'were'} scored again, "
             f"{len(rows) - len(done)} refused."]
    for (task, was, now), n in sorted(flips.items()):
        lines.append(f"  {task}: {n} {'success' if was else 'failure'} -> {'success' if now else 'failure'}")
    for why, n in Counter(r["refused"] for r in rows if "refused" in r).most_common():
        lines.append(f"  refused, {why}: {n}")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="ruikit rescore", description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="a plan's results folder, e.g. runs/paper")
    ap.add_argument("--task", "--tasks", dest="tasks", nargs="+", required=True,
                    help="the tasks to score again")
    ap.add_argument("--dry-run", action="store_true", help="say what would change; write nothing")
    args = ap.parse_args(argv)
    rows = rescore_results(args.out, args.tasks, args.dry_run, say=lambda s: print(s, file=sys.stderr))
    print(report(rows, args.dry_run))
    marked = mark_episodes(args.out, args.dry_run)
    print(f"{len(marked)} results {'would be' if args.dry_run else 'were'} marked: a later "
          "version's change touches them (run them again)")
    for (key, task), n in sorted(Counter((k, r["task"]) for r in marked for k in r["changes"]).items()):
        print(f"  {key}, {task}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
