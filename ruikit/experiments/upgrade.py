"""A plan's results brought up to the version frozen now, spending an agent's
tokens only where they must be spent (v2.3.6).

After a change to the benchmark some results no longer compare with the version
frozen now (``provenance.not_comparable``): their task changed, a change touches
the episode (``provenance.EPISODE_CHANGES``), or their checkout differed from its
freeze.  The runner skips every cell that holds a result, so these are never run
again by themselves.  This command:

1. scores again, from their logs and with no new episode, the results of every
   task changed only in how it is scored (``rescore.rescore_results``);
2. marks the results a change touches though their task did not change
   (``rescore.mark_episodes``);
3. moves every result that still does not compare, with its review, to
   ``OUT/superseded/<its version>/`` -- so the runner runs exactly those again --
   and says what running them again cost before.

Every other result stays where it is and still counts.  Without ``--go`` it only
says what it would do.

    python -m ruikit.experiments.upgrade runs/paper          # what it would do
    python -m ruikit.experiments.upgrade runs/paper --go     # do it, then run the plan
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from ruikit.experiments import provenance
from ruikit.experiments.rescore import mark_episodes, report, rescore_results, result_folders

#: where the results that do not compare go, under the plan's folder
SUPERSEDED = "superseded"
REVIEWS = "reviews.json"


def _read(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def to_score_again(out: Path, expected: str) -> List[str]:
    """The tasks some result of which can be scored again into ``expected``."""
    tasks = set()
    for folder in result_folders(out):
        row = _read(folder / "result.json")
        frozen = (row.get("provenance") or {}).get("frozen")
        if (frozen and frozen != expected and row.get("task")
                in provenance.scored_again_between(frozen, expected)
                and (row.get("scored_again") or {}).get("frozen") != expected
                and row.get("harness") not in provenance.harnesses_changed(frozen, expected)):
            tasks.add(row["task"])
    return sorted(tasks)


def _usage(row: dict) -> Dict[str, int]:
    u = row.get("usage") or {}
    return {k: int(u.get(k) or 0) for k in ("input_tokens", "output_tokens",
                                             "cache_read_input_tokens", "cache_creation_input_tokens")}


def _move(src: Path, dest: Path) -> Path:
    """``src`` to ``dest``, or beside it with a number when that is taken."""
    target, k = dest, 1
    while target.exists():
        target, k = dest.with_name(f"{dest.name}.{k}"), k + 1
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(target))
    return target


def _move_reviews(out: Path, keys: Dict[str, str]) -> None:
    """Each moved episode's review (``OUT/reviews.json``, keyed by its folder under
    OUT) goes with it, to ``OUT/superseded/<version>/reviews.json``: a new result in
    the same folder must not inherit an old one's review."""
    path = out / REVIEWS
    reviews = _read(path)
    if not reviews:
        return
    by_version: Dict[str, dict] = defaultdict(dict)
    for key, version in keys.items():
        if key in reviews:
            by_version[version][key] = reviews.pop(key)
    for version, moved in by_version.items():
        dest = out / SUPERSEDED / version / REVIEWS
        old = _read(dest) or {"_about": reviews.get("_about")}
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps({**old, **moved}, ensure_ascii=False, indent=1))
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(reviews, ensure_ascii=False, indent=1))
    os.replace(tmp, path)


def upgrade(out, go: bool = False, say=print) -> dict:
    """Bring the results under ``out`` up to the version frozen now; with ``go``
    False, only say what would be done."""
    out = Path(out)
    expected = provenance.expected_version()
    tasks = to_score_again(out, expected)
    scored = rescore_results(out, tasks, dry_run=not go) if tasks else []
    scored_ok = {r["folder"] for r in scored if "refused" not in r}
    marked = mark_episodes(out, dry_run=not go)
    marked_at = {r["folder"]: r["changes"] for r in marked}

    keep, move = Counter(), []
    for folder in result_folders(out):
        row = _read(folder / "result.json")
        if not row:
            continue
        if not go:                                   # as the steps above would leave it
            if str(folder) in scored_ok:
                row = {**row, "scored_again": {"frozen": expected}}
            for key in marked_at.get(str(folder), ()):
                row = {**row, key: expected}
        why = provenance.not_comparable(row, expected)
        if why:
            move.append((folder, row, why))
        else:
            keep[folder.parent.name] += 1

    cost = Counter()
    for _, row, _ in move:
        cost.update(_usage(row))
    moved_keys: Dict[str, str] = {}
    if go:
        for folder, row, _ in move:
            version = (row.get("provenance") or {}).get("frozen") or "unknown"
            rel = folder.relative_to(out)
            _move(folder, out / SUPERSEDED / version / rel)
            review = out / "review" / rel
            if review.exists():
                _move(review, out / SUPERSEDED / version / "review" / rel)
            moved_keys[rel.as_posix()] = version
        _move_reviews(out, moved_keys)

    lines = [f"{'Upgraded' if go else 'Would upgrade'} {out} to {expected}."]
    if scored:
        lines.append(report(scored, not go))
    if marked:
        lines.append(f"{len(marked)} results {'were' if go else 'would be'} marked as touched "
                     "by a later change: "
                     + ", ".join(f"{k} {n}" for k, n in Counter(
                         c for r in marked for c in r["changes"]).most_common()))
    lines.append(f"{sum(keep.values())} results compare and stay"
                 + (f" ({', '.join(f'{a}: {n}' for a, n in sorted(keep.items()))})" if keep else ""))
    if move:
        by = Counter((f.parent.name, r.get("task")) for f, r, _ in move)
        reasons = Counter(w.split(",")[0] for _, _, why in move for w in why)
        lines.append(f"{len(move)} results do not compare and {'were' if go else 'would be'} moved "
                     f"to {out / SUPERSEDED}/<version>/, to be run again:")
        for (agent, task), n in sorted(by.items()):
            lines.append(f"  {agent} {task}: {n}")
        for why, n in reasons.most_common():
            lines.append(f"  because {why}: {n}")
        lines.append(f"  they cost {cost['input_tokens'] + cost['cache_read_input_tokens'] + cost['cache_creation_input_tokens']:,} "
                     f"input tokens (with the cache) and {cost['output_tokens']:,} output tokens as they ran")
    else:
        lines.append("Nothing to run again.")
    if not go and (scored_ok or marked or move):
        lines.append("Run it again with --go to do it; then run the plan as before.")
    say("\n".join(lines))
    return {"expected": expected, "scored": scored, "marked": marked, "kept": sum(keep.values()),
            "moved": [str(f) for f, _, _ in move], "cost": dict(cost)}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="a plan's results folder, e.g. runs/paper")
    ap.add_argument("--go", action="store_true", help="do it (without: say what it would do)")
    args = ap.parse_args(argv)
    upgrade(args.out, go=args.go)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
