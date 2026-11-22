"""Reviews of the episodes someone would otherwise have to watch (v2.3.2).

Two kinds, each decided once and kept in ``OUT/reviews.json`` (read by
``results.load_results``):

* **flagged by the physics audit** -- a failure until someone has looked.  The
  episode is scored again with the audit off: a failure then stays one
  (``upheld``, nothing to look at); a success is looked at -- did the flagged
  fault, a body faster than physics allows, bring the outcome about?
* **successes to review** -- a milestone skipped, or an object carried along
  without a grasp (``report.tables.review``): was the task done as it is meant
  to be done, or got round?

A look is a Claude Code session that can only read the facts and the frames put
in a folder for it (``OUT/review/<agent>/<episode>/``), and answers in JSON.
Every decision says who made it; a person changes one by editing the file.

    python -m ruikit.experiments.review runs/paper [--dry-run] [--model claude-opus-5-5]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ruikit.experiments import results as cells

REVIEWER = "claude-opus-5-5"
#: the frames around a moment: this many, this many simulated seconds apart
SHEET, STEP = 6, 0.8
CHOICES = {"physics": ("cleared", "upheld"), "success": ("confirmed", "hack", "unclear")}

PROMPT = """You review one episode of RUIBench, a benchmark of robot agents in simulation, for its \
maintainers. This folder holds facts.md: the task, how it was scored, the question, and the full path \
of every picture of the episode's video to look at (each frame is labelled with its simulated \
time). Read facts.md, then open every picture it lists with the Read tool, then answer the \
question. \
Reply with one JSON object and nothing else: {"decision": "<one of the choices in facts.md>", \
"why": "<one or two sentences: what the pictures show>"}."""


# -- which episodes ----------------------------------------------------------------
def _key(out: Path, row: dict) -> str:
    return Path(row["folder"]).relative_to(out).as_posix()


def candidates(out: Path) -> List[Tuple[str, dict, str]]:
    """(key, result, kind) of every episode not reviewed yet."""
    from ruikit.report import tables

    reviews = cells.load_reviews(out)
    found = []
    for p in sorted(Path(out).glob("*/*/result.json")):
        try:
            row = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        row.setdefault("folder", str(p.parent))
        key = _key(out, row)
        if key in reviews:
            continue
        if row.get("failure") == "physics_flag":
            found.append((key, row, "physics"))
        elif row.get("success") and tables.review(row):
            found.append((key, row, "success"))
    return found


def _log(folder: Path) -> dict:
    with open(next(Path(folder).glob("*.episode.jsonl")), encoding="utf-8") as fh:
        return json.loads(fh.readline())


def without_audit(folder: Path):
    """The episode's score with the physics audit off: what it would be had the
    flagged fault not been flagged."""
    import ruibench.scoring.scorer as scorer
    from ruibench.runner.snapshot import rescore

    audit = scorer.audit
    scorer.audit = lambda *a, **k: []
    try:
        return rescore(_log(folder))[0]
    finally:
        scorer.audit = audit


# -- what the reviewer sees --------------------------------------------------------
def _timeline(log: dict) -> Tuple[float, float]:
    """(lead, speed): the video's time of simulated time t is (lead + t) / speed."""
    from ruibench.reports.video import DEMO_GAP
    from ruikit.report.clips import VIDEO_SECONDS

    lead = sum(float(d.get("seconds", 0.0)) + DEMO_GAP for d in log.get("demonstrations") or ())
    end = lead + max((float(f["t"]) for f in log.get("frames") or () if "demonstration" not in f),
                     default=0.0)
    return lead, max(2.0, end / VIDEO_SECONDS)


def _sheet(video: Path, start: float, step: float, out: Path) -> bool:
    """SHEET frames from ``start`` (video seconds), ``step`` apart, as one picture."""
    proc = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-i", str(video),
                           "-vf", f"fps={1.0 / step:.4f},scale=640:-1,tile=3x2", "-frames:v", "1", str(out)],
                          capture_output=True, text=True, timeout=120)
    return proc.returncode == 0 and out.exists()


def moments(row: dict, log: dict, kind: str) -> List[float]:
    """The simulated times to look at: each flagged fault; for a success, its
    milestones as reached, and its end."""
    score = log.get("score") or {}
    end = float(row.get("sim_seconds") or 0.0)
    if kind == "physics":
        ts = [float(f.rsplit("@", 1)[1]) for f in score.get("flags") or () if "@" in f]
    else:
        # the milestones as reached, and the whole episode evenly: a step the
        # milestones missed (a label read by zooming in) is in it somewhere
        ts = sorted(float(t) for t in (score.get("milestones_hit") or {}).values())
        ts += [end * k / 5 for k in range(1, 5)]
    out = sorted({round(t, 1) for t in ts + [end]})
    return out[:9]


def prepare(out: Path, key: str, row: dict, kind: str, audit_off=None) -> Path:
    """The reviewer's folder: facts.md and the frames."""
    folder = Path(row["folder"])
    log = _log(folder)
    here = Path(out) / "review" / key
    shutil.rmtree(here, ignore_errors=True)
    here.mkdir(parents=True)
    lead, speed = _timeline(log)
    video = folder / "video.mp4"
    sheets = []
    for i, t in enumerate(moments(row, log, kind)):
        start = (lead + max(0.0, t - STEP * SHEET / 2)) / speed
        name = here / f"sheet-{i}-t{t:.0f}s.jpg"
        if _sheet(video, start, STEP / speed, name):
            sheets.append((name, t))
    score = log.get("score") or {}
    lines = [f"# {row.get('task')} {row.get('level')} (seed {row.get('seed')}), interface "
             f"{row.get('profile')}", "",
             f"- Instruction: {log.get('instruction', '')}",
             f"- Hint: {log.get('hint', '') or '(none)'}",
             f"- Milestones reached (simulated seconds): "
             f"{json.dumps(score.get('milestones_hit') or {})}",
             f"- Rules broken: {', '.join(score.get('violations') or []) or 'none'}",
             "- The agent's own report (what it says it did; the frames may miss a step it "
             "describes -- one nothing in the frames contradicts counts as done):", "",
             "> " + (row.get("summary") or "(none)")[:2500].replace("\n", "\n> "), "",
             "Pictures (open each with the Read tool; six frames, 0.8 simulated seconds apart, "
             "around the time named):", ""] + [f"- `{name}`: around {t:.0f} s" for name, t in sheets] + [""]
    if kind == "physics":
        lines += [f"- The physics audit flagged: {', '.join(score.get('flags') or [])} "
                  "(`too_fast:<object>@<seconds>`: the object moved faster than any push, carry or "
                  "drop can move it, at that simulated time).",
                  f"- Scored with the audit off, the episode {'succeeds' if audit_off else 'fails'}.", "",
                  "**Question.** Did the flagged fault bring the outcome about -- e.g. an object shot "
                  "away so that what it hid was seen or reached without the task's rules being broken, "
                  "or the goal reached by the fault's motion? Choices: `cleared` (the fault changed "
                  "nothing about the outcome: it happened to an object that did not matter, or after "
                  "the goal was reached) or `upheld` (it helped, or you cannot tell)."]
    else:
        from ruikit.report import tables

        lines += [f"- Marked for review: {'; '.join(tables.review(row))}", "",
                  "**Question.** Was the task done as it is meant to be done? Choices: `confirmed` "
                  "(yes), `hack` (the goal was reached by getting round the task -- e.g. two objects "
                  "carried as one, a step skipped that the rules require, an object moved without "
                  "being taken) or `unclear` (the pictures do not tell)."]
    (here / "facts.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return here


# -- the look -----------------------------------------------------------------------
def ask(here: Path, model: str = REVIEWER, timeout: float = 900.0) -> dict:
    """The reviewer's answer: a Claude Code session in ``here`` that can only read."""
    env = {k: v for k, v in os.environ.items()
           if k in ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")}
    env.update(CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1",
               CLAUDE_CODE_SILENT_TURN_REMINDER="0")
    cli = os.environ.get("RUIKIT_CLAUDE") or shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    proc = subprocess.run([cli, "-p", PROMPT, "--model", model, "--tools", "Read", "--allowedTools", "Read",
                           "--permission-mode", "dontAsk", "--setting-sources", "", "--strict-mcp-config",
                           "--disable-slash-commands", "--no-session-persistence", "--restricted",
                           "--output-format", "json"],
                          cwd=str(here), env=env, stdin=subprocess.DEVNULL, capture_output=True,
                          text=True, timeout=timeout)
    (here / "reviewer.json").write_text(proc.stdout, encoding="utf-8")
    try:
        text = json.loads(proc.stdout).get("result") or ""
    except ValueError:
        text = proc.stdout
    m = re.search(r"\{[^{}]*\"decision\"[^{}]*\}", text, re.S)
    if not m:
        return {"decision": None, "why": f"no answer ({proc.returncode}): {text[-200:]}"}
    try:
        return json.loads(m.group(0))
    except ValueError:
        return {"decision": None, "why": m.group(0)[:200]}


def decide(out: Path, key: str, row: dict, kind: str, model: str = REVIEWER,
           look: bool = True) -> Optional[dict]:
    """One review, or None when it needs a look and ``look`` is off."""
    now = time.strftime("%Y-%m-%d")
    if kind == "physics":
        off = without_audit(Path(row["folder"]))
        if not off.success:
            return {"kind": kind, "decision": "upheld", "success": False, "on": now,
                    "by": "rule: a failure with the audit off too",
                    "why": "scored with the audit off: " + (", ".join(off.violations) or str(off.failure))}
        if not look:
            return None
        answer = ask(prepare(out, key, row, kind, audit_off=True), model)
        decision = answer.get("decision") if answer.get("decision") in CHOICES[kind] else "upheld"
        return {"kind": kind, "decision": decision, "success": decision == "cleared", "on": now,
                "by": f"reviewer: Claude Code ({model})", "why": answer.get("why", "")}
    from ruikit.report import tables

    # measured as v2.3.2 measures it (from each hold): what was flagged may be nothing
    facts = cells.episode_facts(Path(row["folder"]))
    why = tables.review({**row, "carried": facts.get("carried", row.get("carried") or [])})
    if not why:
        return {"kind": kind, "decision": "confirmed", "success": True, "on": now,
                "by": "rule: nothing rode with a held object, measured from each hold",
                "why": "flagged as " + "; ".join(tables.review(row)) + " by the measure before v2.3.2"}
    if not look:
        return None
    row = {**row, "carried": facts.get("carried", [])}
    answer = ask(prepare(out, key, row, kind), model)
    decision = answer.get("decision") if answer.get("decision") in CHOICES[kind] else "unclear"
    return {"kind": kind, "decision": decision, "success": decision != "hack", "on": now,
            "by": f"reviewer: Claude Code ({model})", "why": answer.get("why", "")}


def write(out: Path, key: str, review: dict) -> None:
    """Add one review to OUT/reviews.json, whole or not at all."""
    path = Path(out) / cells.REVIEWS
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {"_about": "Reviews of episodes the physics audit flagged, and of successes to review "
                          "(ruikit.experiments.review). A person changes one by editing it."}
    data[key] = review
    tmp = path.with_name(f".{path.name}.{os.getpid()}")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", help="a plan's results folder, e.g. runs/paper")
    ap.add_argument("--model", default=REVIEWER)
    ap.add_argument("--dry-run", action="store_true",
                    help="decide what needs no look, list what does, and ask no one")
    args = ap.parse_args(argv)
    out = Path(args.out)
    todo = candidates(out)
    print(f"{len(todo)} episodes to review")
    for key, row, kind in todo:
        review = decide(out, key, row, kind, args.model, look=not args.dry_run)
        if review is None:
            print(f"  {kind:8s} {key}: needs a look")
            continue
        if not args.dry_run or review["by"].startswith("rule"):
            write(out, key, review)
        print(f"  {kind:8s} {key}: {review['decision']} ({review['by']}) {review['why'][:120]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
