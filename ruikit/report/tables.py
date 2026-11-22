"""The experiments' tables, from the results in the shared layout (``cells``).

For each experiment of a plan (``ruikit.experiments.batch --plan``), from the cells
that experiment asks for -- whoever ran them:

- the summary: per agent and interface, the success rate with its Wilson 95%
  interval, the tokens, the time and the videos (also as a page,
  ``ruikit.report.results_page``);
- success per framework + model (rows) and interface (columns), with its
  Wilson 95% interval;
- the same by the five kinds of task and by level;
- what it cost: replies, simulated seconds, tokens;
- for the failures, the last milestone reached, task by task -- the evidence
  for telling a wrong decision from a right one not carried out;
- where the experiment holds the same interface with and without some aids,
  the paired difference: the same agent, task, level and seed, with all the
  aids and with one group taken away.

Each cell is counted once; a cell missing from the run is shown as missing,
never as a failure, and lost cells (the harness's fault) are left out.
"""

from __future__ import annotations

import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from ruikit.experiments import provenance
from ruikit.experiments import results as cells
from ruikit.experiments.plans import Experiment
from ruikit.experiments.results import COLUMNS, base_of

SUITE_NAMES = {"knowledge": "World Knowledge", "learning": "In-Context Learning",
               "discovery": "Discovery", "interaction": "Dynamic Interaction",
               "contact": "Contact-Rich"}


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """The Wilson score interval of k successes in n."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, mid - half), min(1.0, mid + half))


def rate(rows: Sequence[dict]) -> str:
    """'7/20 35% [18-57]' -- successes, rate and Wilson 95% interval."""
    n = len(rows)
    if not n:
        return "-"
    k = sum(bool(r.get("success")) for r in rows)
    lo, hi = wilson(k, n)
    return f"{k}/{n} {100 * k / n:.0f}% [{100 * lo:.0f}-{100 * hi:.0f}]"


def paired(pairs: Sequence[Tuple[bool, bool]], z: float = 1.96) -> Tuple[float, float, int, int]:
    """(difference, half-width, b, c) of paired successes (with, without): the
    difference in rate without minus with, its Wald interval for paired
    proportions, b = done only with, c = done only without."""
    n = len(pairs)
    if not n:
        return (0.0, 0.0, 0, 0)
    b = sum(1 for w, wo in pairs if w and not wo)
    c = sum(1 for w, wo in pairs if wo and not w)
    diff = (c - b) / n
    var = max(0.0, (b + c) - (b - c) ** 2 / n) / (n * n)
    return (diff, z * math.sqrt(var), b, c)


def _agent(r: dict) -> str:
    return r.get("agent") or f"{r.get('harness', 'loop')}-{r.get('model')}"


def _order(columns: Iterable[str]) -> List[str]:
    """The main table's order of columns (``cells.COLUMNS``); a column split by
    aids keeps its place, all the aids first, none last."""
    def rank(c: str):
        base, _, aids = c.partition(" ")
        within = (0 if aids in ("", "aids") else 2 if aids == "none" else 1, aids)
        return (COLUMNS.index(base) if base in COLUMNS else len(COLUMNS), base, within)
    return sorted(set(columns), key=rank)


def select(rows: Sequence[dict], cells: Iterable[tuple]) -> Tuple[List[dict], int]:
    """The results of ``cells`` (agent, task, level, seed, profile), and how many
    of them have none yet."""
    by = {(_agent(r), r["task"], r["level"], int(r["seed"]), r["profile"]): r for r in rows}
    got, missing = [], 0
    for key in cells:
        r = by.get(key)
        if r is None:
            missing += 1
        else:
            got.append(r)
    return got, missing


def column_key(rows: Sequence[dict]):
    """How a result is placed in a column: by its interface's column, and --
    where one column holds several aids settings, or the privileged control
    beside the main setting -- by those too, so no column mixes two conditions."""
    aids, perception = defaultdict(set), defaultdict(set)
    for r in rows:
        col = r.get("column") or r["profile"]
        aids[col].add(r.get("aids") or "none")
        perception[col].add(_privileged(r))

    def key(r: dict) -> str:
        col = r.get("column") or r["profile"]
        out = f"{col} {r.get('aids') or 'none'}" if len(aids[col]) > 1 else col
        return out + ("@privileged" if len(perception[col]) > 1 and _privileged(r) else "")
    return key


def _privileged(r: dict) -> bool:
    return "@privileged" in str(r.get("profile") or "")


def episode_minutes(r: dict) -> Optional[float]:
    """An episode's time, without starting the simulator: a ready-made agent's
    own run (``agent_seconds``), else the simple loop's episode (``wall_seconds``)."""
    t = r.get("agent_seconds") if r.get("agent_seconds") is not None else r.get("wall_seconds")
    return float(t) / 60.0 if t is not None else None


def has_video(r: dict) -> bool:
    return bool(r.get("video")) or (bool(r.get("folder"))
                                    and (Path(str(r["folder"])) / "video.mp4").exists())


def skipped(r: dict) -> List[str]:
    """The milestones a successful episode never reached, in the task's order: a
    success that skipped a step the task is built around (the coin never weighed,
    the doll never in hand) is worth a look at its video -- done another way, or
    the task hacked."""
    if not r.get("success") or not r.get("task"):
        return []
    try:
        order = cells.milestone_order(r["task"], r["level"], int(r["seed"]))
    except Exception:                         # a task this checkout does not have
        return []
    hit = set(r.get("milestones") or ())
    return [m for m in order if m not in hit]


def carried(r: dict) -> List[str]:
    """What a successful episode carried along without a grasp (``results.carried_along``)."""
    return list(r.get("carried") or []) if r.get("success") else []


def review(r: dict) -> List[str]:
    """Why a success is worth a look at its video: a milestone never reached, or
    something carried along with a held object -- none once it has been reviewed
    (``experiments.review``)."""
    if r.get("review"):
        return []
    out = [f"milestone not reached: {m}" for m in skipped(r)]
    out += [f"carried along without a grasp: {n}" for n in carried(r)]
    return out


def _tok(n: float) -> str:
    return (f"{n / 1e9:.2f}G" if n >= 1e9 else f"{n / 1e6:.2f}M" if n >= 1e6
            else f"{n / 1e3:.1f}k" if n >= 1e3 else f"{n:.0f}")


def summary_rows(rows: Sequence[dict]) -> List[dict]:
    """Per agent and interface, then per agent over all its interfaces: episodes,
    successes with the Wilson interval, tokens (median per episode and in all; the
    input counts cached input), time (median per episode and in all) and videos."""
    key = column_key(rows)
    by = defaultdict(list)
    for r in rows:
        by[(_agent(r), key(r))].append(r)
    out = []
    for agent in sorted({a for a, _ in by}):
        cols = _order({c for a, c in by if a == agent})
        for col in cols + [None]:
            group = by[(agent, col)] if col is not None else [r for r in rows if _agent(r) == agent]
            n = len(group)
            k = sum(bool(r.get("success")) for r in group)
            lo, hi = wilson(k, n)
            tin = [int((r.get("usage") or {}).get("input_tokens") or 0) for r in group]
            tout = [int((r.get("usage") or {}).get("output_tokens") or 0) for r in group]
            mins = [m for m in (episode_minutes(r) for r in group) if m is not None]
            out.append({"agent": agent, "interface": col if col is not None else "all",
                        "episodes": n, "successes": k, "rate": k / n if n else 0.0,
                        "to_review": sum(bool(review(r)) for r in group),
                        "lo": lo, "hi": hi,
                        "tokens_in_median": statistics.median(tin) if tin else 0,
                        "tokens_out_median": statistics.median(tout) if tout else 0,
                        "tokens_in": sum(tin), "tokens_out": sum(tout),
                        "minutes_median": statistics.median(mins) if mins else None,
                        "hours": sum(mins) / 60.0, "videos": sum(has_video(r) for r in group)})
    return out


def summary(rows: Sequence[dict]) -> List[str]:
    """The summary as a Markdown table."""
    out = ["| agent | interface | episodes | success [95% interval] | successes to review | "
           "tokens per episode (median, in / out) | tokens in all (in / out) | minutes per "
           "episode (median) | hours in all | videos |", "|---|---|---|---|---|---|---|---|---|---|"]
    for s in summary_rows(rows):
        bold = "**" if s["interface"] == "all" else ""
        minutes = f"{s['minutes_median']:.1f}" if s["minutes_median"] is not None else "-"
        out.append(f"| {s['agent']} | {bold}{s['interface']}{bold} | {s['episodes']} | "
                   f"{bold}{s['successes']}/{s['episodes']} {100 * s['rate']:.0f}%{bold} "
                   f"[{100 * s['lo']:.0f}-{100 * s['hi']:.0f}] | {s['to_review']} | "
                   f"{_tok(s['tokens_in_median'])} / {_tok(s['tokens_out_median'])} | "
                   f"{_tok(s['tokens_in'])} / {_tok(s['tokens_out'])} | {minutes} | "
                   f"{s['hours']:.1f} | {s['videos']}/{s['episodes']} |")
    return out


def grid(rows: Sequence[dict], key, title: str) -> List[str]:
    """Agents down, columns across, one cell each: ``rate``."""
    agents = sorted({_agent(r) for r in rows})
    cols = _order({key(r) for r in rows})
    out = [f"| {title} | " + " | ".join(cols) + " | all |",
           "|---|" + "---|" * (len(cols) + 1)]
    for a in agents:
        mine = [r for r in rows if _agent(r) == a]
        out.append(f"| {a} | " + " | ".join(rate([r for r in mine if key(r) == c]) for c in cols)
                   + f" | {rate(mine)} |")
    return out


def costs(rows: Sequence[dict]) -> List[str]:
    out = ["| agent | interface | episodes | median replies | median simulated s | "
           "tokens in / out (total) | median wall s |", "|---|---|---|---|---|---|---|"]
    by = defaultdict(list)
    key = column_key(rows)
    for r in rows:
        by[(_agent(r), key(r))].append(r)
    for (a, col), group in sorted(by.items()):
        med = lambda k: (f"{statistics.median([g[k] for g in group if g.get(k) is not None]):g}"
                         if any(g.get(k) is not None for g in group) else "-")
        tin = sum(int((g.get("usage") or {}).get("input_tokens") or 0) for g in group)
        tout = sum(int((g.get("usage") or {}).get("output_tokens") or 0) for g in group)
        out.append(f"| {a} | {col} | {len(group)} | {med('replies')} | {med('sim_seconds')} | "
                   f"{tin:,} / {tout:,} | {med('wall_seconds')} |")
    return out


def to_review(rows: Sequence[dict]) -> List[str]:
    """The successes worth a look, one row each, with why and their videos."""
    out = ["| agent | episode | why | video |", "|---|---|---|---|"]
    for r in rows:
        why = review(r)
        if why:
            folder = Path(str(r.get("folder") or ""))
            video = "/".join(folder.parts[-2:] + ("video.mp4",)) if has_video(r) else "-"
            out.append(f"| {_agent(r)} | {folder.name or r.get('task')} | {'; '.join(why)} | "
                       f"{video} |")
    return out if len(out) > 2 else []


def failures(rows: Sequence[dict]) -> List[str]:
    """Per task, how far the failed episodes got: the furthest milestone, counted."""
    out = ["| task | failed | furthest milestone reached (episodes) |", "|---|---|---|"]
    by = defaultdict(list)
    for r in rows:
        if not r.get("success"):
            by[r["task"]].append(r)
    for task in sorted(by):
        c = Counter(r.get("furthest_milestone") or r.get("last_milestone") or "none"
                    for r in by[task])
        out.append(f"| {task} | {len(by[task])} | "
                   + ", ".join(f"{m} ({n})" for m, n in c.most_common()) + " |")
    return out


def claims(rows: Sequence[dict]) -> List[str]:
    """Per agent and interface: how often the agent said it was done, and how
    often it was wrong -- failed though it said so (the cost of having no
    success check), with the 95% interval."""
    out = ["| agent | interface | episodes | declared done | failed though declared done |",
           "|---|---|---|---|---|"]
    by = defaultdict(list)
    key = column_key(rows)
    for r in rows:
        by[(_agent(r), key(r))].append(r)
    for (a, col), group in sorted(by.items()):
        said = [g for g in group if g.get("declared_done")]
        wrong = sum(1 for g in said if not g.get("success"))
        if said:
            lo, hi = wilson(wrong, len(said))
            share = f"{wrong}/{len(said)} = {100 * wrong / len(said):.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]"
        else:
            share = "-"
        out.append(f"| {a} | {col} | {len(group)} | {len(said)} | {share} |")
    return out


def ablation(rows: Sequence[dict]) -> List[str]:
    """The aids ablation: each interface with all the aids against the same with
    a group taken away (or none at all), paired by agent, task, level and seed."""
    full = {(_agent(r), base_of(r["profile"]), r["task"], r["level"], int(r["seed"])):
            bool(r.get("success")) for r in rows if r.get("aids") == "aids"}
    by = defaultdict(list)                   # (removed, column) -> pairs
    for r in rows:
        aids = r.get("aids") or "none"
        if aids == "aids" or not (aids == "none" or aids.startswith("aids-")):
            continue
        key = (_agent(r), base_of(r["profile"]), r["task"], r["level"], int(r["seed"]))
        if key in full:
            removed = "all" if aids == "none" else aids[len("aids-"):]
            by[(removed, r.get("column") or "")].append((full[key], bool(r.get("success"))))
    if not by:
        return []
    removed_order = sorted({k[0] for k in by}, key=lambda g: (g == "all", g))
    cols = _order({k[1] for k in by})
    out = ["| taken away | " + " | ".join(cols) + " | all |", "|---|" + "---|" * (len(cols) + 1)]
    for g in removed_order:
        cells = []
        for c in cols + [None]:
            pairs = [p for (gg, cc), ps in by.items() if gg == g and (c is None or cc == c)
                     for p in ps]
            if not pairs:
                cells.append("-")
                continue
            d, half, b, c_ = paired(pairs)
            cells.append(f"{100 * d:+.0f} ± {100 * half:.0f} pts (n={len(pairs)}; lost {b}, "
                         f"gained {c_})")
        out.append(f"| {g} | " + " | ".join(cells) + " |")
    return out


def report(name: str, rows: Sequence[dict], missing: int = 0, lost: int = 0,
           title: Optional[str] = None, settings: Optional[Dict[str, object]] = None) -> str:
    """One experiment's tables, as Markdown."""
    lines = [f"# {title or name}", ""]
    versions = Counter(r.get("version") or "?" for r in rows)
    lines.append(f"{len(rows)} results; {missing} cells not run yet; {lost} lost to the harness "
                 "(not counted). Benchmark version: "
                 + ", ".join(f"{v} ({n})" for v, n in versions.most_common()) + ".")
    if settings:
        lines.append("Settings: " + "; ".join(f"{k} {v}" for k, v in settings.items()) + ".")
    if len(versions) > 1:
        lines.append("**Mixed versions: these results do not all compare.**")
    if len({bool(r.get("filmed")) for r in rows}) > 1:
        lines.append("**Filmed and unfilmed episodes are mixed: filming changes the pictures "
                     "an agent is shown, so they do not compare.**")
    odd = not_comparable(rows)
    if odd:
        lines.append(f"**{len(odd)} of {len(rows)} results are not comparable**: see the last "
                     "section.")
    if not rows:
        return "\n".join(lines) + "\n"
    key = column_key(rows)
    lines += ["", "## Summary", "",
              "Success with its Wilson 95% interval; tokens as each agent's API counted them "
              "(the input includes cached input); time from the episode's start to its end, "
              "without starting the simulator; videos: episodes filmed (the page "
              "`index.html` plays them).", "", *summary(rows)]
    lines += ["", "## Success by interface", "", *grid(rows, key, "agent \\ interface")]
    lines += ["", "## By kind of task", ""]
    for suite, label in SUITE_NAMES.items():
        mine = [r for r in rows if r.get("suite") == suite]
        if mine:
            lines += [f"### {label}", "", *grid(mine, key, "agent \\ interface"), ""]
    lines += ["## By level", ""]
    for level in ("easy", "hard"):
        mine = [r for r in rows if r.get("level") == level]
        if mine:
            lines += [f"### {level}", "", *grid(mine, key, "agent \\ interface"), ""]
    aids_rows = [r for r in rows if r.get("aids") not in (None, "aids")]
    if aids_rows:
        table = ablation(rows)
        if table:
            lines += ["## Aids taken away (paired with all the aids)", "",
                      "Difference in success rate, in points (negative: worse without), with "
                      "its 95% interval; lost = done with all the aids only, gained = done "
                      "without only.", "", *table, ""]
    listed = to_review(rows)
    if listed:
        lines += ["## Successes to review", "",
                  "Successes that never reached a milestone of their task, or carried an object "
                  "along with a held one: watch the video -- done another way, or the task "
                  "hacked.", "", *listed, ""]
    lines += ["## Cost", "", *costs(rows), "", "## Declared done", "",
              "The agent said it was done (task_done) -- and how often it failed though it said "
              "so.", "", *claims(rows), "", "## How far the failures got", "", *failures(rows), ""]
    if odd:
        lines += ["## Not comparable", "",
                  f"Not on the frozen version {provenance.expected_version() or '(none here)'}, "
                  "not run in a fresh simulator, or without a record of either: counted in the "
                  "tables above, but not comparable with the rest.", "",
                  "| agent | episode | why |", "|---|---|---|"]
        lines += [f"| {r.get('agent', '?')} | {Path(str(r.get('folder', '?'))).name} | "
                  f"{'; '.join(why)} |" for r, why in odd]
        lines.append("")
    return "\n".join(lines) + "\n"


def not_comparable(rows: Sequence[dict]) -> List[Tuple[dict, List[str]]]:
    """The results that do not compare with the rest, each with why
    (``ruikit.experiments.provenance``), and those filmed where most are not, or
    the other way round: filming renders the cameras between the agent's looks,
    which changes the pictures it is shown."""
    expected = provenance.expected_version()
    filmed = Counter(bool(r.get("filmed")) for r in rows)
    usual = filmed.most_common(1)[0][0] if filmed else False
    out = []
    for r in rows:
        why = provenance.not_comparable(r, expected)
        if len(filmed) > 1 and bool(r.get("filmed")) != usual:
            why = why + ["filmed, where most are not" if r.get("filmed") else
                         "not filmed, where most are"]
        if why:
            out.append((r, why))
    return out


# --------------------------------------------------------------------------
# the report
# --------------------------------------------------------------------------
def write_reports(out: Path, experiments: Sequence[Experiment] = ()) -> str:
    """``cells.jsonl`` and ``report.md`` of everything in OUT, for each experiment
    of a plan its own tables (``report-<name>.md``), and the page that shows
    them with the videos (``index.html``, ``ruikit.report.results_page``)."""
    out = Path(out)
    rows = cells.load_results(out)
    cells.write_cells(out, rows)
    lost = {tuple(p.parent.relative_to(out).parts) for p in out.glob("*/*/lost.json")}
    pending = sum(1 for p in out.glob("*/*/api_error.json") if not (p.parent / "result.json").exists())
    text = report("All results", rows, lost=len(lost))
    text += f"\n{pending} episodes ended by the API or a usage limit, waiting to be run again.\n"
    (out / "report.md").write_text(text)
    for e in experiments:
        wanted = e.cells()
        got, missing = select(rows, wanted)
        lost_here = sum(1 for (a, t, lv, s, p) in wanted if (a, cells.stem(t, lv, s, p)) in lost)
        settings = {"levels": ", ".join(e.levels), "tasks": len(e.tasks),
                    "rows": "; ".join(f"{r.harness} {r.model} ({r.seeds}, {r.clock}"
                                      + (f", {e.owner(r)}" if e.owner(r) else "") + ")"
                                      for r in e.agents)}
        body = report(e.name, got, missing=missing - lost_here, lost=lost_here,
                             title=e.title, settings=settings)
        (out / f"report-{e.name}.md").write_text(body)
        text += f"\n{e.name}: {len(got)} of {len(wanted)} cells done -> report-{e.name}.md"
    from ruikit.report import results_page

    page = results_page.write(out, rows, experiments)
    text += f"\nthe page: {page} (success, tokens, time and videos, every episode)"
    return text
