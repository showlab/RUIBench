"""A single self-contained HTML report.

    ruibench report runs/dev.jsonl -o runs/report.html

No build step, no CDN, no dependencies: one file you can email, attach to a PR,
or open on a cluster node.  The point of it is that a result should be
*inspectable* -- every number links back to the episodes it came from, and
anything that disqualifies the run from being a leaderboard says so at the top
instead of in a footnote.
"""

from __future__ import annotations

import html as _html
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Mapping, Optional, Sequence

from ..tasks import SUITES, SUITE_ORDER, get as get_task
from . import svg
from .metrics import Results, summary_table

_CSS = """
* { box-sizing: border-box; }
body { margin:0; background:var(--ground); color:var(--ink);
  font:400 15px/1.6 var(--font-ui); -webkit-font-smoothing:antialiased; }
.wrap { max-width:1060px; margin:0 auto; padding-block:40px 80px; padding-inline:20px; }
h1 { font-family:var(--font-label); font-size:28px; font-weight:700; margin:0 0 4px;
  letter-spacing:0.005em; text-wrap:balance; }
h2 { font-family:var(--font-label); font-size:19px; font-weight:700; margin:44px 0 8px;
  letter-spacing:0.01em; }
h3 { font-family:var(--font-label); font-size:13px; margin:0 0 6px; color:var(--muted);
  font-weight:600; text-transform:uppercase; letter-spacing:0.07em; }
p  { margin:0 0 12px; max-width:68ch; }
.sub { color:var(--muted); margin-bottom:26px; font-family:var(--font-mono); font-size:13px; }
.card { background:var(--surface); border:1px solid var(--rule); border-radius:8px;
  padding:20px 22px; margin:14px 0; overflow-x:auto; }
.note { font-size:13px; color:var(--muted); max-width:72ch; }
.blocker { background:var(--fail-soft); border-radius:8px; padding:14px 18px; margin:18px 0; }
.blocker b { color:var(--fail); }
.ok { background:var(--ok-soft); border-radius:8px; padding:14px 18px; margin:18px 0; }
.ok b { color:var(--ok); }
table { border-collapse:collapse; width:100%; font-size:13.5px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--rule); }
th { font-family:var(--font-label); font-weight:600; color:var(--muted); font-size:12px;
  text-transform:uppercase; letter-spacing:0.06em; }
td.num { text-align:right; font-family:var(--font-mono); font-variant-numeric:tabular-nums; }
.grid { display:grid; gap:16px; grid-template-columns:repeat(auto-fit,minmax(330px,1fr)); }
.kv { display:flex; gap:10px; font-size:13px; color:var(--muted); flex-wrap:wrap; }
.kv span { background:var(--surface-2); border-radius:20px; padding:2px 11px; }
code { font-family:var(--font-mono); background:var(--surface-2); padding:1px 5px;
  border-radius:4px; font-size:13px; }
@media (max-width:600px){ .wrap{padding-block:24px 60px; padding-inline:16px;} h1{font-size:22px;} }
"""


def _e(x: object) -> str:
    return _html.escape(str(x))


def render(logs: Sequence[Mapping], title: str = "RUIBench results",
           run_id: str = "") -> str:
    results = Results.from_logs(logs)
    blockers = results.leaderboard_blockers()
    parts: List[str] = []

    parts.append(f"<h1>{_e(title)}</h1>")
    parts.append(f'<p class="sub">{len(logs)} episodes · '
                 f'{_e(run_id or (logs[0].get("run_id") if logs else "-"))} · '
                 f'generated {datetime.now():%Y-%m-%d %H:%M}</p>')

    # -- honesty banner ----------------------------------------------------
    if blockers:
        parts.append('<div class="blocker"><b>Not a leaderboard.</b><ul>'
                     + "".join(f"<li>{_e(b)}</li>" for b in blockers)
                     + "</ul><p class='note'>These episodes are useful for "
                       "development, scorer checks and negative controls. They are "
                       "not comparable results.</p></div>")
    else:
        parts.append('<div class="ok"><b>Results-grade run.</b> Photorealistic engine, '
                     'benchmark interfaces only.</div>')

    parts.append('<div class="kv">'
                 + "".join(f"<span>{_e(k)}: {v}</span>" for k, v in results.engines.items())
                 + "".join(f"<span>{_e(k)}: {v}</span>" for k, v in results.conditions.items())
                 + "</div>")

    # -- main figure -------------------------------------------------------
    parts.append("<h2>Capability suite × interface</h2>")
    parts.append("<p>Success rate per cell, judged from the physical trajectory. "
                 "No cell is labelled a winner: at these sample sizes the "
                 "best-minus-worst gap is inflated by noise, so read the paired "
                 "differences below for any comparison.</p>")
    hm = results.heatmap()
    ifaces = results.interfaces
    values = {(s, i): (c.rate.value if c.total else None)
              for s, per in hm.items() for i, c in per.items()}
    counts = {(s, i): c.total for s, per in hm.items() for i, c in per.items()}
    labels = {s: SUITES[s][1] for s in SUITE_ORDER}
    present = [s for s in SUITE_ORDER if any(counts.get((s, i), 0) for i in ifaces)]
    if present and ifaces:
        parts.append('<div class="card">'
                     + svg.heatmap(present, ifaces, values, counts, labels) + "</div>")

    # -- main table --------------------------------------------------------
    rows = summary_table(results)
    if rows:
        parts.append("<h2>Main table</h2>")
        body = "".join(
            f"<tr><td>{_e(r['suite_name'])}</td><td>{_e(r['interface'])}</td>"
            f"<td class='num'>{r['success_rate']:.0%}</td>"
            f"<td class='num'>{r['ci_low']:.0%}–{r['ci_high']:.0%}</td>"
            f"<td class='num'>{r['n']}</td>"
            f"<td class='num'>{r['progress']:.2f}</td>"
            f"<td class='num'>${r['usd_per_episode']:.3f}</td>"
            f"<td class='num'>{r['coverage_gaps'] or ''}</td>"
            f"<td class='num'>{r.get('harness_errors') or ''}</td></tr>"
            for r in rows)
        parts.append('<div class="card"><table><thead><tr>'
                     "<th>suite</th><th>interface</th><th>success</th><th>95% CI</th>"
                     "<th>n</th><th>progress</th><th>$/episode</th>"
                     "<th>coverage gaps</th><th>harness errors</th></tr></thead><tbody>"
                     + body + "</tbody></table>"
                     "<p class='note'>Two kinds of episode are counted here and excluded "
                     "from the success rate. <b>Coverage gaps</b>: the interface could not "
                     "express the task at all — “the action language cannot say it” is not "
                     "“the model cannot do it”. <b>Harness errors</b>: our simulator worker "
                     "crashed and the episode was lost after retries — a fact about our "
                     "infrastructure, not about any model.</p>"
                     "</div>")
        bars = [(f"{r['suite_name']} · {r['interface']}", r["success_rate"], r["ci_low"],
                 r["ci_high"], r["n"]) for r in rows if r["n"]]
        if bars:
            parts.append('<div class="card"><h3>success rate with its 95% interval</h3>'
                         + svg.interval_bars(bars) + "</div>")

    # -- knob slopes -------------------------------------------------------
    task_ids = results.tasks
    slopes = []
    for task in task_ids:
        series = {}
        for iface in ifaces:
            curve = results.knob_curve(task, iface)
            if len(curve) == 2:
                series[iface] = {lv: r.value for lv, r in curve.items()}
        if series:
            spec = get_task(task)
            slopes.append(f'<div class="card"><h3>{_e(task)} · knob = '
                          f'{_e(spec.knob.field)}</h3>'
                          f'<p class="note">{_e(spec.knob.description)} · '
                          f'{_e(spec.knob.easy)} → {_e(spec.knob.hard)} '
                          f'{_e(spec.knob.unit)}</p>'
                          + svg.knob_slopes(series) + "</div>")
    if slopes:
        parts.append("<h2>Along the knob</h2>")
        parts.append("<p>Each task varies exactly one declared field. The slope is the "
                     "measurement; the level at either end is not.</p>")
        parts.append('<div class="grid">' + "".join(slopes) + "</div>")

    # -- failures ----------------------------------------------------------
    failure_counts: dict = {}
    for (_t, _i), cell in results.cells.items():
        for k, n in cell.failures.items():
            failure_counts[k] = failure_counts.get(k, 0) + n
    if failure_counts:
        parts.append("<h2>Where episodes ended</h2>")
        parts.append('<div class="card">' + svg.stacked_failures(failure_counts)
                     + '<div class="kv">'
                     + "".join(f"<span>{_e(k)}: {v}</span>"
                               for k, v in sorted(failure_counts.items(), key=lambda kv: -kv[1]))
                     + "</div></div>")

    # -- mechanism ---------------------------------------------------------
    source = logs[0].get("run_id", "") if logs else ""
    parts.append("<h2>Every episode, replayed</h2>")
    parts.append("<p>This page aggregates. To watch any single episode — the plan view of "
                 "the table, the camera frames the model was shown, and every turn and action "
                 f"on one clock — open the viewer: <code>ruibench view &lt;log&gt;.jsonl</code>"
                 f"{' (run ' + _e(source) + ')' if source else ''}.</p>")
    sample = next((l for l in logs if l.get("turns")), None)
    if sample:
        parts.append("<h2>One episode, in time</h2>")
        parts.append("<p>Each turn is drawn from the moment its picture was taken to the "
                     "moment its answer landed, so the bar’s length <em>is</em> the "
                     "observation age. Dashed lines are events; the bottom lane is what "
                     "the arm was doing meanwhile.</p>")
        parts.append(f'<div class="card"><h3>{_e(sample.get("task_id"))} '
                     f'[{_e(sample.get("level"))}] seed {_e(sample.get("seed"))} · '
                     f'{_e(sample.get("interface"))}</h3>'
                     + svg.episode_timeline(sample) + "</div>")

    return _page(title, "".join(parts))


def _page(title: str, body: str) -> str:
    from .viewer import asset

    return (f"<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{_e(title)}</title><style>{asset('tokens.css')}{_CSS}</style></head>"
            f"<body><div class='wrap'>{body}</div></body></html>")


def write(logs: Sequence[Mapping], out: str | Path, title: str = "RUIBench results",
          run_id: str = "") -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render(logs, title=title, run_id=run_id), encoding="utf-8")
    return out
