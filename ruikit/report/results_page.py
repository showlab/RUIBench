"""The results page of a run's folder: for each experiment, every agent and
interface's success rate, tokens, time and videos, and every episode with its
video.

    ruikit report runs/paper
    # -> runs/paper/index.html; with `ruibench serve runs --port 8765` running,
    #    http://localhost:8765/paper/index.html

One file: the numbers are in it, the videos are the episodes' own
(``<agent>/<episode>/video.mp4`` beside the page).  The numbers are those of
``ruikit.report.tables.summary_rows``: success with its Wilson 95% interval;
tokens as each agent's API counted them (the input includes cached input); time
from the episode's start to its end, without starting the simulator.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Optional, Sequence

from ruikit import naming
from ruikit.experiments import provenance
from ruikit.experiments.plans import Experiment
from ruikit.report.tables import (
    _agent,
    column_key,
    episode_minutes,
    carried,
    has_video,
    select,
    skipped,
    summary_rows,
)

#: how the page names a Harness and an interface's column
HARNESS_NAMES = {"loop": "minimal loop", "codex": "Codex", "codex-code": "Codex (code mode)",
                 "claude-code": "Claude Code"}
COLUMN_NAMES = {**{n: naming.label(n) for n in naming.NAMES}, "all": "all interfaces"}


def agent_name(r: dict) -> str:
    """``Claude Code + claude-opus-5-5``, with its clock when not the main one (its
    folder says so: ``loop-openai-<model>@paused``)."""
    clock = str(r.get("agent") or "").partition("@")[2]
    return (f"{HARNESS_NAMES.get(r.get('harness') or 'loop', r.get('harness'))} + {r.get('model')}"
            + (f" (clock {clock})" if clock else ""))


def column_name(key: str) -> str:
    """``pose aids-memory`` -> ``Pose aids-memory``; ``pose@privileged`` -> ``Pose (privileged)``."""
    base, _, rest = key.partition(" ")
    privileged = base.endswith("@privileged") or rest.endswith("@privileged")
    base = base.replace("@privileged", "")
    rest = rest.replace("@privileged", "")
    try:
        base = naming.name_of(base)
    except ValueError:
        pass
    name = COLUMN_NAMES.get(base, base) + (f" {rest}" if rest else "")
    return name + (" (privileged)" if privileged else "")


def _episode(r: dict, key) -> dict:
    folder = Path(str(r.get("folder") or ""))
    rel = "/".join(folder.parts[-2:])
    minutes = episode_minutes(r)
    usage = r.get("usage") or {}
    return {"a": _agent(r), "i": key(r), "p": r.get("profile"), "t": r.get("task"),
            "l": r.get("level"), "s": r.get("seed"), "suite": r.get("suite") or "",
            "ok": bool(r.get("success")), "why": r.get("reason") or "",
            "done": bool(r.get("declared_done")), "r": r.get("replies"),
            "sim": r.get("sim_seconds"), "min": round(minutes, 1) if minutes is not None else None,
            "tin": int(usage.get("input_tokens") or 0), "tout": int(usage.get("output_tokens") or 0),
            "v": f"{rel}/{r.get('video') or 'video.mp4'}" if has_video(r) else "",
            "skip": skipped(r), "carry": carried(r), "sum": str(r.get("summary") or "")[:2000]}


def _experiment(name: str, title: str, rows: Sequence[dict], missing: int) -> dict:
    key = column_key(rows)
    names = {}
    for r in rows:
        names.setdefault(_agent(r), agent_name(r))
    summary = [{**s, "agent_name": names.get(s["agent"], s["agent"]),
                "interface_name": column_name(s["interface"])} for s in summary_rows(rows)]
    mixed = len({bool(r.get("filmed")) for r in rows}) > 1
    return {"name": name, "title": title, "missing": missing, "summary": summary, "mixed": mixed,
            "columns": {k: column_name(k) for k in {key(r) for r in rows}},
            "episodes": [_episode(r, key) for r in rows]}


def page_data(rows: Sequence[dict], experiments: Sequence[Experiment] = ()) -> dict:
    exps = []
    for e in experiments:
        got, missing = select(rows, e.cells())
        exps.append(_experiment(e.name, e.title, got, missing))
    if not exps:
        exps.append(_experiment("all", "all results", rows, 0))
    return {"generated": time.strftime("%Y-%m-%d %H:%M"),
            "frozen": provenance.expected_version() or "", "experiments": exps}


def render(rows: Sequence[dict], experiments: Sequence[Experiment] = ()) -> str:
    data = json.dumps(page_data(rows, experiments), ensure_ascii=False).replace("</", "<\\/")
    return PAGE.replace("__DATA__", data)


def write(out: Path, rows: Sequence[dict], experiments: Sequence[Experiment] = ()) -> Path:
    path = Path(out) / "index.html"
    path.write_text(render(rows, experiments), encoding="utf-8")
    return path


PAGE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>RUIBench results</title>
<style>
:root { --bg: #ffffff; --fg: #1d1d1f; --muted: #6e6e73; --line: #e3e3e8; --head: #f5f5f7;
        --ok: #1a7f37; --ok-bg: #dcf5e3; --no: #b42318; --no-bg: #fde4e1; --pick: #e8f0fe;
        --accent: #2b5fd9; --warn: #b35c00; }
@media (prefers-color-scheme: dark) {
  :root { --bg: #16171a; --fg: #ececf0; --muted: #a0a0aa; --line: #2c2d33; --head: #1f2025;
          --ok: #5fd07c; --ok-bg: #173a22; --no: #ff8a7a; --no-bg: #43201c; --pick: #1d2a44;
          --accent: #8ab4ff; --warn: #f0a050; }
}
* { box-sizing: border-box; }
body { margin: 0; padding: 20px 16px 48px; background: var(--bg); color: var(--fg);
       font: 14px/1.5 -apple-system, "Segoe UI", "PingFang SC", "Noto Sans CJK SC", sans-serif; }
h1 { font-size: 20px; margin: 0 0 4px; }
h2 { font-size: 16px; margin: 24px 0 8px; }
.meta, .note { color: var(--muted); font-size: 13px; }
.tabs { display: flex; flex-wrap: wrap; gap: 6px; margin: 14px 0 4px; }
.tabs button { border: 1px solid var(--line); background: var(--head); color: var(--fg);
               padding: 5px 10px; border-radius: 6px; cursor: pointer; font: inherit; }
.tabs button.on { background: var(--accent); border-color: var(--accent); color: #fff; }
.wrap { overflow-x: auto; }
table { border-collapse: collapse; width: 100%; }
th, td { border-bottom: 1px solid var(--line); padding: 6px 8px; text-align: right;
         white-space: nowrap; }
th { background: var(--head); font-weight: 600; position: sticky; top: 0; }
th:first-child, td:first-child { text-align: left; }
tr.sum td { font-weight: 600; }
tr.agent td { background: var(--head); font-weight: 600; padding-top: 10px; cursor: default; }
tbody.rows tr { cursor: pointer; }
tbody.rows tr:not(.agent):hover, tbody.rows tr.on { background: var(--pick); }
.bar { display: inline-block; width: 60px; height: 6px; background: var(--line);
       border-radius: 3px; vertical-align: middle; margin-left: 6px; overflow: hidden; }
.bar i { display: block; height: 100%; background: var(--ok); }
.ci { color: var(--muted); font-size: 12px; }
.grid td, .grid th { text-align: center; }
.grid td:first-child, .grid th:first-child { text-align: left; }
.ep { display: inline-block; min-width: 28px; margin: 1px; padding: 1px 5px; border-radius: 5px;
      cursor: pointer; font-size: 12px; border: 1px solid transparent; }
.ep.ok { background: var(--ok-bg); color: var(--ok); }
.ep.no { background: var(--no-bg); color: var(--no); }
.ep.novid { opacity: .55; cursor: default; }
.ep.review { outline: 2px solid var(--warn); outline-offset: 1px; }
.warn { color: var(--warn); }
label.only { margin-left: 12px; font-size: 13px; color: var(--muted); cursor: pointer; }
dialog { border: 1px solid var(--line); border-radius: 10px; background: var(--bg);
         color: var(--fg); padding: 16px; width: min(880px, calc(100vw - 32px)); }
dialog::backdrop { background: rgba(0, 0, 0, .45); }
dialog video { width: 100%; max-height: 60vh; background: #000; border-radius: 6px; }
dialog dl { display: grid; grid-template-columns: max-content 1fr; gap: 2px 12px; margin: 10px 0; }
dialog dt { color: var(--muted); }
dialog pre { white-space: pre-wrap; background: var(--head); padding: 8px; border-radius: 6px;
             max-height: 30vh; overflow: auto; font: 12px/1.5 ui-monospace, monospace; }
dialog .close { float: right; border: 0; background: none; color: var(--fg); font-size: 20px;
                cursor: pointer; }
</style>
</head>
<body>
<h1>RUIBench results</h1>
<div class="meta" id="meta"></div>
<div class="tabs" id="tabs"></div>
<div class="note" id="expnote"></div>
<h2>Summary</h2>
<div class="note">Success rates with 95% intervals (Wilson). Tokens as each agent's API counts them, the input including its cached part. Time is an episode's wall-clock time from start to end, not counting the simulator's start. Click a row for its episodes and their videos.</div>
<div class="wrap"><table id="summary"></table></div>
<h2 id="ephead">Episodes</h2>
<div class="note" id="epnote">Click a row of the summary above.</div>
<div class="wrap"><table class="grid" id="episodes"></table></div>
<dialog id="player"><button class="close" id="close" aria-label="Close">×</button>
  <div id="ptitle" style="font-weight:600;margin-bottom:8px"></div>
  <div id="pvideo"></div><dl id="pfacts"></dl><div id="psum"></div></dialog>
<script id="data" type="application/json">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
let cur = 0, pick = null;
const $ = (id) => document.getElementById(id);
const el = (tag, attrs = {}, ...kids) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v; else if (k === "text") e.textContent = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v); else e.setAttribute(k, v);
  }
  for (const k of kids) e.append(k);
  return e;
};
const tok = (n) => n >= 1e9 ? (n / 1e9).toFixed(2) + "G" : n >= 1e6 ? (n / 1e6).toFixed(2) + "M"
  : n >= 1e3 ? (n / 1e3).toFixed(1) + "k" : String(Math.round(n));
const pct = (x) => Math.round(100 * x) + "%";
const REASONS = {success: "goal reached", agent_done: "the agent said it was done", gave_up: "the agent gave up",
  timeout: "simulated time ran out", wall_timeout: "wall-clock time ran out", budget_replies: "replies ran out",
  budget_turns: "turns ran out", budget_calls: "calls ran out", agent_stopped: "the agent stopped", closed: "stopped"};
const why = (r) => REASONS[r] || r || "–";
const review = (ep) => [...(ep.skip || []).map((m) => `milestone ${m} never reached`),
  ...(ep.carry || []).map((n) => `${n} carried along without being gripped`)];

function remember() {
  const e = D.experiments[cur];
  const h = new URLSearchParams({exp: e.name});
  if (pick) { h.set("agent", pick.agent); h.set("iface", pick.interface); }
  history.replaceState(null, "", "#" + h.toString());
}

function recall() {
  const h = new URLSearchParams(location.hash.slice(1));
  const k = D.experiments.findIndex((e) => e.name === h.get("exp"));
  if (k >= 0) cur = k;
  const s = D.experiments[cur].summary.find((x) => x.agent === h.get("agent") && x.interface === h.get("iface"));
  if (s) pick = {agent: s.agent, interface: s.interface, name: s.agent_name, iname: s.interface_name};
}

function tabs() {
  const box = $("tabs"); box.textContent = "";
  D.experiments.forEach((e, k) => box.append(el("button", {class: k === cur ? "on" : "",
    text: e.title, onclick: () => { cur = k; pick = null; remember(); draw(); }})));
}

function summary() {
  const e = D.experiments[cur], t = $("summary"); t.textContent = "";
  t.append(el("thead", {}, el("tr", {}, ...["Interface", "Episodes", "Success", "Successes to review", "Tokens / episode (in / out)",
    "Tokens in all (in / out)", "Time / episode", "Time in all", "Videos"].map((h) => el("th", {text: h})))));
  const body = el("tbody", {class: "rows"});
  let last = null;
  for (const s of e.summary) {
    if (s.agent !== last) {
      body.append(el("tr", {class: "agent"}, el("td", {colspan: "9", text: s.agent_name})));
      last = s.agent;
    }
    const on = pick && pick.agent === s.agent && pick.interface === s.interface;
    const bar = el("span", {class: "bar"}, el("i", {style: `width:${100 * s.rate}%`}));
    const tr = el("tr", {class: (s.interface === "all" ? "sum " : "") + (on ? "on" : ""),
      onclick: () => { pick = {agent: s.agent, interface: s.interface, name: s.agent_name,
                                iname: s.interface_name}; remember(); draw(); }},
      el("td", {text: s.interface_name}), el("td", {text: s.episodes}),
      el("td", {}, `${s.successes}/${s.episodes} ${pct(s.rate)} `,
         el("span", {class: "ci", text: `[${pct(s.lo)}–${pct(s.hi)}]`}), bar),
      el("td", {class: s.to_review ? "warn" : "", text: s.to_review ? String(s.to_review) : "–"}),
      el("td", {text: `${tok(s.tokens_in_median)} / ${tok(s.tokens_out_median)}`}),
      el("td", {text: `${tok(s.tokens_in)} / ${tok(s.tokens_out)}`}),
      el("td", {text: s.minutes_median == null ? "–" : `${s.minutes_median.toFixed(1)} min`}),
      el("td", {text: `${s.hours.toFixed(1)} h`}),
      el("td", {text: `${s.videos}/${s.episodes}`}));
    body.append(tr);
  }
  t.append(body);
}

function badge(ep, label) {
  const b = el("span", {class: "ep " + (ep.ok ? "ok" : "no") + (ep.v ? "" : " novid") +
      (review(ep).length ? " review" : ""),
    title: `${ep.t} ${ep.l} seed ${ep.s}: ${ep.ok ? "success" : "failure"} (${why(ep.why)})` + (ep.v ? "" : ", no video") +
      (review(ep).length ? `; to review: ${review(ep).join("; ")}` : ""),
    text: (label ? label + " " : "") + (ep.ok ? "✓" : "✗")});
  if (ep.v) b.addEventListener("click", () => play(ep));
  return b;
}

let onlyReview = false;
function episodes() {
  const e = D.experiments[cur], t = $("episodes"); t.textContent = "";
  if (!pick) { $("ephead").textContent = "Episodes"; $("epnote").textContent = "Click a row of the summary above."; return; }
  const mine = e.episodes.filter((x) => x.a === pick.agent && (pick.interface === "all" || x.i === pick.interface)
    && (!onlyReview || review(x).length));
  $("ephead").textContent = `Episodes: ${pick.name} · ${pick.iname}`;
  const note = $("epnote"); note.textContent = "✓ success, ✗ failure; click one to play its video (gray: no video). An orange frame is a success to review: a milestone was never reached, or something came along without being gripped -- another way of doing it, or a loophole in the task.";
  const box = el("input", {type: "checkbox"}); box.checked = onlyReview;
  box.addEventListener("change", () => { onlyReview = box.checked; episodes(); });
  note.append(el("label", {class: "only"}, box, " only the successes to review"));
  const tasks = [...new Set(mine.map((x) => x.t))].sort();
  if (pick.interface === "all") {
    const cols = [...new Set(mine.map((x) => x.i))];
    const order = e.summary.filter((s) => s.agent === pick.agent).map((s) => s.interface);
    cols.sort((a, b) => order.indexOf(a) - order.indexOf(b));
    t.append(el("thead", {}, el("tr", {}, el("th", {text: "Task"}),
      ...cols.map((c) => el("th", {text: e.columns[c] || c})))));
    const body = el("tbody");
    for (const task of tasks) {
      body.append(el("tr", {}, el("td", {text: task}), ...cols.map((c) => el("td", {},
        ...mine.filter((x) => x.t === task && x.i === c).map((x) => badge(x, x.l === "easy" ? "E" : "H"))))));
    }
    t.append(body);
    return;
  }
  t.append(el("thead", {}, el("tr", {}, ...["Task", "Easy", "Hard"].map((h) => el("th", {text: h})))));
  const body = el("tbody");
  for (const task of tasks) {
    const cell = (level) => el("td", {}, ...mine.filter((x) => x.t === task && x.l === level).map((x) =>
      el("span", {}, badge(x, ""), el("span", {class: "ci", text: ` ${x.r ?? "–"} replies · ${x.min ?? "–"} min `}))));
    body.append(el("tr", {}, el("td", {text: task}), cell("easy"), cell("hard")));
  }
  t.append(body);
}

function play(ep) {
  const h = new URLSearchParams(location.hash.slice(1));
  h.set("play", [ep.t, ep.l, ep.s, ep.i].join(","));
  history.replaceState(null, "", "#" + h.toString());
  $("ptitle").textContent = `${ep.t} · ${ep.l} · seed ${ep.s} · ${ep.p}`;
  const v = $("pvideo"); v.textContent = "";
  v.append(el("video", {src: ep.v, controls: "", autoplay: "", playsinline: ""}));
  const f = $("pfacts"); f.textContent = "";
  const facts = [["Result", (ep.ok ? "success" : "failure") + ` (${why(ep.why)})` +
      (ep.done && ep.why !== "agent_done" ? ", the agent said it was done" : "")],
    ["Replies", ep.r ?? "–"], ["Simulated time", ep.sim == null ? "–" : `${ep.sim} s`],
    ["Time", ep.min == null ? "–" : `${ep.min} min`], ["Tokens (in / out)", `${tok(ep.tin)} / ${tok(ep.tout)}`]];
  if (review(ep).length) facts.push(["To review", review(ep).join("; ")]);
  for (const [k, val] of facts) f.append(el("dt", {text: k}), el("dd", {text: String(val)}));
  const s = $("psum"); s.textContent = "";
  if (ep.sum) s.append(el("div", {class: "note", text: "The agent's summary"}), el("pre", {text: ep.sum}));
  $("player").showModal();
}

function draw() {
  const e = D.experiments[cur];
  const n = e.episodes.length, vids = e.episodes.filter((x) => x.v).length;
  $("meta").textContent = `Generated ${D.generated}` + (D.frozen ? ` · frozen version ${D.frozen}` : "");
  $("expnote").textContent = `${e.title}: ${n} episodes with a result` + (e.missing ? `, ${e.missing} not yet run` : "") +
    `; ${vids} with a video.` + (e.mixed ? " Note: filmed and unfilmed episodes are mixed; filming changes the pictures the agent sees, so the two do not compare." : "");
  tabs(); summary(); episodes();
}
$("close").addEventListener("click", () => { $("player").close(); });
$("player").addEventListener("close", () => {
  $("pvideo").textContent = "";
  const h = new URLSearchParams(location.hash.slice(1));
  h.delete("play");
  history.replaceState(null, "", "#" + h.toString());
});
recall();
draw();
(() => {                                     // a link to one episode opens its video
  const want = new URLSearchParams(location.hash.slice(1)).get("play");
  if (!want || !pick) return;
  const [t, l, s, i] = want.split(",");
  const ep = D.experiments[cur].episodes.find((x) => x.a === pick.agent && x.t === t &&
    x.l === l && String(x.s) === s && x.i === i);
  if (ep && ep.v) play(ep);
})();
</script>
</body>
</html>
"""
