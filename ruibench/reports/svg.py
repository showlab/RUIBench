"""Inline SVG charts with no dependencies.

The report has to open on any machine, in any browser, with nothing installed
-- including a cluster node with no matplotlib.  Everything here returns an SVG
string; ``ruibench.reports.figures`` has matplotlib versions for the paper.

Design rules followed throughout: a sequential palette for rates, uncertainty
drawn rather than implied, and no cell labelled a "winner" when its interval
overlaps its neighbour's.
"""

from __future__ import annotations

import html
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

# A sequential ramp for rates, readable on both themes' cards.  Neutrals and
# status colours are the page's theme tokens (static/tokens.css), applied
# through ``style`` so they follow light and dark.
RAMP = ["#e7eef5", "#c9d9e8", "#a3c0da", "#7aa3c8", "#5585b1", "#386894", "#244d70"]
INK = "var(--ink)"
MUTED = "var(--muted)"
GRID = "var(--rule)"
ACCENT = "var(--ok)"
WARN = "var(--fail)"
SERIES = ["#3f7cac", "#b0741a", "#7a5aa6", "#2f8f6f", "#a14a5c"]


def _esc(s: object) -> str:
    return html.escape(str(s))


def ramp_color(value: Optional[float]) -> str:
    if value is None:
        return "var(--surface-2)"
    i = min(len(RAMP) - 1, max(0, int(value * len(RAMP))))
    return RAMP[i]


def text_color(value: Optional[float]) -> str:
    return "#ffffff" if (value or 0) > 0.55 else INK


def heatmap(rows: Sequence[str], cols: Sequence[str],
            values: Mapping[Tuple[str, str], Optional[float]],
            counts: Optional[Mapping[Tuple[str, str], int]] = None,
            row_labels: Optional[Mapping[str, str]] = None,
            cell: int = 96, label_w: int = 190, head_h: int = 34) -> str:
    """The main figure: capability suites down, interfaces across.

    Deliberately **not** annotated with a per-row winner: 20 episodes per cell
    only supports coarse differences, and "max minus min" is inflated by noise.
    """
    w = label_w + cell * len(cols) + 16
    h = head_h + cell * len(rows) + 8
    out = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" '
           f'style="max-width:{w}px;font-family:inherit">']
    for j, c in enumerate(cols):
        x = label_w + j * cell + cell / 2
        out.append(f'<text x="{x:.0f}" y="22" text-anchor="middle" font-size="13" '
                   f'font-weight="600" style="fill:{INK}">{_esc(c)}</text>')
    for i, r in enumerate(rows):
        y = head_h + i * cell
        label = (row_labels or {}).get(r, r)
        out.append(f'<text x="{label_w - 12}" y="{y + cell / 2 + 4:.0f}" text-anchor="end" '
                   f'font-size="13" style="fill:{INK}">{_esc(label)}</text>')
        for j, c in enumerate(cols):
            x = label_w + j * cell
            v = values.get((r, c))
            n = (counts or {}).get((r, c), 0)
            out.append(f'<rect x="{x}" y="{y}" width="{cell - 4}" height="{cell - 4}" rx="6" '
                       f'style="fill:{ramp_color(v)};stroke:{GRID}"/>')
            if v is None:
                out.append(f'<text x="{x + cell / 2 - 2:.0f}" y="{y + cell / 2 + 4:.0f}" '
                           f'text-anchor="middle" font-size="13" style="fill:{MUTED}">--</text>')
            else:
                out.append(f'<text x="{x + cell / 2 - 2:.0f}" y="{y + cell / 2:.0f}" '
                           f'text-anchor="middle" font-size="17" font-weight="600" '
                           f'style="fill:{text_color(v)}">{v:.0%}</text>')
                out.append(f'<text x="{x + cell / 2 - 2:.0f}" y="{y + cell / 2 + 17:.0f}" '
                           f'text-anchor="middle" font-size="10.5" '
                           f'style="fill:{text_color(v)}" opacity="0.8">n={n}</text>')
    out.append("</svg>")
    return "".join(out)


def interval_bars(items: Sequence[Tuple[str, float, float, float, int]],
                  width: int = 620, row_h: int = 30, label_w: int = 150) -> str:
    """Rates with their Wilson intervals: (label, value, low, high, n)."""
    h = row_h * len(items) + 30
    plot_w = width - label_w - 60
    out = [f'<svg viewBox="0 0 {width} {h}" width="100%" role="img" '
           f'style="max-width:{width}px;font-family:inherit">']
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        x = label_w + frac * plot_w
        out.append(f'<line x1="{x:.0f}" y1="14" x2="{x:.0f}" y2="{h - 18}" style="stroke:{GRID}"/>')
        out.append(f'<text x="{x:.0f}" y="{h - 4}" text-anchor="middle" font-size="10" '
                   f'style="fill:{MUTED}">{frac:.0%}</text>')
    for i, (label, v, lo, hi, n) in enumerate(items):
        y = 26 + i * row_h
        out.append(f'<text x="{label_w - 10}" y="{y + 4}" text-anchor="end" font-size="12" '
                   f'style="fill:{INK}">{_esc(label)}</text>')
        x0, x1 = label_w + lo * plot_w, label_w + hi * plot_w
        out.append(f'<line x1="{x0:.1f}" y1="{y}" x2="{x1:.1f}" y2="{y}" '
                   f'style="stroke:{MUTED}" stroke-width="2"/>')
        for x in (x0, x1):
            out.append(f'<line x1="{x:.1f}" y1="{y - 4}" x2="{x:.1f}" y2="{y + 4}" '
                       f'style="stroke:{MUTED}" stroke-width="2"/>')
        cx = label_w + v * plot_w
        out.append(f'<circle cx="{cx:.1f}" cy="{y}" r="5" style="fill:{ACCENT}"/>')
        out.append(f'<text x="{label_w + plot_w + 10}" y="{y + 4}" font-size="11" '
                   f'style="fill:{MUTED}">{v:.0%} (n={n})</text>')
    out.append("</svg>")
    return "".join(out)


def knob_slopes(series: Mapping[str, Mapping[str, float]], width: int = 420,
                height: int = 230) -> str:
    """Easy -> hard slope for each interface: the single-factor contrast, drawn.

    A slope chart rather than a bar chart because the claim is about the
    *change* along the knob, not the level at either end.
    """
    pad_l, pad_r, pad_t, pad_b = 46, 96, 18, 32
    plot_h = height - pad_t - pad_b
    x0, x1 = pad_l, width - pad_r
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'style="max-width:{width}px;font-family:inherit">']
    for frac in (0, 0.5, 1.0):
        y = pad_t + (1 - frac) * plot_h
        out.append(f'<line x1="{x0}" y1="{y:.0f}" x2="{x1}" y2="{y:.0f}" style="stroke:{GRID}"/>')
        out.append(f'<text x="{x0 - 8}" y="{y + 4:.0f}" text-anchor="end" font-size="10" '
                   f'style="fill:{MUTED}">{frac:.0%}</text>')
    for label, x in (("easy", x0), ("hard", x1)):
        out.append(f'<text x="{x}" y="{height - 10}" text-anchor="middle" font-size="11" '
                   f'style="fill:{INK}">{label}</text>')
    for k, (name, pair) in enumerate(sorted(series.items())):
        e, hd = pair.get("easy"), pair.get("hard")
        if e is None or hd is None:
            continue
        ye = pad_t + (1 - e) * plot_h
        yh = pad_t + (1 - hd) * plot_h
        col = SERIES[k % len(SERIES)]
        out.append(f'<line x1="{x0}" y1="{ye:.1f}" x2="{x1}" y2="{yh:.1f}" style="stroke:{col}" '
                   f'stroke-width="2.5" opacity="0.9"/>')
        out.append(f'<circle cx="{x0}" cy="{ye:.1f}" r="4" style="fill:{col}"/>')
        out.append(f'<circle cx="{x1}" cy="{yh:.1f}" r="4" style="fill:{col}"/>')
        out.append(f'<text x="{x1 + 8}" y="{yh + 4:.1f}" font-size="11" style="fill:{col}">'
                   f'{_esc(name)}</text>')
    out.append("</svg>")
    return "".join(out)


def episode_timeline(log: Mapping, width: int = 900, height: int = 190) -> str:
    """One episode as a time axis: model turns, events, milestones, actions.

    This is the picture that makes the mechanism claims checkable -- when the
    event fired, how stale the model's picture was, how long the answer took,
    and what was still moving in the meantime.
    """
    traj = log.get("trajectory") or []
    span = max([s.get("t", 0.0) for s in traj] + [1.0])
    pad_l, pad_r, pad_t = 116, 20, 18
    plot_w = width - pad_l - pad_r
    lane = {"turns": pad_t + 20, "events": pad_t + 62, "milestones": pad_t + 104,
            "actions": pad_t + 146}

    def X(t: float) -> float:
        return pad_l + (t / span) * plot_w

    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'style="max-width:{width}px;font-family:inherit">']
    for name, y in lane.items():
        out.append(f'<text x="{pad_l - 12}" y="{y + 4}" text-anchor="end" font-size="11" '
                   f'style="fill:{MUTED}">{name}</text>')
        out.append(f'<line x1="{pad_l}" y1="{y}" x2="{width - pad_r}" y2="{y}" style="stroke:{GRID}"/>')

    for turn in log.get("turns", []):
        a, b = X(turn.get("observed_at", 0)), X(turn.get("returned_at", 0))
        # the bar's length IS the observation age: how stale the picture was by
        # the time the answer landed
        out.append(f'<rect x="{a:.1f}" y="{lane["turns"] - 7}" width="{max(2, b - a):.1f}" '
                   f'height="14" rx="3" style="fill:{ACCENT}" opacity="0.35"/>')
        out.append(f'<circle cx="{b:.1f}" cy="{lane["turns"]}" r="3" style="fill:{ACCENT}"/>')

    for ev in log.get("events", []):
        x = X(ev.get("sim_time", 0))
        out.append(f'<line x1="{x:.1f}" y1="{pad_t}" x2="{x:.1f}" y2="{lane["actions"] + 10}" '
                   f'style="stroke:{WARN}" stroke-dasharray="3 3"/>')
        out.append(f'<circle cx="{x:.1f}" cy="{lane["events"]}" r="5" style="fill:{WARN}"/>')
        out.append(f'<text x="{x + 8:.1f}" y="{lane["events"] - 8}" font-size="10.5" '
                   f'style="fill:{WARN}">{_esc(ev.get("name", "event"))}</text>')

    for name, t in (log.get("milestones_hit") or {}).items():
        x = X(float(t))
        out.append(f'<circle cx="{x:.1f}" cy="{lane["milestones"]}" r="4" style="fill:{INK}"/>')
        out.append(f'<text x="{x + 7:.1f}" y="{lane["milestones"] + 4}" font-size="10" '
                   f'style="fill:{MUTED}">{_esc(name)}</text>')

    for h in log.get("handles", []):
        s, f = h.get("started_at"), h.get("finished_at")
        if s is None:
            continue
        x, w = X(float(s)), max(2.0, X(float(f if f is not None else span)) - X(float(s)))
        col = {"done": ACCENT, "cancelled": WARN, "failed": WARN}.get(h.get("status"), MUTED)
        out.append(f'<rect x="{x:.1f}" y="{lane["actions"] - 5}" width="{w:.1f}" height="10" '
                   f'rx="2" style="fill:{col}" opacity="0.7"/>')

    out.append(f'<text x="{pad_l}" y="{height - 4}" font-size="10" style="fill:{MUTED}">0 s</text>')
    out.append(f'<text x="{width - pad_r}" y="{height - 4}" text-anchor="end" font-size="10" '
               f'style="fill:{MUTED}">{span:.1f} s (simulated)</text>')
    out.append("</svg>")
    return "".join(out)


def stacked_failures(counts: Mapping[str, int], width: int = 620, height: int = 34) -> str:
    """Where failures land: decision / not expressible / execution / timeout."""
    total = sum(counts.values()) or 1
    palette = {"none": ACCENT, "decision_error": SERIES[1], "not_expressible": SERIES[0],
               "execution_failure": SERIES[2], "timeout": SERIES[4], "budget": MUTED,
               "violation": WARN, "harness_error": INK}
    out = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" '
           f'style="max-width:{width}px;font-family:inherit">']
    x = 0.0
    for key, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        w = width * n / total
        out.append(f'<rect x="{x:.1f}" y="6" width="{max(0.0, w - 1):.1f}" height="20" rx="3" '
                   f'style="fill:{palette.get(key, MUTED)}"><title>{_esc(key)}: {n}</title></rect>')
        x += w
    out.append("</svg>")
    return "".join(out)
