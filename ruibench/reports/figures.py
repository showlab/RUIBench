"""Matplotlib versions of the report figures, for the paper.

Optional: matplotlib is imported inside each function, so the HTML report and
the CLI keep working without it.  Styling is deliberately plain -- one font
size, no chart junk, intervals always drawn.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional, Sequence

from ..tasks import SUITES, SUITE_ORDER
from .metrics import Results


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("figures need matplotlib: pip install matplotlib") from exc


def heatmap(results: Results, out: str | Path, cmap: str = "YlGnBu"):
    """Fig. 1 right: four suites x four interfaces."""
    plt = _plt()
    hm = results.heatmap()
    ifaces = results.interfaces
    suites = [s for s in SUITE_ORDER if any(hm[s][i].total for i in ifaces)]
    data = [[hm[s][i].rate.value if hm[s][i].total else float("nan") for i in ifaces]
            for s in suites]

    fig, ax = plt.subplots(figsize=(1.4 * len(ifaces) + 3.2, 0.62 * len(suites) + 1.4))
    im = ax.imshow(data, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(ifaces)), ifaces)
    ax.set_yticks(range(len(suites)), [SUITES[s][1] for s in suites])
    for r, s in enumerate(suites):
        for c, i in enumerate(ifaces):
            cell = hm[s][i]
            if not cell.total:
                ax.text(c, r, "--", ha="center", va="center", color="#888")
                continue
            ax.text(c, r, f"{cell.rate.value:.0%}", ha="center", va="center",
                    color="white" if cell.rate.value > 0.55 else "black", fontsize=10)
    fig.colorbar(im, ax=ax, label="success rate", fraction=0.035)
    ax.set_title("Interface advantage depends on what the task requires", fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return Path(out)


def knob_curve(results: Results, task: str, out: str | Path):
    """One task's easy->hard slope for every interface."""
    plt = _plt()
    fig, ax = plt.subplots(figsize=(4.0, 3.0))
    for iface in results.interfaces:
        curve = results.knob_curve(task, iface)
        if len(curve) < 2:
            continue
        levels = ["easy", "hard"]
        vals = [curve[l].value for l in levels if l in curve]
        lows = [curve[l].interval[0] for l in levels if l in curve]
        highs = [curve[l].interval[1] for l in levels if l in curve]
        ax.errorbar([0, 1], vals,
                    yerr=[[v - lo for v, lo in zip(vals, lows)],
                          [hi - v for v, hi in zip(vals, highs)]],
                    marker="o", capsize=3, label=iface)
    ax.set_xticks([0, 1], ["easy", "hard"])
    ax.set_ylim(-0.02, 1.02)
    ax.set_ylabel("success rate")
    ax.set_title(task, fontsize=11)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return Path(out)


def mechanism(logs: Sequence[Mapping], out: str | Path):
    """The mechanism figure the design requires (v0 §5.3): success alone cannot
    tell a system that pauses and recovers from one that never noticed."""
    plt = _plt()
    ages, responses, cancels = [], [], []
    for log in logs:
        m = log.get("metrics") or {}
        ages.append(m.get("observation_age_mean", 0.0))
        if m.get("response_time") is not None:
            responses.append(m["response_time"])
        cancels.append(m.get("actions_cancelled", 0))

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8))
    for ax, data, label in zip(axes, [ages, responses, cancels],
                               ["observation age (s)", "response time after event (s)",
                                "actions cancelled"]):
        if data:
            ax.hist(data, bins=12, color="#4a6fa5", edgecolor="white")
        ax.set_xlabel(label, fontsize=9)
        ax.tick_params(labelsize=8)
    axes[0].set_ylabel("episodes", fontsize=9)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return Path(out)
