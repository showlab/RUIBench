"""Turning logs into numbers and pictures.

``metrics`` aggregates, ``svg`` draws with no dependencies, ``html`` assembles
the one-page report, ``viewer`` replays episodes (plan view, camera frames,
timeline), ``gallery`` renders every task's scene, ``figures`` produces
matplotlib versions for the paper.  ``static/`` holds the viewer's script and
the design tokens both pages share.
"""

from . import viewer
from .html import render, write
from .metrics import Cell, Rate, Results, summary_table, wilson

__all__ = ["Results", "Cell", "Rate", "wilson", "summary_table", "render", "write", "viewer"]
