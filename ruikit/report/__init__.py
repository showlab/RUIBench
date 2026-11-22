"""Reports of the results: the tables and the results page."""

from __future__ import annotations

from pathlib import Path

STATIC = Path(__file__).resolve().parent / "static"


def static(name: str) -> str:
    """A page asset (CSS or JavaScript) shipped with the package."""
    return (STATIC / name).read_text(encoding="utf-8")
