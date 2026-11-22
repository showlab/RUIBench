"""Operation guides: general practice a Harness adds to what the agent is told, the same for
every task (+guide, +guide:<name>)."""

from __future__ import annotations

from ruikit.core.paths import CONFIGS

#: the operation guides a Harness may add to what the agent is told (``+guide``)
GUIDES_DIR = CONFIGS / "guides"


DEFAULT_GUIDE = "manipulation"


def guide_text(name: str) -> str:
    """An operation guide's text, by its name in ``ruikit/configs/guides``."""
    path = GUIDES_DIR / f"{name}.md"
    if not path.exists():
        have = ", ".join(sorted(p.stem for p in GUIDES_DIR.glob("*.md"))) or "none"
        raise ValueError(f"no operation guide {name!r} (have: {have})")
    return path.read_text(encoding="utf-8").strip()
