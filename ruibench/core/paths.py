"""Where RUIBench keeps what it fetches and builds.

In a source checkout (``pip install -e .``), beside the package, as it always
has: ``third_party/`` and ``runs/``.  Installed as a package, never inside it:
under the user's cache directory (``$XDG_CACHE_HOME/ruibench``, by default
``~/.cache/ruibench``).  An environment variable overrides either
(``RUIBENCH_THIRD_PARTY``, ``RUIBENCH_PRINTS``).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def checkout() -> Optional[Path]:
    """The source checkout this package runs from, if it runs from one."""
    root = Path(__file__).resolve().parents[2]
    return root if (root / "pyproject.toml").exists() and (root / "ruibench").is_dir() else None


def cache_root() -> Path:
    base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(base) / "ruibench"


def data_dir(name: str, in_checkout: str) -> Path:
    """A directory for fetched or built data: ``in_checkout`` (relative to
    the checkout) in a source checkout, else ``name`` under the cache root."""
    root = checkout()
    return root / in_checkout if root is not None else cache_root() / name
