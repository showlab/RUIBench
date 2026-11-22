"""Where things are: the checkout the package runs from, and the presets it ships."""

from __future__ import annotations

from pathlib import Path

#: the checkout's root, the folder that holds ``ruikit/``: runs, secrets and the
#: launch clocks live under it
ROOT = Path(__file__).resolve().parents[2]
#: the package itself
PACKAGE = Path(__file__).resolve().parents[1]
#: the presets it ships: the interfaces, the operation guides, the experiment plans
CONFIGS = PACKAGE / "configs"
