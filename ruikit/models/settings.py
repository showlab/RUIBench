"""Settings for the model channels: the secrets file and the environment."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional

from ruikit.core.paths import ROOT

SECRETS = Path(os.environ.get("RUIKIT_SECRETS", ROOT / ".secrets" / "models.env"))


def load_settings(path: Optional[Path] = None) -> Dict[str, str]:
    """``KEY=value`` lines of the local secrets file, under the shell's own
    variables.  Nothing is exported: a password stays in this dict."""
    out: Dict[str, str] = {}
    path = Path(path or SECRETS)
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = (x.strip() for x in line.split("=", 1))
            out[key] = value.strip("'\"")
    return {**out, **{k: v for k, v in os.environ.items()
                      if k in out or k.startswith(("RUIKIT_", "OPENAI_", "ANTHROPIC_", "QWEN"))}}
