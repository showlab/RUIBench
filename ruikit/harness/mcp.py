"""How a harness reaches the session: the MCP server that serves an episode's tools."""

from __future__ import annotations

import sys
from pathlib import Path

from ruikit.core.paths import ROOT

PYTHON = sys.executable


def _mcp_server(folder: Path) -> dict:
    # the session's folder by its absolute path: an agent starts the server in its
    # own work folder, where a relative one names nothing
    return {"command": PYTHON,
            "args": ["-m", "ruikit.serve.mcp_stdio", "--session", str(Path(folder).resolve())],
            "env": {"PYTHONPATH": str(ROOT)}, "cwd": str(ROOT)}
