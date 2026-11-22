"""Claude Code as a harness: its login, its binary, and how it is started on a
session -- with the session's tools and nothing else."""

from __future__ import annotations

import json
import os
import re
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ruikit.harness.mcp import _mcp_server

CLAUDE_LOGIN = Path.home() / ".claude" / ".credentials.json"

#: its options besides the prompt and the model: the session's server as its one
#: MCP server and its tools the only ones allowed; no built-in tool (``--tools ""``:
#: no shell, files, web or sub-agents); no settings, hooks, skills or slash
#: commands; no session kept on disk; and a restricted session, which also opens
#: no inbox: another Claude session on the machine cannot message it
FLAGS = ("--strict-mcp-config", "--tools", "", "--allowedTools", "mcp__ruibench",
         "--permission-mode", "dontAsk", "--setting-sources", "", "--disable-slash-commands",
         "--no-session-persistence", "--restricted")
#: the reasoning effort, the same in every episode (``--effort``): high, as Codex's
#: (``codex.EFFORT``).  Medium in v2.3.11 alone; every result records its effort
#: (``provenance.not_comparable``)
EFFORT = "high"

#: its environment, besides the few variables every agent gets
ENV = {
    # the first turn waits for the session's server: an agent started without the
    # session's tools has nothing to act with
    "MCP_CONNECTION_NONBLOCKING": "false", "MCP_TIMEOUT": "120000",
    "MCP_TOOL_TIMEOUT": "3600000",
    # no instruction file (CLAUDE.md: managed, the user's, the project's) and no
    # memory folder
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1", "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    # no reminder to keep a user posted: nobody is watching, and it is not the task's
    "CLAUDE_CODE_SILENT_TURN_REMINDER": "0",
    # the tools listed up front, not behind a search; no update, no other traffic
    "ENABLE_TOOL_SEARCH": "false", "DISABLE_AUTOUPDATER": "1",
    # each tool's description whole: Claude Code cuts one at 4,096 characters unless
    # told otherwise, and a program's code_run describes its API in 23 to 36 KB --
    # Codex and the minimal loop pass it whole (v2.3.4)
    "CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH": "100000",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
}


def claude_binary() -> Optional[str]:
    """``RUIKIT_CLAUDE``, else ``claude`` on the PATH, else the newest the VS Code
    extension carries (the real station's workstation has no other)."""
    from glob import glob

    from ruikit.harness.codex import newest

    found = os.environ.get("RUIKIT_CLAUDE") or shutil.which("claude")
    for vscode in (".vscode-server", ".vscode"):
        found = found or newest(glob(str(Path.home() / vscode / "extensions" /
                                         "anthropic.claude-code-*/resources/native-binary/claude")))
    return found


def claude_command(model: str, prompt: str, folder: Path) -> Tuple[List[str], Dict[str, str]]:
    """(argv, environment) of Claude Code on the session in ``folder``; the
    environment is added to the agent's few variables."""
    server = {k: v for k, v in _mcp_server(folder).items() if k != "cwd"}   # not a key it takes
    # it shows the model only the text of a result flagged isError: the pictures of a
    # failed call stay in by not flagging it (the failure is in the result's text)
    server["args"] = [*server["args"], "--keep-pictures"]
    config = json.dumps({"mcpServers": {"ruibench": {"type": "stdio", **server}}})
    argv = [claude_binary(), "-p", prompt, "--model", model, "--effort", EFFORT,
            "--output-format", "stream-json", "--verbose", "--mcp-config", config, *FLAGS]
    return argv, dict(ENV)


def claude_leftovers(work: Path) -> List[Path]:
    """What Claude Code keeps outside a working folder, named after it -- its
    project folder and its MCP logs -- removed with the agent's home."""
    name = re.sub(r"[^A-Za-z0-9]", "-", str(work))
    return [Path.home() / ".claude" / "projects" / name,
            Path.home() / ".cache" / "claude-cli-nodejs" / name]


def claude_start(transcript: Path) -> Optional[str]:
    """Why the agent did not start with the session's tools and nothing else,
    from Claude Code's own record of its start; None when it did."""
    try:
        lines = transcript.read_text(errors="replace").splitlines()
    except OSError:
        return "no record of the agent's start"
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("type") != "system" or r.get("subtype") != "init":
            continue
        servers = {s.get("name"): s.get("status") for s in r.get("mcp_servers") or []}
        others = [t for t in r.get("tools") or [] if not str(t).startswith("mcp__ruibench__")]
        if servers.get("ruibench") != "connected":
            return f"the session's MCP server was {servers.get('ruibench') or 'missing'}"
        if others or set(servers) != {"ruibench"}:
            return f"it was offered more than the session: {sorted(others) or sorted(servers)}"
        if r.get("skills") or r.get("slash_commands"):
            return "it was offered skills or slash commands"
        return None
    return "no record of the agent's start"
