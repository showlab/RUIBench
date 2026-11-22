"""Codex as a harness: its login, its home folder with the session server as its one MCP
server, and its model catalog."""

from __future__ import annotations

import json
import os
import shutil
from glob import glob
from pathlib import Path
from typing import Optional

from ruikit.harness.mcp import _mcp_server

#: the agents' own stores of their logins (linked into a Codex home, never copied)
CODEX_LOGIN = Path.home() / ".codex" / "auth.json"


#: Codex features that would give it something besides the session's tools
CODEX_OFF = ("shell_tool", "unified_exec", "view_image", "multi_agent", "multi_agent_v2",
             "browser_use", "browser_use_external", "computer_use", "image_generation", "apps",
             "plugins", "skill_search", "skill_mcp_dependency_install", "memories", "goals",
             "hooks", "tool_suggest", "in_app_browser", "remote_plugin", "workspace_dependencies",
             "realtime_conversation", "external_agent_memory_import", "code_mode_host")
#: Codex features that keep it from finding more: skills on the host, outside its home
CODEX_ON = ("skip_host_skill_discovery",)


def codex_binary() -> Optional[str]:
    """The Codex CLI: ``RUIKIT_CODEX``, else the newest the VS Code extension carries
    -- a remote VS Code's, else a local one's (on the real station's workstation,
    whose ``codex`` on the PATH is an old snap) -- else ``codex`` on the PATH."""
    if os.environ.get("RUIKIT_CODEX"):
        return os.environ["RUIKIT_CODEX"]
    for vscode in (".vscode-server", ".vscode"):
        found = newest(glob(str(Path.home() / vscode / "extensions/openai.chatgpt-*/bin/"
                                                        "linux-x86_64/codex")))
        if found:
            return found
    return shutil.which("codex")


def newest(paths) -> Optional[str]:
    """The path with the highest version number in it (``2.1.311`` above ``2.1.99``)."""
    import re

    def version(path: str):
        return tuple(int(n) for n in re.findall(r"\d+", Path(path).as_posix()))

    return max(paths, key=version, default=None)


def _toml(value) -> str:
    return json.dumps(value) if not isinstance(value, bool) else ("true" if value else "false")


#: Codex's own list of its models (with how each calls tools), as it last fetched it
CODEX_CATALOG = Path.home() / ".codex" / "models_cache.json"


#: how Codex calls the session's tools, by harness label: ``codex`` one call an
#: item, as the simple loop's model does; ``codex-code`` Codex's own way with
#: these models, from JavaScript it writes -- where a loop of calls is a program
CODEX_MODES = {"codex": "direct", "codex-code": "code"}


def codex_catalog(codex: Path, model: str, mode: str = "direct") -> Optional[Path]:
    """Codex's model list with ``model`` offered the session's tools and none of
    Codex's own: no sub-agents (their calls would not be in the episode's
    record), no file patching, no web search, nothing behind a tool search, no
    instructions for skills, plugins or apps.  ``direct``: each tool called by
    itself, one call an item, as the simple loop's model does -- not from
    JavaScript it writes (``code_mode_only``, Codex's way with these models),
    where a loop of tool calls in one go would be a program in an interface
    that takes one action a reply.  ``code``: Codex's own way, kept.  None when
    the list is not there to copy."""
    try:
        data = json.loads(CODEX_CATALOG.read_text())
    except (OSError, ValueError):
        return None
    models = data.get("models") if isinstance(data, dict) else data
    for m in models or []:
        if m.get("slug") == model:
            if mode == "direct":
                m["tool_mode"] = "direct"
            m.update(supports_search_tool=False, experimental_supported_tools=[],
                     include_skills_usage_instructions=False,
                     include_plugin_usage_instructions=False,
                     include_apps_usage_instructions=False)
            for key in ("multi_agent_version", "apply_patch_tool_type", "web_search_tool_type"):
                m.pop(key, None)
    path = codex / "catalog.json"
    path.write_text(json.dumps({"models": models}))
    return path


#: the reasoning effort, the same in every episode (``model_reasoning_effort``):
#: high, as Claude Code's (``claude_code.EFFORT``).  Until v2.3.12 the model's own
#: default in Codex's list -- medium for GPT-5.6 Sol
EFFORT = "high"


def codex_effort(model: str) -> str:
    """The reasoning effort Codex runs ``model`` at, set so that it does not change
    with Codex's version (``EFFORT``)."""
    return EFFORT


def codex_home(home: Path, folder: Path, model: str, mode: str = "direct") -> Path:
    """A Codex home with nothing but the login (linked) and a config that offers
    the session's tools only: no instruction file (AGENTS.md), and the model's
    own reasoning effort."""
    codex = home / "codex"
    codex.mkdir(parents=True)
    os.symlink(CODEX_LOGIN, codex / "auth.json")
    server = _mcp_server(folder)
    catalog = codex_catalog(codex, model, mode)
    lines = [f"model = {_toml(model)}", f"model_reasoning_effort = {_toml(codex_effort(model))}",
             'approval_policy = "never"', 'sandbox_mode = "read-only"', 'web_search = "disabled"',
             "project_doc_max_bytes = 0", "project_doc_fallback_filenames = []"]
    if catalog is not None:
        lines.append(f"model_catalog_json = {_toml(str(catalog))}")
    lines += ["", "[features]"]
    lines += [f"{name} = false" for name in CODEX_OFF
              if not (name == "code_mode_host" and mode == "code")]
    lines += [f"{name} = true" for name in CODEX_ON]
    lines += ["", "[tools]", "web_search = false", "view_image = false", "",
              "[mcp_servers.ruibench]", f"command = {_toml(server['command'])}",
              f"args = {json.dumps(server['args'])}", f"cwd = {_toml(server['cwd'])}",
              "startup_timeout_sec = 120", "tool_timeout_sec = 3600",
              "", "[mcp_servers.ruibench.env]",
              f"PYTHONPATH = {_toml(server['env']['PYTHONPATH'])}"]
    (codex / "config.toml").write_text("\n".join(lines) + "\n")
    return codex


def _keep_codex_record(home: Path, folder: Path) -> None:
    """Codex keeps its session under its home: the cell keeps a copy (the audit
    and the token count read it; it holds no login)."""
    found = sorted(glob(str(home / "codex" / "sessions" / "**" / "rollout-*.jsonl"),
                        recursive=True))
    if found:
        shutil.copyfile(found[-1], folder / "codex-rollout.jsonl")
