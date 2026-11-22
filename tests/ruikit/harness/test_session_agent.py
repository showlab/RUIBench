"""A ready-made agent's episode (v1.14): a clean agent with no shell, and a
result only when the run counts -- an agent cut off by a usage limit, one that
broke a rule, did not get the session's tools or never used them, or was reached
by anything besides its prompt and the session's results, is run again with a
new agent (v2.1)."""

import json

import pytest

from ruikit.harness import claude_code, codex, selftest
from ruikit.harness import session_agent as harness

TOOLS = ["mcp__ruibench__task_info", "mcp__ruibench__task_done", "mcp__ruibench__camera_look"]


def _start(tools=TOOLS, status="connected", **more):
    """Claude Code's own record of its start."""
    return {"type": "system", "subtype": "init", "tools": list(tools),
            "mcp_servers": [{"name": "ruibench", "status": status}], "skills": [],
            "slash_commands": [], **more}


def _cell(tmp_path, state, transcript):
    folder = tmp_path / "claude-code-claude-opus-5-5" / "DS1_easy_s0_rui.pose-v1+aids"
    folder.mkdir(parents=True)
    (folder / "session.json").write_text(json.dumps(state))
    (folder / "agent.jsonl").write_text("".join(json.dumps(r) + "\n" for r in transcript))
    return folder


def _use(name, inp=None):
    return {"type": "assistant", "message": {"id": f"m-{name}", "role": "assistant", "content": [
        {"type": "tool_use", "name": name, "input": inp or {}}],
        "usage": {"input_tokens": 10, "output_tokens": 5}}}


@pytest.fixture(autouse=True)
def _cli(monkeypatch):
    monkeypatch.setattr(harness, "cli_version", lambda h: "2.1.296 (Claude Code)")


def _collect(folder, code=0):
    return harness._collect("claude-code", "claude-opus-5-5", "DS1", "easy", 0, "rui.pose/v1+aids",
                            folder, "8s", "isaac", agent_code=code, wall_seconds=60.0,
                            agent_seconds=50.0)


def test_a_finished_episode_is_a_result_with_its_cost_and_audit(tmp_path):
    folder = _cell(tmp_path, {"status": "over", "reason": "agent_done", "success": True,
                              "replies": 2, "sim_seconds": 30.5},
                   [_start(), _use("mcp__ruibench__task_info"), _use("mcp__ruibench__task_done"),
                    {"type": "result", "is_error": False, "result": "done; it worked",
                     "num_turns": 3, "total_cost_usd": 0.1,
                     "usage": {"input_tokens": 100, "cache_read_input_tokens": 900,
                               "output_tokens": 40}}])
    row = _collect(folder)
    assert json.loads((folder / "result.json").read_text()) == row
    assert row["success"] is True and row["replies"] == 2 and row["harness"] == "claude-code"
    assert row["usage"]["input_tokens"] == 1000 and row["usage"]["output_tokens"] == 40
    assert row["audit"] == {"tool_uses": 2, "flagged": 0} and row["summary"] == "done; it worked"
    assert row["column"] == "pose" and row["aids"] == "aids"
    assert row["agent_cli"] == "2.1.296 (Claude Code)" and row["effort"] == claude_code.EFFORT


def test_an_agent_cut_off_or_breaking_a_rule_does_not_count(tmp_path):
    folder = _cell(tmp_path / "a", {"status": "over", "reason": "agent_stopped", "success": False},
                   [_use("mcp__ruibench__task_info"),
                    {"type": "result", "is_error": True, "result": "Claude usage limit reached",
                     "usage": {"input_tokens": 1, "output_tokens": 1}}])
    row = _collect(folder, code=1)
    assert "api_error" in row and row["quota"] and not (folder / "result.json").exists()
    folder = _cell(tmp_path / "b", {"status": "over", "reason": "agent_done", "success": True},
                   [_use("mcp__ruibench__task_info"), _use("Bash", {"command": "ls ruikit"})])
    row = _collect(folder)
    assert row["api_error"].startswith("the agent broke the rules") and not row["quota"]
    # an agent that stopped of its own accord ends the episode: that counts
    folder = _cell(tmp_path / "c", {"status": "over", "reason": "agent_stopped", "success": False},
                   [_start(), _use("mcp__ruibench__task_info"),
                    {"type": "result", "is_error": False, "result": "I give up",
                     "usage": {"input_tokens": 1, "output_tokens": 1}}])
    assert _collect(folder)["success"] is False and (folder / "result.json").exists()


def test_the_agents_get_the_session_and_nothing_else(tmp_path, monkeypatch):
    login, catalog = tmp_path / "auth.json", tmp_path / "models_cache.json"
    login.write_text("{}")
    catalog.write_text(json.dumps({"models": [{"slug": "gpt-5.6-sol", "tool_mode": "code_mode_only",
                                               "supports_search_tool": True,
                                               "default_reasoning_level": "medium",
                                               "apply_patch_tool_type": "freeform",
                                               "multi_agent_version": "v2"}]}))
    monkeypatch.setattr(codex, "CODEX_LOGIN", login)
    monkeypatch.setattr(codex, "CODEX_CATALOG", catalog)
    monkeypatch.setattr(harness, "codex_binary", lambda: "codex")
    monkeypatch.setattr(claude_code, "claude_binary", lambda: "claude")
    monkeypatch.setenv("SOME_SECRET", "x")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("TMPDIR", "/repo/runs/tmp")
    monkeypatch.chdir(tmp_path)
    argv, env = harness.agent_command("claude-code", "claude-opus-5-5", "go", tmp_path / "h1",
                                      __import__("pathlib").Path("session"))
    assert argv[argv.index("--tools") + 1] == "" and "--strict-mcp-config" in argv
    assert argv[argv.index("--allowedTools") + 1] == "mcp__ruibench"
    assert argv[argv.index("--effort") + 1] == claude_code.EFFORT
    config = json.loads(argv[argv.index("--mcp-config") + 1])["mcpServers"]["ruibench"]
    # the session by its whole path: the agent starts the server in its own folder; and no
    # reply that carries pictures flagged an error, which Claude Code would show as text (v2.3.4)
    assert config["args"][-3:] == ["--session", str(tmp_path / "session"), "--keep-pictures"]
    assert "cwd" not in config
    assert "SOME_SECRET" not in env and "ANTHROPIC_API_KEY" not in env
    assert env["ENABLE_TOOL_SEARCH"] == "false" and env["MCP_CONNECTION_NONBLOCKING"] == "false"
    # every tool's description whole, the longest program's code_run among them (v2.3.4)
    from ruikit.serve.schemas import mcp_tools

    longest = max(len(t.get("description") or "") for t in mcp_tools("rui.free/v1"))
    assert int(env["CLAUDE_CODE_MAX_MCP_DESCRIPTION_LENGTH"]) > longest > 4096
    assert env["CLAUDE_CODE_DISABLE_CLAUDE_MDS"] == "1"
    assert env["CLAUDE_CODE_DISABLE_AUTO_MEMORY"] == "1"
    assert env["TMPDIR"] == str(tmp_path / "h1" / "tmp")        # its own, not the repository's
    argv, env = harness.agent_command("codex", "gpt-5.6-sol", "go", tmp_path / "h2",
                                      tmp_path / "session")
    home = tmp_path / "h2" / "codex"
    assert env["CODEX_HOME"] == str(home) and (home / "auth.json").resolve() == login.resolve()
    toml = (home / "config.toml").read_text()
    assert "shell_tool = false" in toml and "code_mode_host = false" in toml
    assert "project_doc_max_bytes = 0" in toml and 'model_reasoning_effort = "high"' in toml
    assert "external_agent_memory_import = false" in toml
    assert "skip_host_skill_discovery = true" in toml
    model = json.loads((home / "catalog.json").read_text())["models"][0]
    assert model["tool_mode"] == "direct" and model["supports_search_tool"] is False
    assert "multi_agent_version" not in model and "apply_patch_tool_type" not in model
    # Codex's own code mode, for RUI-Free only: its JavaScript, the same tools
    harness.agent_command("codex-code", "gpt-5.6-sol", "go", tmp_path / "h3", tmp_path / "session")
    toml = (tmp_path / "h3" / "codex" / "config.toml").read_text()
    assert "code_mode_host" not in toml and "project_doc_max_bytes = 0" in toml
    model = json.loads((tmp_path / "h3" / "codex" / "catalog.json").read_text())["models"][0]
    assert model["tool_mode"] == "code_mode_only"
    assert "multi_agent_version" not in model and "apply_patch_tool_type" not in model


def test_an_agent_without_the_sessions_tools_or_reached_from_outside_does_not_count(tmp_path):
    """Not the interface's doing, so not a result: the session's server did not
    connect, Claude Code was offered more than the session, the agent never
    called the session, or something besides the session's results reached it."""
    done = {"status": "over", "reason": "agent_stopped", "success": False}
    end = {"type": "result", "is_error": False, "result": "stopped",
           "usage": {"input_tokens": 1, "output_tokens": 1}}
    cases = {"failed": [_start(status="failed"), _use("mcp__ruibench__task_info"), end],
             "more": [_start(tools=TOOLS + ["Bash"]), _use("mcp__ruibench__task_info"), end],
             "none": [_start(), end],
             "told": [_start(), _use("mcp__ruibench__task_info"),
                      {"type": "user", "message": {"role": "user", "content": [
                          {"type": "text", "text": "the doll is at x=0.41"}]}}, end]}
    for name, transcript in cases.items():
        folder = _cell(tmp_path / name, done, transcript)
        row = _collect(folder)
        assert "api_error" in row and not (folder / "result.json").exists(), name
    assert "server was failed" in _collect(_cell(tmp_path / "f2", done, cases["failed"]))["api_error"]
    assert "x=0.41" in _collect(_cell(tmp_path / "t2", done, cases["told"]))["api_error"]
    # tool results, and the summary of a compacted context, are the session's own
    folder = _cell(tmp_path / "ok", done, [
        _start(), _use("mcp__ruibench__task_info"),
        {"type": "user", "message": {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "1", "content": [{"type": "text", "text": "ok"}]}]}},
        {"type": "system", "subtype": "compact_boundary"},
        {"type": "user", "message": {"role": "user", "content": "Summary of the episode so far"}},
        end])
    assert "api_error" not in _collect(folder)


def test_codex_hears_its_prompt_and_nothing_else(tmp_path):
    from ruikit.experiments.rules_audit import audit_transcript

    def msg(text):
        return {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                     "content": [{"type": "input_text", "text": text}]}}

    rows = [{"type": "session_meta", "payload": {}}, msg("<environment_context>cwd</environment_context>"),
            msg("You are the control agent"), msg("# AGENTS.md instructions: the doll is left")]
    f = tmp_path / "codex-rollout.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = audit_transcript(str(f), own=str(tmp_path))
    assert out["outside"] == ["# AGENTS.md instructions: the doll is left"]
    assert out["session_calls"] == 0


def test_claude_codes_leftovers_are_named_after_its_folder(tmp_path):
    work = __import__("pathlib").Path("/users/u/.cache/ruikit-agents/agent-0123abcd4567/work")
    projects, logs = claude_code.claude_leftovers(work)
    assert projects.name == logs.name == "-users-u--cache-ruikit-agents-agent-0123abcd4567-work"
    assert projects.parent.name == "projects" and logs.parent.name == "claude-cli-nodejs"


def test_the_self_test_judges_by_what_the_agent_did(tmp_path):
    """The verdict comes from the agent's record, not from what it says."""
    word = "MARIGOLD0A1B2C"
    assert all(word in text for text in selftest.planted(word).values())
    good = [_start(), _use("mcp__ruibench__task_info"), _use("mcp__ruibench__task_done"),
            {"type": "result", "is_error": False, "result": "I have no shell.",
             "usage": {"input_tokens": 1, "output_tokens": 1}}]
    state = {"status": "over", "reason": "agent_done", "replies": 2}
    verdict = selftest.judge("claude-code", _cell(tmp_path / "a", state, good), word)
    assert verdict["passed"] and verdict["session_calls"] == 2 and verdict["offered"] == TOOLS
    told = good[:-1] + [{"type": "result", "is_error": False, "result": f"{word}: I have no shell.",
                         "usage": {"input_tokens": 1, "output_tokens": 1}}]
    verdict = selftest.judge("claude-code", _cell(tmp_path / "b", state, told), word)
    assert not verdict["passed"] and "planted" in verdict["failures"][0]
    shelled = [_start(tools=TOOLS + ["Bash"])] + good[1:]
    verdict = selftest.judge("claude-code", _cell(tmp_path / "c", state, shelled), word)
    assert not verdict["passed"]
    verdict = selftest.judge("claude-code", _cell(tmp_path / "d", dict(state, reason="agent_stopped"),
                                                  good), word)
    assert not verdict["passed"]


def test_a_passed_self_test_is_not_run_again_for_the_same_cli_and_code(tmp_path, monkeypatch):
    from ruikit.experiments import provenance

    facts = provenance.launch()
    (tmp_path / "codex-gpt-5.6-sol.json").write_text(json.dumps(
        {"passed": True, "cli": "2.1.296 (Claude Code)", "frozen": facts.get("frozen"),
         "commit": facts.get("commit")}))
    ran = []
    monkeypatch.setattr(selftest, "selftest", lambda *a, **k: ran.append(a) or {"passed": True,
                                                                               "session_calls": 2})
    assert selftest.ensure("codex", "gpt-5.6-sol", tmp_path) is None and not ran
    monkeypatch.setattr(harness, "cli_version", lambda h: "2.1.300 (Claude Code)")
    assert selftest.ensure("codex", "gpt-5.6-sol", tmp_path) is None and len(ran) == 1


def test_programs_run_only_where_the_sandbox_works(monkeypatch):
    from ruikit.experiments import preflight
    from ruikit.interfaces.sandbox import host

    assert preflight.sandbox_check(["rui.pose/v1"]) is None       # no programs: nothing to check
    monkeypatch.setattr(host, "bwrap_command", lambda python: None)
    why = preflight.sandbox_check(["rui.code/v1"])
    assert why and "bubblewrap" in why


def test_the_session_finds_its_log_beside_an_agents_own(tmp_path):
    """The agent's record (agent.jsonl) sits in the session's folder: the
    outcome is read from the tool log, found by what it holds."""
    from ruikit.serve.session_server import _outcome_row

    (tmp_path / "agent.jsonl").write_text(json.dumps({"type": "result", "result": "done"}) + "\n")
    (tmp_path / "DS1_easy_s0_rui.pose-v1.jsonl").write_text(
        json.dumps({"tool": "task_done"}) + "\n" + json.dumps({"outcome": {"reason": "agent_done"}}) + "\n")
    (tmp_path / "DS1_easy_s0_rui.pose-v1.episode.jsonl").write_text(json.dumps({"x": 1}) + "\n")
    assert _outcome_row(tmp_path)["outcome"]["reason"] == "agent_done"


def test_every_harness_label_has_a_harness():
    """The results layout's frameworks are the Harnesses, each with the module
    its batch workers run as; only the simple loop keeps the simulator in its
    own process."""
    import importlib

    from ruikit.experiments.results import HARNESSES
    from ruikit.harness.base import harnesses

    found = harnesses()
    assert tuple(found) == HARNESSES
    for h in found.values():
        assert hasattr(importlib.import_module(h.worker_module), "main")
    assert [h.name for h in found.values() if h.in_process] == ["loop"]


def test_codex_code_modes_calls_from_its_javascript_are_audited(tmp_path):
    """In Codex's code mode the tools are called from the JavaScript it writes:
    each one it reaches there is a use, the session's or not."""
    from ruikit.experiments.rules_audit import audit_transcript

    def run(code):
        return {"type": "response_item", "payload": {"type": "custom_tool_call", "name": "exec",
                                                     "input": code}}

    rows = [{"type": "session_meta", "payload": {}},
            run("const t = await tools.mcp__ruibench__task_info({}); text(JSON.stringify(t));"),
            run('await tools["mcp__ruibench__task_done"]({report: "done"}); ALL_TOOLS.map(x => x.name);'),
            run('await tools.apply_patch("*** Begin Patch");')]
    f = tmp_path / "codex-rollout.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = audit_transcript(str(f), own=str(tmp_path))
    assert out["session_calls"] == 2 and [b["tool"] for b in out["flagged"]] == ["apply_patch"]


def test_a_message_queued_from_outside_does_not_count(tmp_path):
    """Another Claude session's message reaches a running Claude Code as a queued
    command, which its printed output shows only by the command's lifecycle."""
    done = {"status": "over", "reason": "agent_done", "success": False, "replies": 3}
    folder = _cell(tmp_path, done, [
        _start(), _use("mcp__ruibench__task_info"),
        {"type": "command_lifecycle", "command_uuid": "c1", "state": "started"},
        _use("mcp__ruibench__task_done"),
        {"type": "command_lifecycle", "command_uuid": "c1", "state": "completed"},
        {"type": "result", "is_error": False, "result": "done",
         "usage": {"input_tokens": 1, "output_tokens": 1}}])
    row = _collect(folder)
    assert "queued from outside" in row["api_error"]
    argv, _ = claude_code.claude_command("claude-opus-5-5", "go", tmp_path / "s")
    assert "--restricted" in argv                    # it opens no inbox in the first place


def test_a_filmed_episode_keeps_its_video_and_drops_its_frames(tmp_path):
    folder = _cell(tmp_path, {"status": "over", "reason": "agent_done", "success": True, "replies": 2},
                   [_start(), _use("mcp__ruibench__task_info"), _use("mcp__ruibench__task_done"),
                    {"type": "result", "is_error": False, "result": "done",
                     "usage": {"input_tokens": 1, "output_tokens": 1}}])
    (folder / "video.mp4").write_bytes(b"\0")
    (folder / "frames").mkdir()
    (folder / "frames" / "f0.jpg").write_bytes(b"\0")
    row = harness._collect("claude-code", "claude-opus-5-5", "DS1", "easy", 0, "rui.pose/v1+aids",
                           folder, "8s", "isaac", agent_code=0, wall_seconds=60.0,
                           agent_seconds=50.0, filmed=True)
    assert row["filmed"] is True and row["video"] == "video.mp4"
    assert not (folder / "frames").exists()


def test_a_tool_the_cli_turned_away_is_a_slip_not_a_broken_rule(tmp_path):
    """v2.3.2: a tool name the agent's CLI does not have (``code_run`` for
    ``mcp__ruibench__code_run``) reaches nothing and tells the agent only an
    error -- the episode counts; a tool it does have, outside the session, still
    breaks the rules."""
    slip = {"type": "assistant", "message": {"id": "m-slip", "role": "assistant", "content": [
        {"type": "tool_use", "id": "toolu_slip", "name": "code_run", "input": {"source": "show('f1.front')"}}],
        "usage": {"input_tokens": 10, "output_tokens": 5}}}
    turned_away = {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_slip", "is_error": True,
         "content": "<tool_use_error>Error: No such tool available: code_run</tool_use_error>"}]}}
    done = {"type": "result", "is_error": False, "result": "done", "usage": {"input_tokens": 1, "output_tokens": 1}}
    state = {"status": "over", "reason": "agent_done", "success": True, "replies": 2}
    row = _collect(_cell(tmp_path / "a", state, [_start(), _use("mcp__ruibench__task_info"), slip,
                                                  turned_away, _use("mcp__ruibench__task_done"), done]))
    assert "api_error" not in row and row["success"] is True
    # the same call answered -- the CLI had such a tool -- is outside the session
    answered = {"type": "user", "message": {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "toolu_slip", "content": "ran"}]}}
    row = _collect(_cell(tmp_path / "b", state, [_start(), _use("mcp__ruibench__task_info"), slip,
                                                  answered, _use("mcp__ruibench__task_done"), done]))
    assert row["api_error"].startswith("the agent broke the rules")


def test_an_episode_found_not_counted_is_run_again_not_collected_again(tmp_path, monkeypatch):
    """v2.3.2: the retry of an episode whose collection found it not counted
    starts a new agent; before, it collected the same transcript again, three
    times, and the episode was lost."""
    folder = _cell(tmp_path, {"status": "over", "reason": "agent_done", "success": True},
                   [_use("mcp__ruibench__task_info"), _use("Bash", {"command": "ls ruikit"})])
    (folder / "api_error.json").write_text(json.dumps({"api_error": "the agent broke the rules"}))
    collected = []
    monkeypatch.setattr(harness, "_collect", lambda *a, **k: collected.append(a) or {"collected": True})

    class Started(Exception):
        pass

    def fresh(*a, **k):
        raise Started                            # the session of a new try: far enough
    monkeypatch.setattr(harness.subprocess, "Popen", fresh)
    with pytest.raises(Started):
        harness.run_episode("claude-code", "claude-opus-5-5", "DS1", "easy", 0, "rui.pose/v1+aids",
                            str(tmp_path), engine="mock")
    assert not collected and not (folder / "api_error.json").exists()   # cleared, then started anew


def test_a_hung_session_writes_its_stacks_and_a_try_not_counted_is_kept(tmp_path):
    """v2.3.4: a session that does not answer its stop is sent SIGABRT, and with
    PYTHONFAULTHANDLER its threads' stacks land in its log; a try that did not count
    is kept in ``<agent>/.tries/`` before the cell runs again in a clean folder."""
    import os
    import subprocess
    import sys

    from ruikit.harness import session_agent as sa

    log = tmp_path / "server.log"
    server = subprocess.Popen([sys.executable, "-c", "import time\nwhile True: time.sleep(1)"],
                              env=dict(os.environ, PYTHONFAULTHANDLER="1"),
                              stdout=open(log, "w"), stderr=subprocess.STDOUT)
    import time
    time.sleep(1.0)
    sa._dump_stacks(server)
    assert server.poll() is not None
    text = log.read_text()
    assert "Fatal Python error: Aborted" in text and 'File "<string>"' in text
    cell = tmp_path / "claude-code-claude-opus-5-5" / "WK2_hard_s15_rui.code-skill-v1+aids"
    cell.mkdir(parents=True)
    (cell / "agent.jsonl").write_text("{}\n")
    (cell / "server.log").write_text("started\n")
    (cell / "result.json").write_text("{}")
    sa._keep_try(cell)                                   # counted: nothing kept
    assert not (cell.parent / ".tries").exists()
    (cell / "result.json").unlink()
    (cell / "api_error.json").write_text('{"api_error": "the agent was cut off"}')
    sa._keep_try(cell)
    [kept] = list((cell.parent / ".tries").iterdir())
    assert kept.name.startswith(cell.name + ".")
    assert sorted(p.name for p in kept.iterdir()) == ["agent.jsonl", "api_error.json", "server.log"]


def test_an_ended_episode_whose_session_stopped_while_filming_still_counts(tmp_path):
    """A session writes its state 'over' only after scoring and filming the
    episode; one stopped before (a long video on a loaded machine) left 'ready',
    and the episode -- ended, its outcome in its log -- was run again by a new
    agent at its whole cost.  The worker now finishes it from its logs (v2.3.6)."""
    from ruikit.serve.session import _state, _write_state

    folder = tmp_path / "CR1_hard_s15_rui.step-v1+aids"
    folder.mkdir()
    _write_state(folder, {"status": "ready", "task": "CR1"})
    log = folder / f"{folder.name}.jsonl"
    log.write_text(json.dumps({"tool": "task_done"}) + "\n")
    assert not harness._ended(folder)
    harness._settle(folder, record=False, t0=0.0)
    assert _state(folder)["status"] == "ready"                 # not ended: left as it is
    log.write_text(log.read_text() + json.dumps({"outcome": {
        "reason": "agent_done", "success": True, "metrics": {"replies": 60}}}) + "\n")
    assert harness._ended(folder)
    harness._settle(folder, record=False, t0=0.0)
    state = _state(folder)
    assert state["status"] == "over" and state["reason"] == "agent_done" and state["success"]


def test_a_failed_login_stops_the_runner_like_a_usage_limit():
    """An expired login cut every agent off for 18 hours, and each cell used up
    its tries and was lost: it is the account's, not the cell's (v2.3.6)."""
    assert harness._quota("Failed to authenticate: OAuth session expired and could not be "
                          "refreshed", {})
    assert harness._quota("You have hit your usage limit", {})
    assert harness._quota('{"type":"error","message":"You hit your spend cap set by the owner of '
                          'your workspace.","codex_error_info":"usage_limit_exceeded"}', {})
    assert not harness._quota("The cube is in the square hole.", {})


def test_a_cell_whose_ended_episode_was_not_finished_is_collected_not_run_again(tmp_path, monkeypatch):
    """The try before ended its episode but was counted as cut off, its session
    stopped before it wrote its state: the next try finishes and collects it
    from the logs left in the folder, and starts no agent (v2.3.6)."""
    from ruikit.experiments import results as cells
    from ruikit.serve.session import _state, _write_state

    folder = tmp_path / cells.agent_dir("claude-code", "claude-opus-5-5", "8s") / cells.stem(
        "CR1", "hard", 15, "rui.step/v1+aids")
    folder.mkdir(parents=True)
    _write_state(folder, {"status": "ready", "task": "CR1"})
    (folder / "agent.jsonl").write_text("{}\n")
    (folder / "api_error.json").write_text(json.dumps({"api_error": "the agent was cut off"}))
    (folder / f"{folder.name}.jsonl").write_text(json.dumps({"outcome": {
        "reason": "agent_done", "success": True, "metrics": {"replies": 60}}}) + "\n")
    seen = {}
    monkeypatch.setattr(harness, "_collect", lambda *a, **k: seen.setdefault("row", {"ok": 1}))
    monkeypatch.setattr(harness.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(
        AssertionError("no agent, no session")))
    assert harness.run_episode("claude-code", "claude-opus-5-5", "CR1", "hard", 15,
                               "rui.step/v1+aids", str(tmp_path), clock="8s") == {"ok": 1}
    assert _state(folder)["status"] == "over" and not (folder / "api_error.json").exists()


def test_a_collected_episode_is_timed_on_its_files_own_clock(tmp_path, monkeypatch):
    """The file server's clock is not this machine's: an episode collected from
    its logs is timed from its prompt to its last log, both on the files' clock
    (v2.3.7: timed from this machine's clock, a salvaged episode took -6 h)."""
    import os
    import time

    from ruikit.experiments import results as cells
    from ruikit.serve.session import _state, _write_state

    folder = tmp_path / cells.agent_dir("claude-code", "claude-opus-5-5", "8s") / cells.stem(
        "CR1", "hard", 15, "rui.step/v1+aids")
    folder.mkdir(parents=True)
    _write_state(folder, {"status": "ready", "task": "CR1"})
    (folder / "prompt.txt").write_text("go")
    (folder / "agent.jsonl").write_text("{}\n")
    (folder / f"{folder.name}.jsonl").write_text(json.dumps({"outcome": {
        "reason": "agent_done", "success": False, "metrics": {"replies": 60}}}) + "\n")
    ahead = time.time() + 7 * 3600 + 51 * 60                     # the file server's clock
    os.utime(folder / "prompt.txt", (ahead, ahead))
    for p in (folder / "agent.jsonl", folder / f"{folder.name}.jsonl"):
        os.utime(p, (ahead + 3600, ahead + 3600))
    monkeypatch.setattr(harness, "_collect", lambda *a, **k: {"ok": 1})
    harness.run_episode("claude-code", "claude-opus-5-5", "CR1", "hard", 15,
                        "rui.step/v1+aids", str(tmp_path), clock="8s")
    assert abs(_state(folder)["wall_seconds"] - 3600.0) < 1.0
