"""The checks a round of agent tests ends with (ruikit.diagnostics.audit)."""

import json

from ruikit.diagnostics.grasp_audit import audit_grasps
from ruikit.experiments.rules_audit import audit_rules


def _transcript(path, session, uses):
    lines = [{"message": {"role": "user", "content": f"... Your session folder: D={session}\n..."}}]
    lines += [{"message": {"role": "assistant", "content": [
        {"type": "tool_use", "name": n, "input": i}]}} for n, i in uses]
    path.write_text("\n".join(json.dumps(l) for l in lines))


def test_only_the_session_and_its_own_pictures_are_allowed(tmp_path):
    s = "/work/runs/r/clips/DS4-easy-s0-rui-pose"
    _transcript(tmp_path / "a.output", s, [
        ("Bash", {"command": f"D={s} && .venv/bin/python -m ruikit.serve.session call --dir $D "
                             "task_info '{{}}' --why \"read the rules; ls the table\" | head -50"}),
        ("Read", {"file_path": f"{s}/images/f1.front.jpg"}),
        ("Bash", {"command": "grep -rn fake ruibench/tasks/impl/"}),
        ("Read", {"file_path": "/work/runs/r/clips/DS4-easy-s0-rui-code/images/f2.wrist.jpg"}),
        ("Bash", {"command": "cat ~/.claude/projects/x/memory/MEMORY.md"}),
    ])
    (row,) = audit_rules(str(tmp_path))
    assert row["session"] == s and row["tool_uses"] == 5
    flagged = [b.get("command") or b.get("path") for b in row["flagged"]]
    assert len(flagged) == 3, flagged                    # the grep, another run's picture, the memory
    assert not any("task_info" in (f or "") for f in flagged)


def test_a_grasp_is_measured_against_the_true_centre(tmp_path):
    run = tmp_path / "DS4-easy-s0-rui-pose"
    run.mkdir()
    snap = lambda t, g, held: {"t": t, "objects": {"coin0": {"pos": [0.303, 0.004, 0.81]}},
                               "arms": {"robot0/right": {"pos": [0.3, 0.0, 0.82], "gripper": g,
                                                         "holding": held}}}
    traj = [snap(0.0, 0.0, None), snap(0.1, 1.0, None), snap(0.2, 1.0, "coin0")]
    (run / "DS4_easy_s0_rui.pose-v1.episode.jsonl").write_text(json.dumps({"trajectory": traj}))
    (row,) = audit_grasps(str(tmp_path))
    assert row["held"] == "coin0" and row["xy_mm"] == 5.0


def test_a_codex_record_and_an_mcp_run_are_audited(tmp_path):
    """v1.14: Codex's session records are read as Claude Code's transcripts
    are; the session's own MCP tools are allowed, and so is nothing else."""
    import json

    from ruikit.experiments.rules_audit import audit_transcript, command_breaks

    def call(name, args):
        return {"type": "response_item", "payload": {"type": "function_call", "name": name,
                                                       "arguments": json.dumps(args)}}

    rows = [{"type": "session_meta", "payload": {"cwd": "/tmp/agent"}},
            {"type": "response_item", "payload": {"type": "message", "role": "user",
                                                  "content": [{"type": "input_text",
                                                               "text": "You control a robot"}]}},
            call("mcp__ruibench__camera_look", {}),
            {"type": "response_item", "payload": {"type": "function_call", "name": "batch",
                                                  "namespace": "mcp__ruibench",
                                                  "arguments": json.dumps({"calls": []})}},
            call("update_plan", {"plan": []}),
            call("exec_command", {"cmd": "ls ruikit/tools"}),
            call("view_image", {"path": "/tmp/x.png"})]
    f = tmp_path / "rollout-1.jsonl"
    f.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = audit_transcript(str(f))
    assert out["agent"] == "codex" and out["tool_uses"] == 5
    assert [b["tool"] for b in out["flagged"]] == ["exec_command", "view_image"]
    # Claude Code's -p output: MCP tools only
    claude = [{"type": "user", "message": {"role": "user", "content": "You control a robot"}},
              {"type": "assistant", "message": {"role": "assistant", "content": [
                  {"type": "tool_use", "name": "mcp__ruibench__task_info", "input": {}},
                  {"type": "tool_use", "name": "WebSearch", "input": {"query": "ruibench"}}]}}]
    g = tmp_path / "agent.jsonl"
    g.write_text("".join(json.dumps(r) + "\n" for r in claude))
    out = audit_transcript(str(g))
    assert out["agent"] == "claude" and [b["tool"] for b in out["flagged"]] == ["WebSearch"]
    # a loop of calls in the shell is a program in disguise; a program's own loop is not
    call_ = ".venv/bin/python -m ruikit.serve.session call --dir $D"
    assert command_breaks(f"for i in 1 2; do {call_} robot_step '{{}}'; done")
    assert command_breaks(f"{call_} a '{{}}' && {call_} b '{{}}'")
    assert not command_breaks(f"{call_} camera_look '{{}}' | head -40")
    assert not command_breaks(f"{call_} code_run --program - <<'PY'\nfor i in range(3):\n"
                              "    robot.move_to([0, 0, 0.1])\nPY")
