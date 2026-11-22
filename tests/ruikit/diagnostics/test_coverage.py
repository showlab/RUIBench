"""The controlled profiles can do the tasks that made us add a tool (approach-v2 §12).

Each case is a task the profile could not do at all before the addition --
found by ``ruikit coverage`` -- expressed now through the profile's tools only.
The whole table (24 tasks x 2 levels x 4 profiles x 3 seeds) is
``python -m ruikit.diagnostics.coverage``; these are its regression guards.
"""

import json
from pathlib import Path

import pytest

from ruikit import Budget, Episode
from ruikit.diagnostics.reference import EXPRESSORS, style_of


@pytest.mark.parametrize("task,level,profile,needs", [
    ("WK1", "hard", "rui.step/v1", "micro steps: a ring onto a peg"),
    ("DS1", "hard", "rui.step/v1", "tilt steps: a present turned over to show its label"),
    ("WK2", "easy", "rui.step/v1", "robot_wait: the raft crosses on its own"),
    ("DI3", "easy", "rui.step/v1", "robot_wait: a duckling waits at the kerb for the green"),
    ("WK2", "easy", "rui.point/v1", "plane pointing: a passenger lifted off the raft"),
    ("CL4", "easy", "rui.point/v1", "plane pointing: no surface seen under the target"),
])
def test_a_profile_can_do_what_made_us_add_a_tool(task, level, profile, needs):
    ep = Episode(task, level=level, seed=0, profile=profile, budget=Budget(max_tool_calls=6000))
    try:
        ex = EXPRESSORS[style_of(profile)]()
        out = ex.run(ep)
        assert out.success, (needs, out.reason, out.score.get("failure"), ex.unexpressed[:3])
    finally:
        ep.close()


def test_without_tilt_steps_the_step_profile_cannot_turn_a_present_over():
    """The addition is needed, not a convenience: the same run with the tilt
    moves taken out of the vocabulary fails."""
    from dataclasses import replace

    from ruikit.interfaces import load_profile

    base = load_profile("rui.step/v1")
    moves = ["forward", "back", "left", "right", "up", "down", "rotate_ccw", "rotate_cw",
             "grasp", "release"]
    params = {**base.params, "robot_step": {**base.params["robot_step"], "moves": moves}}
    ep = Episode("DS1", level="hard", seed=0, profile=replace(base, params=params),
                 budget=Budget(max_tool_calls=3000))
    try:
        out = EXPRESSORS["step"]().run(ep)
        assert not out.success
    finally:
        ep.close()


def test_a_cell_whose_worker_keeps_dying_is_lost_not_failed(tmp_path, monkeypatch):
    """The Isaac path's worker processes, run on the mock: a crash is retried,
    and a cell that keeps crashing is a harness error -- the others still run."""
    from ruikit.diagnostics.coverage_workers import MAX_ATTEMPTS, run_isolated

    monkeypatch.setenv("RUIKIT_TEST_CRASH_ON", "WK1|easy|0|rui.pose/v1")
    keys = [("WK1", "easy", 0, "rui.pose/v1"), ("WK1", "easy", 0, "rui.code/v1")]
    cells = {c.profile: c for c in run_isolated(keys, "mock", str(tmp_path / "cells.jsonl"), [0],
                                                say=lambda s: None)}
    assert cells["rui.pose/v1"].reason == "harness_error"
    assert str(MAX_ATTEMPTS) in cells["rui.pose/v1"].error
    assert cells["rui.code/v1"].reason != "harness_error"      # ran, whatever the script made of it


def test_a_rerun_into_the_same_folder_takes_its_own_results(tmp_path):
    """Worker files are named by order, GPU and task, so a rerun reuses the
    names: an earlier run's rows left in them must not answer for the rerun."""
    import json

    from ruikit.diagnostics.coverage_workers import run_isolated

    out = tmp_path / "cells.jsonl"
    key = ("DS1", "easy", 0, "rui.code/v1")
    first = run_isolated([key], "mock", str(out), [0], say=lambda s: None)[0]
    assert first.success
    stale = dict(first.as_dict(), success=False, reason="stale row from an earlier run")
    part = next((tmp_path / "cells.workers").glob("*.cells.jsonl"))
    part.write_text(json.dumps(stale) + "\n")          # as the earlier run left it
    out.write_text("")                                  # the cell is to be run again
    again = run_isolated([key], "mock", str(out), [0], say=lambda s: None)[0]
    assert again.success and again.reason != stale["reason"]


def test_isaac_cells_get_a_simulator_each_unless_asked(tmp_path, monkeypatch):
    """A reused Isaac stage keeps state from the episodes before it: by default
    every cell gets a fresh worker process; ``per_worker`` groups the cells of
    one task and level."""
    from ruikit.diagnostics import coverage_workers as cw

    started = []

    class FakeProc:
        pid = 0

        def __init__(self, argv, **kw):
            spec = json.loads(Path(argv[-1]).read_text())
            started.append([tuple(c) for c in spec["cells"]])
            for task, level, seed, profile in spec["cells"]:
                cell = cw.Cell(task, level, seed, profile, True, 1.0, "success", 1, 1.0, 0.1, [])
                with open(spec["out"], "a") as fh:
                    fh.write(json.dumps(cell.as_dict()) + "\n")

        def poll(self):
            return 0

    monkeypatch.setattr(cw.subprocess, "Popen", FakeProc)
    monkeypatch.setattr(cw.time, "sleep", lambda s: None)
    keys = [("WK1", "easy", s, "rui.pose/v1") for s in (0, 1, 2)]
    cw.run_isolated(keys, "mock", str(tmp_path / "a.jsonl"), [0], say=lambda s: None)
    assert [len(c) for c in started] == [1, 1, 1]
    started.clear()
    cw.run_isolated(keys, "mock", str(tmp_path / "b.jsonl"), [0], say=lambda s: None, per_worker=2)
    assert [len(c) for c in started] == [2, 1]


def test_each_cell_records_its_place_in_its_simulator(tmp_path):
    """The worker stamps each cell with what ran it, and its place among the
    cells its simulator ran (1: a fresh one)."""
    from ruikit.diagnostics.coverage_workers import run_isolated

    keys = [("WK1", "easy", 0, "rui.pose/v1"), ("WK1", "easy", 0, "rui.step/v1")]
    cells = run_isolated(keys, "mock", str(tmp_path / "cells.jsonl"), [0], say=lambda s: None,
                         per_worker=2)
    assert [c.provenance["process_episode"] for c in cells] == [1, 2]
    assert {c.provenance["per_worker"] for c in cells} == {2}
    rows = [json.loads(line) for line in (tmp_path / "cells.jsonl").read_text().splitlines()]
    assert all(r["provenance"]["frozen"] and r["provenance"]["python"] for r in rows)
