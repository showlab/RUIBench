"""Surviving a crashed simulator worker.

The Isaac stack can abort with a native heap corruption that does not reproduce
-- the same two-arm scene crashed once and ran cleanly the next time.  These
tests pin down what the harness must do about it: keep everything that
finished, retry what was in flight, give up after a bounded number of attempts,
and never let a crash be scored as a model failure.
"""

import os

from ruibench.core.config import RunConfig
from ruibench.reports import Results
from ruibench.runner.isolated import IsolatedRun, run_isolated

CELLS = [("DI4", "easy", s, "main") for s in (0, 1, 2)]


def test_a_crash_is_retried_and_completed_work_is_kept(tmp_path, monkeypatch):
    out = tmp_path / "run.jsonl"
    # seed 1 always aborts its worker, standing in for a native crash
    monkeypatch.setenv("RUIBENCH_TEST_CRASH_ON", "DI4|easy|1")
    events = []

    job = IsolatedRun(CELLS, RunConfig(engine="mock", run_id="iso"), max_attempts=1)
    logs = run_isolated(job, out, progress=events.append)

    assert len(logs) == 3
    by_seed = {l["seed"]: l for l in logs}
    assert by_seed[0]["score"]["success"], "work finished before the crash must be kept"
    assert by_seed[2]["score"]["success"], "work after the crash must still run"
    assert by_seed[1]["score"]["failure"] == "harness_error"
    assert any("worker died" in e for e in events)


def test_harness_errors_never_count_against_the_model(tmp_path, monkeypatch):
    out = tmp_path / "run.jsonl"
    monkeypatch.setenv("RUIBENCH_TEST_CRASH_ON", "DI4|easy|1")
    job = IsolatedRun(CELLS, RunConfig(engine="mock", run_id="iso"), max_attempts=1)
    logs = run_isolated(job, out)

    results = Results.from_logs(logs)
    cell = results.cells[("DI4", "I0_direct")]
    assert cell.harness_errors == 1
    assert cell.total == 2 and cell.successes == 2      # 100%, not 67%


def test_a_finished_run_is_resumed_not_redone(tmp_path, monkeypatch):
    out = tmp_path / "run.jsonl"
    monkeypatch.delenv("RUIBENCH_TEST_CRASH_ON", raising=False)
    job = IsolatedRun(CELLS[:2], RunConfig(engine="mock", run_id="iso"))
    run_isolated(job, out)
    lines_before = out.read_text().count("\n")

    run_isolated(IsolatedRun(CELLS, RunConfig(engine="mock", run_id="iso")), out)
    lines_after = out.read_text().count("\n")
    assert lines_after == lines_before + 1, "only the missing cell may run again"


def test_a_scene_change_starts_a_fresh_worker_without_charging_a_retry(tmp_path, monkeypatch):
    """The Isaac backend keeps one stage per process, because tearing a stage
    down in-process hung.  A structural change therefore ends the worker -- and
    that must not be mistaken for a crash."""
    monkeypatch.delenv("RUIBENCH_TEST_CRASH_ON", raising=False)
    monkeypatch.setenv("RUIBENCH_MOCK_STRICT_STRUCTURE", "1")
    cells = [("DI4", "easy", 0, "main"), ("DI4", "easy", 1, "main"),
             ("WK5", "easy", 0, "main"), ("WK5", "easy", 1, "main")]
    events = []
    job = IsolatedRun(cells, RunConfig(engine="mock", run_id="iso"), max_attempts=1)
    logs = run_isolated(job, tmp_path / "run.jsonl", progress=events.append)

    assert all(l["score"]["success"] for l in logs), [l["score"]["failure"] for l in logs]
    assert any("new scene structure" in e for e in events)
    assert not any("worker died" in e for e in events)


def test_seeds_of_one_task_share_a_scene_structure():
    """Within a task, seeds must vary poses, colours and rules -- not the set of
    bodies -- or every episode would need its own simulator process."""
    from ruibench.tasks.instance import build, implemented

    for task in implemented():
        for level in ("easy", "hard"):
            keys = {build(task, level, s).scene.structure_key() for s in range(5)}
            assert len(keys) == 1, f"{task}/{level}: {len(keys)} structures across 5 seeds"


def test_a_hung_worker_is_killed_and_the_run_moves_on(tmp_path, monkeypatch):
    """A simulator that stops making progress must not stop the run.  The
    watchdog kills it, the in-flight episode is retried, and after the retry
    budget it is recorded as a harness error -- in seconds, not never."""
    import time as _time

    monkeypatch.delenv("RUIBENCH_TEST_CRASH_ON", raising=False)
    monkeypatch.setenv("RUIBENCH_TEST_HANG_ON", "DI4|easy|1")
    events = []
    job = IsolatedRun(CELLS, RunConfig(engine="mock", run_id="iso"), max_attempts=1,
                      stall_seconds=2.0, startup_seconds=30.0)
    started = _time.time()
    logs = run_isolated(job, tmp_path / "run.jsonl", progress=events.append)

    assert _time.time() - started < 60, "the watchdog must bound a hang"
    by_seed = {l["seed"]: l for l in logs}
    assert by_seed[0]["score"]["success"] and by_seed[2]["score"]["success"]
    assert by_seed[1]["score"]["failure"] == "harness_error"
    assert any("stalled" in e for e in events)


def test_an_exception_in_a_worker_ends_it_and_leaves_the_traceback(tmp_path, monkeypatch):
    """An ordinary exception must not turn into a hang: once a simulator has
    started, a normal interpreter exit runs shutdown hooks that hung for over
    thirteen minutes.  The worker ends at once and its log says why."""
    monkeypatch.delenv("RUIBENCH_TEST_CRASH_ON", raising=False)
    monkeypatch.delenv("RUIBENCH_TEST_HANG_ON", raising=False)
    events = []
    cells = [("NO_SUCH_TASK", "easy", 0, "main")]
    job = IsolatedRun(cells, RunConfig(engine="mock", run_id="iso"), max_attempts=1,
                      stall_seconds=5.0, log_dir=str(tmp_path / "workers"))
    logs = run_isolated(job, tmp_path / "run.jsonl", progress=events.append)

    assert logs[0]["score"]["failure"] == "harness_error"
    assert any("worker died" in e and "ConfigError" in e for e in events), events
    worker_logs = list((tmp_path / "workers").glob("worker-*.log"))
    assert worker_logs and "Traceback" in worker_logs[0].read_text()
