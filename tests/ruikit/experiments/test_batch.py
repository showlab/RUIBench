"""The batch runner of model episodes, on the mock engine against a stand-in model."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from ruikit.experiments import batch
from ruikit.experiments import machine, plans, preflight
from ruikit.experiments.plans import Job, chunks, parse_seeds, plan
from ruikit.experiments.machine import pick_gpu


class Model:
    """Answers every request with one call of task_done -- or with ``status``."""

    def __init__(self, status=200, body=None):
        self.posts = 0
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                outer.posts += 1
                data = json.dumps(body or {"choices": [{"finish_reason": "tool_calls", "message": {
                    "role": "assistant", "content": "", "tool_calls": [{
                        "id": "c0", "type": "function", "function": {
                            "name": "task_done", "arguments": json.dumps({"report": "done"})}}]}}],
                    "usage": {"prompt_tokens": 50, "completion_tokens": 5}}).encode()
                self.send_response(status)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}/v1"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


@pytest.fixture
def model():
    m = Model()
    yield m
    m.close()


def _run(out, url, **kw):
    said = []
    args = dict(engine="mock", workers=1, per_worker=1, base_url=url, say=said.append)
    args.update(kw)
    code = batch.run(["opus5"], ["WK1"], ["easy"], "0-1", ["rui.pose/v1", "rui.step/v1"],
                     str(out), **args)
    return code, said


def test_the_plan_goes_seed_by_seed_and_a_worker_keeps_one_task_and_level():
    jobs = plan(["m"], ["WK1", "DS2"], ["easy", "hard"], "0-1", ["rui.pose/v1", "rui.code/v1"])
    assert len(jobs) == 16
    assert {j.seed for j in jobs[:8]} == {0} and {j.seed for j in jobs[8:]} == {1}
    for c in chunks(jobs, 3):
        assert len(c) <= 3 and len({(j.task, j.level) for j in c}) == 1
    assert Job("openai:m", "WK1", "easy", 3, "rui.pose/v1").folder("o").as_posix() == \
        "o/loop-openai-m/WK1_easy_s3_rui.pose-v1"


def test_seeds_by_range_list_or_split():
    from ruibench.tasks import splits

    assert parse_seeds("0-2", "WK1", "easy") == [0, 1, 2]
    assert parse_seeds("4,1", "WK1", "easy") == [4, 1]
    assert parse_seeds("dev:2", "CL2", "hard") == splits.seeds("CL2", "hard", "dev")[:2]


def test_a_worker_goes_on_a_free_card_none_of_ours_is_on():
    free = {0: 1000, 1: 30000, 2: 20000, 3: 45000}
    assert pick_gpu(free, mine=[], last_start={}, now=100.0) == 3
    assert pick_gpu(free, mine=[3], last_start={}, now=100.0) == 1
    # a card ours started on a moment ago waits; a full one never takes one
    assert pick_gpu(free, mine=[3], last_start={1: 90.0}, now=100.0) == 2
    assert pick_gpu({0: 1000}, mine=[], last_start={}, now=100.0) is None


def test_a_batch_runs_resumes_and_reports(tmp_path, model):
    code, said = _run(tmp_path, model.url, workers=2, per_worker=2)
    assert code == 0
    results = sorted(tmp_path.glob("loop-opus5/*/result.json"))
    assert len(results) == 4 and model.posts == 5          # a check of the model, then 4 episodes
    row = json.loads(results[0].read_text())
    assert row["reason"] == "agent_done" and row["replies"] == 1 and row["engine"] == "mock"
    # what it was run with, for the tables
    assert row["harness"] == "loop" and row["agent"] == "loop-opus5" and row["column"] == "pose"
    assert row["aids"] == "none" and row["suite"] == "knowledge" and row["version"]
    assert row["perception"]["backend"] == "sam3" and "milestones" in row
    assert (results[0].parent / f"{results[0].parent.name}.model.jsonl").exists()
    assert not list(tmp_path.glob("loop-opus5/*/running.json"))
    report = (tmp_path / "report.md").read_text()
    assert "| loop-opus5 | 0/2 0% [0-66] | 0/2 0% [0-66] | 0/4 0% [0-49] |" in report
    assert "4 results; 0 cells not run yet; 0 lost" in report
    assert len((tmp_path / "cells.jsonl").read_text().splitlines()) == 4
    # where each result comes from: two workers of two episodes each
    rows = [json.loads(r.read_text()) for r in results]
    assert sorted(r["provenance"]["process_episode"] for r in rows) == [1, 1, 2, 2]
    assert {r["provenance"]["per_worker"] for r in rows} == {2}
    assert {r["provenance"]["frozen"] for r in rows} == {rows[0]["version"].split("+")[0]}
    assert any("up to 2 episodes per simulator" in s for s in said)
    assert report.count("episode 2 of its simulator process |") == 2
    # the loop's episodes end on the task's limits, as the agents' sessions do:
    # not on real time, nor on a count of tool calls
    log = next(results[0].parent.glob("*_rui.pose-v1.jsonl"))
    budget = json.loads(log.read_text().splitlines()[0])["header"]["budget"]
    assert budget["max_wall_seconds"] == 86400.0 and budget["max_tool_calls"] == 6000
    # again: nothing left to run, nothing asked
    code, said = _run(tmp_path, model.url)
    assert code == 0 and model.posts == 5 and "4 done" in said[0]


def test_a_used_up_quota_stops_new_launches(tmp_path):
    spent = Model(status=402, body={"error": {"message": "monthly quota", "code": "quota_exceeded"}})
    try:
        code, said = _run(tmp_path, spent.url, ping=False)
    finally:
        spent.close()
    assert code == 1 and spent.posts == 1                   # one episode tried, no more launched
    assert not list(tmp_path.glob("loop-opus5/*/result.json"))
    errors = list(tmp_path.glob("loop-opus5/*/api_error.json"))
    assert len(errors) == 1 and json.loads(errors[0].read_text())["quota"]
    assert any("quota" in s and "not starting more" in s for s in said)


def test_a_crashing_worker_is_retried_then_lost_and_the_rest_go_on(tmp_path, model, monkeypatch):
    bad = Job("opus5", "WK1", "easy", 0, "rui.pose/v1")
    monkeypatch.setenv("RUIKIT_TEST_CRASH_ON", bad.tag())
    code, said = _run(tmp_path, model.url)
    lost = bad.folder(tmp_path) / "lost.json"
    assert lost.exists() and json.loads(lost.read_text())["attempts"] == batch.MAX_ATTEMPTS
    assert len(list(tmp_path.glob("loop-opus5/*/result.json"))) == 3   # the other three ran
    assert "3 results; 0 cells not run yet; 1 lost" in (tmp_path / "report.md").read_text()
    assert code == 0                                        # nothing left to run



def test_only_simulators_count_against_the_isaac_limit(monkeypatch):
    """What else of ours sits on a GPU -- the vision server, a local language
    model -- is no Isaac app: counted, it left a batch short of its
    ``--max-isaac`` (RUI-Zero, 10/3).  A simulator is told by Kit being loaded."""
    import os
    import types

    me = os.getpid()
    monkeypatch.setattr(machine.subprocess, "run", lambda *a, **k: types.SimpleNamespace(
        stdout=f"{me}\n"))
    assert not machine._kit_app(me)                 # this test has loaded no simulator
    assert machine.other_isaac([]) == 0
    monkeypatch.setattr(machine, "_kit_app", lambda pid: True)
    assert machine.other_isaac([]) == 1
    assert machine.other_isaac([me]) == 0           # one of our own workers


def test_no_isaac_app_starts_on_a_busy_machine(monkeypatch):
    """What an Isaac app is short of is cores: the runner holds a start while
    the load is over its share of them (the count of four went on 10/3)."""
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 100)
    monkeypatch.setattr(os, "getloadavg", lambda: (85.0, 60.0, 50.0))
    assert machine.busy(0.8) and not machine.busy(0.9)


class Vision:
    """A vision server's front door: ``/info`` and nothing more."""

    def __init__(self, info):
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                data = json.dumps(info).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *a):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()


def test_isaac_runs_are_checked_for_a_vision_server_before_any_start(tmp_path, monkeypatch):
    """Isaac answers from pixels (v1.13): a vision server that is not there, runs
    another detector or has no depth model, or a synthetic profile, stops the run
    before a simulator starts -- not once each of them has."""
    import socket

    with socket.socket() as s:                # a port nobody answers on
        s.bind(("127.0.0.1", 0))
        nobody = s.getsockname()[1]
    monkeypatch.setenv("RUIKIT_VISION_URL", f"http://127.0.0.1:{nobody}")
    assert "not answering" in preflight.vision_check(["rui.pose/v1"])
    assert "synthetic" in preflight.vision_check(["rui.pose/v1@synthetic"])

    up = Vision({"backend": "sam3"})
    try:
        monkeypatch.setenv("RUIKIT_VISION_URL", up.url)
        assert preflight.vision_check(["rui.pose/v1", "rui.code/v1"]) is None
        assert "depth model" in preflight.vision_check(["rui.pose/v1@depth=mono"])
        assert "owlv2-sam2" in preflight.vision_check(["rui.pose/v1@owlv2-sam2"])
        # every server listed is asked: a worker may be given any of them
        monkeypatch.setenv("RUIKIT_VISION_URL", f"{up.url},http://127.0.0.1:{nobody}")
        assert "not answering" in preflight.vision_check(["rui.pose/v1"])
    finally:
        up.close()

    said = []
    code = batch.run(["opus5"], ["WK1"], ["easy"], "0", ["rui.pose/v1"], str(tmp_path),
                     engine="isaac", ping=False, say=said.append)
    assert code == 2 and "cannot run on Isaac" in said[-1]
    assert not (tmp_path / ".workers").exists()       # nothing was started


PLAN = """
defaults: {tasks: [WK1, DS1], levels: [easy], seeds: "0-1", think_seconds: 8}
experiments:
  main:
    title: the main table
    agents: [{harness: loop, model: opus5}]
    interfaces: {dialects: [pose, step], aids: [aids]}
  aids:
    agents: [{harness: loop, model: opus5, seeds: "0"}]
    interfaces: {dialects: [pose], aids: [aids, -memory, none]}
"""


def test_a_plan_runs_each_cell_once_and_reports_each_experiment(tmp_path, model):
    """Two experiments sharing cells (experiment two's "all aids" is the main
    table's): each cell runs once; each experiment gets its own tables."""
    plan_file = tmp_path / "plan.yaml"
    plan_file.write_text(PLAN)
    exps = plans.load_plan(str(plan_file))
    assert [e.name for e in exps] == ["main", "aids"]
    assert exps[1].profiles == ["rui.pose/v1+aids", "rui.pose/v1+aids-memory", "rui.pose/v1"]
    jobs = plans.select(exps)
    # main: 2 tasks x 2 seeds x 2 interfaces = 8; aids adds 2 tasks x 1 seed x 2 new interfaces
    assert len(jobs) == 8 + 4 and len(set(jobs)) == len(jobs)
    assert [j.seed for j in jobs[:8]] == [0] * 8              # seed by seed across experiments
    out = tmp_path / "out"
    said = []
    code = batch.run_jobs(jobs, str(out), engine="mock", workers=2, per_worker=4,
                          base_url=model.url, say=said.append,
                          experiments=exps)
    assert code == 0 and model.posts == 1 + 12               # a check, then each cell once
    main = (out / "report-main.md").read_text()
    assert "8 results; 0 cells not run yet" in main and "the main table" in main
    aids = (out / "report-aids.md").read_text()
    assert "6 results" in aids and "Aids taken away" in aids
    assert "| memory | +0 ± 0 pts (n=2; lost 0, gained 0)" in aids
    row = json.loads(next(out.glob("loop-opus5/WK1_easy_s0_rui.pose-v1+aids-memory/result.json"))
                     .read_text())
    assert row["aids"] == "aids-memory" and row["think_seconds"] == 8.0
    # one person's slice: one column of the main table
    step = plans.select(exps, names=["main"], columns=["step"])
    assert {j.profile for j in step} == {"rui.step/v1+aids"} and len(step) == 4


def test_the_tables_numbers():
    from ruikit.report import tables

    lo, hi = tables.wilson(7, 20)
    assert round(lo, 3) == 0.181 and round(hi, 3) == 0.567
    assert tables.wilson(0, 0) == (0.0, 1.0)
    d, half, b, c = tables.paired([(True, False)] * 3 + [(True, True)] * 5 + [(False, True)])
    assert (b, c) == (3, 1) and round(d, 3) == -0.222 and half > 0
    assert plans.interfaces({"dialects": ["free", "program"], "aids": ["aids", "none"]}) == [
        "rui.free/v1", "rui.code/v1+aids", "rui.code/v1"]


def test_a_plan_is_run_by_setting():
    """The part of the paper's plan to run is chosen by its settings -- the
    experiment, the agent, the model, the interaction form, the action
    abstraction, the interfaces by the paper's names, the tasks, the levels and
    the clock."""
    from ruikit.experiments.plans import describe, expand_tasks, load_plan, select, setting_of

    paper = load_plan("paper")
    claude = dict(names=["rui"], harnesses=["claude-code"], models=["claude-opus-5-5"])
    assert len(select(paper, **claude)) == 13 * 30                    # the easy level, seed 15
    assert {setting_of(j.profile)[0] for j in select(paper, forms=["program"], **claude)} == {"program"}
    assert len(select(paper, forms=["program"], **claude)) == 6 * 30
    assert {j.profile for j in select(paper, actions=["pose"], **claude)} == {
        "rui.pose/v1+aids", "rui.code/v1+aids"}
    assert {j.profile for j in select(paper, forms=["free"], **claude)} == {"rui.free/v1"}
    named = select(paper, columns=["program-skill", "code-joint", "rui-free"], **claude)
    assert {j.profile for j in named} == {"rui.code-skill/v1+aids", "rui.code-joint/v1+aids",
                                         "rui.free/v1"}
    some = select(paper, tasks=["CR"], levels=["easy"], **claude)
    assert {j.task for j in some} == set(expand_tasks(["CR"])) and {j.level for j in some} == {"easy"}
    assert len(some) == 6 * 13
    assert describe(some) == ["claude-code-claude-opus-5-5: 78 episodes -- 13 interfaces, 6 tasks, "
                              "levels easy, seeds 15"]
    assert select(paper, clocks=["paused"]) == []                     # every row on the main clock
    # a row's channel chooses the same rows as its agent does
    assert set(select(paper, owners=["claude-code"])) == set(select(paper, harnesses=["claude-code"]))
    with pytest.raises(ValueError):
        expand_tasks(["XX9"])


def test_a_plan_keeps_its_own_seeds_and_interfaces(tmp_path, capsys):
    for wrong in (["--seeds", "test:2"], ["--split", "dev"]):
        with pytest.raises(SystemExit):
            batch.main(["--plan", "paper", "--out", str(tmp_path), "--dry-run"] + wrong)
        assert "a plan says each row's seeds" in capsys.readouterr().err
    with pytest.raises(SystemExit):                  # a setting of a plan, without one
        batch.main(["--out", str(tmp_path), "--form", "program", "--model", "m", "--dry-run"])
    assert "choose from a plan" in capsys.readouterr().err
