"""The reporting layer, including the guards that stop a non-result being
presented as a result."""

from ruibench.core.config import RunConfig
from ruibench.reports import Results, render, wilson
from ruibench.runner import RunPlan, run_plan


def logs_for(seeds=(0, 1, 2)):
    cfg = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
    results = run_plan(RunPlan(["DI4"], levels=("easy", "hard"), seeds=seeds),
                       agent_factory=lambda i: i.reference_factory(i), config=cfg)
    return [r.log.as_dict() for r in results]


def test_wilson_interval_is_not_degenerate_at_the_extremes():
    lo, hi = wilson(20, 20)
    assert lo < 1.0 and hi == 1.0, "a 20/20 result is not certainty"
    lo, hi = wilson(0, 20)
    assert lo == 0.0 and hi > 0.0


def test_mock_engine_results_are_refused_as_a_leaderboard():
    r = Results.from_logs(logs_for())
    blockers = r.leaderboard_blockers()
    assert any("mock" in b for b in blockers)
    assert any("test double" in b for b in blockers)


def test_report_renders_and_says_so():
    html = render(logs_for(), title="t")
    assert "Not a leaderboard" in html
    assert "<svg" in html


def test_knob_curve_has_both_levels():
    r = Results.from_logs(logs_for())
    curve = r.knob_curve("DI4", "I0_direct")
    assert set(curve) == {"easy", "hard"}
