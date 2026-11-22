"""The design promises, as tests.

If one of these fails, the benchmark has quietly stopped being the thing the
design document describes.
"""

from ruibench import tasks
from ruibench.tasks import SUITES, SplitRole, Status


def test_no_validation_errors():
    findings = tasks.validate()
    errors = tasks.errors(findings)
    assert not errors, "\n".join(str(e) for e in errors)


def test_every_suite_is_six_plus_one():
    assert len(tasks.main_board()) == 6 * len(SUITES)
    assert len(tasks.held_out()) == len(SUITES)
    for suite in SUITES:
        group = tasks.by_suite(suite)[suite]
        assert len(group) == 7, suite


def test_every_id_starts_with_its_suite_prefix():
    """A task's id names its suite; a renamed task keeps its old id in
    ``formerly``."""
    for t in tasks.load_all():
        assert t.id.startswith(SUITES[t.suite][0]), (t.id, t.suite)
    card = {t.id: t for t in tasks.load_all()}
    assert (card["WK6"].name, card["WK6"].formerly.split(",")[0]) == ("tic-tac-toe", "DI1")
    assert (card["DS6"].name, card["DS6"].formerly.split(",")[0]) == ("flame test", "WK6")
    assert (card["DI1"].name, card["DI1"].formerly.split(",")[0]) == ("two-button launch", "DS6")


def test_two_arm_tasks_are_spread_over_the_suites():
    """WK4, CL1, DI1, DI5 and CR6: Dynamic Interaction has two, Discovery none."""
    two = {s: [t.id for t in tasks.by_suite(s)[s]
               if t.arms == 2 and t.split_role is not SplitRole.HOLDOUT] for s in SUITES}
    assert sum(len(v) for v in two.values()) >= 4, two
    assert sum(1 for v in two.values() if v) >= 3, two


def test_every_task_leaves_something_out_and_draws_on_three_abilities():
    """v3: the instruction and the first picture are never enough, and no task
    is a single basic ability."""
    for t in tasks.main_board():
        assert t.gap and t.gap_filled, t.id
        assert len(t.abilities) >= 3, t.id
        assert ("BI" in t.abilities) == (t.arms == 2), t.id


def test_every_suite_has_one_dev_template():
    for suite in SUITES:
        group = tasks.by_suite(suite)[suite]
        dev = [t for t in group if t.split_role is SplitRole.DEV]
        assert len(dev) == 1, f"{suite}: {[t.id for t in dev]}"


def test_held_out_tasks_are_never_implemented():
    """Implementing one spends the benchmark's only clean extrapolation check."""
    for t in tasks.held_out():
        assert t.status is Status.DEFERRED
        assert not t.impl


def test_every_task_declares_its_confounds():
    """A single declared config field does not by itself make a single-factor
    experiment; each card has to say what else moves with the knob."""
    for t in tasks.load_all():
        assert t.confounds, t.id


#: what Gate C set (10/1): GPT-5.6 Sol, Claude Opus 5 and Gemini 3.8 Flash on
#: DI2-DI6 easy through all four interfaces, median replies 7.6 / 6.8 / 4.8 s,
#: 8 s about the 75th percentile of all 4,067 (README, *Time*)
GATE_C_T_REF = 8.0


def test_speed_knobs_are_calibrated_against_the_measured_response_time():
    """Gate C measured real models' reply times and set T_ref from them; every
    speed knob is calibrated against that T_ref.  A new T_ref makes the levels
    provisional again: set ``calibrated: false`` on the cards until
    research/tref_check.py and the Isaac sweeps say the levels still hold."""
    from ruibench.core.config import DEFAULT_T_REF_SECONDS

    assert DEFAULT_T_REF_SECONDS == GATE_C_T_REF, "T_ref changed since Gate C"
    for t in (t for t in tasks.main_board() if t.abilities.get("SP") == "primary"):
        assert t.knob.calibrated, f"{t.id}: a speed knob not calibrated against T_ref"
