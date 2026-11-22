"""Scorer acceptance.

The design requires the scorer to be checked against seven kinds of synthetic
trajectory before it is trusted: must-pass, must-fail, boundary, dropped,
knocked-over, violated-then-recovered, and timed-out (TODO §5).  Each is built
here by hand, so the scorer is tested against intent rather than against itself.
"""

import pytest

from ruibench.core.types import ArmState, ObjectState, Pose, RobotState, WorldState
from ruibench.scoring import (
    AtRest, Custom, Finally, HoldsFor, Milestone, Never, On, Scorer,
)
from ruibench.scoring.failure import FailureLabel

TABLE = 0.8


def world(t, bowl_pos, held=None, version=0, vel=(0.0, 0.0, 0.0)):
    objects = {
        "bowl": ObjectState("bowl", Pose(bowl_pos), (0.05, 0.05, 0.03), "bowl",
                            held_by=held, lin_vel=vel),
        "plate": ObjectState("plate", Pose((0.3, 0.2, TABLE + 0.01)), (0.07, 0.07, 0.01), "plate"),
        "stove": ObjectState("stove", Pose((0.1, -0.2, TABLE + 0.01)), (0.07, 0.07, 0.01), "stove"),
    }
    arms = {"right": ArmState("robot0/right", Pose((0, 0, 1.0)))}
    return WorldState(t, objects, {"robot0": RobotState("robot0", arms)},
                      instruction_version=version)


def on_stove(t, **kw):
    return world(t, (0.1, -0.2, TABLE + 0.05), **kw)


def on_plate(t, **kw):
    return world(t, (0.3, 0.2, TABLE + 0.05), **kw)


def make_scorer():
    return Scorer(
        task_id="T", level="easy",
        success=Finally(On("bowl", "stove")) & HoldsFor(AtRest("bowl") & On("bowl", "stove"), 1.0),
        milestones=(Milestone("grasped", Custom(lambda w: w.obj("bowl").is_held, "held")),
                    Milestone("placed", On("bowl", "stove"))),
        violations=(("never_rests_on_old_target_after_change",
                     Never(Custom(lambda w: (w.instruction_version >= 1
                                             and On("bowl", "plate").holds(w)
                                             and AtRest("bowl").holds(w)),
                                  "settled on the old target"))),),
    )


def score_of(traj, **meta):
    """These are sketches of what happened, not physics -- a bowl is on the
    table, then on the stove -- so the audit's motion checks are off, as for
    the mock engine (``scoring/audit.py``)."""
    return make_scorer().score(traj, {"dynamics": False, **meta})


def test_must_pass():
    traj = [world(0.0, (0.0, 0.0, TABLE + 0.03))]
    traj += [on_stove(t / 10) for t in range(5, 26)]
    score = score_of(traj)
    assert score.success and score.progress == 1.0
    assert score.failure is FailureLabel.NONE


def test_must_fail_never_placed():
    traj = [world(t / 10, (0.0, 0.0, TABLE + 0.03)) for t in range(25)]
    score = score_of(traj)
    assert not score.success
    assert score.progress == 0.0


def test_boundary_not_stable_long_enough():
    """Placed, but only for 0.5 s of the required 1.0 s."""
    traj = [world(t / 10, (0.0, 0.0, TABLE + 0.03)) for t in range(10)]
    traj += [on_stove(1.0 + t / 10) for t in range(6)]
    score = score_of(traj)
    assert not score.success
    assert score.progress > 0.0          # the milestone was still reached


def test_dropped_on_the_way():
    traj = [world(0.0, (0.0, 0.0, TABLE + 0.03))]
    traj += [world(0.1 * t, (0.05 * t, 0.0, 1.0), held="robot0/right") for t in range(1, 5)]
    traj += [world(0.5 + 0.1 * t, (0.2, 0.0, TABLE + 0.03)) for t in range(10)]
    score = score_of(traj)
    assert not score.success
    assert score.failure is FailureLabel.EXECUTION_FAILURE


def test_violation_is_recorded_even_after_recovery():
    """Recovery is allowed, but a violation that happened stays on the record --
    that is what keeps "must not do X" inside the success criterion."""
    traj = [on_plate(t / 10, version=1) for t in range(12)]     # settles on the old target
    traj += [on_stove(1.2 + t / 10, version=1) for t in range(15)]  # then fixes it
    score = score_of(traj)
    assert not score.success
    assert score.failure is FailureLabel.VIOLATION
    assert "never_rests_on_old_target_after_change" in score.violations


def test_completing_the_old_goal_before_the_change_is_not_punished():
    traj = [on_plate(t / 10, version=0) for t in range(12)]     # legal at the time
    traj += [on_stove(1.2 + t / 10, version=1) for t in range(15)]
    score = score_of(traj)
    assert score.success, score.violations


def test_timeout_label():
    traj = [world(t / 10, (0.0, 0.0, TABLE + 0.03)) for t in range(25)]
    score = score_of(traj, timed_out=True)
    assert score.failure is FailureLabel.TIMEOUT


def test_coverage_gap_is_not_a_zero():
    score = make_scorer().score([], {"coverage_gap": "cannot express orientation"})
    assert score.failure is FailureLabel.NOT_EXPRESSIBLE
    assert not score.counted, "coverage gaps must be excluded from success rates"


def test_scoring_is_deterministic():
    """Gate B: re-scoring the same trajectory gives the same answer."""
    traj = [world(0.0, (0.0, 0.0, TABLE + 0.03))] + [on_stove(t / 10) for t in range(5, 26)]
    a = score_of(traj).as_dict()
    b = score_of(traj).as_dict()
    assert a == b


def test_rectangular_footprints_respect_orientation():
    """A 20 x 30 cm bin: a circle of radius 15 cm around it would claim points
    past its short sides."""
    import math

    from ruibench.core.types import ObjectState

    yaw = math.pi / 2                      # long side now along x
    bin_ = ObjectState("bin", Pose((0.0, 0.0, 0.8), (math.cos(yaw / 2), 0, 0, math.sin(yaw / 2))),
                       size=(0.10, 0.15, 0.07))
    assert bin_.contains_xy((0.14, 0.0))       # along the (rotated) long side
    assert not bin_.contains_xy((0.0, 0.14))   # past the (rotated) short side
    assert not bin_.contains_xy((0.12, 0.12))


def test_round_footprints_do_not_claim_their_bounding_box_corners():
    """A can at the corner of a plate's bounding box would fall off the plate."""
    from ruibench.core.types import ObjectState

    plate = ObjectState("plate", Pose((0.0, 0.0, 0.8)), size=(0.075, 0.075, 0.008),
                        footprint="round")
    assert plate.contains_xy((0.07, 0.0))
    assert not plate.contains_xy((0.07, 0.07))
