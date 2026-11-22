"""The skills' pick from the side and tip of the hand (v2.5): what CR4 and DS1's
hard level need, and what the skills did before left as it was."""
import math

import pytest

from ruibench.core.geometry import rotate
from ruikit import Episode
from ruikit.actions.skill import _level, _tipped
from ruikit.env.station import quat_from_rpy_deg


def _same(a, b, tol=1e-9):
    return min(max(abs(x - y) for x, y in zip(a, b)), max(abs(x + y) for x, y in zip(a, b))) < tol


@pytest.mark.parametrize("yaw", [0.0, 37.0, 90.0, -120.0])
def test_a_tip_turns_the_hand_about_its_own_level_line(yaw):
    """From pointing down at heading yaw, -90 leaves the hand level, pointing
    toward yaw - 90 with its fingers closing up and down; +90 turns it back."""
    down = quat_from_rpy_deg((0.0, 0.0, yaw))
    level = _tipped(down, -90.0)
    point, closing = rotate(level, (0.0, 0.0, 1.0)), rotate(level, (0.0, 1.0, 0.0))
    want = math.radians(yaw - 90.0)
    assert abs(point[0] - math.cos(want)) < 1e-9 and abs(point[1] - math.sin(want)) < 1e-9
    assert abs(point[2]) < 1e-9 and abs(closing[2] - 1.0) < 1e-9
    assert _same(_tipped(level, 90.0), down)


@pytest.mark.parametrize("heading", [-90.0, 0.0, 45.0, 180.0])
def test_a_side_pick_holds_the_hand_level_pointing_where_it_is_told(heading):
    class Station:
        @staticmethod
        def pose_to_world(xyz, rpy):
            from ruibench.core.types import Pose

            return Pose(tuple(xyz), quat_from_rpy_deg(rpy))

    class Ctx:
        station = Station()

    q = _level(Ctx(), (0.0, 0.0, 0.0), heading).quat
    point = rotate(q, (0.0, 0.0, 1.0))
    assert abs(math.degrees(math.atan2(point[1], point[0])) - heading) % 360.0 < 1e-6 \
        or abs(abs(math.degrees(math.atan2(point[1], point[0])) - heading) - 360.0) < 1e-6
    assert abs(point[2]) < 1e-9
    # the tip down from it is the hand pointing down at yaw heading + 90
    assert _same(_tipped(q, 90.0), quat_from_rpy_deg((0.0, 0.0, heading + 90.0)))


def _tcp(ep):
    return ep.call("robot_state", {}).data["arms"][0]["tcp"]


def test_the_skills_tip_the_hand_and_turn_it_back_on_the_mock():
    ep = Episode("CL2", level="easy", seed=0, profile="rui.skill/v1")
    try:
        yaw = _tcp(ep)["rpy_deg"][2]
        r = ep.call("skill_turn", {"angle_deg": -90, "axis": "tip"})
        assert not r.is_error, r.text
        roll, pitch, now = _tcp(ep)["rpy_deg"]
        assert abs(roll + 90.0) < 2.0 and abs(pitch) < 2.0
        assert abs(((now - yaw + 180.0) % 360.0) - 180.0) < 2.0
        r = ep.call("skill_turn", {"angle_deg": 90, "axis": "tip"})
        assert not r.is_error, r.text
        roll, pitch, _ = _tcp(ep)["rpy_deg"]
        assert abs(roll) < 2.0 and abs(pitch) < 2.0
        # a move after a tip points the fingers down again, as every skill's does
        ep.call("skill_turn", {"angle_deg": -90, "axis": "tip"})
        here = _tcp(ep)["xyz_m"]
        r = ep.call("skill_move", {"target": {"xyz_m": [here[0], here[1], here[2] + 0.03]}})
        assert not r.is_error, r.text
        roll, pitch, _ = _tcp(ep)["rpy_deg"]
        assert abs(roll) < 2.0 and abs(pitch) < 2.0
    finally:
        ep.close()


@pytest.mark.parametrize("task,level,profile,made", [
    ("CR4", "easy", "rui.skill/v1", '"approach": "side"'),
    ("DS1", "hard", "rui.code-skill/v1", "tip")])
def test_what_needed_a_side_pick_or_a_tip_is_done_through_the_skills(task, level, profile, made,
                                                                     tmp_path):
    """Before v2.5 the skills could not do these: the hand always pointed down.
    The calls show the new skill was made (CR4's reference waits between closing
    the fingers and lifting: the side pick must be seen across that wait)."""
    from ruikit.diagnostics.coverage import run_cell

    cell = run_cell(task, level, 0, profile, oracle_coordinates=True, log_dir=str(tmp_path))
    assert cell.success, (cell.reason, cell.error)
    calls = [p for p in tmp_path.glob("*.jsonl") if not p.name.endswith(".episode.jsonl")]
    assert calls and made in calls[0].read_text(), f"no {made} among the calls"


def test_results_on_the_skills_before_v250_do_not_compare():
    from ruikit.experiments import provenance as P

    row = {"provenance": {f: "x" for f in P.FIELDS} | {"frozen": "v2.4.4", "freeze_clean": True,
                                                      "process_episode": 1},
           "task": "WK1", "harness": "claude-code"}
    for profile in ("rui.skill/v1+aids", "rui.code-skill/v1+aids", "rui.free/v1"):
        assert any("skill_pick" in why for why in P.not_comparable(dict(row, profile=profile),
                                                                   "v2.5.0")), profile
    for profile in ("rui.pose/v1+aids", "rui.code-joint/v1+aids"):
        assert P.not_comparable(dict(row, profile=profile), "v2.5.0") == [], profile
