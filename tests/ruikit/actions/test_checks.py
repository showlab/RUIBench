"""The skills' completion checks (v2.5.2, ``+check``, ``+check:correct``): said
only when they fail, a correction made once, every check logged -- and nothing
changed for an interface without them."""
import json

from ruikit import Episode
from ruikit.interfaces import load_profile


def test_an_interface_without_the_check_is_as_it_was():
    for base in ("rui.skill/v1+aids", "rui.code-skill/v1+aids", "rui.free/v1"):
        prof = load_profile(base)
        assert prof.tool_params("skill_pick") == {} and prof.tool_params("skill_place") == {}
    checked = load_profile("rui.skill/v1+aids+check")
    assert checked.tools == load_profile("rui.skill/v1+aids").tools
    assert checked.tool_params("skill_pick") == {"check": "report"}
    assert load_profile("rui.code-skill/v1+aids+check:correct").tool_params("skill_place") == \
        {"check": "correct"}


def _empty_spot(ep):
    x, y, _ = ep.call("robot_state", {}).data["arms"][0]["tcp"]["xyz_m"]
    return {"xyz_m": [x, y, 0.03]}


def test_a_pick_that_caught_nothing_is_said():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.skill/v1+aids+check")
    try:
        r = ep.call("skill_pick", {"target": _empty_spot(ep)})
        assert not r.is_error, r.text
        assert "check after the pick: the fingers closed on nothing" in r.text
        assert r.data["check"] == {"after": "pick", "ok": False, "why": "the fingers closed on nothing"}
    finally:
        ep.close()


def test_correct_picks_once_more_and_says_how_it_went():
    ep = Episode("WK1", level="easy", seed=0, profile="rui.skill/v1+aids+check:correct")
    try:
        t0 = ep.sim_time
        r = ep.call("skill_pick", {"target": _empty_spot(ep)})
        assert r.data["check"]["corrected"] is True and r.data["check"]["ok_after_correcting"] is False
        assert "the skill picked once more" in r.text
        once = Episode("WK1", level="easy", seed=0, profile="rui.skill/v1+aids+check")
        try:
            t1 = once.sim_time
            once.call("skill_pick", {"target": _empty_spot(once)})
            assert ep.sim_time - t0 > once.sim_time - t1          # the second pick took its time
        finally:
            once.close()
    finally:
        ep.close()


def test_every_check_is_logged_with_what_the_simulator_knows(tmp_path):
    from ruikit.diagnostics.coverage import run_cell

    folder = tmp_path / "cell"                       # made by the episode only at its end
    cell = run_cell("WK1", "easy", 0, "rui.skill/v1+aids+check", oracle_coordinates=True,
                    log_dir=str(folder))
    assert cell.success, (cell.reason, cell.error)
    rows = [json.loads(line) for line in (folder / "checks.jsonl").read_text().splitlines()]
    assert rows and all(r["kind"] == "pick" and r["verdict"]["ok"] for r in rows)
    assert all(r["truth"]["ok"] for r in rows)
    plain = run_cell("WK1", "easy", 0, "rui.skill/v1+aids", oracle_coordinates=True,
                     log_dir=str(tmp_path / "plain"))
    assert (plain.success, plain.replies, plain.tool_calls) == (cell.success, cell.replies, cell.tool_calls)
    assert not (tmp_path / "plain" / "checks.jsonl").exists()



def test_the_place_check_compares_what_was_seen_before_and_after():
    """v2.5.3: from every camera, only cells seen both times compare: a thin ring on a stack
    counts; cells first seen after (the wrist camera now over the target) do not."""
    import numpy as np

    from ruikit.actions.checks import RISEN_CELLS, _risen

    rng = np.random.default_rng(0)

    def disc(z, r0=0.0, r1=0.04, n=4000):
        a, d = rng.uniform(0, 2 * np.pi, n), np.sqrt(rng.uniform(r0 ** 2, r1 ** 2, n))
        return np.c_[0.5 + d * np.cos(a), d * np.sin(a), z + rng.normal(0, 0.001, n)]

    stack = disc(0.05)
    ring = np.r_[disc(0.05, 0.0, 0.006), disc(0.058, 0.006, 0.02), disc(0.05, 0.02, 0.04)]
    xy, r, cap = (0.5, 0.0), 0.04, 0.3
    assert _risen(stack, ring, xy, r, cap)[0] >= RISEN_CELLS
    assert _risen(stack, disc(0.05), xy, r, cap)[0] < RISEN_CELLS
    lid = disc(0.068, 0.02, 0.04)
    assert _risen(lid, disc(0.068), xy, r, cap)[0] < RISEN_CELLS     # the hole first seen after


def test_the_fixed_camera_sees_what_came_into_a_box():
    """v2.5.3: pixel by pixel, a block sunk into a hole shows where the box's far wall was
    before; an unchanged picture, or the robot's own pixels, show nothing new."""
    from types import SimpleNamespace as NS

    import numpy as np

    from ruikit.actions.checks import NEW_CELLS, _appeared

    n = 120
    xs, ys = np.meshgrid(np.linspace(0.45, 0.55, n), np.linspace(-0.05, 0.05, n))
    in_hole = np.hypot(xs - 0.5, ys) < 0.02
    wall = np.where(in_hole, 0.02, 0.068)                     # through the hole: the far wall, low
    sunk = np.where(in_hole, 0.057, 0.068)                    # the block's top, just under the lid
    points = np.stack([xs, ys, sunk], axis=-1)
    nobody = np.zeros((n, n), dtype=bool)
    views = {"b": NS(height=wall, points=points, robot=nobody),
             "a": NS(height=sunk, points=points, robot=nobody),
             "hand": NS(height=sunk, points=points, robot=in_hole)}
    ctx = NS(station=NS(two_arms=False, origin=(0.0, 0.0, 0.0)),
             perception=NS(view=lambda f: views[f.id]))
    look = lambda i: [NS(camera="front", id=i), NS(camera="wrist", id="w")]
    new, compared = _appeared(ctx, look("b"), look("a"), (0.5, 0.0), 0.04, below=0.2)
    assert new >= NEW_CELLS and compared > new
    assert _appeared(ctx, look("a"), look("a"), (0.5, 0.0), 0.04, below=0.2)[0] == 0
    assert _appeared(ctx, look("b"), look("hand"), (0.5, 0.0), 0.04, below=0.2)[0] == 0
    assert _appeared(ctx, look("b"), look("a"), (0.5, 0.0), 0.04, below=0.05)[0] == 0   # the arm's height


def test_where_a_dropped_thing_fell_is_found_by_the_fixed_camera():
    """v2.5.3: +check:correct looks for what fell near the target among the fixed camera's new
    pixels, not among what the wrist camera first saw after."""
    from types import SimpleNamespace as NS

    import numpy as np

    from ruikit.actions.checks import _dropped

    n = 200
    xs, ys = np.meshgrid(np.linspace(0.40, 0.60, n), np.linspace(-0.10, 0.10, n))
    table = np.zeros((n, n))
    fell = np.where(np.hypot(xs - 0.565, ys - 0.0) < 0.02, 0.04, 0.0)    # 6.5 cm off the target
    points = np.stack([xs, ys, fell], axis=-1)
    nobody = np.zeros((n, n), dtype=bool)
    views = {"b": NS(height=table, points=points, robot=nobody),
             "a": NS(height=fell, points=points, robot=nobody)}
    ctx = NS(station=NS(two_arms=False, origin=(0.0, 0.0, 0.0),
                        describe_pose=lambda pose: {"xyz_m": [0.5, 0.0, 0.2]}),
             perception=NS(view=lambda f: views[f.id]), tcp=lambda arm: None)
    look = lambda i: [NS(camera="front", id=i)]
    x, y, top = _dropped(ctx, "right", look("b"), look("a"), (0.5, 0.0), 0.03)
    assert abs(x - 0.565) < 0.015 and abs(y) < 0.015 and abs(top - 0.04) < 0.005
    assert _dropped(ctx, "right", look("b"), look("b"), (0.5, 0.0), 0.03) is None
