"""The Harness components' implementations, on the painted mock: each reads pixels and
the robot's own readings only."""

import math

import pytest

np = pytest.importorskip("numpy")
import painted_mock as pixels  # noqa: E402


def _look(ep, camera="front"):
    return ep.call("camera_look", {"cameras": [camera]}).data["frames"][0]["frame_id"]


@pytest.mark.parametrize("style", ["dots", "masks", "boxes"])
def test_marks_are_drawn_three_ways(style):
    ep = pixels.episode("DS4", profile=f"rui.pose/v1+marks:{style}")
    try:
        fid = _look(ep)
        r = ep.call("vision_marks", {"frame_id": fid, "kind": "objects"})
        assert not r.is_error and len(r.data["marks"]) >= 3, r.text
        png = r.images[0].png
        from io import BytesIO

        from PIL import Image

        drawn = np.asarray(Image.open(BytesIO(png)).convert("RGB")).astype(int)
        plain = np.asarray(ep.frames.get(fid).image)[..., :3].astype(int)
        changed = (np.abs(drawn - plain).sum(axis=-1) > 0).sum()
        assert changed > 200, (style, changed)               # something was drawn
        if style != "dots":
            # an outline or a box reaches past the dot in the middle
            from ruikit.harness.components.marks import _draw_marks

            dots = np.asarray(Image.open(BytesIO(_draw_marks(ep.frames.get(fid), r.data["marks"],
                                                             "dots"))).convert("RGB")).astype(int)
            assert changed > 1.5 * (np.abs(dots - plain).sum(axis=-1) > 0).sum(), style
    finally:
        ep.close()


def test_a_vlm_points_and_its_time_is_charged(monkeypatch):
    from ruikit.extras.backends import ASKERS

    class Fake:
        name = "fake-vlm"

        def ask(self, images, question, system=""):
            assert images and "coin" in question and "512 pixels wide" in question
            return 'Sure: {"points": [[100, 200], [9999, 5], ["x", 1]]}'

    monkeypatch.setitem(ASKERS, "fake-vlm", Fake)
    ep = pixels.episode("DS4", profile="rui.pose/v1+pointer:vlm/fake-vlm")
    try:
        fid = _look(ep)
        r = ep.call("vision_point", {"frame_id": fid, "query": "coin"})
        assert not r.is_error, r.text
        assert r.data["points"] == [{"u": 100, "v": 200, "label": "coin"}]   # the rest dropped
    finally:
        ep.close()


#: implementations that need an optional package (``pip install -e ".[dev]"`` has them)
NEEDS = {"template": "cv2", "antipodal": "scipy", "pca": "scipy"}


@pytest.mark.parametrize("impl", ["redetect", "template"])
def test_a_still_thing_is_followed_both_ways(impl):
    if impl in NEEDS:
        pytest.importorskip(NEEDS[impl])
    ep = pixels.episode("DS4", profile=f"rui.pose/v1+track:{impl}")
    try:
        fid = _look(ep)
        coin = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"][0]
        start = ep.call("vision_track", {"action": "start", "frame_id": fid, "id": coin["id"],
                                         "hz": 4.0}).data
        ep.call("robot_wait", {"seconds": 1.5})
        read = ep.call("vision_track", {"action": "read", "track": start["track"]}).data
        assert not read["lost"] and read["samples"] >= 4, (impl, read)
        assert math.dist(read["xyz_m"][:2], start["xyz_m"][:2]) < 0.02
        assert max(abs(v) for v in read["velocity_mps"]) < 0.05
    finally:
        ep.close()


def test_the_scene_memory_keeps_what_perception_said():
    ep = pixels.episode("DS4", profile="rui.pose/v1+memory:scene")
    try:
        fid = _look(ep)
        coins = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"]
        pose = ep.call("vision_pose", {"frame_id": fid, "id": coins[0]["id"]}).data
        r = ep.call("memory_scene", {"label": "coin"})
        assert not r.is_error, r.text
        things = r.data["things"]
        assert len(things) == 1 and things[0]["xyz_m"] == pose["xyz_m"]
        assert len(r.data["unplaced"]) == len(coins) - 1     # found, never posed
        assert ep.call("memory_scene", {"label": "nothing"}).data == {"things": [], "unplaced": []}
    finally:
        ep.close()


@pytest.mark.parametrize("impl", ["box", "antipodal", "pca"])
def test_grasps_three_ways_on_what_was_measured(impl):
    if impl in NEEDS:
        pytest.importorskip(NEEDS[impl])
    ep = pixels.episode("DS4", profile=f"rui.pose/v1+grasps:{impl}")
    try:
        fid = _look(ep)
        coin = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"][0]
        pose = ep.call("vision_pose", {"frame_id": fid, "id": coin["id"]}).data
        r = ep.call("plan_grasps", {"target": {"frame_id": fid, "id": coin["id"]}, "top_k": 3})
        assert not r.is_error, r.text
        best = r.data["grasps"][0]
        assert best["method"] == impl and best["score"] > 0
        assert math.dist(best["xyz_m"][:2], pose["xyz_m"][:2]) < 0.02
        assert -90.0 <= best["yaw_deg"] <= 90.0 or impl == "box"
        assert 0.0 < best["opening_m"] < 0.08
    finally:
        ep.close()


@pytest.mark.parametrize("impl", ["lift", "rrt"])
def test_paths_round_what_the_cameras_saw(impl):
    """From above one coin to above another, low: the lift planner goes over
    what stands between, RRT-Connect round or over it -- and neither path's
    hand comes into what was seen."""

    ep = pixels.episode("DS4", profile=f"rui.pose/v1+paths:{impl}+motion")
    try:
        fid = _look(ep)
        coins = ep.call("vision_detect", {"frame_id": fid, "query": "coin"}).data["objects"]
        a, b = (ep.call("vision_pose", {"frame_id": fid, "id": c["id"]}).data["xyz_m"]
                for c in (coins[0], coins[-1]))
        ep.call("robot_move_to", {"xyz_m": [a[0], a[1], 0.08]})
        _look(ep)
        r = ep.call("plan_motion", {"xyz_m": [b[0], b[1], 0.05]})
        assert not r.is_error, r.text
        pts = r.data["waypoints"]
        assert 1 <= len(pts) <= 12 and r.data["planner"] == impl
        assert pts[-1]["xyz_m"] == pytest.approx([b[0], b[1], 0.05], abs=1e-3)
        if r.data["clearance_m"] is not None:
            assert r.data["clearance_m"] > 0.0
        moved = ep.call("robot_waypoints", {"points": pts})
        assert not moved.is_error, moved.text
    finally:
        ep.close()


def test_a_path_stays_under_how_high_the_arm_reaches():
    """The ARX X5 reaches lower the farther out (v1.15): the lift planner's way
    across stays under that, or is refused when clearing what was seen needs
    more; RRT-Connect keeps its samples, and so its path, under it.  Without a
    ceiling -- the FR3 -- both plan exactly as before."""
    from ruikit.harness.components.planning.paths import _lift_path, _rrt_path, _TooHigh

    ceil = lambda xy: 0.20 - 0.28 * max(0.0, float(xy[0]))       # noqa: E731 (0.074 at 0.45)
    start, goal = np.array([0.1, 0.0, 0.15]), np.array([0.45, 0.0, 0.03])

    def thing(x, half, tall):                                       # seen points of a box
        g = np.linspace(-1.0, 1.0, 5)
        return np.array([[x + 0.02 * a, half * b, tall * (c + 1) / 2] for a in g for b in g for c in g])

    over = _lift_path(start, goal, thing(0.27, 0.02, 0.06), 0.6, ceil)
    assert isinstance(over, _TooHigh) and over.need == pytest.approx(0.11)
    assert over.ceiling == pytest.approx(ceil((0.45, 0.0)))
    low = thing(0.27, 0.02, 0.02)
    path = _lift_path(start, goal, low, 0.6, ceil)
    assert all(q[2] <= ceil(q) + 1e-9 for q in path[1:]), path
    assert max(q[2] for q in path) == pytest.approx(0.15)           # it only starts higher
    assert _lift_path(start, goal, low, 0.6, lambda xy: math.inf) == _lift_path(start, goal, low, 0.6)
    wall = thing(0.27, 0.1, 0.12)                                   # too tall to pass over here
    path = _rrt_path(start, goal, wall, 0.6, np.random.default_rng(0), ceil)
    assert path is not None and all(q[2] <= ceil(q) + 1e-9 for q in path[1:]), path
    assert max(abs(q[1]) for q in path) > 0.1                       # round it
    assert _rrt_path(start, goal, wall, 0.6, np.random.default_rng(0), lambda xy: math.inf) \
        == _rrt_path(start, goal, wall, 0.6, np.random.default_rng(0))


def test_plan_motion_refuses_what_the_x5_cannot_reach():
    """On the two-arm station a goal higher than the arm reaches there is
    refused in plan_ik's words, and a path far out keeps under the ceiling."""
    from ruibench.engines import robots

    ep = pixels.episode("WK4", profile="rui.pose/v1+paths:lift+motion")
    try:
        st = ep.station
        base = st.to_station(robots.arm_base(st.scene.robots[0], "right", st.table_height))
        far = (base[0] + 0.45, base[1])
        # the hand pointing down, as the painted mock draws it (folded, the
        # painted hand and the one the depth leaves out would not agree)
        ep.call("robot_move_to", {"arm": "right", "xyz_m": [base[0] + 0.25, base[1], 0.10]})
        _look(ep, "head")
        r = ep.call("plan_motion", {"xyz_m": [far[0], far[1], 0.12]})
        assert r.error == "unreachable", r.text
        assert r.text == "not reachable: " + st.too_high(st.ceiling("right", far)), r.text
        r = ep.call("plan_motion", {"xyz_m": [far[0], far[1], 0.03]})
        assert not r.is_error, r.text
        assert all(w["xyz_m"][2] <= st.ceiling("right", w["xyz_m"][:2]) + 1e-3
                   for w in r.data["waypoints"]), r.data["waypoints"]
    finally:
        ep.close()

