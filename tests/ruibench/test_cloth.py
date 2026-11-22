"""Cloth (``engines/cloth.py``): what a cloth is made of, the rule a hand takes
it by, the fold that stands in for physics on the mock, and its place in the
state and in the log.

The Isaac side -- the particle cloth, the attachment to the hand -- is checked
on Isaac (``research/cloth_dev``); here: everything both engines share.
"""

import math

import pytest

from ruibench.core.types import GRASP_DOWN, Pose
from ruibench.engines import cloth as C
from ruibench.engines.mock import MockEngine
from ruibench.runner import snapshot
from ruibench.tasks import clearance, sightlines
from ruibench.tasks.impl._common import LEFT, RIGHT, STATION_X, TABLE, station_scene

H = 0.15
SQUARE = ((-H, -H), (H, -H), (H, H), (-H, H))
CORNERS = (("near_right", (-H, -H)), ("far_right", (H, -H)), ("far_left", (H, H)),
           ("near_left", (-H, H)))
#: a small T: a body 20 cm wide and 26 long, sleeves out to 32 cm across its top 8 cm
TEE = ((-0.13, -0.10), (0.05, -0.10), (0.05, -0.16), (0.13, -0.16), (0.13, 0.16), (0.05, 0.16),
       (0.05, 0.10), (-0.13, 0.10))


def _towel(yaw=0.0, ahead=0.32):
    spec = C.cloth_spec(SQUARE, CORNERS)
    pose = Pose((STATION_X + ahead, 0.0, TABLE + spec.radius),
                (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))
    return C.cloth_body("towel", spec, pose, category="towel", color="fabric:navy")


def _scene(body=None, extra=()):
    return station_scene("towel_dev", [body or _towel(), *extra], two_arms=True)


def _down(x, y, z):
    return Pose((x, y, TABLE + z), GRASP_DOWN)


def _flat(points):
    """Points as one list of numbers (``approx`` does not look into nested tuples)."""
    return [v for p in points for v in p]


# -- what a cloth is made of ------------------------------------------------------------
def test_a_towel_is_a_grid_of_points_its_marks_three_centimetres_apart():
    spec = C.cloth_spec(SQUARE, CORNERS)
    assert (len(spec.nodes), len(spec.triangles), len(spec.marks)) == (61 * 61, 2 * 60 * 60, 11 * 11)
    assert spec.bounds() == pytest.approx((-H, H, -H, H))
    # every key point is a corner node, and a mark
    for name, at in CORNERS:
        node = dict(spec.keys)[name]
        assert spec.nodes[node] == pytest.approx(at) and spec.marks[spec.key(name)] == node
    # neighbouring marks are 3 cm apart, and every triangle is half a cell
    xs = sorted({round(spec.nodes[m][0], 6) for m in spec.marks})
    assert all(b - a == pytest.approx(0.03) for a, b in zip(xs, xs[1:]))
    for a, b, c in spec.triangles[::97]:
        (ax, ay), (bx, by), (cx, cy) = spec.nodes[a], spec.nodes[b], spec.nodes[c]
        assert abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax)) / 2 == pytest.approx(C.CELL ** 2 / 2)
    assert hash(spec) == hash(C.cloth_spec(SQUARE, CORNERS))             # it goes into a scene's key


def test_an_outline_with_corners_cut_out_keeps_only_what_is_inside():
    spec = C.cloth_spec(TEE, (("sleeve_left", (0.09, 0.16)), ("hem_left", (-0.13, 0.10))))
    assert all(C.inside(TEE, p) for p in spec.nodes)
    # nothing beside the body below the sleeves
    assert not any(x < 0.05 - 1e-9 and abs(y) > 0.10 + 1e-9 for x, y in spec.nodes)
    area = len(spec.triangles) * C.CELL ** 2 / 2
    assert area == pytest.approx(0.26 * 0.20 + 2 * 0.08 * 0.06, rel=0.01)
    assert spec.nodes[dict(spec.keys)["sleeve_left"]] == pytest.approx((0.09, 0.16))
    with pytest.raises(ValueError):
        C.ClothSpec(spec.nodes, spec.triangles, marks=(0, 1), keys=(("corner", 5),))


def test_a_cloths_body_is_the_flat_box_round_it_and_the_checks_take_it_as_that():
    body = _towel()
    assert body.shape == C.SHAPE and C.is_cloth(body) and not body.fixed
    assert body.size == pytest.approx((H, H, C.CELL / 2))
    assert body.attrs["graspable"] is False
    scene = _scene(body)
    assert C.scene_problems(scene) == []
    assert clearance.check_scene(scene) == [] and clearance.falls_through(scene) == []
    assert clearance.out_of_reach(scene) == [] and sightlines.check_scene(scene) == []
    # another cloth is another stage; the same cloth somewhere else is not
    other = C.cloth_body("towel", C.cloth_spec(SQUARE, CORNERS, mark_every=3), body.pose)
    assert _scene(other).structure_key() != scene.structure_key()
    assert _scene(_towel(yaw=0.1, ahead=0.30)).structure_key() == scene.structure_key()


def test_a_cloths_scene_has_nothing_that_is_moved_by_writing_its_pose():
    from ruibench.engines.base import ArticulationSpec, BodySpec

    box = BodySpec("tray", "box", (0.05, 0.05, 0.01), Pose((0.2, 0.3, TABLE + 0.01)), fixed=True)
    assert any("tray is fixed" in line for line in C.scene_problems(_scene(extra=[box])))
    loose = BodySpec("peg", "box", (0.01, 0.01, 0.01), Pose((0.2, 0.3, TABLE + 0.01)))
    assert C.scene_problems(_scene(extra=[loose])) == []
    tilted = C.cloth_body("towel", C.cloth_spec(SQUARE), Pose((0.0, 0.0, TABLE + 0.0025), GRASP_DOWN))
    assert any("starts flat" in line for line in C.scene_problems(_scene(tilted)))
    high = C.cloth_body("towel", C.cloth_spec(SQUARE), Pose((0.0, 0.0, TABLE + 0.05)))
    assert any("on the table" in line for line in C.scene_problems(_scene(high)))
    two = [C.cloth_body("rag", C.cloth_spec(SQUARE), Pose((0.0, 0.3, TABLE + 0.0025)))]
    assert any("one a scene" in line for line in C.scene_problems(_scene(extra=two)))
    import dataclasses
    jointed = dataclasses.replace(_scene(extra=[loose]),
                                  articulations=(ArticulationSpec("slide", parent="peg"),))
    assert any("joints" in line for line in C.scene_problems(jointed))


# -- the grasp ------------------------------------------------------------------------
def test_a_closing_hand_takes_the_patch_under_its_fingertips_and_only_near_them():
    body = _towel()
    points = C.flat_points(body)
    cx, cy, cz = body.pose.pos
    top = cz + C.CELL / 2                                    # the cloth's upper side
    # fingertips at the cloth's top: the tool centre point is 8 mm above them
    patch = C.pinched(points, Pose((cx, cy, top + C.TIP_BELOW_TCP), GRASP_DOWN))
    xs = [points[k][0] - cx for k in patch]
    ys = [points[k][1] - cy for k in patch]
    assert max(xs) == pytest.approx(0.010) and min(xs) == pytest.approx(-0.010)      # across: +-1.4 cm
    assert max(ys) == pytest.approx(0.010) and min(ys) == pytest.approx(-0.010)      # along: +-1.1 cm
    assert len(patch) == 5 * 5
    # 9 mm up it still takes hold, 2 cm up it closes on air
    assert C.pinched(points, Pose((cx, cy, top + C.TIP_BELOW_TCP + 0.006), GRASP_DOWN))
    assert C.pinched(points, Pose((cx, cy, top + C.TIP_BELOW_TCP + 0.02), GRASP_DOWN)) == []
    # beside the cloth: nothing; at its edge: the part of the patch that is cloth
    assert C.pinched(points, Pose((cx + H + 0.03, cy, top + C.TIP_BELOW_TCP), GRASP_DOWN)) == []
    assert 0 < len(C.pinched(points, Pose((cx + H, cy, top + C.TIP_BELOW_TCP), GRASP_DOWN))) < 25
    # two layers under the fingers: both
    lifted = points + [(x, y, z + C.CELL) for x, y, z in points]
    both = C.pinched(lifted, Pose((cx, cy, top + C.CELL + C.TIP_BELOW_TCP), GRASP_DOWN))
    assert len(both) == 50


# -- the fold that stands in for physics ------------------------------------------------
def test_what_is_let_go_at_b_after_being_taken_at_a_folds_along_the_bisector():
    pts = [(x / 20, y / 10) for x in range(-4, 5) for y in range(-2, 3)]
    layers = [0] * len(pts)
    # one hand, the far middle to the near middle: the crease is the line x = 0
    crease = C.crease_of([((0.2, 0.0), (-0.2, 0.0))])
    assert C.side((0.1, 0.3), crease) * C.side((0.1, -0.3), crease) > 0
    assert C.side((0.1, 0.0), crease) * C.side((-0.1, 0.0), crease) < 0
    out, lay = C.fold_by(pts, layers, [((0.2, 0.0), (-0.2, 0.0))])
    for (x, y), (fx, fy), layer in zip(pts, out, lay):
        assert (fx, fy) == pytest.approx((-abs(x), y)) and layer == (1 if x > 0 else 0)
    # two hands, each its own corner: the same fold
    two, _ = C.fold_by(pts, layers, [((0.2, 0.2), (-0.2, 0.2)), ((0.2, -0.2), (-0.2, -0.2))])
    assert _flat(two) == pytest.approx(_flat(out))
    # folded again the same way, along x = -0.1: four layers, and of what went
    # over, what was uppermost is undermost
    again, lay2 = C.fold_by(out, lay, [((0.0, 0.2), (-0.2, 0.2)), ((0.0, -0.2), (-0.2, -0.2))])
    assert sorted(set(lay2)) == [0, 1, 2, 3]
    assert min(x for x, _ in again) == pytest.approx(-0.2) and max(x for x, _ in again) == pytest.approx(-0.1)
    for (x, y), (fx, fy), layer in zip(pts, again, lay2):
        if x == pytest.approx(0.05):          # went over twice: on top after the first fold
            assert (fx, fy) == pytest.approx((-0.15, y)) and layer == 2
        if x == pytest.approx(-0.05):         # went over once: under it then, over it now
            assert (fx, fy) == pytest.approx((-0.15, y)) and layer == 3
    # a hand that did not move folds nothing
    assert C.fold_by(pts, layers, [((0.2, 0.0), (0.2, 0.0))]) == (pts, layers)


def test_a_cloth_pulled_by_its_edge_away_from_itself_goes_along_whole():
    pts = [(x / 10, y / 10) for x in range(-2, 3) for y in range(-2, 3)]
    out, lay = C.fold_by(pts, [0] * len(pts), [((0.2, 0.0), (0.8, 0.0))])
    assert _flat(out) == pytest.approx(_flat((x + 0.6, y) for x, y in pts)) and set(lay) == {0}


def test_the_stand_in_carries_a_patch_and_folds_when_the_last_hand_lets_go():
    body = _towel()
    cloth = C.FlatCloth(body)
    spec = C.cloth_of(body)
    cx, cy, cz = body.pose.pos
    start = cloth.state()
    assert _flat(start.points) == pytest.approx(_flat(C.flat_points(body, spec.marks)))
    assert start.pose.pos == pytest.approx(body.pose.pos) and start.size == pytest.approx(body.size)
    grab = {LEFT: (cx + H - 0.01, cy + H - 0.01), RIGHT: (cx + H - 0.01, cy - H + 0.01)}
    z = cz + C.CELL / 2 + C.TIP_BELOW_TCP
    for arm, (x, y) in grab.items():
        assert cloth.take(arm, Pose((x, y, z), GRASP_DOWN))
    assert cloth.held_by == LEFT and not cloth.take("nobody", Pose((cx, cy, z + 0.05), GRASP_DOWN))
    # the corner marks are in the patches and go with the hands; the rest stays
    for arm, (x, y) in grab.items():
        cloth.carry(arm, Pose((x - 0.15, y, z + 0.08), GRASP_DOWN), (-0.2, 0.0, 0.0))
    mid = cloth.state()
    assert C.key_point(mid, "far_left")[2] == pytest.approx(cz + 0.08)
    assert C.key_point(mid, "near_left") == pytest.approx(C.key_point(start, "near_left"))
    assert mid.speed == pytest.approx(0.2) and mid.size[2] > 0.03
    # one hand lets go: nothing folds yet
    for arm, (x, y) in grab.items():
        cloth.carry(arm, Pose((x - 0.28, y, z + 0.01), GRASP_DOWN))
    cloth.let_go(LEFT)
    assert cloth.held_by == RIGHT and set(cloth.layers) == {0}
    cloth.let_go(RIGHT)
    done = cloth.state()
    assert done.held_by is None and done.speed == 0.0
    # far corners on the near corners, a layer up; the box round it half as long
    for side in ("left", "right"):
        far, near = C.key_point(done, f"far_{side}"), C.key_point(done, f"near_{side}")
        assert far[:2] == pytest.approx(near[:2], abs=1e-9) and far[2] - near[2] == pytest.approx(C.CELL)
    assert done.size == pytest.approx((H / 2, H, C.CELL))
    assert done.pose.pos == pytest.approx((cx - H / 2, cy, cz + C.CELL / 2))


# -- in the mock engine, the state and the log ---------------------------------------------
def _go(engine, targets, grip=None):
    for arm, pose in targets.items():
        engine.set_arm_target(arm, pose)
    for _ in range(400):
        engine.advance(0.05)
        if all(engine.arm_at_target(a) for a in targets):
            break
    if grip is not None:
        for arm in targets:
            engine.set_gripper(arm, grip)
        engine.advance(0.6)
    return engine.state()


def test_on_the_mock_two_hands_fold_a_towel_and_the_log_gives_it_back():
    scene = _scene()
    engine = MockEngine()
    assert engine.capabilities.cloth and not engine.capabilities.dynamics
    engine.load(scene, 0)
    trajectory = [engine.reset()]
    towel = trajectory[0].obj("towel")
    assert towel.points is not None and len(towel.points) == 121 and towel.held_by is None
    far = {LEFT: C.key_point(towel, "far_left"), RIGHT: C.key_point(towel, "far_right")}
    near = {LEFT: C.key_point(towel, "near_left"), RIGHT: C.key_point(towel, "near_right")}
    inward = {LEFT: -0.01, RIGHT: 0.01}
    # closed 5 cm over the cloth: on air, and a hand closed on air takes nothing on the way down
    w = _go(engine, {a: _down(p[0] - 0.01, p[1] + inward[a], 0.07) for a, p in far.items()}, grip=1.0)
    assert all(w.arm(a).holding is None for a in far)
    w = _go(engine, {a: _down(p[0] - 0.01, p[1] + inward[a], 0.012) for a, p in far.items()})
    assert all(w.arm(a).holding is None for a in far)
    w = _go(engine, {a: _down(p[0] - 0.01, p[1] + inward[a], 0.012) for a, p in far.items()}, grip=0.0)
    # closed at the cloth: both hands have it
    w = _go(engine, {a: _down(p[0] - 0.01, p[1] + inward[a], 0.012) for a, p in far.items()}, grip=1.0)
    trajectory.append(w)
    assert all(w.arm(a).holding == "towel" for a in far) and w.obj("towel").held_by == LEFT
    assert all(w.arm(a).public().grasping for a in far)
    w = _go(engine, {a: _down(p[0] - 0.15, p[1] + inward[a], 0.09) for a, p in far.items()})
    trajectory.append(w)
    assert C.key_point(w.obj("towel"), "far_left")[2] > TABLE + 0.07
    w = _go(engine, {a: _down(p[0] + 0.01, p[1] + inward[a], 0.02) for a, p in near.items()}, grip=0.0)
    trajectory.append(w)
    towel = w.obj("towel")
    assert towel.held_by is None and all(w.arm(a).holding is None for a in far)
    for side in ("left", "right"):
        gap = math.dist(C.key_point(towel, f"far_{side}")[:2], C.key_point(towel, f"near_{side}")[:2])
        assert gap < 0.001
    assert towel.size == pytest.approx((H / 2, H, C.CELL), abs=1e-3)
    assert engine.visible_objects("agentview") == ("towel",)
    # the log: every snapshot gives the cloth back, marks and box and all
    back = snapshot.load_all(snapshot.dump_all(trajectory, scene), scene)
    for a, b in zip(trajectory, back):
        x, y = a.obj("towel"), b.obj("towel")
        assert _flat(y.points) == pytest.approx(_flat(x.points), abs=1e-4)
        assert y.size == pytest.approx(x.size, abs=1e-4)
        assert y.pose.pos == pytest.approx(x.pose.pos, abs=1e-4) and y.held_by == x.held_by
        assert C.key_point(y, "far_left") == pytest.approx(C.key_point(x, "far_left"), abs=1e-4)
    # a rigid body's snapshot has no points
    assert "points" in snapshot.dump(trajectory[0], None, scene)["objects"]["towel"]


def test_on_the_mock_a_loose_thing_beside_the_cloth_is_still_taken_as_a_box():
    from ruibench.engines.base import BodySpec

    peg = BodySpec("peg", "box", (0.012, 0.012, 0.012), Pose((STATION_X + 0.30, -0.25, TABLE + 0.012)))
    engine = MockEngine()
    engine.load(_scene(extra=[peg]), 0)
    w = engine.reset()
    at = w.obj("peg").pose.pos
    w = _go(engine, {RIGHT: _down(at[0], at[1], 0.012)}, grip=1.0)
    assert w.arm(RIGHT).holding == "peg" and w.obj("towel").held_by is None
    assert w.obj("peg").points is None
