"""CR6, folding a T-shirt: the garment and what counts as folded.

What a cloth is made of, the rule a hand takes it by and the fold that stands
in for physics are the engines' (``test_cloth.py``).  Here: the garments'
cut and folds, how a fold is scored -- by its shape, not by where it lies --
the way a hand goes to make one, and the task on the development engine.
"""

import math
from collections import Counter

import pytest

from ruibench.core.config import RunConfig
from ruibench.engines import cloth as C
from ruibench.engines import prints
from ruibench.policies.agents import build_agent
from ruibench.runner import RunPlan, run_plan
from ruibench.tasks.impl import _garment as G
from ruibench.tasks.impl import cr6_fold as CR6
from ruibench.tasks.impl._common import LEFT, RIGHT, STATION_X, TABLE
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
LEVELS = ("easy", "hard")


def _run(level, spec="reference", seeds=(0, 1, 2)):
    return run_plan(RunPlan(["CR6"], levels=(level,), seeds=seeds),
                    agent_factory=lambda i: build_agent(spec, i), config=CFG)


def _at(g, pts, name):
    return pts[g.spec().key(name)]


def _rim(spec):
    """The mesh's own edge: the sides of triangles that no other triangle shares."""
    sides = Counter(tuple(sorted((tri[k], tri[(k + 1) % 3]))) for tri in spec.triangles for k in range(3))
    return [side for side, n in sides.items() if n == 1]


def _area_of(outline):
    n = len(outline)
    return abs(sum(outline[k][0] * outline[(k + 1) % n][1] - outline[(k + 1) % n][0] * outline[k][1]
                   for k in range(n))) / 2


# -- a cloth cut to its outline -----------------------------------------------------
def test_fold_a_square_cut_is_the_plain_grid():
    g = G.towel()
    spec = g.spec()
    plain = C.cloth_spec(g.outline, g.keypoints)
    assert len(spec.nodes) == len(plain.nodes)
    assert max(math.dist(a, b) for a, b in zip(spec.nodes, plain.nodes)) < 1e-12
    assert spec.marks == plain.marks and spec.keys == plain.keys
    assert len(spec.nodes) == 49 * 49 and len(spec.triangles) == 2 * 48 * 48     # 24 cm at 5 mm
    assert len(spec.marks) == 9 * 9                                              # every 3 cm


@pytest.mark.parametrize("make", [G.towel, G.tshirt])
def test_fold_a_cloth_is_cut_to_its_outline_and_not_to_the_grid(make):
    """Its edge is the outline's -- no steps along a slanted shoulder or round
    the neck --, it covers what the outline encloses, and no triangle of it is
    turned over or a sliver."""
    g = make()
    spec = g.spec()
    rim = _rim(spec)
    assert max(G.to_outline(g.outline, spec.nodes[n])[0] for side in rim for n in side) < 1e-9
    length = sum(math.dist(spec.nodes[a], spec.nodes[b]) for a, b in rim)
    around = sum(math.dist(g.outline[k], g.outline[(k + 1) % len(g.outline)]) for k in range(len(g.outline)))
    assert length == pytest.approx(around, rel=1e-6)
    areas = [G._area(*(spec.nodes[n] for n in tri)) for tri in spec.triangles]
    assert min(areas) > 0.1 * spec.cell ** 2 / 2
    assert sum(areas) == pytest.approx(_area_of(g.outline), rel=1e-6)
    sides = {tuple(sorted((tri[k], tri[(k + 1) % 3]))) for tri in spec.triangles for k in range(3)}
    assert min(math.dist(spec.nodes[a], spec.nodes[b]) for a, b in sides) > 0.4 * spec.cell
    assert set(range(len(spec.nodes))) == {n for tri in spec.triangles for n in tri}   # no loose point
    marks = g.marks()
    for name, p in g.keypoints:
        assert math.dist(marks[spec.key(name)], p) <= spec.cell / 2 + 1e-9


def test_fold_the_tshirt_is_cut_as_its_print_is_drawn():
    g = G.tshirt()
    drawn = prints.tee_outline()
    assert len(g.outline) == len(drawn)
    assert max(math.dist(p, (x / 1000, y / 1000)) for p, (x, y) in zip(g.outline, drawn)) < 1e-12
    xs, ys = [p[0] for p in g.spec().nodes], [p[1] for p in g.spec().nodes]
    assert (min(xs), max(xs)) == pytest.approx((-0.13, 0.13))                    # 26 cm long
    assert (min(ys), max(ys)) == pytest.approx((-0.165, 0.165))                  # 33 cm across the sleeves
    assert prints.SIZES["tee"] == (260.0, 330.0)                                 # the box round it, mm
    assert not C.inside(g.outline, (0.0, 0.13))          # nothing beside the body below the sleeves
    assert not C.inside(g.outline, (0.120, 0.0))         # the neck is cut out
    assert C.inside(g.outline, (0.100, 0.0))             # down to 2.5 cm
    assert C.inside(g.outline, (0.07, 0.15)) and C.inside(g.outline, (-0.13, 0.10))
    # the shoulders slope: their ends are lower than the neck
    assert g.keypoint("shoulder_left")[0] < max(xs) - 0.01
    # the corners a hand takes are the cloth's own
    marks = g.marks()
    for name in ("hem_left", "hem_right", "shoulder_left", "shoulder_right", "sleeve_left", "sleeve_right"):
        assert math.dist(marks[g.spec().key(name)], g.keypoint(name)) < 1e-9


def test_fold_the_tshirts_print_has_its_stripes_its_neckband_and_its_ground():
    img = prints.DESIGNS["tee"]("navy")
    assert img.size == tuple(round(v * prints.TEE_PX_PER_MM) for v in prints.SIZES["tee"])

    def colour(x_mm, y_mm):
        col = round((x_mm - prints.TEE["hem"]) * prints.TEE_PX_PER_MM)
        row = round((prints.SIZES["tee"][1] / 2 - y_mm) * prints.TEE_PX_PER_MM)
        return img.getpixel((col, row))

    navy, ground = prints.TEES["navy"], prints.TEE_GROUND
    near = lambda c, want: all(abs(a - b) < 40 for a, b in zip(c, want))
    assert near(colour(-129.0, 0.0), navy)               # the first stripe, by the hem
    assert near(colour(-120.0, 0.0), ground)             # and the white after it
    assert near(colour(102.0, 0.0), navy)                # the neckband, under the scoop
    assert near(colour(-129.0 + 16.0 * 5, 60.0), navy)   # stripes every 16 mm, right across


# -- its folds ------------------------------------------------------------------------
def test_fold_in_half_lays_the_towels_far_edge_on_its_near_edge():
    g = G.towel()
    marks = g.marks()
    pts, layers = G.goal(g, "easy", marks)
    assert (min(p[0] for p in pts), max(p[0] for p in pts)) == pytest.approx((-0.12, 0.0))
    for far, near in (("far_left", "near_left"), ("far_right", "near_right")):
        assert math.dist(_at(g, pts, far), _at(g, pts, near)) < 1e-9
        assert _at(g, layers, far) == 1 and _at(g, layers, near) == 0
    # and again, every point of it: a strip a quarter as deep
    pts, layers = G.goal(g, "hard", list(g.spec().nodes))
    assert max(p[0] for p in pts) - min(p[0] for p in pts) == pytest.approx(0.06)
    assert sorted(set(layers)) == [0, 1, 2, 3]


def test_fold_the_tshirts_folds_leave_a_block_its_bodys_width_and_half_its_length():
    g = G.tshirt()
    marks = g.marks()
    pts, _ = G.goal(g, "easy", marks)                    # the hem up, the sleeves still out
    assert math.dist(_at(g, pts, "hem_left"), g.keypoint("shoulder_left")) < 1e-9
    # taken by its corners, 20 cm apart: two X5 hands side by side come no nearer than 17 cm
    assert g.fold("hem_up").moved == ("hem_left", "hem_right")
    assert g.keypoint("hem_left")[1] - g.keypoint("hem_right")[1] >= 0.18
    every, _ = G.goal(g, "easy", list(g.spec().nodes))
    assert max(abs(p[1]) for p in every) == pytest.approx(0.165)
    every, _ = G.goal(g, "hard", list(g.spec().nodes))
    assert max(abs(p[1]) for p in every) == pytest.approx(0.10)                  # the sleeves are in
    fold = (g.keypoint("hem_left")[0] + g.keypoint("shoulder_left")[0]) / 2
    assert fold <= min(p[0] for p in every) <= fold + g.cell                     # nothing below the fold
    assert max(p[0] for p in every) == pytest.approx(0.13)
    pts, _ = G.goal(g, "hard", marks)
    assert math.dist(_at(g, pts, "sleeve_left"), g.keypoint("chest_left")) <= g.cell / 2
    assert math.dist(_at(g, pts, "sleeve_right"), g.keypoint("chest_right")) <= g.cell / 2


@pytest.mark.parametrize("make", [G.towel, G.tshirt])
def test_fold_every_fold_lays_the_points_it_takes_on_the_points_it_names(make):
    """So a hand need only carry the one to where the other lies (to half a
    cell: a point named inside the cloth is the cloth's point nearest it)."""
    g = make()
    flat = g.marks()
    for level, names in g.levels:
        pts, layers = list(flat), [0] * len(flat)
        for name in names:
            f = g.fold(name)
            assert len(f.moved) == len(f.onto)
            for moved, onto in zip(f.moved, f.onto):
                assert math.dist(C.mirror(_at(g, pts, moved), f.crease), _at(g, pts, onto)) <= g.cell / 2 + 1e-9
            pts, layers = C.fold_flat(pts, layers, f.crease, _at(g, pts, f.moved[0]))
        want, _ = G.goal(g, level, flat)
        assert max(math.dist(a, b) for a, b in zip(pts, want)) < 1e-9


def test_fold_is_scored_by_its_shape_not_by_where_it_lies():
    g = G.tshirt()
    marks = g.marks()
    want, _ = G.goal(g, "easy", marks)
    c, s = math.cos(0.35), math.sin(0.35)                # the same fold, turned 20 degrees and shifted
    moved = [(c * x - s * y + 0.31, s * x + c * y - 0.04) for x, y in want]
    assert G.fit(want, moved)[0] < 1e-9
    assert G.fit(want, marks)[0] > 0.03                  # not folded at all
    # folded 3 cm short of the shoulders
    crease = g.fold("hem_up").crease
    short = ((crease[0][0] - 0.015, -1.0), (crease[0][0] - 0.015, 1.0))
    lazy, _ = C.fold_flat(marks, [0] * len(marks), short, g.keypoint("hem_left"))
    mean, worst = G.fit(want, lazy)
    assert CR6.MEAN_PER_FOLD < mean < 0.02 and worst < CR6.WORST   # too far off on average to count
    # the shoulders down to the hem: the mirror image, and as good
    other, _ = C.fold_flat(marks, [0] * len(marks), crease, g.keypoint("shoulder_left"))
    assert G.fit(want, other)[0] < 1e-9
    # left over right is another fold
    side, _ = C.fold_flat(marks, [0] * len(marks), ((-1.0, 0.0), (1.0, 0.0)), (0.0, 0.1))
    assert G.fit(want, side)[0] > 0.03


def test_fold_a_hand_goes_round_the_crease_no_higher_than_an_x5_holds_its_hand_down():
    pinch, over = G.fold_way((0.18, 0.10), (0.425, 0.10))                        # the hem's left corner
    assert pinch == pytest.approx((0.18 + G.INSET, 0.10, G.PINCH))
    assert over[-1][2] == pytest.approx(G.LAY) and abs(over[-1][0] - 0.425) < 0.02
    heights = [z for _, _, z in over]
    assert max(heights) == pytest.approx(G.PEAK)                                 # as high as the hand can be held
    assert heights.index(max(heights)) == len(G.ROUND) // 2                      # highest half way
    assert all(abs(y - 0.10) < 1e-9 for _, y, _ in over)
    # never farther from the crease than where it lets the cloth go: nothing is pulled after it
    crease = (pinch[0] + over[-1][0]) / 2
    farthest = math.hypot(pinch[0] - crease, G.LAY)
    assert all(math.hypot(x - crease, z) <= farthest + 1e-9 for x, _, z in over)
    assert over[0][0] > pinch[0] and over[0][2] > G.LAY                          # up and over at once
    # a short fold goes round half a circle: a sleeve's end to the chest
    pinch, over = G.fold_way((0.375, 0.1575), (0.375, 0.0425))
    half = math.dist(pinch[:2], over[-1][:2]) / 2
    assert max(z for _, _, z in over) == pytest.approx(half) and half < G.PEAK
    # with physics the bit in the hand is left doubled back: the hand goes nearly all the way
    _, real = G.fold_way((0.18, 0.06), (0.425, 0.06), turned=False)
    _, played = G.fold_way((0.18, 0.06), (0.425, 0.06))
    assert real[-1][0] == pytest.approx(0.425 - G.SHORT_ACROSS)
    assert played[-1][0] == pytest.approx(0.425 - C.PINCH_ACROSS)
    # and not as near when its fingers open the way the cloth goes: they turn that bit over themselves
    _, sleeve = G.fold_way((0.375, 0.1575), (0.375, 0.0425), turned=False)
    assert sleeve[-1][1] == pytest.approx(0.0425 + G.SHORT_ALONG) and G.SHORT_ALONG > G.SHORT_ACROSS
    assert G.short((math.sqrt(0.5), math.sqrt(0.5))) == pytest.approx((G.SHORT_ALONG + G.SHORT_ACROSS) / 2)
    # over cloth that lies there already it lets go that much higher, and is never lower on the way
    _, high = G.fold_way((0.18, 0.06), (0.425, 0.06), turned=False, lay=G.LAY + 0.005)
    assert high[-1][2] == pytest.approx(G.LAY + 0.005) == pytest.approx(min(z for _, _, z in high))
    assert [p[:2] for p in high] == [pytest.approx(p[:2]) for p in real]
    with pytest.raises(ValueError):
        G.fold_way((0.3, 0.1), (0.3, 0.1))


def test_fold_two_hands_carry_an_edge_as_one():
    # the shoulders lie 3 cm nearer one another than the hem is wide: each corner still goes
    # straight up, and the two come down as far apart as they were taken
    ways = G.one_move([((0.18, 0.10), (0.425, 0.085)), ((0.18, -0.10), (0.425, -0.085))])
    assert [a for a, _ in ways] == [(0.18, 0.10), (0.18, -0.10)]
    assert ways[0][1] == pytest.approx((0.425, 0.10)) and ways[1][1] == pytest.approx((0.425, -0.10))
    # a cloth lying askew: both go the mean of the two ways
    (_, b0), (_, b1) = G.one_move([((0.0, 0.1), (0.24, 0.12)), ((0.0, -0.1), (0.26, -0.08))])
    assert b0 == pytest.approx((0.25, 0.12)) and b1 == pytest.approx((0.25, -0.08))
    # one hand's way is its own
    assert G.one_move([((0.3, 0.16), (0.3, 0.04))]) == [((0.3, 0.16), pytest.approx((0.3, 0.04)))]


def test_fold_how_high_the_cloth_lies_is_read_off_its_marks():
    from types import SimpleNamespace

    one = TABLE + CR6.GARMENT.spec().radius
    flat = [(0.0, 0.0, one), (0.03, 0.0, one), (0.2, 0.0, one)]
    world = lambda pts: SimpleNamespace(obj=lambda name: SimpleNamespace(points=pts))
    assert CR6.lying(world(flat), (0.0, 0.0)) == 0.0
    # a layer over it near by: that much higher; the same layer farther off: nothing
    folded = flat + [(0.02, 0.02, one + 0.005)]
    assert CR6.lying(world(folded), (0.0, 0.0)) == pytest.approx(0.005)
    assert CR6.lying(world(folded), (0.02 + CR6.NEAR + 0.001, 0.02)) == 0.0
    assert CR6.lying(world(folded), (0.2, 0.0)) == 0.0
    assert CR6.lying(world(folded), (0.5, 0.5)) == 0.0                           # nothing of it there


# -- the task -----------------------------------------------------------------------
def test_fold_the_task_folds_the_tshirt():
    assert CR6.GARMENT.name == "tshirt" and CR6.CLOTH == "tshirt"
    assert CR6.FINISH.startswith("paper:tee:") and CR6.FINISH.split(":")[2] in prints.TEES
    assert len(CR6.goals("easy")) == 1 and len(CR6.goals("hard")) == 3


def test_fold_the_scene_is_the_cloth_and_nothing_else_within_the_hands_reach():
    for seed in range(10):
        scene = build("CR6", "easy", seed).scene
        assert [b.name for b in scene.bodies] == [CR6.CLOTH]
        assert C.scene_problems(scene) == []
        body = scene.bodies[0]
        assert body.pose.pos[2] == pytest.approx(TABLE + C.cloth_of(body).radius)
        points = C.flat_points(body)
        ahead = [p[0] - STATION_X for p in points]
        side = [abs(p[1]) for p in points]
        # every point of it where a hand pointing down is held 5 cm over the table: from 15 cm
        # ahead to 45 cm, 10 cm to a side and more (measured 10/5, on a grid of 3 cm)
        assert 0.15 <= min(ahead) and max(ahead) <= 0.455 and max(side) <= 0.19, seed


@pytest.mark.parametrize("level", LEVELS)
def test_fold_the_reference_folds_it_with_both_hands(level):
    asked = len(CR6.GARMENT.asked(level))
    for r in _run(level):
        assert r.success, r.score.failure
        # (at the hard level the first fold is a sleeve, one hand's work: it is laid before
        # the cloth is ever in both hands)
        assert sorted(r.score.milestones_hit) == ["cloth_pinched", "first_fold_laid", "folded_hands_off",
                                                   "in_both_hands"]
        assert r.log.task_metrics["off_mean_mm"] <= 1000 * CR6.MEAN_PER_FOLD * asked / 2
        last = r.trajectory[-1]
        assert last.obj(CR6.CLOTH).held_by is None
        assert all(last.arm(a).holding is None for a in (LEFT, RIGHT))
        # the hem lies at the shoulders
        hem, shoulder = (C.key_point(last.obj(CR6.CLOTH), k) for k in ("hem_left", "shoulder_left"))
        assert math.dist(hem[:2], shoulder[:2]) < 0.01
        # and the hands took it by the hem's corners
        took = next(w for w in r.trajectory if all(w.arm(a).holding == CR6.CLOTH for a in (LEFT, RIGHT)))
        corner = C.key_point(took.obj(CR6.CLOTH), "hem_left")
        assert math.dist(took.arm(LEFT).eef_pose.pos[:2], corner[:2]) < 0.02
        # they let go at one height, on top of what lay there: over the folded sleeves, higher
        both = [w for w in r.trajectory if all(w.arm(a).holding == CR6.CLOTH for a in (LEFT, RIGHT))]
        let_go = [both[-1].arm(a).eef_pose.pos[2] - TABLE for a in (LEFT, RIGHT)]
        assert let_go[0] == pytest.approx(let_go[1], abs=1e-6)
        assert (let_go[0] > G.LAY + 0.004) if asked > 1 else (let_go[0] == pytest.approx(G.LAY, abs=1e-6))


@pytest.mark.parametrize("level", LEVELS)
def test_fold_a_cloth_moved_as_a_block_is_not_folded(level):
    for r in _run(level, "reference+ignore_rules"):
        assert not r.success
        assert list(r.score.milestones_hit) == ["cloth_pinched"]
        goals = CR6.goals(level)
        assert CR6.off(r.trajectory[-1], goals[-1])[0] > CR6.MEAN_PER_FOLD * len(goals)


def test_fold_a_cloth_left_alone_is_not_folded():
    for r in run_plan(RunPlan(["CR6"], levels=LEVELS, seeds=(0,)),
                      agent_factory=lambda i: build_agent("do_nothing", i), config=CFG):
        assert not r.success and not r.score.milestones_hit
