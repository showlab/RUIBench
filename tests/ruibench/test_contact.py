"""The Contact-Rich Manipulation suite: what each task's contact is made of.

The generic checks -- a reference that solves every level, doing nothing never
working, one scene structure per task -- run on these tasks with the rest
(``test_tasks.py``, ``test_isolated.py``).  Here: the holes, the stone, and
the controls that show the "how" is what these tasks are about.
"""

import math

import pytest

from ruibench.core.config import RunConfig
from ruibench.core.types import GRASP_DOWN, Pose
from ruibench.engines.mock import MockEngine
from ruibench.policies.agents import build_agent
from ruibench.runner import RunPlan, run_plan
from ruibench.tasks.impl import cr1_shape_sorter as CR1
from ruibench.tasks.impl import cr3_curling as CR3
from ruibench.tasks.instance import build

CFG = RunConfig(engine="mock", interface="I0_direct", clock="zero_latency")
LEVELS = ("easy", "hard")


def _run(task, level, spec="reference", seeds=(0, 1, 2), factory=None):
    return run_plan(RunPlan([task], levels=(level,), seeds=seeds),
                    agent_factory=factory or (lambda i: build_agent(spec, i)), config=CFG)


def _yaw(angle):
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


# -- CR1 ------------------------------------------------------------------------
def _dropped(level, shape, hole, turn, seed=0):
    """``shape`` let go over ``hole``'s middle, turned ``turn`` from the way
    that fits its own hole: the engine after it has come to rest."""
    scene = build("CR1", level, seed).scene
    engine = MockEngine()
    engine.load(scene, seed)
    world = engine.reset()
    sorter = world.obj(CR1.SORTER)
    fits = CR1.target(sorter, shape, CR1.CLEARANCE[level], 0.0)[1]
    if shape == hole:
        (x, y), _ = CR1.target(sorter, shape, CR1.CLEARANCE[level], fits)
        if shape == "triangle":
            # turned about its centroid, which is not the middle of its box
            off = CR1.triangle_height() / 6
            cx, cy = x + math.sin(fits) * off, y - math.cos(fits) * off
            x, y = cx - math.sin(fits + turn) * off, cy + math.cos(fits + turn) * off
    else:
        x, y = CR1.cell_centre(sorter, hole)
    engine.teleport(shape, Pose((x, y, CR1.lid_top(sorter) + CR1.BLOCK_HALF_HEIGHT + 0.01),
                                _yaw(fits + turn)))
    engine.advance(0.2)
    return engine


@pytest.mark.parametrize("level", LEVELS)
def test_cr1_a_block_passes_its_hole_only_turned_within_the_holes_play(level):
    for shape in CR1.SHAPES:
        assert CR1.through(_dropped(level, shape, shape, 0.0).state(), shape), shape
        if shape == "round":
            continue
        fit = CR1.fit_yaw(shape, CR1.CLEARANCE[level])
        assert CR1.through(_dropped(level, shape, shape, 0.8 * fit).state(), shape), shape
        assert not CR1.through(_dropped(level, shape, shape, fit + math.radians(3)).state(), shape), shape


def test_cr1_some_holes_take_other_blocks_so_the_scorer_keeps_the_rule():
    """At the hard level only the round block passes another hole (the
    square one, which is as wide as it is); at the easy level most do -- a
    square turned cornerwise goes through the cross.  Which hole a block went
    through is recorded as it drops, and another shape's hole is a foul."""
    turns = [math.radians(a) for a in range(0, 120, 5)]

    def passes(level):
        return {(shape, hole) for shape in CR1.SHAPES for hole in CR1.SHAPES if hole != shape
                and any(CR1.through(_dropped(level, shape, hole, t).state(), shape) for t in turns)}

    assert passes("hard") == {("round", "square")}
    assert {("square", "round"), ("square", "cross"), ("round", "square")} <= passes("easy")
    engine = _dropped("easy", "square", "round", 0.0)
    assert CR1.through(engine.state(), "square")
    CR1.Sorter().tick(engine, engine.state())
    world = engine.state()
    assert CR1.wrong_hole(world) and not CR1.own_hole(world, "square")


def test_cr1_every_block_but_the_round_one_starts_out_of_its_holes_play():
    for seed in range(20):
        box_yaw, blocks = CR1.layout(seed)
        for shape, (_, yaw) in blocks.items():
            if shape == "round":
                continue
            sym = CR1.SYMMETRY[shape]
            off = (yaw - box_yaw + (math.pi / 2 if shape == "triangle" else 0.0)) % sym
            assert min(off, sym - off) >= CR1.MIN_TURN - 1e-9, (seed, shape)
            # ...and the fingers close across the table, where the camera sees the block
            assert abs(math.sin(yaw)) <= math.sin(CR1.STRAIGHT) + 1e-9, (seed, shape)
    assert CR1.fit_yaw("cross", CR1.CLEARANCE["easy"]) < CR1.MIN_TURN
    assert CR1.fit_yaw("triangle", CR1.CLEARANCE["easy"]) < CR1.MIN_TURN


@pytest.mark.parametrize("level", LEVELS)
def test_cr1_the_reference_turns_each_block_and_one_put_down_as_it_lies_stays_out(level):
    for r in _run("CR1", level):
        assert r.success and r.log.task_metrics["tries"] == len(CR1.SHAPES)
    for r in _run("CR1", level, "reference+ignore_rules"):
        assert not r.success
        world = r.trajectory[-1]
        assert not CR1.through(world, "cross") and not CR1.through(world, "triangle")
        assert CR1.through(world, "round")


# -- CR3 ------------------------------------------------------------------------
@pytest.mark.parametrize("level", LEVELS)
def test_cr3_the_stone_is_pushed_home_and_carrying_it_there_is_a_foul(level):
    for r in _run("CR3", level):
        assert r.success and r.log.task_metrics["pushes"] >= 1 and not r.score.violations
    for r in _run("CR3", level, "reference+ignore_rules"):
        assert not r.success and "never_gripped" in r.score.violations


def test_cr3_the_zone_is_the_knob():
    """A stone left 5 cm from the tee is in the house and not on the button."""
    def short(inst):
        policy = inst.reference_factory(inst)
        tx, ty = policy.tee
        _, start = CR3.layout(inst.seed)
        d = math.dist(start, (tx, ty))
        # aim 5 cm short of the tee, along the line from where the stone starts
        policy.tee = (tx + (start[0] - tx) * 0.05 / d, ty + (start[1] - ty) * 0.05 / d)
        policy.near = 0.004
        return policy

    assert all(r.success for r in _run("CR3", "easy", factory=short))
    for r in _run("CR3", "hard", factory=short):
        assert not r.success and not r.score.violations
        assert 0.03 < r.log.task_metrics["final_distance_m"] < 0.07


def test_cr3_on_the_mock_a_push_off_the_stones_middle_sends_it_sideways():
    scene = build("CR3", "easy", 0).scene
    engine = MockEngine()
    engine.load(scene, 0)
    world = engine.reset()
    reactor = CR3.Pushing()
    x, y, _ = world.obj(CR3.STONE).pose.pos
    z = CR3.SHEET_TOP + CR3.PUSH_Z
    # along +y, on a line 2 cm to the side of the stone's middle
    engine.set_gripper(CR3.RIGHT, 1.0)
    for target in ((x + 0.02, y - 0.08, z + 0.1), (x + 0.02, y - 0.08, z), (x + 0.02, y + 0.06, z)):
        engine.set_arm_target(CR3.RIGHT, Pose(target, GRASP_DOWN))
        for _ in range(40):
            engine.advance(0.1)
            reactor.tick(engine, engine.state())
    sx, sy, sz = engine.state().obj(CR3.STONE).pose.pos
    assert sx < x - 0.01                      # shoved aside, away from the hand's line
    assert sy > y and abs(sz - CR3.STONE_Z) < 1e-6


# -- CR5 ------------------------------------------------------------------------
from ruibench.tasks.impl import cr5_bricks as CR5  # noqa: E402


def _cr5(level="hard", seed=0):
    inst = build("CR5", level, seed, CFG)
    engine = MockEngine()
    engine.load(inst.scene, seed)
    engine.reset()
    return inst, engine, inst.reactor_factory()


def _steps(engine, clutch, seconds):
    for _ in range(int(round(seconds / 0.05))):
        engine.advance(0.05)
        clutch.tick(engine, engine.state())
    return engine.state()


def _over_red(engine, name, k, off=(0.0, 0.0), up=0.03):
    """A brick let go ``up`` over its place, ``off`` from the middle."""
    red = engine.state().obj(CR5.RED).pose
    engine.teleport(name, Pose((red.pos[0] + off[0], red.pos[1] + off[1],
                                CR5.level_top(k) + CR5.BODY_H / 2 + up), red.quat))


def _press(engine, clutch, name, k, seconds=1.0):
    """Closed fingers brought down on the brick's studs to where they are when
    it is home, held there, and taken away."""
    obj = engine.state().obj(name).pose
    top = CR5.level_top(k) + CR5.BODY_H + CR5.fingers_on(name) + CR5.FINGERTIP
    engine.set_gripper(CR5.RIGHT, 1.0)
    for z, t in ((top + 0.05, 1.5), (top, 1.5 + seconds), (top + 0.05, 1.5)):
        engine.set_arm_target(CR5.RIGHT, Pose((obj.pos[0], obj.pos[1], z), GRASP_DOWN))
        _steps(engine, clutch, t)
    return engine.state()


@pytest.mark.parametrize("level", LEVELS)
def test_cr5_the_tower_is_built_and_a_brick_put_down_askew_cannot_be_pressed_home(level):
    """The reference puts each brick over its place and presses it home.  The
    stress agent puts the first brick down half a stud off and presses: it
    rides on top of the studs, no pressing takes it down, and nothing after
    it counts."""
    results = _run("CR5", level, seeds=(0, 1, 2, 3))
    assert all(r.success for r in results), [r.score.failure for r in results]
    assert all(r.log.task_metrics["bricks_home"] == CR5.COUNT[level] for r in results)
    blind = _run("CR5", level, spec="reference+ignore_rules", seeds=(0, 1, 2, 3))
    assert not any(r.success for r in blind)
    assert all(r.log.task_metrics["bricks_home"] == 0 for r in blind)
    for r in blind:
        order = CR5.BRICKS[:CR5.COUNT[level]]
        w = r.trajectory[-1]
        # up on the studs, half a stud along, and it was pressed: the hand came down on it
        assert CR5.gap(w, order, order[0]) == pytest.approx(CR5.STUD_H, abs=1e-3)
        assert abs(CR5.off_middle(w, order, order[0])[0]) == pytest.approx(CR5.ASKEW, abs=2e-3)
        low = min(s.arm(CR5.RIGHT).eef_pose.pos[2] for s in r.trajectory if s.arm(CR5.RIGHT).gripper >= 0.5
                  and not s.arm(CR5.RIGHT).holding)
        assert low < CR5.level_top(0) + CR5.BODY_H + CR5.STUD_H + CR5.FINGERTIP + 0.002


def test_cr5_a_seat_gives_only_under_a_brick_that_is_over_its_place():
    """Sprung under its brick, as good as solid under a brick that sits a stud
    along: there the hand that presses is stopped (the engine is told how hard
    the seat is held)."""
    inst, engine, clutch = _cr5()
    told = []
    engine.set_joint_drive = lambda joint, stiffness, damping: told.append((joint, stiffness))
    _steps(engine, clutch, 0.2)                                   # the yellow brick is still on the table
    assert told[0] == (CR5.seat_name("yellow"), CR5.SEAT_STIFF[0])
    _over_red(engine, "yellow", 0)
    _steps(engine, clutch, 0.5)
    assert told[-1] == (CR5.seat_name("yellow"), CR5.SEAT_STIFFNESS)
    _over_red(engine, "yellow", 0, off=(CR5.PITCH, 0.0))            # a stud along: not its place
    _steps(engine, clutch, 0.5)
    assert told[-1] == (CR5.seat_name("yellow"), CR5.SEAT_STIFF[0])


def test_cr5_a_brick_rides_on_its_seat_until_pressed_home_and_the_next_seat_comes_up():
    """Let go over the red brick, the yellow one stops on its seat, 5 mm up;
    closed fingers on its studs take it down, and it stays down.  Only then is
    the blue brick's seat at its place -- before, the blue brick put on the
    yellow one goes straight down, and holds nothing."""
    inst, engine, clutch = _cr5()
    order = CR5.BRICKS
    _over_red(engine, "yellow", 0)
    w = _steps(engine, clutch, 1.0)
    assert CR5.gap(w, order, "yellow") == pytest.approx(CR5.REST, abs=1e-4)
    assert not CR5.home(w, order, "yellow") and not clutch.on["yellow"]
    # the blue brick's seat is still down inside the red brick
    assert w.articulations[CR5.seat_name("blue")] < -CR5.BODY_H
    # a touch that does not go all the way down: the brick comes back up
    top = CR5.level_top(0) + CR5.BODY_H + CR5.fingers_on("yellow") + CR5.FINGERTIP
    obj = w.obj("yellow").pose
    engine.set_gripper(CR5.RIGHT, 1.0)
    engine.set_arm_target(CR5.RIGHT, Pose((obj.pos[0], obj.pos[1], top + 0.003), GRASP_DOWN))
    w = _steps(engine, clutch, 2.5)
    assert CR5.gap(w, order, "yellow") == pytest.approx(0.003, abs=5e-4) and not clutch.on["yellow"]
    engine.set_arm_target(CR5.RIGHT, Pose((obj.pos[0], obj.pos[1], top + 0.05), GRASP_DOWN))
    w = _steps(engine, clutch, 1.5)
    assert CR5.gap(w, order, "yellow") == pytest.approx(CR5.REST, abs=1e-4)
    # pressed home: it stays, and the next seat comes up to its place
    w = _press(engine, clutch, "yellow", 0)
    assert CR5.home(w, order, "yellow") and clutch.on["yellow"]
    w = _steps(engine, clutch, 1.5)
    assert w.articulations[CR5.seat_name("blue")] == pytest.approx(0.0, abs=1e-4)
    _over_red(engine, "blue", 1)
    w = _steps(engine, clutch, 1.0)
    assert CR5.gap(w, order, "blue") == pytest.approx(CR5.REST, abs=1e-4)
    assert CR5.homes(w, order) == 1


def test_cr5_a_brick_too_far_off_sits_on_the_studs_and_no_pressing_moves_it():
    """A brick goes down over the studs only from near enough for their
    rounded tops to take it in.  Put 12 mm off, it stays up on them; pressed,
    it stays there."""
    inst, engine, clutch = _cr5()
    order = CR5.BRICKS
    assert CR5.CLEAR == pytest.approx(0.002) and CR5.OVER_XY == pytest.approx(0.007)
    # the fingers, over the middle of a brick, stand on the studs' shoulders
    assert CR5.STUD_STRAIGHT < CR5.fingers_on("yellow") < CR5.STUD_H
    # 4 mm off: it slides home, and stands on its seat, its walls clear of the studs
    _over_red(engine, "yellow", 0, off=(0.004, 0.0))
    w = _steps(engine, clutch, 1.0)
    assert CR5.gap(w, order, "yellow") == pytest.approx(CR5.REST, abs=1e-4)
    assert w.obj("yellow").pose.planar_distance_to(w.obj(CR5.RED).pose.pos) <= CR5.CLEAR
    _over_red(engine, "yellow", 0, off=(0.012, 0.0))
    w = _steps(engine, clutch, 1.0)
    assert CR5.gap(w, order, "yellow") == pytest.approx(CR5.STUD_H, abs=1e-4)
    w = _press(engine, clutch, "yellow", 0)
    assert CR5.gap(w, order, "yellow") == pytest.approx(CR5.STUD_H, abs=1e-4)
    assert not clutch.on["yellow"] and CR5.homes(w, order) == 0


def test_cr5_the_seats_are_not_drawn_and_meet_only_their_own_brick():
    """A seat is a collider and nothing else: no camera shows it, it is no
    word of the task's, and every body but its own brick passes through it."""
    from ruibench.engines.base import drawn, passes

    inst, engine, clutch = _cr5()
    bodies = {b.name: b for b in inst.scene.bodies}
    seats = [bodies[CR5.seat_name(n)] for n in CR5.BRICKS]
    for name, seat in zip(CR5.BRICKS, seats):
        assert not drawn(seat) and seat.category == "object"
        assert not passes(seat, bodies[name])
        assert all(passes(seat, b) for n, b in bodies.items() if n not in (name, seat.name))
    seen = set(engine.visible_objects("agentview")) | set(engine.visible_objects("wrist"))
    assert not seen & {s.name for s in seats} and {"yellow", "red"} <= seen
    # the bricks are the size the card says, and a wall stands 2 mm off the studs
    assert CR5.half("yellow") == pytest.approx((0.064, 0.032, 0.0192))
    assert CR5.half("blue") == pytest.approx((0.032, 0.032, 0.0192))
    assert 2 * CR5.half("yellow")[1] < 0.08                      # the FR3's fingers open 8 cm


# -- CR2 ------------------------------------------------------------------------
from ruibench.tasks.impl import cr2_bulb as CR2  # noqa: E402


def _cr2(level="hard", seed=0):
    inst = build("CR2", level, seed, CFG)
    engine = MockEngine()
    engine.load(inst.scene, seed)
    engine.reset()
    return inst, engine, inst.reactor_factory()


def _bulb_at(engine, thread, xy, bottom, yaw, seconds=0.2):
    """The bulb stood at ``xy``, its cap's end ``bottom`` over the table,
    turned ``yaw`` -- and the thread given a look at it."""
    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    engine.teleport(CR2.BULB, Pose((xy[0], xy[1], bottom + CR2.BULB_H / 2), quat))
    for _ in range(int(round(seconds / 0.05))):
        engine.advance(0.05)
        thread.tick(engine, engine.state())
    return engine.state()


@pytest.mark.parametrize("level", LEVELS)
def test_cr2_the_lamp_lights_after_its_turns_and_never_turned_the_other_way(level):
    """The reference sets the bulb in the socket and turns it clockwise until
    the lamp lights -- in one stroke at the easy level, in several at the hard
    one.  The stress agent turns it the other way, as far: the lamp stays dark."""
    results = _run("CR2", level, seeds=(0, 1, 2, 3))
    assert all(r.success for r in results), [r.score.failure for r in results]
    assert all(r.log.task_metrics["lit_at_the_end"] for r in results)
    blind = _run("CR2", level, spec="reference+ignore_rules", seeds=(0, 1, 2, 3))
    assert not any(r.success for r in blind)
    assert not any(r.log.task_metrics["lit_at_the_end"] for r in blind)
    # the other way round, the bulb did get into the socket
    assert all("cap_in_socket" in r.score.milestones_hit for r in blind)


def test_cr2_the_thread_counts_clockwise_turns_while_the_cap_is_in_the_socket():
    """Clockwise seen from above screws the bulb in, the other way out again,
    and no further in than all the way; out of the socket it starts again."""
    inst, engine, thread = _cr2("easy")
    lamp = next(b for b in inst.scene.bodies if b.name == CR2.LAMP).pose.pos
    floor = CR2.socket_floor()
    quarter = math.pi / 2
    w = _bulb_at(engine, thread, lamp, floor, 0.0)
    assert CR2.in_socket(w, lamp) and not CR2.lit(w) and thread.turned == 0.0
    # counter-clockwise first: nothing
    for k in range(1, 4):
        w = _bulb_at(engine, thread, lamp, floor, k * quarter / 3)
    assert not CR2.lit(w) and thread.turned == 0.0
    # back, and a quarter turn clockwise past where it started: lit
    for k in range(2, -4, -1):
        w = _bulb_at(engine, thread, lamp, floor, k * quarter / 3)
    assert CR2.lit(w) and thread.turned == pytest.approx(quarter)
    assert w.obj(CR2.GLOW).color == CR2.ON
    # no further in than all the way: a third of a turn more counts for nothing
    for k in range(-4, -8, -1):
        w = _bulb_at(engine, thread, lamp, floor, k * quarter / 3)
    assert thread.turned == pytest.approx(quarter)
    # ten degrees back: out
    w = _bulb_at(engine, thread, lamp, floor, -7 * quarter / 3 + math.radians(10))
    assert not CR2.lit(w) and w.obj(CR2.GLOW).color == CR2.OFF
    w = _bulb_at(engine, thread, lamp, floor, -7 * quarter / 3)
    assert CR2.lit(w)
    # taken out of the socket and stood on the table: it starts again
    w = _bulb_at(engine, thread, (lamp[0] + 0.12, lamp[1]), CR2.TABLE, -7 * quarter / 3)
    assert not CR2.in_socket(w, lamp) and not CR2.lit(w) and thread.turned == 0.0


def test_cr2_a_bulb_leaning_in_the_bore_is_still_in_the_socket():
    """The socket is judged where the cap is: a lit bulb leaning 5 degrees,
    its cap's end still on the socket's axis, keeps the lamp lit.  Its middle
    is then 4.7 mm off the axis -- judged there, the lamp went out when a
    finger brushed the glass (Claude Code's runs on v2.3.2)."""
    from ruibench.core.geometry import quat_mul, rotate

    inst, engine, thread = _cr2("easy")
    lamp = next(b for b in inst.scene.bodies if b.name == CR2.LAMP).pose.pos
    floor = CR2.socket_floor()
    quarter = math.pi / 2
    for k in range(0, -4, -1):
        w = _bulb_at(engine, thread, lamp, floor, k * quarter / 3)
    assert CR2.lit(w)
    lean = math.radians(5)
    quat = quat_mul((math.cos(-quarter / 2), 0.0, 0.0, math.sin(-quarter / 2)),
                    (math.cos(lean / 2), math.sin(lean / 2), 0.0, 0.0))
    axis = rotate(quat, (0.0, 0.0, 1.0))
    middle = tuple(c + CR2.BULB_H / 2 * a for c, a in zip((lamp[0], lamp[1], floor), axis))
    # (not advanced: the mock rests a leaning body on what its footprint meets -- the mouth)
    engine.teleport(CR2.BULB, Pose(middle, quat))
    thread.tick(engine, engine.state())
    w = engine.state()
    assert math.hypot(middle[0] - lamp[0], middle[1] - lamp[1]) > CR2.IN_XY
    assert CR2.tilt(w) == pytest.approx(lean, abs=1e-6)
    assert w.obj(CR2.BULB).pose.pos == pytest.approx(middle, abs=1e-9)
    assert CR2.in_socket(w, lamp) and CR2.lit(w) and thread.turned == pytest.approx(quarter)
    # leaning past what the bore allows is out, and starts again
    lean = math.radians(15)
    quat = quat_mul((math.cos(-quarter / 2), 0.0, 0.0, math.sin(-quarter / 2)),
                    (math.cos(lean / 2), math.sin(lean / 2), 0.0, 0.0))
    axis = rotate(quat, (0.0, 0.0, 1.0))
    engine.teleport(CR2.BULB, Pose(tuple(c + CR2.BULB_H / 2 * a for c, a in
                                         zip((lamp[0], lamp[1], floor), axis)), quat))
    thread.tick(engine, engine.state())
    w = engine.state()
    assert not CR2.in_socket(w, lamp) and not CR2.lit(w) and thread.turned == 0.0


def test_cr2_the_bulb_fits_the_socket_and_two_turns_take_more_than_one_stroke():
    """The cap's corners turn free in the bore with half a millimetre to
    spare; the mouth takes in a bulb let go 8 mm off; and the hard level's two
    turns are more than the wrist makes in one go, the easy one's are not."""
    inst, engine, thread = _cr2("hard")
    assert CR2.CAP_FLAT / math.cos(math.pi / 8) < CR2.BORE < CR2.CAP_FLAT / math.cos(math.pi / 8) + 0.001
    assert CR2.MOUTH_TAKES == pytest.approx(0.0085)
    stroke = CR2.TURNED_BACK - CR2.TURNED_ON
    assert 2 * math.pi * CR2.TURNS["easy"] < stroke < 2 * math.pi * CR2.TURNS["hard"] / 2
    assert CR2.STEP < math.pi / 2
    lamp = next(b for b in inst.scene.bodies if b.name == CR2.LAMP).pose.pos
    # let go in the mouth, 6 mm off: it slides down into the cup
    w = _bulb_at(engine, thread, (lamp[0] + 0.006, lamp[1]), CR2.TABLE + CR2.RIM + 0.002, 0.3, seconds=0.5)
    assert CR2.in_socket(w, lamp)
    # the bulb starts standing in its box, and the glass is seen while the filament rides in it
    inst, engine, thread = _cr2("easy")
    w = engine.state()
    box = w.obj(CR2.BOX).pose.pos
    assert w.obj(CR2.BULB).pose.planar_distance_to(box) < 1e-6 and CR2.tilt(w) < 1e-6
    _bulb_at(engine, thread, lamp, CR2.socket_floor(), 0.4)
    w = engine.state()
    assert w.obj(CR2.GLOW).pose.planar_distance_to(w.obj(CR2.BULB).pose.pos) < 1e-6


# -- CR4 ------------------------------------------------------------------------
from ruibench.core.geometry import rotate  # noqa: E402
from ruibench.engines.parts import part_corners  # noqa: E402
from ruibench.engines.robots import FR3  # noqa: E402
from ruibench.tasks.impl import cr4_graphics_card as CR4  # noqa: E402


def test_cr4_the_card_lies_flat_its_top_edge_over_the_pads_edge():
    """Fan up, where a hand from above can only take it lying; its top edge
    hangs over the pad's edge, where a finger can get under it, high enough
    over the bench for the hand to work level."""
    for seed in range(10):
        scene = build("CR4", "easy", seed).scene
        card = next(b for b in scene.bodies if b.name == CR4.CARD)
        box = next(b for b in scene.bodies if b.name == CR4.BOX)
        through = rotate(card.pose.quat, (0.0, 1.0, 0.0))
        assert through[2] > 0.99                                   # the fan side up
        top_edge = CR4.compose(card.pose, Pose((0.0, 0.0, CR4.CARD_HALF[2]))).pos
        pad_edge = box.pose.pos[1] + CR4.BOX_HALF[1]
        assert 0.02 <= top_edge[1] - pad_edge <= 0.03, seed
        # its weight is on the pad
        middle = CR4.compose(card.pose, Pose((0.0, 0.0, CR4.BODY_Z))).pos
        assert middle[1] < pad_edge - 0.005
        # the hand level beside it: its body, 20.8 cm along the fingers' travel
        # and upright now, clears the bench (it gets down to 11-13 cm, Isaac 10/5)
        assert card.pose.pos[2] - scene.table_height >= 0.13 + 0.02
    # across its thickness the hand closes on it; across its height too, from above -- lying
    assert CR4.CARD_T < CR4.CARD_H < FR3.max_opening


@pytest.mark.parametrize("level", LEVELS)
def test_cr4_the_slots_mouth_is_the_knob_and_it_narrows_to_grip_the_edge(level):
    board = CR4.board_body(level, (0.0, 0.0), 0.0)
    sx = CR4.SLOT_AT[0]
    floor = -(CR4.BOARD_TOP + CR4.SLOT_H) / 2 + CR4.BOARD_TOP + CR4.SLOT_H - CR4.GROOVE
    walls = [p for p in board.attrs["parts"] if p[0] == "box" and p[1][0] > sx + 0.0005
             and p[2][1] > 0.03]
    inner = sorted({(round(c[0] - sx, 5), round(c[2] - floor, 5)) for p in walls for c in part_corners(p)})
    by_height = {z: min(x for x, zz in inner if zz == z) for _, z in inner}
    tab = CR4.TAB_T / 2
    assert by_height[CR4.GROOVE] == pytest.approx(tab + CR4.FIT[level], abs=1e-5)      # the mouth
    assert by_height[CR4.GRIP_DEPTH] == pytest.approx(tab + CR4.TIGHT, abs=1e-5)       # the grip
    assert CR4.FIT["hard"] < CR4.FIT["easy"]


@pytest.mark.parametrize("level", LEVELS)
def test_cr4_the_reference_takes_it_from_the_side_and_presses_it_home(level):
    for r in _run("CR4", level):
        assert r.success and "card_in" in r.trajectory[-1].events_fired
        assert r.log.task_metrics["depth_mm"] >= 1000 * (CR4.GROOVE - CR4.HOME_Z)
        # it was taken while it lay: the hand level, not pointing down
        taken = next(w for w in r.trajectory if w.obj(CR4.CARD).is_held)
        pointing = rotate(taken.arm(CR4.RIGHT).eef_pose.quat, (0.0, 0.0, 1.0))
        assert abs(pointing[2]) < 0.2


@pytest.mark.parametrize("level", LEVELS)
def test_cr4_taken_from_above_as_it_lies_it_never_goes_in(level):
    for r in _run("CR4", level, "reference+ignore_rules"):
        assert not r.success
        assert not CR4.seated(r.trajectory[-1], level)
        assert "card_in" not in r.trajectory[-1].events_fired
        assert r.log.task_metrics["depth_mm"] == 0.0               # across the slot is not in it
        # it did take the card, and carried it lying all the way
        held = [w for w in r.trajectory if w.obj(CR4.CARD).is_held]
        assert held and all(CR4.tilt(w) > math.radians(60) for w in held)
        assert set(r.score.milestones_hit) == {"card_taken"}


def test_cr4_a_card_set_in_the_slot_and_let_go_is_not_in_until_it_is_pressed():
    """Stood in the mouth and let go, it rests on the seat, 5 mm up: the
    slot has not taken it (pressing it home is the reference's test above)."""
    level, seed = "easy", 0
    scene = build("CR4", level, seed).scene
    engine = MockEngine()
    engine.load(scene, seed)
    world = engine.reset()
    slot = CR4.Slot(level)
    (gx, gy), run, floor = CR4.groove(world.obj(CR4.BOARD))

    def stand(up):
        quat = CR4.turn_z(run + math.pi)
        ox, oy, _ = rotate(quat, (CR4.TAB_ALONG, CR4.PCB_Y, 0.0))
        engine.teleport(CR4.CARD, Pose((gx - ox, gy - oy, floor + up + CR4.CARD_HALF[2]), quat))

    stand(CR4.REST + 0.001)
    for _ in range(10):
        engine.advance(0.1)
        slot.tick(engine, engine.state())
    world = engine.state()
    over, up = CR4.in_groove(world, level)
    assert over and up == pytest.approx(CR4.REST, abs=0.0015) and not CR4.seated(world, level)
    assert "card_in" not in world.events_fired
