"""The ways round the tasks' checks found by reading the scorers after a rollout
broke WK1's rules and still succeeded (v2.3), each closed: the trajectory that
went round a check now fails it, and the honest one beside it still passes.

Each trajectory starts from the mock's first state and is edited by hand; the
tasks' own reactors are ticked on it where the check reads their events."""

import math
from dataclasses import replace

from ruibench.core.geometry import rotate
from ruibench.core.types import Pose
from ruibench.engines import make_engine
from ruibench.engines import parts as P
from ruibench.engines.robots import model_of
from ruibench.tasks.instance import build


class FakeEngine:
    """Just enough engine for a reactor: it records events and targets."""

    def __init__(self):
        self.events, self.instruction, self.version, self.targets = [], "", 0, {}

    def note_event(self, name):
        self.events.append(name)

    def set_instruction(self, text, initial=False):
        self.instruction, self.version = text, self.version + 1

    def set_articulation_target(self, joint, value):
        self.targets[joint] = value

    def set_color(self, body, color):
        pass

    def set_joint_drive(self, joint, stiffness, damping):
        pass


def first(task, level, seed=15):
    inst = build(task, level=level, seed=seed)
    e = make_engine("mock")
    e.load(inst.scene, seed)
    e.reset()
    w = e.state()                                    # sight lines: none (the mock works none out)
    return inst, replace(w, objects={n: replace(o, visible_from=o.visible_from or ())
                                     for n, o in w.objects.items()})


def moved(w, t=None, **objs):
    """``w`` with objects changed: name -> a position, or a dict of fields."""
    out = dict(w.objects)
    for n, change in objs.items():
        fields = change if isinstance(change, dict) else {"pose": Pose(change, out[n].pose.quat)}
        out[n] = replace(out[n], **{"lin_vel": (0.0, 0.0, 0.0), "held_by": None, **fields})
    return replace(w, objects=out, sim_time=w.sim_time if t is None else t)


def with_arm(w, arm, **fields):
    robot, _, part = arm.partition("/")
    r = w.robots[robot]
    arms = {**r.arms, part: replace(r.arms[part], **fields)}
    return replace(w, robots={**w.robots, robot: replace(r, arms=arms)})


def on_top(w, obj, support):
    b, a = w.obj(support), w.obj(obj)
    return b.pose.pos[0], b.pose.pos[1], b.pose.pos[2] + b.size[2] + a.size[2]


def score(inst, traj):
    return inst.scorer.score(traj, {"dynamics": False})


def held(w, name):
    return {"held_by": w.arm_ids[0], "pose": Pose((*w.obj(name).pose.pos[:2], 1.1), w.obj(name).pose.quat)}


# -- Discovery -----------------------------------------------------------------
def test_ds4_coins_tried_in_the_dish_one_by_one_fail():
    inst, w0 = first("DS4", "hard")
    fake = inst.metadata["fake"]
    real = next(c for c in ("coin0", "coin1") if c != fake)
    tried = moved(w0, 5.0, **{real: on_top(w0, real, "dish")})
    back = moved(tried, 10.0, **{real: w0.obj(real).pose.pos})
    right = moved(back, 15.0, **{fake: on_top(back, fake, "dish")})
    s = score(inst, [w0, tried, back, right])
    assert not s.success and "no_real_coin_in_the_dish" in s.violations
    # the light coin straight into the dish breaks no rule of the dish -- only, with
    # no weighing before, the one that it be found out first (``fake_found_first``)
    s = score(inst, [w0, moved(w0, 15.0, **{fake: on_top(w0, fake, "dish")})])
    assert s.violations == ["fake_found_first"], s.violations


def test_ds1_a_present_still_moving_on_the_mat_does_not_count():
    inst, w0 = first("DS1", "hard")
    mine = inst.metadata["target"]
    sliding = moved(w0, 5.0, **{mine: {"pose": Pose(on_top(w0, mine, "red_mat")), "lin_vel": (0.2, 0.0, 0.0)}})
    assert not score(inst, [w0, sliding]).success
    assert score(inst, [w0, moved(w0, 5.0, **{mine: on_top(w0, mine, "red_mat")})]).success


# -- World knowledge -----------------------------------------------------------
def test_wk2_a_moment_aboard_does_not_let_a_hand_carry_a_thing_over():
    from ruibench.tasks.impl import wk2_river as R

    inst, w0 = first("WK2", "easy")
    rules = R.RiverRules(R.things("easy"), tuple(inst.scorer.violations[0][1].rules))
    raft = w0.obj(R.RAFT).pose.pos
    goat = w0.obj(R.GOAT)
    deck = (raft[0], raft[1], R.DECK + goat.size[2] + 0.001)
    far_x = R.ahead(R.RIVER[1])[0] + 0.06
    far = (far_x, goat.pose.pos[1], goat.pose.pos[2])
    aboard = moved(w0, 1.0, **{R.GOAT: deck})
    assert R.banks(aboard, R.things("easy"))[R.GOAT] == R.BOAT
    carried = moved(aboard, 2.0, **{R.GOAT: held(aboard, R.GOAT)})
    over = moved(carried, 3.0, **{R.GOAT: far})
    breach = rules.first_breach([w0, aboard, carried, over])
    assert breach and "without the raft" in breach[1]
    # rowed over: the raft at the far bank, the goat on it, then put ashore
    docked = replace(aboard, sim_time=4.0, articulations={**aboard.articulations, R.FERRY: R.TRAVEL})
    shifted = (raft[0] + R.TRAVEL, raft[1], raft[2])
    docked = moved(docked, **{R.RAFT: shifted, R.GOAT: (shifted[0], shifted[1], deck[2])})
    assert R.raft_side(docked) == R.FAR
    ashore = moved(docked, 5.0, **{R.GOAT: far})
    ashore = replace(ashore, articulations=docked.articulations)
    assert rules.first_breach([w0, aboard, docked, ashore]) is None


def test_wk3_a_domino_already_on_the_bell_does_not_ring_it():
    from ruibench.tasks.impl import wk3_dominoes as D

    inst, w0 = first("WK3", "easy")
    bell = w0.obj(D.BELL).pose.pos
    leant = moved(w0, 1.0, domino5={"pose": Pose((bell[0], bell[1], bell[2] + 0.02),
                                                 P.turn_y(math.radians(45)))})
    assert D.on_bell(leant, "domino5")
    let_go = replace(leant, sim_time=2.0, events_fired=(D.LET_GO,))
    rang = replace(let_go, sim_time=3.0, events_fired=(D.LET_GO, D.RANG))
    assert not D.RangAfterLetGo().evaluate([w0, leant, let_go, rang])
    clear = [replace(w, objects=w0.objects) for w in (let_go, rang)]
    assert D.RangAfterLetGo().evaluate([w0] + clear)


def test_wk5_a_weight_slid_into_the_next_pan_is_a_foul():
    from ruibench.tasks.impl import wk5_lever as L

    inst, w0 = first("WK5", "easy")
    weight = L.weights("easy")[1]
    size = w0.obj(weight).size[2]

    def in_pan(w, step, t):
        cx, cy, cz = L.pan_floor(1, step, L.lift_of(w, 1, step))
        return moved(w, t, **{weight: (cx, cy, cz + size + 0.002)})

    right = L.solution("easy")[weight]
    wrong = next(k for k in L.STEPS if k != right)
    s = score(inst, [w0, in_pan(w0, wrong, 2.0), in_pan(w0, right, 4.0)])
    assert "each_weight_in_one_pan" in s.violations
    assert "each_weight_in_one_pan" not in score(inst, [w0, in_pan(w0, right, 4.0)]).violations


def test_wk6_two_x_moves_in_one_turn_are_a_foul():
    inst, w0 = first("WK6", "hard")
    two = replace(w0, sim_time=5.0, events_fired=("x:0", "x:4", "o:8", "x:2", "won:X"))
    assert "one_move_a_turn" in score(inst, [w0, two]).violations
    fair = replace(w0, sim_time=5.0, events_fired=("x:0", "o:8", "x:4", "o:2", "x:6", "won:X"))
    s = score(inst, [w0, fair])
    assert "one_move_a_turn" not in s.violations


# -- Contact -------------------------------------------------------------------
def test_cr1_inside_the_box_is_in_its_own_axes():
    from ruibench.tasks.impl import cr1_shape_sorter as S

    inst, w0 = first("CR1", "easy", seed=0)
    box = w0.obj(S.SORTER)
    assert abs(math.remainder(S.yaw_of(box.pose.quat), math.pi / 2)) > math.radians(20)
    inner = S.BOX_HALF[0] - S.WALL
    low = box.pose.pos[2] + S.BOX_HALF[2] - S.LID - S.BLOCK_HALF_HEIGHT - 0.005

    def at(local):
        dx, dy, _ = rotate(box.pose.quat, (local[0], local[1], 0.0))
        return moved(w0, 1.0, **{S.SHAPES[0]: (box.pose.pos[0] + dx, box.pose.pos[1] + dy, low)})

    assert S.through(at((inner - 0.004, inner - 0.004)), S.SHAPES[0])     # a corner inside
    assert not S.through(at((inner + 0.015, 0.0)), S.SHAPES[0])           # by a wall, outside


def test_cr3_a_stone_squeezed_by_a_part_closed_grip_is_a_foul():
    from ruibench.tasks.impl import cr3_curling as C
    from ruibench.tasks.impl._common import ONE_ARM

    inst, w0 = first("CR3", "easy")
    stone = w0.obj(C.STONE).pose.pos
    arm = next(a for a in w0.arm_ids if a.endswith(C.RIGHT))
    command = 0.3                                    # robot_gripper(width_m=...): below GRIP_CLOSING
    sent = (1 - command) * model_of(ONE_ARM).max_opening
    gripped = with_arm(moved(w0, 1.0), arm, gripper=command, opening=sent + 0.02,
                       eef_pose=Pose(stone, w0.arm(C.RIGHT).eef_pose.quat))
    assert "never_squeezed" in score(inst, [w0, gripped]).violations
    pushing = with_arm(gripped, arm, opening=sent)  # the fingers where they were sent: nothing between
    assert "never_squeezed" not in score(inst, [w0, pushing]).violations


# -- Dynamic interaction -------------------------------------------------------
def test_di3_two_ducklings_on_the_road_at_once_is_a_foul():
    from ruibench.tasks.impl import di3_red_light as D

    inst, w0 = first("DI3", "easy")
    x, _ = D.ahead(D.ROAD_X, D.ROAD_Y)
    a, b = D.DUCKS[:2]
    both = moved(w0, 1.0, **{a: (x, w0.obj(a).pose.pos[1], w0.obj(a).pose.pos[2]),
                             b: (x, w0.obj(b).pose.pos[1], w0.obj(b).pose.pos[2])})
    assert "one_at_a_time" in score(inst, [w0, both]).violations
    one = moved(w0, 1.0, **{a: (x, w0.obj(a).pose.pos[1], w0.obj(a).pose.pos[2])})
    assert "one_at_a_time" not in score(inst, [w0, one]).violations


def _simon(level, hedge):
    """DI4 played to every call Simon makes: the named toy alone, or every toy."""
    from ruibench.tasks.impl import di4_simon_says as M

    inst, w0 = first("DI4", level)
    calls = M.schedule(level, 15)
    toys = list(M.LEVEL_TOYS[level])
    reactor, eng = inst.reactor_factory(), FakeEngine()
    arm = next(a for a in w0.arm_ids if a.endswith(M.RIGHT))
    tap = {t: (*w0.obj(t).pose.pos[:2], w0.obj(t).pose.pos[2] + w0.obj(t).size[2] + 0.01) for t in toys}
    end = calls[-1][0] + M.CALLS[level][1] * 8.0
    plan = {}
    for when, simon, named in calls:
        if simon:
            for i, toy in enumerate(toys if hedge else [named]):
                plan[round(when + 4.0 + 1.5 * i, 1)] = toy
    traj, t, until, cur = [w0], 0.0, -1.0, None
    while t < end + 8.0:
        t = round(t + 0.1, 1)
        if t in plan:
            cur, until = plan[t], t + 0.5
        pos = tap[cur] if (cur and t < until) else (0.0, 0.0, 1.2)
        w = with_arm(replace(w0, sim_time=t, events_fired=tuple(eng.events), instruction=eng.instruction,
                             instruction_version=eng.version), arm,
                     eef_pose=Pose(pos, w0.arm(M.RIGHT).eef_pose.quat))
        reactor.tick(eng, w)
        traj.append(replace(w, events_fired=tuple(eng.events)))
    return score(inst, traj)


def test_di4_touching_every_toy_at_each_call_is_a_foul():
    s = _simon("easy", hedge=True)
    assert not s.success and "only_what_simon_says" in s.violations
    s = _simon("easy", hedge=False)
    assert s.success, s.violations


def test_di5_a_mallet_held_over_the_holes_whacks_nothing():
    """Hovering where a risen mole meets the mallet, never swinging: no whack."""
    from ruibench.tasks.impl import _toys as T
    from ruibench.tasks.impl import di5_whack_a_mole as M

    inst, w0 = first("DI5", "easy")
    reactor, eng = inst.reactor_factory(), FakeEngine()
    _, holes = M.build_scene("easy")
    sides = {a: [n for aa, _, n in M.hole_names() if aa == a] for a in (M.LEFT, M.RIGHT)}
    z_head = T.MOLE_TOP + T.MOLE_RISE + 0.005 + 0.010 + M.HEAD_R
    plan = M.schedule("easy", 15)
    end = M.game_start(8.0) + plan[-1][0] + M.SCHEDULE["easy"][1] * 8.0 + 1.0
    traj, t = [w0], 0.0
    while t < end:
        t = round(t + 0.1, 1)
        w = replace(w0, sim_time=t, events_fired=tuple(eng.events),
                    articulations={**w0.articulations, **eng.targets})
        for arm in (M.LEFT, M.RIGHT):
            name, k = M.mallet_name(arm), int(t / 0.6) % 3
            o = w0.obj(name)
            hx, hy = holes[sides[arm][k]]
            dx, dy, dz = rotate(o.pose.quat, (M.HEAD_X, M.HEAD_DY, M.HEAD_DZ))
            w = moved(w, **{name: {"pose": Pose((hx - dx, hy - dy, z_head - dz), o.pose.quat),
                                   "held_by": arm}})
        reactor.tick(eng, w)
        traj.append(replace(w, events_fired=tuple(eng.events),
                            articulations={**w0.articulations, **eng.targets}))
    assert not reactor.whacked and not score(inst, traj).success


def test_di6_covers_looked_under_one_by_one_are_a_foul():
    from ruibench.tasks.impl import di6_carousel as M

    def play(peek):
        inst, w0 = first("DI6", "hard")
        arm = w0.arm_ids[0]
        clock = [30.0]

        def snap(w):
            clock[0] += 0.1
            return replace(w, sim_time=clock[0], events_fired=("turn_1", M.STOPPED))

        w = snap(w0)
        traj = [w0, w]
        for c in (("cover2", "cover3") if peek else ()):
            p = w.obj(c).pose.pos
            w = moved(w, **{c: {"held_by": arm, "pose": Pose((p[0], p[1], p[2] + 0.1))}})
            traj.append(snap(w))
            w = moved(w, **{c: Pose(p).pos})
            traj.append(snap(w))
        p = w.obj("cover1").pose.pos
        w = moved(w, **{"cover1": {"held_by": arm, "pose": Pose((p[0], p[1], p[2] + 0.1))}})
        traj.append(snap(w))
        w = moved(w, **{"cover1": (p[0], p[1] + 0.3, 0.8495)})
        traj.append(snap(w))
        d = w.obj("plate").pose.pos
        w = moved(w, cake=(d[0], d[1], d[2] + 0.025))
        traj.append(snap(w))
        return score(inst, traj)

    s = play(peek=True)
    assert not s.success and "only_one_cover_lifted" in s.violations
    assert "only_one_cover_lifted" not in play(peek=False).violations


# -- v2.3.4: checks too strict, or games begun before the first look -------------
def test_di5_the_moles_wait_for_the_first_looks_and_a_strike_from_low_counts():
    """The first mole comes up 3 T_ref in, after an agent has read, looked and got
    ready; and a head brought over a risen mole a centimetre above it, then
    struck straight down through it, whacks it (until v2.3.4 the first moles were
    gone before any reply, and such a strike did not count)."""
    from ruibench.tasks.impl import _toys as T
    from ruibench.tasks.impl import di5_whack_a_mole as M

    inst, w0 = first("DI5", "easy")
    reactor, eng = inst.reactor_factory(), FakeEngine()
    _, holes = M.build_scene("easy")
    assert M.game_start(8.0) == 24.0
    mole = M.schedule("easy", 15)[0][1][0]
    arm = M.LEFT if mole.startswith("mole_left") else M.RIGHT
    name, o = M.mallet_name(arm), w0.obj(M.mallet_name(arm))
    dx, dy, dz = rotate(o.pose.quat, (M.HEAD_X, M.HEAD_DY, M.HEAD_DZ))
    hx, hy = holes[mole]
    top = T.MOLE_TOP + T.MOLE_RISE
    clock = [0.0]

    def tick(x, y, under):
        clock[0] = round(clock[0] + 0.1, 1)
        w = replace(w0, sim_time=clock[0], events_fired=tuple(eng.events),
                    articulations={**w0.articulations, **eng.targets})
        w = moved(w, **{name: {"pose": Pose((x - dx, y - dy, under + M.HEAD_R - dz), o.pose.quat),
                               "held_by": arm}})
        reactor.tick(eng, w)

    while clock[0] < 23.8:                            # the head waits 8 cm off the hole
        tick(hx + 0.08, hy, top + 0.010)
    assert M.STARTED not in eng.events and not any(e.endswith("_up") for e in eng.events)
    for _ in range(5):
        tick(hx + 0.08, hy, top + 0.010)
    assert f"{mole}_up" in eng.events
    for k in range(8, -1, -1):                       # in sideways a centimetre over the mole
        tick(hx + 0.01 * k, hy, top + 0.010)
    assert not reactor.whacked
    for k in range(1, 7):                            # and straight down through it
        tick(hx, hy, top + 0.010 - 0.005 * k)
    assert reactor.whacked == [mole]


def test_di6_the_table_turns_only_after_the_first_looks():
    from ruibench.tasks.impl import di6_carousel as M

    inst, w0 = first("DI6", "hard")
    reactor, eng = inst.reactor_factory(), FakeEngine()
    assert M.turn_start(8.0) == 24.0
    t = 0.0
    while t < 30.0:
        t = round(t + 0.1, 1)
        reactor.tick(eng, replace(w0, sim_time=t, events_fired=tuple(eng.events)))
        assert t >= 24.0 or "turn_1" not in eng.events
    assert "turn_1" in eng.events


def test_cl6_a_knight_well_inside_its_square_stands_on_it():
    from ruibench.tasks.impl import cl6_chess as C

    cx, cy = C.centre((3, 3))
    assert C.square_of((cx + 0.020, cy)) == (3, 3)
    assert C.square_of((cx + 0.014, cy - 0.014)) == (3, 3)
    assert C.square_of((cx + 0.025, cy)) is None     # nearer the edge than the middle


def test_wk6_a_piece_brushed_once_the_game_is_won_is_no_foul():
    inst, w0 = first("WK6", "easy")
    after = replace(w0, sim_time=5.0, events_fired=("x:6", "won:X", "foul:moved_a_piece"))
    assert "fair_play" not in score(inst, [w0, after]).violations
    during = replace(w0, sim_time=5.0, events_fired=("foul:moved_a_piece", "x:6", "won:X"))
    assert "fair_play" in score(inst, [w0, during]).violations


def test_cr3_closed_fingers_pushing_the_stone_do_not_grip_it():
    """Closed fingers pushing the stone were spread 2-3 mm by its rim, and the
    engine called it held: a push onto the button failed as a grip (v2.3.4)."""
    from ruibench.tasks.impl import cr3_curling as C

    inst, w0 = first("CR3", "easy")
    arm = next(a for a in w0.arm_ids if a.endswith(C.RIGHT))
    stone = w0.obj(C.STONE)
    beside = Pose((stone.pose.pos[0], stone.pose.pos[1] - 0.038, stone.pose.pos[2]),
                  w0.arm(C.RIGHT).eef_pose.quat)
    pushing = with_arm(moved(w0, 1.0, **{C.STONE: {"held_by": arm, "pose": stone.pose}}), arm,
                       gripper=1.0, opening=0.0025, eef_pose=beside)
    assert "never_gripped" not in score(inst, [w0, pushing]).violations
    gripping = with_arm(pushing, arm, opening=0.058,
                        eef_pose=Pose(stone.pose.pos, w0.arm(C.RIGHT).eef_pose.quat))
    assert "never_gripped" in score(inst, [w0, gripping]).violations


def test_cl5_a_piece_a_little_off_its_place_still_fills_it():
    from ruibench.tasks.impl import cl5_tangram as M

    inst, w0 = first("CL5", "easy")
    mine, _ = M.shapes_for(15)

    def laid(shift=0.0, turn=0.0):
        out = {}
        for k, (name, (xy, yaw)) in enumerate(M.solved(mine).items()):
            o = w0.obj(name)
            if k == 0:                              # the first piece: a large triangle
                xy, yaw = (xy[0] + shift, xy[1]), yaw + math.radians(turn)
            out[name] = {"pose": Pose((xy[0], xy[1], o.pose.pos[2]),
                                      (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)))}
        return moved(w0, 1.0, **out)

    assert M.filled(laid(), mine) == len(M.PIECES)
    assert M.filled(laid(shift=0.014, turn=14.0), mine) == len(M.PIECES)
    assert M.filled(laid(shift=0.017), mine) == len(M.PIECES) - 1
