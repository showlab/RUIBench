"""WK1's rules hold whatever a gripper is detected to hold (v2.3).

A rollout showed an agent lifting two rings as one: only the top ring counted as
held, the one under it rode along, and the move-by-move check -- which reads
moves from what is held -- saw a single legal move.  The rules are now also
checked from where the rings are.  Each trajectory below is built by hand."""

from ruibench.core.types import ArmState, ObjectState, Pose, RobotState, WorldState
from ruibench.tasks.impl import wk1_hanoi as H
from ruibench.tasks.instance import build

INST = build("WK1", level="easy", seed=15)
STAND = INST.metadata["stand"]
RINGS = H.rings_for("easy")                                   # smallest first
SMALL, MEDIUM, LARGE = RINGS
START = {n: H.PEGS.index(p) for n, p in INST.metadata["start"].items()}


def on(k, level):
    """A ring's pose threaded on peg ``k``, ``level`` rings up from the base."""
    x, y = H.peg_xy(STAND, k)
    return Pose((x, y, H.BASE_TOP + H.RING_HALF + 2 * H.RING_HALF * level))


def above(k, dz):
    x, y = H.peg_xy(STAND, k)
    return Pose((x, y, H.PEG_TOP + H.RING_HALF + dz))


def world(t, poses, held=()):
    objects = {n: ObjectState(n, p, (2 * H.OUTER[n],) * 2 + (2 * H.RING_HALF,), "ring",
                              held_by="robot0/right" if n in held else None)
               for n, p in poses.items()}
    objects[H.STAND] = ObjectState(H.STAND, Pose((*STAND[:2], H.TABLE + 0.01)))
    arms = {"right": ArmState("robot0/right", Pose((0.3, 0.0, 1.1)))}
    return WorldState(t, objects, {"robot0": RobotState("robot0", arms)})


def start_poses():
    piles = {}
    poses = {}
    for name in reversed(RINGS):                              # largest first: the bottom
        k = START[name]
        poses[name] = on(k, piles.get(k, 0))
        piles[k] = piles.get(k, 0) + 1
    return poses


def solve(t0, poses):
    """The three legal moves from the start, each ring alone, then the tower still."""
    frames, t = [], t0
    middle = 1
    for ring, peg, level in ((SMALL, middle, 0), (MEDIUM, H.GOAL, 1), (SMALL, H.GOAL, 2)):
        frames.append(world(t, {**poses, ring: above(START.get(ring, 0), 0.02)}, held=(ring,)))
        poses = {**poses, ring: on(peg, level)}
        frames.append(world(t + 0.5, poses))
        t += 1.0
    frames += [world(t + 0.5 * i, poses) for i in range(6)]   # 2.5 s still
    return frames


def score(traj):
    return INST.scorer.score(traj, {"dynamics": False})


def test_a_legal_solution_counts():
    assert START[LARGE] == H.GOAL and START[SMALL] == START[MEDIUM] != H.GOAL
    s = score([world(0.0, start_poses())] + solve(1.0, start_poses()))
    assert s.success, (s.violations, s.illegal_moves, s.flags)


def test_two_rings_lifted_as_one_break_the_rules():
    """The top ring held, the one under it carried along: one held, two moved."""
    k = START[SMALL]
    lifted = [world(0.5 + 0.25 * i, {**start_poses(), SMALL: above(k, 0.03 + H.RING_HALF * 2),
                                     MEDIUM: above(k, 0.03)}, held=(SMALL,)) for i in range(5)]
    traj = [world(0.0, start_poses())] + lifted + solve(3.0, start_poses())
    s = score(traj)
    assert not s.success and "one_ring_at_a_time" in s.violations
    assert INST.scorer.machine.violations(traj) == []         # what the move-by-move check missed


def test_a_ring_left_off_the_pegs_and_one_on_a_smaller_ring_break_the_rules():
    poses = start_poses()
    x, y = H.peg_xy(STAND, 1)
    lying = {**poses, MEDIUM: Pose((x + 0.05, y, H.BASE_TOP + H.RING_HALF))}   # on the board
    traj = ([world(0.0, poses)] + [world(0.5 + 0.5 * i, lying) for i in range(4)]
            + solve(3.0, poses))
    s = score(traj)
    assert not s.success and "rings_put_down_on_pegs" in s.violations
    upside = {**poses, SMALL: on(1, 0), MEDIUM: on(1, 1)}                      # medium on small
    traj = ([world(0.0, poses)] + [world(0.5 + 0.5 * i, upside) for i in range(3)]
            + solve(3.0, poses))
    s = score(traj)
    assert not s.success and "never_on_a_smaller_ring" in s.violations
