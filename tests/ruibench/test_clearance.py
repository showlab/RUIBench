"""Scene geometry: the gripper can get at what a task asks it to pick up.

The first CL3 run on Isaac failed every episode with descents stopping 7 cm
short.  The cause was the scene, not the controller: bins stood inside the
Panda hand's 20 cm footprint.  These tests keep that kind of bug out of CI.
"""

from ruibench.core.rng import SeedBundle
from ruibench.core.types import Pose
from ruibench.engines.base import BodySpec, SceneSpec
from ruibench.tasks.clearance import FINGER_OUTER, HAND_HALF_Y, check_scene, grasp_conflicts
from ruibench.tasks.impl._common import BIN_SIZE, BIN_XY, CUBE, basket, cube, grid_layout

TABLE = 0.80


def _scene(*bodies):
    return SceneSpec("t", bodies=tuple(bodies), table_height=TABLE)


def test_the_layout_that_failed_on_isaac_is_flagged():
    """Bins at y = +-0.24, 14 cm deep, and a cube 5.6 cm off the centre line:
    the first CL3 layout, whose descents all stopped on the bin wall."""
    tall_bin = BodySpec("bin", "bin", (0.10, 0.15, 0.07), Pose((0.20, -0.24, TABLE + 0.07),
                                                                (0.7071, 0.0, 0.0, 0.7071)),
                        fixed=True, attrs={"container": "open"})
    scene = _scene(cube("cube0", (0.154, -0.056), "red"), tall_bin)
    assert grasp_conflicts(scene, "cube0") == ["hand would hit bin"]


def test_a_low_bin_under_the_hand_is_fine_but_one_beside_the_fingers_is_not():
    """Height matters as much as footprint: the hand passes 4 cm above the grasp."""
    low = BodySpec("mat", "box", (0.10, 0.10, 0.002), Pose((0.10, 0.10, TABLE + 0.002)), fixed=True)
    assert grasp_conflicts(_scene(cube("c", (0.10, 0.05), "red"), low), "c") == []
    neighbour = cube("n", (0.10, 0.05 + FINGER_OUTER + CUBE[0]), "blue")
    assert grasp_conflicts(_scene(cube("c", (0.10, 0.05), "red"), neighbour), "c") == [
        "finger would hit n"]


def test_the_shared_layout_leaves_room_for_the_hand():
    """The constants the sorting tasks share, checked against the hand directly:
    the object furthest off the centre line, over many seeds, still leaves the
    end of the hand short of the bin's inner wall."""
    furthest = max(abs(y) for seed in range(50)
                   for _, y in grid_layout(SeedBundle(seed), "t", 6))
    inner_wall = BIN_XY[1][1] - BIN_SIZE[0]          # bins are turned: x half extent is along y
    assert furthest + HAND_HALF_Y < inner_wall
    scene = _scene(cube("c", (0.15, furthest), "red"), basket("b", BIN_XY[1], "blue"))
    assert check_scene(scene) == []


def test_fixed_bodies_and_opted_out_bodies_need_no_grasp():
    wall = BodySpec("wall", "box", (0.10, 0.01, 0.10), Pose((0.10, 0.0, TABLE + 0.10)), fixed=True)
    prop = cube("prop", (0.10, 0.03), "red", graspable=False)
    assert check_scene(_scene(wall, prop)) == []


def test_a_body_on_an_undriven_joint_must_clear_the_table():
    """BC2's sprung drawer rested on the table and, after a hand had pressed it
    down, stayed open on Isaac in 3 releases of 8."""
    from ruibench.engines.base import ArticulationSpec
    from ruibench.tasks.clearance import JOINT_CLEARANCE, resting_on_table

    def drawer(bottom, actuated):
        body = BodySpec("drawer", "box", (0.10, 0.10, 0.02), Pose((0.20, 0.0, bottom + 0.02)))
        joint = ArticulationSpec("slide", axis=(1.0, 0.0, 0.0), limits=(-0.2, 0.0),
                                 actuated=actuated, stiffness=30.0, parent="drawer")
        return SceneSpec("t", bodies=(body,), articulations=(joint,), table_height=TABLE)

    assert resting_on_table(drawer(TABLE, actuated=False)) == [
        "drawer on slide is 0.0 mm above the table, needs 2 mm"]
    assert resting_on_table(drawer(TABLE + JOINT_CLEARANCE, actuated=False)) == []
    assert resting_on_table(drawer(TABLE, actuated=True)) == []      # a drive pushes through
