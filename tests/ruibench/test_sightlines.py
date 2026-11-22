"""The fixed camera sees what a task needs: marks, objects, the grasp."""

import pytest

from ruibench.core.types import Pose
from ruibench.engines.base import BodySpec, MarkerSpec, marker_strokes
from ruibench.tasks import sightlines
from ruibench.tasks.instance import build


def _block_scene():
    """A red block on the single-arm table -- where v0's WK5 started it."""
    from ruibench.core.rng import SeedBundle
    from ruibench.tasks.impl._common import cube, table_scene

    return table_scene("block", (cube("block", SeedBundle(0).jitter("block", (0.08, -0.16), 0.03),
                                      "red"),))


def test_a_steep_camera_hides_the_grasp(monkeypatch):
    import dataclasses
    import math

    import ruibench.engines.base as base
    from ruibench.engines import robots

    scene = _block_scene()
    assert sightlines.hidden_in_grasp(scene, "block") <= sightlines.MAX_HIDDEN_IN_GRASP
    # a camera looking down steeply (v0's first view was 71 degrees down): the
    # hand covers most of what it holds
    monkeypatch.setitem(robots.MODELS, "fr3",
                        dataclasses.replace(robots.FR3, camera_pitch=math.radians(75)))
    base._fitted_camera.cache_clear()
    try:
        assert sightlines.hidden_in_grasp(scene, "block") > sightlines.MAX_HIDDEN_IN_GRASP
        assert any("under the hand" in line for line in sightlines.check_scene(scene))
    finally:
        monkeypatch.undo()          # the camera fit is cached by model name
        base._fitted_camera.cache_clear()


def test_an_object_behind_a_wall_is_flagged_unless_hidden_on_purpose():
    import dataclasses

    scene = _block_scene()
    block = next(b for b in scene.bodies if b.name == "block")
    x, y, z = block.pose.pos
    # a tall panel between the block and the camera (which is on the +x side)
    wall = BodySpec("panel", "box", (0.01, 0.10, 0.10), Pose((x + 0.08, y, z + 0.08)), fixed=True)
    blocked = dataclasses.replace(scene, bodies=scene.bodies + (wall,))
    assert sightlines.hidden_at_start(blocked, "block") > sightlines.MAX_HIDDEN_AT_START
    assert any("at the start" in line for line in sightlines.check_scene(blocked))
    meant = dataclasses.replace(block, attrs={**block.attrs, "hidden_at_start": True})
    on_purpose = dataclasses.replace(blocked, bodies=(meant, wall) + tuple(
        b for b in scene.bodies if b.name != "block"))
    assert not any("at the start" in line for line in sightlines.check_scene(on_purpose))


def test_marks_are_drawn_from_strokes_every_renderer_shares():
    cross = marker_strokes(MarkerSpec("m", shape="cross", size=(0.03, 0.03, 0.001)))
    assert len(cross) == 2 and max(abs(v) for seg in cross for p in seg for v in p) == 0.03
    assert len(marker_strokes(MarkerSpec("s", shape="square", size=(0.045, 0.045, 0.001)))) == 4
    letter = marker_strokes(MarkerSpec("l", shape="label", size=(0.035, 0.035, 0.001), text="A"))
    xs = [p[0] for seg in letter for p in seg]
    # upright in the fixed camera: the apex of the A is the point farthest from it (-x)
    assert min(xs) == pytest.approx(-0.0175) and max(xs) == pytest.approx(0.0175)
    with pytest.raises(ValueError):
        marker_strokes(MarkerSpec("l", shape="label", text="@"))




def test_the_arms_where_they_start_hide_what_lies_under_them():
    """v13: the start check counts the arms folded at rest.  The ARX X5s' head
    camera looks past the folded grippers, which hid WK4's near cutlery while
    the check, counting other bodies only, said 0%."""
    from dataclasses import replace

    from ruibench.tasks import sightlines
    from ruibench.tasks.impl import wk4_set_table as W
    from ruibench.tasks.impl._common import ahead

    scene = W.build_scene("easy", 0)
    assert not sightlines.check_scene(scene)
    # the fork laid where the near row used to be, under the left gripper
    x, y = ahead(0.255, 0.18, arm="left")
    bodies = tuple(replace(b, pose=replace(b.pose, pos=(x, y, b.pose.pos[2])))
                   if b.name == "fork_left" else b for b in scene.bodies)
    hidden = sightlines.hidden_at_start(replace(scene, bodies=bodies), "fork_left")
    assert hidden > sightlines.MAX_HIDDEN_AT_START
    # and a thing an arm starts out holding is not counted as hidden by it
    from ruibench.tasks.instance import build
    cl1 = build("CL1", "easy", 0).scene
    held = [s.holding for s in cl1.arm_starts if s.holding]
    assert held and all(sightlines.hidden_at_start(cl1, n) < 0.3 for n in held)
