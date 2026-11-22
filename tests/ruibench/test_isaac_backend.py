"""The Isaac backend, checked as far as a machine without it can check.

Everything here runs with no GPU and no Isaac install: it verifies the parts
that must hold *before* hardware gets involved -- that nothing imports Isaac at
module level, that the engine declares itself results-grade while the others do
not, and that the asset indirection holds.

The behaviour that needs hardware lives in `ruibench doctor` (Gate A), not here:
a test that silently skips is worse than a check that reports what is missing.
"""

import importlib
import subprocess
import sys

import pytest

from ruibench.engines import ENGINES, results_grade
from ruibench.engines.isaac import assets


def test_only_the_photorealistic_engine_can_produce_results():
    """Frontier models are judged on what they see, so a low-fidelity render
    would measure domain gap rather than ability.  The real station's pictures
    are the world itself; its results are kept apart from the simulation's."""
    assert results_grade("isaac")
    assert not results_grade("mock")
    assert results_grade("franka")
    assert set(ENGINES.names()) == {"isaac", "mock", "franka"}


def test_importing_ruibench_does_not_import_isaac():
    """SimulationApp must start before any omni/pxr/isaaclab import, and the
    core has to work on a machine with no GPU.  Checked in a fresh interpreter
    because this session may have imported things already."""
    code = (
        "import ruibench, ruibench.tasks, ruibench.reports, ruibench.doctor, "
        "ruibench.engines.isaac, sys;"
        "bad=[m for m in sys.modules if m.split('.')[0] in "
        "{'omni','pxr','isaacsim','isaaclab','carb'}];"
        "print(bad)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", f"eagerly imported: {out.stdout.strip()}"


def test_every_logical_name_is_either_a_mesh_or_a_declared_primitive():
    """A name has to resolve one way or the other.  Isaac's YCB subset ships no
    plate and no ball, so those are primitives -- a chosen shape, with the
    reason recorded, not a silent gap."""
    known = set(assets.load_catalogue()) | set(assets.primitives())
    for name in ("ball", "basket", "plate", "table"):
        assert name in known, name


def test_task_cards_never_name_a_usd_path():
    """One indirection, so a moved asset changes one file and not 28 cards."""
    from ruibench import tasks

    for task in tasks.load_all():
        known = set(assets.load_catalogue()) | set(assets.primitives())
        for named in task.catalogue_assets:
            assert ".usd" not in named
            assert named in known, (task.id, named)


def test_primitives_record_why_they_are_primitives():
    """"We built it" is a decision with consequences -- a cylinder plate has no
    rim -- so the reason is part of the catalogue, not folklore."""
    for name, prim in assets.primitives().items():
        assert len(prim.reason) > 20, name


def test_unverified_assets_are_reported_not_assumed():
    """Until `ruibench doctor --assets` resolves them on a live asset server,
    every catalogue entry is unverified and the validator says so."""
    from ruibench import tasks

    unverified = set(assets.unverified())
    warned = {f.where for f in tasks.validate()
              if f.level == "warning" and "asset" in f.message}
    using = {t.id for t in tasks.load_all()
             if set(t.catalogue_assets) & unverified}
    assert using <= warned


def test_eula_is_not_accepted_on_the_users_behalf():
    """Accepting a licence agreement is the operator's decision."""
    from ruibench.engines.isaac import app

    source = (importlib.import_module("ruibench.engines.isaac.app").__file__)
    text = open(source).read()
    assert 'os.environ["OMNI_KIT_ACCEPT_EULA"] = ' not in text
    assert 'setdefault("OMNI_KIT_ACCEPT_EULA"' not in text
    assert "OMNI_KIT_ACCEPT_EULA" in app.EULA_HINT


def test_the_doctor_never_starts_the_simulator_in_its_own_process():
    """A simulation app can be created once per process and does not tear down
    cleanly, so every check that needs one runs in a child process."""
    import inspect

    from ruibench import doctor

    source = inspect.getsource(doctor)
    assert "app.launch(" not in source
    assert "ruibench.engines.isaac.checks" in source


def test_hardware_checks_are_importable_without_isaac():
    from ruibench.engines.isaac import checks

    assert set(checks.CHECKS) >= {"ik", "dual_arm", "render", "table"}


def test_the_controller_tracks_the_tool_centre_point():
    """Task code gives object positions; a grasp happens between the fingertips.
    Controlling the hand frame put every target 10 cm off."""
    import math

    from ruibench.core.types import GRASP_DOWN, Pose
    from ruibench.engines.isaac.engine import hand_to_tcp, tcp_to_hand
    from ruibench.engines.robots import FR3

    hand = Pose((0.1, 0.0, 1.0), GRASP_DOWN)
    tcp = hand_to_tcp(hand)
    assert math.isclose(tcp.pos[2], 1.0 - FR3.tcp_from_body.pos[2], abs_tol=1e-9)   # tool points down
    back = tcp_to_hand(tcp)
    assert all(math.isclose(a, b, abs_tol=1e-9) for a, b in zip(back.pos, hand.pos))


def test_commanded_motion_is_speed_limited():
    """A move_to must not teleport the joint targets: that is what made the
    arm swing behind its own base."""
    from ruibench.core.types import Pose
    from ruibench.engines.isaac.engine import MAX_LINEAR_SPEED, _step_towards

    cur, goal = Pose((0.0, 0.0, 0.0)), Pose((1.0, 0.0, 0.0))
    nxt = _step_towards(cur, goal, 0.1)
    assert abs(nxt.pos[0] - MAX_LINEAR_SPEED * 0.1) < 1e-9


def test_reference_policies_point_the_tool_down():
    """The identity quaternion points a real gripper at the ceiling."""
    from ruibench.core.config import RunConfig
    from ruibench.core.types import GRASP_DOWN
    from ruibench.interfaces.commands import MoveTo
    from ruibench.interfaces.base import TurnContext
    from ruibench.tasks.instance import build

    inst = build("WK5", "hard", 0, RunConfig())
    policy = inst.reference_factory(inst)
    from ruibench.engines import make_engine
    engine = make_engine("mock")
    engine.load(inst.scene, 0)
    world = engine.state()
    ctx = TurnContext(observation=None, task_id="WK5", instruction="", arm_ids=world.arm_ids,
                      turn_index=0, sim_time=0.0, budget_left={}, extras={"world": world})
    moves = [c for c in policy.plan(world, ctx) if isinstance(c, MoveTo)]
    assert moves and all(m.pose.quat == GRASP_DOWN for m in moves)


def _self_calls_and_definitions(cls_node):
    import ast

    defined = {n.name for n in cls_node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assigned = set()
    called = set()
    for node in ast.walk(cls_node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id == "self":
            if isinstance(getattr(node, "ctx", None), ast.Store):
                assigned.add(node.attr)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) \
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "self":
            called.add(node.func.attr)
    return called, defined, assigned


def test_the_isaac_engine_defines_every_method_it_calls():
    """Isaac-only code paths are not exercised without a GPU.  A method lost in
    an edit (it happened: ``eef_pose``) then surfaces only as a dead worker on
    the hardware.  This catches it statically."""
    import ast
    import inspect

    from ruibench.engines.base import Engine
    from ruibench.engines.isaac import engine as isaac_engine

    tree = ast.parse(inspect.getsource(isaac_engine))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "IsaacEngine")
    called, defined, assigned = _self_calls_and_definitions(cls)
    inherited = {n for n in dir(Engine) if not n.startswith("__")}
    missing = called - defined - assigned - inherited
    assert not missing, f"IsaacEngine calls undefined methods: {sorted(missing)}"


def test_every_engine_implements_the_whole_contract():
    """Abstract methods left unimplemented only fail at instantiation, which for
    the Isaac engine means on the hardware."""
    import inspect

    from ruibench.engines.base import Engine
    from ruibench.engines.isaac.engine import IsaacEngine
    from ruibench.engines.mock import MockEngine

    for cls in (IsaacEngine, MockEngine):
        assert not inspect.isabstract(cls), f"{cls.__name__}: {cls.__abstractmethods__}"


def test_verifying_assets_never_rewrites_the_hand_written_catalogue(tmp_path, monkeypatch):
    """The catalogue carries the reasons behind its choices in comments.  An
    earlier version re-dumped it on every check and erased them."""
    from ruibench.engines.isaac import assets

    before = assets.CATALOGUE_PATH.read_text()
    monkeypatch.setattr(assets, "LOCK_PATH", tmp_path / "assets.lock.json")
    assets.mark_verified({name: True for name in assets.load_catalogue()})
    assert assets.CATALOGUE_PATH.read_text() == before
    assert (tmp_path / "assets.lock.json").exists()


def test_hollow_assets_are_not_convex_hulls():
    """A convex hull fills a bowl in: "the ball is in the bowl" would be false
    while it looks true."""
    from ruibench.engines.isaac import assets

    catalogue = assets.load_catalogue()
    for name in ("rattan_basket", "sand_pail", "doll_bed", "doll_sofa"):
        assert catalogue[name].collision_approximation == "convexDecomposition", name


def test_every_asset_a_task_shows_says_who_made_it():
    """CC BY asks to be credited wherever the thing is shown: every catalogue
    asset a built task puts on the table carries its licence and attribution."""
    from ruibench.engines.isaac.assets import LICENCE_NAMES, credits
    from ruibench.tasks.instance import assets_shown

    lines = credits(assets_shown())             # raises for an asset with no attribution
    assert {c.licence for c in lines} <= set(LICENCE_NAMES.values())
    assert sum(len(c.items) for c in lines) >= 20              # not an empty list


def test_every_catalogue_path_is_well_formed():
    """A bad edit once wrote ``{NUCLEUS, footprint: round}`` into three USD
    paths; nothing failed until a scene tried to load them on the GPU."""
    import re

    from ruibench.engines.isaac import assets

    pattern = re.compile(r"^\{NUCLEUS\}/[A-Za-z0-9_./-]+\.usd$")
    for name, entry in assets.load_catalogue().items():
        if entry.source:
            # v2: made ready by ``prepare`` from a named source, with its terms
            assert entry.usd == f"{{PREPARED}}/{entry.name}/{entry.name}.usda", name
            assert entry.source.split(":", 1)[0] in ("polyhaven", "gso", "objaverse", "ycb", "nucleus"), name
            assert entry.licence and entry.attribution, f"{name}: licence and attribution"
        else:
            assert pattern.match(entry.usd), f"{name}: malformed path {entry.usd!r}"
        assert entry.footprint in ("rect", "round"), name
        assert entry.up in ("y", "z"), name


def test_every_catalogue_entry_is_verified_and_measured_where_it_matters():
    """Graspable props need measured geometry: their sizes feed the predicates."""
    from ruibench.engines.isaac import assets

    catalogue = assets.load_catalogue()
    assert all(e.verified for e in catalogue.values()), assets.unverified()
    for name, entry in catalogue.items():
        if entry.category != "table":
            assert entry.half_extents is not None, f"{name} not measured"


def test_the_external_camera_frames_the_whole_workspace():
    """Every object and mark, each commanded arm's base, the gripper at working
    height and the gripper where it starts fit in one 512 px image -- by
    projection, no renderer needed.  Isaac Lab's stacking-task camera cut the
    bins off; the next framing left the starting gripper above the image; the
    first station camera cut the arm off at the top and wasted the bottom.
    Since v2 a single-arm task frames its own part of the table: there the
    arm's base may fall outside, the fingers and the lower hand may not."""
    from ruibench.core.geometry import quat_conj, rotate
    from ruibench.engines.base import (
        HORIZONTAL_APERTURE, arm_start_pose, camera_focal_length, external_camera_pose,
        marker_strokes_world,
    )
    from ruibench.engines.robots import arm_base, model_of
    from ruibench.tasks.clearance import bounds
    from ruibench.tasks.instance import build

    size = 512
    from ruibench.tasks.instance import implemented

    for task, level, seed in [(t, lv, sd) for t in implemented()
                              for lv in ("easy", "hard") for sd in (0, 1, 2)]:
        scene = build(task, level, seed).scene
        cam = external_camera_pose(scene)
        fixed = next(c for c in scene.cameras if c.kind == "external")
        f = size * camera_focal_length(scene, fixed) / HORIZONTAL_APERTURE
        points = []
        for robot in scene.robots:
            for arm in robot.arms:
                base = arm_base(robot, arm, scene.table_height)
                start = arm_start_pose(scene, robot, arm).pos
                if not scene.camera_frame:
                    points += [(base[0], base[1], scene.table_height), base]
                # the gripper at its start pose: fingertips to the top of the hand (to the
                # hand's lower half in a task's own frame), both fingers.  A station's
                # head camera sits between the arms and the hands start 30 cm out to
                # either side, which is wider than anything else the picture holds: what
                # is asked there is that each hand comes into the picture from its side
                # (its inner half), not that the whole hand box is inside (9/21 -- a lens
                # wide enough for that drew the bench no bigger than v3 did)
                top = 0.07 if scene.camera_frame else 0.13
                inward = -1.0 if base[1] > 0 else 1.0
                # a head camera sees the hand's inner edge, so that is what is asked of it
                dys = (inward * 0.0725,) if model_of(robot).ego_camera else (-0.05, 0.05)
                points += [(start[0], start[1] + dy, start[2] + dz)
                           for dy in dys for dz in (-0.01, top)]
        for b in scene.bodies:
            (x0, x1), (y0, y1), (z0, z1) = bounds(b)
            points += [(x, y, z) for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)]
        # every mark, stroke by stroke: a mark the instruction names is in the picture
        points += [(x, y, scene.table_height) for m in scene.markers
                   for seg in marker_strokes_world(m) for x, y in seg]
        for p in points:
            x, y, z = rotate(quat_conj(cam.quat), tuple(a - e for a, e in zip(p, cam.pos)))
            u, v = size / 2 + f * x / z, size / 2 + f * y / z
            assert z > 0 and 20 <= u <= size - 20 and 20 <= v <= size - 20, (task, level, p, u, v)


def test_camera_orientation_matches_isaac_labs_convention():
    """``camera_quat`` reproduces the stacking task's hand-written quaternion
    from its position and where it looks -- the "ros" optical convention."""
    import math

    from ruibench.core.geometry import camera_quat

    eye = (1.0, 0.0, 0.4)
    target = (1.0 - math.cos(math.radians(30)), 0.0, 0.4 - math.sin(math.radians(30)))
    q = camera_quat(eye, target)
    expected = (0.35355, -0.61237, -0.61237, 0.35355)
    assert abs(abs(sum(a * b for a, b in zip(q, expected))) - 1.0) < 1e-4


def test_wk1_right_peg_is_on_the_right_of_the_fixed_camera_image():
    """WK1's goal is "the right peg": right as the fixed camera shows it, which
    is mirrored from the robot's own right."""
    from ruibench.engines.base import external_camera_pose, project
    from ruibench.tasks.impl.wk1_hanoi import GOAL, PEGS, peg_xy
    from ruibench.tasks.instance import build

    for level in ("easy", "hard"):
        inst = build("WK1", level, 0)
        cam = external_camera_pose(inst.scene)
        stand = inst.metadata["stand"]
        columns = [project(cam, (*peg_xy(stand, k), 0.85), 512, 512)[0] for k in range(3)]
        assert columns == sorted(columns)
        assert PEGS[GOAL] == "right" and columns[GOAL] == max(columns)
        assert inst.metadata["goal"] == "right"
