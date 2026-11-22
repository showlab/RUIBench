"""Hardware checks of the Isaac backend -- Gate A, the parts that need a GPU.

Each check runs in its **own process** and prints one ``RESULT {...}`` line:

    python -m ruibench.engines.isaac.checks {ik|dual_arm|render|table|perf|geometry}
    python -m ruibench.engines.isaac.checks snapshot TASK LEVEL SEED

One process per check is not a style choice.  Three things learned the hard way
force it: a simulation app can be created once per process; tearing down a stage
that has cameras attached hung for over fifteen minutes; and a native crash in
one check must not take the others with it.  ``ruibench doctor`` calls these
through ``subprocess`` and records whatever they print.
"""

from __future__ import annotations

import json
import os
import sys
import time

T0 = time.time()

#: where checks save the images they render (generated output, not tracked)
CHECK_DIR = "runs/checks"


def say(msg: str) -> None:
    print(f"[{time.time() - T0:7.1f}s] {msg}", flush=True)


def _fmt(v) -> str:
    return "(" + ", ".join(f"{x:+.3f}" for x in v) + ")"


def _single_arm(name: str, bodies=(), cameras=()):
    """The single-arm tasks' robot: the FR3 at the station table's back edge
    (v13: these checks built a Panda, which no task has used since v2 -- a
    machine could pass them and still build no task)."""
    from ..base import SceneSpec
    from ...tasks.impl._common import FR3_ONE, TABLE

    return SceneSpec(name=name, bodies=tuple(bodies), cameras=tuple(cameras), table_height=TABLE,
                     robots=(FR3_ONE,))


# --------------------------------------------------------------------- checks
def check_ik() -> dict:
    """The shared controller reaches the poses tasks actually use.

    It is the premise of the whole design -- change the interface, not the
    controller -- so it is checked before anything else is measured.
    """
    from ...core.types import GRASP_DOWN, Pose
    from .. import make_engine

    from ...tasks.impl._common import TABLE, ahead

    engine = make_engine("isaac")
    engine.load(_single_arm("check_ik"), seed=0)
    arm = "robot0/right"
    # where the single-arm tasks put things: from the station's arm line
    targets = tuple((label, ahead(dx, dy) + (TABLE + dz,)) for label, dx, dy, dz in (
        ("centre", 0.40, 0.00, 0.15), ("far-left", 0.55, 0.15, 0.10),
        ("near-right", 0.25, -0.20, 0.08), ("low", 0.35, 0.05, 0.04),
        ("low-far", 0.55, -0.10, 0.03)))
    rows = []
    for label, pos in targets:
        engine.reset()
        goal = Pose(pos, GRASP_DOWN)
        engine.set_arm_target(arm, goal)
        engine.advance(4.0)
        end = engine.state().arm(arm).eef_pose
        err, rot = end.distance_to(goal), end.angle_to(goal) * 57.2958
        rows.append({"target": label, "pos_err_mm": round(err * 1000, 1),
                     "rot_err_deg": round(rot, 1)})
        say(f"IK {label:10s} -> {_fmt(end.pos)}  {err * 1000:5.1f} mm  {rot:4.1f} deg")
    worst = max(r["pos_err_mm"] for r in rows)
    return {"pass": worst < 10.0, "worst_pos_err_mm": worst, "targets": rows}


def check_dual_arm() -> dict:
    """A2: two arms, two goals, commanded in the same step -- both move at once.

    Concurrency is judged at half-time: a serialising controller would have one
    arm still parked when the other is half way.
    """
    from ...core.types import GRASP_DOWN, Pose
    from ..base import CameraSpec, SceneSpec
    from .. import make_engine
    from ...tasks.impl._common import LEFT, RIGHT, TABLE, TWO_ARMS, ahead

    # the two-arm tasks' station: the ARX X5 pair, from where they fold at rest
    scene = SceneSpec(
        name="check_dual_arm", table_height=TABLE, robots=(TWO_ARMS,),
        cameras=(CameraSpec("agentview", "external", width=128, height=128),))
    engine = make_engine("isaac")
    engine.load(scene, seed=0)
    a0, a1 = RIGHT, LEFT
    s0 = engine.state()
    g0 = Pose(ahead(0.30, 0.08, arm="right") + (TABLE + 0.10,), GRASP_DOWN)
    g1 = Pose(ahead(0.30, -0.08, arm="left") + (TABLE + 0.10,), GRASP_DOWN)
    engine.set_arm_target(a0, g0)
    engine.set_arm_target(a1, g1)
    engine.advance(1.0)
    half = engine.state()
    engine.advance(3.0)
    end = engine.state()
    moved0 = half.arm(a0).eef_pose.distance_to(s0.arm(a0).eef_pose)
    moved1 = half.arm(a1).eef_pose.distance_to(s0.arm(a1).eef_pose)
    err0 = end.arm(a0).eef_pose.distance_to(g0)
    err1 = end.arm(a1).eef_pose.distance_to(g1)
    say(f"A2 after 1 s: arm0 moved {moved0 * 100:.1f} cm, arm1 moved {moved1 * 100:.1f} cm")
    say(f"A2 after 4 s: arm0 error {err0 * 1000:.1f} mm, arm1 error {err1 * 1000:.1f} mm")
    return {"pass": moved0 > 0.05 and moved1 > 0.05 and err0 < 0.01 and err1 < 0.01,
            "moved_at_1s_cm": [round(moved0 * 100, 1), round(moved1 * 100, 1)],
            "final_err_mm": [round(err0 * 1000, 1), round(err1 * 1000, 1)]}


def check_render(size: int = 512, steps: int = 60, save_dir: str = CHECK_DIR) -> dict:
    """A3: what rendering what the model will look at costs.

    Measured with both cameras rendered on every control step -- an upper bound:
    a real episode renders only when the model is asked something.
    """
    from ...core.types import Pose
    from ..base import BodySpec, CameraSpec
    from .. import make_engine

    scene = _single_arm(
        "check_render",
        bodies=(BodySpec("probe_cube", "box", (0.02, 0.02, 0.02), Pose((0.10, 0.0, 0.82)),
                         color="red"),),
        cameras=(CameraSpec("agentview", "external", width=size, height=size),
                 CameraSpec("wrist", "wrist", attached_to="robot0/right",
                            width=size, height=size)))
    engine = make_engine("isaac", render_every_step=True, track_visibility=True)
    engine.load(scene, seed=0)
    engine.advance(0.5)
    visible = engine.visible_objects("agentview")
    saved = _save_frames(engine, ("agentview", "wrist"), save_dir)
    t = time.time()
    engine.advance(steps / engine.control_hz)
    dt = (time.time() - t) / steps
    say(f"A3 {dt * 1000:.0f} ms per control step with two {size}x{size} cameras rendered "
        f"each step; agentview sees {visible}")
    return {"pass": "probe_cube" in visible,
            "ms_per_control_step": round(dt * 1000, 1),
            "realtime_factor": round(1.0 / (dt * engine.control_hz), 3),
            "visible_from_agentview": list(visible), "frames": saved}


def check_table() -> dict:
    """Where the work surface really is: drop two cubes and see where they stop."""
    from ...core.types import Pose
    from ..base import BodySpec
    from .. import make_engine

    scene = _single_arm("check_table", bodies=(
        BodySpec("drop_a", "box", (0.02, 0.02, 0.02), Pose((0.15, 0.00, 1.0)), color="red"),
        BodySpec("drop_b", "box", (0.02, 0.02, 0.02), Pose((0.30, -0.25, 1.0)), color="blue")))
    engine = make_engine("isaac")
    engine.load(scene, seed=0)
    engine.advance(1.5)
    tops = [round(engine.state().obj(n).pose.pos[2] - 0.02, 4) for n in ("drop_a", "drop_b")]
    say(f"TABLE surface measured at {tops} (design value 0.800)")
    return {"pass": all(abs(v - 0.80) < 0.002 for v in tops), "surface_z": tops}


def check_perf(n: int = 240) -> dict:
    """Milliseconds per physics step, by component."""
    from ...core.types import GRASP_DOWN, Pose
    from .. import make_engine

    engine = make_engine("isaac")
    engine.load(_single_arm("check_perf"), seed=0)
    engine.set_arm_target("robot0/right", Pose((0.15, 0.05, 0.95), GRASP_DOWN))
    dt = 1.0 / engine.sim_hz
    t = time.time()
    engine.advance(n * dt)
    ms = (time.time() - t) / n * 1000
    say(f"PERF {ms:.2f} ms per physics step ({1000 / (ms * engine.sim_hz):.1f}x realtime)")
    return {"pass": True, "ms_per_physics_step": round(ms, 3),
            "realtime_factor": round(1000 / (ms * engine.sim_hz), 2)}


def check_asset_geometry() -> dict:
    """Measure every catalogue mesh: which way is up, its size, its origin.

    Predicates like On() and In() use ``BodySpec.size`` as half-extents and the
    body pose as its centre, so for a mesh asset both have to match the real
    mesh.  Measured, not assumed -- and measured **before any physics step**:
    Isaac 4.5 writes simulated poses to Fabric, not USD, so a USD bounding box
    taken after stepping describes where the object was spawned, not where it
    is now.

    Each asset is measured as authored and rotated +90 degrees about x; the
    catalogue's ``up`` field says which of the two is standing.  Results go to
    ``assets.lock.json``, never to the hand-written catalogue.
    """
    import omni.usd
    from pxr import Usd, UsdGeom

    from ...core.types import Pose
    from ..base import BodySpec
    from .. import make_engine
    from .assets import load_catalogue
    from .scene_builder import body_key

    catalogue = load_catalogue()
    names = [n for n, e in catalogue.items()
             if e.category not in ("table", "environment", "cabinet") and n == e.name]
    rot_x90 = (0.70710678, 0.70710678, 0.0, 0.0)
    spawn = {}
    bodies = []
    for i, n in enumerate(names):
        for k, (label, quat) in enumerate((("as_authored", (1.0, 0.0, 0.0, 0.0)),
                                           ("x+90", rot_x90))):
            pos = (-0.9 + 0.3 * i, 2.0 + 0.6 * k, 1.5)       # far from the robot, in the air
            spawn[(n, label)] = pos
            bodies.append(BodySpec(f"{n}__{label.replace('+', 'p')}", pose=Pose(pos, quat),
                                   asset=n, mass=0.1, fixed=True,
                                   attrs={"raw_orientation": True}))
    engine = make_engine("isaac")
    engine.load(_single_arm("check_geometry", bodies=tuple(bodies)), seed=0)
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])

    out = {}
    for n in names:
        rows = {}
        for label in ("as_authored", "x+90"):
            key = f"{n}__{label.replace('+', 'p')}"
            box = cache.ComputeWorldBound(
                stage.GetPrimAtPath(f"/World/envs/env_0/{body_key(key)}")).ComputeAlignedRange()
            lo, hi = box.GetMin(), box.GetMax()
            pos = spawn[(n, label)]
            rows[label] = {
                "half_extents": [round((hi[j] - lo[j]) / 2, 4) for j in range(3)],
                "centre_offset": [round((hi[j] + lo[j]) / 2 - pos[j], 4) for j in range(3)],
            }
        # which way is up is a property of the asset (see assets.yaml `up:`),
        # not something to infer: "lowest orientation" would lay a can down
        entry = catalogue[n]
        upright = "x+90" if entry.up == "y" else "as_authored"
        chosen = rows[upright]
        footprint = sorted(chosen["half_extents"][:2])
        out[n] = {**rows, "upright": upright,
                  "narrow_width_m": round(2 * footprint[0], 4),
                  "graspable": 2 * footprint[0] <= 0.075}
        say(f"GEOM {n:12s} up={entry.up} half(upright)={chosen['half_extents']} "
            f"offset={chosen['centre_offset']} narrow side {2 * footprint[0] * 100:4.1f} cm "
            f"{'graspable' if out[n]['graspable'] else '-- too wide for an 8 cm gripper'}")
    from .assets import record_geometry
    record_geometry(out)
    return {"pass": True, "assets": out}


def check_snapshot(task: str = "CL3", level: str = "hard", seed: int = 0,
                   save_dir: str = CHECK_DIR, tag: str = "") -> dict:
    """Build a task's scene, let it settle, and show what is actually there.

    Reports where every body came to rest (a body resting higher than its
    design height is sitting on something) and every collider's world bounds,
    and saves an external and a top-down image.  The tool to reach for when a
    motion is blocked by something nobody can see.
    """
    import omni.usd
    from pxr import Usd, UsdGeom, UsdPhysics

    from ...core.types import Pose
    from ...tasks.instance import build
    from ..base import CameraSpec, SceneSpec
    from .. import make_engine

    inst = build(task, level, int(seed))
    top = CameraSpec("topdown", "external", width=512, height=512,
                     pose=Pose((0.15, 0.0, 2.2), (0.0, 0.70710678, 0.70710678, 0.0)))
    scene = SceneSpec(**{**inst.scene.__dict__, "cameras": inst.scene.cameras + (top,)})
    engine = make_engine("isaac", render_every_step=False)
    engine.load(scene, seed=int(seed))
    engine.advance(1.0)
    st = engine.state()
    seen = {cam.name: engine.pixel_counts(cam.name) for cam in scene.cameras}
    for cam, counts in seen.items():
        say(f"SEEN {cam:10s} " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            + f"  -> visible {list(engine.visible_objects(cam))}"
            + f"  from {_fmt(engine.camera_position(cam))}")
    rest = {}
    for body in scene.bodies:
        o = st.obj(body.name)
        rest[body.name] = {"design_z": round(body.pose.pos[2], 4),
                           "rest": [round(v, 4) for v in o.pose.pos]}
        drift = o.pose.pos[2] - body.pose.pos[2]
        say(f"SNAP {body.name:14s} design z {body.pose.pos[2]:.3f} rests at "
            f"{_fmt(o.pose.pos)}  dz {drift * 1000:+6.1f} mm")
    stage = omni.usd.get_context().get_stage()
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render", "guide", "proxy"])
    colliders = {}
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI) and "/envs/env_0/obj_" in str(prim.GetPath()):
            box = cache.ComputeWorldBound(prim).ComputeAlignedRange()
            lo, hi = box.GetMin(), box.GetMax()
            colliders[str(prim.GetPath())] = [[round(v, 3) for v in lo], [round(v, 3) for v in hi]]
    for path, (lo, hi) in sorted(colliders.items()):
        say(f"COLL {path.split('env_0/')[-1][:60]:60s} {lo} .. {hi}")
    saved = _save_frames(engine, ("agentview", "wrist", "topdown"), save_dir, tag)
    return {"pass": True, "rest": rest, "seen_pixels": seen, "colliders": colliders,
            "frames": saved}


def _save_frames(engine, cameras, save_dir: str, tag: str = "") -> list:
    try:
        import numpy as np
        from PIL import Image

        os.makedirs(save_dir, exist_ok=True)
        out = []
        for cam in cameras:
            img = engine.render(cam).image
            img = img.detach().cpu().numpy() if hasattr(img, "detach") else img
            path = os.path.join(save_dir, f"check_{tag + '_' if tag else ''}{cam}.png")
            Image.fromarray(np.asarray(img).astype("uint8")).save(path)
            out.append(path)
        return out
    except Exception as exc:                      # pragma: no cover - diagnostics only
        say(f"could not save frames: {exc}")
        return []


CHECKS = {"ik": check_ik, "dual_arm": check_dual_arm, "render": check_render,
          "table": check_table, "perf": check_perf, "geometry": check_asset_geometry,
          "snapshot": check_snapshot}


def main(name: str) -> None:
    from . import app

    app.launch(headless=True)
    say(f"{name}: app up")
    result = CHECKS[name](*sys.argv[2:])
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    try:
        main(sys.argv[1])
    except Exception:
        import traceback
        print("RESULT " + json.dumps({"pass": False, "error": traceback.format_exc()[-800:]}),
              flush=True)
    finally:
        sys.stdout.flush()
        os._exit(0)                               # never tear an Isaac stage down in-process
