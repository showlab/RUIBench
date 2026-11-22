"""A dependency-free kinematic engine.

**It is not a physics simulator and never produces benchmark numbers.**  What
it is for:

* the whole harness -- clock, events, executor, scoring, reports, CLI -- runs
  and is unit-tested without a simulator, on any machine, in milliseconds;
* the negative controls and the scorer acceptance trajectories (do-nothing,
  first-frame-only, ignore-feedback, ignore-change) need a world that is
  cheap to drive to an exact state;
* a new interface implementation can be smoke-tested before anyone has a GPU.

Motion is first-order: the end effector slews toward its target at a constant
speed, grasping attaches a body to the gripper, releasing drops it onto the
nearest support.  Runs marked ``engine="mock"`` are refused by the reporting
layer's leaderboard path.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from ..core.rng import SeedBundle
from ..core.types import (
    ArmState, CameraFrame, ObjectState, Pose, RobotState, Vec3, WorldState,
)
from ..core.geometry import compose, quat_conj, quat_mul, relative_pose, yaw_of
from .base import (
    ARRIVAL_POS_TOL, ARRIVAL_ROT_TOL, GRASP_MARGIN, GRIP_CLOSING, ArticulationSpec,
    BodySpec, Engine,
    EngineCapabilities, SceneSpec, arm_camera_model, arm_start_pose,
    attachments, collides, container_floor, external_camera_pose, joint_program, joint_start,
    drawn, in_grasp, legibility_reach, legible_from, passes, wrist_camera_pose,
)
from .cloth import FlatCloth, is_cloth
from .kinematics import arm_frame, arm_kinematics, forward, inverse
from .robots import model_of

EEF_SPEED = 0.25          # m/s
#: m/s^2, for what the mock throws (``launch``)
GRAVITY = 9.81
GRIPPER_SPEED = 4.0       # units/s (0 open -> 1 closed in 0.25 s)
GRASP_CLOSED = GRIP_CLOSING   # gripper value from which a grasp is attempted
JOINT_SPEED = 0.25        # m/s for driven slides
HINGE_SPEED = 2.0         # rad/s for driven hinges
PART_OVERLAP = 0.001      # m: a body rests on a part it overlaps by more than this
RAISED_PART = 0.002       # m: a part's underside this far above the body's own is a raised one
#: a closing hand takes hold only where its fingertips reach into the body, at
#: least this far below its top (v1.2): the box test alone (``base.in_grasp``,
#: give or take ``GRASP_MARGIN``) let a hand whose fingertips ended 5 mm above
#: a 1 cm coin hold it, and a pick that closes on air on Isaac worked here
FINGERTIP_OVERLAP = 0.002
#: opt-in: with this set to 1 in the environment, a hand pointing down stops
#: where its parts would newly meet a body, or its fingertips the table top, as
#: it does on Isaac; the executor then calls the move blocked.  Off (the
#: default), the hand follows the commanded line through whatever is in the
#: way: a program practised that way skimmed a balance pan's rim and stopped
#: short there on Isaac (RUI-Zero I-013, 10/2).  Not obstacles: what the hand
#: holds, and what it is meant to press (``pressable``, ``lever``).
HAND_COLLIDES_ENV = "RUIBENCH_MOCK_HAND_COLLIDES"
#: what the switch models, for a run's record: what collides has grown, and
#: the same switch value meant less before (10/2: the hand only)
HAND_COLLIDES_MODEL = "2026-10-03: the hand (the FR3's own collision shapes) and what it holds"
#: where the switch is not to be trusted: the task-levels whose own scripted
#: reference -- which does the task on Isaac -- the mock stops with it on,
#: because of what the mock still cannot do: it pushes nothing aside, lets
#: nothing fall into a narrow mouth, and keeps a jointed body rigid.  Practise
#: these with the switch off.  (task, level) -> (seeds stopped, seeds run, why),
#: from the reference on every dealt seed -- dev, test and spare, 40 a
#: task-level, 1,920 episodes -- under this ``HAND_COLLIDES_MODEL``; the
#: other 44 task-levels pass on all 40, and all of these pass with it off.
HAND_COLLIDES_UNSOUND = {
    ("CL4", "hard"): (40, 40, "a pebble let go over the jar rests on its mouth; on Isaac it "
                              "falls in"),
    ("DS4", "hard"): (27, 40, "a balance pan is rigid; on Isaac it gives under the fingertips"),
    ("DS5", "easy"): (40, 40, "the palm's end is stopped on the glass rod; on Isaac it nudges "
                              "the rod aside in its block"),
    ("DS5", "hard"): (40, 40, "the palm's end is stopped on the glass rod; on Isaac it nudges "
                              "the rod aside in its block"),
}
#: the fingertips may touch the table top, not go into it
TABLE_SLACK = 0.0005
#: What Isaac collides the hand with, where it is known: the FR3's collision
#: shapes as Isaac 4.5's ``Robots/Franka/FR3/fr3.usd`` authors them (read
#: with usd-core, 10/2) -- far slimmer than the robot model's envelope, which
#: is for keeping clear (``tasks/clearance``).  Each finger is four boxes:
#: (half width across the fingers' line, its reach out from the finger's inner
#: face, its span above the tool centre point); the tip is 1.5 cm thick, not
#: the envelope's 2.6.  The hand's body is narrow at its bottom (x +-1.8 cm)
#: and full width from 7 cm up (its ends: +-9.75 cm up to 4.5 cm, then +-10).  With the envelope, the references stopped on
#: a pan's rim (DS4), a bowl's wall (CL4) and a glass rod (DS5) they clear on
#: Isaac.  The wrist camera's housing and bracket come from the robot model.
HAND_COLLIDERS = {
    "fr3": {
        "finger": ((0.0088, (0.0000, 0.0152), (-0.0089, 0.0096)),
                   (0.0088, (0.0070, 0.0248), (0.0053, 0.0292)),
                   (0.0110, (0.0110, 0.0260), (0.0246, 0.0446)),
                   (0.0110, (0.0024, 0.0112), (0.0415, 0.0453))),
        "hand": (((-0.0173, 0.0173), (-0.0975, 0.0975), (0.038, 0.045)),
                 ((-0.019, 0.019), (-0.1004, 0.1004), (0.045, 0.070)),
                 ((-0.0316, 0.0316), (-0.104, 0.104), (0.070, 0.130))),
    },
}
#: with visibility tracked, a body is seen when this much of its camera-facing
#: surface is in sight (Isaac: 16 pixels of it)
MIN_IN_SIGHT = 0.1


def _lerp_towards(cur: Vec3, goal: Vec3, max_step: float) -> Vec3:
    d = [g - c for c, g in zip(cur, goal)]
    dist = math.sqrt(sum(x * x for x in d))
    if dist <= max_step or dist == 0.0:
        return tuple(goal)
    s = max_step / dist
    return tuple(c + x * s for c, x in zip(cur, d))


@dataclass
class _Arm:
    name: str
    pose: Pose
    target: Pose
    gripper: float = 0.0
    gripper_target: float = 0.0
    holding: Optional[str] = None
    home: Pose = field(default_factory=Pose)
    #: the fingers closed on nothing: they take nothing until they open again
    closed_empty: bool = False
    #: the robot model, and where the arm's base stands (``engines.kinematics``)
    model: object = None
    base: Optional[Pose] = None
    #: joints known exactly at the pose they were for (the start, a joint target reached)
    joints: Optional[Tuple[float, ...]] = None
    joints_at: Optional[Pose] = None
    #: the joint target it is moving to
    joint_goal: Optional[Tuple[float, ...]] = None


@dataclass
class _Body:
    spec: BodySpec
    pose: Pose
    lin_vel: Vec3 = (0.0, 0.0, 0.0)
    carrier: Optional[str] = None      # articulation joint carrying this body
    #: arms holding it, in the order they took hold, each with where the body
    #: sits in that hand's frame -- two during a handover
    grips: Dict[str, Pose] = field(default_factory=dict)
    #: its velocity while thrown (``launch``); None on the ground
    flight: Optional[Vec3] = None

    @property
    def held_by(self) -> Optional[str]:
        return next(iter(self.grips), None)


@dataclass
class _Joint:
    spec: ArticulationSpec
    value: float
    target: float
    origin: Pose
    #: the axis in the world now (a mounted joint's turns with what it stands on)
    axis: Vec3 = (1.0, 0.0, 0.0)
    #: a mounted joint's zero pose and axis in its mount's frame
    mount_rel: Optional[Pose] = None
    axis_rel: Optional[Vec3] = None
    #: where the carried body was after the last tick: what rides it moves by the difference
    last: Optional[Vec3] = None
    #: the joint value then: what stands on a turntable turns by the difference
    last_value: Optional[float] = None


class MockEngine(Engine):
    capabilities = EngineCapabilities(
        name="mock", concurrent_arms=True, springs=True, actuated_joints=True,
        rendering=False, mesh_assets=False, max_camera_size=512,
        photoreal=False, results_grade=False, cloth=True,
    )

    def __init__(self) -> None:
        #: line-of-sight occlusion costs ray tests, so -- as on Isaac -- only
        #: tasks that ask for it get ``visible_from`` worked out in every
        #: ``state()``; for the rest it is None.  ``visible_objects`` always
        #: tests sight lines
        self.track_visibility = False
        self._sight_cache: Dict[tuple, Tuple[str, ...]] = {}
        self._scene: Optional[SceneSpec] = None
        self._t = 0.0
        self._arms: Dict[str, _Arm] = {}
        self._bodies: Dict[str, _Body] = {}
        #: the cloths, folded flat for want of physics (``cloth.FlatCloth``);
        #: each also has a body, the box round it, for everything that asks
        #: where a body is
        self._cloths: Dict[str, FlatCloth] = {}
        self._joints: Dict[str, _Joint] = {}
        self._instruction = ""
        self._instruction_version = 0
        self._events_fired: List[str] = []
        self._seed = 0
        self._attached: Dict[str, Tuple[str, Pose]] = {}
        self.hand_collides = os.environ.get(HAND_COLLIDES_ENV, "") == "1"
        #: each body's solid pieces where it was last looked at (hand_collides)
        self._pieces: Dict[str, tuple] = {}

    # -- lifecycle ---------------------------------------------------------
    def load(self, scene: SceneSpec, seed: int) -> None:
        # Optionally behave like the Isaac backend, which keeps one stage per
        # process: a structural change must go to a fresh worker.  Used to test
        # the isolated runner's rebuild path without a GPU.
        import os
        if (os.environ.get("RUIBENCH_MOCK_STRICT_STRUCTURE") and self._scene is not None
                and scene.structure_key() != self._scene.structure_key()):
            from ..core.errors import SceneRebuildRequired
            raise SceneRebuildRequired(f"{self._scene.name} -> {scene.name}")
        self._scene = scene
        self._seed = seed
        self.reset()

    def reset(self) -> WorldState:
        assert self._scene is not None, "call load() first"
        sc, seeds = self._scene, SeedBundle(self._seed)
        self._t = 0.0
        self._events_fired = []
        self._instruction, self._instruction_version = "", 0

        self._arms = {}
        for r in sc.robots:
            model = model_of(r)
            kin = arm_kinematics(model.name)
            for a in r.arms:
                home = arm_start_pose(sc, r, a)
                arm = self._arms[f"{r.name}/{a}"] = _Arm(f"{r.name}/{a}", home, home, home=home,
                                                         model=model)
                if kin is not None:
                    arm.base = arm_frame(r, a, sc.table_height)
                    if sc.arm_start(f"{r.name}/{a}") is None:    # the ready posture
                        arm.joints, arm.joints_at = kin.ready, home

        self._bodies = {}
        for b in sc.bodies:
            pose = b.pose
            # a free body starts a few millimetres off, by seed -- unless it
            # starts fitted into something (CR2's bulb, its cap in the collar's
            # hole with a millimetre of room)
            if not b.fixed and not b.attrs.get("starts_fitted"):
                jitter = seeds.jitter(f"layout/{b.name}", pose.pos[:2], 0.004)
                pose = Pose((jitter[0], jitter[1], pose.pos[2]), pose.quat)
            self._bodies[b.name] = _Body(spec=b, pose=pose)
        self._cloths = {b.name: FlatCloth(replace(b, pose=self._bodies[b.name].pose))
                        for b in sc.bodies if is_cloth(b)}
        spec = {b.name: b for b in sc.bodies}
        self._knocked: set = set()                 # bodies a thrown one has knocked over
        self._attached = {child: (parent, relative_pose(spec[parent].pose, spec[child].pose))
                          for child, parent in attachments(sc.bodies).items()}
        self._carry_attached()

        self._joints = {}
        for j in sc.articulations:
            origin = self._bodies[j.parent].pose if j.parent in self._bodies else Pose()
            start = joint_start(j)
            self._joints[j.name] = joint = _Joint(spec=j, value=start, target=start, origin=origin,
                                                  axis=tuple(j.axis))
            if j.mounted_on:
                from ..core.geometry import rotate

                mount = self._bodies[j.mounted_on].pose
                joint.mount_rel = relative_pose(mount, origin)
                joint.axis_rel = rotate(quat_conj(mount.quat), tuple(j.axis))
            body = self._bodies.get(j.parent)
            if body is not None and start != 0.0:
                # the body's spec pose is the joint's zero: put it where it starts
                body.pose = _joint_pose(joint, body)
                joint.last = body.pose.pos
        self._carry_attached()
        for start in sc.arm_starts:
            # the hand closed on what it starts with (``ArmStart``)
            arm = self._arms[start.arm]
            arm.gripper = arm.gripper_target = 1.0
            arm.closed_empty = not start.holding
            if start.holding:
                body = self._bodies[start.holding]
                body.pose = spec[start.holding].pose            # no jitter: it is in the hand
                arm.holding = start.holding
                body.grips[arm.name] = relative_pose(arm.pose, body.pose)
        return self.state()

    def close(self) -> None:
        self._scene = None

    # -- time --------------------------------------------------------------
    @property
    def sim_time(self) -> float:
        return self._t

    def advance(self, seconds: float) -> None:
        if seconds <= 0:
            return
        dt = 0.02
        steps, rest = divmod(seconds + 1e-12, dt)
        for _ in range(int(steps)):
            self._tick(dt)
        if rest > 1e-9:
            self._tick(rest)

    def _tick(self, dt: float) -> None:
        self._t += dt
        for arm in self._arms.values():
            prev = arm.pose
            arm.pose = Pose(_lerp_towards(arm.pose.pos, arm.target.pos, EEF_SPEED * dt),
                            arm.target.quat)
            if arm.joint_goal is not None and arm.pose == arm.target:
                arm.joints, arm.joints_at, arm.joint_goal = arm.joint_goal, arm.pose, None
            # only a step that moves the hand into something stops it: the mock
            # turns the hand at once, and a turn alone near a neighbour is not
            # what stops a hand on Isaac, where it turns on the way
            if self.hand_collides and arm.pose.pos != prev.pos and (
                    self._hand_meets(arm, arm.pose)
                    - self._hand_meets(arm, Pose(prev.pos, arm.pose.quat))):
                arm.pose = Pose(prev.pos, arm.pose.quat)   # stopped where it meets something
            step = tuple((n - p) / dt for n, p in zip(arm.pose.pos, prev.pos))
            arm.gripper = _towards_scalar(arm.gripper, arm.gripper_target, GRIPPER_SPEED * dt)
            self._update_grasp(arm, step)
        for name, cloth in self._cloths.items():
            self._bodies[name].pose = cloth.state().pose
        self._tick_joints(dt)
        self._fly(dt)
        self._settle()
        self._carry_attached()

    def _carry_attached(self) -> None:
        """Visual-only marks keep their place on the body they are painted on."""
        for child, (parent, offset) in self._attached.items():
            self._bodies[child].pose = compose(self._bodies[parent].pose, offset)

    def _roofed(self, point: Vec3) -> bool:
        """Is there something fixed right above this point?  A hand coming down
        from above could not have got there -- the mock has no collisions, so
        this is what stops it reaching into a shut cabinet."""
        from .sight import box_bounds

        from .parts import body_parts, footprints

        carried = {j.spec.parent for j in self._joints.values()}
        for name, b in self._bodies.items():
            # something fixed, or a lid on its hinge (which may be tipped up)
            lid = name in carried and b.spec.attrs.get("falls")
            if not (b.spec.fixed or lid) or not collides(b.spec):
                continue
            if lid:
                shadows = footprints(body_parts(b.spec.shape, b.spec.size, b.spec.attrs),
                                     tuple(b.pose.pos), tuple(b.pose.quat))
                if any(bottom > point[2] and _in_hull(point, hull) for hull, bottom, _ in shadows):
                    return True
                continue
            (x0, x1), (y0, y1), (z0, _) = box_bounds(b.spec.shape, b.spec.size, b.pose)
            if x0 <= point[0] <= x1 and y0 <= point[1] <= y1 and z0 > point[2]:
                return True
        return False

    def _pushed_to(self, j: _Joint, body: _Body) -> Optional[float]:
        """How far down a closed hand's fingertips hold a pressable body: the
        joint value that puts its top at the lowest fingertip over it."""
        from .base import arm_camera_model

        if j.spec.kind == "hinge":
            return self._lever_pushed_to(j, body)
        best = None
        for arm in self._arms.values():
            if arm.gripper < GRASP_CLOSED:
                continue
            x, y, z = arm.pose.pos
            bx, by, _ = body.pose.pos
            if abs(x - bx) > body.spec.size[0] or abs(y - by) > body.spec.size[1]:
                continue
            fingers = [p for p in arm_camera_model(self._scene, arm.name).closed_hand
                       if p.name == "finger"]
            tip = z + min((p.bottom for p in fingers), default=0.0)
            value = (tip - body.spec.size[2] - j.origin.pos[2]) / (j.axis[2] or 1.0)
            best = value if best is None else min(best, value)
        return best

    def _lever_pushed_to(self, j: _Joint, body: _Body) -> Optional[float]:
        """A hinge worked by a lever (``attrs["lever"]``: its span across the
        hinge, its reach from it, its top above the hinge -- all at rest, in
        the body's frame): a fingertip pushing the lever's top down turns the
        hinge.  The hinge is assumed to turn about +y, the lever to point -x."""
        from ..core.geometry import rotate
        from .base import arm_camera_model

        (y0, y1), (r0, r1), top = body.spec.attrs["lever"]
        pivot = _hinge_pivot(j)
        inv = quat_conj(j.origin.quat)
        best = None
        for arm in self._arms.values():
            if arm.gripper < GRASP_CLOSED:
                continue
            fingers = [p for p in arm_camera_model(self._scene, arm.name).closed_hand
                       if p.name == "finger"]
            tip = arm.pose.pos[:2] + (arm.pose.pos[2] + min((p.bottom for p in fingers),
                                                            default=0.0),)
            lx, ly, lz = rotate(inv, tuple(t - p for t, p in zip(tip, pivot)))
            reach = math.hypot(lx, lz)
            if not (y0 <= ly - j.spec.pivot[1] <= y1 and r0 <= reach <= r1 and lx < 0):
                continue
            # the lever's top passes through the fingertip at this angle
            below = math.atan2(-lz, -lx) + math.asin(min(1.0, top / reach))
            value = -below
            best = value if best is None else min(best, value)
        return best

    def _opening(self, arm: _Arm) -> float:
        """How far apart the fingers are: across what they hold (the fingers
        travel along the tool's y), else as far as the command opens them --
        closed on nothing, not at all."""
        from ..core.geometry import rotate

        widest = arm_camera_model(self._scene, arm.name).max_opening
        if arm.holding is not None:
            b = self._bodies[arm.holding]
            across = rotate(arm.pose.quat, (0.0, 1.0, 0.0))
            half = b.spec.size
            if footprint_of(b.spec) == "round":
                half = (half[0], half[0], half[2])
            width = 2.0 * sum(abs(sum(u * v for u, v in zip(across, rotate(b.pose.quat, e)))) * h
                              for e, h in zip(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), half))
            return round(min(widest, width), 4)
        if arm.gripper >= GRASP_CLOSED:
            return 0.0
        return round(widest * (1.0 - arm.gripper), 4)

    def _update_grasp(self, arm: _Arm, vel: Vec3) -> None:
        if arm.gripper >= GRASP_CLOSED and arm.holding is None and not arm.closed_empty:
            # a button is pressed, never picked up; nor is a link on a joint
            # (WK2's raft, tapped with a closed hand, v9): on Isaac it cannot be
            near = [] if self._roofed(arm.pose.pos) else [
                (arm.pose.distance_to(b.pose), n) for n, b in self._bodies.items()
                if not b.spec.fixed and collides(b.spec) and not b.spec.attrs.get("pressable")
                and not b.spec.attrs.get("carried_by") and not is_cloth(b.spec)
                and in_grasp(b.spec, b.pose, arm.pose.pos) and self._fingers_reach(arm, b)]
            # a cloth is taken by the patch under the fingertips, not as a box
            cloth = None if near else next(
                (n for n, c in self._cloths.items() if c.take(arm.name, arm.pose)), None)
            if near:
                _, name = min(near)
                arm.holding = name
                body = self._bodies[name]
                body.grips[arm.name] = relative_pose(arm.pose, body.pose)
                body.carrier = None
            elif cloth is not None:
                arm.holding = cloth
            else:
                # closed on nothing: a hand closed first and then pushed onto
                # a body does not take it (bench-issue #21: a doll pushed down
                # with closed fingers was "held" from then on)
                arm.closed_empty = True
        elif arm.gripper < GRASP_CLOSED:
            arm.closed_empty = False
            if arm.holding in self._cloths:
                self._cloths[arm.holding].let_go(arm.name)
                arm.holding = None
            elif arm.holding is not None:
                self._bodies[arm.holding].grips.pop(arm.name, None)
                arm.holding = None
        if arm.holding in self._cloths:
            self._cloths[arm.holding].carry(arm.name, arm.pose, vel)
        elif arm.holding:
            b = self._bodies[arm.holding]
            if b.held_by == arm.name:
                # it moves, and turns, with the hand that took it first
                b.pose = compose(arm.pose, b.grips[arm.name])
                b.lin_vel = vel
            else:
                # the second hand of a handover: it takes over from where it is
                b.grips[arm.name] = relative_pose(arm.pose, b.pose)

    def _tick_joints(self, dt: float) -> None:
        # mounted joints last: they stand on bodies the others carry
        for j in sorted(self._joints.values(), key=lambda j: j.mount_rel is not None):
            if j.mount_rel is not None:
                from ..core.geometry import rotate

                mount = self._bodies[j.spec.mounted_on].pose
                j.origin = compose(mount, j.mount_rel)
                j.axis = rotate(mount.quat, j.axis_rel)
            spec = j.spec
            body = self._bodies.get(spec.parent) if spec.parent else None
            lo, hi = spec.limits
            if body is not None and body.held_by is not None and not spec.actuated:
                # a hand moves a free joint, but only along (or about) its axis
                # and within its limits
                along = (_hinge_angle(j, body.pose) if spec.kind == "hinge" else
                         sum((p - o) * a for p, o, a in zip(body.pose.pos, j.origin.pos, j.axis)))
                j.value = min(max(along, lo), hi)
                if spec.stiffness <= 0:
                    j.target = j.value                   # a free joint stays where it is left
            else:
                # a spring pulls toward its target -- where it rests, unless
                # something (a reactor letting a pendulum go) moves it
                goal = joint_program(spec, self._t) if spec.oscillate else j.target
                if body is not None and body.spec.attrs.get("falls") and spec.kind == "hinge":
                    # a lid nothing holds: shut, or -- past upright -- wide open
                    goal = lo if j.value < -math.pi / 2 else 0.0
                speed = HINGE_SPEED if spec.kind == "hinge" else JOINT_SPEED
                j.value = min(max(_towards_scalar(j.value, goal, speed * dt), lo), hi)
                if body is not None and body.spec.attrs.get("pressable") and not spec.actuated:
                    # a closed hand pushing down on a button or a key
                    pushed = self._pushed_to(j, body)
                    if pushed is not None and pushed < j.value:
                        j.value = max(pushed, lo)
            if body is None:
                continue
            new = _joint_pose(j, body)
            delta = tuple(n - p for n, p in zip(new.pos, j.last or body.pose.pos))
            turned = j.value - (j.last_value if j.last_value is not None else j.value)
            j.last, j.last_value = new.pos, j.value
            body.lin_vel = tuple(d / dt for d in delta)
            body.pose = new
            if spec.carries and spec.kind == "hinge" and turned:
                self._turn_riders(j, body, turned, dt)
                continue
            # anything resting on or in the carried body rides along
            carries = spec.carries and spec.kind != "hinge"
            for other in (self._bodies.values() if carries else ()):
                if other is body or other.held_by is not None or other.spec.fixed:
                    continue
                # on it or in it: over it, and not lower than its underside (a
                # toy standing on the table beside a passing wagon stays put)
                bottom = other.pose.pos[2] - other.spec.size[2]
                if other.pose.planar_distance_to(body.pose) < max(body.spec.size[:2]) \
                        and -0.002 <= bottom - (body.pose.pos[2] - body.spec.size[2]) < 0.12:
                    other.pose = Pose(tuple(o + d for o, d in zip(other.pose.pos, delta)),
                                      other.pose.quat)
                    other.lin_vel = body.lin_vel

    def _turn_riders(self, j: "_Joint", body, angle: float, dt: float) -> None:
        """A turntable (a carrying hinge): what stands on it turns with it,
        about the hinge's axis -- position and heading both."""
        from ..core.geometry import rotate

        ax = j.axis
        turn = (math.cos(angle / 2),) + tuple(math.sin(angle / 2) * a for a in ax)
        pivot = _hinge_pivot(j)
        top = body.pose.pos[2] + body.spec.size[2]
        for other in self._bodies.values():
            if other is body or other.held_by is not None or other.spec.fixed:
                continue
            bottom = other.pose.pos[2] - other.spec.size[2]
            if other.pose.planar_distance_to(body.pose) < max(body.spec.size[:2]) \
                    and -0.004 <= bottom - top < 0.12:
                arm = rotate(turn, tuple(o - q for o, q in zip(other.pose.pos, pivot)))
                pos = tuple(q + a for q, a in zip(pivot, arm))
                other.lin_vel = tuple((n - o) / dt for n, o in zip(pos, other.pose.pos))
                other.pose = Pose(pos, quat_mul(turn, other.pose.quat))

    def _settle(self) -> None:
        """Released bodies drop straight down onto the nearest support.

        Resolved bottom-up, and a body may only rest on something **strictly
        below** it -- and not on anything round on top (``curved_top``): a
        fruit dropped on a fruit rolls off it in a physics engine.  Without that guard two objects sharing a footprint each
        raise the other every tick and climb away from the table.
        """
        table = self._scene.table_height if self._scene else 0.0
        order = sorted(self._bodies.items(), key=lambda kv: kv[1].pose.pos[2])
        containers = [o for o in self._bodies.values()
                      if str(o.spec.attrs.get("container", "")) == "open"]
        carried = {j.spec.parent for j in self._joints.values() if j.spec.parent}
        for name, b in order:
            if b.held_by is not None or b.spec.fixed or name in carried or b.flight is not None \
                    or name in self._cloths:
                continue
            # Inside an open container: rest on its floor -- or, dropped onto
            # something already in it, on top of that.  Isaac stacks three cubes
            # let go over the same spot into a tower whose top is above the rim,
            # and a mock that laid them side by side hid exactly that failure.
            # (dropping only goes down: a body under a container -- a ball under the
            # cabinet a basket stands on -- is not in it)
            home = next((c for c in containers if c is not b and _over_opening(b, c)
                         and b.pose.pos[2] > c.pose.pos[2] - c.spec.size[2]), None)
            if home is not None and not _fits_inside(b, home):
                # too big to go in -- a lid: it rests on the rim
                rim = home.pose.pos[2] + home.spec.size[2] + b.spec.size[2]
                b.pose = Pose((b.pose.pos[0], b.pose.pos[1], rim), b.pose.quat)
                b.lin_vel = (0.0, 0.0, 0.0)
                continue
            if home is not None:
                rest = (home.pose.pos[2] - home.spec.size[2] + container_floor(home.spec.attrs)
                        + b.spec.size[2])
                for other_name, o in self._bodies.items():
                    # only on what is in the same container: WK5's weight, in a
                    # pan, was stood on the beam's bounding box beside it (v8)
                    if (other_name == name or o is home or o.held_by is not None
                            or o.spec.fixed or o.pose.pos[2] >= b.pose.pos[2]
                            or o.spec.attrs.get("curved_top") or not _over_opening(o, home)
                            or passes(b.spec, o.spec)):
                        continue
                    reach = max(b.spec.size[0], b.spec.size[1]) + max(o.spec.size[0], o.spec.size[1])
                    if b.pose.planar_distance_to(o.pose) < reach:
                        rest = max(rest, o.pose.pos[2] + o.spec.size[2] + b.spec.size[2])
                if abs(b.pose.pos[2] - rest) > 1e-6:
                    b.pose = Pose((b.pose.pos[0], b.pose.pos[1], rest), b.pose.quat)
                b.lin_vel = home.lin_vel             # it moves with what holds it
                continue
            rest = table + b.spec.size[2]
            cover = b.spec.attrs.get("container") == "inverted"
            for other_name, o in self._bodies.items():
                if other_name == name or o.held_by is not None or not collides(o.spec):
                    continue
                if passes(b.spec, o.spec):
                    continue                      # it drops through (CL4's stones, the well's water)
                if o.spec.attrs.get("curved_top"):
                    continue                      # a fruit holds nothing up: it rolls off
                if cover and o.pose.pos[2] < b.pose.pos[2] and _fits_under(o, b):
                    continue                      # a cup comes down around it
                if o.spec.shape == "compound" and o.spec.attrs.get("parts"):
                    # a toy built from parts holds things up with its parts: a
                    # ring round a peg rests on the stand's base, not on the peg
                    rest = max(rest, _part_support(o, b))
                    continue
                if o.pose.pos[2] >= b.pose.pos[2]:
                    continue                      # not a support: not below us
                if (b.pose.planar_distance_to(o.pose) < max(o.spec.size[0], o.spec.size[1])
                        or _shadows_overlap(b, o)):
                    top = o.pose.pos[2] + o.spec.size[2]
                    inside = str(o.spec.attrs.get("container", "")) == "open"
                    rest = max(rest, o.pose.pos[2] if inside else top + b.spec.size[2])
            if b.pose.pos[2] > rest + 1e-6:
                b.pose = Pose((b.pose.pos[0], b.pose.pos[1], rest), b.pose.quat)
                b.lin_vel = (0.0, 0.0, 0.0)
            elif b.pose.pos[2] < rest - 1e-6:
                b.pose = Pose((b.pose.pos[0], b.pose.pos[1], rest), b.pose.quat)
            else:
                b.lin_vel = (0.0, 0.0, 0.0)

    # -- observation -------------------------------------------------------
    def state(self) -> WorldState:
        objects = {}
        cameras = self._scene.cameras if self._scene else ()
        # sight lines only for a task that asks (``needs_visibility``), as on
        # Isaac: otherwise ``visible_from`` is None, "not worked out"
        seen = ({c.name: set(self.visible_objects(c.name)) for c in cameras}
                if self.track_visibility else None)
        for name, b in self._bodies.items():
            vis = None if seen is None else tuple(c.name for c in cameras if name in seen[c.name])
            if name in self._cloths:
                objects[name] = self._cloths[name].state(visible_from=vis)
                continue
            objects[name] = ObjectState(
                name=name, pose=b.pose, size=b.spec.size, category=b.spec.category,
                color=b.spec.color, held_by=b.held_by, lin_vel=b.lin_vel,
                visible_from=vis, attrs=dict(b.spec.attrs),
                footprint=footprint_of(b.spec),
            )
        robots: Dict[str, Dict[str, ArmState]] = {}
        for aid, arm in self._arms.items():
            robot, _, part = aid.partition("/")
            robots.setdefault(robot, {})[part] = ArmState(
                name=aid, eef_pose=arm.pose, gripper=arm.gripper, holding=arm.holding,
                busy=not (self.arm_at_target(aid) and self.gripper_settled(aid)),
                opening=self._opening(arm),
            )
        return WorldState(
            sim_time=self._t,
            objects=objects,
            robots={r: RobotState(r, arms) for r, arms in robots.items()},
            articulations={n: j.value for n, j in self._joints.items()},
            instruction=self._instruction,
            instruction_version=self._instruction_version,
            events_fired=tuple(self._events_fired),
        )

    def render(self, camera: str) -> CameraFrame:
        """No pixels: a placeholder frame carrying the capture time, which is
        what observation-age accounting actually needs."""
        return CameraFrame(camera=camera, width=0, height=0, captured_at=self._t, image=None)

    def camera_pose(self, camera: str) -> Pose:
        """Where a camera is.  The wrist camera rides the arm, which is how
        Active Perception gets a viewpoint change with no extra camera rig."""
        for cam in (self._scene.cameras if self._scene else ()):
            if cam.name != camera:
                continue
            if cam.kind == "wrist":
                arm = self._arms.get(cam.attached_to or next(iter(self._arms)))
                model = arm_camera_model(self._scene, arm.name) if arm else None
                return wrist_camera_pose(arm.pose, model) if arm else Pose()
            return cam.pose or external_camera_pose(self._scene)
        return Pose()

    def visible_objects(self, camera: str) -> Tuple[str, ...]:
        """Visibility, to the extent a kinematic stand-in can model it.

        Three ways to be hidden: out of the camera's picture or behind other
        bodies or the arms (``engines/sight.py``: a body needs ``MIN_IN_SIGHT``
        of its camera-facing surface clear); shut in a drawer; or facing away
        -- a body may *face* a direction, and is then only visible from a
        camera standing on that side of it.  That is what makes "walk round and
        look at the back" a real action rather than a formality -- DS1's marker
        carries ``attrs={"facing": (x, y, z)}``.

        Always by sight lines (v15): until then, a task that did not ask for
        them got a guess -- hidden only under something resting right on top
        of it -- and DI6's tart showed through its cover to every service.
        The real answer comes from an instance-segmentation pass on the Isaac
        backend; this is a stand-in for developing task logic.
        """
        pose = self.camera_pose(camera)
        eye = pose.pos
        occluded = self._unseen(camera, pose)
        spec = next((c for c in self._scene.cameras if c.name == camera), None)
        reach = legibility_reach(self._scene, spec) if spec is not None else 1.0
        out = []
        for name, b in self._bodies.items():
            if name in occluded:
                continue
            blocked = False
            container = b.spec.attrs.get("inside_joint")
            if container and self._joints.get(str(container)) is not None:
                blocked = blocked or self._joints[str(container)].value < 0.05
            if not blocked:
                blocked = not legible_from(b.spec, b.pose.pos, b.pose.quat, eye, reach)
            if not blocked:
                out.append(name)
        return tuple(out)

    def _bodies_held_by(self, arm: str) -> set:
        return {n for n, b in self._bodies.items() if arm in b.grips}

    def _unseen(self, camera: str, pose: Pose) -> set:
        """Bodies out of this camera's frame or hidden by other bodies --
        cached on where everything is."""
        from ..tasks.sightlines import hand_boxes
        from .base import camera_apertures, camera_focal_length, project
        from .sight import box_bounds, crosses, facing_samples, sight_boxes

        spec = next(c for c in self._scene.cameras if c.name == camera)
        focal = camera_focal_length(self._scene, spec)
        apertures = camera_apertures(self._scene, spec)
        eye = pose.pos
        key = (camera, tuple(round(v, 4) for v in eye), tuple(round(v, 4) for v in pose.quat),
               tuple((n, tuple(round(v, 4) for v in b.pose.pos), b.held_by is None)
                     for n, b in self._bodies.items()),
               tuple(tuple(round(v, 3) for v in a.pose.pos) for a in self._arms.values()))
        hit = self._sight_cache.get(key)
        if hit is not None:
            return set(hit)
        boxes = {n: sight_boxes(b.spec.shape, b.spec.size, b.pose, b.spec.attrs)
                 for n, b in self._bodies.items() if collides(b.spec) and drawn(b.spec)}
        # a hand is in the way as much as any body -- but not of what it holds,
        # and not of its own wrist camera, which looks past it
        own = spec.attached_to if spec.kind == "wrist" else None
        hands = {}
        for a in self._arms.values():
            if a.name == own:
                continue
            held = [self._bodies[n].spec for n in self._bodies_held_by(a.name)]
            hands[a.name] = hand_boxes(a.pose.pos, arm_camera_model(self._scene, a.name),
                                       yaw_of(a.pose.quat),
                                       grip_half=held[0].size[1] if held else 0.0)
        out = []
        for name, b in self._bodies.items():
            if not drawn(b.spec):
                out.append(name)                 # nothing of it is drawn: no camera shows it
                continue
            others = [box for n, bs in boxes.items() if n != name for box in bs]
            others += [box for arm, bs in hands.items() if name not in self._bodies_held_by(arm)
                       for box in bs]
            points = facing_samples(box_bounds(b.spec.shape, b.spec.size, b.pose), eye, 3)
            clear = 0
            for p in points:
                uv = project(pose, p, spec.width, spec.height, focal, apertures)
                if (uv is not None and 0 <= uv[0] <= spec.width and 0 <= uv[1] <= spec.height
                        and not any(crosses(p, eye, box) for box in others)):
                    clear += 1
            if not points or clear < MIN_IN_SIGHT * len(points):
                out.append(name)
        if len(self._sight_cache) > 256:
            self._sight_cache.clear()
        self._sight_cache[key] = tuple(out)
        return set(out)

    # -- control -----------------------------------------------------------
    def set_arm_target(self, arm: str, pose: Pose) -> None:
        self._arms[arm].target = pose
        self._arms[arm].joint_goal = None

    def set_arm_joints(self, arm: str, joints) -> Pose:
        """The hand goes, in a straight line as for a pose, to where the joints
        put it (``engines.kinematics``); there the joints are these."""
        a = self._arms[arm]
        kin = arm_kinematics(a.model.name) if a.model is not None else None
        if kin is None or a.base is None:
            raise NotImplementedError(f"no joint model for {getattr(a.model, 'name', arm)}")
        if len(joints) != len(kin.joints):
            raise ValueError(f"{arm} has {len(kin.joints)} joints, not {len(joints)}")
        q = tuple(min(max(float(v), lo), hi) for v, lo, hi in zip(joints, kin.lower, kin.upper))
        a.target = forward(a.model, a.base, q)
        a.joint_goal = q
        return a.target

    def arm_joints(self, arm: str):
        """Exact after a joint target or at the start; elsewhere the joints the
        inverse kinematics finds for where the hand is, from the last ones."""
        a = self._arms[arm]
        kin = arm_kinematics(a.model.name) if a.model is not None else None
        if kin is None or a.base is None:
            return None
        if a.joints is None or a.joints_at != a.pose:
            sol = inverse(a.model, a.base, a.pose, seed=a.joints or kin.ready)
            a.joints, a.joints_at = sol.joints, a.pose
        return a.joints

    def set_gripper(self, arm: str, value: float) -> None:
        self._arms[arm].gripper_target = min(1.0, max(0.0, value))

    def arm_at_target(self, arm: str, pos_tol: float = ARRIVAL_POS_TOL,
                      rot_tol: float = ARRIVAL_ROT_TOL) -> bool:
        a = self._arms[arm]
        return a.pose.distance_to(a.target) <= pos_tol

    def gripper_settled(self, arm: str) -> bool:
        a = self._arms[arm]
        return abs(a.gripper - a.gripper_target) <= 0.02

    def set_articulation_target(self, joint: str, value: float) -> None:
        j = self._joints[joint]
        j.target = min(max(value, j.spec.limits[0]), j.spec.limits[1])

    def set_joint_drive(self, joint: str, stiffness: float, damping: float) -> None:
        # the mock has no gravity on joints: a joint let go still goes to its
        # target, which is where the task says it ends up
        if joint not in self._joints:
            raise KeyError(joint)

    def launch(self, body: str, velocity) -> None:
        b = self._bodies[body]
        if b.held_by is not None:
            raise RuntimeError(f"refusing to launch {body!r}: it is grasped")
        b.flight = tuple(float(v) for v in velocity)

    def _fly(self, dt: float) -> None:
        """Thrown bodies: a parabola under gravity, until they come down on
        the table.  A free body the thrown one runs into falls over, away from
        it -- the mock's stand-in for knocking a can down -- and the thrown
        body carries on, slowed."""
        from ..core.geometry import rotate

        table = self._scene.table_height if self._scene else 0.0
        for name, b in self._bodies.items():
            if b.flight is None:
                continue
            vx, vy, vz = b.flight
            vz -= GRAVITY * dt
            x, y, z = (p + v * dt for p, v in zip(b.pose.pos, (vx, vy, vz)))
            b.pose = Pose((x, y, z), b.pose.quat)
            b.flight = (vx, vy, vz)
            for other_name, o in self._bodies.items():
                if (o is b or o.spec.fixed or o.held_by is not None or not collides(o.spec)
                        or other_name in self._knocked):
                    continue
                if all(abs(p - q) <= hb + ho for p, q, hb, ho in
                       zip(b.pose.pos, o.pose.pos, b.spec.size, o.spec.size)):
                    heading = math.atan2(vy, vx)
                    lie = (math.cos(math.pi / 4), -math.sin(heading) * math.sin(math.pi / 4),
                           math.cos(heading) * math.sin(math.pi / 4), 0.0)
                    ox, oy, _ = o.pose.pos
                    reach = o.spec.size[2]
                    o.pose = Pose((ox + reach * math.cos(heading), oy + reach * math.sin(heading),
                                   o.pose.pos[2]), quat_mul(lie, o.pose.quat))
                    self._knocked.add(other_name)
                    b.flight = (vx * 0.4, vy * 0.4, vz)
            if vz < 0 and z - b.spec.size[2] <= table + 1e-4:
                b.flight = None
                b.pose = Pose((x, y, table + b.spec.size[2]), b.pose.quat)

    def teleport(self, body: str, pose: Pose) -> None:
        b = self._bodies[body]
        if b.held_by is not None:
            raise RuntimeError(f"refusing to teleport {body!r}: it is grasped")
        if body in self._cloths:
            raise RuntimeError(f"refusing to teleport {body!r}: a cloth has no one pose to set")
        b.pose = pose

    # -- hooks used by the event scheduler ---------------------------------
    def set_instruction(self, text: str, initial: bool = False) -> None:
        """``initial=True`` sets the opening instruction without counting as a
        revision -- otherwise every episode would start at version 1 and every
        "has the instruction changed?" predicate would be true from t=0."""
        self._instruction = text
        if not initial:
            self._instruction_version += 1

    def note_event(self, name: str) -> None:
        self._events_fired.append(name)

    def _hand_meets(self, arm: _Arm, pose: Pose) -> set:
        """What the hand's parts -- palm, camera, fingers at their present
        opening -- and what it holds overlap at ``pose``: bodies by name, and
        "table" when either goes below its top.  Only for a hand pointing down, which is
        what the robot model's ``HandBox`` describes."""
        from ..core.geometry import rotate

        if rotate(pose.quat, (0.0, 0.0, 1.0))[2] > -0.9:
            return set()
        model = arm_camera_model(self._scene, arm.name)
        x = rotate(pose.quat, (1.0, 0.0, 0.0))
        c, s = x[0], x[1]
        n = math.hypot(c, s) or 1.0
        c, s = c / n, s / n
        gx, gy, gz = pose.pos
        boxes = []                        # (x span, y span, z span), the tool's frame
        known = HAND_COLLIDERS.get(model.name)
        if known is None:
            inward = model.max_opening / 2 - self._opening(arm) / 2
            for part in model.open_hand:
                y = part.y
                if part.name == "finger":
                    y = (y[0] - inward, y[1] - inward) if y[0] >= 0 else (y[0] + inward, y[1] + inward)
                boxes.append((part.x, y, (part.bottom, part.bottom + part.height)))
        else:
            face = self._opening(arm) / 2         # each finger's inner face
            for half, (y0, y1), up in known["finger"]:
                boxes.append(((-half, half), (face + y0, face + y1), up))
                boxes.append(((-half, half), (-face - y1, -face - y0), up))
            boxes += list(known["hand"])
            boxes += [(p.x, p.y, (p.bottom, p.bottom + p.height))
                      for p in model.open_hand if p.name == "camera"]
        hand = []
        for x, y, (lo, hi) in boxes:
            corners = [(gx + c * px - s * py, gy + s * px + c * py) for px in x for py in y]
            hand.append(_prism([corners[0], corners[1], corners[3], corners[2]], gz + lo, gz + hi))
        # what it holds comes along, and is as solid as on Isaac: brought down
        # beside a placed piece it is stopped there (WK6, I-015), and a held
        # beaker meets its mat before the palm meets the rod beyond it (DS5)
        held = self._bodies.get(arm.holding) if arm.holding else None
        grip = held.grips.get(arm.name) if held is not None else None
        if grip is not None:
            hand += _solid_pieces(held.spec, compose(pose, grip))
        met = set()
        if min(h[2][2][0] for h in hand) < self._scene.table_height - TABLE_SLACK:
            met.add("table")
        for name, b in self._bodies.items():
            # (a cloth gives: the fingers go down through it to what it lies on)
            if name == arm.holding or not collides(b.spec) or b.spec.attrs.get("pressable") \
                    or b.spec.attrs.get("lever") or name in self._cloths:
                continue
            key = (tuple(b.pose.pos), tuple(b.pose.quat))
            cached = self._pieces.get(name)
            if cached is None or cached[0] != key:
                cached = self._pieces[name] = (key, _solid_pieces(b.spec, b.pose))
            if any(_convex_overlap(h, piece) for piece in cached[1] for h in hand):
                met.add(name)
        return met

    def _fingers_reach(self, arm: _Arm, body) -> bool:
        """Do the fingertips reach into what the hand closes on, as they must
        on Isaac to take hold of it: ``FINGERTIP_OVERLAP`` below its top, at
        least -- the top of the body's box, or of its handle when the hand is
        at the handle (CL5's pieces are held by a knob 2.5 cm above the
        plate).  They reach below the tool centre point by the robot's own
        finger length (8 mm on the FR3 and the X5)."""
        from ..core.geometry import rotate

        model = arm_camera_model(self._scene, arm.name)
        tips = arm.pose.pos[2] + min(b.bottom for b in model.open_hand if b.name == "finger")
        spec, pose = body.spec, body.pose
        local = rotate(quat_conj(pose.quat), tuple(t - p for t, p in zip(arm.pose.pos, pose.pos)))
        regions = [((0.0, 0.0, 0.0), spec.size)]
        if spec.attrs.get("handle"):
            regions.append(tuple(spec.attrs["handle"]))
        for centre, half in regions:
            if not all(abs(v - c) <= h + GRASP_MARGIN for v, c, h in zip(local, centre, half)):
                continue
            mid = rotate(pose.quat, tuple(centre))
            top = pose.pos[2] + mid[2] + sum(
                abs(rotate(pose.quat, e)[2]) * h
                for e, h in zip(((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)), half))
            if tips <= top - FINGERTIP_OVERLAP:
                return True
        return False

    def set_color(self, body: str, color: str) -> None:
        b = self._bodies[body]
        b.spec = replace(b.spec, color=color)


def _aabb(points) -> tuple:
    return tuple((min(p[i] for p in points), max(p[i] for p in points)) for i in range(3))


def _prism(corners, lo: float, hi: float) -> tuple:
    """An upright prism -- a turned rectangle from ``lo`` to ``hi`` -- as
    (its corner points, its axes, its bounds)."""
    (ax, ay), (bx, by) = corners[0], corners[1]
    n = math.hypot(bx - ax, by - ay) or 1.0
    u = ((bx - ax) / n, (by - ay) / n, 0.0)
    points = [(x, y, z) for x, y in corners for z in (lo, hi)]
    return points, [(0.0, 0.0, 1.0), u, (-u[1], u[0], 0.0)], _aabb(points)


#: an octagon's side directions, and up: what a round part is taken as
_OCTAGON_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0),
                 (math.sqrt(0.5), -math.sqrt(0.5), 0.0), (0.0, 0.0, 1.0))
_BOX_AXES = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _solid_pieces(spec: BodySpec, pose: Pose) -> list:
    """A body's solid pieces as Isaac collides with them -- each part's own
    turned box, a round one as the octagon round it -- not the world-axis
    bounds around each (``sight.part_boxes``): a pan's rim of 16 turned boxes
    was 2 cm fatter that way, and every pick from the pan stopped short.
    Each piece is (its corner points, its axes, its bounds)."""
    from ..core.geometry import quat_mul, rotate
    from .parts import box_pieces, octagon_points

    attrs = spec.attrs or {}
    if attrs.get("proxy"):
        parts = list(attrs["proxy"])
    elif spec.shape == "compound" and attrs.get("parts"):
        parts = [p for p in attrs["parts"] if p[5]]
    else:
        parts = _shape_parts(spec)
    out = []
    for part in parts:
        # a roof block or a tangram triangle as the prism it is, not slabs:
        # two triangles laid along their long side were slabs that overlapped
        pieces = [part] if part[0] == "prism" else box_pieces(part)
        for kind, centre, dims, rot, _, _ in pieces:
            q = quat_mul(pose.quat, rot)
            mid = tuple(a + b for a, b in zip(pose.pos, rotate(pose.quat, centre)))
            if kind == "prism":
                hx, hy, hz = dims
                local = [(x, y, z) for y in (-hy, hy) for x, z in ((-hx, -hz), (hx, -hz), (0.0, hz))]
                n = math.hypot(2 * hz, hx) or 1.0
                axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0),
                        (2 * hz / n, 0.0, hx / n), (-2 * hz / n, 0.0, hx / n),
                        (-hx / n, 0.0, 2 * hz / n), (hx / n, 0.0, 2 * hz / n))
            elif kind in ("cylinder", "capsule", "sphere", "octagon") or (
                    kind.startswith("lathe:") and kind != "lathe:slab"):
                r = dims[0]
                hz = {"cylinder": dims[1], "capsule": dims[1] + dims[0], "sphere": dims[0]}.get(
                    kind, dims[2])
                local = [(x, y, z) for x, y in octagon_points(r) for z in (-hz, hz)]
                axes = _OCTAGON_AXES
            else:
                hx, hy, hz = (dims[0], dims[1], 0.0005) if kind == "print" else dims
                local = [(sx * hx, sy * hy, sz * hz)
                         for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
                axes = _BOX_AXES
            points = [tuple(m + v for m, v in zip(mid, rotate(q, p))) for p in local]
            out.append((points, [rotate(q, a) for a in axes], _aabb(points)))
    return out


def _shape_parts(spec: BodySpec) -> list:
    """A plain body's solid parts in its own frame: its shape, or an open
    container's walls and floor (an inverted one's walls and top)."""
    from .base import container_floor

    hx, hy, hz = spec.size
    attrs = spec.attrs or {}
    kind, wall = attrs.get("container"), float(attrs.get("wall") or 0.0)
    if kind in ("open", "inverted") and wall:
        w = wall / 2
        parts = [("box", (0.0, -hy + w, 0.0), (hx, w, hz)), ("box", (0.0, hy - w, 0.0), (hx, w, hz)),
                 ("box", (-hx + w, 0.0, 0.0), (w, hy, hz)), ("box", (hx - w, 0.0, 0.0), (w, hy, hz))]
        if kind == "open":
            f = container_floor(attrs) / 2
            parts.append(("box", (0.0, 0.0, -hz + f), (hx, hy, f)))
        else:
            parts.append(("box", (0.0, 0.0, hz - w), (hx, hy, w)))
    elif spec.shape == "cylinder":
        parts = [("cylinder", (0.0, 0.0, 0.0), (hx, hz, 0.0))]
    elif spec.shape == "sphere":
        parts = [("sphere", (0.0, 0.0, 0.0), (hx, 0.0, 0.0))]
    else:
        parts = [("box", (0.0, 0.0, 0.0), (hx, hy, hz))]
    return [p + ((1.0, 0.0, 0.0, 0.0), None, True) for p in parts]


def _convex_overlap(a, b) -> bool:
    """Do two convex pieces -- (points, axes, bounds) -- overlap?  Their
    bounds first, then separating axes: each one's axes and their crossings."""
    pa, ea, ba = a
    pb, eb, bb = b
    if any(ba[i][1] <= bb[i][0] or bb[i][1] <= ba[i][0] for i in range(3)):
        return False
    axes = list(ea) + list(eb)
    for d1 in ea:
        for d2 in eb:
            c = (d1[1] * d2[2] - d1[2] * d2[1], d1[2] * d2[0] - d1[0] * d2[2],
                 d1[0] * d2[1] - d1[1] * d2[0])
            n = math.sqrt(c[0] * c[0] + c[1] * c[1] + c[2] * c[2])
            if n > 1e-9:
                axes.append((c[0] / n, c[1] / n, c[2] / n))
    for ax in axes:
        p1 = [p[0] * ax[0] + p[1] * ax[1] + p[2] * ax[2] for p in pa]
        p2 = [p[0] * ax[0] + p[1] * ax[1] + p[2] * ax[2] for p in pb]
        if max(p1) <= min(p2) + 1e-9 or max(p2) <= min(p1) + 1e-9:
            return False
    return True


def _in_hull(point, hull) -> bool:
    """Is (x, y) inside a counter-clockwise convex polygon?"""
    x, y = point[0], point[1]
    n = len(hull)
    return n >= 3 and all(
        (hull[(i + 1) % n][0] - hull[i][0]) * (y - hull[i][1])
        - (hull[(i + 1) % n][1] - hull[i][1]) * (x - hull[i][0]) >= 0 for i in range(n))


def _hinge_pivot(j: _Joint) -> Vec3:
    from ..core.geometry import rotate

    d = rotate(j.origin.quat, j.spec.pivot)
    return tuple(o + v for o, v in zip(j.origin.pos, d))


def _joint_pose(j: "_Joint", body) -> Pose:
    """Where a joint at its current value puts the body it carries."""
    if j.spec.kind == "hinge":
        return _hinge_pose(j, j.value)
    return Pose(tuple(o + a * j.value for o, a in zip(j.origin.pos, j.axis)), body.pose.quat)


def _hinge_pose(j: _Joint, angle: float) -> Pose:
    """Where a hinge puts its body at ``angle``: turned about the axis through
    the pivot."""
    from ..core.geometry import rotate

    ax = j.axis
    turn = (math.cos(angle / 2),) + tuple(math.sin(angle / 2) * a for a in ax)
    pivot = _hinge_pivot(j)
    arm = rotate(turn, tuple(o - p for o, p in zip(j.origin.pos, pivot)))
    return Pose(tuple(p + a for p, a in zip(pivot, arm)), quat_mul(turn, j.origin.quat))


def _hinge_angle(j: _Joint, pose: Pose) -> float:
    """How far a body at ``pose`` has been turned about a hinge's axis."""
    ax = j.axis
    pivot = _hinge_pivot(j)
    u = tuple(o - p for o, p in zip(j.origin.pos, pivot))
    v = tuple(o - p for o, p in zip(pose.pos, pivot))

    def flat(w):
        along = sum(a * b for a, b in zip(w, ax))
        return tuple(a - along * b for a, b in zip(w, ax))

    u, v = flat(u), flat(v)
    if math.hypot(*u) < 1e-6:
        # the axis runs through the body's own middle (a turntable turned by
        # a knob on its rim): how far it has turned is in its heading
        turn = quat_mul(pose.quat, quat_conj(j.origin.quat))
        return 2 * math.atan2(sum(t * a for t, a in zip(turn[1:], ax)), turn[0])
    if math.hypot(*v) < 1e-9:
        return j.value
    cross = (u[1] * v[2] - u[2] * v[1], u[2] * v[0] - u[0] * v[2], u[0] * v[1] - u[1] * v[0])
    return math.atan2(sum(c * a for c, a in zip(cross, ax)), sum(a * b for a, b in zip(u, v)))


#: how far two bodies' shadows must overlap for one to hold the other up: a
#: plank across two cubes rests on both, though its middle is over neither
SUPPORT_OVERLAP = 0.01


def _shadow(body: "_Body"):
    """A body's rectangle on the table, turned with it."""
    from ..core.geometry import yaw_of

    hx, hy = body.spec.size[0], body.spec.size[1]
    yaw = yaw_of(body.pose.quat)
    c, s = math.cos(yaw), math.sin(yaw)
    x, y = body.pose.pos[0], body.pose.pos[1]
    return tuple((x + c * px - s * py, y + s * px + c * py)
                 for px, py in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)))


def _shadows_overlap(body: "_Body", support: "_Body") -> bool:
    from .parts import overlap

    return overlap(_shadow(body), _shadow(support)) > SUPPORT_OVERLAP


def _part_support(support: "_Body", body: "_Body") -> float:
    """The height ``body``'s centre rests at on ``support``'s parts: the top of
    the highest solid part below its centre whose shadow on the table overlaps
    one of the body's (a ring on a ring overlaps its rim, not its hole; a block
    turned square to a hole in a lid falls through it).  A support may ask for
    a finer overlap than ``PART_OVERLAP`` (``fit_tolerance``).

    A part of the body whose underside is above the body's own -- the roof of
    a hollow brick, over the studs of the brick below -- comes down that much
    further before it meets what is under it: the brick goes down over the
    studs until its walls stand on something (CR5)."""
    from .parts import body_parts, footprints, overlap

    spec = body.spec
    mine = footprints(body_parts(spec.shape, spec.size, spec.attrs),
                      tuple(body.pose.pos), tuple(body.pose.quat))
    tol = float(support.spec.attrs.get("fit_tolerance", PART_OVERLAP))
    z = body.pose.pos[2]
    lowest = min((bottom for _, bottom, _ in mine), default=z - spec.size[2])
    best = -math.inf
    for shadow, _, top in footprints(tuple(support.spec.attrs["parts"]),
                                     tuple(support.pose.pos), tuple(support.pose.quat)):
        if top >= z:
            continue
        for m, bottom, _ in mine:
            raised = bottom - lowest
            rest = top + spec.size[2] - (raised if raised > RAISED_PART else 0.0)
            if rest > best and overlap(shadow, m) > tol:
                best = rest
    return best


def _over_opening(body: "_Body", container: "_Body") -> bool:
    """Is ``body``'s centre over the container's opening (inside its walls)?"""
    wall = float(container.spec.attrs.get("wall") or 0.0)
    dx = body.pose.pos[0] - container.pose.pos[0]
    dy = body.pose.pos[1] - container.pose.pos[1]
    if footprint_of(container.spec) == "round":
        return math.hypot(dx, dy) < container.spec.size[0] - wall
    yaw = yaw_of(container.pose.quat)
    c, s = math.cos(yaw), math.sin(yaw)
    lx, ly = c * dx + s * dy, -s * dx + c * dy
    return (abs(lx) < container.spec.size[0] - wall
            and abs(ly) < container.spec.size[1] - wall)


def _fits_inside(body: "_Body", container: "_Body") -> bool:
    wall = float(container.spec.attrs.get("wall") or 0.0)
    return (min(body.spec.size[:2]) < min(container.spec.size[:2]) - wall
            and max(body.spec.size[:2]) < max(container.spec.size[:2]) - wall)


def _fits_under(small: "_Body", cup: "_Body") -> bool:
    """Would ``small`` be inside ``cup`` put down over it?"""
    wall = float(cup.spec.attrs.get("wall") or 0.0)
    return (abs(small.pose.pos[0] - cup.pose.pos[0]) + small.spec.size[0] <= cup.spec.size[0] - wall
            and abs(small.pose.pos[1] - cup.pose.pos[1]) + small.spec.size[1] <= cup.spec.size[1] - wall
            and 2 * small.spec.size[2] <= 2 * cup.spec.size[2] - wall)


def footprint_of(spec: BodySpec) -> str:
    """Round for cylinders and spheres, and for catalogue meshes marked round."""
    if spec.shape in ("cylinder", "sphere"):
        return "round"
    return str(spec.attrs.get("footprint", "rect"))


def _towards_scalar(cur: float, goal: float, step: float) -> float:
    if abs(goal - cur) <= step:
        return goal
    return cur + math.copysign(step, goal - cur)
