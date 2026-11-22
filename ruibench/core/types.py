"""Engine-agnostic value types shared by every layer of RUIBench.

Nothing in this module may import an engine, a simulator, or numpy.  Task
logic (predicates, milestones, event scripts, scorers) is written against
these types only, which is what keeps the engine behind ``ruibench.engines``
(see ``docs/ARCHITECTURE.md`` and design doc v0 §7.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Mapping, Optional, Sequence, Tuple

Vec3 = Tuple[float, float, float]
Quat = Tuple[float, float, float, float]  # (w, x, y, z)

IDENTITY_QUAT: Quat = (1.0, 0.0, 0.0, 0.0)

#: The end-effector orientation for tabletop grasping: tool pointing straight
#: down.  Scripted references use it for every approach, place and look-down.
#: The identity quaternion is the wrong default for a manipulator -- on a real
#: arm it points the gripper at the ceiling.
GRASP_DOWN: Quat = (0.0, 1.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Pose:
    """A 6-DoF pose.  Position in metres, orientation as a (w,x,y,z) quaternion."""

    pos: Vec3 = (0.0, 0.0, 0.0)
    quat: Quat = IDENTITY_QUAT

    def __post_init__(self) -> None:
        object.__setattr__(self, "pos", tuple(float(v) for v in self.pos))
        object.__setattr__(self, "quat", tuple(float(v) for v in self.quat))

    def translated(self, d: Vec3) -> "Pose":
        return replace(self, pos=(self.pos[0] + d[0], self.pos[1] + d[1], self.pos[2] + d[2]))

    def distance_to(self, other: "Pose | Vec3") -> float:
        p = other.pos if isinstance(other, Pose) else other
        return math.dist(self.pos, p)

    def planar_distance_to(self, other: "Pose | Vec3") -> float:
        p = other.pos if isinstance(other, Pose) else other
        return math.dist(self.pos[:2], p[:2])

    def angle_to(self, other: "Pose") -> float:
        """Geodesic angle between the two orientations, in radians."""
        dot = abs(sum(a * b for a, b in zip(self.quat, other.quat)))
        return 2.0 * math.acos(max(-1.0, min(1.0, dot)))

    def as_dict(self) -> dict:
        return {"pos": list(self.pos), "quat": list(self.quat)}

    @classmethod
    def from_dict(cls, d: Mapping) -> "Pose":
        return cls(tuple(d.get("pos", (0, 0, 0))), tuple(d.get("quat", IDENTITY_QUAT)))


# --------------------------------------------------------------------------
# world state
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ObjectState:
    """Ground-truth state of one rigid body.

    This is *privileged* information: it feeds predicates and scripted
    reference policies, never a model.  What a model may see is decided by
    the interface (see ``ruibench.interfaces``) and its perception tools (RUIKit's).
    """

    name: str
    pose: Pose
    size: Vec3 = (0.05, 0.05, 0.05)
    category: str = "object"          # cube / bowl / plate / basket / ball / ...
    color: Optional[str] = None
    held_by: Optional[str] = None     # "<robot>/<arm>" when grasped
    lin_vel: Vec3 = (0.0, 0.0, 0.0)
    #: the cameras with a line of sight to it -- ``()`` seen by none, ``None``
    #: not worked out (a renderer only does it for a task that asks, and "seen
    #: by none" read as "not worked out" let a service report what was hidden)
    visible_from: Optional[Tuple[str, ...]] = None
    attrs: Mapping[str, object] = field(default_factory=dict)
    #: "rect" or "round": how the body occupies the table, seen from above.
    #: Predicates test containment against this shape -- a circle around a
    #: rectangular bin reaches past its short sides, and a square around a round
    #: plate reaches past its rim.
    footprint: str = "rect"
    #: a cloth's marks, where they are now (``engines.cloth``): for a body that
    #: is not rigid, ``pose`` and ``size`` are the box round these.  None for
    #: every rigid body
    points: Optional[Tuple[Vec3, ...]] = None

    def contains_xy(self, point, margin: float = 0.0) -> bool:
        """Is ``point`` (x, y, ...) over this body's footprint, grown by ``margin``?"""
        dx, dy = point[0] - self.pose.pos[0], point[1] - self.pose.pos[1]
        if self.footprint == "round":
            r = max(self.size[0], self.size[1]) + margin
            return dx * dx + dy * dy <= r * r
        w, x, y, z = self.pose.quat
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        c, s = math.cos(-yaw), math.sin(-yaw)
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= self.size[0] + margin and abs(ly) <= self.size[1] + margin

    @property
    def is_held(self) -> bool:
        return self.held_by is not None

    def seen(self, camera: Optional[str] = None) -> bool:
        """Is it in sight of ``camera`` (of any camera, for None)?  Asking a
        world that did not work sight lines out is a harness error, never "no"
        or "yes" (``TaskInstance.metadata["needs_visibility"]``)."""
        if self.visible_from is None:
            raise VisibilityNotComputed(
                f"no sight lines for {self.name!r}: the task must ask for them "
                "(metadata needs_visibility)")
        return bool(self.visible_from) if camera is None else camera in self.visible_from

    @property
    def speed(self) -> float:
        return math.sqrt(sum(v * v for v in self.lin_vel))


class VisibilityNotComputed(RuntimeError):
    """A reader asked about sight lines the engine did not work out."""


@dataclass(frozen=True)
class ArmReading:
    """What an arm tells an interface under test about itself: where the hand
    is, how far the fingers are closed, whether they close on something --
    never *what* (a held body's name is the scorer's: DS1's "present2",
    DS6's "splint_cucl2")."""

    name: str
    eef_pose: Pose
    gripper: float = 0.0
    grasping: bool = False
    busy: bool = False
    opening: Optional[float] = None
    joints: Optional[Tuple[float, ...]] = None


@dataclass(frozen=True)
class ArmState:
    name: str
    eef_pose: Pose
    gripper: float = 0.0              # 0 = fully open, 1 = fully closed: the command
    holding: Optional[str] = None     # privileged: the scorer's and the log's
    busy: bool = False                # an action is currently executing
    #: how far apart the fingers are (m), measured: on something they close
    #: on, its width, where ``gripper`` still says 1 (bench-issue #17).  None:
    #: the engine cannot tell
    opening: Optional[float] = None
    #: the arm's joint angles (radians, base to wrist), measured -- the robot's
    #: own readings (v1.13).  None: the engine has no joints to read (the mock)
    joints: Optional[Tuple[float, ...]] = None

    def public(self) -> ArmReading:
        return ArmReading(self.name, self.eef_pose, self.gripper, self.holding is not None,
                          self.busy, self.opening, self.joints)


@dataclass(frozen=True)
class RobotState:
    name: str
    arms: Mapping[str, ArmState]

    @property
    def arm_names(self) -> Tuple[str, ...]:
        return tuple(self.arms)


@dataclass(frozen=True)
class WorldState:
    """One privileged snapshot of the world at ``sim_time``.

    ``instruction`` lives here on purpose: an event may rewrite it mid-episode
    (task DI4), and predicates need to know which version was in force at a
    given moment.
    """

    sim_time: float
    objects: Mapping[str, ObjectState]
    robots: Mapping[str, RobotState]
    articulations: Mapping[str, float] = field(default_factory=dict)
    instruction: str = ""
    instruction_version: int = 0
    events_fired: Tuple[str, ...] = ()

    def obj(self, name: str) -> ObjectState:
        try:
            return self.objects[name]
        except KeyError as exc:  # pragma: no cover - defensive
            raise KeyError(f"no object {name!r}; have {sorted(self.objects)}") from exc

    def arm(self, arm: str) -> ArmState:
        """Look up an arm by ``"<robot>/<arm>"`` or by a unique bare arm name."""
        if "/" in arm:
            robot, _, part = arm.partition("/")
            return self.robots[robot].arms[part]
        matches = [a for r in self.robots.values() for n, a in r.arms.items() if n == arm]
        if len(matches) != 1:
            raise KeyError(f"arm {arm!r} is not unique; qualify it as '<robot>/<arm>'")
        return matches[0]

    @property
    def arm_ids(self) -> Tuple[str, ...]:
        return tuple(f"{r}/{a}" for r, rob in self.robots.items() for a in rob.arms)


# --------------------------------------------------------------------------
# observations
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CameraFrame:
    """One rendered frame.  ``image`` is an opaque handle (path, array, bytes)."""

    camera: str
    width: int
    height: int
    captured_at: float          # sim time at capture -- the source of "observation age"
    image: object = None
    #: metres along the optical axis, one value a pixel -- what a depth camera
    #: measures, before any sensor noise -- when the camera was asked for it
    #: (``CameraSpec.depth``); else None
    depth: object = None

    def age_at(self, now: float) -> float:
        return max(0.0, now - self.captured_at)


@dataclass(frozen=True)
class Observation:
    """What the harness captured; each interface decides how much of it to expose."""

    sim_time: float
    frames: Mapping[str, CameraFrame]
    #: ``ArmReading`` for an interface under test, ``ArmState`` for the
    #: harness's own test double (I0)
    proprioception: Mapping[str, object]
    instruction: str
    instruction_version: int = 0
    feedback: Tuple[str, ...] = ()       # 1-bit "correct"/"wrong" messages, In-Context suites
    notes: Mapping[str, object] = field(default_factory=dict)

    @property
    def observation_age(self) -> float:
        if not self.frames:
            return 0.0
        return max(f.age_at(self.sim_time) for f in self.frames.values())


__all__ = [
    "Vec3", "Quat", "IDENTITY_QUAT", "GRASP_DOWN", "Pose", "ObjectState", "ArmState",
    "ArmReading", "VisibilityNotComputed", "RobotState", "WorldState", "CameraFrame",
    "Observation",
]
