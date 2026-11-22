"""The engine boundary.

Everything above this line -- predicates, milestones, event scripts, knobs,
scorers, interfaces -- is engine-agnostic.  Everything below it is a simulator.
Adding RoboLab/Isaac later means writing one more subclass of :class:`Engine`,
not touching a single task (design doc v0 §7.1).

A scene is described declaratively by :class:`SceneSpec` so that the same task
card can be instantiated on any backend.
"""

from __future__ import annotations

import functools
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.types import CameraFrame, Pose, Vec3, WorldState

#: The named colours a task may use, as they should look (sRGB).  One table for
#: the renderer and for every drawing of a scene, so a "red" cube is the same
#: red in both; the renderer takes them to linear first (``finishes.linear``).
COLORS: Mapping[str, Tuple[float, float, float]] = {
    "red": (0.9, 0.1, 0.1), "green": (0.1, 0.8, 0.2), "blue": (0.15, 0.3, 0.9),
    "yellow": (0.95, 0.85, 0.1), "white": (0.95, 0.95, 0.95), "black": (0.08, 0.08, 0.08),
    "orange": (0.95, 0.5, 0.1), "purple": (0.6, 0.2, 0.8), "grey": (0.5, 0.52, 0.55),
    # the v1 toy palette
    "pink": (0.95, 0.45, 0.6), "brown": (0.45, 0.25, 0.12), "tan": (0.8, 0.58, 0.3),
    "cream": (0.95, 0.88, 0.7), "gold": (0.95, 0.72, 0.15), "teal": (0.1, 0.6, 0.6),
    "navy": (0.1, 0.15, 0.45), "lime": (0.55, 0.85, 0.15), "dark": (0.18, 0.18, 0.2),
    "silver": (0.75, 0.77, 0.8), "sky": (0.45, 0.7, 0.95), "wood": (0.7, 0.5, 0.3),
    # dyed felt: a play mat's grass and river (v2), duller than paint
    "grass": (0.36, 0.56, 0.26), "river": (0.28, 0.48, 0.72),
    # a card table's baize: where CL6's taken king is set down (v8)
    "baize": (0.09, 0.33, 0.2),
    # water, as a liquid finish tints it ("glass:water", v8)
    "water": (0.55, 0.75, 0.9),
    # a pale green glaze: CL4's bowl (v8)
    "celadon": (0.70, 0.81, 0.73),
    # pale green bottle glass: CL4's jar (v11)
    "bottle": (0.62, 0.80, 0.64),
    # glazes for DS2's flower pots, and the bare clay at their feet (v10)
    "cobalt": (0.13, 0.30, 0.68), "sunflower": (0.96, 0.74, 0.12), "mint": (0.50, 0.80, 0.64),
    "terracotta": (0.70, 0.38, 0.24),
    # printing ink: black, and exactly black, so a "flame:" finish in it emits
    # nothing at all until a reactor recolours it (DS3's ultraviolet mark)
    "ink": (0.0, 0.0, 0.0),
}


def color_rgb(color) -> Tuple[float, float, float]:
    """A colour name from :data:`COLORS`, an (r, g, b) tuple as it is, or the
    colour of a finish ("paint:red", "wood:oak" -- ``engines.finishes``)."""
    if isinstance(color, (tuple, list)):
        return tuple(float(v) for v in color)
    from .finishes import finish_rgb, is_finish

    if is_finish(color):
        return finish_rgb(color)
    return COLORS.get(color or "white", COLORS["white"])


def color_hex(color: Optional[str]) -> str:
    return "#" + "".join(f"{round(c * 255):02x}" for c in color_rgb(color))


# --------------------------------------------------------------------------
# declarative scene description
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class BodySpec:
    """One rigid body.  ``asset`` names a mesh from the asset library; when it is
    None the engine builds the primitive given by ``shape``.  A body built of
    parts may name its own friction (``attrs["friction"]``: CR5's seats, next
    to none); the others have the physics engine's 0.5, and between two
    bodies the smaller counts."""

    name: str
    shape: str = "box"                 # box | cylinder | sphere | mesh | compound | cloth
    size: Vec3 = (0.025, 0.025, 0.025)
    pose: Pose = field(default_factory=Pose)
    category: str = "object"
    color: Optional[str] = None
    mass: float = 0.05
    fixed: bool = False
    asset: Optional[str] = None        # e.g. "libero/akita_black_bowl"
    attrs: Mapping[str, object] = field(default_factory=dict)


def collides(body: BodySpec) -> bool:
    """False for a visual-only body -- a mark painted on something.  It pushes
    nothing, supports nothing and is never in a gripper's way."""
    return bool(body.attrs.get("collides", True))


def rides(body: BodySpec) -> bool:
    """True for a body that rides on another (``attached_to``): a mark moved
    with its parent every step and by no physics of its own -- on Isaac a
    kinematic body with no collider at all.  Until v2.3.8 it was a dynamic
    body whose pose was written each step: it gathered free-fall speed all
    episode, and one with a colliding part (DS3's detector lights) shot away
    whatever it touched."""
    return attached_to(body) is not None


def drawn(body: BodySpec) -> bool:
    """False for a body nothing of which is drawn -- a mechanism's collider
    and no more (CR5's sprung seats, which hold a brick up off the one under
    it until it is pressed home): no camera shows it, and it hides nothing."""
    parts = body.attrs.get("parts")
    if body.shape == "compound" and parts:
        return any(part[5] != "hidden" for part in parts)
    return True


def attached_to(body: BodySpec) -> Optional[str]:
    """The body this one rides on, keeping its initial offset, if any.

    For visual-only marks: DS1's colour patch moves with its block.  Something
    that collides must never be carried this way -- a teleported collider
    pushes without being pushed back (see ``ArticulationSpec``).
    """
    parent = body.attrs.get("attached_to")
    return str(parent) if parent else None


def legible_from(body: BodySpec, pos: Vec3, quat, eye: Vec3, reach: float = 1.0) -> bool:
    """Can a mark on one face of ``body`` be read from ``eye``?

    A body with ``attrs["facing"]`` (the face's outward normal, in the world, at
    the body's initial pose) is readable only from within ``facing_cos_min`` of
    that normal; the normal turns with the body.  One with
    ``attrs["legible_within"]`` only from that close -- lettering a centimetre
    tall is not read from across the table (DS1's tags, v8).  Anything else:
    yes.  Whether something is in the way is a separate question -- the
    renderer's, on Isaac.

    ``reach``: how far this camera reads, against the lens the distances were
    set for (``legibility_reach``).

    Both engines ask this, so "seen" means the same thing on both: a camera
    grazing the face a couple of degrees behind its plane catches pixels of it
    (34 on Isaac, from DS1's start pose) but could not read a colour off them.
    """
    import math

    from ..core.geometry import quat_conj, quat_mul, rotate

    far = body.attrs.get("legible_within")
    if far is not None and math.dist(tuple(eye), tuple(pos)) > float(far) * reach:
        return False
    facing = body.attrs.get("facing")
    if not facing:
        return True
    turned = quat_mul(tuple(quat), quat_conj(body.pose.quat))
    normal = rotate(turned, tuple(float(v) for v in facing))
    to_eye = [e - p for e, p in zip(eye, pos)]
    norm = math.sqrt(sum(v * v for v in to_eye)) or 1.0
    cos = sum(n * v for n, v in zip(normal, to_eye)) / norm
    return cos >= float(body.attrs.get("facing_cos_min", 0.35))


def passes(a: BodySpec, b: BodySpec) -> bool:
    """Do these two pass through each other?  A body's ``attrs["passes"]``
    names the bodies it does not collide with -- CL4's floating water in the
    well holds up a piece of foam, and the stones drop through it (v9)."""
    return b.name in (a.attrs.get("passes") or ()) or a.name in (b.attrs.get("passes") or ())


def attachments(bodies: Sequence[BodySpec]) -> Dict[str, str]:
    """``{child: parent}`` for every attached body, checked."""
    from ..core.errors import ConfigError

    names = {b.name for b in bodies}
    out = {}
    for b in bodies:
        parent = attached_to(b)
        if parent is None:
            continue
        if parent not in names:
            raise ConfigError(f"{b.name} is attached to {parent!r}, which is not in the scene")
        if collides(b):
            raise ConfigError(f"{b.name} is attached to {parent} but collides; only "
                              "visual-only bodies (collides: False) may be carried along")
        out[b.name] = parent
    return out


@dataclass(frozen=True)
class ArticulationSpec:
    """A joint we drive or that has a spring: drawer, slider, hinge.

    Moving objects use a **position-actuated joint, never a mocap body**: mocap
    bodies are kinematic and take no force, so contact with anything they carry
    is inconsistent (design doc details §5.1).
    """

    name: str
    kind: str = "slide"                # slide | hinge (values in metres | radians)
    axis: Vec3 = (1.0, 0.0, 0.0)
    limits: Tuple[float, float] = (0.0, 0.3)
    initial: float = 0.0
    actuated: bool = True
    stiffness: float = 0.0             # > 0 gives the self-closing drawer of BC2
    damping: float = 1.0
    #: the body this joint carries.  Its ``BodySpec.pose`` is where it is at
    #: joint value 0; the joint's axis is in the world frame
    parent: Optional[str] = None
    #: (amplitude, speed, phase): the joint drives back and forth around
    #: ``initial`` by this much at this speed, starting ``phase`` (0-1) into
    #: the cycle -- DI2's travelling bowl
    oscillate: Optional[Tuple[float, float, float]] = None
    #: does what rests on the body travel with it?  A tray does; a trapdoor
    #: slides out from under what it held (the physics engines work that out
    #: themselves; the mock is told).  Only slides carry in the mock.
    carries: bool = True
    #: a hinge turns about the axis through this point, in the carried body's
    #: frame (a lid's back edge, a pendulum's pivot)
    pivot: Vec3 = (0.0, 0.0, 0.0)
    #: the body this joint stands on, when that is not the world: a body
    #: another joint carries (a catapult's arm on its turning base).  The
    #: joint's zero pose and axis are as the scene gives them at the start and
    #: move with that body from then on
    mounted_on: Optional[str] = None


def joint_program(joint: "ArticulationSpec", t: float) -> float:
    """Where an oscillating joint is told to be at time ``t``: a triangle wave,
    out to +amplitude, back through ``initial`` to -amplitude and home."""
    amplitude, speed, start = joint.oscillate
    period = 4.0 * amplitude / speed
    phase = (t / period + start) % 1.0
    wave = 4 * phase if phase < 0.25 else (2 - 4 * phase if phase < 0.75 else 4 * phase - 4)
    return joint.initial + amplitude * wave


def joint_start(joint: "ArticulationSpec") -> float:
    """Where a joint is when the episode starts: an oscillating one already
    where its program has it.  Started at ``initial``, DI2's wagon showed in
    every first picture at the middle of its track and dashed to its seeded
    place once the clock ran."""
    return joint_program(joint, 0.0) if joint.oscillate else joint.initial


@dataclass(frozen=True)
class MarkerSpec:
    """A mark drawn flat on the table: it collides with nothing, and every
    renderer draws it from the same strokes (:func:`marker_strokes`).

    ``shape``: ``cross`` (a plus sign, arms ``size[0]`` long from the centre),
    ``square`` (an outline, half its side ``size[0]``), ``rect`` (an outline,
    half sides ``size[0]`` and ``size[1]``), ``circle`` (an outline of radius
    ``size[0]``), ``segment`` (a bare stroke ``size[0]`` to either side along
    the mark's x), ``line`` (the same, with a short tick at each end) or
    ``label`` (``text`` in capital letters or digits ``size[0]`` tall, upright
    in the fixed camera).  A mark on something flat (a placemat) takes that
    thing's top as its height.
    """

    name: str
    pose: Pose = field(default_factory=Pose)
    shape: str = "cross"
    size: Vec3 = (0.02, 0.02, 0.001)
    color: str = "white"
    text: str = ""


#: Every mark is drawn with flat strokes this wide and this proud of the table.
STROKE_WIDTH = 0.006
STROKE_HEIGHT = 0.001

#: A stroke font on a 4 x 6 grid (x to the right, y up), as polylines: capital
#: letters, digits and a few signs.  A missing glyph is an error, not a blank.
_O = ((1, 0), (0, 1), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0))
_P = ((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3))
GLYPHS: Dict[str, Tuple[Tuple[Tuple[float, float], ...], ...]] = {
    "A": (((0, 0), (2, 6), (4, 0)), ((1, 3), (3, 3))),
    "B": (((0, 0), (0, 6), (3, 6), (4, 5), (4, 4), (3, 3), (0, 3)),
          ((3, 3), (4, 2), (4, 1), (3, 0), (0, 0))),
    "C": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1)),),
    "D": (((0, 0), (0, 6), (2.5, 6), (4, 4.5), (4, 1.5), (2.5, 0), (0, 0)),),
    "E": (((4, 0), (0, 0), (0, 6), (4, 6)), ((0, 3), (3, 3))),
    "F": (((0, 0), (0, 6), (4, 6)), ((0, 3), (3, 3))),
    "G": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 1), (1, 0), (3, 0), (4, 1), (4, 3), (2, 3)),),
    "H": (((0, 0), (0, 6)), ((4, 0), (4, 6)), ((0, 3), (4, 3))),
    "I": (((1, 0), (3, 0)), ((2, 0), (2, 6)), ((1, 6), (3, 6))),
    "J": (((1, 6), (4, 6)), ((3, 6), (3, 1), (2, 0), (1, 0), (0, 1))),
    "K": (((0, 0), (0, 6)), ((4, 6), (0, 2)), ((1, 3), (4, 0))),
    "L": (((0, 6), (0, 0), (4, 0)),),
    "M": (((0, 0), (0, 6), (2, 3), (4, 6), (4, 0)),),
    "N": (((0, 0), (0, 6), (4, 0), (4, 6)),),
    "O": (_O,),
    "P": (_P,),
    "Q": (_O, ((2, 2), (4, 0))),
    "R": (_P, ((2, 3), (4, 0))),
    "S": (((4, 5), (3, 6), (1, 6), (0, 5), (0, 4), (1, 3), (3, 3), (4, 2), (4, 1), (3, 0),
           (1, 0), (0, 1)),),
    "T": (((0, 6), (4, 6)), ((2, 6), (2, 0))),
    "U": (((0, 6), (0, 1), (1, 0), (3, 0), (4, 1), (4, 6)),),
    "V": (((0, 6), (2, 0), (4, 6)),),
    "W": (((0, 6), (1, 0), (2, 4), (3, 0), (4, 6)),),
    "X": (((0, 0), (4, 6)), ((0, 6), (4, 0))),
    "Y": (((0, 6), (2, 3), (4, 6)), ((2, 3), (2, 0))),
    "Z": (((0, 6), (4, 6), (0, 0), (4, 0)),),
    "0": (_O, ((0.5, 1), (3.5, 5))),
    "1": (((1, 5), (2, 6), (2, 0)), ((1, 0), (3, 0))),
    "2": (((0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (0, 0), (4, 0)),),
    "3": (((0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1.5, 3)),
          ((3, 3), (4, 2), (4, 1), (3, 0), (1, 0), (0, 1))),
    "4": (((3, 0), (3, 6), (0, 2), (4, 2)),),
    "5": (((4, 6), (0, 6), (0, 3.5), (3, 3.5), (4, 2.5), (4, 1), (3, 0), (1, 0), (0, 1)),),
    "6": (((3.5, 6), (1.5, 6), (0, 4), (0, 1), (1, 0), (3, 0), (4, 1), (4, 2.5), (3, 3.5),
           (0, 3.5)),),
    "7": (((0, 6), (4, 6), (1.5, 0)),),
    "8": (((1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (3, 3), (1, 3), (0, 2),
           (0, 1), (1, 0), (3, 0), (4, 1), (4, 2), (3, 3)),),
    "9": (((4, 3), (1, 3), (0, 4), (0, 5), (1, 6), (3, 6), (4, 5), (4, 1), (3, 0), (1, 0)),),
    "-": (((1, 3), (3, 3)),),
    "?": (((0, 5), (1, 6), (3, 6), (4, 5), (4, 4), (2, 2.5), (2, 1.5)), ((2, 0.4), (2, 0))),
    "!": (((2, 6), (2, 1.5)), ((2, 0.4), (2, 0))),
    # lower case for chemical formulas (DS6's NaCl, CuCl2), the rest upper
    "a": (((4, 4), (4, 0)), ((4, 3), (3, 4), (1, 4), (0, 3), (0, 1), (1, 0), (3, 0), (4, 1))),
    "u": (((0, 4), (0, 1), (1, 0), (3, 0), (4, 1)), ((4, 4), (4, 0))),
    "i": (((2, 0), (2, 4)), ((2, 5.2), (2, 5.8))),
    "l": (((2, 0), (2, 6)),),
    "\u2082": (((0.5, 2.2), (1.2, 2.9), (2.3, 2.9), (3, 2.2), (3, 1.7), (0.5, -0.5), (3, -0.5)),),
}


def text_strokes(text: str, height: float) -> Tuple["Segment", ...]:
    """The strokes of a line of text ``height`` tall, centred on the origin, in
    the text's own plane: (right, up), metres."""
    unit, advance = height / 6.0, 6.0
    width = advance * len(text) - 2.0
    out = []
    for k, ch in enumerate(text):
        if ch == " ":
            continue
        ch = ch if ch in GLYPHS else ch.upper()
        if ch not in GLYPHS:
            raise ValueError(f"no glyph for {ch!r} in {text!r}")
        for line in GLYPHS[ch]:
            pts = [((gx + k * advance - width / 2) * unit, (gy - 3.0) * unit) for gx, gy in line]
            out += list(zip(pts[:-1], pts[1:]))
    return tuple(out)


Segment = Tuple[Tuple[float, float], Tuple[float, float]]


def marker_strokes(marker: MarkerSpec) -> Tuple[Segment, ...]:
    """The straight strokes that draw a mark, in the mark's own frame (metres).

    Text reads upright from the fixed camera, which looks at the robot along
    -x: a letter's "up" is -x and its "right" is +y.
    """
    s = float(marker.size[0])
    if marker.shape == "cross":
        return (((-s, 0.0), (s, 0.0)), ((0.0, -s), (0.0, s)))
    if marker.shape == "line":
        tick = min(0.015, s)
        return (((-s, 0.0), (s, 0.0)), ((-s, -tick), (-s, tick)), ((s, -tick), (s, tick)))
    if marker.shape == "square":
        c = ((-s, -s), (-s, s), (s, s), (s, -s))
        return tuple((c[i], c[(i + 1) % 4]) for i in range(4))
    if marker.shape == "rect":
        sy = float(marker.size[1])
        c = ((-s, -sy), (-s, sy), (s, sy), (s, -sy))
        return tuple((c[i], c[(i + 1) % 4]) for i in range(4))
    if marker.shape == "segment":
        return (((-s, 0.0), (s, 0.0)),)
    if marker.shape == "circle":
        import math
        n = 20
        pts = [(s * math.cos(2 * math.pi * k / n), s * math.sin(2 * math.pi * k / n))
               for k in range(n)]
        return tuple((pts[k], pts[(k + 1) % n]) for k in range(n))
    if marker.shape == "label":
        try:
            strokes = text_strokes(marker.text, s)
        except ValueError as exc:
            raise ValueError(f"marker {marker.name!r}: {exc}") from None
        # text right -> +y, text up -> -x
        return tuple(tuple((-up, right) for right, up in seg) for seg in strokes)
    raise ValueError(f"marker {marker.name!r}: unknown shape {marker.shape!r}")


def marker_strokes_world(marker: MarkerSpec) -> Tuple[Segment, ...]:
    from ..core.geometry import yaw_of
    import math

    yaw = yaw_of(marker.pose.quat)
    c, sn = math.cos(yaw), math.sin(yaw)
    x0, y0 = marker.pose.pos[0], marker.pose.pos[1]
    return tuple(tuple((x0 + c * x - sn * y, y0 + sn * x + c * y) for x, y in seg)
                 for seg in marker_strokes(marker))


@dataclass(frozen=True)
class CameraSpec:
    name: str
    kind: str = "external"             # external | wrist
    attached_to: Optional[str] = None  # arm id for wrist cameras
    width: int = 512
    height: int = 512
    pose: Optional[Pose] = None
    #: also deliver a depth image (``CameraFrame.depth``): an RGB-D camera.
    #: Off unless a run asks -- perception from pixels needs it (RUIKit's
    #: ``perception: model``); nothing else reads it, and the picture is the same
    depth: bool = False


@dataclass(frozen=True)
class RobotSpec:
    name: str = "robot0"
    #: a key of ``engines.robots.MODELS`` (case-insensitive)
    model: str = "fr3"
    #: the arms a task may command; a station's other arms stay in their
    #: ready pose
    arms: Tuple[str, ...] = ("right",)
    base_pose: Pose = field(default_factory=Pose)


@dataclass(frozen=True)
class ArmStart:
    """Where an arm starts an episode, if not at its model's rest pose, and
    what it holds from the first frame (v6, 9/22: DI5's and CL1's hands start
    with their mallets).  ``holding`` names a body whose spec pose is already
    in the hand at ``pose``; the gripper starts closed on it.  Not part of the
    scene's structure: written at reset, like every pose."""

    arm: str                        # arm id, "robot0/left"
    pose: Pose                      # the tool centre point's start pose (world)
    holding: Optional[str] = None


@dataclass(frozen=True)
class SceneSpec:
    """A complete, engine-independent description of one scene."""

    name: str
    robots: Tuple[RobotSpec, ...] = (RobotSpec(),)
    bodies: Tuple[BodySpec, ...] = ()
    articulations: Tuple[ArticulationSpec, ...] = ()
    markers: Tuple[MarkerSpec, ...] = ()
    cameras: Tuple[CameraSpec, ...] = (
        CameraSpec("agentview", "external"),
        CameraSpec("wrist", "wrist", attached_to="robot0/right"),
    )
    table_height: float = 0.8
    #: catalogue name of the work surface, or None for a bare floor.  Two-arm
    #: scenes need a table whose footprint leaves room for both bases.
    table: Optional[str] = "table"
    base_env: Optional[str] = None     # backend hint, e.g. robosuite "TwoArmHandover"
    #: the points a task's fixed camera must show (world), besides the gripper
    #: where it starts: the part of the table the task uses.  Empty: the arm's
    #: whole workspace (``RobotModel.camera_frame``)
    camera_frame: Tuple[Vec3, ...] = ()
    #: arms that start somewhere else than their rest pose (``ArmStart``)
    arm_starts: Tuple[ArmStart, ...] = ()

    def arm_start(self, arm_id: str) -> Optional[ArmStart]:
        return next((s for s in self.arm_starts if s.arm == arm_id), None)

    @property
    def arm_ids(self) -> Tuple[str, ...]:
        return tuple(f"{r.name}/{a}" for r in self.robots for a in r.arms)

    def structure_key(self) -> tuple:
        """Everything that needs a new stage to change.

        Poses and colours are deliberately **not** in it: they are written at
        reset.  An engine that keeps a stage alive across episodes rebuilds only
        when this key changes -- which is why a task should randomise positions,
        colours and rules across seeds, but keep its set of bodies fixed.
        """
        return (
            tuple((r.name, r.model, r.arms, r.base_pose.pos, r.base_pose.quat)
                  for r in self.robots),
            tuple((b.name, b.shape, b.size, b.asset, b.fixed, b.category,
                   tuple(b.attrs.get("parts") or ()), b.attrs.get("angular_damping"))
                  # what a cloth is made of, and how heavy: its mesh and its
                  # particles are built with the stage
                  + ((b.attrs["cloth"], b.mass) if b.shape == "cloth" else ())
                  for b in self.bodies),
            tuple((j.name, j.kind, j.axis, j.limits, j.parent, j.stiffness, j.damping,
                   j.actuated, j.pivot, j.mounted_on)
                  for j in self.articulations),
            tuple((c.name, c.kind, c.attached_to, c.width, c.height) for c in self.cameras),
            tuple((m.name, m.shape, m.size, m.text, m.color) for m in self.markers),
            self.table, round(self.table_height, 6),
            tuple(tuple(round(v, 6) for v in p) for p in self.camera_frame),
        )

    def summary(self) -> dict:
        """A compact, JSON-ready picture of the scene, kept in every episode log.

        Enough to draw it -- footprints, colours, containers, robots -- so an
        episode can be replayed from its log alone, on any machine, long after
        the task code has moved on.
        """
        from ..core.geometry import yaw_of

        return {
            "table_height": self.table_height,
            "robots": [{"name": r.name, "base": [round(v, 4) for v in r.base_pose.pos],
                        "arms": list(r.arms)} for r in self.robots],
            "bodies": [{"name": b.name, "shape": b.shape,
                        "size": [round(v, 4) for v in b.size],
                        "pos": [round(v, 4) for v in b.pose.pos],
                        "yaw": round(yaw_of(b.pose.quat), 4),
                        "color": b.color, "category": b.category, "fixed": b.fixed,
                        "container": b.attrs.get("container") == "open",
                        "wall": b.attrs.get("wall"), "footprint": b.attrs.get("footprint"),
                        "collides": collides(b), "attached_to": attached_to(b)}
                       for b in self.bodies],
            "markers": [{"name": m.name, "shape": m.shape, "size": [round(v, 4) for v in m.size],
                         "pos": [round(v, 4) for v in m.pose.pos], "color": m.color,
                         "text": m.text, "width": STROKE_WIDTH,
                         "strokes": [[round(v, 4) for p in seg for v in p]
                                     for seg in marker_strokes_world(m)]}
                        for m in self.markers],
            "cameras": [c.name for c in self.cameras],
        }


#: A move has arrived when the tool is this close to its commanded pose.  Tight
#: on purpose: the next command usually closes the gripper, and a hand still
#: 4 mm above a 1 cm slice closes on its top edge and pushes it away.
ARRIVAL_POS_TOL = 0.002           # m
ARRIVAL_ROT_TOL = 0.02            # rad
#: A move that has come to rest this close is done as well (the executor
#: decides when it is at rest): pressing onto a stack can stop it short.
NEAR_POS_TOL = 0.005              # m
NEAR_ROT_TOL = 0.05               # rad

#: A gripper command at or above this closes the fingers on whatever is
#: between them; below it, they open to (1 - value) of the way -- on both
#: engines (the Isaac FR3 and X5 squeezed from 0.5, the mock from 0.6: a
#: command of 0.5 closed the hand on Isaac and left it open on the mock, v13)
GRIP_CLOSING = 0.5

#: fingers commanded closed hold something only while it keeps them this far
#: apart (m): closed on nothing they meet, and a closed hand pushed onto a
#: body is not holding it.  The thinnest thing any task takes is DS6's 4 mm
#: splint (bench-issue #21: a doll pushed with a closed hand stayed "held")
HELD_MIN_GAP = 0.002

#: a closing hand takes hold of a body when its tool centre point is inside the
#: body's box, give or take this much
GRASP_MARGIN = 0.015


def in_grasp(spec: BodySpec, pose: Pose, tcp: Vec3, margin: float = GRASP_MARGIN) -> bool:
    """Is a tool centre point where closing the fingers would take hold of the
    body?  Inside its box, give or take ``margin``.  The Isaac engine's
    holding test as well as the mock's (in ``engines.mock`` until v1.2, when
    the mock left the freeze's fingerprint and this, which shapes Isaac's
    episodes too, came here word for word)."""
    from ..core.geometry import quat_conj, rotate

    local = rotate(quat_conj(pose.quat), tuple(t - p for t, p in zip(tcp, pose.pos)))
    if all(abs(v) <= h + margin for v, h in zip(local, spec.size)):
        return True
    handle = spec.attrs.get("handle")         # ((centre), (half)) in the body's frame
    return bool(handle) and all(abs(v - c) <= h + margin
                                for v, c, h in zip(local, handle[0], handle[1]))


def container_floor(attrs) -> float:
    """How high an open container's floor is above its bottom -- where what
    is put in it comes to rest: ``attrs["floor"]`` where a mesh says so (CL2's
    doll bed: its mattress, 3.6 cm up, bench-issue #20), else its wall's
    thickness."""
    floor = attrs.get("floor")
    return float(floor if floor is not None else (attrs.get("wall") or 0.0))

def wrist_camera_pose(tcp: Pose, model=None) -> Pose:
    """Where the wrist camera is when the tool centre point is at ``tcp``.
    Where the camera is, not where the fingers are, decides what can be seen
    -- so a viewpoint is planned for the camera and the tool pose derived."""
    from . import robots

    return robots.wrist_camera_pose(tcp, model or robots.FR3)


def tcp_for_wrist_camera(camera: Pose, model=None) -> Pose:
    """The tool pose that puts the wrist camera at ``camera``."""
    from . import robots

    return robots.tcp_for_wrist_camera(camera, model or robots.FR3)


def arm_start_pose(scene: "SceneSpec", robot: "RobotSpec", arm: Optional[str] = None) -> Pose:
    """Where an arm's tool centre point starts, pointing straight down.
    Engine-independent for the same reason as the camera: "has the arm moved
    yet" is a task-level question."""
    from . import robots

    return robots.arm_start_pose(scene, robot, arm or robot.arms[0])


def arm_camera_model(scene: "SceneSpec", arm_id: str):
    """The robot model of the arm a wrist camera is attached to."""
    from . import robots

    name = arm_id.split("/")[0]
    robot = next((r for r in scene.robots if r.name == name), scene.robots[0])
    return robots.model_of(robot)


#: Pinhole intrinsics: every camera has the same 20.955 mm sensor (square
#: images).  Wrist cameras and the single-arm front view use a 24 mm lens,
#: about 47 degrees across; a robot model may give its fixed camera another
#: (``RobotModel.camera_focal_length``: the station's wide head camera), and
#: its wrist cameras another (``RobotModel.wrist_focal_length``: the X5's D405).
FOCAL_LENGTH = 24.0
HORIZONTAL_APERTURE = 20.955


def camera_apertures(scene: "SceneSpec", camera: "CameraSpec") -> Tuple[float, Optional[float]]:
    """(horizontal, vertical) aperture of one of a scene's cameras, in mm; the
    vertical None for square pixels.  The station's head camera is RoboDojo's
    Gemini 345Lg: 22.212 x 14.266 mm behind a 10 mm lens, 96 x 71 degrees."""
    from . import robots

    if camera.kind == "wrist" or not scene.robots:
        return HORIZONTAL_APERTURE, None
    model = robots.model_of(scene.robots[0])
    return model.camera_apertures or (HORIZONTAL_APERTURE, None)


def legibility_reach(scene: "SceneSpec", camera: "CameraSpec") -> float:
    """How far this camera reads lettering, as a fraction of the distances in
    ``attrs["legible_within"]``: those were set for a 24 mm lens, 512 pixels
    across the 20.955 mm sensor, and lettering is read by the pixels it
    covers.  The FR3's wrist camera, behind the D405's 18 mm since v17, reads
    DS1's tags from 34 cm, not 45 -- at 45 its new place, looking ahead, read
    Tom's tag from where the hand starts."""
    horizontal, _ = camera_apertures(scene, camera)
    focal_px = camera_focal_length(scene, camera) / horizontal * camera.width
    return focal_px / (FOCAL_LENGTH / HORIZONTAL_APERTURE * 512)


def camera_focal_length(scene: "SceneSpec", camera: "CameraSpec") -> float:
    """The lens of one of a scene's cameras, in mm.

    Every fixed camera on a robot uses its model's own lens.  The station's
    head camera is fixed, as RoboDojo's is: a task cannot zoom it (v3.1 tried,
    and the fit answered by pushing the camera back behind the arms)."""
    from . import robots

    if not scene.robots:
        return FOCAL_LENGTH
    if camera.kind == "wrist":
        return arm_camera_model(scene, camera.attached_to or "").wrist_focal_length
    return robots.model_of(scene.robots[0]).camera_focal_length


def head_camera_room(scene: "SceneSpec"):
    """``point -> how far either side of its axis the station's head camera
    sees at that point's range``, or None for a scene without one.

    The head camera is fixed (RoboDojo's is too), so it is the layout that has
    to fit the camera and not the other way round.  Until 9/21 nothing checked
    this and all four two-arm tasks were laid out wider than the picture: DS6
    put its wire loops at y = +-0.48 where the head sees +-0.28 at that range,
    so the thing the task is about was off the side of every observation.
    Keeping the work between the arms is the other half of copying RoboDojo."""
    from ..core.geometry import quat_conj, rotate
    from . import robots

    if not scene.robots:
        return None
    model = robots.model_of(scene.robots[0])
    if not model.ego_camera:
        return None
    pose = ego_camera_pose(scene)
    across, down = model.camera_apertures or (HORIZONTAL_APERTURE, None)
    edge = (0.5 - CAMERA_FIT_MARGIN) * across / model.camera_focal_length
    width, height = model.camera_size or (1, 1)
    # the picture sees less up and down than across: by its aspect, or by the
    # sensor's own vertical aperture
    tall = down / across if down else height / width

    def room(point: Vec3):
        """(how far off the axis the point is, how far the camera sees there),
        the vertical offset scaled to the picture's width."""
        x, y, z = rotate(quat_conj(pose.quat), tuple(p - c for p, c in zip(point, pose.pos)))
        return (max(abs(x), abs(y) / tall), edge * z) if z > 1e-6 else (1.0, -1.0)

    return room


def project(camera: Pose, point: Vec3, width: int, height: int,
            focal_length: float = FOCAL_LENGTH,
            apertures: Tuple[float, Optional[float]] = (HORIZONTAL_APERTURE, None)):
    """Pixel coordinates and depth of a world point, optical convention (the
    camera looks along its +z, image x to the right, y down).  ``apertures``:
    (horizontal, vertical) in mm, the vertical None for square pixels."""
    from ..core.geometry import quat_conj, rotate

    x, y, z = rotate(quat_conj(camera.quat), tuple(p - c for p, c in zip(point, camera.pos)))
    if z <= 1e-6:
        return None
    across, down = apertures
    fx = width * focal_length / across
    fy = height * focal_length / down if down else fx
    return width / 2 + fx * x / z, height / 2 + fy * y / z, z




def work_centre(scene: "SceneSpec") -> Tuple[float, float]:
    """The middle of the commanded arms' bases, on the table."""
    from . import robots

    points = [robots.arm_base(r, a, scene.table_height)[:2] for r in scene.robots for a in r.arms]
    if not points:
        return (0.0, 0.0)
    return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))


#: How far inside the image edge a fitted camera keeps what it must show, as a
#: fraction of the image width (24 px of 512).
CAMERA_FIT_MARGIN = 24 / 512


def fit_camera(points: Sequence[Vec3], pitch: float, margin: float = CAMERA_FIT_MARGIN,
               focal_length: float = FOCAL_LENGTH) -> Pose:
    """The closest camera looking toward -x, ``pitch`` below the horizon, that
    shows every point at least ``margin`` inside its (square) image.

    With the orientation fixed, each image edge is a linear constraint on the
    eye: writing it as ``x a + y b + z c`` (image right, image down, view),
    the left and right edges bound ``x`` by ``z`` and meet at the largest
    ``z`` they allow, likewise the top and bottom for ``y``; the closer of the
    two meeting points wins and the other axis is centred.
    """
    from ..core.geometry import camera_quat

    c = (-math.cos(pitch), 0.0, -math.sin(pitch))
    a = (0.0, 1.0, 0.0)                       # c x up, normalised: the image's right
    b = (c[1] * a[2] - c[2] * a[1], c[2] * a[0] - c[0] * a[2], c[0] * a[1] - c[1] * a[0])
    k = (0.5 - margin) * HORIZONTAL_APERTURE / focal_length

    def dot(u, v):
        return u[0] * v[0] + u[1] * v[1] + u[2] * v[2]

    left = max(dot(a, p) - k * dot(c, p) for p in points)          # x - k z >= left
    right = max(-dot(a, p) - k * dot(c, p) for p in points)        # -x - k z >= right
    top = max(dot(b, p) - k * dot(c, p) for p in points)
    bottom = max(-dot(b, p) - k * dot(c, p) for p in points)
    z = min(-(left + right), -(top + bottom)) / (2 * k)
    x = (left - right) / 2
    y = (top - bottom) / 2
    eye = tuple(x * a[i] + y * b[i] + z * c[i] for i in range(3))
    return Pose(eye, camera_quat(eye, tuple(e + d for e, d in zip(eye, c))))


def external_camera_pose(scene: "SceneSpec") -> Pose:
    """Where the external camera stands in ``scene``, in the optical convention."""
    from ..core.geometry import camera_quat
    from . import robots

    model = robots.model_of(scene.robots[0]) if scene.robots else robots.FR3
    if model.ego_camera is not None:
        return ego_camera_pose(scene)
    if model.camera_pitch is not None:
        return _fitted_camera(model.name, tuple(
            robots.arm_base(r, a, scene.table_height)[:2] for r in scene.robots for a in r.arms),
            scene.table_height, tuple(scene.camera_frame))
    eye_off, target_off = model.camera_eye, model.camera_target
    cx, cy = work_centre(scene)
    eye = (cx + eye_off[0], cy + eye_off[1], scene.table_height + eye_off[2])
    target = (cx + target_off[0], cy + target_off[1], scene.table_height + target_off[2])
    return Pose(eye, camera_quat(eye, target))


def ego_camera_pose(scene: "SceneSpec") -> Pose:
    """A head camera between a station's arms: ``RobotModel.ego_camera`` ahead
    of their bases and over the table, looking forward ``camera_pitch`` down."""
    from ..core.geometry import camera_quat
    from . import robots

    robot = scene.robots[0]
    model = robots.model_of(robot)
    bases = [robots.arm_base(robot, a, scene.table_height) for a in model.arm_mounts]
    cx = sum(b[0] for b in bases) / len(bases)
    cy = sum(b[1] for b in bases) / len(bases)
    ahead, height = model.ego_camera
    eye = (cx + ahead, cy, scene.table_height + height)
    pitch = model.camera_pitch or 0.0
    look = (math.cos(pitch), 0.0, -math.sin(pitch))
    return Pose(eye, camera_quat(eye, tuple(e + d for e, d in zip(eye, look))))


@functools.lru_cache(maxsize=256)
def _fitted_camera(model_name: str, bases: Tuple[Tuple[float, float], ...],
                   table_height: float, frame: Tuple[Vec3, ...] = ()) -> Pose:
    """Fitted to the task's own frame and the gripper where it starts, or --
    with no frame -- to the arm's whole workspace (v2: a compact task's toys
    were a few pixels across in a picture of everything the Franka reaches)."""
    from . import robots

    model = robots.robot_model(model_name)
    if frame:
        points = list(frame) + [(bx + dx, by + dy, table_height + dz)
                                for bx, by in bases for dx, dy, dz in model.camera_hand]
    else:
        points = [(bx + dx, by + dy, table_height + dz)
                  for bx, by in bases for dx, dy, dz in model.camera_frame]
    return fit_camera(points, model.camera_pitch, focal_length=model.camera_focal_length)


# --------------------------------------------------------------------------
# the engine contract
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class EngineCapabilities:
    """What a backend can actually do.  The task validator checks a task's
    requirements against this, so an unsupported task fails loudly at load
    time rather than silently scoring 0."""

    name: str
    concurrent_arms: bool = False
    springs: bool = False
    actuated_joints: bool = False
    rendering: bool = False
    mesh_assets: bool = False
    max_camera_size: int = 512
    #: Does this backend render at a fidelity a frontier model should be judged
    #: on?  The models under test are never fine-tuned on our images, so what
    #: they see has to look like the world -- a low-fidelity rasterisation
    #: measures domain gap, not ability.
    photoreal: bool = False
    #: May results from this backend appear on a leaderboard?  False for the
    #: mock engine and for development backends; the reporting layer enforces it.
    results_grade: bool = False
    #: Does the backend work out how things tip, roll and knock each other
    #: over?  The mock only drops them straight down; a toy that needs a chain
    #: reaction plays it out itself there (WK3's dominoes).
    dynamics: bool = False
    #: Can it hold a cloth (``engines.cloth``: a body whose shape is "cloth")?
    #: Isaac simulates it; the mock folds it flat when it is let go.
    cloth: bool = False
    #: Is it a physical robot (``engines.franka``)?  Then the world runs on the
    #: wall clock, whether or not a model is thinking (``advance`` waits time
    #: out, it does not step it); the pictures and the arm's readings come from
    #: real sensors, which err as they err (no simulated error is added); and
    #: nothing knows where the things on the table are -- ``state`` holds the
    #: robot alone, and an operator judges the episode.
    real: bool = False


class Engine(ABC):
    """Physics + robot + cameras.  Stateful; one instance drives one episode."""

    capabilities: EngineCapabilities

    def describe(self) -> dict:
        """What this backend actually ran with, for the episode log.

        The run config only asks; this is the answer.  A log that repeated the
        requested rates while the engine used its own would be wrong provenance.
        """
        out = {"name": self.capabilities.name}
        for key in ("sim_hz", "control_hz", "device", "render_passes", "render_settings"):
            if hasattr(self, key):
                out[key] = getattr(self, key)
        return out

    # -- lifecycle ---------------------------------------------------------
    @abstractmethod
    def load(self, scene: SceneSpec, seed: int) -> None:
        """Build ``scene``; ``seed`` fixes every randomised placement."""

    @abstractmethod
    def reset(self) -> WorldState:
        """Return to the loaded initial state and return the first snapshot."""

    @abstractmethod
    def close(self) -> None: ...

    # -- time --------------------------------------------------------------
    @property
    @abstractmethod
    def sim_time(self) -> float: ...

    @abstractmethod
    def advance(self, seconds: float) -> None:
        """Step physics forward.  The only way time passes; nothing reads the
        wall clock (design doc details §1.3)."""

    # -- observation -------------------------------------------------------
    @abstractmethod
    def state(self) -> WorldState:
        """Privileged ground truth.  Never handed to a model directly."""

    @abstractmethod
    def render(self, camera: str) -> CameraFrame: ...

    # -- robot control (servo level; queueing lives in ruibench.executor) ---
    def camera_pose(self, camera: str) -> Pose:
        """Where ``camera`` is now, in the optical convention (:func:`project`:
        it looks along its +z, image x right, y down)."""
        raise NotImplementedError(f"{type(self).__name__} does not say where its cameras are")

    @abstractmethod
    def set_arm_target(self, arm: str, pose: Pose) -> None: ...

    @abstractmethod
    def set_gripper(self, arm: str, value: float) -> None:
        """0 open, 1 closed; ``GRIP_CLOSING`` and above close on what is there."""

    @abstractmethod
    def arm_at_target(self, arm: str, pos_tol: float = ARRIVAL_POS_TOL,
                      rot_tol: float = ARRIVAL_ROT_TOL) -> bool:
        """Has the end effector reached its commanded pose (or, after
        :meth:`set_arm_joints`, the arm its joints)?"""

    def set_arm_joints(self, arm: str, joints: Sequence[float]) -> Pose:
        """Drive the arm's joints to ``joints`` (radians, base to wrist, inside
        their limits; ``engines.kinematics``), each along its own way, and say
        where that puts the tool centre point (v1.17)."""
        raise NotImplementedError(f"{type(self).__name__} cannot drive an arm's joints")

    def arm_joints(self, arm: str) -> Optional[Tuple[float, ...]]:
        """The arm's joint readings (radians, base to wrist), or None."""
        return self.state().arm(arm).joints

    def object_masks(self, camera: str) -> Dict[str, object]:
        """Each body's pixels in what ``camera`` shows now (bool HxW), where
        the engine segments its pictures; ``{}`` where it does not.  Ground
        truth: only the privileged control condition reads it (v1.17)."""
        return {}

    def gripper_settled(self, arm: str) -> bool:
        """Has the last gripper command finished?

        Kept apart from :meth:`arm_at_target` on purpose: a gripper closed on an
        object keeps twitching against it while the arm moves, and a move must
        not wait for that to stop.
        """
        return True

    def stop_arm(self, arm: str) -> None:
        """Cancel the arm's current motion by holding its present pose."""
        self.set_arm_target(arm, self.state().arm(arm).eef_pose)

    # -- world manipulation, for the event scheduler ------------------------
    @abstractmethod
    def set_articulation_target(self, joint: str, value: float) -> None: ...

    def set_joint_drive(self, joint: str, stiffness: float, damping: float) -> None:
        """Change how hard a joint is held: a pendulum held up by its spring is
        let go by taking the spring away."""
        raise NotImplementedError(f"{type(self).__name__} cannot change a joint's drive")

    def launch(self, body: str, velocity: Vec3) -> None:
        """Set a free body flying at ``velocity`` (m/s) -- what a toy's spring
        does to the ball in it: a catapult's throw."""
        raise NotImplementedError(f"{type(self).__name__} cannot launch a body")

    def teleport(self, body: str, pose: Pose) -> None:
        """Hard-set a body pose.  Only legal for smoke tests: it violates
        velocity/constraint consistency and must never touch a body that is
        grasped or in contact (design doc details §5.1)."""
        raise NotImplementedError(f"{type(self).__name__} does not support teleport")

    # -- introspection -----------------------------------------------------
    @abstractmethod
    def visible_objects(self, camera: str) -> Tuple[str, ...]:
        """Objects with an unoccluded line of sight from ``camera``.  Backs the
        visibility predicates of the Active Perception suite."""
