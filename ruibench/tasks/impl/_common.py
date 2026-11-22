"""Shared scene and policy pieces for the task implementations.

Kept small on purpose: a task module should read as a description of *that*
task, not as a call into a framework.  What lives here is only what three or
more tasks genuinely share.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing import List, Optional, Sequence, Tuple

from ...core.rng import SeedBundle
from ...core.types import Pose, Vec3
from ...engines.base import ArmStart, BodySpec, CameraSpec, MarkerSpec, RobotSpec, SceneSpec

TABLE = 0.80
CUBE = (0.022, 0.022, 0.022)

#: one robot, one table, two cameras -- the base every single-arm task uses
CAMERAS = (CameraSpec("agentview", "external"),
           CameraSpec("wrist", "wrist", attached_to="robot0/right"))
ARM = "robot0/right"

#: Where the robot base stands, relative to the work surface.  Objects start in
#: x in [0, 0.35] and the bins' centres are 0.63 m out, so a base at x = -0.35
#: keeps every grasp and every drop inside the Franka's comfortable reach
#: (about 0.85 m), and close to the 0.5 m Isaac Lab's own Franka tasks use.
#: Further back and the arm works at the edge of its workspace, where IK takes
#: odd routes.
BASE_X = -0.35


#: The two-arm station -- since v4 (9/22) RoboDojo's: two ARX X5 arms 0.6 m
#: apart standing on the table (``engines.robots.ARX5``), the right one at
#: y = -0.3 -- runs the bimanual tasks.  Until then two Pipers on 20 cm
#: pedestals (a model since removed, v16), which hid much of the bench.
STATION_X = -0.30
TWO_ARMS = RobotSpec("robot0", "arx5", ("left", "right"), Pose((STATION_X, 0.0, 0.0)))
RIGHT, LEFT = "robot0/right", "robot0/left"
#: Since v2 (2026-09-18) every single-arm task runs on one Franka Research 3,
#: centred at the table's back edge (``engines.robots.FR3``).  It stands
#: ``FR3_BACK`` behind the station's arm line, so a v1 layout -- written from
#: the Piper's base, 12 to 40 cm ahead of it -- lands 30 to 58 cm in front of
#: the FR3, where it reaches comfortably, and only moves sideways to the middle.
FR3_BACK = 0.18
FR3_ONE = RobotSpec("robot0", "fr3", ("right",), Pose((STATION_X - FR3_BACK, 0.0, 0.0)))
#: the single-arm robot, for tasks that need its base
ONE_ARM = FR3_ONE
ONE_ARM_CAMERAS = (CameraSpec("agentview", "external"),
                   CameraSpec("wrist", "wrist", attached_to=RIGHT))
TWO_ARM_CAMERAS = (CameraSpec("agentview", "external"),
                   CameraSpec("wrist_right", "wrist", attached_to=RIGHT),
                   CameraSpec("wrist_left", "wrist", attached_to=LEFT))


def ahead(dx: float, dy: float = 0.0, arm: Optional[str] = None) -> Tuple[float, float]:
    """The table point ``dx`` in front of the arm line and ``dy`` to the
    robot's left (+y).  Layouts are written this way because what an arm can
    reach is measured from its base.  With ``arm`` (the bimanual suite): from
    that arm's base.  Without (single-arm tasks): from the middle of the
    table, on the line the station's arms stand on, ``FR3_BACK`` ahead of the FR3."""
    from ...engines.robots import model_of

    if arm is None:
        return (STATION_X + dx, dy)
    mx, my, _ = model_of(TWO_ARMS).arm_mounts[arm]
    return (STATION_X + mx + dx, my + dy)


def rest_pose(arm: str):
    """Where a station arm rests: folded low beside its base, as RoboDojo
    keeps an arm that is not working (``engines.robots.ARX5``)."""
    from types import SimpleNamespace

    from ...engines.robots import arm_start_pose

    return arm_start_pose(SimpleNamespace(table_height=TABLE), TWO_ARMS, arm.split("/")[1])


#: a hand lower than this over the table is lifted to it, straight up, before
#: it goes home: the way home keeps the hand no lower than where it starts
LOW_HAND = 0.06


def to_rest(world, arms: Sequence[str], clear: float = LOW_HAND, start=None) -> list:
    """Moves that take ``arms`` back to rest: one move home, which the Isaac
    engine takes in joint space along one smooth path -- still at its ends,
    past the model's ``home_via`` posture for an arm reaching far out
    (``engines/isaac/engine.HOME_VIA_MARGIN``) -- the hand kept no lower than
    where it starts; a hand lower than ``clear`` over the table is lifted to
    it first.  v5 went straight up 14 cm first, v10 then back over the table
    with the fingers down: three moves, each stopping at its end, the first
    out of the wrist's reach at WK4's far places (v11, the advisor: "still not
    smooth").  ``start``: arm -> the pose the hand will be at by then, when
    earlier moves in the same batch take it elsewhere."""
    from ...interfaces.commands import MoveTo

    out = []
    for a in arms:
        pose = (start or {}).get(a) or world.arm(a).eef_pose
        x, y, z = pose.pos
        if z < TABLE + clear:
            out.append(MoveTo(a, Pose((x, y, TABLE + clear), pose.quat)))
    return out + [MoveTo(a, rest_pose(a)) for a in arms]


def put_back(world, arms: Sequence[str], grips, carry_z: float) -> list:
    """What each hand holds, set down where it was taken up (``grips``: arm ->
    the grasp pose it was taken with), and the arms back to rest."""
    from ...interfaces.commands import MoveTo, SetGripper, Wait

    out = []
    for a in arms:
        x, y, _ = world.arm(a).eef_pose.pos
        g = grips[a]
        out += [MoveTo(a, Pose((x, y, carry_z), g.quat)), MoveTo(a, Pose((g.pos[0], g.pos[1], carry_z), g.quat)),
                MoveTo(a, Pose((g.pos[0], g.pos[1], g.pos[2] + 0.003), g.quat))]
    out += [SetGripper(a, 0.0) for a in arms] + [Wait(0.5)]
    return out + to_rest(world, arms, start={a: grips[a] for a in arms})


@dataclass(frozen=True)
class Frame:
    """The part of the table a single-arm task uses, which its fixed camera
    shows (with the gripper where it starts): x ahead of the station line, y
    across, and how high things stand.  The same at both levels and in every
    seed -- ``validate`` checks every body, mark and slide stays inside it."""

    x: Tuple[float, float]
    y: Tuple[float, float]
    top: float = 0.10

    def corners(self) -> Tuple[Tuple[float, float, float], ...]:
        return tuple((STATION_X + x, y, TABLE + z)
                     for x in self.x for y in self.y for z in (0.0, self.top))


def station_scene(name: str, bodies: Sequence[BodySpec], markers: Sequence[MarkerSpec] = (),
                  two_arms: bool = False, cameras: Optional[Sequence[CameraSpec]] = None,
                  articulations: Sequence = (), frame: Optional[Frame] = None,
                  arm_starts: Sequence[ArmStart] = ()) -> SceneSpec:
    """A scene at the station table: the ARX pair for two arms (their head
    camera), the FR3 for one (a front camera on ``frame``, the task's part of
    the table).  ``arm_starts``: arms that start elsewhere than at rest,
    perhaps holding something (``engines.base.ArmStart``)."""
    robot = TWO_ARMS if two_arms else FR3_ONE
    cams = cameras if cameras is not None else (TWO_ARM_CAMERAS if two_arms else ONE_ARM_CAMERAS)
    return SceneSpec(name=name, robots=(robot,), bodies=tuple(bodies), markers=tuple(markers),
                     cameras=tuple(cams), articulations=tuple(articulations),
                     table_height=TABLE, table="station_table",
                     camera_frame=frame.corners() if frame else (),
                     arm_starts=tuple(arm_starts))


def table_scene(name: str, bodies: Sequence[BodySpec],
                markers: Sequence[MarkerSpec] = (),
                robots: Sequence[RobotSpec] = (FR3_ONE,),
                cameras: Sequence[CameraSpec] = CAMERAS) -> SceneSpec:
    return SceneSpec(name=name, robots=tuple(robots), bodies=tuple(bodies),
                     markers=tuple(markers), cameras=tuple(cameras), table_height=TABLE)


def cube(name: str, xy: Tuple[float, float], color: str, size: Vec3 = CUBE,
         **attrs) -> BodySpec:
    return BodySpec(name, "box", size, Pose((xy[0], xy[1], TABLE + size[2])),
                    category="block", color=color, mass=0.04, attrs=attrs)


#: Where the two sorting bins stand, and the band between them that objects
#: are laid out in.  The spacing is set by the hand, not the objects: the Panda
#: hand is 20 cm long along the direction its fingers open (y), so grasping an
#: object at the edge of the band puts the end of the hand 16.5 cm out, and
#: the bins' inner walls stand at 20 cm.  With them at 14 cm every descent near
#: a bin stopped on its wall (`ruibench.tasks.clearance` checks this in CI).
BIN_XY = ((0.20, -0.30), (0.20, 0.30))
#: Two rows 11 cm apart: an open finger reaches 6.6 cm from the grasp centre,
#: so a neighbouring 4.4 cm cube needs 9.3 cm with margin -- even after jitter.
WORK_BAND = ((0.00, 0.30), (-0.11, 0.11))
BIN_YAW = (0.70710678, 0.0, 0.0, 0.70710678)        # long side along x


#: Half extents of a sorting bin: 20 x 30 cm, 8 cm deep.  Deeper hid what was
#: inside from the front camera -- a 14 cm front wall covered a cube at the
#: bin's centre -- and a sorting task the model cannot check is a different task.
BIN_SIZE = (0.10, 0.15, 0.04)


def basket(name: str, xy: Tuple[float, float], color: str) -> BodySpec:
    """An open bin in ``color``: things released above it come to rest inside.

    Built from five boxes on the Isaac backend (see the catalogue's primitives
    for why not Isaac's KLT bin mesh), a box of the same size on the mock engine.
    """
    return BodySpec(name, "bin", BIN_SIZE, Pose((xy[0], xy[1], TABLE + BIN_SIZE[2]), BIN_YAW),
                    category="basket", color=color, fixed=True,
                    attrs={"container": "open", "footprint": "rect", "wall": 0.01})


def neutral_bin(name: str, xy: Tuple[float, float], color: str = "white") -> BodySpec:
    """A sorting bin that says nothing: both of a pair look alike unless a
    task gives them colours on purpose."""
    b = basket(name, xy, color)
    return BodySpec(b.name, b.shape, b.size, b.pose, category="bin", color=color,
                    fixed=True, asset=b.asset, attrs=b.attrs)


#: the In-Context suite's objects: a cube or a cylinder, in some colour
CYLINDER = (0.02, 0.02, 0.025)


def shaped(name: str, xy: Tuple[float, float], color: str, shape: str) -> BodySpec:
    if shape == "cube":
        return BodySpec(name, "box", CUBE, Pose((xy[0], xy[1], TABLE + CUBE[2])),
                        category="block", color=color, mass=0.04, attrs={"shape": shape})
    return BodySpec(name, "cylinder", CYLINDER, Pose((xy[0], xy[1], TABLE + CYLINDER[2])),
                    category="cylinder", color=color, mass=0.04, attrs={"shape": shape})


def grid_layout(seeds: SeedBundle, label: str, n: int,
                band=WORK_BAND, jitter: float = 0.006) -> List[Tuple[float, float]]:
    """``n`` positions on a jittered grid inside ``band``, in seed-shuffled order.

    Random placement with a minimum gap stops finding room well before the band
    is full; a grid always has room, and the jitter and shuffle keep layouts
    from being identical across seeds.
    """
    import math

    (x0, x1), (y0, y1) = band
    cols = max(1, math.ceil(math.sqrt(n * (x1 - x0) / (y1 - y0))))
    rows = math.ceil(n / cols)
    cells = [(x0 + (x1 - x0) * (c + 0.5) / cols, y0 + (y1 - y0) * (r + 0.5) / rows)
             for r in range(rows) for c in range(cols)]
    rng = seeds.stream(f"grid/{label}")
    rng.shuffle(cells)
    return [(x + rng.uniform(-jitter, jitter), y + rng.uniform(-jitter, jitter))
            for x, y in cells[:n]]


def asset_body(name: str, asset: str, xy: Tuple[float, float], *,
               fallback_size: Vec3, color: Optional[str] = None, mass: float = 0.1,
               fixed: bool = False, category: Optional[str] = None, yaw: float = 0.0,
               on: float = TABLE, **attrs) -> BodySpec:
    """A catalogue mesh standing on the table -- or on something lying on it,
    whose top is at ``on``.

    Its half extents come from the **measured** geometry in the asset lock file
    (upright frame), so the predicates that use ``size`` agree with the mesh the
    camera shows.  ``fallback_size`` is used until the geometry check has run,
    and by the mock engine, which has no meshes.
    """
    from ...engines.isaac.assets import get

    entry = get(asset)
    size = tuple(entry.half_extents) if entry.half_extents else fallback_size
    shape = "cylinder" if entry.footprint == "round" else "box"
    import math

    quat = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    return BodySpec(name, shape, size, Pose((xy[0], xy[1], on + size[2]), quat),
                    category=category or entry.category, color=color, mass=mass,
                    fixed=fixed, asset=asset, attrs={"footprint": entry.footprint, **attrs})
