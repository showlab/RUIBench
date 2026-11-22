"""DS1 "whose present?" -- Discovery.

Knob: how many presents, and where their names are (three, the tags on top
-> five, one of them Tim's, the names on labels underneath).

Wrapped presents in a row -- 6.4 cm boxes in patterned wrapping paper, a
satin ribbon round each both ways, a bow on the corner where the ribbons
cross -- and a red mat.  "Put Tom's present on the red mat."  Nothing in
the first picture says whose present is whose.

Easy (Tom, Amy and Leo): a gift tag lies on top of each present beside its
bow, the name in a script hand -- too small to read from the fixed camera
across the table, and out of the wrist camera's picture where the arm
starts.  With the hand over the row, pointing down as it hangs, the wrist
camera reads them all.

Hard (Tom, Tim, Amy, Leo and Mia, in a seed-drawn order): the names are on
labels stuck underneath.  A present has to be picked up and turned, its
bottom to the fixed camera, to be read -- and put back if it is not Tom's.
Tim is there to be misread.

v6 (2026-09-22): the cards faced the robot at both levels -- the easy
level's had faced the fixed camera and could be read from the first
picture.  v7 (9/23): one look at the row from its front right, not a tour.
v8 (9/24), the advisor: the presents did not look like presents (glossy
blocks, a ribbon and two balls for a bow); the easy level should need no
turning or tipping of the hand, the natural downward wrist view seeing the
names; at the hard level the label is stuck underneath -- and the labels
should be prettier.  So: wrapping paper, satin and a bow; printed tags
(``engines/prints.py``); the easy tags on top and the hard labels
underneath.  A name is read, as a real one is, from close enough: the tags
within 45 cm (``legible_within``), the labels, printed large, within 1.6 m.

The reference reads a name only once a camera can see it (never from where
it is), so a plan made from the first picture can only guess: the
``first_frame_only`` control is judged against chance (one in three, one in
five).
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.geometry import compose, quat_mul, rotate
from ...core.rng import SeedBundle
from ...core.types import IDENTITY_QUAT, Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec
from ...engines.prints import WRAPS as PAPERS
from ...engines.robots import model_of
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...policies.scripted import turned_down
from ...scoring import AtRest, Custom, Finally, Held, Milestone, On, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _toys as T
from ._common import ONE_ARM, RIGHT, TABLE, Frame, ahead, station_scene

OWNER = "Tom"
#: whose presents there are, per level: the hard level adds Tim, who looks
#: like Tom from afar, and Mia
NAMES = {"easy": ("Tom", "Amy", "Leo"), "hard": ("Tom", "Tim", "Amy", "Leo", "Mia")}
PRESENTS = {level: tuple(f"present{i}" for i in range(len(names))) for level, names in NAMES.items()}
#: the presents' papers (``engines/prints.py``) and satin ribbons
WRAPS = (("red_dots", "metal:gold"), ("blue_stars", "plastic:white"),
         ("green_stripes", "plastic:red"), ("kraft_hearts", "plastic:red"),
         ("purple_snow", "metal:aluminium"))
#: the name tags (easy) and labels (hard), one body each, carried with its present
CARDS = {n: f"card_{n.lower()}" for n in NAMES["hard"]}
MAT = "red_mat"
#: the row, 11 cm apart (an open finger beside a 6.4 cm present clears the
#: next one), centred, 33 cm out: from where the arm starts the wrist
#: camera's picture ends short of it
ROW_X, ROW_PITCH = 0.33, 0.11
MAT_AT, MAT_HALF = (0.16, -0.25), (0.05, 0.05)

# -- a present: a 6.4 cm box (the Franka's fingers open to 8 cm) ---------------------
BOX = 0.032
BOX_MASS = 0.05
#: the ribbons: 9 mm satin, crossing 2.2 cm from the middle towards the far
#: right corner -- the bow there -- leaving the quarter of the top (and of the
#: bottom) opposite free for the name
RIB, RIB_T = 0.0045, 0.0004
CROSS = (0.022, -0.022)
FREE = ((-BOX + CROSS[0] - RIB) / 2, (CROSS[1] + RIB + BOX) / 2)
#: the tag on top (a script hand, 1 cm tall), in the quarter the ribbons
#: leave free; the label underneath across the whole bottom, over the ribbon
#: (bold capitals 1.7 cm tall)
TAG_WIDTH, LABEL_WIDTH = 0.046, 0.058
#: read from close enough, and from within 57 degrees of square on: the tag
#: by the wrist camera over the row, the label by the fixed camera from the
#: showing pose (53 cm) -- neither from across the table
TAG_READ = {"facing_cos_min": 0.55, "legible_within": 0.45}
LABEL_READ = {"facing_cos_min": 0.55, "legible_within": 0.75}
#: held 12 mm above its middle, across the row (the fingers on its left and
#: right faces: the fixed camera, in front, sees it in the hand)
GRASP_HEIGHT = 0.012

WRIST, FIXED = "wrist", "agentview"
MODEL = model_of(ONE_ARM)
#: the easy level's look: the wrist camera this high over the tops of the
#: presents, the hand pointing down as it hangs
#: v17: 25 cm, not 30 -- the FR3's wrist camera, behind a wider lens, reads
#: the tags from 34 cm (``base.legibility_reach``), and from 30 cm up the row's
#: end tags were 34-35 cm away
LOOK_HEIGHT = 0.25
#: short of the half turn an upright picture asks for, so that the hand turns
#: the way its wrist has room: asked for 180 degrees (or -165) it turned the
#: other way, its wrist ran out of travel at -127 and the hand stopped 7-9 mm
#: short of the look, 3 s a seed (Isaac, v16 -- the camera on the hand's robot
#: side wants the half turn; on the other side it wanted none).  The tags lean
#: 10 degrees in the picture
LOOK_TURN_SHORT = math.radians(10)
#: the hard level's showing: the present lifted to here, in the middle of
#: the table and out towards the fixed camera (Isaac's picture of the label
#: from 27 cm out, 68 cm from the camera, was too small to read), and turned
#: bottom-first towards it: the hand turned a quarter about the vertical,
#: then tipped a quarter turn forward -- each turn square to the line
#: between the fingertips, so the present does not swing round it
SHOW_X, SHOW_Y, SHOW_Z = 0.42, 0.0, 0.30
SHOW_TIP = math.pi / 2
SHOW_YAW = math.pi / 2
#: the label underneath, turned so that it reads upright once shown
#: (its top to the present's -y, its right to +x: a half turn about x - y)
LABEL_TURN = (0.0, math.sqrt(0.5), -math.sqrt(0.5), 0.0)
#: a present is carried this high: clear over the others (lifted less, it
#: was carried down across its neighbour and knocked it, Isaac 9/24)
CARRY_Z = TABLE + 0.20

#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.075, 0.375), (-0.325, 0.275), 0.07)


def row_y(level: str) -> Tuple[float, ...]:
    n = len(NAMES[level])
    return tuple((k - (n - 1) / 2) * ROW_PITCH for k in range(n))


def owners_for(seed: int, level: str = "hard") -> List[str]:
    """Whose present stands where.  Tom's is at every place once in any n
    consecutive seeds (n presents), in a seed-drawn order -- uniform by
    construction, so a blind guess is right exactly one time in n."""
    names = NAMES[level]
    group, place = divmod(seed, len(names))
    order = list(range(len(names)))
    SeedBundle(group).stream("tom").shuffle(order)
    others = [n for n in names if n != OWNER]
    SeedBundle(seed).stream("others").shuffle(others)
    out = others[:]
    out.insert(order[place], OWNER)
    return out


def present(name: str, xy, paper: str, ribbon: str, level: str) -> BodySpec:
    """A box in wrapping paper, a satin ribbon round it both ways and a bow
    where they cross.  Only the box collides."""
    h = BOX
    ground = tuple(c / 255.0 for c in PAPERS[paper][0])
    parts = [P.box((0, 0, 0), (h, h, h), ground)]
    o = h + 0.0002
    faces = (((0, 0, o), IDENTITY_QUAT), ((0, 0, -o), P.turn_x(math.pi)),
             ((o, 0, 0), P.turn_y(math.pi / 2)), ((-o, 0, 0), P.turn_y(-math.pi / 2)),
             ((0, o, 0), P.turn_x(-math.pi / 2)), ((0, -o, 0), P.turn_x(math.pi / 2)))
    parts += [P.printed(c, 2 * h, f"wrap:{paper}", q) for c, q in faces]
    cx, cy = CROSS
    r, t, r2 = h + 0.0006, RIB_T, h + 0.0010
    parts += [P.box((0, cy, s * r), (r, RIB, t), ribbon, collides=False) for s in (1, -1)]
    parts += [P.box((s * r, cy, 0), (t, RIB, r), ribbon, collides=False) for s in (1, -1)]
    parts += [P.box((cx, 0, s * r2), (RIB, r2, t), ribbon, collides=False) for s in (1, -1)]
    parts += [P.box((cx, s * r2, 0), (RIB, t, r2), ribbon, collides=False) for s in (1, -1)]
    parts += bow((cx, cy, h + 0.0014), ribbon)
    return BodySpec(name, "compound", (h, h, h), Pose((xy[0], xy[1], TABLE + h)),
                    category="present", color=ground, mass=BOX_MASS,
                    attrs={"parts": tuple(parts), "footprint": "rect", "grasp_height": GRASP_HEIGHT,
                           "grasp_yaw": 0.0, "paper": paper})


def bow(at, ribbon: str) -> List[tuple]:
    """Two loops along the box's diagonal, rising outwards, a knot, and two
    short tails to the edges by the corner."""
    x, y, z = at
    along = math.pi / 4                         # the loops, across the corner's diagonal
    out = []
    for s in (1, -1):
        d = 0.0085
        c = (x + s * d * math.cos(along), y + s * d * math.sin(along), z + 0.0048)
        rise = quat_mul(P.turn_z(along), P.turn_y(-s * math.radians(24)))
        out.append(P.ellipsoid(c, (0.0105, 0.0042, 0.0058), ribbon, rise))
    out.append(P.ellipsoid((x, y, z + 0.0038), (0.0046, 0.0046, 0.0040), ribbon))
    for yaw in (math.radians(-12), math.radians(-78)):
        length = 0.0105
        c = (x + length / 2 * math.cos(yaw), y + length / 2 * math.sin(yaw), z + 0.0004)
        out.append(P.box(c, (length / 2, 0.0021, 0.0003), ribbon, P.turn_z(yaw), collides=False))
    return out


def name_tag(owner: str, box: BodySpec, level: str) -> BodySpec:
    """``owner``'s name, carried with ``box``: a gift tag on top beside the
    bow (easy) or a label stuck underneath (hard), in the quarter the ribbons
    leave free.  The tag's lettering reads from the robot's side; the label's
    reads upright to the fixed camera once the box is turned bottom-first
    towards it (``SHOW_TIP``)."""
    x, y, z = box.pose.pos
    fx, fy = FREE
    if level == "easy":
        part = P.printed((0, 0, 0), TAG_WIDTH, f"tag:{owner}")
        at, facing, read = (x + fx, y + fy, z + BOX + 0.0009), (0.0, 0.0, 1.0), TAG_READ
    else:
        part = P.printed((0, 0, 0), LABEL_WIDTH, f"sticker:{owner}", LABEL_TURN)
        at, facing, read = (x, y, z - BOX - 0.0014), (0.0, 0.0, -1.0), LABEL_READ
    hx, hy, _ = part[2]
    return BodySpec(CARDS[owner], "compound", (hx, hy, 0.0005), Pose(at),
                    category="gift tag" if level == "easy" else "name label", color="white",
                    attrs={"parts": (part,), "collides": False, "graspable": False,
                           "attached_to": box.name, "facing": facing, **read, "text": owner})


def build_scene(level: str, seed: int):
    presents = [present(name, ahead(ROW_X, y), paper, ribbon, level)
                for name, y, (paper, ribbon) in zip(PRESENTS[level], row_y(level), WRAPS)]
    # the names in a fixed order: which present each is on changes, the set
    # of bodies does not
    on = dict(zip(owners_for(seed, level), presents))
    tags = [name_tag(owner, on[owner], level) for owner in NAMES[level]]
    mat = T.mat(MAT, ahead(*MAT_AT), MAT_HALF, "fabric:red")
    return station_scene(f"ds1_whose_present_{level}", presents + tags + [mat], frame=FRAME)


def look_down_pose() -> Pose:
    """The hand over the row, pointing down as it hangs -- turned about the
    vertical so the tags' lettering stands upright in the wrist camera's
    picture -- and placed so that the picture is centred on the row."""
    best = None
    for yaw in (0.0, math.pi / 2, math.pi, -math.pi / 2):
        cam = compose(Pose((0.0, 0.0, 0.0), turned_down(yaw)), MODEL.camera_from_tcp)
        up = rotate(cam.quat, (0.0, -1.0, 0.0))          # the picture's up, in the world
        if best is None or up[0] > best[0]:
            best = (up[0], yaw, cam)
    _, yaw, cam = best
    if abs(abs(yaw) - math.pi) < 1e-9:
        yaw = math.pi - LOOK_TURN_SHORT
        cam = compose(Pose((0.0, 0.0, 0.0), turned_down(yaw)), MODEL.camera_from_tcp)
    look = rotate(cam.quat, (0.0, 0.0, 1.0))
    tops = TABLE + 2 * BOX
    s = LOOK_HEIGHT / -look[2]                       # along the view, down to the tops
    eye = (ahead(ROW_X)[0] - s * look[0], -s * look[1], tops + LOOK_HEIGHT)
    return Pose(tuple(e - c for e, c in zip(eye, cam.pos)), turned_down(yaw))


def show_pose() -> Pose:
    """The hand holding a present bottom-first to the fixed camera: turned a
    quarter about the vertical, then tipped a quarter turn forward."""
    q = quat_mul(P.turn_y(-SHOW_TIP), turned_down(SHOW_YAW))
    return Pose((ahead(SHOW_X)[0], SHOW_Y, TABLE + SHOW_Z), q)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DS1")
    scene = build_scene(level, seed)
    owners = owners_for(seed, level)
    presents = PRESENTS[level]
    mine = presents[owners.index(OWNER)]
    others = [p for p in presents if p != mine]
    # at rest (v2.3): a present still sliding or bouncing over the mat counted
    placed = Custom(lambda w: On(mine, MAT).holds(w) and AtRest(mine).holds(w)
                    and not any(On(p, MAT).holds(w) for p in others),
                    f"{OWNER}'s present at rest on the mat, and no other")
    milestones = (
        Milestone("card_read", Custom(lambda w: w.obj(CARDS[OWNER]).seen(),
                                      f"{OWNER}'s name read by a camera")),
        Milestone("present_taken", Held(mine)),
        Milestone("on_the_mat", Custom(lambda w: On(mine, MAT).holds(w), "it is on the mat")),
        Milestone("only_it", placed),
    )
    if level == "easy":
        factory = lambda inst: LookDownPolicy(look_down_pose())
    else:
        factory = lambda inst, order=presents: TurnOverPolicy(order)
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction=f"Put {OWNER}'s present on the red mat.",
        scorer=Scorer(task_id="DS1", level=level, success=Finally(placed), milestones=milestones),
        milestones=milestones,
        reference_factory=factory,
        metadata={"knob": spec.knob.field,
                  "knob_value": len(presents),
                  # hidden from the agent; here so the matched control can say it
                  "owners": dict(zip(presents, owners)), "target": mine,
                  "needs_visibility": True, "max_model_calls": 60 if level == "easy" else 120},
    )


def read(world: WorldState, cameras=(WRIST, FIXED)) -> Optional[str]:
    """Whose present Tom's is, if a camera can read his name now."""
    card = world.obj(CARDS[OWNER])
    if any(card.seen(c) for c in cameras):
        return str(card.attrs["attached_to"])
    return None


class _Carrier(ScriptedPolicy):
    """Takes a present up to ``CARRY_Z``, over where it goes, and down."""

    arm = RIGHT

    def _pick_up(self, world: WorldState, name: str) -> List[Command]:
        obj = world.obj(name)
        x, y, z = obj.pose.pos
        q = turned_down(float(obj.attrs.get("grasp_yaw", 0.0)))
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose((x, y, CARRY_Z), q)),
                MoveTo(self.arm, Pose((x, y, z + GRASP_HEIGHT), q)), SetGripper(self.arm, 1.0),
                Wait(0.4), MoveTo(self.arm, Pose((x, y, CARRY_Z), q))]

    def _set_down(self, world: WorldState, name: str, xy, on: float = TABLE) -> List[Command]:
        """From over wherever the hand is, carried high, then down at ``xy``
        onto a surface at height ``on``."""
        q = turned_down(float(world.obj(name).attrs.get("grasp_yaw", 0.0)))
        x, y = xy
        down = on + BOX + GRASP_HEIGHT + 0.006
        return [MoveTo(self.arm, Pose((x, y, CARRY_Z), q)), MoveTo(self.arm, Pose((x, y, down), q)),
                SetGripper(self.arm, 0.0), Wait(0.5), MoveTo(self.arm, Pose((x, y, CARRY_Z), q))]

    def _to_mat(self, world: WorldState, name: str) -> List[Command]:
        mat = world.obj(MAT)
        return (self._pick_up(world, name)
                + self._set_down(world, name, mat.pose.pos[:2], mat.pose.pos[2] + mat.size[2]))


class LookDownPolicy(_Carrier):
    """The easy level: the hand over the row, pointing down, reads the tags;
    then Tom's present onto the mat."""

    name = "whose_present"

    def __init__(self, view: Pose, arm: str = RIGHT) -> None:
        self.view = view
        self.arm = arm
        self.reset()

    def reset(self) -> None:
        self._owner: Optional[str] = None
        self._looked = False
        self._moved = False

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._owner is None:
            self._owner = read(world)
        if self._owner is None:
            if self._looked:
                return [Done("could not read the tags")]
            self._looked = True
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, self.view), Wait(0.8)]
        if self._moved:
            return [Done("Tom's present is on the mat")]
        self._moved = True
        return self._to_mat(world, self._owner)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """From the first picture: read a tag if one shows, else guess the
        first present -- right one time in three."""
        return self._to_mat(world, read(world) or "present0") + [Done("planned")]


class TurnOverPolicy(_Carrier):
    """The hard level: one present at a time, nearest the mat first -- pick
    it up across its front and back, turn it bottom-first to the fixed
    camera, read the label; Tom's goes onto the mat, any other back to its
    place."""

    name = "whose_present"

    def __init__(self, order: Sequence[str], arm: str = RIGHT) -> None:
        self.order = list(order)
        self.arm = arm
        self.reset()

    def reset(self) -> None:
        self._queue: Optional[List[str]] = None
        self._home: Dict[str, Tuple[float, float]] = {}
        self._showing: Optional[str] = None
        self._owner: Optional[str] = None
        self._moved = False

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._queue is None:
            self._home = {n: tuple(world.obj(n).pose.pos[:2]) for n in self.order}
            self._queue = sorted(self.order, key=lambda n: self._home[n][1])   # the mat is on the right
        if self._owner is None:
            self._owner = read(world)
        if self._showing is not None:
            shown, self._showing = self._showing, None
            hand = world.arm(self.arm).eef_pose.pos
            # untipped, then turned back: square to the fingertips' line, as on the way out
            upright = [MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), turned_down(SHOW_YAW))),
                       MoveTo(self.arm, Pose((hand[0], hand[1], CARRY_Z), turned_down(0.0)))]
            if self._owner == shown:
                self._moved = True
                mat = world.obj(MAT)
                return (upright + self._set_down(world, shown, mat.pose.pos[:2],
                                                 mat.pose.pos[2] + mat.size[2])
                        + [Done("Tom's present is on the mat")])
            return upright + self._set_down(world, shown, self._home[shown])
        if self._moved:
            return [Done("Tom's present is on the mat")]
        if self._owner is not None:
            # read without turning it over (a camera caught it): straight to the mat
            self._moved = True
            return self._to_mat(world, self._owner)
        if not self._queue:
            return [Done("no label said Tom")]
        self._showing = self._queue.pop(0)
        x, y, _ = world.obj(self._showing).pose.pos
        return (self._pick_up(world, self._showing)
                + [MoveTo(self.arm, Pose((x, y, CARRY_Z), turned_down(SHOW_YAW))),
                   MoveTo(self.arm, show_pose()), Wait(1.0)])

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """From the first picture: guess the first present -- right one time
        in five."""
        return self._to_mat(world, read(world) or "present0") + [Done("planned")]


def secrets(inst) -> Tuple[str, ...]:
    """What names the answer (``TaskInstance.secrets``): each present's and
    each name card's own name says whose it is."""
    owners = dict(inst.metadata["owners"])
    return (inst.metadata["target"], *owners, *(CARDS[n] for n in owners.values() if n in CARDS))
