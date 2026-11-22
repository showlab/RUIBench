"""DS2 "egg hunt" -- Discovery.

Knob: where the egg may be hidden (under one of three pots -> under one of two
pots or behind one of two tall boxes).

A painted Easter egg is hidden and nothing in the first picture says where:
under one of the glazed flower pots turned upside down, or, at the hard level,
behind one of two grocery boxes standing at the ends of the row (behind as the
fixed camera sees it, on the robot's side; v3 folds "hide and seek" in here).
A pot is lifted to look; behind a box, the wrist camera is taken round -- and
a box must not be knocked over.  Which place hides it is drawn uniformly per
seed, and which pot or box stands where is shuffled on its own, so a look says
nothing either.  Put the egg in the wicker basket.  The reference sets each
pot down beside its place and looks with the hand empty: the Franka's wrist
camera is beside the hand, and a pot held up hides what is under it.

What the first frame cannot show, a plan made from it can only guess: the
``first_frame_only`` control is judged against chance (1 in 3, 1 in 4).

v8 (2026-09-24), the advisor: the task reads at a glance, but its things were
neither pretty nor real -- plastic garden pots in turquoise, avocado and red,
a teal plastic basket, a gold-foil egg.  Now three clay pots (two scans and
a Poly Haven model), the Poly Haven wicker basket, and a Hutsul pysanka
scanned by the Virtual Museums of Malopolska (CC0).  v10, the advisor: now too
dull -- brown pots in a brown basket on a brown table.  The pots are glazed
stoneware in cobalt, sunflower yellow and mint, turned from parts, bare clay
at their feet.  v12: the basket stands on the picnic cloth (``BASKET_ON``);
on the table, under the cloth's printed face, it looked bottomless.
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.base import BodySpec
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, In, Milestone, Never, Scorer, Upright
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

EGG, BASKET = "egg", "basket"
KINDS = ("blue pot", "yellow pot", "green pot")
#: each kind's glaze.  The pots: glazed stoneware, 9.6 cm across the rim and
#: 8.4 cm tall, drawn round (``parts``' "flowerpot", and "pot_foot" for the
#: bare clay), each over hidden rings of blocks following its taper and a
#: hidden base -- every pot upside down, no drainage hole (a camera above saw
#: the egg through the scanned pots' until v8 plugged them)
GLAZES = {"blue pot": "ceramic:cobalt", "yellow pot": "ceramic:sunflower",
          "green pot": "ceramic:mint"}
POT_R, POT_HZ = 0.048, 0.042
UPSIDE_DOWN = (0.0, 1.0, 0.0, 0.0)
#: a red gingham picnic cloth under it all (v10): where it lies, how wide
#: (across the table; its depth follows the print's shape, 50 cm) -- inside
#: the fixed camera's frame, the hard level's boxes on it
CLOTH, CLOTH_AT, CLOTH_WIDTH = "cloth", (0.29, 0.0), 0.76
#: the cloth's printed face, half a millimetre over the table -- drawn only
CLOTH_FACE = TABLE + 0.0005
#: the hard level's boxes, standing, their broad faces to the fixed camera:
#: 20-21 cm, they hide an egg up to 17 cm behind them (hide and seek, v2)
BOXES = ("crackers", "cheez_it")
BOX_SIZE = {"crackers": (0.0701, 0.0302, 0.1046), "cheez_it": (0.082, 0.0359, 0.1067)}
BOX_X, BOX_YAW, HIDE_X = 0.47, math.pi / 2, 0.36
#: knocked over: tilted further than this from upright
KNOCKED = 0.52
#: the hiding places per level: (kind of place, y).  Pots 17 cm apart, so the
#: open hand over one (21 cm across) clears the green pot beside it; the boxes
#: at the ends, far enough out that the hand lifting a pot clears them
PLACES = {"easy": (("pot", -0.17), ("pot", 0.0), ("pot", 0.17)),
          "hard": (("box", -0.30), ("pot", -0.085), ("pot", 0.085), ("box", 0.30))}
ROW_X = 0.46
#: the basket on the robot's side of the row: the fixed camera looks from the
#: far side, and a 13 cm basket there hid half of the pots behind it.  Its
#: drop points stay 31 cm or more in front of the arm's base: at 25 cm, 16 cm
#: up, the Franka could not get there with the egg (Isaac, 2026-09-18)
BASKET_AT = (0.16, 0.0)
#: a rattan basket woven in red, orange and maroon, 18 cm across and 8.4 cm
#: deep, empty (v10: the wicker one was one more brown thing; v11, the
#: advisor: no Easter grass in it)
BASKET_SIZE = (0.09, 0.09, 0.0419)
#: the basket stands on the cloth, its bottom a millimetre over the cloth's
#: face.  Standing on the table, its woven floor (a fraction of a millimetre
#: thick) was under the face, and the gingham showed inside it, as if it had
#: no bottom (v12, the advisor: "the basket should have a bottom")
BASKET_ON = CLOTH_FACE + 0.001
#: where a lifted thing is set down: its own place, 14 cm nearer the robot --
#: never between the camera and the place it leaves open
ASIDE = 0.14
#: looking at a place: the wrist camera (4.5 cm out on the hand's far side)
#: straight above it, LOOK_Z over the table
CAMERA_OUT, LOOK_Z = 0.045, 0.30
#: the egg lies on its side, pointing at the robot: a large chocolate egg, 5.6
#: cm long and 4 across, gripped across (the hand's long side across the fixed
#: camera's view) 3 mm above its girth -- the fingertips still reach 5 mm below
#: it.  The fixed camera sees 44% of it in the hand; a 5 cm egg, 35%
EGG_HALF = (0.028, 0.0213, 0.0213)
EGG_GRASP_HEIGHT = 0.003
CAMERAS = ("agentview", "wrist")


#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.025, 0.550), (-0.400, 0.400), 0.22)


def cloth() -> BodySpec:
    """The picnic cloth: a printed face on the table, drawn only -- what
    stands on it stands on the table, except the basket, whose floor would
    be under the face (``BASKET_ON``)."""
    x, y = ahead(*CLOTH_AT)
    face = P.printed((0.0, 0.0, 0.0), CLOTH_WIDTH, "cloth:gingham_red")
    return BodySpec(CLOTH, "compound", (face[2][0], face[2][1], CLOTH_FACE - TABLE),
                    Pose((x, y, CLOTH_FACE)),
                    category="cloth", color="red", fixed=True,
                    attrs={"parts": (face,), "collides": False, "graspable": False})


def pot(name: str, look: str, xy) -> BodySpec:
    """A glazed pot standing upside down: drawn round, what the hand and the
    egg meet hidden rings of blocks -- four, stepping out with the wall as
    it flares -- the rim band's, and the base."""
    r, hz, t = POT_R, POT_HZ, P.POT_T
    band = P.POT_RIM * 2 * hz
    parts = [P.cylinder((0.0, 0.0, -hz + t / 2), P.POT_BASE * r, t / 2, "grey", collides="hidden")]
    below = hz - band
    for k in range(4):
        z0 = -hz + (below + hz) * k / 4
        z1 = -hz + (below + hz) * (k + 1) / 4
        outer = P._pot_outer(r, hz, (z0 + z1) / 2)
        parts += [q[:5] + ("hidden",) for q in P.ring((0.0, 0.0, (z0 + z1) / 2), outer, outer - t,
                                                      (z1 - z0) / 2, "grey", segments=16)]
    parts += [q[:5] + ("hidden",) for q in P.ring((0.0, 0.0, hz - band / 2), r, r - t, band / 2,
                                                  "grey", segments=16)]
    parts += [P.lathe((0.0, 0.0, 0.0), "flowerpot", r, hz, GLAZES[look]),
              P.lathe((0.0, 0.0, 0.0), "pot_foot", r, hz, "paint:terracotta")]
    return BodySpec(name, "compound", (r, r, hz), Pose((xy[0], xy[1], TABLE + hz), UPSIDE_DOWN),
                    category=look, color=GLAZES[look], mass=0.2,
                    attrs={"parts": tuple(parts), "footprint": "round", "container": "inverted",
                           "wall": t, "kind": look})


def hidden_under(seed: int, n: int) -> int:
    """Which place hides the egg: every place exactly once in any n
    consecutive seeds, in a seed-drawn order -- uniform by construction, so a
    blind guess succeeds exactly one time in n."""
    group, place = divmod(seed, n)
    order = list(range(n))
    SeedBundle(group).stream(f"under/{n}").shuffle(order)
    return order[place]


def build_scene(level: str, seed: int):
    """The places (pots, and boxes at the hard level), the egg under or behind
    one, the basket.  Returns the scene, the places' names, what kind each is,
    where each is and where the egg hides for it, and which one hides it."""
    layout = PLACES[level]
    n = len(layout)
    under = hidden_under(seed, n)
    stream = SeedBundle(seed)
    pots = [k for k in KINDS][:sum(1 for kind, _ in layout if kind == "pot")]
    stream.stream("pots").shuffle(pots)
    boxes = list(BOXES)
    stream.stream("boxes").shuffle(boxes)
    bodies, names, kinds, homes, hides = [cloth()], [], [], [], []
    # each place is named for what it is, so every seed builds the same bodies
    # and only where they stand changes
    for i, (kind, y) in enumerate(layout):
        if kind == "pot":
            look = pots.pop()
            name = look.replace(" ", "_")
            xy = ahead(ROW_X, y)
            bodies.append(pot(name, look, xy))
            hides.append(xy)
        else:
            look = boxes.pop()
            name = f"{look}_box"
            hx, hy, hz = BOX_SIZE[look]
            xy = ahead(BOX_X, y)
            bodies.append(asset_body(name, look, xy, fallback_size=(hx, hy, hz), mass=0.3,
                                     category="box", yaw=BOX_YAW, graspable=False, kind="box"))
            hides.append(ahead(HIDE_X, y))
        names.append(name)
        kinds.append(kind)
        homes.append(xy)
    egg = easter_egg(EGG, hides[under])
    basket = asset_body(BASKET, "rattan_basket", ahead(*BASKET_AT), fallback_size=BASKET_SIZE,
                        mass=0.3, fixed=True, category="basket", container="open", wall=0.006,
                        graspable=False, on=BASKET_ON)
    bodies = sorted(bodies, key=lambda b: b.name)
    scene = station_scene(f"ds2_egg_hunt_{level}", [egg] + bodies + [basket], frame=FRAME)
    return scene, names, kinds, homes, hides, under


def easter_egg(name: str, xy) -> BodySpec:
    """A painted egg lying on its side: the scanned pysanka drawn over a
    hidden capsule of its length and girth."""
    a, r = EGG_HALF[0], EGG_HALF[1]
    parts = [P.visual("easter_egg", (0, 0, 0), EGG_HALF),
             P.capsule((0, 0, 0), r, a - r, "grey", rot=P.turn_y(math.pi / 2),
                       collides="hidden")]
    return BodySpec(name, "compound", EGG_HALF, Pose((xy[0], xy[1], TABLE + r)),
                    category="egg", color="gold", mass=0.03,
                    attrs={"parts": tuple(parts), "curved_top": True, "hidden_at_start": True,
                           "grasp_height": EGG_GRASP_HEIGHT})


def lifted(world: WorldState, name: str, home) -> bool:
    obj = world.obj(name)
    return obj.pose.pos[2] > TABLE + obj.size[2] + 0.03 or math.dist(obj.pose.pos[:2], home) > 0.05


def seen(world: WorldState) -> bool:
    return any(world.obj(EGG).seen(c) for c in CAMERAS)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("DS2")
    scene, places, kinds, spots, hides, under = build_scene(level, seed)
    homes = dict(zip(places, spots))
    in_basket = In(EGG, BASKET)
    pots = [p for p, k in zip(places, kinds) if k == "pot"]
    boxes = [p for p, k in zip(places, kinds) if k == "box"]
    looks = [(x - CAMERA_OUT, y) for x, y in hides]
    knocked = Custom(lambda w: any(not Upright(b, tol=KNOCKED).holds(w) for b in boxes),
                     "a box knocked over")

    milestones = (
        Milestone("went_to_look", Custom(
            lambda w: any(lifted(w, p, homes[p]) for p in pots)
            or min(math.dist(w.arm(RIGHT).eef_pose.pos[:2], q) for q in looks) < 0.05,
            "a pot lifted, or the hand over a hiding place")),
        Milestone("egg_found", Custom(seen, "the egg in sight")),
        Milestone("egg_in_basket", in_basket),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Find the painted egg and put it in the basket.",
        scorer=Scorer(task_id="DS2", level=level,
                      success=Finally(Custom(lambda w: in_basket.holds(w) and AtRest(EGG).holds(w),
                                             "the egg at rest in the basket")),
                      milestones=milestones,
                      violations=(("nothing_knocked_over", Never(knocked)),)),
        milestones=milestones,
        reference_factory=lambda inst, p=tuple(places), k=tuple(kinds), h=homes, hd=tuple(hides):
            HuntPolicy(p, h, k, hd),
        metadata={"knob": spec.knob.field, "knob_value": spec.knob.levels[level],
                  # hidden from the agent; here so the gap-filled control can name it
                  "under": places[under], "kinds": dict(zip(places, kinds)),
                  "max_model_calls": 60 + 20 * len(places), "needs_visibility": True},
    )


class HuntPolicy(ScriptedPolicy):
    """Place by place: lift a pot, set it down beside its place and look where
    it stood; look behind a box with the wrist camera from above.  The egg in
    sight: take it to the basket.  Whether it has been found is read off what
    the cameras see, never off where it is."""

    name = "egg_hunt"

    def __init__(self, places: Sequence[str], homes, kinds=None, hides=None,
                 arm: str = RIGHT) -> None:
        self.places = list(places)
        self.homes = dict(homes)
        self.kinds = dict(zip(self.places, kinds or ["pot"] * len(self.places)))
        self.hides = dict(zip(self.places, hides or [self.homes[p] for p in self.places]))
        self.arm = arm
        self.reset()

    def aside(self, name: str):
        """Beside where ``name`` stood, nearer the robot."""
        x, y = self.homes[name]
        return (x - ASIDE, y)

    def reset(self) -> None:
        self._next = 0
        self._found = False
        self._looking: Optional[str] = None
        self._fetch = None

    def _lift(self, world: WorldState, name: str) -> List[Command]:
        from ...policies.scripted import grasp_pose

        grasp = grasp_pose(world.obj(name))
        above = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + 0.10), GRASP_DOWN)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above), MoveTo(self.arm, grasp),
                SetGripper(self.arm, 1.0), MoveTo(self.arm, above)]

    def _put_down(self, world: WorldState, name: str, xy) -> List[Command]:
        arm = world.arm(self.arm)
        held = world.obj(name)
        offset = arm.eef_pose.pos[2] - held.pose.pos[2]
        down = Pose((xy[0], xy[1], TABLE + held.size[2] + 0.004 + offset), GRASP_DOWN)
        up = Pose((xy[0], xy[1], down.pos[2] + 0.10), GRASP_DOWN)
        return [MoveTo(self.arm, up), MoveTo(self.arm, down), Wait(0.3),
                SetGripper(self.arm, 0.0), MoveTo(self.arm, up)]

    def _look(self, name: str) -> List[Command]:
        x, y = self.hides[name]
        return [MoveTo(self.arm, Pose((x - CAMERA_OUT, y, TABLE + LOOK_Z), GRASP_DOWN)),
                Wait(0.6)]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import MAX_GRASP_ATTEMPTS, PickAndPlacePolicy

        if ctx.extras.get("executor_busy"):
            return []
        holding = world.arm(self.arm).holding
        if holding in self.places:
            # set it down beside its place, then look where it stood
            self._looking = holding
            return self._put_down(world, holding, self.aside(holding))
        if self._looking is not None:
            name, self._looking = self._looking, None
            return self._look(name)
        self._found = self._found or seen(world)
        if self._found:
            if self._fetch is None:
                self._fetch = PickAndPlacePolicy(EGG, lambda w: BASKET, arm=self.arm,
                                                 place_clearance=0.01)
            cmds = self._fetch.commands(world, ctx)
            if self._fetch.attempts > MAX_GRASP_ATTEMPTS:
                return [Done("could not take the egg to the basket")]
            return cmds
        if self._next < len(self.places):
            name = self.places[self._next]
            self._next += 1
            if self.kinds[name] == "box":
                return [SetGripper(self.arm, 0.0)] + self._look(name)
            return self._lift(world, name)
        return [Done("the egg is in none of them")]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Blind: guess the first place -- right one time in n."""
        from ...policies.scripted import _pick_and_place_sequence

        first = self.places[0]
        egg = world.obj(EGG)
        hx, hy = self.hides[first]
        guess = replace(egg, pose=Pose((hx, hy, egg.pose.pos[2]), egg.pose.quat))
        basket = world.obj(BASKET)
        out: List[Command] = []
        if self.kinds[first] == "pot":
            aside = self.aside(first)
            out += _pick_and_place_sequence(self.arm, world.obj(first), (aside[0], aside[1], TABLE),
                                            0.0, 0.10, 0.004)
        out += _pick_and_place_sequence(self.arm, guess, basket.pose.pos, basket.size[2],
                                        0.12, 0.01)
        return out + [Done("guessed")]


def secrets(inst) -> Tuple[str, ...]:
    """What the egg is under (``TaskInstance.secrets``)."""
    return (inst.metadata["under"],)
