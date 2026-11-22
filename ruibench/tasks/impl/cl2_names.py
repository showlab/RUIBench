"""CL2 "the dolls' names" -- In-Context Learning.

Knob: how many names there are to learn (1 -> 2).

Six wooden dolls lie on their backs on the table (scans of two dollhouse
families, v2 revision 2026-09-18), beside a doll's bed and a little sofa.
Before the episode a film shows the robot lifting one of the dolls: "This is
Momo."  At the hard level a second film lifts another: "This is Kiki."  Then
the dolls are laid out again in new places, and: "Put Momo to bed." (easy) or
"Put Momo to bed and Kiki on the sofa." (hard).  The names have to be bound
to how the dolls look: where they lay in the film says nothing about where
they lie now, and swapping the two names fails.  Nothing else may be moved.

v3 (2026-09-19): the film is shot at another table, in another light -- a
blue or a teal cloth, or a pale ash table, the studio light dimmer or
brighter, warmer or cooler and from another side (``Demonstration.look``) --
so what is learnt has to be the doll, not the picture.  And the six are two
look-alike pairs and two others: the two fathers both in a blue striped shirt
and jeans, the two mothers both in a yellow jacket -- told apart by their hair
(dark or fair; black or red) and the mothers' dresses, not by "the man in the
striped shirt".

Which names are used, which dolls they name, and where everything lies are
drawn per seed; the six dolls, the bed and the sofa are the same at both
levels.  It replaces "what is a dax?" (invented words for odd toys, a basket
and a plate): the same test -- binding a new word from one showing -- in a
form a viewer reads at once (v2-task-revisions.md §3.2).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Dict, List, Optional, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import AtRest, Custom, Finally, Held, In, Milestone, Never, Scorer
from ..instance import Demonstration, TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, asset_body, station_scene

#: each doll's catalogue entry and measured half extents (for the mock): 9 cm
#: long, lying along the fixed camera's line of sight (head to the robot), the
#: hand across the view taking it across the body -- their hands 6-7 cm apart,
#: inside the gripper's 8 cm.  Lying across the view, the hand along the line
#: of sight hid 75-81% of a doll and its side met the bed
LOOKS = {"dark-haired man in a striped shirt": ("doll_family_1", (0.034, 0.045, 0.0146)),
         "fair-haired man in a striped shirt": ("doll_modern_1", (0.0307, 0.045, 0.0136)),
         "black-haired woman in a yellow jacket": ("doll_family_0", (0.0355, 0.045, 0.0192)),
         "red-haired woman in a yellow jacket": ("doll_modern_0", (0.0329, 0.045, 0.023)),
         "boy in a green shirt": ("doll_family_3", (0.0325, 0.045, 0.0133)),
         "boy in an orange shirt": ("doll_modern_3", (0.032, 0.045, 0.0134))}
#: the look-alike pairs: the same clothes, other hair
LOOKALIKES = (("dark-haired man in a striped shirt", "fair-haired man in a striped shirt"),
              ("black-haired woman in a yellow jacket", "red-haired woman in a yellow jacket"))
#: where the films are shot, one per seed: another table, another light
FILM_LOOKS = ({"table": "fabric:sky", "light": 0.7, "light_colour": (1.0, 0.86, 0.68), "light_turn": 120.0},
              {"table": "wood:ash", "light": 1.4, "light_colour": (0.86, 0.93, 1.0), "light_turn": -90.0},
              {"table": "fabric:teal", "light": 0.85, "light_colour": (1.0, 0.97, 0.92), "light_turn": 200.0})
KINDS = tuple(LOOKS)
DOLLS = tuple(f"doll{i}" for i in range(len(KINDS)))
BED, SOFA = "bed", "sofa"
BED_HALF, SOFA_HALF = (0.0872, 0.0621, 0.0303), (0.1048, 0.0434, 0.0511)
#: what the bed and the sofa are solid as, off the meshes' height maps
#: (bench-issue #20: registered as trays with a 6 mm floor and walls all
#: round, they gave depth 3 cm under the mattress, and the sofa a front wall
#: it has not got).  Each box: from -x to +x and -y to +y as fractions of the
#: half extents, and its top above the bottom (m).  Things put in rest on the
#: mattress, 3.6 cm up, and on the seat, 5.1 cm up -- a doll laid on them on
#: Isaac lay 3.7 cm and 5.2 cm higher than on the table
BED_FLOOR, SOFA_FLOOR = 0.036, 0.051
BED_SHAPE = ((-1.0, 1.0, -1.0, 1.0, BED_FLOOR),            # the mattress
             (0.40, 0.72, -1.0, 1.0, 0.044),               # the pillows
             (0.72, 1.0, -1.0, 1.0, 0.0606),               # the headboard
             (-1.0, -0.85, -0.57, 0.57, 0.050))            # the footboard
SOFA_SHAPE = ((-0.72, 0.72, -1.0, 0.35, SOFA_FLOOR),       # the seat
              (-1.0, -0.72, -1.0, 0.35, 0.066),            # the arms
              (0.72, 1.0, -1.0, 0.35, 0.066),
              (-1.0, 1.0, 0.35, 1.0, 0.098))               # the back
#: names nobody has heard before, a pair drawn per seed
NAMES = ("Momo", "Kiki", "Lulu", "Bibi", "Nono", "Pipa", "Tuti", "Zaza")
#: the dolls lie in two staggered rows of three, 15 cm apart along a row; the
#: hand across the view taking the front row's right one clears the sofa
SPOTS = [(0.22, y) for y in (-0.27, -0.12, 0.03)] + [(0.36, y) for y in (-0.21, -0.06, 0.09)]
#: the dolls' turn: head toward the robot, along the line of sight
DOLL_YAW = math.pi / 2
#: the bed and the sofa on the right, long side along the line of sight like
#: the dolls: the bed's head toward the robot, the sofa's back to the right
BED_AT, SOFA_AT = (0.20, 0.225), (0.465, 0.245)
BED_YAW, SOFA_YAW = 0.0, 0.0
#: a doll counts as moved once it is this far from where it lay
MOVED = 0.015


#: the part of the table the fixed camera shows, with the gripper where it
#: starts (``_common.Frame``)
FRAME = Frame((0.075, 0.575), (-0.350, 0.325), 0.11)


def film_look(seed: int) -> Dict[str, object]:
    """Where this seed's films are shot."""
    return dict(SeedBundle(seed).choice("film_look", FILM_LOOKS))


def spots() -> List[Tuple[float, float]]:
    return [ahead(x, y) for x, y in SPOTS]


def draw(seed: int):
    """(the doll kinds by body, the two names, which doll each names).  The
    bodies keep their kinds -- the same bodies in every seed, so one simulator
    process serves them all; the names, whom they name and the places are drawn."""
    rng = SeedBundle(seed).stream("names")
    first, second = rng.sample(range(len(NAMES)), 2)
    momo, kiki = rng.sample(range(len(DOLLS)), 2)
    return list(KINDS), (NAMES[first], NAMES[second]), (momo, kiki)


def doll_body(name: str, kind: str, xy):
    asset, size = LOOKS[kind]
    return asset_body(name, asset, xy, fallback_size=size, mass=0.04, category="doll",
                      kind=kind, yaw=DOLL_YAW, grasp_yaw=0.0, grasp_yaw_at=DOLL_YAW)


def solid_as(body, shape):
    """``body`` with what it is solid as (``attrs["proxy"]``): the boxes of
    ``shape`` in its own frame, for what has no mesh to ask -- sight lines,
    depth, the design checks.  Isaac collides with the mesh itself."""
    hx, hy, hz = body.size
    boxes = tuple(P.box(((x0 + x1) / 2 * hx, (y0 + y1) / 2 * hy, -hz + top / 2),
                        ((x1 - x0) / 2 * hx, (y1 - y0) / 2 * hy, top / 2), "proxy")
                  for x0, x1, y0, y1, top in shape)
    return replace(body, attrs={**body.attrs, "proxy": boxes})


def scene_for(seed: int, label: str, kinds):
    where = spots()
    SeedBundle(seed).stream(f"places/{label}").shuffle(where)
    bodies = [doll_body(name, kind, xy) for name, kind, xy in zip(DOLLS, kinds, where)]
    bodies.append(solid_as(asset_body(BED, "doll_bed", ahead(*BED_AT), fallback_size=BED_HALF,
                                      yaw=BED_YAW, fixed=True, category="bed", container="open",
                                      wall=0.006, floor=BED_FLOOR, graspable=False), BED_SHAPE))
    bodies.append(solid_as(asset_body(SOFA, "doll_sofa", ahead(*SOFA_AT), fallback_size=SOFA_HALF,
                                      yaw=SOFA_YAW, fixed=True, category="sofa", container="open",
                                      wall=0.006, floor=SOFA_FLOOR, graspable=False), SOFA_SHAPE))
    return station_scene("cl2_names", bodies, frame=FRAME)


class ShowDoll(ScriptedPolicy):
    """The film: pick one doll up, hold it high for a moment, lay it back."""

    name = "show_doll"

    def __init__(self, doll: str, arm: str = RIGHT) -> None:
        self.doll = doll
        self.arm = arm
        self._sent = False

    def reset(self) -> None:
        self._sent = False

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import grasp_pose

        if ctx.extras.get("executor_busy"):
            return []
        if self._sent:
            return [Done("shown")]
        self._sent = True
        grasp = grasp_pose(world.obj(self.doll))
        x, y, z = grasp.pos
        above = Pose((x, y, z + 0.10), grasp.quat)
        show = Pose((x, y, TABLE + 0.20), grasp.quat)
        down = Pose((x, y, z + 0.003), grasp.quat)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above), MoveTo(self.arm, grasp),
                SetGripper(self.arm, 1.0), MoveTo(self.arm, show), Wait(1.5),
                MoveTo(self.arm, down), SetGripper(self.arm, 0.0), MoveTo(self.arm, above)]


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CL2")
    kinds, (first, second), (momo_i, kiki_i) = draw(seed)
    momo, kiki = DOLLS[momo_i], DOLLS[kiki_i]
    film_scene = scene_for(seed, "film", kinds)
    look = film_look(seed)
    films = [Demonstration(scene=film_scene, demonstrator=lambda t=momo: ShowDoll(t),
                           caption=f"This is {first}.", look=look)]
    if level == "hard":
        films.append(Demonstration(scene=film_scene, demonstrator=lambda t=kiki: ShowDoll(t),
                                   caption=f"This is {second}.", look=look))
    scene = scene_for(seed, "episode", kinds)
    start = {b.name: b.pose.pos for b in scene.bodies if b.name in DOLLS}
    moves = [(momo, BED)] + ([(kiki, SOFA)] if level == "hard" else [])
    placed = [In(doll, target) for doll, target in moves]
    moving = {doll for doll, _ in moves}
    others = [d for d in DOLLS if d not in moving]

    def still(w: WorldState) -> bool:
        return all(math.dist(w.obj(d).pose.pos[:2], start[d][:2]) < MOVED for d in others)

    instruction = (f"Put {first} to bed." if level == "easy"
                   else f"Put {first} to bed and {second} on the sofa.")
    done = Custom(lambda w: all(p.holds(w) for p in placed)
                  and all(AtRest(d).holds(w) for d in moving) and still(w),
                  f"{instruction[4:-1]}, nothing else moved")
    # a doll set down where it does not belong fails at once: a wrong doll taken
    # out again is not a second guess (v2.3, from a review of how an agent could
    # get round the task)
    targets = {t for _, t in moves}
    misplaced = Custom(lambda w: any(In(d, t).holds(w) for d in DOLLS for t in targets
                                     if (d, t) not in moves), "a doll where it does not belong")
    milestones = (
        Milestone("film_watched", Custom(lambda w: True, "the episode began after the film")),
        Milestone("a_doll_picked", Custom(lambda w: any(w.obj(d).is_held for d in DOLLS),
                                          "a doll in hand")),
        Milestone("named_doll_picked", Held(momo)),
        Milestone("placed", done),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene, instruction=instruction,
        scorer=Scorer(task_id="CL2", level=level, success=Finally(done) & Never(misplaced),
                      milestones=milestones),
        milestones=milestones,
        demonstrations=tuple(films),
        reference_factory=lambda inst, lv=level, names=(first, second): NameBinder(lv, names),
        metadata={"knob": spec.knob.field, "knob_value": 1 if level == "easy" else 2,
                  # hidden from the agent; here so the matched control can define the names
                  "names": {first: kinds[momo_i], **({second: kinds[kiki_i]}
                                                     if level == "hard" else {})},
                  "target": BED if level == "easy" else [BED, SOFA],
                  "film_look": dict(look), "lookalikes": [list(p) for p in LOOKALIKES],
                  "max_model_calls": 30 if level == "easy" else 45},
    )


class NameBinder(ScriptedPolicy):
    """Watch the films: which doll was lifted, under which name.  Find the
    doll that looks the same (its kind) in the episode, and do as told.
    Without the films (``ignore_demonstration``) it takes the first dolls."""

    name = "name_binder"

    def __init__(self, level: str, names: Tuple[str, str], arm: str = RIGHT) -> None:
        self.level = level
        self.names = names
        self.arm = arm
        self.reset()

    def reset(self) -> None:
        self._inner = None

    @staticmethod
    def learned(watched, names) -> Dict[str, str]:
        """{name: the lifted doll's kind}, from the films' captions and frames."""
        out = {}
        for film in watched:
            name = next((n for n in names if film.caption == f"This is {n}."), None)
            first = film.states[0] if film.states else None
            for state in film.states:
                held = next((d for d in DOLLS if state.obj(d).is_held), None)
                if held and name:
                    out[name] = first.obj(held).attrs.get("kind")
                    break
        return out

    def _moves(self, world: WorldState, ctx: TurnContext) -> List[Tuple[str, str]]:
        known = self.learned(ctx.extras.get("demonstrations", ()), self.names)

        def doll(name: str, fallback: int) -> str:
            kind = known.get(name)
            match = [d for d in DOLLS if world.obj(d).attrs.get("kind") == kind]
            return match[0] if match else DOLLS[fallback]

        out = [(doll(self.names[0], 0), BED)]
        if self.level == "hard":
            out.append((doll(self.names[1], 1), SOFA))
        return out

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import SequentialPickAndPlacePolicy

        if self._inner is None:
            self._inner = SequentialPickAndPlacePolicy(self._moves(world, ctx), arm=self.arm,
                                                       lift_height=0.10)
        return self._inner.commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        from ...policies.scripted import SequentialPickAndPlacePolicy

        return SequentialPickAndPlacePolicy(self._moves(world, ctx), arm=self.arm,
                                            lift_height=0.10).plan(world, ctx)


def secrets(inst) -> Tuple[str, ...]:
    """The dolls the names belong to (``TaskInstance.secrets``)."""
    _, _, (momo_i, kiki_i) = draw(inst.seed)
    return (DOLLS[momo_i],) + ((DOLLS[kiki_i],) if inst.level == "hard" else ())
