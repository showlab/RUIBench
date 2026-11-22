"""CR6 "fold the T-shirt" -- Contact-Rich Manipulation, two arms.

Knob: how many folds are asked for (one: the hem up to the shoulders ->
three: each sleeve in first).

A child's striped T-shirt lies flat between the arms, its collar away from
the robot.  "Fold the T-shirt in half: its bottom edge up to its shoulders."
What to do is all there.  How is not: a cloth is taken by pinching it, a
corner in each hand; the two hands have to go together, up and round the
fold -- carried flat along the table, the cloth is pushed into a heap, and
with one corner carried alone the other is dragged askew -- and let go where
the corners are to lie.  At the hard level each sleeve is one hand's work,
and the hem then goes up over them.

(10/5: the user asked for a real garment, not a towel.  The towel the task
was first made with is still in ``_garment.py``, a plain thing to try the
engine on.)

What is not simulated: the pinch.  A hand that closes its fingers on the
cloth holds the bit of it under its fingertips fast until it opens
(``engines/cloth.pinched``; friction between two fingers does not hold a
cloth in this engine: Isaac, 10/5).  The hint says as much, in a person's
words.  Where there is no physics the cloth folds, when it is let go, along
the perpendicular bisector of the way it was carried: one hand does there
what takes two on Isaac, and how high the cloth was carried does not count.

The cloth is solved on the graphics card; every other task's physics is not.
Two runs of one mission agree to a hundredth of a millimetre, not to the bit.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

from ...core.config import RunConfig
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines.cloth import cloth_body, key_point
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import Custom, Finally, HoldsFor, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from . import _garment as G
from ._common import Frame, LEFT, RIGHT, TABLE, ahead, station_scene, to_rest

#: the garment (the user, 10/5: a real one), and what it is called in the scene
GARMENT = G.tshirt()
CLOTH = GARMENT.name

# -- where it lies: its middle this far ahead of the arms' line, between them,
# a little off and a little turned by seed.  An X5's hand pointing down gets to
# the table from 15 cm ahead to 48 cm, 13 cm to a side and more, and 5 cm above
# it to 45 cm (10/5, REPORT-2.md, on a grid of 3 cm): the cloth's corners,
# turned, stay inside that with the hand a little above them
AT = (0.31, 0.0)
JITTER, TURN = (0.006, 0.010), math.radians(4)
#: its print (``engines/prints.py``: drawn to the cut the cloth is cut to)
FINISH = "paper:tee:navy"

# -- folded: the marks lie on the goal's -- as well as turning and shifting the
# goal can lay it on them, not flipping it: a one-layer shirt is not mirrored
# without being turned over, and the mirror took the shoulders folded down onto
# the hem for the hem folded up (v2.3) -- this near on average, and this near at
# worst; nothing of it more than this high; no hand on it; and so for 2 s.  A
# corner read off the cameras is good to about 5 mm, and a fold made from it
# lies as far off: 8 mm for one fold lets a careful hand pass, and the folds'
# errors add up as the square root of their number (v2.3: 8 mm for each fold
# let a hem 5 cm short of the shoulders pass at three).
# On Isaac (10/5, seeds 0-24) the reference's one fold lies 2.7 mm off on
# average and 4.4 mm at worst, its three folds 3.6 mm and 6.8 mm; with one
# corner carried at a time the hem lies 12-16 mm and 37-48 mm off (after the
# sleeves, 16-22 mm and 44-64 mm).  (Not asked: that no point of it moves --
# a point of a pile at rest still trembles at 3-8 cm/s.)
MEAN_PER_FOLD, WORST = 0.008, 0.040
LOW, STABLE_SECONDS = 0.06, 2.0

# -- the reference's hands: how long they wait for a hold or for letting go
# (the engine takes most of a second over either), and where they go once the
# cloth is out of them -- this high over the table, this far out to their side
# (no higher: 42 cm ahead, where a hem is laid, an X5's hand pointing down
# gets 8 cm up only with its wrist on its limit, and stopped short of it)
SETTLE, AGAIN, WAITS = 0.6, 0.4, 3
CLEAR, ASIDE = 0.06, 0.05
#: the cloth's marks this near a place tell how high it lies there.  They are
#: 3 cm apart, and the narrow end of a folded sleeve has none nearer than this
#: to the shoulder it lies at: better a layer too many counted (the cloth is
#: let go 5 mm over the pile) than one too few (it is pressed into it)
NEAR = 0.05
MAX_TRIES = 2
LAID_ALREADY = 0.015                          # a point this near where it is to lie is left there

FRAME = Frame((0.14, 0.50), (-0.22, 0.22), 0.12)

INSTRUCTIONS = {"easy": "Fold the T-shirt in half: its bottom edge up to its shoulders.",
                "hard": "Fold the T-shirt: both sleeves in over the front, then its bottom edge up "
                        "to its shoulders."}
HINT = ("Closing a hand's fingers on the cloth pinches it: the cloth under the fingertips comes "
        "with the hand until the fingers open.")


def turn_z(angle: float):
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


# -- the cloth ------------------------------------------------------------------------
def cloth(xy, turn: float):
    spec = GARMENT.spec()
    return cloth_body(CLOTH, spec, Pose((xy[0], xy[1], TABLE + spec.radius), turn_z(turn)),
                      category=CLOTH, color=FINISH, mass=0.1)


def marks(w: WorldState) -> List[Tuple[float, float, float]]:
    return list(w.obj(CLOTH).points)


def key(w: WorldState, name: str) -> Tuple[float, float, float]:
    return tuple(key_point(w.obj(CLOTH), name))


# -- the scene ------------------------------------------------------------------------
def layout(seed: int):
    rng = SeedBundle(seed).stream("fold")
    return (ahead(AT[0] + rng.uniform(-JITTER[0], JITTER[0]), AT[1] + rng.uniform(-JITTER[1], JITTER[1])),
            rng.uniform(-TURN, TURN))


def build_scene(level: str, seed: int):
    xy, turn = layout(seed)
    # nothing else on the table: on the graphics card a fixed body, a joint or
    # a carried mark is moved by writing its pose, which the solver does not see
    return station_scene(f"cr6_fold_{CLOTH}_{level}", [cloth(xy, turn)], two_arms=True, frame=FRAME)


# -- what counts ------------------------------------------------------------------
def goals(level: str) -> List[List[Tuple[float, float]]]:
    """Where the marks lie after each of the level's folds in turn (the last:
    the level's goal), in the garment's frame."""
    flat = GARMENT.marks()
    out, pts, layers = [], list(flat), [0] * len(flat)
    for f in GARMENT.asked(level):
        taken = pts[G.nearest(flat, GARMENT.keypoint(f.moved[0]))]
        pts, layers = G.fold_flat(pts, layers, f.crease, taken)
        out.append(list(pts))
    return out


def off(w: WorldState, goal: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    """(mean, worst) distance of the cloth's marks from the goal's."""
    return G._fit(goal, [(x, y) for x, y, _ in marks(w)])


def laid(w: WorldState, goal, folds: int = 1) -> bool:
    mean, worst = off(w, goal)
    return (mean <= MEAN_PER_FOLD * math.sqrt(folds) and worst <= WORST
            and max(z for _, _, z in marks(w)) <= TABLE + LOW)


def lying(w: WorldState, at) -> float:
    """How much higher the cloth lies at a place than one layer of it flat on
    the table does: by its highest mark within ``NEAR`` of there."""
    one = TABLE + GARMENT.spec().radius
    return max(0.0, max((p[2] for p in marks(w) if math.dist(p[:2], at) <= NEAR), default=one) - one)


def hands_off(w: WorldState) -> bool:
    return w.obj(CLOTH).held_by is None and all(w.arm(a).holding != CLOTH for a in (LEFT, RIGHT))


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CR6")
    after = goals(level)

    def done(w: WorldState) -> bool:
        return laid(w, after[-1], len(after)) and hands_off(w)

    folded = Custom(done, "the cloth folded as asked, no hand on it")
    milestones = (
        Milestone("cloth_pinched", Custom(lambda w: not hands_off(w), "the cloth pinched by a hand")),
        Milestone("in_both_hands", Custom(lambda w: all(w.arm(a).holding == CLOTH for a in (LEFT, RIGHT)),
                                          "the cloth in both hands at once")),
        Milestone("first_fold_laid", Custom(lambda w: laid(w, after[0], 1) and hands_off(w),
                                            "the first fold laid")),
        Milestone("folded_hands_off", folded),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=build_scene(level, seed),
        instruction=INSTRUCTIONS[level], hint=HINT,
        scorer=Scorer(task_id="CR6", level=level,
                      success=Finally(folded) & HoldsFor(folded, STABLE_SECONDS),
                      milestones=milestones),
        milestones=milestones,
        # with physics the bit of cloth in a hand does not turn over with the fold
        reference_factory=lambda inst, lv=level, played=(config is None or config.engine == "mock"):
            FoldPolicy(lv, turned=played),
        metadata={"knob": spec.knob.field, "knob_value": len(after), "garment": CLOTH,
                  "max_model_calls": 40 + 20 * len(after)},
        extra_metrics={"off_mean_mm": lambda traj: round(1000 * off(traj[-1], after[-1])[0], 1),
                       "off_worst_mm": lambda traj: round(1000 * off(traj[-1], after[-1])[1], 1)},
    )


# -- the reference ------------------------------------------------------------------
def hand_at(p, quat=GRASP_DOWN) -> Pose:
    return Pose((p[0], p[1], TABLE + p[2]), quat)


class FoldPolicy(ScriptedPolicy):
    """Fold by fold: a hand to each point that is to go over, down to pinch
    it; the hands up and over together in an arch, to where the points are to
    lie; let go, and up.  Then the arms back to rest.
    Two hands with one edge make one move (``_garment.one_move``), and the
    cloth is let go on top of what lies there already (``lying``), not pressed
    into it: pressed against the fingertips, a hem corner went 4 cm aside with
    the finger that opened over it, and once it stayed there (Isaac, 10/5).
    ``serial`` (a stress agent): one hand at a time -- one corner carried over
    and laid, then the other.
    ``ignore_rules`` (the blind control): the cloth handled as a block would
    be -- taken by its middle in one hand and set down a little nearer."""

    name = "fold_cloth"

    def __init__(self, level: str, turned: bool = True) -> None:
        self.level = level
        #: does the bit of cloth in a hand turn over with the fold (no physics)?
        self.turned = turned
        self.serial = False
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._steps: Optional[List[Tuple[Tuple[str, str], ...]]] = None
        self._k = 0
        self._stage = "pinch"
        self._tries = 0
        self._waited = 0
        self._ways = {}

    # the steps: each a tuple of (the key point taken, the key point it is laid on)
    def _plan_steps(self) -> List[Tuple[Tuple[str, str], ...]]:
        steps = []
        for f in GARMENT.asked(self.level):
            pairs = tuple(zip(f.moved, f.onto))
            steps += [(p,) for p in pairs] if self.serial else [pairs]
        return steps

    @staticmethod
    def _arm_for(world: WorldState, taken: str, alone: bool, pairs) -> str:
        """The hand on the point's own side: of two points, the one further left
        is the left hand's; a point alone, by which side of the cloth it is on."""
        y = key(world, taken)[1]
        if alone:
            return LEFT if y >= world.obj(CLOTH).pose.pos[1] else RIGHT
        other = next(key(world, t)[1] for t, _ in pairs if t != taken)
        return LEFT if y >= other else RIGHT

    def _pinch(self, world: WorldState, pairs) -> List[Command]:
        self._ways = {}
        ways = [(taken, key(world, taken)[:2], key(world, onto)[:2]) for taken, onto in pairs]
        ways = [w for w in ways if math.dist(w[1], w[2]) >= LAID_ALREADY]      # the rest lie there already
        if not ways:
            return []
        if len(ways) > 1:
            # two hands with one edge make one move
            ways = [(taken, a, b) for (taken, _, _), (a, b)
                    in zip(ways, G.one_move([(a, b) for _, a, b in ways]))]
        # on top of what lies there already, and both hands as high
        lay = G.LAY + max(lying(world, b) for _, _, b in ways)
        above: List[Command] = []
        down: List[Command] = []
        for taken, a, b in ways:
            arm = self._arm_for(world, taken, len(pairs) == 1, pairs)
            pinch, over = G.fold_way(a, b, turned=self.turned, lay=lay)
            self._ways[arm] = over
            above += [SetGripper(arm, 0.0), MoveTo(arm, hand_at((pinch[0], pinch[1], G.HOVER)))]
            down += [MoveTo(arm, hand_at(pinch))]
        closing = [SetGripper(arm, 1.0) for arm in self._ways]
        return above + [Wait(0.1)] + down + [Wait(0.1)] + closing + [Wait(SETTLE)]

    def _carry(self) -> List[Command]:
        """The hands over together, stopping together at every point on the way."""
        out: List[Command] = []
        n = len(next(iter(self._ways.values())))
        for k in range(n):
            out += [MoveTo(arm, hand_at(way[k])) for arm, way in self._ways.items()]
            out.append(Wait(0.05))
        return out + [Wait(SETTLE)]

    def _clear(self, world: WorldState) -> List[Command]:
        """Straight up clear of the cloth, and a little out to the hand's own
        side: a hand that went off low over a towel folded twice caught its
        upper corner and took it along (Isaac, 10/5)."""
        out: List[Command] = []
        for arm in self._ways:
            x, y, _ = world.arm(arm).eef_pose.pos
            out.append(MoveTo(arm, hand_at((x, y, CLEAR))))
        out.append(Wait(0.1))
        for arm in self._ways:
            x, y, _ = world.arm(arm).eef_pose.pos
            out.append(MoveTo(arm, hand_at((x, y + (ASIDE if arm == LEFT else -ASIDE), CLEAR))))
        return out + [Wait(1.0)]

    def _as_a_block(self, world: WorldState) -> List[Command]:
        x, y, _ = world.obj(CLOTH).pose.pos
        return [SetGripper(LEFT, 0.0), MoveTo(LEFT, hand_at((x, y, G.HOVER))),
                MoveTo(LEFT, hand_at((x, y, G.PINCH))), SetGripper(LEFT, 1.0), Wait(SETTLE),
                MoveTo(LEFT, hand_at((x, y, G.HOVER))),
                MoveTo(LEFT, hand_at((x - 0.08, y, G.HOVER))),
                MoveTo(LEFT, hand_at((x - 0.08, y, G.LAY))), SetGripper(LEFT, 0.0), Wait(SETTLE),
                MoveTo(LEFT, hand_at((x - 0.08, y, G.HOVER)))] + to_rest(world, [LEFT]) + [Wait(1.0)]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self.ignore_rules:
            self._k += 1
            return self._as_a_block(world) if self._k == 1 else [Done("moved the cloth as a block")]
        if self._steps is None:
            self._steps = self._plan_steps()
        if self._k >= len(self._steps):
            if self._stage != "rest":
                self._stage = "rest"
                return to_rest(world, [LEFT, RIGHT]) + [Wait(STABLE_SECONDS + 0.5)]
            return [Done("folded")]
        pairs = self._steps[self._k]
        if self._stage == "pinch":
            cmds = self._pinch(world, pairs)
            if not cmds:                                    # nothing left of this step to do
                self._k += 1
                return [Wait(0.1)]
            self._stage, self._waited = "carry", 0
            return cmds
        if self._stage == "carry":
            if not all(world.arm(arm).holding == CLOTH for arm in self._ways):
                if self._waited < WAITS:                    # the hold is not there yet
                    self._waited += 1
                    return [Wait(AGAIN)]
                # a hand closed on nothing: open, and once more
                self._tries += 1
                if self._tries > MAX_TRIES:
                    return [SetGripper(arm, 0.0) for arm in self._ways] + [Done("could not pinch the cloth")]
                self._stage = "pinch"
                return [SetGripper(arm, 0.0) for arm in self._ways] + [Wait(0.3)]
            self._stage = "let_go"
            return self._carry()
        if self._stage == "let_go":
            self._stage, self._waited = "clear", 0
            return [SetGripper(arm, 0.0) for arm in self._ways] + [Wait(SETTLE)]
        # away only once the cloth is out of the hands -- a hand that rose while
        # it still held a corner took it 7 cm along (Isaac, 10/5) -- and on to
        # the next fold
        if any(world.arm(arm).holding == CLOTH for arm in self._ways) and self._waited < WAITS:
            self._waited += 1
            return [Wait(AGAIN)]
        self._stage, self._k, self._tries = "pinch", self._k + 1, 0
        return self._clear(world)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        # all that the first picture decides: where the folded cloth lies is what the next fold goes by
        steps = self._plan_steps()
        cmds = self._pinch(world, steps[0]) + self._carry()
        cmds += [SetGripper(arm, 0.0) for arm in self._ways]
        return cmds + [Done("one fold made; the next needs looking")]
