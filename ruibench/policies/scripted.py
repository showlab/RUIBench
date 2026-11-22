"""Reusable scripted references.

Deliberately simple and readable: these are a *specification of the intended
behaviour*, not an attempt at a strong policy.  If a scripted reference cannot
finish a task, the task is broken -- that is the point of layer 1.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from ..core.geometry import quat_mul, rotate, yaw_of
from ..core.types import GRASP_DOWN, Pose, WorldState
from ..interfaces.base import TurnContext
from ..interfaces.commands import Cancel, Command, Done, MoveTo, SetGripper, Wait
from ..scoring.predicates import inside
from .base import ScriptedPolicy


#: How far below an object's top the fingertips close.  Grasping a tall object
#: at its centre drove the palm into it: a 10 cm can blocked the descent and
#: the move waited out its timeout.
GRASP_DEPTH = 0.025


#: Never grasp lower than this above an object's bottom: the fingertips reach
#: this far below the tool centre point on the Piper (7.7 mm; 5 mm on the
#: Panda), and a thin slice grasped at its centre put them into the table.
MIN_GRASP_HEIGHT = 0.009


def grasp_offset(obj) -> float:
    """Height of the grasp point above the object's centre -- the object's own
    ``grasp_height`` if it names one (a toy car is held by its cabin)."""
    attrs = getattr(obj, "attrs", None) or {}
    if "grasp_height" in attrs:
        return float(attrs["grasp_height"])
    return max(obj.size[2] - GRASP_DEPTH, MIN_GRASP_HEIGHT - obj.size[2], 0.0)


#: A reference that keeps re-trying a grasp that keeps failing only burns the
#: call budget; after this many attempts on one object it gives up.
MAX_GRASP_ATTEMPTS = 3

#: How far the hand backs off after letting go.
RETREAT_HEIGHT = 0.10

#: A held thing turned further than this from upright hangs crooked in the
#: fingers, and is put down by its lowest corner rather than its half height.
CROOKED = math.radians(10)


def reach_below(obj) -> float:
    """How far a thing reaches below its middle as it is turned now: its half
    height when upright, more when it hangs crooked.  A goat held crooked
    after its grasp slipped came down on one hoof 1.2 cm above where its
    half height put the deck, the hand pressed on, and the fingers opening
    on it flung it off the raft (WK2, Isaac, v9)."""
    q = obj.pose.quat
    if rotate(q, (0.0, 0.0, 1.0))[2] >= math.cos(CROOKED):
        return obj.size[2]
    axes = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return sum(abs(rotate(q, axis)[2]) * half for axis, half in zip(axes, obj.size))


def retreat_pose(arm, height: float = RETREAT_HEIGHT) -> Pose:
    """Straight up from where the hand let go.

    Leaving open fingers around a placed object kept it twitching against them,
    and "at rest" then never held for long enough -- the check was measuring the
    hand, not the object.
    """
    x, y, z = arm.eef_pose.pos
    return Pose((x, y, z + height), GRASP_DOWN)


def grasp_pose(obj) -> Pose:
    """Where the hand takes a thing: over its middle, turned by its
    ``grasp_yaw`` -- the turn ``validate`` checks the grasp's clearance and
    sightlines for (WK2's animals are taken across the back, 2026-09-18).
    A thing that also names the heading it starts at (``grasp_yaw_at``) is
    taken as far round again as it has turned since: a doll knocked round a
    third of a turn was taken along its length and slipped out (CL2, v13)."""
    yaw = float(obj.attrs.get("grasp_yaw", 0.0))
    if "grasp_yaw_at" in obj.attrs:
        # the least turn that follows it: a parallel grasp is the same half way round
        yaw += fold_half_turn(yaw_of(obj.pose.quat) - float(obj.attrs["grasp_yaw_at"]))
    return Pose((obj.pose.pos[0], obj.pose.pos[1], obj.pose.pos[2] + grasp_offset(obj)),
                turned_down(yaw))


def is_open_container(body) -> bool:
    return body.attrs.get("container") == "open"


def container_spots(container, margin: float = 0.0) -> List[Tuple[float, float]]:
    """Drop points inside an open container: two rows of four along its long side.

    In the container's own frame, at a quarter and three quarters of the inner
    half-widths -- 7 cm apart along a 30 cm bin, 9 cm across it -- then turned
    and moved into the world.  ``margin`` (the thing's own half width) keeps
    all of it inside: a 9 cm duck let go 6.5 cm from a bowl's middle hung on
    the rim (v2, real sizes).
    """
    wall = float(container.attrs.get("wall", 0.0))
    hx = max(0.0, container.size[0] - wall - margin)
    hy = max(0.0, container.size[1] - wall - margin)
    yaw = yaw_of(container.pose.quat)
    turn = (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))
    cx, cy = container.pose.pos[0], container.pose.pos[1]
    out = []
    for fy in (-0.75, -0.25, 0.25, 0.75):
        for fx in (-0.5, 0.5):
            dx, dy, _ = rotate(turn, (fx * hx, fy * hy, 0.0))
            out.append((cx + dx, cy + dy))
    return out


def drop_spot(world: WorldState, obj, target,
              taken: Sequence[Tuple[float, float]] = ()) -> Tuple[float, float]:
    """Where to let go of ``obj`` over ``target``.

    A surface: its centre.  An open container: the drop point furthest from
    everything already in it (and from ``taken``, points an open-loop plan has
    already promised).  Letting everything go over a bin's centre stacked three
    cubes into a tower on Isaac, and the top one came to rest above the rim --
    not in the bin.
    """
    if not is_open_container(target):
        return target.pose.pos[0], target.pose.pos[1]
    occupied = [o.pose.pos[:2] for o in world.objects.values()
                if o.name not in (obj.name, target.name) and inside(o, target)]
    occupied += list(taken)
    spots = container_spots(target, margin=max(obj.size[0], obj.size[1]))
    if not occupied:
        return spots[0]
    return max(spots, key=lambda p: min(math.dist(p, q) for q in occupied))


def placed_on(obj, target) -> bool:
    """Has ``obj`` been put down on (or, for a container, in) ``target``?

    For a container this is the scorer's own test, so a reference can never
    consider an object put away while the score says it is not.
    """
    if is_open_container(target):
        return inside(obj, target)
    return (not obj.is_held
            and obj.pose.planar_distance_to(target.pose) <= max(target.size[0], target.size[1])
            and obj.pose.pos[2] >= target.pose.pos[2])


class PickAndPlacePolicy(ScriptedPolicy):
    """Pick one object up and put it on a target that may change mid-episode.

    ``target_for`` is re-evaluated on every turn, which is exactly the behaviour
    the Responsiveness suite is testing for -- and exactly what the
    ``ignore_change`` negative control removes.
    """

    name = "pick_and_place"

    def __init__(self, obj: str, target_for: Callable[[WorldState], str],
                 arm: str = "robot0/right", lift_height: float = 0.12,
                 place_clearance: float = 0.04) -> None:
        self.obj = obj
        self.target_for = target_for
        self.arm = arm
        self.lift_height = lift_height
        self.place_clearance = place_clearance
        self.phase = "approach"
        self._plan_version = 0
        self.attempts = 0

    def reset(self) -> None:
        self.phase = "approach"
        self._plan_version = 0
        self.attempts = 0

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        # The instruction was revised: drop whatever is still in flight and
        # re-plan on the next turn.  Reacting is the whole point of the
        # Responsiveness suite, so the reference policy has to do it -- and it
        # is exactly what the `ignore_change` negative control removes.
        if world.instruction_version != self._plan_version:
            self._plan_version = world.instruction_version
            return [Cancel(self.arm)]

        # Previous commands are still executing.  Re-planning on top of them is
        # how a policy ends up re-opening the gripper it just closed; a
        # non-blocking interface gets asked again long before its actions finish.
        if ctx.extras.get("executor_busy"):
            return []

        obj = world.obj(self.obj)
        target = world.obj(self.target_for(world))
        arm = world.arm(self.arm)
        # where the *hand* must be for the object's centre to end up just above
        # the target: the object hangs below the grasp point
        x, y = drop_spot(world, obj, target)
        drop = Pose((x, y, target.pose.pos[2] + target.size[2] + obj.size[2]
                     + self.place_clearance + self._held_offset(obj, arm)), GRASP_DOWN)

        if obj.is_held:
            if arm.eef_pose.distance_to(drop) < 0.02:
                self.phase = "release"
                return [SetGripper(self.arm, 0.0), MoveTo(self.arm, retreat_pose(arm)),
                        Wait(1.5, self.arm)]
            self.phase = "carry"
            lift = Pose((obj.pose.pos[0], obj.pose.pos[1],
                         obj.pose.pos[2] + self.lift_height), GRASP_DOWN)
            return [MoveTo(self.arm, lift), MoveTo(self.arm, drop)]

        if placed_on(obj, target):
            return [Done("placed")]

        self.attempts += 1
        if self.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"gave up on {self.obj} after {MAX_GRASP_ATTEMPTS} grasp attempts")]
        self.phase = "approach"
        grasp = grasp_pose(obj)
        above = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + self.lift_height), GRASP_DOWN)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above),
                MoveTo(self.arm, grasp), SetGripper(self.arm, 1.0)]

    @staticmethod
    def _held_offset(obj, arm) -> float:
        """Grasp point above the object's centre -- measured when it is held,
        predicted from its size when it is not."""
        if obj.is_held:
            return arm.eef_pose.pos[2] - obj.pose.pos[2]
        return grasp_offset(obj)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """Pick and place is fully determined by the first frame."""
        obj = world.obj(self.obj)
        target = world.obj(self.target_for(world))
        return _pick_and_place_sequence(self.arm, obj, _spot3(world, obj, target),
                                        target.size[2], self.lift_height,
                                        self.place_clearance) + [Done("planned")]


class SequentialPickAndPlacePolicy(ScriptedPolicy):
    """Move several objects to several targets, one at a time.

    The reference for the Long-Horizon suite, and the shape most other scripted
    references take.
    """

    name = "sequential_pick_and_place"

    def __init__(self, pairs: List[tuple], arm: str = "robot0/right",
                 lift_height: float = 0.12) -> None:
        self.pairs = list(pairs)          # [(object, target), ...]
        self.arm = arm
        self.lift_height = lift_height
        self._index = 0
        self._subs = {}

    def reset(self) -> None:
        self._index = 0
        self._subs = {}

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        out: List[Command] = []
        taken: Dict[str, list] = {}
        for obj_name, target_name in self.pairs:
            obj, target = world.obj(obj_name), world.obj(target_name)
            spot = _spot3(world, obj, target, taken.setdefault(target_name, []))
            taken[target_name].append(spot[:2])
            out += _pick_and_place_sequence(self.arm, obj, spot,
                                            target.size[2], self.lift_height, 0.05)
        return out + [Done("planned")]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        while self._index < len(self.pairs):
            obj_name, target_name = self.pairs[self._index]
            sub = self._subs.setdefault(obj_name, PickAndPlacePolicy(
                obj_name, lambda w, t=target_name: t, arm=self.arm,
                lift_height=self.lift_height))
            obj, target = world.obj(obj_name), world.obj(target_name)
            if placed_on(obj, target):
                self._index += 1
                continue
            cmds = sub.commands(world, ctx)
            if sub.attempts > MAX_GRASP_ATTEMPTS:
                self._index += 1              # give up on this object, go on to the next
                continue
            # a sub-policy finishing one object is not the episode finishing
            return [c for c in cmds if not isinstance(c, Done)]
        return [Done("all placed")]


class PlaceAtPointPolicy(ScriptedPolicy):
    """Pick one object up and put it down at a world point.

    The Fine-Grained reference.  It approaches the point from directly above and
    releases at a fixed clearance -- deliberately simple, because the scripted
    reference exists to show the task is *achievable*, not to be good at it.
    """

    name = "place_at_point"
    SETTLE = 0.5

    def __init__(self, obj: str, point, arm: str = "robot0/right",
                 lift_height: float = 0.13, release_clearance: float = 0.004,
                 tolerance: float = 0.004, carry_z: Optional[float] = None,
                 settle_above: float = 0.0) -> None:
        self.obj = obj
        self.point = tuple(point)
        self.arm = arm
        self.lift_height = lift_height
        #: when set, the hand travels at this height (world z) between the
        #: two points, straight up and straight down at either end -- for a
        #: thing on a raised plate, where lifting ``lift_height`` more is out
        #: of reach and a slanting path clips the plate's edge
        self.carry_z = carry_z
        #: with ``carry_z``: stop over the drop this long before going down, in
        #: a turn of its own -- a thing held high above its middle (a cup by
        #: its knob) swings after a quick carry, and lowered swinging it lands
        #: crooked
        self.settle_above = settle_above
        self._settled = 0
        self.release_clearance = release_clearance
        #: close enough to call it placed; further off, it tries again
        self.tolerance = tolerance
        self._plan_version = 0
        self.attempts = 0

    def reset(self) -> None:
        self._plan_version = 0
        self.attempts = 0
        self._settled = 0

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        obj = world.obj(self.obj)
        arm = world.arm(self.arm)
        down = grasp_pose(obj).quat            # the hand keeps the grasp's turn throughout
        drop = Pose((self.point[0], self.point[1],
                     self.point[2] + reach_below(obj) + self.release_clearance
                     + PickAndPlacePolicy._held_offset(obj, arm)), down)

        if obj.is_held:
            if arm.eef_pose.distance_to(drop) < 0.015:
                up = Pose(retreat_pose(arm).pos, down)
                return [SetGripper(self.arm, 0.0), MoveTo(self.arm, up), Wait(2.5, self.arm)]
            # waiting lets the controller close the last millimetres before
            # the fingers open
            if self.carry_z is not None:
                hand = arm.eef_pose.pos
                above = Pose((drop.pos[0], drop.pos[1], self.carry_z), down)
                # twice at most: at the edge of its reach an arm may stop a
                # centimetre short of the point, and would stop there forever
                if (self.settle_above and self._settled < 2
                        and math.dist(hand[:2], above.pos[:2]) > 0.02):
                    self._settled += 1
                    return [MoveTo(self.arm, Pose((hand[0], hand[1], self.carry_z), down)),
                            MoveTo(self.arm, above), Wait(self.settle_above, self.arm)]
                if self.settle_above:
                    return [MoveTo(self.arm, drop), Wait(self.SETTLE, self.arm)]
                return [MoveTo(self.arm, Pose((hand[0], hand[1], self.carry_z), down)),
                        MoveTo(self.arm, above), MoveTo(self.arm, drop), Wait(self.SETTLE, self.arm)]
            return [MoveTo(self.arm, Pose((obj.pose.pos[0], obj.pose.pos[1],
                                           obj.pose.pos[2] + self.lift_height), down)),
                    MoveTo(self.arm, drop), Wait(self.SETTLE, self.arm)]

        if obj.pose.planar_distance_to(self.point) < self.tolerance:
            return [Done("placed")]

        self.attempts += 1
        if self.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"gave up on {self.obj} after {MAX_GRASP_ATTEMPTS} grasp attempts")]
        grasp = grasp_pose(obj)
        high = (grasp.pos[2] + self.lift_height if self.carry_z is None
                else max(self.carry_z, grasp.pos[2] + 0.03))
        above = Pose((grasp.pos[0], grasp.pos[1], high), grasp.quat)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above),
                MoveTo(self.arm, grasp), SetGripper(self.arm, 1.0)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        obj = world.obj(self.obj)
        return _pick_and_place_sequence(
            self.arm, obj, self.point, 0.0, self.lift_height,
            self.release_clearance) + [Done("planned")]


def turned_down(yaw: float):
    """A straight-down tool orientation turned ``yaw`` about the vertical."""
    return quat_mul((math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)), GRASP_DOWN)


def fold_half_turn(angle: float) -> float:
    """An angle folded into [-90, 90) degrees: a parallel grasp looks the same
    turned half way round."""
    return (angle + math.pi / 2) % math.pi - math.pi / 2


class PlacePosePolicy(ScriptedPolicy):
    """Pick one object up across its own heading and put it down at a point,
    turned to a heading -- the reference for placing things *square* to
    something (a car in its space, a block over its hole).

    ``symmetry`` is the object's own: pi for a car that may face either way,
    pi / 2 for a square block.  Where the object sits in the fingers is taken
    off at the drop, turned with the hand (a grasp a few millimetres off put
    FG2's block on a plate's top, retry after retry).

    ``lift``: how high the hand carries the object -- over its grasp, over its
    place, and after letting go (``HOVER`` unless the task says otherwise).
    """

    name = "place_pose"
    SETTLE = 0.5
    HOVER = 0.10

    #: how many in-place corrections ``refine`` makes before letting go anyway
    MAX_REFINES = 6
    #: ...and how many times it brings a held piece down to its place from the
    #: hover: stopped short of it every time (a centimetre or more above, by
    #: something in the way), it had tried 130 times when the call budget ran
    #: out (CL5, Isaac, v13)
    MAX_LOWERINGS = 4

    def __init__(self, obj: str, point, yaw: float, arm: str = "robot0/right",
                 symmetry: float = math.pi, release_clearance: float = 0.004,
                 tolerance: float = 0.004, yaw_tolerance: float = 0.05,
                 base=None, refine: bool = False, release_tolerance: float = 0.004,
                 insert: float = 0.0, lift: Optional[float] = None) -> None:
        self.obj = obj
        self.point = tuple(point)
        self.yaw = yaw
        self.arm = arm
        self.symmetry = symmetry
        self.release_clearance = release_clearance
        self.tolerance = tolerance
        self.yaw_tolerance = yaw_tolerance
        #: the arm's base, to keep grasp headings on the wrist's comfortable side
        self.base = base
        #: ``refine``: near the drop, correct in place (not from the hover) until
        #: the hand is within ``release_tolerance`` and the object within
        #: ``yaw_tolerance`` of its heading -- for fits of a millimetre or two
        self.refine = refine
        self.release_tolerance = release_tolerance
        #: ``insert``: square over the drop, lower the object this much further
        #: (into its hole) before letting go.  Squeezed hard, a light piece
        #: jumps a few millimetres sideways as the fingers open; a hole's walls
        #: then keep it in line, and it slides through
        self.insert = insert
        self.lift = self.HOVER if lift is None else lift
        self.attempts = 0
        self._refines = 0
        self._lowerings = 0

    def reset(self) -> None:
        self.attempts = 0
        self._refines = 0
        self._lowerings = 0

    def placed(self, world: WorldState, obj) -> bool:
        return (obj.pose.planar_distance_to(self.point) < self.tolerance
                and self._heading_error(obj) < self.yaw_tolerance)

    def _heading_error(self, obj) -> float:
        d = (yaw_of(obj.pose.quat) - self.yaw) % self.symmetry
        return min(d, self.symmetry - d)

    def _hand_yaw(self, yaw: float, at) -> float:
        from ..engines.robots import natural_yaw

        if self.base is None:
            return fold_half_turn(yaw)
        return natural_yaw(self.base, at, yaw)

    def _drop(self, world: WorldState, height: float) -> Pose:
        hand, obj = world.arm(self.arm).eef_pose, world.obj(self.obj)
        # turn the object by the least amount that squares it
        err = (yaw_of(obj.pose.quat) - self.yaw) % self.symmetry
        turn_by = -err if err <= self.symmetry / 2 else self.symmetry - err
        dx, dy = obj.pose.pos[0] - hand.pos[0], obj.pose.pos[1] - hand.pos[1]
        c, s = math.cos(turn_by), math.sin(turn_by)
        x = self.point[0] - (c * dx - s * dy)
        y = self.point[1] - (s * dx + c * dy)
        z = self.point[2] + obj.size[2] + height + hand.pos[2] - obj.pose.pos[2]
        return Pose((x, y, z), turned_down(yaw_of(hand.quat) + turn_by))

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        obj = world.obj(self.obj)
        arm = world.arm(self.arm)
        if obj.is_held:
            low = self._drop(world, self.release_clearance)
            gap = arm.eef_pose.distance_to(low)
            square = not self.refine or self._heading_error(obj) < self.yaw_tolerance
            if (gap < self.release_tolerance and square) or self._refines >= self.MAX_REFINES \
                    or self._lowerings >= self.MAX_LOWERINGS:
                self._refines = self._lowerings = 0
                let_go = [SetGripper(self.arm, 0.0), MoveTo(self.arm, retreat_pose(arm, self.lift)),
                          Wait(2.0, self.arm)]
                if self.insert:
                    let_go.insert(0, MoveTo(self.arm, self._drop(world, self.release_clearance
                                                                 - self.insert)))
                return let_go
            if self.refine and gap < 0.01:
                self._refines += 1
                return [MoveTo(self.arm, low), Wait(self.SETTLE, self.arm)]
            self._lowerings += 1
            return [MoveTo(self.arm, self._drop(world, self.lift)), MoveTo(self.arm, low),
                    Wait(self.SETTLE, self.arm)]
        if self.placed(world, obj):
            return [Done("placed")]
        self.attempts += 1
        if self.attempts > MAX_GRASP_ATTEMPTS:
            return [Done(f"gave up on {self.obj} after {MAX_GRASP_ATTEMPTS} grasp attempts")]
        grasp = grasp_pose(obj)
        yaw = self._hand_yaw(yaw_of(obj.pose.quat), grasp.pos)
        down = turned_down(yaw)
        at = Pose(grasp.pos, down)
        above = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + self.lift), down)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above), MoveTo(self.arm, at),
                SetGripper(self.arm, 1.0), MoveTo(self.arm, above)]

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        obj = world.obj(self.obj)
        grasp = grasp_pose(obj)
        yaw0 = self._hand_yaw(yaw_of(obj.pose.quat), grasp.pos)
        err = (yaw_of(obj.pose.quat) - self.yaw) % self.symmetry
        turn_by = -err if err <= self.symmetry / 2 else self.symmetry - err
        down0, down1 = turned_down(yaw0), turned_down(yaw0 + turn_by)
        z = self.point[2] + obj.size[2] + self.release_clearance + grasp_offset(obj)
        above = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + self.lift), down0)
        low = Pose((self.point[0], self.point[1], z), down1)
        hover = Pose((self.point[0], self.point[1], z + self.lift), down1)
        inserted = [MoveTo(self.arm, Pose((low.pos[0], low.pos[1], z - self.insert), down1))] \
            if self.insert else []
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, above),
                MoveTo(self.arm, Pose(grasp.pos, down0)), SetGripper(self.arm, 1.0),
                MoveTo(self.arm, above), MoveTo(self.arm, hover), MoveTo(self.arm, low),
                *inserted, SetGripper(self.arm, 0.0), MoveTo(self.arm, hover), Wait(2.0, self.arm),
                Done("planned")]


class LookThenActPolicy(ScriptedPolicy):
    """Go and look, and only then decide.

    The Active Perception reference, and the reason it is written this way: it
    reads the deciding information **from the observation**, not from privileged
    state.  A policy that peeked at ground truth would still finish the task and
    would make the `first_frame_only` control look unnecessary -- which is
    exactly the confound the suite exists to avoid.

    ``viewpoints`` are poses to try in order; ``read`` extracts the decision from
    the world *if* the evidence is currently visible, and returns None otherwise.
    """

    name = "look_then_act"

    def __init__(self, obj: str, viewpoints: Sequence[Pose],
                 read: Callable[[WorldState], Optional[str]],
                 target_for: Callable[[str], str], arm: str = "robot0/right",
                 lift_height: float = 0.13,
                 candidates: Optional[Sequence[str]] = None) -> None:
        self.candidates = list(candidates or [])
        self.obj = obj
        self.viewpoints = list(viewpoints)
        self.read = read
        self.target_for = target_for
        self.arm = arm
        self.lift_height = lift_height
        self._decision: Optional[str] = None
        self._next_viewpoint = 0

    def reset(self) -> None:
        self._decision = None
        self._next_viewpoint = 0

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        if self._decision is None:
            self._decision = self.read(world)
        if self._decision is None:
            if self._next_viewpoint >= len(self.viewpoints):
                return [Done("could not obtain the deciding information")]
            pose = self.viewpoints[self._next_viewpoint]
            self._next_viewpoint += 1
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, pose), Wait(0.3, self.arm)]

        return PickAndPlacePolicy(
            self.obj, lambda w, t=self.target_for(self._decision): t,
            arm=self.arm, lift_height=self.lift_height).commands(world, ctx)

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        """From the first frame the evidence is not visible, so the best an
        open-loop plan can do is guess.  That is the finding, not a defect:
        judge this control against chance, not against zero."""
        decision = self.read(world)
        if decision is None:
            if not self.candidates:
                return [Done("cannot plan without looking")]
            decision = self.candidates[0]          # a guess, deliberately unbiased
        obj = world.obj(self.obj)
        target = world.obj(self.target_for(decision))
        return _pick_and_place_sequence(self.arm, obj, _spot3(world, obj, target),
                                        target.size[2], self.lift_height,
                                        0.05) + [Done("guessed")]


def _spot3(world: WorldState, obj, target, taken=()) -> Tuple[float, float, float]:
    """:func:`drop_spot`, at the target's height -- what the open-loop plans take."""
    x, y = drop_spot(world, obj, target, taken)
    return x, y, target.pose.pos[2]


def _pick_and_place_sequence(arm: str, obj, target_pos, target_half_height: float,
                             lift_height: float, clearance: float) -> List[Command]:
    """The canonical open-loop pick-and-place, shared by every plan() above."""
    grasp = grasp_pose(obj)
    above = Pose((grasp.pos[0], grasp.pos[1], grasp.pos[2] + lift_height), grasp.quat)
    drop = Pose((target_pos[0], target_pos[1],
                 target_pos[2] + target_half_height + obj.size[2] + clearance
                 + grasp_offset(obj)), grasp.quat)
    carry = Pose((drop.pos[0], drop.pos[1], drop.pos[2] + lift_height), grasp.quat)
    return [SetGripper(arm, 0.0), MoveTo(arm, above), MoveTo(arm, grasp),
            SetGripper(arm, 1.0), MoveTo(arm, above), MoveTo(arm, carry),
            MoveTo(arm, drop), SetGripper(arm, 0.0), MoveTo(arm, carry), Wait(2.5, arm)]
