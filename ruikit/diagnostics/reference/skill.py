"""The reference through the semantic skills."""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from ruibench.core.geometry import rotate
from ruibench.core.types import Pose
from ruibench.interfaces.commands import (
    Cancel,
    MoveDelta,
    MoveTo,
    SetGripper,
)
from ruikit.actions.skill import SIDE_IN_M
from ruikit.diagnostics.reference.base import Expressor, _arm_name, _grip_args
from ruikit.diagnostics.reference.grounding import _targets, _with_notes
from ruikit.env.station import rpy_deg_from_quat

#: a pose tipped this far from pointing down (its roll, at pitch 0) is a tip the
#: skills make on purpose (``skill_turn`` axis 'tip'); less is the sway of an
#: arm carrying something, which the skills, pointing down, leave out
TIPPED_DEG = 20.0


def _level(at: Pose) -> bool:
    """The fingers point level -- a hand taking something from the side -- not down."""
    return abs(rotate(at.quat, (0.0, 0.0, 1.0))[2]) < 0.5


def _tip(at: Pose) -> float:
    """How far ``at`` is tipped from pointing down, about the hand's own x axis
    (its roll at pitch 0); 0 when it is not a tip or barely one."""
    roll, pitch, _ = rpy_deg_from_quat(at.quat)
    return roll if abs(pitch) < 10.0 and abs(roll) >= TIPPED_DEG else 0.0


class SkillExpressor(Expressor):
    """The skill abstraction: the reference's acts, not its moves.  Fingers closing
    where a thing can be taken become ``skill_pick`` there -- from the side where
    the reference's hand comes level; closing on nothing takeable,
    ``skill_press``; opening on something held, ``skill_place`` where it was let
    go.  Every move between is left to the skills, which approach and lift off on
    their own, and point the fingers down; a move to a tipped hand is that move
    and then ``skill_turn``'s tip (v2.5)."""
    style = "skill"

    def __init__(self, *a, **k) -> None:
        super().__init__(*a, **k)
        #: arm -> a pick from the side just made: the level moves after it are its own
        self.side: Dict[str, bool] = {}
        #: the turn's acts in full: its waits split them for ``express``, and the
        #: lift after a side grasp's fingers close lies past the wait that lets
        #: them settle (CR4's reference)
        self.turn: List = []
        self.aim: Dict[str, Pose] = {}          # arm -> the pose a gripper event acts at
        self.held: Dict[str, bool] = {}
        self.lifted: Dict[str, Pose] = {}       # arm -> where a skill acted and lifted from
        self.before: Dict[str, Pose] = {}       # arm -> where its last act was

    @staticmethod
    def graspable(ep, at: Pose) -> Optional[str]:
        from ruibench.engines.base import collides, in_grasp

        world = ep.oracle().world()
        best, best_d = None, None
        for body in ep.scene.bodies:
            if body.fixed or not collides(body) or (body.attrs or {}).get("pressable") \
                    or body.name not in world.objects:
                continue
            pose = world.objects[body.name].pose
            if in_grasp(body, pose, at.pos):
                d = math.dist(pose.pos, at.pos)
                if best_d is None or d < best_d:
                    best, best_d = body.name, d
        return best

    def pick(self, ep, arm: str, at: Pose, body: str) -> dict:
        if _level(at):
            return self.side_pick(ep, arm, at)
        d = ep.station.describe_pose(at)
        x, y, z = d["xyz_m"]
        # a point picks 2 cm below it: give the grip point 2 cm higher
        return {"arm": arm, "target": {"xyz_m": [x, y, round(z + 0.02, 4)]},
                "yaw_deg": d["rpy_deg"][2]}

    @staticmethod
    def side_pick(ep, arm: str, at: Pose) -> dict:
        """The reference's hand, level at ``at``: a pick from the side pointing
        where its fingers point, at the point its tool centre stops short of
        by the skill's own reach past the target."""
        point = rotate(at.quat, (0.0, 0.0, 1.0))
        heading = math.atan2(point[1], point[0])
        x, y, z = ep.station.describe_pose(at)["xyz_m"]
        return {"arm": arm, "approach": "side", "yaw_deg": round(math.degrees(heading), 2),
                "target": {"xyz_m": [round(x - math.cos(heading) * SIDE_IN_M, 4),
                                     round(y - math.sin(heading) * SIDE_IN_M, 4), z]}}

    def place(self, ep, arm: str, at: Pose) -> dict:
        return {"arm": arm, "target": {"xyz_m": ep.station.describe_pose(at)["xyz_m"]},
                "height_m": 0.0, "yaw_deg": ep.station.describe_pose(at)["rpy_deg"][2]}

    def press(self, ep, arm: str, at: Pose) -> dict:
        return {"arm": arm, "target": {"xyz_m": ep.station.describe_pose(at)["xyz_m"]},
                "depth_m": 0.0}

    def move(self, ep, arm: str, at: Pose) -> dict:
        d = ep.station.describe_pose(at)
        return {"arm": arm, "target": {"xyz_m": d["xyz_m"]}, "yaw_deg": d["rpy_deg"][2]}

    #: a move this far up from where the fingers acted, over the same spot, is
    #: the lift a pick, a place or a press makes on its own
    LIFT = 0.04

    def _rise(self, frm: Pose, to: Pose) -> bool:
        return (to.pos[2] - frm.pos[2] >= self.LIFT
                and math.hypot(to.pos[0] - frm.pos[0], to.pos[1] - frm.pos[1]) < 0.02)

    def _goto(self, ep, arm: str, frm: Pose, to: Pose) -> List[tuple]:
        """A move between acts: a turn where the hand is, else a skill_move --
        and, to a tipped hand, the tip after it (``TIPPED_DEG``)."""
        tip_to, tip_from = _tip(to), _tip(frm)
        if math.dist(frm.pos, to.pos) < 0.003:
            turn = ((rpy_deg_from_quat(to.quat)[2] - rpy_deg_from_quat(frm.quat)[2] + 180.0)
                    % 360.0) - 180.0
            calls = [("skill_turn", {"arm": arm, "angle_deg": round(turn, 2)})] if abs(turn) > 0.5 else []
            if (tip_to or tip_from) and abs(tip_to - tip_from) > 0.5:
                calls.append(("skill_turn", {"arm": arm, "angle_deg": round(tip_to - tip_from, 2),
                                             "axis": "tip"}))
            return calls
        calls = [("skill_move", self.move(ep, arm, to))]
        if tip_to:
            calls.append(("skill_turn", {"arm": arm, "angle_deg": round(tip_to, 2), "axis": "tip"}))
        return calls

    def on_commands(self, ep, cmds) -> None:
        super().on_commands(ep, cmds)
        self.turn = [c for c in cmds if isinstance(c, (MoveTo, MoveDelta, SetGripper, Cancel))]

    def _after(self, c) -> list:
        """This arm's acts after ``c`` in the whole turn, past its waits."""
        for k, x in enumerate(self.turn):
            if x is c:
                return [y for y in self.turn[k + 1:] if getattr(y, "arm", None) == c.arm]
        return []

    @staticmethod
    def _side_ahead(later) -> bool:
        """The reference's next grasp (in ``later``, this arm's acts to come) is
        made with the hand level: a pick from the side, which approaches by itself."""
        for j, x in enumerate(later):
            if isinstance(x, SetGripper):
                before = later[j - 1] if j else None
                return x.value >= 0.5 and isinstance(before, MoveTo) and _level(before.pose)
        return False

    def express(self, ep, acts):
        world = ep.oracle().world()
        acts = list(acts)
        per: Dict[str, List[tuple]] = {}
        hand: Dict[str, Pose] = {}            # where each arm will be, as the calls go
        for i, c in enumerate(acts):
            arm = _arm_name(c.arm) if getattr(c, "arm", None) else None
            if isinstance(c, Cancel):
                per.setdefault(arm or "*", []).append(("robot_stop", {"arm": arm} if arm else {}))
                continue
            here = hand.get(arm) or world.arm(c.arm).eef_pose
            later = [x for x in acts[i + 1:] if getattr(x, "arm", None) == c.arm]
            if isinstance(c, (MoveTo, MoveDelta)):
                to = here.translated(c.dpos) if isinstance(c, MoveDelta) else c.pose
                if later and isinstance(later[0], SetGripper):
                    self.aim[arm] = to                  # the gripper event takes it
                    hand[arm] = to
                    continue
                if not self.held.get(arm) and self._side_ahead(later):
                    hand[arm] = to                      # a side pick's approach: its own
                    continue
                if self.lifted.get(arm) and self._rise(self.lifted[arm], to):
                    self.lifted.pop(arm)                # the skill lifted already
                    hand[arm] = here
                    continue
                if self.side.get(arm) and _level(to):
                    self.lifted.pop(arm, None)          # a side pick lifts and backs out level
                    hand[arm] = to
                    continue
                self.side.pop(arm, None)
                self.lifted.pop(arm, None)
                per.setdefault(arm, []).extend(self._goto(ep, arm, here, to))
                hand[arm] = to
                continue
            if not isinstance(c, SetGripper):
                continue
            aim = self.aim.pop(arm, None)
            at = aim or here
            nxt = next((x for x in later if isinstance(x, (MoveTo, MoveDelta))), None)
            if nxt is None and aim is not None and _level(aim):
                nxt = next((x for x in self._after(c) if isinstance(x, (MoveTo, MoveDelta))), None)
            rises = nxt is not None and self._rise(at, at.translated(nxt.dpos) if isinstance(
                nxt, MoveDelta) else nxt.pose)
            held = self.held.get(arm, world.arm(c.arm).holding is not None)
            came = aim is not None and math.dist(aim.pos, (self.before.get(arm) or here).pos) >= 0.02
            calls = per.setdefault(arm, [])
            if c.value >= 0.5 and not held and came and rises:
                body = self.graspable(ep, at)
                if body is not None:
                    calls.append(("skill_pick", self.pick(ep, arm, at, body)))
                    self.held[arm] = True
                    self.side[arm] = _level(at)
                else:
                    calls.append(("skill_press", self.press(ep, arm, at)))
                self.lifted[arm] = at
            elif c.value < 0.5 and held and came and rises:
                calls.append(("skill_place", self.place(ep, arm, at)))
                self.held[arm] = False
                self.lifted[arm] = at
            else:
                if aim is not None:
                    calls.extend(self._goto(ep, arm, self.before.get(arm) or here, aim))
                calls.append(("robot_gripper", _grip_args(ep, c)))
                self.held[arm] = c.value >= 0.5 and (held or self.graspable(ep, at) is not None)
            hand[arm] = at
            self.before[arm] = at
        k = 0
        while True:                      # the arms' k-th calls together, each arm's in order
            row = [calls[k] for calls in per.values() if k < len(calls)]
            if not row:
                return
            ep.call_batch(row)
            if ep.over:
                return
            k += 1


class GroundedSkill(_with_notes(SkillExpressor)):
    """The skill abstraction on estimates: a pick names the thing by the id the
    detector gave it (the skill then estimates its own grip, as a model's
    call would make it do); a place and a press go where the estimate puts
    the reference's point."""

    def on_commands(self, ep, cmds):
        super().on_commands(ep, cmds)
        targets = _targets(self, ep, cmds)
        if targets:
            self.grounder.prepare(targets)

    def _grounded(self, ep, at: Pose):
        got = self.grounder.ground(at)
        if got is None:
            self.unexpressed.append("nothing estimated to act by")
            return at, None
        self.note(ep, got.note)
        return got.pose, got.seen

    def pick(self, ep, arm, at, body):
        pose, seen = self._grounded(ep, at)
        if _level(at):
            # from the side: at the estimated point, the way the reference's
            # fingers point (an id would grip the estimate's near side instead,
            # at its middle -- the reference's edge is a point of its own)
            return self.side_pick(ep, arm, Pose(pose.pos, at.quat))
        if seen is not None and seen.body == body:
            return {"arm": arm, "target": {"frame_id": seen.frame_id, "id": seen.obj_id}}
        return super().pick(ep, arm, pose, body)

    def place(self, ep, arm, at):
        return super().place(ep, arm, self._grounded(ep, at)[0])

    def press(self, ep, arm, at):
        return super().press(ep, arm, self._grounded(ep, at)[0])
