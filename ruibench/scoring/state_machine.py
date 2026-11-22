"""Legal-move automata for the rule-bound tasks.

Slot rearrangement and the Tower of Hanoi (WK1) are only well-posed if the
legal moves are written out completely; otherwise the single-buffer constraint
can be side-stepped and the minimum-move formula stops holding (v0 §2,
Reasoning control requirements).  The rules implemented here:

* one object per move -- never carry a stack;
* only the top object of a pile may be picked up;
* an object may only be released at a declared location (slot, peg, buffer);
* size order: a larger object may not be placed on a smaller one.

A violation is recorded with the time and the move that caused it, and the
episode's success predicate is ``Never(violated)`` -- constraint violations
belong inside the success criterion, not in a second metric.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from ..core.types import Vec3, WorldState
from .predicates import Trajectory


@dataclass(frozen=True)
class Location:
    """A declared place an object may be released at."""

    name: str
    center: Vec3
    radius: float = 0.05
    capacity: int = 1


@dataclass
class MoveRecord:
    obj: str
    src: Optional[str]
    dst: Optional[str]
    sim_time: float
    legal: bool = True
    reason: str = ""

    def as_dict(self) -> dict:
        return {"obj": self.obj, "src": self.src, "dst": self.dst,
                "sim_time": self.sim_time, "legal": self.legal, "reason": self.reason}


@dataclass
class LegalMoveMachine:
    locations: Tuple[Location, ...]
    size_order: Mapping[str, int] = field(default_factory=dict)  # bigger number = bigger object
    forbid_larger_on_smaller: bool = True
    allow_offboard: bool = False        # may an object rest outside every location?
    #: the objects the game is played with; empty: every object.  A peg stand is
    #: not a piece, though a ring sits below its centre.
    pieces: Tuple[str, ...] = ()

    def where(self, pos: Vec3) -> Optional[str]:
        for loc in self.locations:
            dx, dy = pos[0] - loc.center[0], pos[1] - loc.center[1]
            if (dx * dx + dy * dy) ** 0.5 <= loc.radius:
                return loc.name
        return None

    def analyse(self, traj: Trajectory) -> List[MoveRecord]:
        """Segment the trajectory into pick/place moves and judge each one."""
        moves: List[MoveRecord] = []
        held: Dict[str, Tuple[str, Optional[str]]] = {}   # obj -> (arm, src location)
        prev = None
        counts = (lambda name: name in self.pieces) if self.pieces else (lambda name: True)
        for w in traj:
            for name, obj in w.objects.items():
                if not counts(name):
                    continue
                if obj.is_held and name not in held:
                    held[name] = (obj.held_by, self.where(obj.pose.pos))
                    # what was on it is judged just *before* the pick: once it
                    # is lifted, whatever lay on it may already have slid off
                    before = prev if prev is not None else w
                    base = before.obj(name)
                    others = [o for o in before.objects.values()
                              if o.name != name and counts(o.name) and not o.is_held
                              and o.pose.planar_distance_to(base.pose) < 0.03
                              and o.pose.pos[2] > base.pose.pos[2]]
                    if others:
                        moves.append(MoveRecord(name, held[name][1], None, w.sim_time,
                                                False, "picked an object that was not on top"))
                    if sum(1 for o in w.objects.values() if o.is_held
                           and o.held_by == obj.held_by) > 1:
                        moves.append(MoveRecord(name, held[name][1], None, w.sim_time,
                                                False, "more than one object in one hand"))
                elif (not obj.is_held) and name in held:
                    _, src = held.pop(name)
                    dst = self.where(obj.pose.pos)
                    rec = MoveRecord(name, src, dst, w.sim_time)
                    if dst is None and not self.allow_offboard:
                        rec.legal, rec.reason = False, "released outside any declared location"
                    elif self.forbid_larger_on_smaller and dst is not None:
                        under = [o for o in w.objects.values()
                                 if o.name != name and counts(o.name)
                                 and o.pose.planar_distance_to(obj.pose) < 0.04
                                 and o.pose.pos[2] < obj.pose.pos[2]]
                        mine = self.size_order.get(name)
                        for o in under:
                            theirs = self.size_order.get(o.name)
                            if mine is not None and theirs is not None and mine > theirs:
                                rec.legal, rec.reason = False, f"placed {name} on smaller {o.name}"
                    moves.append(rec)
            prev = w
        return moves

    def violations(self, traj: Trajectory) -> List[MoveRecord]:
        return [m for m in self.analyse(traj) if not m.legal]

    def move_count(self, traj: Trajectory) -> int:
        return sum(1 for m in self.analyse(traj) if m.dst is not None)
