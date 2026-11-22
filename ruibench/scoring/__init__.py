"""Scoring: predicates, milestones, legal-move automata, failure labels.

All of it is a pure function of a trajectory -- no engine, no clock, no
randomness.  (A stored log keeps positions and headings only, not everything a
predicate reads, so ``ruibench rescore`` checks stored scores for consistency
rather than scoring again.)
"""

from .failure import FailureLabel
from .milestones import Milestone, MilestoneTracker, replay
from .predicates import (
    AllOfT, And, AnyOfT, AtRest, Before, Custom, Ever, Finally, Held, HoldsFor, In,
    InstructionIs, JointOpen, KnownBefore, NearPoint, Never, Not, NotT, On, Or,
    RelativeStatic, StatePredicate, Trajectory, TrajectoryPredicate, Upright,
    VisibleFrom, inside,
)
from .placements import LastKCorrect, Placement, bin_of, segment_placements
from .scorer import EpisodeScore, Scorer
from .state_machine import LegalMoveMachine, Location, MoveRecord

__all__ = [
    "FailureLabel", "Milestone", "MilestoneTracker", "replay",
    "StatePredicate", "TrajectoryPredicate", "Trajectory",
    "And", "Or", "Not", "On", "In", "NearPoint", "Held", "AtRest",
    "RelativeStatic", "JointOpen", "Upright", "VisibleFrom", "InstructionIs", "Custom",
    "Finally", "Ever", "Never", "HoldsFor", "Before", "KnownBefore", "AllOfT", "AnyOfT", "NotT",
    "Scorer", "EpisodeScore", "LegalMoveMachine", "Location", "MoveRecord",
    "LastKCorrect", "Placement", "bin_of", "segment_placements", "inside",
]
