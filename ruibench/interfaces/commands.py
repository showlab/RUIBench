"""The shared command vocabulary.

Every interface -- semantic step, pose, pointing, program -- compiles its own
action language down to *these* commands, and the executor below knows nothing
else.  That is what makes "same robot, same controller, same perception, only the
interface changes" true in code rather than only in the paper (design doc v0 §4).

If you are implementing one of the four interfaces: your job is
``str -> list[Command]``.  Do not add engine calls; if you need something the
vocabulary cannot express, that is a finding -- record it as a coverage gap
(:class:`ruibench.core.errors.CoverageGap`) instead of widening one interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Optional, Tuple, Union

from ..core.types import Pose, Vec3


@dataclass(frozen=True)
class MoveTo:
    """Move an arm's end effector to an absolute world pose."""
    arm: str
    pose: Pose
    speed: float = 1.0


@dataclass(frozen=True)
class MoveJoints:
    """Move an arm's joints to target angles (radians, base to wrist; the
    robot's own order, ``engines.kinematics``), each along its own way at the
    shared joint speed.  The joint-level interface compiles to this (v1.17)."""
    arm: str
    joints: Tuple[float, ...]


@dataclass(frozen=True)
class MoveDelta:
    """Move an arm by a relative offset, optionally in the camera frame.

    The semantic-step interface ("move left 5cm", "go down") is expected to
    compile to this; the pose interface is expected never to emit it.
    """
    arm: str
    dpos: Vec3 = (0.0, 0.0, 0.0)
    drot: Vec3 = (0.0, 0.0, 0.0)
    frame: str = "world"        # world | camera | eef


@dataclass(frozen=True)
class SetGripper:
    arm: str
    value: float                # 0 open, 1 closed; from 0.5 up it closes on what is
                                # there, below it opens to (1 - value) of the way


@dataclass(frozen=True)
class Wait:
    """Let the world run for a while without commanding anything.

    A pause at its place among the commands: what comes after it starts once
    what came before it is done and ``seconds`` more have passed -- "move down,
    wait 0.3 s, open" lets the hand settle before it opens.  Without an arm it
    holds every arm (each finishes what came before, then all pause
    together); with one, only that arm's commands pause.  On its own, the
    world runs that long.

    Legal for every interface.  It is also how "wait for the bowl to come back"
    is expressed without a delegation primitive -- we deliberately do not
    provide ``watch(condition)`` (v0 §4.1: that belongs to the Approach layer).
    """
    seconds: float
    arm: Optional[str] = None


@dataclass(frozen=True)
class Cancel:
    """Abort in-flight motion.  Whether an interface may issue this is part of
    its contract, and it is one of the mechanisms Responsiveness measures."""
    arm: Optional[str] = None       # None cancels every arm
    handle_id: Optional[str] = None


@dataclass(frozen=True)
class Done:
    """The model declares it is finished.

    This ends the episode but **never** decides success: success comes from the
    physical trajectory and the task rules (v0 §2.1).
    """
    note: str = ""


Command = Union[MoveTo, MoveJoints, MoveDelta, SetGripper, Wait, Cancel, Done]

def describe(cmd: Command) -> str:
    """One-line human-readable form, used in logs and the episode viewer."""
    if isinstance(cmd, MoveTo):
        x, y, z = cmd.pose.pos
        return f"move_to({cmd.arm}, [{x:.3f}, {y:.3f}, {z:.3f}])"
    if isinstance(cmd, MoveJoints):
        return f"move_joints({cmd.arm}, {tuple(round(v, 3) for v in cmd.joints)})"
    if isinstance(cmd, MoveDelta):
        return f"move_delta({cmd.arm}, {tuple(round(v, 3) for v in cmd.dpos)}, frame={cmd.frame})"
    if isinstance(cmd, SetGripper):
        return f"gripper({cmd.arm}, {cmd.value:.2f})"
    if isinstance(cmd, Wait):
        return f"wait({cmd.arm + ', ' if cmd.arm else ''}{cmd.seconds:.2f}s)"
    if isinstance(cmd, Cancel):
        return f"cancel({cmd.arm or 'all'})"
    if isinstance(cmd, Done):
        return f"done({cmd.note!r})" if cmd.note else "done()"
    return repr(cmd)
