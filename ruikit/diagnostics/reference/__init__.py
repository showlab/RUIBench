"""The scripted reference through each interface: the task's own policy decides,
the interface's tools act.  One module per action abstraction, each with the
reference that reads the simulator's coordinates and one that reads only what
the perception tools estimate (``Grounded*``)."""

from __future__ import annotations

from ruikit.diagnostics.reference.delta import DeltaExpressor, GroundedDelta
from ruikit.diagnostics.reference.joint import GroundedJoint, JointExpressor
from ruikit.diagnostics.reference.point import GroundedPoint, PointExpressor
from ruikit.diagnostics.reference.pose import GroundedPose, PoseExpressor
from ruikit.diagnostics.reference.program import CodeExpressor, GroundedCode, ProgramOver
from ruikit.diagnostics.reference.skill import GroundedSkill, SkillExpressor
from ruikit.diagnostics.reference.step import GroundedStep, StepExpressor

EXPRESSORS = {"pose": PoseExpressor, "step": StepExpressor, "point": PointExpressor,
              "code": CodeExpressor, "delta": DeltaExpressor, "skill": SkillExpressor,
              "joint": JointExpressor,
              "code-joint": lambda: ProgramOver(JointExpressor()),
              "code-delta": lambda: ProgramOver(DeltaExpressor()),
              "code-step": lambda: ProgramOver(StepExpressor()),
              "code-point": lambda: ProgramOver(PointExpressor()),
              "code-skill": lambda: ProgramOver(SkillExpressor())}


#: an interface without a named form (a file of its own) is expressed as its
#: action abstraction is
DIALECT_STYLE = {"pose": "pose", "step": "step", "point": "point", "program": "code",
                 "delta": "delta", "skill": "skill", "joint": "joint"}


def style_of(profile_id: str) -> str:
    """The expressor style for an interface id, composed ones included: the
    action abstraction of a per-action interface; ``code`` for the program over
    the pose, ``code-<abstraction>`` for a program over another."""
    from ruikit.interfaces.catalog import form_of

    form, action = form_of(profile_id.split("@")[0].split("+")[0])
    if form == "per_action":
        return action
    if form == "program":
        return "code" if action == "pose" else f"code-{action}"
    from ruikit.interfaces import load_profile

    dialect = load_profile(profile_id).dialect
    if dialect not in DIALECT_STYLE:
        raise KeyError(f"no expressor for {profile_id!r} (dialect {dialect!r})")
    return DIALECT_STYLE[dialect]


GROUNDED = {"pose": GroundedPose, "step": GroundedStep, "point": GroundedPoint,
            "delta": GroundedDelta, "skill": GroundedSkill, "joint": GroundedJoint,
            "code": GroundedCode,
            # programs over the other action abstractions: their calls made from programs
            "code-joint": lambda: ProgramOver(GroundedJoint()),
            "code-delta": lambda: ProgramOver(GroundedDelta()),
            "code-step": lambda: ProgramOver(GroundedStep()),
            "code-point": lambda: ProgramOver(GroundedPoint()),
            "code-skill": lambda: ProgramOver(GroundedSkill())}
