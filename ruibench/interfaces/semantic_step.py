"""I1 -- Semantic step.  ``left 5cm`` / ``down`` / ``grasp``.

Action language after Show-Harness (arXiv 2609.10522).  The model speaks most
often of the four interfaces: one short step per turn, the executor turns it
into an increment and servos.  Metric expectation is delegated to the
interpreter; re-entry period is the shortest, which is why it is the interface
most likely to survive the hard Responsiveness levels without a delegation
primitive (v0 §4.1).

TO IMPLEMENT
------------
``parse`` should accept a small, frozen grammar, roughly::

    move <left|right|forward|back|up|down> <n><cm|mm>
    rotate <cw|ccw> <n>deg
    grasp | release
    look <up|down|left|right>        # wrist camera, realised as an arm move
    done

and emit :class:`~ruibench.interfaces.commands.MoveDelta`,
:class:`~ruibench.interfaces.commands.SetGripper`,
:class:`~ruibench.interfaces.commands.Done`.  Directions are resolved in the
**camera frame** -- say so in the prompt, and keep the mapping identical across
tasks; a frame flip silently changes what the suite measures.
"""

from __future__ import annotations

from .base import Expressivity, InterfaceContract, NotImplementedInterface


class SemanticStepInterface(NotImplementedInterface):
    id = "I1"
    name = "semantic step"
    source = "Show-Harness (arXiv 2609.10522)"

    @property
    def contract(self) -> InterfaceContract:
        return InterfaceContract(
            position=Expressivity.SYMBOLIC,
            orientation=Expressivity.SYMBOLIC,
            gripper=Expressivity.SYMBOLIC,
            blocking=True,
            cancellable=True,          # the next step supersedes the current one
            concurrent_arms=False,     # needs an explicit "both arms" word to do BC
            camera_control="via_arm",
            services=("detect", "segment", "depth_at", "pose_of"),
            feedback_delivery="between_turns",
            reentry="per_step",
            notes="shortest re-entry period; metric grounding delegated to the interpreter",
        )
