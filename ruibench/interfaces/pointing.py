"""I3 -- Pointing.  A pixel in the image plus an action type.

Action language after MOKA / PIVOT / Gemini-ER.  The point is lifted to a pose
by the depth service and then executed by **exactly the same executor as I2**,
so the measured difference between I2 and I3 is the grounding path, not the
controller.

THE KNOWN HOLE, DECIDE IT BEFORE WRITING CODE
---------------------------------------------
A point plus an action type cannot express a 6-DoF orientation.  WK3 (standing
placement) and the observation turns of DS1-DS2 all
need one.  TODO §3 makes this an explicit decision, not an accident:

* either extend the language (e.g. a second point giving an approach direction,
  or a discrete set of canonical approaches) -- and then say so in the
  contract, in the prompt and in the paper appendix;
* or declare the task a **coverage gap** via :meth:`coverage_gap`, which is
  reported in its own column and never scored as a zero.

Whichever you choose, it must be the same choice for every task in a suite.

TO IMPLEMENT
------------
Parse e.g. ``pick(x=412, y=233)`` / ``place(x=190, y=280)`` /
``look(x=..., y=...)``, call the depth service to lift ``(x, y)`` to a world
point, and emit ``MoveTo`` + ``SetGripper``.  Record the lift error separately:
"pixel -> 3D grounding error" is one of the paper's three vision-side
contributions (v0 §8).
"""

from __future__ import annotations

from typing import Optional

from .base import Expressivity, InterfaceContract, NotImplementedInterface


class PointingInterface(NotImplementedInterface):
    id = "I3"
    name = "pointing"
    source = "MOKA / PIVOT / Gemini-ER"

    #: Tasks whose orientation requirement this action language cannot express.
    #: Keep the list here, in one place, so the coverage-gap column is auditable.
    ORIENTATION_DEPENDENT_TASKS = ("WK3",)

    @property
    def contract(self) -> InterfaceContract:
        return InterfaceContract(
            position=Expressivity.PIXEL,
            orientation=Expressivity.NONE,     # <- revisit if you extend the language
            gripper=Expressivity.SYMBOLIC,
            blocking=True,
            cancellable=True,
            concurrent_arms=False,
            camera_control="via_arm",
            services=("detect", "segment", "depth_at", "pose_of"),
            feedback_delivery="between_turns",
            reentry="per_target",
            notes="target must be visible to be pointed at; orientation is the open question",
        )

    def coverage_gap(self, task) -> Optional[str]:
        task_id = getattr(task, "id", str(task))
        if task_id in self.ORIENTATION_DEPENDENT_TASKS:
            return ("a pixel plus an action type cannot express the required "
                    "end-effector orientation")
        return None
