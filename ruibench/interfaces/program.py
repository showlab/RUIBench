"""I4 -- Program.  A block of code calling the same API as I2.

Action language after CaP-X.  The model writes a program; it runs to
completion, **but it is allowed to execute a single primitive and return**,
which is how a program interface can imitate step-by-step control when the
world is moving (v0 §1.3 -- "plans well" and "cannot change its mind" are not
necessarily correlated).

The API exposed to the program must be **item-for-item identical to I2**.  That
identity is the benchmark's cleanest contrast, so it is enforced by a test, not
by discipline: see ``tests/ruibench/test_interface_contract.py``.

TO IMPLEMENT
------------
* a sandbox (no filesystem, no network, no ``import``) with a wall-clock and
  step budget;
* the callable surface: ``move_to``, ``gripper``, ``wait``, ``cancel``,
  ``detect``, ``segment``, ``depth_at``, ``pose_of``, ``feedback()``, ``done``;
* ``feedback()`` must be readable from inside the program -- the 1-bit
  correct/wrong signal reaches every interface including this one;
* ``parse`` returns the commands the program emitted this turn, so the executor
  and the clock stay the single source of truth about time.
"""

from __future__ import annotations

from .base import Expressivity, InterfaceContract, NotImplementedInterface


class ProgramInterface(NotImplementedInterface):
    id = "I4"
    name = "program"
    source = "CaP-X (arXiv 2603.22435)"

    #: The callable surface the sandbox must expose.  Kept next to I2's contract
    #: so any drift shows up as a failing test rather than as a silent advantage.
    API_SURFACE = (
        "move_to", "gripper", "wait", "cancel",
        "detect", "segment", "depth_at", "pose_of", "feedback", "done",
    )

    @property
    def contract(self) -> InterfaceContract:
        return InterfaceContract(
            position=Expressivity.ABSOLUTE,
            orientation=Expressivity.ABSOLUTE,
            gripper=Expressivity.ABSOLUTE,
            blocking=False,            # the program keeps running while the world moves
            cancellable=True,
            concurrent_arms=True,      # two handles + wait_any / wait_all
            camera_control="via_arm",
            services=("detect", "segment", "depth_at", "pose_of"),
            feedback_delivery="inline",
            reentry="per_program",
            notes="may execute one primitive and return, emulating step-by-step re-entry",
        )
