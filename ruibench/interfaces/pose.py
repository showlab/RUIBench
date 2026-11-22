"""I2 -- Pose.  An absolute 6-DoF target plus a gripper state.

Action language after Robocurve's ``move_to``.  One turn per target: the
executor runs IK, plans and drives the arm there.  I2 and I4 are the cleanest
pair in the benchmark -- **identical callable API, the only difference being
step-by-step calls versus a written program** (v0 §4) -- so anything added here
must be added to I4 as well, and ``tests/ruibench/test_interface_contract.py``
checks that the two call surfaces match item by item.

I2 is also the interface M1 runs on: DI4 "restatement" through I2 is the first
full-pipeline milestone (details §6).

TO IMPLEMENT
------------
Parse a small JSON or function-call form, e.g.::

    move_to(arm="robot0/right", pos=[0.31, 0.12, 0.86], quat=[1,0,0,0])
    gripper(arm="robot0/right", value=1.0)
    done()

emitting :class:`~ruibench.interfaces.commands.MoveTo` /
:class:`~ruibench.interfaces.commands.SetGripper` / ``Done``.  Poses are in the
world frame, metres; state the table height and the reachable box in the prompt.
"""

from __future__ import annotations

from .base import Expressivity, InterfaceContract, NotImplementedInterface


class PoseInterface(NotImplementedInterface):
    id = "I2"
    name = "pose"
    source = "Robocurve move_to"

    @property
    def contract(self) -> InterfaceContract:
        return InterfaceContract(
            position=Expressivity.ABSOLUTE,
            orientation=Expressivity.ABSOLUTE,
            gripper=Expressivity.ABSOLUTE,
            blocking=True,
            cancellable=True,
            concurrent_arms=False,     # sequential calls; see ruibench.executor handles
            camera_control="via_arm",
            services=("detect", "segment", "depth_at", "pose_of"),
            feedback_delivery="between_turns",
            reentry="per_target",
            notes="same callable API as I4 by construction",
        )
