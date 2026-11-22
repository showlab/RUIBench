"""The real station: the lab's Franka Panda, its NUC and its RealSense cameras.

    python -m ruibench.engines.franka.check        # can this machine reach the robot and cameras?
    python -m ruibench.engines.franka.calibrate    # the cameras and the table, from a board
    python -m ruikit.experiments.real ...          # episodes, with an operator at the robot

The engine real-robot episodes run on (``FrankaEngine``).  Everything an
episode does above the engine -- interfaces, tools, the executor, perception,
the Harnesses -- is what it is in simulation; what is real is decided here:
the clock is the wall clock, the pictures are the cameras', the arm's
readings the robot's, and nothing knows where the things on the table are.

The robot's model (``model``), the station's settings (``rig.yaml``) and its
calibration (``calibration.json``) are measured on the lab's robot and change
with it; no simulation result depends on them, so none of this package is in
the frozen benchmark's fingerprint (``ruibench.tasks.freeze``).  Each
real-robot result records its commit and the calibration it ran with.

Nothing here imports the robot's or the cameras' libraries at module level
(``zerorpc``, ``pyrealsense2``): the rest of RUIBench works where they are not
installed.
"""

from . import model
from .engine import FAKE_ENV, FrankaEngine, RobotFault, home
from .scene import station_scene

__all__ = ["FAKE_ENV", "FrankaEngine", "RobotFault", "home", "model", "station_scene"]
