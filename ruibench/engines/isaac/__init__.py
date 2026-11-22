"""The Isaac Sim backend: photorealistic rendering, PhysX, USD assets.

    ruibench doctor --hint              # how to install Isaac Sim and Isaac Lab
    ruibench setup                      # fetch and prepare robots, room, objects
    ruibench doctor                     # check the install and the asset server
    ruibench run DI4 --engine isaac     # run for real

Nothing in this package imports Isaac at module level, so the rest of RUIBench
keeps working where Isaac is not installed.
"""

from . import app, assets
from .engine import DEFAULT_CONTROL_HZ, DEFAULT_SIM_HZ, IK_CONTROLLER, IsaacEngine

__all__ = ["IsaacEngine", "app", "assets", "IK_CONTROLLER",
           "DEFAULT_SIM_HZ", "DEFAULT_CONTROL_HZ"]
