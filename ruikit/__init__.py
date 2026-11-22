"""RUIKit: Robot Agents for RUIBench, built from interchangeable parts.

A Robot Agent is a model, a Harness, an interface and a perception backend.
An interface is an interaction form applied to an action abstraction; the
agent acts on the robot only through the interface's tools, every answer
computed from the cameras' pictures and the robot's own readings.

    import ruikit
    agent = ruikit.Agent(model="openai:<model>", interface="program-pose")   # with the aids
    ruikit.run(agent, task="CL2", level="easy", seed=15)

    ep = ruikit.Episode("WK1", profile="rui.pose/v1", engine="mock")   # one episode, by hand
    ep.call("camera_look", {"cameras": ["front"]})

``docs/ruikit/`` describes the parts and how to add one.
"""

from ruikit.agent import Agent, run
from ruikit.conditions import Condition
from ruikit.core import ToolError, ToolResult, param, tool
from ruikit.env import Budget, Episode
from ruikit.interfaces import load_profile

__all__ = ["Agent", "Budget", "Condition", "Episode", "ToolError", "ToolResult", "load_profile",
           "param", "run", "tool"]
__version__ = "1.0.0"
