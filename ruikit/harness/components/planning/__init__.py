"""The planning aids: geometry only.  Nothing here moves the robot.

``plan_*`` does geometry only -- reach, grasp candidates, a path round what was
seen -- from what the cameras measured: a thing's points in the depth image,
never its true shape.  The grasps and paths components (``plan_grasps``,
``plan_motion``) have several implementations: grasps from the estimated box,
from opposite faces of the thing's points, or across their principal axis; a
path that lifts over the highest thing on the way, or one RRT-Connect finds in
an occupancy grid.  There is deliberately **no** task solver (no
Tower-of-Hanoi, river-crossing, chess or tangram planner), no physics predictor
and no success checker: those would answer the question the task is asking.

The tools that ask other models (``model_*``) are extras: ``ruikit.extras.models``.
"""
