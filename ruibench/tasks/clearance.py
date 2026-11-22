"""Can the gripper get at what a task asks it to pick up?

A straight-down grasp needs room for more than the fingers.  The Panda hand is
20 cm long along the direction its fingers open, and its underside passes 4 cm
above the fingertips.  Sorting bins whose walls stood inside that footprint
stopped every descent 7 cm short on Isaac: a scene bug that looked like a
controller bug, and that no amount of controller tuning would have fixed.  So
scenes are checked against the hand's envelope -- geometrically, in CI, with
no simulator.

The envelope is **measured**, not taken from a datasheet: the collider boxes
of the Isaac backend's Panda in the hand frame (2026-09-16), written here
around the tool centre point for a grasp straight down with the fingers
opening along world y.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ..core.types import Pose
from ..engines.base import BodySpec, SceneSpec, collides, passes
from ..engines.sight import Box, box_bounds, occluder_boxes

#: The Franka hand's envelope under its v0 names (the FR3's hand;
#: ``engines.robots.FR3`` holds each model's own numbers).  The hand body: half size across (x) and along (y) the finger
#: direction, and the height of its underside above the tool centre point.
HAND_HALF_X = 0.032
HAND_HALF_Y = 0.104
HAND_BOTTOM = 0.041
#: An open finger: inner face at the opening (4 cm from the centre), 2.6 cm
#: thick, 2 cm wide, and reaching 5 mm below the tool centre point.
FINGER_INNER = 0.040
FINGER_OUTER = 0.066
FINGER_HALF_X = 0.010
FINGER_BOTTOM = -0.005
#: Touching is not clearing: this much air is required everywhere.
MARGIN = 0.005

def bounds(body: BodySpec) -> Box:
    """World-axis-aligned bounds of an upright body, where it starts."""
    return box_bounds(body.shape, body.size, body.pose)


def turned_footprint(x: Tuple[float, float], y: Tuple[float, float], yaw: float):
    """The world-axis-aligned bounds of a footprint turned by ``yaw``."""
    c, s = math.cos(yaw), math.sin(yaw)
    corners = [(c * px - s * py, s * px + c * py) for px in x for py in y]
    return ((min(p[0] for p in corners), max(p[0] for p in corners)),
            (min(p[1] for p in corners), max(p[1] for p in corners)))


def gripper_boxes(grasp, model=None, yaw: float = 0.0,
                  opening: Optional[float] = None) -> List[Tuple[str, Box]]:
    """The hand and both open fingers, for a straight-down grasp at ``grasp``
    -- opened ``opening`` either side of the middle, if not all the way (a
    body's ``grasp_open``: CL6's knight, taken among the pieces).

    Each box runs from its lowest point upward without limit: anything taller
    than that point, inside the footprint, is in the way of the descent.  The
    parts come from the robot model (the FR3's by default), turned by the
    grasp's yaw.
    """
    from ..engines.robots import FR3

    model = model or FR3
    gx, gy, gz = grasp
    out = []
    for part in model.open_hand:
        y = part.y
        if part.name == "finger" and opening is not None:
            # opened only so far: each finger nearer the middle by what it
            # does not open
            inward = model.max_opening / 2 - opening
            y = (y[0] - inward, y[1] - inward) if y[0] >= 0 else (y[0] + inward, y[1] + inward)
        (x0, x1), (y0, y1) = turned_footprint(part.x, y, yaw)
        out.append((part.name, ((gx + x0, gx + x1), (gy + y0, gy + y1), (gz + part.bottom, math.inf))))
    return out


def gripper_parts(grasp, model=None, yaw: float = 0.0, opening: Optional[float] = None):
    """The hand's parts as ``gripper_boxes`` has them, but each as its own
    turned rectangle -- (name, four corners, lowest point) -- not as the
    upright box around it: turned 45 degrees, the box round the 21 cm hand is
    17 cm square, and a knight among chessmen (CL6) was never clear of it."""
    from ..engines.robots import FR3

    model = model or FR3
    gx, gy, gz = grasp
    c, s = math.cos(yaw), math.sin(yaw)
    out = []
    for part in model.open_hand:
        y = part.y
        if part.name == "finger" and opening is not None:
            inward = model.max_opening / 2 - opening
            y = (y[0] - inward, y[1] - inward) if y[0] >= 0 else (y[0] + inward, y[1] + inward)
        corners = [(gx + c * px - s * py, gy + s * px + c * py) for px in part.x for py in y]
        out.append((part.name, [corners[0], corners[1], corners[3], corners[2]], gz + part.bottom))
    return out


def _turned_overlap(corners, bottom: float, box: Box, margin) -> bool:
    """Does a turned rectangle, reaching up without limit from ``bottom``,
    come within ``margin`` of an upright box?  Separating axes: the box's
    two and the rectangle's two."""
    m = margin if isinstance(margin, tuple) else (margin,) * 3
    (x0, x1), (y0, y1), (_, z1) = box
    if not bottom < z1 + m[2]:
        return False
    theirs = [(x0 - m[0], y0 - m[1]), (x1 + m[0], y0 - m[1]),
              (x1 + m[0], y1 + m[1]), (x0 - m[0], y1 + m[1])]
    axes = [(1.0, 0.0), (0.0, 1.0)]
    for (ax, ay), (bx, by) in zip(corners, corners[1:] + corners[:1]):
        length = math.hypot(bx - ax, by - ay)
        if length > 1e-12:
            axes.append(((by - ay) / length, -(bx - ax) / length))
    for ux, uy in axes:
        mine = [x * ux + y * uy for x, y in corners]
        other = [x * ux + y * uy for x, y in theirs]
        if max(mine) <= min(other) or max(other) <= min(mine):
            return False
    return True


def scene_model(scene: SceneSpec):
    """The robot model a scene's grasps are checked against."""
    from ..engines.robots import FR3, model_of

    return model_of(scene.robots[0]) if scene.robots else FR3


def _overlap(a: Box, b: Box, margin) -> bool:
    margins = margin if isinstance(margin, tuple) else (margin,) * 3
    return all(lo1 < hi2 + m and lo2 < hi1 + m
               for ((lo1, hi1), (lo2, hi2)), m in zip(zip(a, b), margins))


def supports(scene: SceneSpec, name: str, other: BodySpec) -> bool:
    """Does ``name`` stand on ``other`` -- on its top, or on the top of one of
    its parts (a Hanoi stand's base, below its pegs)?  The fingers stop above
    what an object stands on, however thin the object: no air is asked for
    below."""
    target = next(b for b in scene.bodies if b.name == name)
    (x0, x1), (y0, y1), (z0, _) = bounds(target)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    tops = [bounds(other)] + occluder_boxes(other.shape, other.size, other.pose, other.attrs)
    return any(abs(top - z0) < 0.003 and ox0 <= cx <= ox1 and oy0 <= cy <= oy1
               for (ox0, ox1), (oy0, oy1), (_, top) in tops)


def stacked_on(scene: SceneSpec, name: str) -> List[str]:
    """Everything resting on ``name``, directly or higher up the same pile, and
    any cup put down over it.

    A pile is taken apart from the top and a cup is lifted off, so these are in
    the way of grasping the object -- and hide it -- only until they have been
    moved; the checks here leave them out.
    """
    target = next(b for b in scene.bodies if b.name == name)
    (tx0, tx1), (ty0, ty1), (tz0, tz1) = bounds(target)
    out: List[str] = []
    for b in scene.bodies:
        (x0, x1), (y0, y1), (z0, z1) = bounds(b)
        if (b.attrs.get("container") == "inverted" and b.name != name
                and x0 < tx0 and tx1 < x1 and y0 < ty0 and ty1 < y1 and z0 <= tz0 and tz1 < z1):
            out.append(b.name)
        covered = b.attrs.get("covers")         # a lid: whatever is in its container
        if covered and b.name != name:
            (cx0, cx1), (cy0, cy1), (cz0, cz1) = bounds(next(c for c in scene.bodies
                                                         if c.name == covered))
            if cx0 < tx0 and tx1 < cx1 and cy0 < ty0 and ty1 < cy1 and cz0 <= tz0 and tz1 <= z0:
                out.append(b.name)
    frontier = [name]
    while frontier:
        current = frontier.pop()
        below = next(b for b in scene.bodies if b.name == current)
        (bx0, bx1), (by0, by1), (_, top) = bounds(below)
        for b in scene.bodies:
            if b.name in out or b.name == below.name:
                continue
            (x0, x1), (y0, y1), (z0, _) = bounds(b)
            cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
            if abs(z0 - top) < 0.003 and bx0 <= cx <= bx1 and by0 <= cy <= by1:
                out.append(b.name)
                frontier.append(b.name)
    return out


def grasp_conflicts(scene: SceneSpec, name: str) -> List[str]:
    """What a straight-down grasp of ``name``, where it starts, would run into.

    The grasp point is the scripted reference's (see ``grasp_pose``).  Empty
    means the way down is clear -- once whatever is piled on the object is
    gone (:func:`stacked_on`).
    """
    from ..policies.scripted import grasp_offset

    target = next(b for b in scene.bodies if b.name == name)
    x, y, z = target.pose.pos
    grasp = (x, y, z + grasp_offset(target))
    above = set(stacked_on(scene, name))
    out = []
    for other in scene.bodies:
        if other.name == name or not collides(other) or other.name in above:
            continue
        # a container is its walls and floor: reaching into one is fine
        boxes = occluder_boxes(other.shape, other.size, other.pose, other.attrs)
        margin = (MARGIN, MARGIN, 0.0) if supports(scene, name, other) else MARGIN
        opening = target.attrs.get("grasp_open")
        hit = next((part for part, corners, bottom in gripper_parts(
                        grasp, scene_model(scene), float(target.attrs.get("grasp_yaw", 0.0)),
                        None if opening is None else float(opening))
                    if any(_turned_overlap(corners, bottom, box, margin) for box in boxes)), None)
        if hit:
            out.append(f"{hit} would hit {other.name}")
    return out


def must_be_graspable(scene: SceneSpec) -> List[str]:
    """Every loose body, unless it says otherwise: ``graspable: False``, or
    ``graspable_later`` for what only a moved joint brings within reach (a
    drawer's contents) -- those the reference run on Isaac has to show."""
    return [b.name for b in scene.bodies
            if not b.fixed and b.attrs.get("graspable", True)
            and not b.attrs.get("graspable_later")]


def check_scene(scene: SceneSpec) -> List[str]:
    """One line per blocked grasp, in the form ``"cube0: hand would hit bin"``."""
    return [f"{name}: {why}" for name in must_be_graspable(scene)
            for why in grasp_conflicts(scene, name)]


def falls_through(scene: SceneSpec) -> List[str]:
    """One line per free body with nothing on it that collides: on Isaac it
    drops through the table as the episode starts (the mock has no gravity, so
    only the render shows it -- DS3's and DI3's lamps, 9/21).  Something only
    to look at is fixed, carried by a joint, or tied to a body that is."""
    out = []
    for b in scene.bodies:
        parts = b.attrs.get("parts") or ()
        nothing = b.attrs.get("collides") is False or (
            bool(parts) and all(len(p) > 5 and p[5] is False for p in parts))
        if nothing and not b.fixed and not b.attrs.get("carried_by") \
                and not b.attrs.get("attached_to"):
            out.append(f"{b.name} is free and nothing on it collides")
    return out


#: A body on a joint nothing drives -- a free or sprung slide -- rides at least
#: this far above the table.  Resting on it, a drawer a hand had pressed down
#: stayed pressed into it (the joint allows no give), and the friction held it
#: open against its spring on Isaac.  A position-driven joint pushes through.
JOINT_CLEARANCE = 0.002


def resting_on_table(scene: SceneSpec) -> List[str]:
    """One line per undriven joint whose body touches the table."""
    bodies = {b.name: b for b in scene.bodies}
    out = []
    for joint in scene.articulations:
        body = bodies.get(joint.parent)
        if joint.actuated or body is None:
            continue
        gap = bounds(body)[2][0] - scene.table_height
        if gap < JOINT_CLEARANCE - 1e-9:
            out.append(f"{body.name} on {joint.name} is {gap * 1000:.1f} mm above the table, "
                       f"needs {JOINT_CLEARANCE * 1000:.0f} mm")
    return out


def out_of_reach(scene: SceneSpec) -> List[str]:
    """One line per object a task needs picked up that no commanded arm can
    reach with a straight-down grasp (the robot model's measured reach) -- and
    per fixed open container one of whose drop points none can: DS2's basket
    stood 25 cm in front of the Franka's base, and the egg never got in."""
    from ..engines.robots import within_reach
    from ..policies.scripted import container_spots

    def reachable(point) -> bool:
        return any(within_reach(r, a, scene.table_height, point)
                   for r in scene.robots for a in r.arms)

    out = []
    for name in must_be_graspable(scene):
        body = next(b for b in scene.bodies if b.name == name)
        if not reachable(body.pose.pos):
            out.append(f"{name} at ({body.pose.pos[0]:.2f}, {body.pose.pos[1]:.2f}) is out of reach")
    for body in scene.bodies:
        if not body.fixed or body.attrs.get("container") != "open":
            continue
        top = body.pose.pos[2] + body.size[2]
        far = next(((x, y) for x, y in container_spots(body, margin=0.02)
                    if not reachable((x, y, top))), None)
        if far:
            out.append(f"{body.name}: a drop point at ({far[0]:.2f}, {far[1]:.2f}) is out of reach")
    return out


def arm_start_problems(scene: SceneSpec) -> List[str]:
    """One line per ``ArmStart`` the engines could not set up: a start pose
    out of that arm's reach, or a body said to be held that the closed hand
    at that pose would not have hold of (``base.in_grasp``, the rule both
    engines use), that is fixed, or that nothing can touch."""
    from ..engines.base import in_grasp
    from ..engines.robots import within_reach

    bodies = {b.name: b for b in scene.bodies}
    robots = {r.name: r for r in scene.robots}
    out = []
    for start in scene.arm_starts:
        robot_name, arm = start.arm.split("/")
        robot = robots.get(robot_name)
        if robot is None or arm not in robot.arms:
            out.append(f"{start.arm} starts somewhere, but no commanded arm has that name")
            continue
        if not within_reach(robot, arm, scene.table_height, start.pose.pos):
            x, y, z = start.pose.pos
            out.append(f"{start.arm} starts out of its reach, at ({x:.2f}, {y:.2f}, {z:.2f})")
        if start.holding is None:
            continue
        body = bodies.get(start.holding)
        if body is None:
            out.append(f"{start.arm} starts holding {start.holding}, which is not in the scene")
        elif body.fixed or not collides(body):
            out.append(f"{start.arm} starts holding {body.name}, which is fixed or touches nothing")
        elif not in_grasp(body, body.pose, start.pose.pos):
            out.append(f"{start.arm} starts holding {body.name}, but its hand is not on it")
    return out


#: how close a jointed body may come to a fixed one anywhere along its travel
JOINT_SWEEP_MARGIN = 0.0005


def joint_sweeps(scene: SceneSpec) -> List[str]:
    """One line per slide whose body, somewhere along its travel, runs into
    something fixed -- a button cap sitting on its housing cannot be pressed.
    Only a physics engine would notice; the mock has no collisions."""
    bodies = {b.name: b for b in scene.bodies}
    fixed = [b for b in scene.bodies if b.fixed and collides(b)]
    out = []
    for joint in scene.articulations:
        body = bodies.get(joint.parent)
        # a body that collides with nothing cannot stick (CL6's smoothie rises
        # out of the blender's base)
        if body is None or joint.kind != "slide" or not collides(body):
            continue
        lo, hi = joint.limits
        for k in range(9):
            value = lo + (hi - lo) * k / 8
            moved = tuple(p + a * value for p, a in zip(body.pose.pos, joint.axis))
            mine = occluder_boxes(body.shape, body.size, Pose(moved, body.pose.quat), body.attrs)
            # what it passes through does not stop it (CR5's seats, which wait
            # inside the brick they come up through)
            hit = next((o.name for o in fixed if not passes(body, o)
                        for box in occluder_boxes(o.shape, o.size, o.pose, o.attrs)
                        if any(_overlap(m, box, -JOINT_SWEEP_MARGIN) for m in mine)), None)
            if hit:
                out.append(f"{body.name} on {joint.name} runs into {hit} at {value * 1000:.1f} mm")
                break
    return out


def camera_touches(scene: SceneSpec, trajectory) -> List[str]:
    """Where a wrist camera's housing or bracket meets a body along an
    episode's world states -- the mock's, whose hand follows the commanded
    path through whatever is in the way.  On Isaac the housing is solid and
    the hand stops where it meets something: v16 moved the FR3's camera to the
    hand's robot side, and WK3's hand, turned as before over the first domino,
    came down with the camera on the pendulum's frame and let every domino go
    4.4 cm high.  Checked while a hand points down (``HandBox`` is written for
    that), one line per body, at the first time it is met.  A body a joint has
    turned is covered as it stands then (the toy box of v1.8's CR6, its lid
    open past upright: the bounds of the whole board reached 5 cm out under
    its lean, where the hand carrying the bear passes 2 cm clear of it)."""
    from ..core.geometry import rotate
    from ..engines.robots import model_of

    bodies = {b.name: b for b in scene.bodies if collides(b)}
    met = {}
    for world in trajectory:
        for robot in scene.robots:
            parts = [p for p in model_of(robot).closed_hand if p.name == "camera"]
            state = world.robots.get(robot.name)
            if not parts or state is None:
                continue
            for arm in state.arms.values():
                tcp = arm.eef_pose
                if rotate(tcp.quat, (0.0, 0.0, 1.0))[2] > -0.9:
                    continue
                x = rotate(tcp.quat, (1.0, 0.0, 0.0))
                yaw = math.atan2(x[1], x[0])
                gx, gy, gz = tcp.pos
                mine = []
                for part in parts:
                    (x0, x1), (y0, y1) = turned_footprint(part.x, part.y, yaw)
                    mine.append(((gx + x0, gx + x1), (gy + y0, gy + y1),
                                 (gz + part.bottom, gz + part.bottom + part.height)))
                for name, obj in world.objects.items():
                    body = bodies.get(name)
                    if body is None or name in met:
                        continue
                    theirs = occluder_boxes(body.shape, body.size, obj.pose, body.attrs,
                                            turned=True)
                    if any(_overlap(m, box, 0.0) for m in mine for box in theirs):
                        met[name] = world.sim_time
    return [f"the wrist camera meets {name} at {t:.1f} s"
            for name, t in sorted(met.items(), key=lambda kv: kv[1])]
