"""What the fixed camera sees of each object -- checked by geometry.

Two things a layout must not do, whatever the renderer:

* hide an object behind other bodies at the start -- or behind the arms, as
  they start: the model would be asked about something it cannot see (unless
  the task hides it on purpose, and says so with ``attrs["hidden_at_start"]``).
  v12 (9/24): the ARX X5s' head camera looks past their folded grippers, which
  hid WK4's near cutlery (the advisor: "the knife and fork are hard to see");
  this check counted bodies only;
* let the hand hide the object it is grasping: the fixed camera then shows
  nothing of the grasp.  A steep, near top-down view did exactly this -- over
  every task, 29% of an object stayed in sight under the hand, and some
  objects vanished completely.

Sight lines are tested against boxes (``engines/sight.py``): the bodies, and
the hand, its closed fingers and a column above it for the wrist and forearm.
"""

from __future__ import annotations

import math
from typing import List

from ..core.types import Vec3
from ..core.geometry import yaw_of
from ..engines.base import SceneSpec, collides, external_camera_pose
from ..engines.sight import Box, facing_samples, hidden_fraction, sight_boxes
from .clearance import bounds, must_be_graspable, scene_model, stacked_on, turned_footprint

#: at most this much of an object's camera-facing surface may be hidden...
MAX_HIDDEN_AT_START = 0.3
#: ...and at most this much while the hand is over it to grasp it.  A toy
#: narrower than the Piper's fingers are wide (3.4 cm) is more than half
#: covered by them from a camera off to its side, and a tall one's top by the
#: hand (IC1's strawberry and pear, 56-58%); the wrist camera shows those
#: grasps.  The steep v0 view this check was written against hid 71%.
MAX_HIDDEN_IN_GRASP = 0.6


def hand_boxes(grasp: Vec3, model=None, yaw: float = 0.0, grip_half: float = 0.0) -> List[Box]:
    """The hand holding something ``grip_half`` wide (half, along the fingers'
    travel) at ``grasp``, closed fingers and all, and a column above it for the
    wrist and forearm (the robot model's, the FR3's by default)."""
    from ..engines.robots import FR3

    model = model or FR3
    gx, gy, gz = grasp
    out: List[Box] = []
    top = gz
    for part in model.closed_hand:
        sides = ([(grip_half + part.y[0], grip_half + part.y[1]),
                  (-grip_half - part.y[1], -grip_half - part.y[0])]
                 if part.from_side else [part.y])
        for y_range in sides:
            (x0, x1), (y0, y1) = turned_footprint(part.x, y_range, yaw)
            out.append(((gx + x0, gx + x1), (gy + y0, gy + y1),
                        (gz + part.bottom, gz + part.bottom + part.height)))
        top = max(top, gz + part.bottom + part.height)
    half, height = model.wrist_column
    out.append(((gx - half, gx + half), (gy - half, gy + half), (top, top + height)))
    return out


def rest_hand_boxes(scene: SceneSpec) -> List[Box]:
    """Every arm's hand where it starts -- fingers open, with the column of its
    wrist behind them -- turned as the arm holds it there: straight down, or
    as the model rests (the ARX X5 folded, its tool pointing forward)."""
    import itertools

    from ..core.geometry import rotate
    from ..engines.robots import arm_start_pose, model_of

    out: List[Box] = []
    for robot in scene.robots:
        model = model_of(robot)
        half, height = model.wrist_column
        top = max(p.bottom + p.height for p in model.open_hand)
        parts = [(p.x, p.y, (p.bottom, p.bottom + p.height)) for p in model.open_hand]
        parts.append(((-half, half), (-half, half), (top, top + height)))
        for arm in robot.arms:
            tcp = arm_start_pose(scene, robot, arm)
            for x_range, y_range, (z0, z1) in parts:
                # a hand box is drawn around a straight-down grasp: its x and y
                # are the world's, its heights up the tool -- in the tool frame
                # (x, -y, -height) -- and turned from there as the hand starts
                corners = [tuple(c + d for c, d in zip(tcp.pos, rotate(tcp.quat, (x, -y, -z))))
                           for x, y, z in itertools.product(x_range, y_range, (z0, z1))]
                xs, ys, zs = zip(*corners)
                out.append(((min(xs), max(xs)), (min(ys), max(ys)), (min(zs), max(zs))))
    return out


def occluders(body) -> List[Box]:
    return sight_boxes(body.shape, body.size, body.pose, body.attrs)


def samples(body, eye: Vec3) -> List[Vec3]:
    return facing_samples(bounds(body), eye)


def _eye(scene: SceneSpec) -> Vec3:
    return external_camera_pose(scene).pos


def hidden_at_start(scene: SceneSpec, name: str) -> float:
    """Hidden by other bodies and by the arms where they start -- not counting
    what is piled on it, which covers its top only until the pile is taken
    apart, nor the hand a thing starts in (``hidden_in_grasp`` is about that)."""
    eye = _eye(scene)
    body = next(b for b in scene.bodies if b.name == name)
    above = set(stacked_on(scene, name))
    boxes = [box for o in scene.bodies if o.name != name and collides(o) and o.name not in above
             for box in occluders(o)]
    if not any(start.holding == name for start in scene.arm_starts):
        boxes += rest_hand_boxes(scene)
    return hidden_fraction(samples(body, eye), eye, boxes)


def hidden_in_grasp(scene: SceneSpec, name: str) -> float:
    from ..policies.scripted import grasp_offset

    eye = _eye(scene)
    body = next(b for b in scene.bodies if b.name == name)
    x, y, z = body.pose.pos
    yaw = float(body.attrs.get("grasp_yaw", 0.0))
    # the fingers close on the body's sides across the hand, whichever way the
    # body itself is turned
    across = yaw - yaw_of(body.pose.quat)
    boxes = hand_boxes((x, y, z + grasp_offset(body)), scene_model(scene), yaw,
                       grip_half=body.size[1] if abs(math.sin(across)) < 0.5 else body.size[0])
    return hidden_fraction(samples(body, eye), eye, boxes)


#: how far past its task's frame a thing may reach and still be in the picture
#: (the fit keeps every frame point 2% of the image inside the edge)
FRAME_SLACK = 0.01


def outside_frame(scene: SceneSpec) -> List[str]:
    """One line per body, painted mark or slide's travel that reaches past the
    task's own camera frame (``SceneSpec.camera_frame``) -- the fixed camera
    is fitted to that frame, not to what is on the table."""
    from dataclasses import replace

    from ..core.types import Pose

    if not scene.camera_frame:
        return []
    xs = [p[0] for p in scene.camera_frame]
    ys = [p[1] for p in scene.camera_frame]
    zs = [p[2] for p in scene.camera_frame]
    lo = (min(xs) - FRAME_SLACK, min(ys) - FRAME_SLACK)
    hi = (max(xs) + FRAME_SLACK, max(ys) + FRAME_SLACK, max(zs) + FRAME_SLACK)

    def past(box) -> bool:
        (x0, x1), (y0, y1), (_, z1) = box
        return x0 < lo[0] or y0 < lo[1] or x1 > hi[0] or y1 > hi[1] or z1 > hi[2]

    out = []
    by_name = {b.name: b for b in scene.bodies}
    for b in scene.bodies:
        if past(bounds(b)):
            out.append(f"{b.name} reaches past the camera frame")
    for m in scene.markers:
        x, y, z = m.pose.pos
        hy = m.size[0] if m.shape == "circle" else m.size[1]
        if past(((x - m.size[0], x + m.size[0]), (y - hy, y + hy), (z, z))):
            out.append(f"mark {m.name} lies past the camera frame")
    for j in scene.articulations:
        body = by_name.get(j.parent)
        if body is None or j.kind != "slide" or body.attrs.get("flies_off"):
            continue
        for v in j.limits:
            moved = tuple(p + a * v for p, a in zip(body.pose.pos, j.axis))
            if past(bounds(replace(body, pose=Pose(moved, body.pose.quat)))):
                out.append(f"{body.name} slides past the camera frame")
                break
    return out


def off_the_picture(scene: SceneSpec) -> List[str]:
    """One line per thing the station's head camera does not have in shot.

    The single-arm camera is fitted to its task's frame, so it always covers
    it; the head camera is fixed, and a layout wider than its cone simply
    falls off the side of every observation -- which is what all four two-arm
    tasks did until 9/21.  What is checked is the things themselves, not the
    task's frame box: the box over-claims near the lens, where its top corners
    are metres outside a cone nothing that tall stands in."""
    from dataclasses import replace

    from ..core.types import Pose
    from ..engines.base import head_camera_room

    room = head_camera_room(scene)
    if room is None:
        return []

    def unseen(box) -> bool:
        (x0, x1), (y0, y1), (z0, z1) = box
        return any(room((x, y, z))[0] > room((x, y, z))[1]
                   for x in (x0, x1) for y in (y0, y1) for z in (z0, z1))

    out = []
    by_name = {b.name: b for b in scene.bodies}
    for b in scene.bodies:
        if unseen(bounds(b)):
            out.append(f"{b.name} is outside the head camera's picture")
    for m in scene.markers:
        x, y, z = m.pose.pos
        hy = m.size[0] if m.shape == "circle" else m.size[1]
        if unseen(((x - m.size[0], x + m.size[0]), (y - hy, y + hy), (z, z))):
            out.append(f"mark {m.name} is outside the head camera's picture")
    for j in scene.articulations:
        body = by_name.get(j.parent)
        if body is None or j.kind != "slide" or body.attrs.get("flies_off"):
            continue
        for v in j.limits:
            moved = tuple(p + a * v for p, a in zip(body.pose.pos, j.axis))
            if unseen(bounds(replace(body, pose=Pose(moved, body.pose.quat)))):
                out.append(f"{body.name} slides outside the head camera's picture")
                break
    return out


def check_scene(scene: SceneSpec) -> List[str]:
    """One line per object the fixed camera cannot see well enough."""
    out = off_the_picture(scene) + outside_frame(scene)
    for name in must_be_graspable(scene):
        body = next(b for b in scene.bodies if b.name == name)
        if not collides(body):
            continue
        start = hidden_at_start(scene, name)
        if start > MAX_HIDDEN_AT_START and not body.attrs.get("hidden_at_start"):
            out.append(f"{name}: {start:.0%} hidden by other bodies or the arms at the start")
        grasp = hidden_in_grasp(scene, name)
        if grasp > MAX_HIDDEN_IN_GRASP:
            out.append(f"{name}: {grasp:.0%} hidden under the hand while grasped")
    return out
