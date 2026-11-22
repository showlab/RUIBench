"""Toys the v1 tasks share: wooden play fruit, a hungry monkey.

Built from primitives (``engines/parts.py``): each is one rigid body with a
few solid parts and some painted-on detail.  Every toy's origin is the middle
of its solid parts' bounds, so ``BodySpec.size`` is honest about where it
rests.  Sizes are chosen for the Piper's hand: nothing is wider than 5.4 cm
where the fingers close on it (they open 7 cm).
"""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Tuple

from ...core.geometry import quat_mul
from ...core.types import Pose
from ...engines import parts as P
from ...engines.base import ArticulationSpec, BodySpec
from ._common import TABLE

# -- shared helpers -------------------------------------------------------------


def centred(parts) -> list:
    """The parts moved so their solid bounds are centred on the origin -- the
    rule every toy here keeps, so ``BodySpec.size`` says where it rests."""
    from ...engines.parts import part_corners

    lo, hi = [math.inf] * 3, [-math.inf] * 3
    for part in parts:
        if part[5]:
            for p in part_corners(part):
                lo = [min(a, b) for a, b in zip(lo, p)]
                hi = [max(a, b) for a, b in zip(hi, p)]
    return P.offset(parts, tuple(-(a + b) / 2 for a, b in zip(lo, hi)))


# -- a mat to lay things out on -----------------------------------------------
#: a flat mat, 4 mm thick.  Not a tray: a rim low enough for fingers to pass
#: over is no rim at all, and one high enough to hold round fruit is in the way
MAT_HALF_Z = 0.002


def mat(name: str, xy, half_xy: Tuple[float, float], color="wood") -> BodySpec:
    return BodySpec(name, "box", (half_xy[0], half_xy[1], MAT_HALF_Z),
                    Pose((xy[0], xy[1], TABLE + MAT_HALF_Z)), category="mat", color=color,
                    fixed=True, attrs={"footprint": "rect", "graspable": False})


# -- a toy train -------------------------------------------------------------------
#: A wooden push-along train, painted.  The wagon: an open box, long along the
#: track (y), 9 x 10 cm inside -- room for a 7.5 cm plush bear lying across it
#: (v2) and 2.3 cm either way along the track; it rides its slide with its
#: floor 1.2 cm above the table, wheels on the track
WAGON_HALF, WAGON_WALL = (0.05, 0.055, 0.02), 0.005
WAGON_LIFT = 0.012


def wagon(name: str, xy, joint: str, color: str = "paint:red") -> BodySpec:
    hx, hy, hz = WAGON_HALF
    w = WAGON_WALL
    red, black = color, "paint:black"
    parts = [P.box((0, 0, -hz + w / 2), (hx, hy, w / 2), red),
             P.box((hx - w / 2, 0, 0), (w / 2, hy, hz), red),
             P.box((-hx + w / 2, 0, 0), (w / 2, hy, hz), red),
             P.box((0, hy - w / 2, 0), (hx - w, w / 2, hz), red),
             P.box((0, -hy + w / 2, 0), (hx - w, w / 2, hz), red)]
    parts += [P.cylinder((sx * (hx - 0.009), sy * (hy - 0.02), -hz - 0.002), 0.011, 0.004, black,
                         P.turn_y(math.pi / 2), collides=False)
              for sx in (-1, 1) for sy in (-1, 1)]
    parts.append(P.box((0, 0, -hz + w + 0.0004), (hx - w - 0.002, hy - w - 0.002, 0.0004),
                       "wood:ash", collides=False))
    return BodySpec(name, "compound", WAGON_HALF,
                    Pose((xy[0], xy[1], TABLE + WAGON_LIFT + WAGON_HALF[2])), category="wagon",
                    color=color.split(":")[-1], mass=0.3,
                    attrs={"parts": tuple(parts), "container": "open", "wall": w,
                           "footprint": "rect", "carried_by": joint, "driven": True,
                           "graspable": False})


def locomotive(name: str, wagon_body: BodySpec, ahead_by: float = 0.115) -> BodySpec:
    """The engine in front of the wagon (+y), painted on: it rides with the
    wagon and collides with nothing."""
    green, black, brass = "paint:green", "paint:black", "metal:brass"
    along_y = P.turn_x(math.pi / 2)
    parts = [P.box((0, 0, -0.012), (0.04, 0.055, 0.006), black, collides=False),
             P.cylinder((0, 0.014, 0.013), 0.025, 0.037, green, along_y, collides=False),
             P.box((0, -0.032, 0.022), (0.035, 0.02, 0.03), green, collides=False),
             P.box((0, -0.032, 0.056), (0.039, 0.024, 0.004), black, collides=False),
             P.cylinder((0, 0.032, 0.048), 0.008, 0.013, black, collides=False),
             P.sphere((0, 0.052, 0.013), 0.009, brass, collides=False)]
    parts += [P.cylinder((sx * 0.036, sy, -0.018), 0.013 if sy < 0 else 0.01, 0.004, "paint:red",
                         P.turn_y(math.pi / 2), collides=False)
              for sx in (-1, 1) for sy in (-0.033, 0.0, 0.033)]
    x, y, z = wagon_body.pose.pos
    return BodySpec(name, "compound", P.half_extents(parts, solid_only=False),
                    Pose((x, y + ahead_by, z - WAGON_HALF[2] + 0.024)), category="locomotive",
                    color="green", fixed=False,
                    attrs={"parts": tuple(parts), "collides": False, "graspable": False,
                           "attached_to": wagon_body.name})


def track(name: str, xy, half_length: float) -> BodySpec:
    """A wooden track along y, lying on the table: a board with two grooves
    for the wheels."""
    groove = (0.28, 0.2, 0.14)
    parts = [P.box((0, 0, 0.003), (0.05, half_length, 0.003), "wood:ash", collides=False)]
    parts += [P.box((sx * (WAGON_HALF[0] - 0.009), 0, 0.0062), (0.003, half_length, 0.0003),
                    groove, collides=False) for sx in (-1, 1)]
    return BodySpec(name, "compound", P.half_extents(parts, solid_only=False),
                    Pose((xy[0], xy[1], TABLE)), category="track", color="wood", fixed=True,
                    attrs={"parts": tuple(parts), "collides": False, "graspable": False})


# -- a little wheeled robot with a tray ----------------------------------------------
#: navy plastic, a white tray 10 x 13 cm on its back: room for a 7 cm apple
#: and 2.6 cm either way along its drive (v2)
BOT_HALF = (0.05, 0.065, 0.035)
BOT_LIFT = 0.002


def little_robot(name: str, xy, joint: str) -> BodySpec:
    """A small wheeled robot carrying a tray on its back (its top face)."""
    hx, hy, hz = BOT_HALF
    body, dark, tray_c = "plastic:navy", "plastic:black", "plastic:white"
    parts = [P.box((0, 0, -0.008), (hx - 0.01, hy - 0.015, hz - 0.008), body),
             P.box((0, 0, hz - 0.003), (hx, hy, 0.003), tray_c)]
    parts += [P.box((sx * (hx - 0.002), 0, hz + 0.003), (0.002, hy, 0.003), tray_c)
              for sx in (-1, 1)]
    parts += [P.box((0, sy * (hy - 0.002), hz + 0.003), (hx - 0.004, 0.002, 0.003), tray_c)
              for sy in (-1, 1)]
    parts += [P.cylinder((0, sy * (hy - 0.012), -hz + 0.017), 0.017, 0.004, dark,
                         P.turn_x(math.pi / 2), collides=False) for sy in (-1, 1)]
    parts.append(P.box((hx - 0.009, 0, 0.004), (0.001, 0.022, 0.009), "sky", collides=False))
    parts += [P.sphere((hx - 0.008, s * 0.01, 0.006), 0.0038, "white", collides=False)
              for s in (-1, 1)]
    parts = centred(parts)
    size = P.half_extents(parts)
    return BodySpec(name, "compound", size,
                    Pose((xy[0], xy[1], TABLE + BOT_LIFT + size[2])), category="robot",
                    color="blue", mass=0.4,
                    attrs={"parts": tuple(parts), "footprint": "rect", "carried_by": joint,
                           "driven": True, "graspable": False})


#: the basket version (DI3's customer): 11 x 15 cm inside, room for two
#: things side by side (an apple and a lemon, 7 + 5.4 cm), walls 3.5 cm high
SHOP_BOT_HALF = (0.06, 0.08, 0.0225)
BASKET_WALL, BASKET_H = 0.004, 0.035


def basket_robot(name: str, xy, joint: str) -> BodySpec:
    """A little wheeled robot with a basket on its back, its face (a screen,
    two eyes) toward the robot arm."""
    hx, hy, hz = SHOP_BOT_HALF
    body, dark, weave = "plastic:navy", "plastic:black", "wood:ash"
    w, bh = BASKET_WALL, BASKET_H / 2
    top = hz                                           # the chassis's top, the basket's floor
    parts = [P.box((0, 0, 0), (hx - 0.006, hy - 0.012, hz), body),
             P.box((0, 0, top + w / 2), (hx, hy, w / 2), weave)]
    parts += [P.box((sx * (hx - w / 2), 0, top + bh), (w / 2, hy, bh), weave) for sx in (-1, 1)]
    parts += [P.box((0, sy * (hy - w / 2), top + bh), (hx - w, w / 2, bh), weave) for sy in (-1, 1)]
    parts += [P.cylinder((0, sy * (hy - 0.016), -hz + 0.004), 0.016, 0.004, dark,
                         P.turn_x(math.pi / 2), collides=False) for sy in (-1, 1)]
    parts.append(P.box((-hx + 0.0055, 0, 0.002), (0.001, 0.026, 0.011), "sky", collides=False))
    parts += [P.sphere((-hx + 0.005, s * 0.011, 0.004), 0.004, "white", collides=False)
              for s in (-1, 1)]
    parts = centred(parts)
    size = P.half_extents(parts)
    return BodySpec(name, "compound", size,
                    Pose((xy[0], xy[1], TABLE + BOT_LIFT + size[2])), category="robot",
                    color="blue", mass=0.5,
                    attrs={"parts": tuple(parts), "footprint": "rect", "carried_by": joint,
                           "driven": True, "graspable": False, "container": "open",
                           "wall": BASKET_WALL})


def toy_house(name: str, xy, roof: str, scale: float = 1.0) -> BodySpec:
    """A toy house: a painted box, a roof block, a door -- 8.4 cm across and
    8.6 cm tall, times ``scale``."""
    k = scale
    parts = [P.box((0, 0, -0.012 * k), (0.035 * k, 0.035 * k, 0.028 * k), "paint:cream"),
             P.prism((0, 0, 0.031 * k), (0.042 * k, 0.042 * k, 0.015 * k), roof, P.turn_z(math.pi / 2)),
             P.box((-0.0355 * k, 0, -0.024 * k), (0.0005, 0.010 * k, 0.016 * k), "paint:brown",
                   collides=False)]
    parts = centred(parts)
    size = P.half_extents(parts)
    return BodySpec(name, "compound", size, Pose((xy[0], xy[1], TABLE + size[2])),
                    category="house", color="cream", fixed=True,
                    attrs={"parts": tuple(parts), "footprint": "rect", "graspable": False})


# -- a river and its passengers -------------------------------------------------------
# -- a cabinet whose drawer slides out sideways ---------------------------------------
#: The drawer is long along y and slides along it, out of the cabinet's -y
#: side, so the fixed camera (+x) sees into it once it is out.  Sized for five
#: real small toys (v2): 8.9 cm deep inside, under the cabinet's top an 8 cm
#: figure stands clear of it; two rows of three 8.4 x 9 cm places.  It comes
#: all the way out, so its far places are clear of the cabinet's top.
SIDE_DRAWER_HALF = (0.09, 0.14, 0.047)
SIDE_DRAWER_WALL = 0.005
#: it rides its slide this far above the table: a free slide resting on the
#: table was held by friction (BC2, v0)
SIDE_DRAWER_LIFT = 0.004
SIDE_DRAWER_TRAVEL = 0.28
#: The handle: a post against the drawer's -y face, in the drawer's frame,
#: standing 2.7 cm above the cabinet so the fingers close on it clear of the
#: cabinet's top.  The cabinet stops 3 mm short of the drawer's face, so the
#: post passes it.
_POST_TOP = SIDE_DRAWER_HALF[2] + 0.014 + 0.027
_POST_BOTTOM = -SIDE_DRAWER_HALF[2] + 0.010
SIDE_DRAWER_POST = ((0.0, -SIDE_DRAWER_HALF[1] - 0.011, (_POST_TOP + _POST_BOTTOM) / 2),
                    (0.011, 0.011, (_POST_TOP - _POST_BOTTOM) / 2))
#: where the tool centre point goes to hold the post, in the drawer's frame:
#: 14 mm below its top, on its middle (the Franka's fingers are 2 cm wide)
SIDE_DRAWER_GRIP = (0.0, SIDE_DRAWER_POST[0][1], _POST_TOP - 0.014)


def side_drawer(prefix: str, xy_shut, joint: str, color="wood:ash",
                cabinet_color="paint:cream"):
    """The drawer (a carried body), its slide, and the cabinet's plates."""
    hx, hy, hz = SIDE_DRAWER_HALF
    cx, cy = xy_shut
    bottom = TABLE + SIDE_DRAWER_LIFT
    drawer = BodySpec(f"{prefix}drawer", "bin", SIDE_DRAWER_HALF,
                      Pose((cx, cy, bottom + hz)), category="drawer", color=color, mass=0.3,
                      attrs={"container": "open", "wall": SIDE_DRAWER_WALL, "footprint": "rect",
                             "handle": SIDE_DRAWER_POST, "carried_by": joint,
                             "graspable": False})
    slide = ArticulationSpec(joint, "slide", axis=(0.0, 1.0, 0.0),
                             limits=(-SIDE_DRAWER_TRAVEL - 0.005, 0.0), initial=0.0,
                             actuated=False, stiffness=0.0, damping=3.0, parent=drawer.name)
    top_z = bottom + 2 * hz + 0.004              # the cabinet top's underside
    side_h = (top_z - TABLE) / 2
    y0, y1 = cy - hy + 0.003, cy + hy + 0.016    # the cabinet's extent along the slide
    ym, yh = (y0 + y1) / 2, (y1 - y0) / 2
    plates = {
        "cabinet_top": ((cx, ym, top_z + 0.005), (hx + 0.012, yh, 0.005)),
        "cabinet_front": ((cx + hx + 0.008, ym, TABLE + side_h), (0.004, yh, side_h)),
        "cabinet_back": ((cx - hx - 0.008, ym, TABLE + side_h), (0.004, yh, side_h)),
        "cabinet_end": ((cx, y1 - 0.004, TABLE + side_h), (hx + 0.004, 0.004, side_h)),
    }
    cabinet = [BodySpec(f"{prefix}{name}", "box", half, Pose(centre), category="cabinet",
                        color=cabinet_color, fixed=True, attrs={"graspable": False})
               for name, (centre, half) in plates.items()]
    return drawer, slide, cabinet


# -- a cross-shaped block (tic-tac-toe's X) ---------------------------------------------
#: the block: a plus sign 4.4 cm across with arms 1.6 cm wide, 2.4 cm thick
CROSS_ARM = (0.022, 0.008)
CROSS_HALF_HEIGHT = 0.012


def cross_block(name: str, xy, yaw: float, color="paint:green",
                half_height: float = CROSS_HALF_HEIGHT, grasp_height: float = 0.008) -> BodySpec:
    a, b = CROSS_ARM
    h = half_height
    parts = (P.box((0, 0, 0), (a, b, h), color), P.box((0, 0, 0), (b, a, h), color))
    return BodySpec(name, "compound", (a, a, h),
                    Pose((xy[0], xy[1], TABLE + h), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
                    category="cross block", color=color, mass=0.03,
                    # held by the upper half of its arms' ends: the hand stays
                    # higher, and the camera sees more of the block under it
                    attrs={"parts": parts, "kind": "cross", "footprint": "rect",
                           "grasp_height": grasp_height})


# -- dominoes -------------------------------------------------------------------------
#: A domino stands 6 cm tall, 3 cm wide (x) and 1 cm thick (y): a chain runs
#: along y.  The hand takes it by its narrow sides, turned a quarter.
DOMINO_HALF = (0.015, 0.005, 0.03)
DOMINO_COLORS = ("paint:red", "paint:orange", "paint:yellow", "paint:green", "paint:blue")
RACK_FLOOR = 0.004


def domino(name: str, xy, color, on: float = TABLE, yaw: float = 0.0,
           grasp_yaw: float = -math.pi / 2) -> BodySpec:
    """A domino turned ``yaw``; ``grasp_yaw`` is the hand's turn the scene is
    checked for (the hand takes it by its narrow sides)."""
    hx, hy, hz = DOMINO_HALF
    return BodySpec(name, "box", DOMINO_HALF,
                    Pose((xy[0], xy[1], on + hz), (math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2))),
                    category="domino", color=color if isinstance(color, str) else tuple(color),
                    mass=0.015,
                    attrs={"grasp_height": 0.02, "grasp_yaw": grasp_yaw, "footprint": "rect"})


def domino_rack(name: str, xy, slots: int, pitch: float, color="wood:ash") -> BodySpec:
    """A low wooden rack along y: a groove between two rails that holds the
    dominoes upright, faces to the front."""
    half_len = slots * pitch / 2
    groove = DOMINO_HALF[1] + 0.001
    rail = 0.012
    parts = (P.box((0, 0, -0.004), (groove + rail, half_len, RACK_FLOOR / 2), color),
             P.box((groove + rail / 2, 0, 0.0), (rail / 2, half_len, 0.004), color),
             P.box((-groove - rail / 2, 0, 0.0), (rail / 2, half_len, 0.004), color))
    return BodySpec(name, "compound", (groove + rail, half_len, 0.006),
                    Pose((xy[0], xy[1], TABLE + 0.006)),
                    category="domino rack", color=color, fixed=True,
                    attrs={"parts": parts, "graspable": False, "footprint": "rect"})


def rack_slot(rack: BodySpec, k: int, slots: int, pitch: float):
    """Where the k-th domino stands in a rack: (x, y) and the floor's height."""
    x, y, _ = rack.pose.pos
    return (x, y - slots * pitch / 2 + (k + 0.5) * pitch), TABLE + RACK_FLOOR


#: the pendulum: a 10 cm arm with a ball, on a pivot 13.5 cm up, turning about x
PENDULUM_ARM = 0.10
PENDULUM_BALL = 0.012


def pendulum(prefix: str, pivot_xy, pivot_z: float, joint: str, up: float, down: float,
             color="paint:red", frame_color="wood:ash"):
    """The hammer (a carried body, hanging straight down at joint 0), its hinge,
    and the frame it hangs from: a post beside the chain and a bar over it.
    The hinge is a soft spring with a lot of damping: let go, the ball comes
    down in about a second and leans on what it meets."""
    from ...engines.base import ArticulationSpec

    px, py = pivot_xy
    length, r = PENDULUM_ARM, PENDULUM_BALL
    parts = (P.box((0, 0, 0.0), (0.003, 0.003, (length - r) / 2), frame_color),
             P.sphere((0, 0, -(length - r) / 2 - r), r, color))
    centre_z = pivot_z - (length - r) / 2
    hammer = BodySpec(f"{prefix}hammer", "compound", P.half_extents(parts),
                      Pose((px, py, centre_z)), category="pendulum", color=color, mass=0.03,
                      attrs={"parts": parts, "graspable": False, "carried_by": joint})
    hinge = ArticulationSpec(joint, "hinge", axis=(1.0, 0.0, 0.0), limits=(down, up + 0.05),
                             initial=up, actuated=False, stiffness=0.2, damping=0.1,
                             parent=hammer.name, pivot=(0.0, 0.0, pivot_z - centre_z))
    post_x = px + 0.05
    frame = [
        BodySpec(f"{prefix}post", "box", (0.006, 0.006, (pivot_z + 0.012 - TABLE) / 2),
                 Pose((post_x, py, (pivot_z + 0.012 + TABLE) / 2)), category="frame",
                 color=frame_color, fixed=True, attrs={"graspable": False}),
        BodySpec(f"{prefix}bar", "box", (0.03, 0.005, 0.005), Pose((px + 0.024, py, pivot_z + 0.012)),
                 category="frame", color=frame_color, fixed=True, attrs={"graspable": False}),
    ]
    return hammer, hinge, frame


BELL_QUIET, BELL_RINGING = (0.35, 0.3, 0.2), (1.0, 0.95, 0.45)


def bell(name: str, xy, color="metal:brass"):
    """A little bell on a stand, 3.5 cm tall, and the light on top of it that
    comes on when it rings (``<name>_light``)."""
    parts = [P.cylinder((0, 0, -0.0155), 0.016, 0.002, "wood:oak"),
             P.cylinder((0, 0, -0.004), 0.013, 0.0095, color),
             P.cylinder((0, 0, 0.0085), 0.009, 0.003, color)]
    parts = centred(parts)
    size = P.half_extents(parts)
    body = BodySpec(name, "compound", size, Pose((xy[0], xy[1], TABLE + size[2])),
                    category="bell", color=color, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "round"})
    light = BodySpec(f"{name}_light", "sphere", (0.005, 0.005, 0.005),
                     Pose((xy[0], xy[1], TABLE + 2 * size[2] + 0.004)),
                     category="lamp", color=BELL_QUIET, fixed=True,
                     attrs={"collides": False, "graspable": False})
    return body, light


# -- a two-button launcher ------------------------------------------------------------
#: A button's cap rides a sprung slide 1 cm above its housing; a closed hand
#: pushes it down (``pressable``).
BUTTON_RADIUS, BUTTON_HALF, BUTTON_TRAVEL = 0.022, 0.008, 0.010
CONSOLE_TOP = TABLE + 0.03
BUTTON_RING = 0.004                                        # the housing's height
#: the cap's middle, up: 1.2 cm of air over the housing for 1 cm of travel
BUTTON_REST_Z = CONSOLE_TOP + BUTTON_RING + 0.012 + BUTTON_HALF
#: how high the cap's dome rises over its flat top
BUTTON_DOME = 0.004
#: v14, an emergency stop's look: the red mushroom head on a dark collar (the
#: button's stem) that sinks with it into a yellow base; radius, height --
#: down to 2 mm inside the base while the button is up
BUTTON_COLLAR = (0.011, 0.014)
BUTTON_BASE_R = 0.032


def push_button(name: str, xy, joint: str, color="plastic:red", radius: float = BUTTON_RADIUS):
    """The cap (a carried body) and its spring.  An emergency stop's red
    mushroom head on its dark collar, which sinks with it into the yellow
    base (v14; a flat cap hung over a ring until then).  The head is only
    drawn: the fingers push on the flat top of the hidden disc under its
    dome.  (A flat top mirrored the lab's lights, and DI1's right button
    rendered pink beside the red left one, Isaac, 9/23.)"""
    from ...engines.base import ArticulationSpec

    collar_r, collar_h = BUTTON_COLLAR
    parts = (P.cylinder((0, 0, 0), radius, BUTTON_HALF, color, collides="hidden"),
             P.lathe((0, 0, BUTTON_DOME / 2), "mushroom", radius, BUTTON_HALF + BUTTON_DOME / 2, color),
             P.lathe((0, 0, -BUTTON_HALF - collar_h / 2), "cylinder", collar_r, collar_h / 2,
                     "plastic:dark"))
    cap = BodySpec(name, "compound", (radius, radius, BUTTON_HALF),
                   Pose((xy[0], xy[1], BUTTON_REST_Z)), category="button", color=color, mass=0.02,
                   attrs={"parts": parts, "pressable": True, "graspable": False, "carried_by": joint,
                          "footprint": "round"})
    spring = ArticulationSpec(joint, "slide", axis=(0.0, 0.0, 1.0),
                              limits=(-BUTTON_TRAVEL, 0.0), initial=0.0, actuated=False,
                              stiffness=300.0, damping=5.0, parent=name, carries=False)
    return cap, spring


def launch_console(name: str, centre_xy, half_length: float, buttons_xy,
                   color="metal:black", ring="plastic:yellow") -> BodySpec:
    """A long low control box -- dark powder-coated steel, a brushed
    aluminium plate screwed down on top -- with a yellow base under each
    button (v14: a grey plastic bar with yellow rings until then)."""
    cx, cy = centre_xy
    hz = (CONSOLE_TOP - TABLE) / 2
    hx, plate = 0.035, 0.0004
    parts = [P.box((0, 0, 0), (hx, half_length, hz), color, collides="hidden"),
             P.slab((0, 0, 0), (hx, half_length, hz), color),
             P.slab((0, 0, hz), (hx - 0.004, half_length - 0.004, plate), "metal:aluminium")]
    for sx in (-1, 1):
        for sy in (-1, 1):
            parts.append(P.ellipsoid((sx * (hx - 0.009), sy * (half_length - 0.009), hz + plate),
                                     (0.002, 0.002, 0.0009), "metal:chrome"))
    for bx, by in buttons_xy:
        parts.append(P.lathe((bx - cx, by - cy, hz + plate + BUTTON_RING / 2), "plinth", BUTTON_BASE_R,
                             BUTTON_RING / 2, ring))
    return BodySpec(name, "compound", (hx, half_length, hz + BUTTON_RING),
                    Pose((cx, cy, TABLE + hz)), category="console", color=color, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


#: an indicator lamp: a glass dome on a black post, in a chrome bezel (v14)
LAMP_RADIUS, LAMP_POST_R, LAMP_POST_H = 0.015, 0.011, 0.025
LAMP_DARK = (0.06, 0.07, 0.06)


def lamp_posts(name: str, spots) -> BodySpec:
    """The posts the lamps stand on (``indicator_lamp``), one body: black
    stands on round feet, a chrome bezel round the lamp at the top of each."""
    cx, cy = (sum(x for x, _ in spots) / len(spots), sum(y for _, y in spots) / len(spots))
    hz = LAMP_POST_H / 2
    parts = []
    for x, y in spots:
        dx, dy = x - cx, y - cy
        parts += [P.cylinder((dx, dy, 0), LAMP_POST_R, hz, "plastic:black", collides="hidden"),
                  P.lathe((dx, dy, -hz + 0.002), "plinth", LAMP_POST_R + 0.005, 0.002, "plastic:black"),
                  P.lathe((dx, dy, 0.001), "cylinder", LAMP_POST_R, hz - 0.001, "plastic:black"),
                  P.lathe((dx, dy, hz), "washer", LAMP_RADIUS + 0.003, 0.002, "metal:chrome",
                          inner=LAMP_RADIUS - 0.003)]
    return BodySpec(name, "compound", P.half_extents(parts), Pose((cx, cy, TABLE + hz)),
                    category="lamp post", color="plastic:black", fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def indicator_lamp(name: str, xy) -> BodySpec:
    """A dome on top of its post that glows when the reactor lights it
    (``SetColor``)."""
    half = (LAMP_RADIUS, LAMP_RADIUS, LAMP_RADIUS * 0.8)
    return BodySpec(name, "compound", half, Pose((xy[0], xy[1], TABLE + LAMP_POST_H)),
                    category="lamp", color="flame:dark", fixed=True,
                    attrs={"parts": (P.ellipsoid((0, 0, 0), half, "flame:dark"),), "collides": False,
                           "graspable": False, "recolorable": True})


#: how far the rocket has to leap off its pad to have lifted off
ROCKET_LIFT = 0.16
#: and how far it flies: on up out of the picture (v8, 9/24 -- the advisor:
#: "the rocket should keep flying"; until then it stopped 16 cm up and hung
#: there), no higher than the room's ceiling
ROCKET_FLIGHT = 0.8
#: the rocket, as turned (v14): its body tube's radius; from its engine bell's
#: mouth up, where the tube starts and ends and the nose's tip; its launch lug
#: rides the pad's rod, this far out between two fins
ROCKET_R = 0.016
ROCKET_TUBE, ROCKET_TIP = (0.008, 0.108), 0.150
ROCKET_LUG = ROCKET_R + 0.0022


def rocket(name: str, xy, joint: str, pad_top: float):
    """A toy rocket standing on its pad (carried by a driven slide that lifts
    it off), its flame (dark until it burns), and the pad.

    v14 (cylinders of plastic until then, a blue ring round the body):
    turned and enamelled -- a white body tube, a red ogive nose and a red
    band, four red fins, a porthole, a dark engine bell; on a steel launch
    stand with an aluminium blast plate and the launch rod its lug rides up.
    Only drawn, but for a hidden cylinder as tall as it."""
    from ...engines.base import ArticulationSpec

    white, red = "paint:white", "paint:red"
    r = ROCKET_R
    lo, hi = ROCKET_TUBE
    lug = (ROCKET_LUG * math.cos(math.pi / 4), ROCKET_LUG * math.sin(math.pi / 4))
    parts = [P.cylinder((0, 0, ROCKET_TIP / 2), r, ROCKET_TIP / 2, white, collides="hidden"),
             P.lathe((0, 0, lo / 2 + 0.0005), "nozzle", 0.011, lo / 2 + 0.0005, "metal:black"),
             P.lathe((0, 0, (lo + hi) / 2), "cylinder", r, (hi - lo) / 2, white),
             P.lathe((0, 0, hi - 0.018), "cylinder", r + 0.0003, 0.004, red),
             P.lathe((0, 0, (hi + ROCKET_TIP) / 2), "nose", r, (ROCKET_TIP - hi) / 2, red),
             # a porthole toward the arms, in a chrome rim
             P.lathe((-r - 0.0003, 0, 0.068), "cylinder", 0.0045, 0.0005, "plastic:navy",
                     rot=P.turn_y(math.pi / 2)),
             P.lathe((-r - 0.0002, 0, 0.068), "washer", 0.006, 0.0006, "metal:chrome",
                     rot=P.turn_y(math.pi / 2), inner=0.0044),
             P.lathe((lug[0], lug[1], 0.052), "cylinder", 0.0022, 0.009, white)]
    # four swept fins: triangles standing out from the tube's foot
    for k in range(4):
        a = k * math.pi / 2
        out = r + 0.0075 - 0.0005
        parts.append(P.prism((out * math.cos(a), out * math.sin(a), lo + 0.021), (0.021, 0.0012, 0.0075),
                             red, quat_mul(P.turn_z(a), P.turn_y(math.pi / 2)), collides=False))
    parts = centred(parts)
    size = P.half_extents(parts)
    z = pad_top + size[2] + 0.001
    body = BodySpec(name, "compound", size, Pose((xy[0], xy[1], z)), category="rocket",
                    color=white, mass=0.05,
                    attrs={"parts": tuple(parts), "graspable": False, "carried_by": joint,
                           # its flight leaves the picture on purpose
                           "flies_off": True})
    lift = ArticulationSpec(joint, "slide", axis=(0.0, 0.0, 1.0), limits=(0.0, ROCKET_FLIGHT),
                            initial=0.0, actuated=True, parent=name, carries=False)
    # the exhaust: a plume from the bell's mouth, dark and down in the pad and
    # the table until it lights (an orange ball until v14)
    plume = (0.010, 0.026)
    flame = BodySpec(f"{name}_flame", "compound", (plume[0], plume[0], plume[1]),
                     Pose((xy[0], xy[1], z - size[2] - plume[1])), category="flame",
                     color="fire:ink", fixed=False, mass=0.001,
                     attrs={"parts": (P.lathe((0, 0, 0), "flame", plume[0], plume[1], "fire:ink",
                                              rot=P.turn_x(math.pi)),),
                            "collides": False, "graspable": False, "attached_to": name})
    hz = (pad_top - TABLE) / 2
    rod = 0.11
    pad_parts = (P.cylinder((0, 0, 0), 0.035, hz, "metal:steel", collides="hidden"),
                 P.lathe((0, 0, 0), "plinth", 0.038, hz, "metal:steel"),
                 P.slab((0, 0, hz + 0.0007), (0.022, 0.022, 0.0007), "metal:aluminium"),
                 P.lathe((lug[0], lug[1], hz + 0.004), "cylinder", 0.0035, 0.004, "metal:black"),
                 P.lathe((lug[0], lug[1], hz + rod), "cylinder", 0.0012, rod, "metal:chrome"))
    pad = BodySpec(f"{name}_pad", "compound", (0.035, 0.035, hz),
                   Pose((xy[0], xy[1], (pad_top + TABLE) / 2)), category="launch pad",
                   color="metal:steel", fixed=True,
                   attrs={"parts": pad_parts, "graspable": False, "footprint": "round"})
    return body, lift, flame, pad


# -- a toy xylophone ------------------------------------------------------------------
#: Five bars side by side along y, longest on the left, each on a sprung slide
#: 1 cm above the frame; pressed 5 mm, it has played.
XYLO_COLORS = {"red": "paint:red", "orange": "paint:orange", "yellow": "paint:yellow",
               "green": "paint:green", "blue": "paint:blue"}
XYLO_PITCH = 0.047
XYLO_KEY_WIDTH = 0.02                    # half, along y
XYLO_KEY_HALF_HEIGHT = 0.005
XYLO_LENGTHS = (0.05, 0.047, 0.044, 0.041, 0.038)     # half, along x
XYLO_TRAVEL = 0.008
XYLO_FRAME_TOP = TABLE + 0.02
XYLO_KEY_Z = XYLO_FRAME_TOP + 0.01 + XYLO_KEY_HALF_HEIGHT


def xylophone_key(name: str, xy, joint: str, color, half_length: float):
    from ...engines.base import ArticulationSpec

    key = BodySpec(name, "box", (half_length, XYLO_KEY_WIDTH, XYLO_KEY_HALF_HEIGHT),
                   Pose((xy[0], xy[1], XYLO_KEY_Z)), category="xylophone key", color=color,
                   mass=0.02, attrs={"pressable": True, "graspable": False, "carried_by": joint})
    spring = ArticulationSpec(joint, "slide", axis=(0.0, 0.0, 1.0), limits=(-XYLO_TRAVEL, 0.0),
                              initial=0.0, actuated=False, stiffness=300.0, damping=5.0,
                              parent=name, carries=False)
    return key, spring


def xylophone_frame(name: str, xy, keys: int, color="wood:ash") -> BodySpec:
    """Two wooden rails under the bars, and a base between them."""
    half_y = keys * XYLO_PITCH / 2 + 0.01
    hz = (XYLO_FRAME_TOP - TABLE) / 2
    parts = (P.box((0.03, 0, 0), (0.008, half_y, hz), color),
             P.box((-0.03, 0, 0), (0.008, half_y, hz), color),
             P.box((0, 0, -hz + 0.003), (0.04, half_y, 0.003), color))
    return BodySpec(name, "compound", (0.038, half_y, hz), Pose((xy[0], xy[1], TABLE + hz)),
                    category="xylophone", color=color, fixed=True,
                    attrs={"parts": parts, "graspable": False, "footprint": "rect"})


# -- a tic-tac-toe board whose O pieces come up from inside ----------------------------
#: A hollow board 21 cm square and 6 cm tall.  Each of its nine cells is a
#: square post (an X stands on it) in a square well; in the well hangs an O
#: ring on a lift, out of sight until the opponent plays there.
TTT_PITCH = 0.07
TTT_HALF = 0.105
TTT_TOP = TABLE + 0.06
TTT_PLATE = 0.006
TTT_WELL = 0.024                    # half the well's side
TTT_POST = 0.012                    # half the post's side
O_RING = (0.022, 0.018, 0.006)      # outer and inner radius, half height
O_DOWN = -0.042                     # the lift's travel: the ring's top 3 cm down
O_COLOR = "paint:blue"


def ttt_board(name: str, centre_xy, color="wood:ash", inside=(0.25, 0.22, 0.2)) -> BodySpec:
    cx, cy = centre_xy
    h, w, p = TTT_HALF, TTT_WELL, TTT_PITCH
    zc = (TTT_TOP - TABLE) / 2                   # the board's middle, above the table
    top = zc - TTT_PLATE / 2                     # the plate's middle, in the body frame
    parts = [P.box((0, 0, -zc + 0.003), (h, h, 0.003), inside)]
    for s in (-1, 1):
        parts.append(P.box((s * (h - 0.004), 0, 0), (0.004, h, zc), color))
        parts.append(P.box((0, s * (h - 0.004), 0), (h - 0.008, 0.004, zc), color))
    # the plate: strips across between the rows of wells, pieces between wells
    edges = [-h] + [c + s * w for c in (-p, 0.0, p) for s in (-1, 1)] + [h]
    gaps = [(edges[k], edges[k + 1]) for k in range(0, len(edges), 2)]
    for x0, x1 in gaps:
        parts.append(P.box(((x0 + x1) / 2, 0, top), ((x1 - x0) / 2, h, TTT_PLATE / 2), color))
    for c in (-p, 0.0, p):
        for y0, y1 in gaps:
            parts.append(P.box((c, (y0 + y1) / 2, top), (w, (y1 - y0) / 2, TTT_PLATE / 2), color))
    # the posts, and a painted grid
    post_h = (2 * zc - 0.006) / 2
    for r in (-p, 0.0, p):
        for c in (-p, 0.0, p):
            parts.append(P.box((r, c, -zc + 0.006 + post_h), (TTT_POST, TTT_POST, post_h), color))
    for s in (-0.5, 0.5):
        parts.append(P.box((s * p, 0, zc + 0.0004), (0.0015, 1.5 * p - 0.004, 0.0004), inside,
                           collides=False))
        parts.append(P.box((0, s * p, zc + 0.0004), (1.5 * p - 0.004, 0.0015, 0.0004), inside,
                           collides=False))
    return BodySpec(name, "compound", (h, h, zc), Pose((cx, cy, TABLE + zc)),
                    category="tic-tac-toe board", color=color, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def ttt_cell(board: BodySpec, cell: int):
    """A cell's middle: row 0 nearest the robot, column 0 on the camera's left (-y)."""
    r, c = divmod(cell, 3)
    x, y, _ = board.pose.pos
    return (x + (r - 1) * TTT_PITCH, y + (c - 1) * TTT_PITCH)


def o_ring(name: str, xy, joint: str, initial: float):
    """An O on its lift: up (0) it stands on the board round the post."""
    from ...engines.base import ArticulationSpec

    outer, inner, hz = O_RING
    parts = tuple(P.ring((0, 0, 0), outer, inner, hz, O_COLOR, segments=12))
    body = BodySpec(name, "compound", (outer, outer, hz), Pose((xy[0], xy[1], TTT_TOP + hz)),
                    category="O piece", color=O_COLOR, mass=0.02,
                    attrs={"parts": parts, "graspable": False, "carried_by": joint,
                           "footprint": "round"})
    lift = ArticulationSpec(joint, "slide", axis=(0.0, 0.0, 1.0), limits=(O_DOWN, 0.0),
                            initial=initial, actuated=True, parent=name, carries=False)
    return body, lift


X_YAW = -math.pi / 4


def x_piece(name: str, xy, on: float = TABLE, yaw: float = X_YAW,
            color="paint:red") -> BodySpec:
    """A cross; turned 45 degrees it is an X."""
    body = cross_block(name, xy, yaw, color)
    return replace(body, pose=Pose((xy[0], xy[1], on + CROSS_HALF_HEIGHT), body.pose.quat),
                   category="X piece", attrs={**body.attrs, "kind": "X", "grasp_yaw": yaw})


# -- a gold coin -------------------------------------------------------------------------
GOLD = "metal:gold"


COIN_HALF = (0.016, 0.005)          # across its flats, and its thickness


def gold_coin(name: str, xy, on: float, mass: float = 0.01, half=COIN_HALF, **attrs) -> BodySpec:
    """A gold coin, 1 cm thick.  What the fingers meet is eight-sided -- two
    squares a quarter turned, hidden: a round one held by its rim slipped
    out of the fingers on the way (Isaac) -- and what is drawn is round: a
    raised rim, a sunken field, a boss in the middle of each face (v8, 9/24:
    the advisor had seen the brass eight-pointed star too often).  ``half``:
    across the flats, and the thickness, halved."""
    parts = octagon_core(half) + (P.lathe((0, 0, 0), "coin", ROUND_OVER * half[0], half[1], GOLD),)
    size = P.half_extents(parts, solid_only=False)
    return BodySpec(name, "compound", size, Pose((xy[0], xy[1], on + half[1])),
                    category="gold coin", color=GOLD, mass=mass,
                    attrs={"parts": parts, "footprint": "round", **attrs})


#: a round look over an eight-sided core reaches this far past the flats: the
#: core's corners (8.2% out) stay inside it, the fingers on the flats
ROUND_OVER = 1.085


def octagon_core(half, z: float = 0.0) -> tuple:
    """A hidden eight-sided prism ``half`` (across the flats, and the
    thickness, halved) whose flats the fingers close on (``parts.octagon``)."""
    w, h = half
    return (P.octagon((0, 0, z), w, h, "grey", collides="hidden"),)


# -- a toy balance ---------------------------------------------------------------
#: A toy pan balance of the kind whose pans sit on top and stay level (a
#: Roberval balance): each pan rides a slide straight up and down, and the
#: beam between them only shows which way it has gone.  What it shows is worked
#: out from what lies in the pans (``ds4_fake_coin.Balance``), so a few grams
#: tip it every time and where a coin lies in its pan does not matter.  Pans
#: that tilted with the beam tipped a coin in the fingers: taken off a raised
#: pan, it slipped out on every lift (Isaac, 2026-09-19).
BALANCE_ARM = 0.12                   # from the pivot to each pan's middle, along y
BALANCE_PIVOT_Z = TABLE + 0.08
PAN_RADIUS, PAN_RIM = 0.075, 0.004       # three coins side by side
PAN_UP = 0.025                       # a pan's floor above the pivot, level
BALANCE_TILT = math.radians(8.0)     # how far the beam tips
PAN_TRAVEL = BALANCE_ARM * math.sin(BALANCE_TILT)       # how far a pan goes up or down


def balance(prefix: str, xy, hinge_name: str, slide_names):
    """The beam (a carried body on a hinge about x, only showing which way:
    nothing touches it), the two pans (carried bodies on slides, side -1 then +1), and
    the stand.  A positive hinge angle takes the +y end up.

    v8 (9/24), the advisor: the materials were not pretty -- a red beam, blue
    plastic pans.  Now a school balance's: brass pans (drawn round over the
    same colliders), a brass beam with a black pointer, a brass post on a
    wooden base."""
    arm, r = BALANCE_ARM, PAN_RADIUS
    brass, wood = "metal:brass", "wood:oak"
    # the bar keeps a collider (a link needs mass and inertia) though nothing
    # ever reaches it: the pans sit above its ends, the post stops below it
    beam_parts = [P.box((0, 0, 0), (0.006, arm + 0.012, 0.005), brass),
                  P.box((0, 0, 0.024), (0.0012, 0.0012, 0.022), "paint:black", collides=False),
                  P.sphere((0, 0, 0.047), 0.0028, "paint:black", collides=False)]
    beam_parts += [P.cylinder((dx, 0, 0), 0.009, 0.002, brass, collides=False,
                              rot=P.turn_y(math.pi / 2)) for dx in (-0.008, 0.008)]
    beam_parts += [P.sphere((0, s * (arm + 0.012), 0), 0.006, brass, collides=False) for s in (-1, 1)]
    beam = BodySpec(f"{prefix}beam", "compound", P.half_extents(beam_parts, solid_only=False),
                    Pose((xy[0], xy[1], BALANCE_PIVOT_Z)), category="balance beam", color=brass,
                    mass=0.1, attrs={"parts": tuple(beam_parts), "graspable": False,
                                     "carried_by": hinge_name})
    hinge = ArticulationSpec(hinge_name, "hinge", axis=(1.0, 0.0, 0.0),
                             limits=(-BALANCE_TILT - 0.02, BALANCE_TILT + 0.02), initial=0.0,
                             actuated=True, parent=beam.name, carries=False)
    pans, slides = [], []
    for side, slide_name in zip((-1, 1), slide_names):
        hidden = [p[:5] + ("hidden",) for p in
                  P.ring((0, 0, 0.002 + PAN_RIM / 2), r, r - 0.004, PAN_RIM / 2, brass, segments=16)]
        parts = [P.cylinder((0, 0, 0), r, 0.002, brass, collides="hidden"), *hidden,
                 P.lathe((0, 0, 0.002), "pan", r, 0.004, brass),
                 P.cylinder((0, 0, -(PAN_UP - 0.004) / 2 - 0.002), 0.004, (PAN_UP - 0.004) / 2,
                            brass, collides=False)]
        pan = BodySpec(f"{prefix}pan_{'left' if side < 0 else 'right'}", "compound",
                       P.half_extents(parts), Pose((xy[0], xy[1] + side * arm,
                                                    BALANCE_PIVOT_Z + PAN_UP + 0.002)),
                       category="balance pan", color=brass, mass=0.05,
                       attrs={"parts": tuple(parts), "graspable": False, "carried_by": slide_name,
                              "container": "open", "wall": 0.004})
        pans.append(pan)
        slides.append(ArticulationSpec(slide_name, "slide", axis=(0.0, 0.0, 1.0),
                                       limits=(-PAN_TRAVEL - 0.005, PAN_TRAVEL + 0.005),
                                       initial=0.0, actuated=True, parent=pan.name, carries=True))
    top = BALANCE_PIVOT_Z - 0.008 - TABLE      # the post stops short of the beam
    stand = [P.box((0, 0, -top / 2 + 0.006), (0.05, 0.07, 0.006), wood),
             P.cylinder((0, 0, -top / 2 + 0.013), 0.016, 0.001, brass, collides=False),
             P.cylinder((0, 0, 0.006), 0.007, top / 2 - 0.006, brass)]
    base = BodySpec(f"{prefix}stand", "compound", (0.05, 0.07, top / 2),
                    Pose((xy[0], xy[1], TABLE + top / 2)), category="balance", color=brass,
                    fixed=True, attrs={"parts": tuple(stand), "graspable": False})
    return beam, hinge, pans, slides, base


def pan_centre(pivot, lift: float, side: int):
    """Where a pan's floor is, the pan ``lift`` above level (side -1: -y)."""
    return (pivot[0], pivot[1] + side * BALANCE_ARM, pivot[2] + PAN_UP + 0.004 + lift)


# -- a whack-a-mole machine -----------------------------------------------------------
#: A long cabinet between the arms, its top grass-green with a row of holes at
#: each end; a mole in each hole on a driven slide.  Down, a mole's head is
#: inside the cabinet (the holes are wider than it, so nothing fixed touches
#: what moves); up, it stands MOLE_RISE above the top, face to the robot.
#: v8 (9/24), the advisor: a mole coming up was easy to miss.  It now rises
#: 5 cm, its whole head and a collar of body above the top (3.5 until then,
#: the dome and no more); the top is 1.5 cm lower and the body shorter, so a
#: head waiting over a risen mole is where it was, within the X5's reach
MOLE_TOP = TABLE + 0.055             # the cabinet's top
MOLE_RADIUS, MOLE_HOLE = 0.022, 0.030
MOLE_RISE = 0.05
MOLE_BODY = 0.0125                   # half the height of its body, below the dome
MOLE_DOWN_Z = MOLE_TOP - 0.005 - MOLE_RADIUS - MOLE_BODY     # its middle, down


def mole(name: str, xy, joint: str):
    """A mole (a carried body, down at joint 0) and its slide (0 -> MOLE_RISE):
    brown felt, a pale muzzle and a pink nose, round white eyes with black
    pupils and two little ears, looking at the robot (-x)."""
    fur, black, white = "fabric:brown", "paint:black", "paint:white"
    r, b = MOLE_RADIUS, MOLE_BODY
    parts = [P.cylinder((0, 0, 0), r, b, fur), P.sphere((0, 0, b), r, fur)]
    for dy in (-0.0085, 0.0085):
        parts += [P.sphere((-r * 0.80, dy, b + 0.007), 0.0048, white, collides=False),
                  P.sphere((-r * 0.80 - 0.0036, dy, b + 0.007), 0.0027, black, collides=False),
                  P.ellipsoid((-0.004, dy * 1.9, b + r * 0.72), (0.0045, 0.0028, 0.0055), fur)]
    parts += [P.ellipsoid((-r * 0.72, 0, b - 0.005), (0.009, 0.011, 0.0075), "fabric:tan"),
              P.sphere((-r * 1.04, 0, b - 0.002), 0.0048, "plastic:pink", collides=False)]
    body = BodySpec(name, "compound", (r, r, b + r / 2), Pose((xy[0], xy[1], MOLE_DOWN_Z)),
                    category="mole", color="brown", mass=0.05,
                    attrs={"parts": tuple(parts), "graspable": False, "carried_by": joint,
                           "footprint": "round"})
    slide = ArticulationSpec(joint, "slide", axis=(0.0, 0.0, 1.0), limits=(0.0, MOLE_RISE + 0.005),
                             initial=0.0, actuated=True, parent=name, carries=False)
    return body, slide


def mole_cabinet(name: str, centre_xy, half: Tuple[float, float], holes_xy) -> BodySpec:
    """The cabinet: a wooden toy's -- sides painted red, an ash top of strips
    around the holes, a black ring round each (v8; a grass-green top with
    yellow rings until then) -- open underneath, all fixed."""
    cx, cy = centre_xy
    hx, hy = half
    hz = (MOLE_TOP - TABLE) / 2
    side, top = "paint:red", "wood:ash"
    t = 0.006                                               # the boards' thickness
    # no bottom board: it stands on the table, and a mole down reaches to 3 mm above it
    parts = [P.box((hx - t / 2, 0, 0), (t / 2, hy, hz), side),
             P.box((-hx + t / 2, 0, 0), (t / 2, hy, hz), side),
             P.box((0, hy - t / 2, 0), (hx, t / 2, hz), side),
             P.box((0, -hy + t / 2, 0), (hx, t / 2, hz), side)]
    # the top: along y, strips between the hole rows' edges; across each row,
    # blocks between its holes
    rows = sorted({round(y - cy, 4) for _, y in holes_xy})
    ys = [-hy]
    for y in rows:
        ys += [y - MOLE_HOLE, y + MOLE_HOLE]
    ys.append(hy)
    for a, b in zip(ys[0::2], ys[1::2]):
        if b - a > 1e-4:
            parts.append(P.box((0, (a + b) / 2, hz - t / 2), (hx, (b - a) / 2, t / 2), top))
    for y in rows:
        xs = sorted(x - cx for x, yy in holes_xy if abs(yy - cy - y) < 1e-4)
        edges = [-hx] + [e for x in xs for e in (x - MOLE_HOLE, x + MOLE_HOLE)] + [hx]
        for a, b in zip(edges[0::2], edges[1::2]):
            if b - a > 1e-4:
                parts.append(P.box(((a + b) / 2, y, hz - t / 2), ((b - a) / 2, MOLE_HOLE, t / 2), top))
        for x in xs:
            # to look at only: turned boxes, whose bounds would reach into the hole
            parts += [p[:5] + (False,) for p in P.ring((x, y, hz + 0.0012), MOLE_HOLE + 0.006,
                                                       MOLE_HOLE, 0.0012, "paint:dark",
                                                       segments=24)]
    return BodySpec(name, "compound", (hx, hy, hz), Pose((cx, cy, TABLE + hz)), category="cabinet",
                    color="wood:ash", fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


