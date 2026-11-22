"""CR4 "seat the graphics card" -- Contact-Rich Manipulation.

Knob: the play of the card's edge in the mouth of its slot (3 mm each side
-> 1 mm).  Below the mouth the slot narrows to grip the edge, as a real
one's contacts do.

A small motherboard lies on the bench, its long slot empty.  Beside it, on
the foam pad on top of its box, lies the graphics card, fan up, its top edge
over the pad's edge.  "Fit the graphics card into the long white slot on the
motherboard and press it all the way in."  What to do is all there.  How is
not: taken from above, across its height, the card is held lying, its gold
edge against a finger -- it cannot go into a slot like that, and the hand
cannot turn it over.  So the hand comes level from the side, takes the edge
that overhangs between a finger above and a finger below, lifts, and turns
the wrist down: the card now hangs from the fingers, its gold edge at the
bottom.  Then it is turned to the slot, lowered until the edge is in the
slot's mouth, and pressed: the slot holds the card up 5 mm (a sprung seat
nobody sees stands for its contacts) until it is pushed home, where it stays.

The hand works level only above 11 cm over the table (its body is 20.8 cm
long along the fingers' travel, and that stands upright when they close
upward): which is why the card lies on its box, 15 cm up (measured on Isaac,
runs/cr4/probe_reach.json, 10/5: pointing to the robot's right with the
fingers closing upward it reaches down to 11-13 cm at 45-55 cm ahead; pointing
left, or ahead with the fingers closing sideways, it does not get there).

EmbodiedSWE's assembly.pc_gpu (2026) is this task with the grasp given away:
its card welds to the hand when the fingers close on it, because "the lying
default is ungraspable by a Franka gripper".

The look (v1.12): what such parts show is drawn on them -- a circuit board's
traces and small parts, a fan's blades, a memory module's chips, a carton's
tape and shipping label (``engines/prints_pc.py``).  Until then they were flat
colours, and a collaborator found the clips "a bit cartoon".  Nothing that a
hand, the card or the slot meets changed with it: every collider is where
and what it was.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

from ...core.config import RunConfig
from ...core.geometry import quat_conj, quat_mul, rotate, yaw_of
from ...core.rng import SeedBundle
from ...core.types import GRASP_DOWN, Pose, WorldState
from ...engines import parts as P
from ...engines.prints_pc import FOAM_RGB, KRAFT_RGB, PCB_BLACK, PCB_GREEN
from ...engines.base import ArticulationSpec, BodySpec, Engine
from ...events.reactor import Reactor
from ...interfaces.base import TurnContext
from ...interfaces.commands import Command, Done, MoveTo, SetGripper, Wait
from ...policies.base import ScriptedPolicy
from ...scoring import Custom, Finally, HoldsFor, Milestone, Scorer
from ..instance import TaskInstance
from ..loader import get
from ._common import RIGHT, TABLE, Frame, ahead, station_scene

CARD, BOARD, BOX, SEAT = "card", "motherboard", "box", "slot_seat"

# -- the card: a low-profile, single-slot one.  Its own frame is the way it
# stands in a slot: x along its length, y through it (the fan side is +y),
# z up; its gold edge -- the tab -- hangs 8 mm below the board
CARD_L, CARD_T, CARD_H = 0.150, 0.017, 0.069
TAB_L, TAB_T, TAB_H = 0.085, 0.0016, 0.008
TAB_ALONG = -0.0225                      # the tab's middle from the card's, along its length
PCB_Y = -(CARD_T - TAB_T) / 2            # the circuit board is the card's back face
CARD_HALF = (CARD_L / 2, CARD_T / 2, (CARD_H + TAB_H) / 2)
BODY_Z = TAB_H / 2                       # the body's middle, and the tab's, in that frame
TAB_Z = -CARD_HALF[2] + TAB_H / 2
CARD_MASS = 0.15
#: lying fan up, its top edge toward the robot's left (+y): half a turn about
#: the line between its y and z
LYING = (0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5))

# -- the box it lies on, a foam pad on top: 15 cm up, the edge of the pad
# 2.6 cm short of the card's top edge (the card's weight is then a centimetre
# inside the pad's edge)
BOX_AT = (0.30, -0.25)
BOX_HALF = (0.08, 0.07, 0.07)
PAD_HALF = (0.07, 0.045, 0.005)
PAD_TOP = TABLE + 2 * BOX_HALF[2] + 2 * PAD_HALF[2]
OVERHANG = 0.026
CARD_JITTER, CARD_TURN = 0.012, math.radians(5)

# -- the motherboard: 17 cm square, its top 8 mm over the bench, the slot's
# body 11 mm more; the groove 8 mm deep
BOARD_AT = (0.30, 0.06)
BOARD_HALF = (0.085, 0.085)
BOARD_TOP, SLOT_H, GROOVE = 0.008, 0.011, 0.008
SLOT_AT = (0.045, -0.010)                # the groove's middle on the board (the groove runs along its y)
SLOT_WALL = 0.003
GROOVE_HALF_L = TAB_L / 2 + 0.004       # 4 mm of play along the groove, at both levels
BOARD_JITTER, BOARD_TURN = 0.010, math.radians(8)
#: the knob: the play of the tab in the groove's mouth, each side.  The mouth
#: is the top 5 mm; from there the walls close in to ``TIGHT`` a side for the
#: last 3 mm, where they hold the card upright -- with the mouth's play all the
#: way down, a card let go leant 14 degrees in its slot and its weight, off to
#: the fan's side of its edge, kept it there (Isaac, 10/5)
FIT = {"easy": 0.0030, "hard": 0.0010}
TIGHT, GRIP_DEPTH = 0.0002, 0.003

# -- the slot's hold on the card (as CR5's studs): a sprung seat in the groove
# keeps the tab ``REST`` above the floor -- 2 mm of sag under the card, 4 N at
# the bottom -- and, held down with the tab on it, stays down
REST = 0.005
SEAT_T, SEAT_MASS, SEAT_STIFFNESS, SEAT_DAMPING = 0.002, 0.01, 800.0, 15.0
LATCH_AT, DWELL, UNDER, GONE = 0.004, 0.3, 0.0005, 0.5

#: seated: the tab this near the groove's floor, the card this near upright,
#: for this long with the hand off it
HOME_Z, HOME_TILT, STABLE_SECONDS = 0.0015, math.radians(12), 2.0
HAND_OFF = 0.06

# -- the reference's hand: the fingertips this far in from the card's edge,
# coming from this far out; the card carried this high over the slot
GRIP_IN, COME_FROM, LIFT, CLEAR_OF_BOX, HOVER = 0.013, 0.07, 0.10, 0.07, 0.05
MOUTH, JUST_OVER, IN_THE_MOUTH, PAST, SETTLE = 0.002, 0.0003, 0.0015, 0.004, 0.5
MAX_REFINES, MAX_TRIES = 6, 2
#: the blind play lets the card go with the hand this high over the slot's top:
#: lying in the fingers it is then just clear of the cooler beside the slot
FLAT_OVER = 0.032

FRAME = Frame((0.12, 0.46), (-0.34, 0.20), 0.24)


# -- poses ------------------------------------------------------------------------
def turn_z(angle: float):
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def compose(a: Pose, b: Pose) -> Pose:
    """``b``, given in ``a``'s frame, in the frame ``a`` is given in."""
    return Pose(tuple(p + r for p, r in zip(a.pos, rotate(a.quat, b.pos))), quat_mul(a.quat, b.quat))


def inverse(a: Pose) -> Pose:
    qi = quat_conj(a.quat)
    return Pose(tuple(-v for v in rotate(qi, a.pos)), qi)


def angle_between(a, b) -> float:
    """The turn that takes one orientation to the other."""
    dot = abs(sum(x * y for x, y in zip(a, b)))
    return 2 * math.acos(max(-1.0, min(1.0, dot)))


def quat_of_axes(x, y, z):
    """The orientation whose own x, y, z axes point along these."""
    m = ((x[0], y[0], z[0]), (x[1], y[1], z[1]), (x[2], y[2], z[2]))
    t = m[0][0] + m[1][1] + m[2][2]
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        return (0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s)
    if m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2
        return ((m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s)
    if m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2
        return ((m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s)
    s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2
    return ((m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s)


def cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


# -- the bodies -------------------------------------------------------------------
#: a print lies this far proud of the face it is on
PROUD = 0.0002


def face(centre, width: float, spec: str, normal, top, wrap: float = 0.0):
    """A printed face (``engines/prints_pc.py``) on a side of a part: looking
    out along ``normal``, its image's top toward ``top``, ``width`` wide --
    wrapped, with ``wrap``, round a thing of that radius behind it."""
    return P.printed(centre, width, spec, quat_of_axes(top, cross(normal, top), normal), wrap=wrap)


def card_body(xy, turn: float) -> BodySpec:
    """The card, lying on the pad.  What a hand and the slot meet is its slab
    and its tab; what is drawn is what a low-profile card shows: a black
    circuit board, its back printed; the shroud over it, the fan in its well
    and the heatsink's fins behind a grille printed on its face; the fins'
    tops between the two; the gold edge with its key; the steel bracket across
    the end, a socket and its vents in it."""
    hx, hy, hz = CARD_HALF
    body = (hx, hy, CARD_H / 2)
    out = hy + PROUD
    end = -hx - 0.0001                                    # just off the bracket's plate
    parts = [
        P.box((0, 0, BODY_Z), body, "grey", collides="hidden"),
        P.box((TAB_ALONG, PCB_Y, TAB_Z), (TAB_L / 2, TAB_T / 2, TAB_H / 2), "metal:gold"),
        # the key: the gap in the gold a slot's ridge goes into
        P.box((TAB_ALONG - TAB_L / 2 + 0.0125, PCB_Y, TAB_Z + 0.0005),
              (0.00095, TAB_T / 2 + 0.0001, TAB_H / 2 - 0.0005), PCB_BLACK, collides=False),
        # the circuit board, the shroud over it, the fins between them at the top
        P.box((0, PCB_Y, BODY_Z), (hx, TAB_T / 2, CARD_H / 2), PCB_BLACK, collides=False),
        P.box((0.004, 0.0012, BODY_Z - 0.002), (hx - 0.008, hy - 0.0014, CARD_H / 2 - 0.004),
              "plastic:black", collides=False),
        P.box((0.004, -0.002, BODY_Z + CARD_H / 2 - 0.003), (hx - 0.012, 0.004, 0.003),
              "metal:aluminium", collides=False),
        # its two faces: the shroud with its fan (the bracket's end to the
        # right of that picture), the board's back (the bracket's end to the left)
        face((0.004, out, BODY_Z - 0.002), 2 * (hx - 0.008), "gpu_shroud:", (0, 1, 0), (0, 0, 1)),
        face((0, -out, BODY_Z), CARD_L, "gpu_back:", (0, -1, 0), (0, 0, 1)),
        # the fins' tops, along the card's top edge
        face((0.004, -0.002, BODY_Z + CARD_H / 2 + PROUD), 2 * (hx - 0.012), "gpu_fins:", (0, 0, 1), (0, 1, 0)),
        # the bracket's plate across the end, its two sockets and the vents under them
        P.box((-hx + 0.0006, 0.0, BODY_Z), (0.0006, hy, CARD_H / 2), "metal:steel", collides=False),
        P.box((end, 0.0, BODY_Z + 0.019), (0.0001, 0.0029, 0.0075), "plastic:black", collides=False),
        P.box((end, 0.0, BODY_Z - 0.001), (0.0001, 0.0026, 0.0081), "plastic:black", collides=False),
    ]
    parts += [P.box((end, 0.0, BODY_Z - 0.031 + 0.0022 * k), (0.0001, 0.0056, 0.0006), "plastic:black",
                    collides=False) for k in range(8)]
    quat = quat_mul(turn_z(turn), LYING)
    return BodySpec(CARD, "compound", CARD_HALF, Pose((xy[0], xy[1], PAD_TOP + hy), quat),
                    category="graphics card", color="plastic:black", mass=CARD_MASS,
                    attrs={"parts": tuple(parts), "footprint": "rect",
                           # taken from above it is held lying, which is no use
                           # here: the grasp the checks would look at is not
                           # the one the task is about
                           "graspable": False})


def box_body() -> BodySpec:
    """The carton the card came in -- corrugated board, its flaps taped, a
    shipping label on the end the fixed camera sees -- and on top of it a pad
    of pink anti-static foam."""
    x, y = ahead(*BOX_AT)
    hx, hy, hz = BOX_HALF
    px, py, pz = PAD_HALF
    pad = (0.0, hy - py, hz + pz)                          # flush with the box's left edge
    total = hz + pz
    mid = -pz                                              # the carton's middle, in the body's frame
    parts = [P.box((0, 0, mid), BOX_HALF, KRAFT_RGB, collides="hidden"),
             P.box((0, 0, mid), BOX_HALF, KRAFT_RGB, collides=False),
             face((0, 0, mid + hz + PROUD), 2 * hx, "carton:top", (0, 0, 1), (0, 1, 0)),
             face((0, hy + PROUD, mid), 2 * hx, "carton:side", (0, 1, 0), (0, 0, 1)),
             face((0, -hy - PROUD, mid), 2 * hx, "carton:side", (0, -1, 0), (0, 0, 1)),
             face((hx + PROUD, 0, mid), 2 * hy, "carton_end:label", (1, 0, 0), (0, 0, 1)),
             face((-hx - PROUD, 0, mid), 2 * hy, "carton_end:plain", (-1, 0, 0), (0, 0, 1)),
             P.box((pad[0], pad[1], pad[2] - pz), PAD_HALF, FOAM_RGB),
             face((pad[0], pad[1], pad[2] + PROUD), 2 * px, "foam_pad:", (0, 0, 1), (0, 1, 0))]
    return BodySpec(BOX, "compound", (hx, hy, total), Pose((x, y, TABLE + total)),
                    category="box", color=KRAFT_RGB, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def groove_half(level: str) -> float:
    """Half the groove's width at its mouth."""
    return TAB_T / 2 + FIT[level]


#: the cooler on the processor: where it stands on the board, its radius and
#: height (what the card meets is a cylinder this size), and its fan's blades
COOLER_AT, COOLER_R, COOLER_H = (-0.030, 0.020), 0.030, 0.028
SINK_H, BLADES, BLADE_PITCH = 0.016, 7, math.radians(32)


def cooler(up: float) -> List[tuple]:
    """A low round cooler: an aluminium heatsink, its fins a band round it,
    and on it a fan in a round frame -- hub, label, seven blades.  Solid as
    one cylinder (a lathe is drawn only: without it a card laid on the board
    lay half inside the cooler -- the blind play, Isaac, 10/5)."""
    cx, cy = COOLER_AT
    black = "plastic:black"
    band = 2 * math.pi * (COOLER_R - 0.0012)
    parts = [
        P.cylinder((cx, cy, up + COOLER_H / 2), COOLER_R, COOLER_H / 2, "grey", collides="hidden"),
        P.lathe((cx, cy, up + SINK_H / 2), "cylinder", COOLER_R - 0.0025, SINK_H / 2, "metal:aluminium"),
        face((cx + band / (2 * math.pi), cy, up + SINK_H / 2), band, "fins:", (1, 0, 0), (0, 0, 1),
             wrap=band / (2 * math.pi)),
        P.lathe((cx, cy, up + (SINK_H + COOLER_H - 0.001) / 2), "washer", COOLER_R,
                (COOLER_H - 0.001 - SINK_H) / 2, black, inner=COOLER_R - 0.0032),
        P.lathe((cx, cy, up + SINK_H + 0.0045), "cylinder", 0.009, 0.0045, black),
        P.lathe((cx, cy, up + SINK_H + 0.0092), "cylinder", 0.0066, 0.0002, "metal:aluminium"),
    ]
    inner, outer = 0.0085, COOLER_R - 0.0040
    for k in range(BLADES):
        a = 2 * math.pi * k / BLADES
        r = (inner + outer) / 2
        parts.append(P.box((cx + r * math.cos(a), cy + r * math.sin(a), up + SINK_H + 0.0048),
                           ((outer - inner) / 2, 0.0062, 0.0004), "plastic:dark",
                           rot=quat_mul(P.turn_z(a), P.turn_x(BLADE_PITCH)), collides=False))
    return parts


def board_body(level: str, xy, turn: float) -> BodySpec:
    """The motherboard: its slab, the slot -- two walls, two ends and a floor
    round the groove -- and what makes it one: its printed face, a cooler on
    its processor, memory in its sockets, the connectors along its edges."""
    bx, by = BOARD_HALF
    top = BOARD_TOP + SLOT_H
    hz = top / 2
    g = groove_half(level)
    sx, sy = SLOT_AT
    z0 = -hz                                               # the bench
    slab = (0, 0, z0 + BOARD_TOP / 2)
    wall_z = z0 + BOARD_TOP + SLOT_H / 2
    floor_h = SLOT_H - GROOVE
    # a green board, its long slot white -- the one long white thing on it --
    # the rest of its sockets black
    black, white, steel, nylon = "plastic:black", "plastic:white", "metal:steel", "plastic:cream"
    up = z0 + BOARD_TOP
    # the board on the sheet of anti-static foam it came on: 6 mm of foam, 2 mm of board
    parts = [P.box(slab, (bx, by, BOARD_TOP / 2), PCB_GREEN, collides="hidden"),
             P.box((0, 0, z0 + 0.003), (bx + 0.006, by + 0.006, 0.003), FOAM_RGB, collides=False),
             P.box((0, 0, z0 + 0.007), (bx, by, 0.001), PCB_GREEN, collides=False),
             P.printed((0, 0, up + PROUD), 2 * by, "mainboard:")]
    floor = z0 + BOARD_TOP + floor_h
    tight = TAB_T / 2 + TIGHT
    # a wall's upper part leans out from the grip to the mouth
    rise = GROOVE - GRIP_DEPTH
    lean = math.atan2(g - tight, rise)
    long = GROOVE_HALF_L + SLOT_WALL
    for s in (-1, 1):
        # below the mouth: upright, tight on the tab, down to the board
        low_h = floor_h + GRIP_DEPTH
        parts.append(P.box((sx + s * (tight + SLOT_WALL / 2), sy, z0 + BOARD_TOP + low_h / 2),
                           (SLOT_WALL / 2, long, low_h / 2), white))
        mx, mz = (tight + g) / 2, floor + GRIP_DEPTH + rise / 2
        parts.append(P.box((sx + s * (mx + math.cos(lean) * SLOT_WALL / 2), sy,
                            mz - math.sin(lean) * SLOT_WALL / 2),
                           (SLOT_WALL / 2, long, math.hypot(g - tight, rise) / 2), white,
                           rot=P.turn_y(s * lean)))
        parts.append(P.box((sx, sy + s * (GROOVE_HALF_L + SLOT_WALL / 2), wall_z),
                           (g, SLOT_WALL / 2, SLOT_H / 2), white))
    parts.append(P.box((sx, sy, z0 + BOARD_TOP + floor_h / 2), (g, GROOVE_HALF_L, floor_h / 2), white))
    # -- what stands on the board, on the robot's side of the slot, and is
    # solid: the cooler, the two memory sockets and the module in the first,
    # the block of ports at the back, the power connector
    parts += cooler(up)
    parts += [
        P.box((-0.030, -0.055, up + 0.004), (0.045, 0.004, 0.004), black),
        P.box((-0.030, -0.066, up + 0.004), (0.045, 0.004, 0.004), black),
        P.box((-0.030, -0.055, up + 0.016), (0.040, 0.0012, 0.012), PCB_BLACK),
        P.box((-0.068, 0.070, up + 0.010), (0.014, 0.012, 0.010), steel),
        P.box((0.070, 0.060, up + 0.004), (0.010, 0.016, 0.004), nylon),
    ]
    # -- and what is only drawn: the module's two sides and the power
    # connector's top, printed; the sockets' latches; the capacitors' cans by
    # the processor; the heatsinks on the chipset and on the regulators; the
    # connectors along the rear and the front edge; the clock's battery.  None
    # of it stands higher than the slot, but for the rear connectors, which
    # are a hand's width from where a seated card is
    parts += [face((-0.030, -0.055 - s * (0.0012 + PROUD), up + 0.016), 0.080, "dimm:", (0, -s, 0), (0, 0, 1))
              for s in (1, -1)]
    parts.append(P.printed((0.070, 0.060, up + 0.008 + PROUD), 0.032, "atx:"))
    parts += [P.box((-0.030 + s * 0.0475, y, up + 0.007), (0.0025, 0.0036, 0.007), "plastic:grey", collides=False)
              for y in (-0.055, -0.066) for s in (-1, 1)]
    parts += [P.cylinder((x, 0.055, up + 0.004), 0.0031, 0.004, "metal:aluminium", collides=False)
              for x in (-0.048, -0.038, -0.028, -0.018)]
    parts.append(P.box((0.030, 0.062, up + 0.0015), (0.014, 0.014, 0.0015), steel, collides=False))
    parts += [P.box((0.030, 0.062 - 0.012 + 0.004 * k, up + 0.0055), (0.014, 0.0009, 0.0025), steel, collides=False)
              for k in range(7)]
    parts.append(P.box((-0.0795, 0.0205, up + 0.004), (0.0033, 0.031, 0.004), steel, collides=False))
    parts += [P.box((-0.0795 + dx, 0.0205, up + 0.0095), (0.0007, 0.031, 0.0015), steel, collides=False)
              for dx in (-0.0024, 0.0, 0.0024)]
    parts += [
        P.box((-0.040, 0.077, up + 0.006), (0.010, 0.006, 0.006), nylon, collides=False),
        P.box((-0.017, 0.0775, up + 0.006), (0.009, 0.007, 0.006), steel, collides=False),
        P.box((0.005, 0.0775, up + 0.0075), (0.009, 0.007, 0.0075), black, collides=False),
        P.box((0.030, -0.077, up + 0.004), (0.0075, 0.007, 0.004), black, collides=False),
        P.box((0.048, -0.077, up + 0.004), (0.0075, 0.007, 0.004), black, collides=False),
        P.lathe((0.070, 0.012, up + 0.0012), "washer", 0.0112, 0.0012, black, inner=0.0098),
        P.lathe((0.070, 0.012, up + 0.0016), "coin", 0.0098, 0.0016, steel),
    ]
    return BodySpec(BOARD, "compound", (bx, by, hz),
                    Pose((xy[0], xy[1], TABLE + hz), turn_z(turn)),
                    category="motherboard", color=PCB_GREEN, fixed=True,
                    attrs={"parts": tuple(parts), "graspable": False, "footprint": "rect"})


def groove(board) -> Tuple[Tuple[float, float], float, float]:
    """(the groove's middle, the way it runs -- an angle --, its floor's height)."""
    yaw = yaw_of(board.pose.quat)
    dx, dy, _ = rotate(board.pose.quat, (SLOT_AT[0], SLOT_AT[1], 0.0))
    floor = board.pose.pos[2] - (BOARD_TOP + SLOT_H) / 2 + BOARD_TOP + SLOT_H - GROOVE
    return (board.pose.pos[0] + dx, board.pose.pos[1] + dy), yaw + math.pi / 2, floor


def seat(level: str, board: BodySpec) -> Tuple[BodySpec, ArticulationSpec]:
    """The sprung seat in the groove, nothing of it drawn, which only the card
    meets."""
    (gx, gy), _, floor = groove(board)
    # as wide as the mouth (it passes through the walls): the tab meets it
    # wherever in the mouth it comes down, and slides on it to the middle
    size = (groove_half(level) - 0.0001, GROOVE_HALF_L - 0.0005, SEAT_T / 2)
    body = BodySpec(SEAT, "compound", size, Pose((gx, gy, floor + REST - SEAT_T / 2), board.pose.quat),
                    mass=SEAT_MASS,
                    attrs={"parts": (P.box((0, 0, 0), size, "grey", collides="hidden"),),
                           "graspable": False, "footprint": "rect", "carried_by": SEAT,
                           "passes": (BOARD, BOX), "friction": 0.05, "starts_fitted": True})
    joint = ArticulationSpec(SEAT, "slide", axis=(0.0, 0.0, 1.0), limits=(-(REST + 2 * UNDER), 0.0),
                             initial=0.0, actuated=False, stiffness=SEAT_STIFFNESS,
                             damping=SEAT_DAMPING, parent=SEAT, carries=True)
    return body, joint


def layout(seed: int):
    """(the card's place and turn on its pad, the board's place and turn)."""
    rng = SeedBundle(seed).stream("bench")
    bx, by = ahead(*BOX_AT)
    pad_edge = by + BOX_HALF[1]
    # the card's middle: its top edge OVERHANG past the pad's edge
    card = (bx + rng.uniform(-CARD_JITTER, CARD_JITTER), pad_edge + OVERHANG - CARD_HALF[2])
    card_turn = rng.uniform(-CARD_TURN, CARD_TURN)
    board = ahead(BOARD_AT[0] + rng.uniform(-BOARD_JITTER, BOARD_JITTER),
                  BOARD_AT[1] + rng.uniform(-BOARD_JITTER, BOARD_JITTER))
    return card, card_turn, board, rng.uniform(-BOARD_TURN, BOARD_TURN)


def build_scene(level: str, seed: int):
    card_xy, card_turn, board_xy, board_turn = layout(seed)
    board = board_body(level, board_xy, board_turn)
    seat_body, joint = seat(level, board)
    bodies = [box_body(), board, seat_body, card_body(card_xy, card_turn)]
    return station_scene(f"cr4_graphics_card_{level}", bodies, articulations=[joint], frame=FRAME)


# -- what counts ------------------------------------------------------------------
def tab_bottom(w: WorldState) -> Tuple[float, float, float]:
    card = w.obj(CARD).pose
    return compose(card, Pose((TAB_ALONG, PCB_Y, -CARD_HALF[2]))).pos


def tilt(w: WorldState) -> float:
    """How far the card's top edge is from straight up."""
    return math.acos(max(-1.0, min(1.0, rotate(w.obj(CARD).pose.quat, (0.0, 0.0, 1.0))[2])))


def in_groove(w: WorldState, level: str, slack: float = 0.001) -> Tuple[bool, float]:
    """(is the tab's bottom over the groove, how far above the groove's floor it is)."""
    (gx, gy), run, floor = groove(w.obj(BOARD))
    x, y, z = tab_bottom(w)
    dx, dy = x - gx, y - gy
    along = math.cos(run) * dx + math.sin(run) * dy
    across = -math.sin(run) * dx + math.cos(run) * dy
    over = abs(along) <= GROOVE_HALF_L - TAB_L / 2 + slack and abs(across) <= FIT[level] + slack
    return over, z - floor


def depth(w: WorldState, level: str) -> float:
    """How far down the groove the tab's bottom is: nothing for a card whose
    edge is not over it (one lying across the slot is no way in)."""
    over, up = in_groove(w, level)
    return max(0.0, GROOVE - up) if over else 0.0


def hand_off(w: WorldState, arm: str = RIGHT) -> bool:
    hand = w.arm(arm)
    return hand.holding != CARD and (hand.gripper < 0.5 or math.dist(
        hand.eef_pose.pos, w.obj(CARD).pose.pos) > HAND_OFF)


def seated(w: WorldState, level: str) -> bool:
    over, up = in_groove(w, level)
    return over and up <= HOME_Z and tilt(w) <= HOME_TILT


class Slot(Reactor):
    """What stands for the slot's contacts: the seat.  It gives under the tab
    and, held all the way down with the tab on it, stays down -- the card is
    in.  A card pulled out again lets it go.  On an engine without contact
    the pressing is played here: the seat goes down with the tab a hand
    brings down on it."""

    def __init__(self, level: str) -> None:
        super().__init__()
        self.level = level
        self.on = False
        self._down_for = 0.0
        self._gone_for = 0.0
        self._then: Optional[float] = None

    def tick(self, engine: Engine, world: WorldState) -> None:
        dt = max(0.0, world.sim_time - self._then) if self._then is not None else 0.0
        self._then = world.sim_time
        over, up = in_groove(world, self.level, slack=0.002)
        standing = over and tilt(world) <= HOME_TILT
        value = world.articulations.get(SEAT, 0.0)
        if self.on:
            goal = -(REST + UNDER)
            self._gone_for = 0.0 if standing and up <= REST / 2 else self._gone_for + dt
            if self._gone_for >= GONE:
                self.on = False
                self.note(engine, world, "card_out")
                goal = 0.0
        else:
            played = not engine.capabilities.dynamics
            # without contact: the seat is where the tab a hand holds has it
            held = world.obj(CARD).is_held
            goal = -min(max(REST - up, 0.0), REST) if played and standing and held else 0.0
            down = standing and up <= HOME_Z and (value <= -LATCH_AT or (played and held))
            self._down_for = self._down_for + dt if down else 0.0
            if self._down_for >= DWELL:
                self.on = True
                self._gone_for = 0.0
                self.note(engine, world, "card_in")
                goal = -(REST + UNDER)
        engine.set_articulation_target(SEAT, goal)


def build(level: str, seed: int, config: Optional[RunConfig] = None) -> TaskInstance:
    spec = get("CR4")
    scene = build_scene(level, seed)
    start = next(b for b in scene.bodies if b.name == CARD).pose.pos

    def home(w: WorldState) -> bool:
        return seated(w, level) and hand_off(w)

    def taken(w: WorldState) -> bool:
        c = w.obj(CARD)
        return c.is_held and math.dist(c.pose.pos, start) > 0.04

    done = Custom(home, "the card pressed home, the hand off it")
    milestones = (
        Milestone("card_taken", Custom(taken, "the card off its pad, in the hand")),
        Milestone("card_upright", Custom(lambda w: w.obj(CARD).is_held and tilt(w) <= math.radians(20),
                                         "the card held with its gold edge down")),
        Milestone("edge_in_the_slot", Custom(lambda w: in_groove(w, level)[0]
                                             and in_groove(w, level)[1] <= GROOVE - 0.001,
                                             "the gold edge in the mouth of the slot")),
        Milestone("pressed_home", done),
    )
    return TaskInstance(
        spec=spec, level=level, seed=seed, scene=scene,
        instruction="Fit the graphics card into the long white slot on the motherboard and "
                    "press it all the way in.",
        scorer=Scorer(task_id="CR4", level=level,
                      success=Finally(done) & HoldsFor(done, STABLE_SECONDS),
                      milestones=milestones),
        milestones=milestones,
        reactor_factory=lambda lv=level: Slot(lv),
        reference_factory=lambda inst, lv=level: CardPolicy(lv),
        metadata={"knob": spec.knob.field, "knob_value": FIT[level],
                  "groove_width_mm": round(2000 * groove_half(level), 2), "max_model_calls": 90},
        extra_metrics={"depth_mm": lambda traj, lv=level: round(1000 * depth(traj[-1], lv), 2)},
    )


# -- the reference ------------------------------------------------------------------
class CardPolicy(ScriptedPolicy):
    """Take the card by the edge that overhangs, from the side, a finger above
    it and one below; lift it and bring it over the slot standing -- the
    wrist turned down, the card hanging from the fingers, turned to the slot
    --, square it over the groove, lower its edge into the mouth and press it
    home; wait for the slot to take it, and let go.  ``ignore_rules`` (the
    blind control): takes the card as it lies, from above, and brings it to
    the slot like that."""

    name = "seat_card"

    def __init__(self, level: str, arm: str = RIGHT) -> None:
        self.level, self.arm = level, arm
        self.ignore_rules = False
        self.reset()

    def reset(self) -> None:
        self._tries = 0
        self._refines = 0
        self._twin = 0
        self._stage = "over"
        #: where the hand was last sent, while it holds the card
        self._cmd: Optional[Pose] = None
        self._flat = 0

    # where the hand has to be for the card to be at ``card``
    def _hand_for(self, world: WorldState, card: Pose) -> Pose:
        now_card, now_hand = world.obj(CARD).pose, world.arm(self.arm).eef_pose
        return compose(card, compose(inverse(now_card), now_hand))

    def _card_over_slot(self, world: WorldState, up: float) -> Pose:
        """The card standing over the groove, its tab ``up`` above the floor
        (the fan toward the camera, or -- the other twin -- away from it)."""
        (gx, gy), run, floor = groove(world.obj(BOARD))
        yaw = run + (math.pi if self._twin else 0.0) + math.pi      # the card's x back along the groove
        quat = turn_z(yaw)
        ox, oy, _ = rotate(quat, (TAB_ALONG, PCB_Y, 0.0))
        return Pose((gx - ox, gy - oy, floor + up + CARD_HALF[2]), quat)

    def _take(self, world: WorldState) -> List[Command]:
        card = world.obj(CARD).pose
        out = rotate(card.quat, (0.0, 0.0, 1.0))                    # the way its top edge points
        up = rotate(card.quat, (0.0, 1.0, 0.0))                     # its fan side
        point = tuple(-v for v in out)                              # the fingers point in at it
        quat = quat_of_axes(cross(up, point), up, point)
        edge = compose(card, Pose((0.0, 0.0, CARD_HALF[2]))).pos
        at = tuple(e - o * GRIP_IN for e, o in zip(edge, out))
        back = tuple(e + o * COME_FROM for e, o in zip(edge, out))
        high = (back[0], back[1], back[2] + LIFT)
        lifted = (at[0], at[1], at[2] + LIFT)
        clear = tuple(p + o * CLEAR_OF_BOX for p, o in zip(lifted, out))
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose(high, GRASP_DOWN)),
                MoveTo(self.arm, Pose(high, quat)), MoveTo(self.arm, Pose(back, quat)),
                MoveTo(self.arm, Pose(at, quat)), SetGripper(self.arm, 1.0), Wait(SETTLE, self.arm),
                MoveTo(self.arm, Pose(lifted, quat)), MoveTo(self.arm, Pose(clear, quat))]

    def _take_standing(self, world: WorldState) -> List[Command]:
        """The card standing (in the slot, let go before it was home): taken
        again by its top edge, from above."""
        card = world.obj(CARD).pose
        through = rotate(card.quat, (0.0, 1.0, 0.0))
        point = tuple(-v for v in rotate(card.quat, (0.0, 0.0, 1.0)))
        quat = quat_of_axes(cross(through, point), through, point)
        at = compose(card, Pose((0.0, 0.0, CARD_HALF[2] - GRIP_IN))).pos
        above = (at[0], at[1], at[2] + LIFT)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose(above, quat)),
                MoveTo(self.arm, Pose(at, quat)), SetGripper(self.arm, 1.0), Wait(SETTLE, self.arm)]

    def _flat_hand(self, world: WorldState):
        """(the middle of the card's slab as it lies, the hand pointing down
        with its fingers across the card's height)."""
        card = world.obj(CARD).pose
        along = rotate(card.quat, (1.0, 0.0, 0.0))
        # either way round is the same hold: the one that turns the wrist
        # least.  Sent half a turn about, the wrist ran into its stop and the
        # fingers came down on top of the card (Isaac, 10/5)
        yaw = (math.atan2(along[1], along[0]) + math.pi / 2) % math.pi - math.pi / 2
        return compose(card, Pose((0.0, 0.0, BODY_Z))).pos, quat_mul(turn_z(yaw), GRASP_DOWN)

    def _over_flat(self, world: WorldState) -> List[Command]:
        (x, y, z), down = self._flat_hand(world)
        return [SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose((x, y, z + LIFT), down))]

    def _take_flat(self, world: WorldState) -> List[Command]:
        """The blind play: from above, across the card as it lies, and off to
        the slot with it lying."""
        (x, y, z), down = self._flat_hand(world)
        (gx, gy), _, floor = groove(world.obj(BOARD))
        over = floor + GROOVE + FLAT_OVER
        return [MoveTo(self.arm, Pose((x, y, z + 0.004), down)), SetGripper(self.arm, 1.0),
                Wait(SETTLE, self.arm), MoveTo(self.arm, Pose((x, y, z + LIFT), down)),
                MoveTo(self.arm, Pose((gx, gy, z + LIFT), down)),
                MoveTo(self.arm, Pose((gx, gy, over), down)), Wait(1.0, self.arm),
                SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose((gx, gy, over + LIFT), down)),
                Wait(1.0, self.arm)]

    def commands(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        if ctx.extras.get("executor_busy"):
            return []
        card, hand = world.obj(CARD), world.arm(self.arm)
        if self.ignore_rules:
            # over the card first, then by where it lies under the hand
            self._flat += 1
            if self._flat == 1:
                return self._over_flat(world)
            if self._flat == 2:
                return self._take_flat(world)
            return [Done("put the card on the slot as it lay")]
        if seated(world, self.level) and not card.is_held:
            if hand_off(world):
                return [Done("the card is in")]
            return [SetGripper(self.arm, 0.0), MoveTo(self.arm, Pose(
                (hand.eef_pose.pos[0], hand.eef_pose.pos[1], hand.eef_pose.pos[2] + LIFT),
                hand.eef_pose.quat)), Wait(1.0, self.arm)]
        if not card.is_held:
            self._tries += 1
            if self._tries > MAX_TRIES + 2:
                return [Done("could not take the card up")]
            self._refines, self._cmd = 0, None
            if tilt(world) <= math.radians(25):
                return self._take_standing(world)                    # it stands in the slot, not home
            if tilt(world) < math.radians(60):
                return [Done("the card no longer lies where it can be taken")]
            return self._take(world)
        # held.  The arm gives a millimetre or two under what it carries (its
        # joints are sprung), so where the hand is sent is corrected by what
        # the card is still off -- the command moved, not the hand's own place:
        # sent to where the hand already hangs, it stays there (Isaac, 10/5)
        now = card.pose

        def send(want: Pose, wait: float = SETTLE) -> List[Command]:
            self._cmd = compose(compose(want, inverse(now)), self._cmd)
            return [MoveTo(self.arm, self._cmd), Wait(wait, self.arm)]

        over = self._card_over_slot(world, GROOVE + HOVER)
        mouth = self._card_over_slot(world, GROOVE + MOUTH)
        far = math.dist(now.pos[:2], mouth.pos[:2]) > 0.03
        if self._cmd is None or tilt(world) > math.radians(25) or (far and self._stage != "over"):
            self._cmd, self._stage, self._refines = hand.eef_pose, "over", 0
            return send(over)
        if self._stage == "over":
            self._stage = "mouth"
            return send(mouth)
        if self._stage in ("mouth", "top"):
            # square over the mouth, then again with the edge just over the
            # slot's top: sent the whole way down from 2 mm up, the hand ran a
            # millimetre wide on the way and set the edge down on the wall
            # (hard level, Isaac, 10/5).  Steps this small it takes at rest
            here = mouth if self._stage == "mouth" else self._card_over_slot(world, GROOVE + JUST_OVER)
            across, along = self._off(world)
            if (across <= FIT[self.level] / 4 and along <= 0.0015) or self._refines >= MAX_REFINES:
                self._refines = 0
                if self._stage == "mouth":
                    self._stage = "top"
                    return send(self._card_over_slot(world, GROOVE + JUST_OVER))
                self._stage = "lip"
                return send(self._card_over_slot(world, GROOVE - IN_THE_MOUTH))
            self._refines += 1
            return send(here)
        if self._stage == "lip":
            self._stage = "pressed"
            return send(self._card_over_slot(world, -PAST), DWELL + 0.7)
        # pressed and still in the hand: let go if it is in, else lift and try again
        if seated(world, self.level):
            self._cmd = None
            return [SetGripper(self.arm, 0.0), Wait(SETTLE, self.arm), MoveTo(self.arm, Pose(
                (hand.eef_pose.pos[0], hand.eef_pose.pos[1], hand.eef_pose.pos[2] + LIFT),
                hand.eef_pose.quat)), Wait(1.0, self.arm)]
        self._tries += 1
        if self._tries > MAX_TRIES + 2:
            return [SetGripper(self.arm, 0.0), Done("the card would not go in")]
        self._stage, self._refines = "over", 0
        return send(over)

    def _off(self, world: WorldState) -> Tuple[float, float]:
        """How far the tab is off the groove: the worse of its two ends,
        across; and its middle, along."""
        (gx, gy), run, _ = groove(world.obj(BOARD))
        card = world.obj(CARD).pose
        across, along = 0.0, 0.0
        for end in (-1, 0, 1):
            x, y, _ = compose(card, Pose((TAB_ALONG + end * TAB_L / 2, PCB_Y, -CARD_HALF[2]))).pos
            dx, dy = x - gx, y - gy
            if end:
                across = max(across, abs(-math.sin(run) * dx + math.cos(run) * dy))
            else:
                along = abs(math.cos(run) * dx + math.sin(run) * dy)
        return across, along

    def plan(self, world: WorldState, ctx: TurnContext) -> List[Command]:
        # all that the first picture decides: how the card hangs in the hand
        # once it is taken is what the fitting goes by
        return self._take(world) + [Done("the card taken; fitting it needs looking")]
