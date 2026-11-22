"""Toys built from primitives: one rigid body, several coloured parts.

The v1 tasks are made of things people recognise -- a burger, a toy car, rings
on a peg, a treasure chest -- and most of them are, like the wooden toys they
imitate, a handful of boxes and cylinders.  Building them from parts keeps
their physics simple and exact (convex pieces, no mesh decomposition) and
their look consistent.

A part is a plain tuple, so a body's parts can live in ``BodySpec.attrs`` and
in the scene's structure key:

    (kind, centre, dims, rot, color, collides)

* ``kind``: ``box`` (dims = half extents), ``cylinder`` (radius, half height,
  along the part's z), ``sphere`` (radius), ``capsule`` (radius, half height of
  the straight part, along z), ``asset`` (a catalogue mesh drawn there, dims =
  its half extents; see :func:`visual`), ``lathe:<profile>`` (a surface of
  revolution, drawn only, dims = (radius, radius, half height); see
  :func:`lathe`; ``lathe:slab``, a rounded board, dims = half extents: see
  :func:`slab`), ``print`` (a printed face, drawn only; see :func:`printed`)
* ``centre`` and ``rot``: in the body's frame; ``rot`` is (w, x, y, z)
* ``color``: a name from ``engines.base.COLORS`` or an (r, g, b) tuple
* ``collides``: False for decoration (a painted stripe, an eye); ``"hidden"``
  for a collider nobody sees (see :func:`grip`)

Engines that do not build parts (the mock) treat the body as its bounding box,
which is ``BodySpec.size``; :func:`half_extents` computes it.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Iterable, List, Sequence, Tuple, Union

from ..core.types import IDENTITY_QUAT, Quat, Vec3

Color = Union[str, Tuple[float, float, float]]
Part = Tuple[str, Vec3, Vec3, Quat, Color, bool]


def box(centre: Vec3, half: Vec3, color: Color, rot: Quat = IDENTITY_QUAT,
        collides: bool = True) -> Part:
    return ("box", tuple(centre), tuple(half), tuple(rot), color, collides)


def cylinder(centre: Vec3, radius: float, half_height: float, color: Color,
             rot: Quat = IDENTITY_QUAT, collides: bool = True) -> Part:
    return ("cylinder", tuple(centre), (radius, half_height, 0.0), tuple(rot), color, collides)


def sphere(centre: Vec3, radius: float, color: Color, collides: bool = True) -> Part:
    return ("sphere", tuple(centre), (radius, 0.0, 0.0), IDENTITY_QUAT, color, collides)


def capsule(centre: Vec3, radius: float, half_height: float, color: Color,
            rot: Quat = IDENTITY_QUAT, collides: bool = True) -> Part:
    return ("capsule", tuple(centre), (radius, half_height, 0.0), tuple(rot), color, collides)


def prism(centre: Vec3, half: Vec3, color: Color, rot: Quat = IDENTITY_QUAT,
          collides: bool = True) -> Part:
    """A triangular prism -- a roof block: in its own x-z plane a triangle,
    base ``2 * half[0]`` wide at ``-half[2]`` and apex at ``+half[2]``, drawn
    out ``half[1]`` either way along y."""
    return ("prism", tuple(centre), tuple(half), tuple(rot), color, collides)


def ellipsoid(centre: Vec3, half: Vec3, color: Color, rot: Quat = IDENTITY_QUAT) -> Part:
    """A stretched sphere -- an egg's look -- drawn over ``hidden`` colliders:
    it never collides itself (PhysX has no stretched spheres)."""
    return ("ellipsoid", tuple(centre), tuple(half), tuple(rot), color, False)


def octagon(centre: Vec3, half_flat: float, half_height: float, color: Color,
            rot: Quat = IDENTITY_QUAT, collides=True) -> Part:
    """An eight-sided prism, its flats ``half_flat`` from the middle, two of
    them square to x and two to y: what the fingers close on for a round
    thing drawn over it (a coin, a weight, a tart).  A round collider held by
    its rim slipped out of the fingers (Isaac); two squares a quarter turned
    -- the collider until v8 -- are an eight-pointed star, reaching 41% past
    the flats, which is what the coins looked like."""
    return ("octagon", tuple(centre), (half_flat, half_flat, half_height), tuple(rot), color, collides)


def octagon_points(half_flat: float):
    """The prism's corners round it, in its own plane."""
    r = half_flat / math.cos(math.pi / 8)
    return [(r * math.cos(math.pi / 8 + k * math.pi / 4), r * math.sin(math.pi / 8 + k * math.pi / 4))
            for k in range(8)]


def lathe(centre: Vec3, profile: str, radius: float, half_height: float, color: Color,
          rot: Quat = IDENTITY_QUAT, inner: float = 0.0) -> Part:
    """A surface of revolution, drawn only: :data:`PROFILES`' ``profile``
    turned about the part's z axis, ``radius`` at its widest, ``half_height``
    half its height.  A glass vessel is one -- its wall a single sheet, as
    thin-walled glass is drawn: a wall of boxes (:func:`ring`) came out a
    birdcage, and whatever stood in it black (v8, research/v8_beaker_probe.py)
    -- and so are a bell, a cloche, a flowerpot.  What a hand meets is the
    task's hidden parts."""
    if profile not in PROFILES:
        raise ValueError(f"unknown lathe profile {profile!r}")
    # the second of the dims is the hole's radius for a profile with one (a
    # washer); the others ignore it
    return (f"lathe:{profile}", tuple(centre), (radius, inner or radius, half_height), tuple(rot),
            color, False)


def vessel(centre: Vec3, radius: float, half_height: float, color: Color = "glass:clear",
           profile: str = "beaker") -> Part:
    """A glass vessel open at the top -- its floor, its wall and the lip it
    turns over at the rim."""
    return lathe(centre, profile, radius, half_height, color)


def slab(centre: Vec3, half: Vec3, color: Color, rot: Quat = IDENTITY_QUAT) -> Part:
    """A board -- a stand's base, a tray -- its corners rounded in plan and
    its top edge rounded over (:func:`slab_radii`), drawn only, over the
    task's hidden box.  USD's cube has knife edges: WK1's ash stand drawn as
    one read as fibreboard (v9)."""
    return ("lathe:slab", tuple(centre), tuple(half), tuple(rot), color, False)


def slab_radii(dims: Vec3) -> Tuple[float, float]:
    """(the top edge's rounding, the corners' in plan) of a slab of ``dims``."""
    edge = min(0.005, 0.6 * float(dims[2]))
    return edge, max(edge, min(0.012, 0.25 * min(float(dims[0]), float(dims[1]))))


def _cylinder_profile(r: float, hz: float):
    return ((0.0, -hz), (r, -hz), (r, hz), (0.0, hz))


#: a vessel's lip: how far the glass turns in over the rim, and down inside it
LIP = 0.0012


def _beaker_profile(r: float, hz: float):
    return ((0.0, -hz), (r, -hz), (r, hz), (r - LIP, hz), (r - LIP, hz - 2 * LIP))


def _test_tube_profile(r: float, hz: float, steps: int = 8):
    """A round bottom, a straight wall, the lip."""
    arc = tuple((r * math.sin(a), -hz + r - r * math.cos(a))
                for a in (i * math.pi / 2 / steps for i in range(steps + 1)))
    return arc + ((r, hz), (r - LIP, hz), (r - LIP, hz - 2 * LIP))


def _sphere_profile(r: float, hz: float, steps: int = 16):
    """A sphere (``hz`` = r), or a spheroid."""
    return tuple((r * math.sin(a), -hz * math.cos(a))
                 for a in (i * math.pi / steps for i in range(steps + 1)))


def _capsule_profile(r: float, hz: float, steps: int = 8):
    """A capsule: ``hz`` half its straight part, a hemisphere of ``r`` on each end."""
    ends = [i * math.pi / 2 / steps for i in range(steps + 1)]
    return (tuple((r * math.sin(a), -hz - r * math.cos(a)) for a in ends)
            + tuple((r * math.cos(a), hz + r * math.sin(a)) for a in ends))


def _round_bottom_profile(r: float, hz: float, steps: int = 8):
    """A solid with a round bottom and a flat top: what fills a test tube."""
    return _test_tube_profile(r, hz, steps)[:steps + 1] + ((r, hz), (0.0, hz))


def _arc(r0, z0, r1, z1, steps, outward_first=True):
    """From (r0, z0) to (r1, z1) along a quarter ellipse: leaving level and
    arriving upright (``outward_first``), or the reverse."""
    out = []
    for i in range(steps + 1):
        t = i * math.pi / 2 / steps
        if outward_first:
            out.append((r0 + (r1 - r0) * math.sin(t), z0 + (z1 - z0) * (1 - math.cos(t))))
        else:
            out.append((r0 + (r1 - r0) * (1 - math.cos(t)), z0 + (z1 - z0) * math.sin(t)))
    return out


def _bowl_profile(r: float, hz: float, steps: int = 10):
    """A small bowl, a solid glazed all over: a foot ring, the body curving
    out and up to the rim, a flat rim, the inside down to its floor."""
    h = 2 * hz
    foot_r, foot_h, rim, floor = 0.44 * r, 0.10 * h, 0.05 * r, 0.16 * h
    outer = _arc(foot_r + 0.03 * r, -hz + foot_h, r, hz, steps)
    inside = [((r - rim) * math.cos(i * math.pi / 2 / steps),
               -hz + floor + (h - floor) * (1 - math.sin(i * math.pi / 2 / steps)))
              for i in range(steps + 1)]
    return ((0.0, -hz), (foot_r, -hz), (foot_r + 0.03 * r, -hz + foot_h)) + tuple(outer[1:]) + \
        tuple(inside) + ((0.0, -hz + floor),)


def _plate_profile(r: float, hz: float, steps: int = 8):
    """A plate: a foot ring, the underside rising to the rim, the rim's
    edge, the well sinking to a flat middle."""
    h = 2 * hz
    well_r, rim_w = 0.62 * r, 0.03 * r
    return ((0.0, -hz + 0.12 * h), (0.40 * r, -hz + 0.12 * h), (0.42 * r, -hz), (0.46 * r, -hz),
            (0.50 * r, -hz + 0.25 * h), (r, hz - 0.18 * h), (r, hz - 0.05 * h), (r - rim_w, hz),
            (well_r + 0.06 * r, hz - 0.28 * h), (well_r, -hz + 0.45 * h), (0.0, -hz + 0.45 * h))


def _dome_profile(r: float, hz: float, steps: int = 14):
    """A cloche: a dome over a short upright skirt with a rolled lip, open
    underneath, the knob on top left to a part of its own.  A sheet of
    porcelain 3 mm thick: the outside up to the crown, the inside back down."""
    t = 0.003
    skirt = 0.10 * 2 * hz
    outside = [(r, -hz), (r + 0.002, -hz + 0.002), (r, -hz + 0.004), (r, -hz + skirt)]
    dome_h = 2 * hz - skirt
    outside += [(r * math.cos(i * math.pi / 2 / steps),
                 -hz + skirt + dome_h * math.sin(i * math.pi / 2 / steps)) for i in range(1, steps + 1)]
    inside = [((r - t) * math.cos(i * math.pi / 2 / steps),
               -hz + skirt + (dome_h - t) * math.sin(i * math.pi / 2 / steps)) for i in range(steps, -1, -1)]
    inside += [(r - t, -hz), (r, -hz)]                 # the underside of the lip: closed round
    return tuple(outside) + tuple(inside)


def _stand_profile(r: float, hz: float, steps: int = 8):
    """A cake stand: a round plate with a raised lip on a stem and a foot."""
    h = 2 * hz
    top, plate_t = hz, 0.10 * h
    return ((0.0, -hz), (0.55 * r, -hz), (0.56 * r, -hz + 0.05 * h), (0.30 * r, -hz + 0.14 * h),
            (0.14 * r, -hz + 0.30 * h), (0.12 * r, top - plate_t - 0.12 * h),
            (0.30 * r, top - plate_t - 0.02 * h), (r * 0.98, top - plate_t), (r, top - plate_t * 0.4),
            (r - 0.012, top), (r - 0.016, top - plate_t * 0.35), (0.0, top - plate_t * 0.35))


def _plinth_profile(r: float, hz: float):
    """A round base with a bevelled top edge: a desk bell's."""
    bevel = min(0.3 * 2 * hz, 0.12 * r)
    return ((0.0, -hz), (r, -hz), (r, hz - bevel), (r - bevel, hz), (0.0, hz))


def _bell_profile(r: float, hz: float, steps: int = 16):
    """A desk bell's dome: a flattened half sphere on a short upright rim, a
    solid (what is inside it is never seen)."""
    rim = 0.12 * 2 * hz
    out = [(0.0, -hz), (r, -hz), (r, -hz + rim)]
    out += [(r * math.cos(i * math.pi / 2 / steps),
             -hz + rim + (2 * hz - rim) * math.sin(i * math.pi / 2 / steps)) for i in range(1, steps + 1)]
    return tuple(out)


def _washer_profile(r: float, hz: float, inner: float, steps: int = 6):
    """A disc with a hole through its middle -- a Tower of Hanoi's, a rack's
    rim round a hole: its outer edge rounded over, the hole's edges eased."""
    e = min(0.45 * hz, 0.25 * (r - inner))        # the outer edge's rounding
    c = min(0.25 * hz, 0.2 * (r - inner))         # the hole's ease
    out = [(inner + c, -hz), (r - e, -hz)]
    out += [(r - e + e * math.sin(t), -hz + e - e * math.cos(t))
            for t in (i * math.pi / 2 / steps for i in range(1, steps + 1))]
    out += [(r - e + e * math.cos(t), hz - e + e * math.sin(t))
            for t in (i * math.pi / 2 / steps for i in range(0, steps + 1))]
    out += [(inner + c, hz), (inner, hz - c), (inner, -hz + c), (inner + c, -hz)]
    return tuple(out)


def _dowel_profile(r: float, hz: float, steps: int = 8):
    """A peg: straight, its top rounded over."""
    top = [(r * math.cos(t), hz - r + r * math.sin(t))
           for t in (i * math.pi / 2 / steps for i in range(steps + 1))]
    return ((0.0, -hz), (r, -hz)) + tuple(top)


def _flame_profile(r: float, hz: float, steps: int = 18):
    """A flame: from the burner's mouth (half its width) swelling to its
    widest a third of the way up, then drawn out to a point."""
    out = [(0.0, -hz)]
    for i in range(steps + 1):
        u = i / steps
        if u <= 0.3:
            w = 0.5 + 0.5 * math.sin(u / 0.3 * math.pi / 2)
        else:
            w = math.cos((u - 0.3) / 0.7 * math.pi / 2) ** 0.85
        out.append((r * w, -hz + 2 * hz * u))
    out[-1] = (0.0, hz)
    return tuple(out)


def _pan_profile(r: float, hz: float):
    """A balance pan: a shallow dish, its floor 4 mm thick, its wall flaring
    out to a thin rim."""
    t = min(0.004, hz)
    return ((0.0, -hz), (0.88 * r, -hz), (r, hz), (r - 0.0025, hz), (0.88 * r - 0.002, -hz + t),
            (0.0, -hz + t))


def _chamfer_profile(r: float, hz: float):
    """A disc weight with its edges chamfered at 45 degrees: two stacked show
    a V between them (WK5, v8)."""
    c = min(0.14 * r, 0.3 * hz)
    return ((0.0, -hz), (r - c, -hz), (r, -hz + c), (r, hz - c), (r - c, hz), (0.0, hz))


def _coin_profile(r: float, hz: float):
    """A coin: a raised rim round a sunken field on either face, a low boss
    in the middle of each (the relief's place), a bevelled edge."""
    rim, d, e = 0.13 * r, min(0.24 * hz, 0.0012), min(0.16 * hz, 0.0008)
    f, b = hz - d, 0.7 * d
    face = [(0.0, f + b), (0.35 * r, f + 0.55 * b), (0.55 * r, f), (r - rim, f),
            (r - rim + 0.6 * d, hz), (r - e, hz), (r, hz - e)]
    bottom = [(x, -z) for x, z in face]
    return tuple(bottom) + tuple(reversed(face))


#: a glazed flower pot, upright, as fractions of its rim's radius and its
#: height: its base, the wall under the rolled rim, the rim band's height;
#: the wall 5 mm thick, and the bare clay at its foot this high
POT_BASE, POT_WALL, POT_RIM, POT_T, POT_FOOT = 0.68, 0.90, 0.16, 0.005, 0.004


def _pot_outer(r: float, hz: float, z: float) -> float:
    """The pot's outside radius at height ``z``, below its rim band."""
    rb, rw, top = POT_BASE * r, POT_WALL * r, hz - POT_RIM * 2 * hz
    return rb + (rw - rb) * (z + hz) / (top + hz)


def _flowerpot_profile(r: float, hz: float):
    """A flower pot, upright: the narrow base, the wall flaring out to a
    rolled rim band, over its lip, and the inside back down to the floor --
    a solid wall, glazed all over (DS2, v10: the scanned clay pots made a
    dull brown scene)."""
    t, e = POT_T, 0.002
    rb, rw, band = POT_BASE * r, POT_WALL * r, POT_RIM * 2 * hz
    return ((0.0, -hz), (rb - e, -hz), (rb, -hz + e), (rw, hz - band), (r - e, hz - band),
            (r, hz - band + e), (r, hz - e), (r - e, hz), (r - t + e, hz), (r - t, hz - e),
            (r - t, hz - band), (rw - t, hz - band - e), (rb - t, -hz + t), (0.0, -hz + t))


def _pot_foot_profile(r: float, hz: float):
    """The bare clay at a glazed pot's foot: a band round its base, a hair
    proud of the glaze, its top inside the pot's floor."""
    z1, a = -hz + POT_FOOT, 0.0003
    return ((0.0, -hz - a), (POT_BASE * r - 0.002, -hz - a), (POT_BASE * r + a, -hz + 0.002),
            (_pot_outer(r, hz, z1) + a, z1), (0.0, z1))


def _nose_profile(r: float, hz: float, steps: int = 16):
    """A rocket's nose cone: a tangent ogive from its base (``r``) to a
    point -- a cone's sides bowed out, meeting the body tube without a kink."""
    length = 2 * hz
    rho = (r * r + length * length) / (2 * r)
    out = [(0.0, -hz), (r, -hz)]
    for i in range(1, steps):
        d = length * (1 - i / steps)                     # from the tip
        out.append((math.sqrt(max(rho * rho - (length - d) ** 2, 0.0)) + r - rho, hz - d))
    return tuple(out) + ((0.0, hz),)


def _nozzle_profile(r: float, hz: float, steps: int = 8):
    """A rocket engine's bell: its throat (half its mouth) on top, flaring
    to its mouth at the bottom."""
    out = [(0.0, -hz), (r, -hz)]
    out += [(r * (0.5 + 0.5 * (1 - i / steps) ** 2), -hz + 2 * hz * i / steps)
            for i in range(1, steps + 1)]
    return tuple(out) + ((0.0, hz),)


def _mushroom_profile(r: float, hz: float, steps: int = 12):
    """A mushroom-head push button's cap: a flat underside, a rolled rim and a
    low dome over it -- an emergency stop's."""
    h = 2 * hz
    out = [(0.0, -hz), (0.84 * r, -hz), (0.95 * r, -hz + 0.07 * h), (r, -hz + 0.22 * h)]
    dome = 0.78 * h
    out += [(r * math.cos(t), -hz + 0.22 * h + dome * math.sin(t))
            for t in (i * math.pi / 2 / steps for i in range(1, steps))]
    return tuple(out) + ((0.0, hz),)


def _bulb_profile(r: float, hz: float, steps: int = 18):
    """A light bulb's glass, cap down: from the neck (0.45 of the globe's
    width, where the cap takes it) drawn out into the globe, round over the
    top.  Open at the neck -- a sheet of glass, as a vessel is."""
    neck, zc = 0.45 * r, hz - r                      # the neck's radius, the globe's middle
    low = math.radians(128.0)                        # where the globe gives way to the neck
    p0, p2 = (neck, -hz), (r * math.sin(low), zc + r * math.cos(low))
    ctrl = (neck, p2[1] - 0.25 * (p2[1] + hz))
    out = []
    for i in range(steps // 2 + 1):
        t = i / (steps // 2)
        out.append(tuple((1 - t) ** 2 * a + 2 * t * (1 - t) * c + t * t * b
                         for a, c, b in zip(p0, ctrl, p2)))
    out += [(r * math.sin(a), zc + r * math.cos(a))
            for a in (low * (1 - i / steps) for i in range(1, steps))]
    return tuple(out) + ((0.0, hz),)


def _funnel_profile(r: float, hz: float, inner: float):
    """A flared collar -- a lamp holder's mouth: a wall 2 mm thick opening
    from the bore (``inner``) at the bottom to ``r`` at the top."""
    t = 0.002
    return ((inner, -hz), (r - t, hz), (r, hz), (inner + t, -hz), (inner, -hz))


#: a spinning top, as fractions of its height from the tip: where the body is
#: widest, how high the shoulder rises over to the peg, and the peg's
#: radius as a fraction of the body's
TOP_WIDEST, TOP_SHOULDER, TOP_PEG = 0.58, 0.12, 0.2
#: its painted bands, each a slice of the one outline (from, to): the steel
#: tip, a red body, a yellow band at its widest, a blue shoulder, the peg
TOP_BANDS = {"top_tip": (0.0, 0.05), "top_body": (0.05, 0.50), "top_band": (0.50, 0.62),
             "top_shoulder": (0.62, 0.705), "top_peg": (0.705, 1.0)}


def _top_outline(r: float, hz: float, steps: int = 24):
    """A turned spinning top from its tip up: the flank swelling out to the
    widest, the shoulder turned over in a quarter ellipse to the peg, the
    peg straight up and rounded over at its end."""
    h, peg = 2 * hz, TOP_PEG * r
    out = [(r * (TOP_WIDEST * i / steps / TOP_WIDEST) ** 0.75, -hz + h * TOP_WIDEST * i / steps)
           for i in range(steps + 1)]
    turn = math.acos(TOP_PEG)
    out += [(r * math.cos(turn * i / steps), -hz + h * (TOP_WIDEST + TOP_SHOULDER * math.sin(turn * i / steps)))
            for i in range(1, steps + 1)]
    out += [(peg * math.cos(a), hz - peg + peg * math.sin(a))
            for a in (math.pi / 2 * i / 8 for i in range(9))]
    return out


def _top_slice(u0: float, u1: float):
    """The part of the top's outline between two heights (fractions of its
    height from the tip), closed on the axis at both ends: a painted band."""

    def profile(r: float, hz: float):
        z0, z1 = -hz + 2 * hz * u0, -hz + 2 * hz * u1
        outline = _top_outline(r, hz)

        def radius(z):
            for (ra, za), (rb, zb) in zip(outline[:-1], outline[1:]):
                if za <= z <= zb and zb > za:
                    return ra + (rb - ra) * (z - za) / (zb - za)
            return outline[-1][0] if z >= outline[-1][1] else outline[0][0]

        inside = [(rr, z) for rr, z in outline if z0 < z < z1]
        return ((0.0, z0), (radius(z0), z0)) + tuple(inside) + ((radius(z1), z1), (0.0, z1))

    return profile


#: surfaces of revolution: a profile's name -> its points (r, z) for a part
#: of radius r and half height hz, traced with the thing's inside on the left
#: -- its surface faces right.  A profile that starts and ends on the axis is
#: a closed solid; one that does not, a sheet (drawn from both sides)
PROFILES = {"cylinder": _cylinder_profile, "sphere": _sphere_profile, "capsule": _capsule_profile,
            "beaker": _beaker_profile, "test_tube": _test_tube_profile,
            "round_bottom": _round_bottom_profile, "bowl": _bowl_profile, "plate": _plate_profile,
            "dome": _dome_profile, "stand": _stand_profile, "chamfer": _chamfer_profile,
            "coin": _coin_profile, "pan": _pan_profile, "plinth": _plinth_profile,
            "bell": _bell_profile, "flame": _flame_profile, "washer": _washer_profile,
            "dowel": _dowel_profile, "flowerpot": _flowerpot_profile, "pot_foot": _pot_foot_profile,
            "nose": _nose_profile, "nozzle": _nozzle_profile, "mushroom": _mushroom_profile,
            "bulb": _bulb_profile, "funnel": _funnel_profile,
            **{name: _top_slice(*band) for name, band in TOP_BANDS.items()}}
#: the profiles that take a hole's radius as well
HOLED = ("washer", "funnel")
#: a turn sharper than this between two stretches of a profile is an edge;
#: gentler, the surface is drawn smooth across it
CREASE = math.radians(40.0)
#: sides round the axis: this many per metre of the widest rim, within these
#: bounds.  RTX draws USD's own cylinder as a prism of a dozen sides: a
#: beaker's water, a coin, a bell's rim came out cut from wood (v8)
SIDES_PER_M, SIDES = 1500.0, (24, 72)


def lathe_profile(kind: str, dims: Vec3):
    """The (r, z) points of a ``lathe:<profile>`` part of ``dims``."""
    name = kind.split(":", 1)[1]
    if name in HOLED:
        return tuple(PROFILES[name](float(dims[0]), float(dims[2]), float(dims[1])))
    return tuple(PROFILES[name](float(dims[0]), float(dims[2])))


def is_sheet(profile) -> bool:
    """Open at an end away from the axis, and not closed round on itself."""
    if tuple(profile[0]) == tuple(profile[-1]):
        return False
    return abs(profile[0][0]) > 1e-9 or abs(profile[-1][0]) > 1e-9


def lathe_sides(radius: float) -> int:
    return int(min(max(round(2 * math.pi * radius * SIDES_PER_M), SIDES[0]), SIDES[1]))


def _stretches(profile):
    """The profile's stretches, and each one's normals at its two ends: its
    own, or where the turn to the next is gentler than :data:`CREASE`, the
    two averaged -- smooth across it."""

    def unit(x, y):
        n = math.hypot(x, y)
        return (x / n, y / n)

    stretches = [(a, b) for a, b in zip(profile[:-1], profile[1:]) if a != b]
    normals = [unit(b[1] - a[1], -(b[0] - a[0])) for a, b in stretches]

    def smooth(m, n):
        return math.acos(max(-1.0, min(1.0, m[0] * n[0] + m[1] * n[1]))) < CREASE

    ends = []
    for i, n in enumerate(normals):
        start = end = n
        if i > 0 and smooth(normals[i - 1], n):
            start = unit(normals[i - 1][0] + n[0], normals[i - 1][1] + n[1])
        if i + 1 < len(normals) and smooth(n, normals[i + 1]):
            end = unit(n[0] + normals[i + 1][0], n[1] + normals[i + 1][1])
        ends.append((start, end))
    return stretches, ends


def revolve(profile, sides: int):
    """The profile turned full circle, as a mesh: (points, normals, face vertex
    counts, face vertex indices), a normal per vertex -- smooth round the axis,
    and along the profile but at its creases."""
    turn = [(math.cos(2 * math.pi * k / sides), math.sin(2 * math.pi * k / sides))
            for k in range(sides)]
    points, vertex_normals, counts, indices = [], [], [], []
    for ((a, b), (na, nb)) in zip(*_stretches(profile)):
        base = len(points)
        for (r, z), (nr, nz) in ((a, na), (b, nb)):
            for c, s in turn:
                points.append((r * c, r * s, z))
                vertex_normals.append((nr * c, nr * s, nz))
        for k in range(sides):
            j = (k + 1) % sides
            quad = [base + k, base + j, base + sides + j, base + sides + k]
            if abs(a[0]) < 1e-9:              # from the axis: a fan of triangles
                quad = [quad[0], quad[2], quad[3]]
            elif abs(b[0]) < 1e-9:            # to the axis
                quad = quad[:3]
            counts.append(len(quad))
            indices.extend(quad)
    return points, vertex_normals, counts, indices


def slab_mesh(dims: Vec3, steps: int = 6):
    """A ``slab`` part as a mesh, as :func:`revolve` makes one: its edge's
    profile -- up the side and over the rounded top edge -- carried round a
    rectangle with rounded corners instead of a circle, and a flat cap on
    top and underneath."""
    hx, hy, hz = (float(v) for v in dims)
    edge, corner = slab_radii(dims)
    cx, cy = hx - corner, hy - corner
    # r: out from the rectangle the corners' centres make
    profile = [(corner, -hz), (corner, hz - edge)] + [
        (corner - edge + edge * math.cos(a), hz - edge + edge * math.sin(a))
        for a in (math.pi / 2 * k / steps for k in range(1, steps + 1))]
    path = []                                 # (corner's centre, outward), counter-clockwise
    for (sx, sy), start in (((1, 1), 0.0), ((-1, 1), 0.5), ((-1, -1), 1.0), ((1, -1), 1.5)):
        for k in range(steps + 1):
            a = math.pi * (start + 0.5 * k / steps)
            path.append(((sx * cx, sy * cy), (math.cos(a), math.sin(a))))
    n = len(path)
    points, normals, counts, indices = [], [], [], []
    for ((a, b), (na, nb)) in zip(*_stretches(profile)):
        base = len(points)
        for (r, z), (nr, nz) in ((a, na), (b, nb)):
            for (px, py), (ux, uy) in path:
                points.append((px + r * ux, py + r * uy, z))
                normals.append((nr * ux, nr * uy, nz))
        for k in range(n):
            j = (k + 1) % n
            counts.append(4)
            indices.extend([base + k, base + j, base + n + j, base + n + k])
    for (r, z), up in ((profile[-1], 1.0), (profile[0], -1.0)):
        middle = len(points)
        points.append((0.0, 0.0, z))
        points.extend((px + r * ux, py + r * uy, z) for (px, py), (ux, uy) in path)
        normals.extend([(0.0, 0.0, up)] * (n + 1))
        for k in range(n):
            j = (k + 1) % n
            counts.append(3)
            indices.extend([middle, middle + 1 + k, middle + 1 + j] if up > 0
                           else [middle, middle + 1 + j, middle + 1 + k])
    return points, normals, counts, indices


def printed(centre: Vec3, width: float, spec: str, rot: Quat = IDENTITY_QUAT,
            wrap: float = 0.0) -> Part:
    """A printed face (``engines/prints.py``), drawn only: ``spec``'s image
    ``width`` wide on a rectangle facing the part's +z, the image's top
    towards +x and its right towards -y -- read, lying flat, by someone at
    -x, as :func:`text` is.  Its height follows the design's aspect.
    ``wrap``: the radius of the round thing it is stuck on, its axis along
    the image's height, behind the face -- a label round a beaker; negative,
    in front of it -- the label's back, seen from inside the glass."""
    from .prints import aspect

    half = (width / aspect(spec) / 2, width / 2, float(wrap))
    return ("print", tuple(centre), half, tuple(rot), f"paper:{spec}", False)


def print_sheet(dims: Vec3, columns: int = 16):
    """A print's face as a mesh: (points, normals, texture coordinates, face
    vertex counts, face vertex indices) -- flat, or wrapped round its radius."""
    hx, hy, wrap = (float(v) for v in dims)
    cols = columns if wrap else 1
    points, normals, st = [], [], []
    for i in range(cols + 1):
        s = -hy + 2 * hy * i / cols                  # along the face, from its right (-y)
        if wrap:
            phi = s / wrap
            y, z, n = wrap * math.sin(phi), wrap * math.cos(phi) - wrap, (0.0, math.sin(phi), math.cos(phi))
        else:
            y, z, n = s, 0.0, (0.0, 0.0, 1.0)
        u = 1.0 - i / cols
        for x, t in ((-hx, 0.0), (hx, 1.0)):
            points.append((x, y, z))
            normals.append(n)
            st.append((u, t))
    counts, indices = [], []
    for i in range(cols):
        a, b = 2 * i, 2 * (i + 1)
        counts.append(4)
        indices.extend([a, a + 1, b + 1, b])            # counter-clockwise, seen from the front
    return points, normals, st, counts, indices


def visual(asset: str, centre: Vec3, half: Vec3, rot: Quat = IDENTITY_QUAT,
           scale: float = 1.0) -> Part:
    """A real thing's look on a toy built from exact parts (v2).

    A scanned flower pot, a chest or a cabinet is what the cameras should see;
    what the physics should feel is the exact, convex shape its task needs --
    a pot's walls around an egg, a drawer that slides.  So the colliders are
    ``hidden`` parts and the scan is drawn over them: ``asset`` names a
    catalogue entry (``engines/isaac/assets.yaml``), ``half`` its half extents
    after ``scale`` (for everything that treats a part as a box)."""
    return ("asset", tuple(centre), tuple(half), tuple(rot), f"{asset}@{scale}", False)


def grip(centre: Vec3, half: Vec3, rot: Quat = IDENTITY_QUAT) -> Part:
    """An invisible box collider, for a round handle drawn by a decoration
    part.  Flat pads on a round handle touch it along a line, and on Isaac a
    rolling pin held by one handle turned about that line until it fell out;
    on a box's face they hold it (research/pin_grip_probe.py)."""
    return ("box", tuple(centre), tuple(half), tuple(rot), "grey", "hidden")


def turn_z(angle: float) -> Quat:
    return (math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2))


def turn_x(angle: float) -> Quat:
    return (math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0)


def turn_y(angle: float) -> Quat:
    return (math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0)


def ring(centre: Vec3, outer: float, inner: float, half_height: float, color: Color,
         segments: int = 12) -> List[Part]:
    """A flat ring (a washer) as ``segments`` boxes around the z axis."""
    mid = (outer + inner) / 2
    half_w = (outer - inner) / 2
    half_len = mid * math.tan(math.pi / segments) + half_w * 0.2
    out = []
    for k in range(segments):
        a = 2 * math.pi * k / segments
        c = (centre[0] + mid * math.cos(a), centre[1] + mid * math.sin(a), centre[2])
        out.append(box(c, (half_w, half_len, half_height), color, turn_z(a)))
    return out


def open_ring(centre: Vec3, outer: float, inner: float, half_height: float, color: Color,
              segments: int = 24, open_share: float = 0.28) -> List[Part]:
    """A ring with the segments nearest the fixed camera left out.

    The fixed camera stands beyond the far edge of the table and looks back,
    so it sees a body's +x faces: leaving those segments out makes a vessel
    that shows what is inside it.  The wall it draws is only the drawing --
    what stops a hand is the hidden shell the task builds separately.  Real
    transparency is not available (this renderer draws a material with
    opacity as nothing at all; see ``engines/isaac/finishes``).
    """
    keep = []
    for part in ring(centre, outer, inner, half_height, color, segments):
        dx = part[1][0] - centre[0]
        dy = part[1][1] - centre[1]
        angle = math.atan2(dy, dx)
        if abs(angle) > math.pi * open_share:      # 0 is straight at the camera
            keep.append(part)
    return keep


def text(message: str, centre: Vec3, facing: str, height: float, color: Color,
         stroke: float = 0.0, depth: float = 0.0006) -> List[Part]:
    """Lettering painted on a face of a toy: thin decoration boxes, readable
    from outside the face.  ``facing`` is the face's outward normal in the
    body's frame: ``+x``, ``-x``, ``+y``, ``-y``, or ``+z`` for a card lying
    flat on the bench, read by someone standing at -x -- which is where the
    arms, and the head camera between them, look from.  Flat is how a label
    is worth drawing at all under that camera: it looks down 60 degrees, so
    lettering on an upright face loses half its height to foreshortening on
    top of being small (DS6's stand labels came out 3 px tall, 9/21)."""
    from .base import text_strokes

    stroke = stroke or max(0.0012, height / 9.0)
    if facing == "+z":
        out = []
        for (r0, u0), (r1, u1) in text_strokes(message, height):
            mr, mu = (r0 + r1) / 2, (u0 + u1) / 2
            length = math.hypot(r1 - r0, u1 - u0) + stroke
            angle = math.atan2(u1 - u0, r1 - r0)
            # the reader's right is -y, the tops of the letters point away, +x
            out.append(box((centre[0] + mu, centre[1] - mr, centre[2]),
                           (length / 2, stroke / 2, depth), color,
                           turn_z(angle - math.pi / 2), collides=False))
        return out
    # (right, up) on the face, as body-frame directions, for a reader outside it
    right = {"+x": (0.0, 1.0), "-x": (0.0, -1.0), "+y": (-1.0, 0.0), "-y": (1.0, 0.0)}[facing]
    out = []
    for (r0, u0), (r1, u1) in text_strokes(message, height):
        mr, mu = (r0 + r1) / 2, (u0 + u1) / 2
        length = math.hypot(r1 - r0, u1 - u0) + stroke
        angle = math.atan2(u1 - u0, r1 - r0)            # within the face, from "right"
        c = (centre[0] + right[0] * mr, centre[1] + right[1] * mr, centre[2] + mu)
        if facing in ("+x", "-x"):
            # the face spans y (right) and z (up): turn about x
            sign = 1.0 if facing == "+x" else -1.0
            rot = turn_x(sign * angle)
            half = (depth, length / 2, stroke / 2)
        else:
            sign = 1.0 if facing == "-y" else -1.0
            rot = turn_y(-sign * angle)
            half = (length / 2, depth, stroke / 2)
        out.append(box(c, half, color, rot, collides=False))
    return out


def box_pieces(part: Part, layers: int = 4) -> List[Part]:
    """A part as boxes that cover it: itself, or for a prism a stack of
    slabs, each as wide as the triangle at its bottom -- a hand passing beside
    a roof's ridge is not in the way of the roof.  A box leaning off the
    vertical is a stack of slabs along its lean: as one, its bounds filled the
    wedge behind it -- CL3's card, leaning back 35 degrees on its easel, stood
    in the way of the wrist camera 1 cm clear of it (v17)."""
    import math

    from ..core.geometry import rotate

    kind, centre, dims, rot, color, collides = part
    if kind == "box":
        axes = [rotate(rot, e) for e in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))]
        tilted = [i for i, a in enumerate(axes) if 0.1 < abs(a[2]) < 0.995]
        if not tilted:
            return [part]
        i = max(tilted, key=lambda k: dims[k] * math.hypot(axes[k][0], axes[k][1]))
        out = []
        for j in range(layers):
            d = -dims[i] + (j + 0.5) * 2 * dims[i] / layers
            half = tuple(dims[k] / layers if k == i else dims[k] for k in range(3))
            out.append(("box", tuple(c + v * d for c, v in zip(centre, axes[i])), half, rot, color,
                        collides))
        return out
    if kind != "prism":
        return [part]
    hx, hy, hz = dims
    out = []
    for j in range(layers):
        dz = -hz + (j + 0.5) * 2 * hz / layers
        off = rotate(rot, (0.0, 0.0, dz))
        out.append(("box", tuple(c + o for c, o in zip(centre, off)),
                    (hx * (1 - j / layers), hy, hz / layers), rot, color, collides))
    return out


def part_corners(part: Part) -> Iterable[Vec3]:
    """The eight corners of a part's own bounding box, in the body's frame."""
    return _corners(part)


def _corners(part: Part) -> Iterable[Vec3]:
    from ..core.geometry import rotate

    kind, centre, dims, rot, _, _ = part
    if kind == "lathe:slab":
        hx, hy, hz = dims
    elif kind.startswith("lathe:"):
        hx = hy = dims[0]
        hz = dims[2]
    elif kind == "prism":
        # its own corners, not its box's: laid on its side (a sorter's triangle
        # block) its shadow is a triangle, and the box's would not pass a
        # triangular hole.  Standing as a roof the shadow is the same either way
        hx, hy, hz = dims
        for x, z in ((-hx, -hz), (hx, -hz), (0.0, hz)):
            for sy in (-1, 1):
                d = rotate(rot, (x, sy * hy, z))
                yield (centre[0] + d[0], centre[1] + d[1], centre[2] + d[2])
        return
    elif kind in ("box", "asset", "ellipsoid"):
        hx, hy, hz = dims
    elif kind == "print":
        hx, hy, hz = dims[0], dims[1], 0.0005
    elif kind == "octagon":
        # its own corners, not its box's: the box's reach 41% further on the diagonals
        for x, y in octagon_points(dims[0]):
            for sz in (-1, 1):
                d = rotate(rot, (x, y, sz * dims[2]))
                yield (centre[0] + d[0], centre[1] + d[1], centre[2] + d[2])
        return
    elif kind == "sphere":
        hx = hy = hz = dims[0]
    elif kind == "cylinder":
        hx = hy = dims[0]
        hz = dims[1]
    else:                                               # capsule
        hx = hy = dims[0]
        hz = dims[1] + dims[0]
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                d = rotate(rot, (sx * hx, sy * hy, sz * hz))
                yield (centre[0] + d[0], centre[1] + d[1], centre[2] + d[2])


def half_extents(parts: Sequence[Part], solid_only: bool = True) -> Vec3:
    """The half extents of the parts' bounding box about the body's origin
    (symmetric: the largest reach along each axis).  Decoration is left out
    unless ``solid_only`` is False: it does not change where the toy rests."""
    reach = [0.0, 0.0, 0.0]
    for part in parts:
        if solid_only and not part[5]:
            continue
        for p in _corners(part):
            for i in range(3):
                reach[i] = max(reach[i], abs(p[i]))
    return tuple(round(v, 5) for v in reach)


def offset(parts: Sequence[Part], by: Vec3) -> List[Part]:
    """The same parts moved by ``by`` in the body's frame."""
    return [(k, (c[0] + by[0], c[1] + by[1], c[2] + by[2]), d, r, col, hit)
            for k, c, d, r, col, hit in parts]


__all__ = ["Part", "box", "cylinder", "sphere", "capsule", "prism", "ellipsoid", "lathe", "vessel", "slab",
           "printed", "print_sheet", "octagon", "octagon_points",
           "PROFILES", "lathe_profile", "lathe_sides", "revolve", "slab_mesh", "slab_radii", "is_sheet",
           "visual", "ring", "open_ring", "text", "turn_x", "turn_y",
           "turn_z", "half_extents", "offset", "part_corners"]


# -- shadows on the table -----------------------------------------------------------
Polygon = Tuple[Tuple[float, float], ...]


def convex_hull(points: Iterable[Tuple[float, float]]) -> Polygon:
    """The convex hull, counter-clockwise (Andrew's monotone chain)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return tuple(pts)

    def turn(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    upper: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], p) <= 1e-15:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], p) <= 1e-15:
            upper.pop()
        upper.append(p)
    return tuple(lower[:-1] + upper[:-1])


def overlap(a: Polygon, b: Polygon) -> float:
    """How deep two convex polygons overlap: the least overlap of their shadows
    on the separating-axis candidates (zero or less: they are apart)."""
    best = math.inf
    for poly in (a, b):
        for i in range(len(poly)):
            (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
            ax, ay = y1 - y0, x0 - x1
            norm = math.hypot(ax, ay)
            if norm < 1e-12:
                continue
            ax, ay = ax / norm, ay / norm
            pa = [ax * x + ay * y for x, y in a]
            pb = [ax * x + ay * y for x, y in b]
            best = min(best, min(max(pa), max(pb)) - max(min(pa), min(pb)))
            if best <= 0:
                return best
    return best


@lru_cache(maxsize=4096)
def footprints(parts: Tuple[Part, ...], pos: Vec3, quat: Quat):
    """Each solid part's shadow on the table -- the hull of its corners in world
    x and y -- with its bottom and top, for a body at (``pos``, ``quat``).  A
    round part casts its bounding square."""
    from ..core.geometry import rotate

    out = []
    for part in parts:
        if not part[5]:
            continue
        pts = [rotate(quat, c) for c in _corners(part)]
        pts = [(p[0] + pos[0], p[1] + pos[1], p[2] + pos[2]) for p in pts]
        out.append((convex_hull((p[0], p[1]) for p in pts),
                    min(p[2] for p in pts), max(p[2] for p in pts)))
    return tuple(out)


def body_parts(shape: str, size: Vec3, attrs) -> Tuple[Part, ...]:
    """A body as parts: its own, or one part its size."""
    if shape == "compound" and attrs.get("parts"):
        return tuple(attrs["parts"])
    if shape in ("cylinder", "sphere"):
        return (cylinder((0.0, 0.0, 0.0), size[0], size[2], "grey"),)
    return (box((0.0, 0.0, 0.0), size, "grey"),)
