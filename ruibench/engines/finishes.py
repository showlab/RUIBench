"""Finishes: real surfaces for what RUIBench builds itself (v2, 2026-09-18).

A task names each part's colour.  Since v2 it may name a *finish* instead --
the surface a real thing of that kind has:

    wood:oak  wood:ash  wood:plywood       photographed grain (wood:walnut, wood:cherry,
                                           wood:ebony: the oak's, darkened, reddened,
                                           stained black; wood:stain_<colour>: the
                                           ash's, dyed)
    paint:<colour>                         painted wood (a toy's enamel)
    plastic:<colour>  ceramic:<colour>     glossy, glazed
    fabric:<colour>                        linen weave, dyed
    metal:gold  metal:brass  metal:steel  metal:chrome  metal:nickel  metal:aluminium
    metal:copper  metal:black
    paper:<spec>                           a printed face (``engines/prints.py``)
    glass:<colour>                         clear glass or a clear liquid, tinted
                                           (``glass:clear`` is colourless, a thin
                                           wall; ``glass:solid`` a rod of it)
    flame:<colour>                         a lamp: glows in its colour; one
                                           material per body, so it can be recoloured
    fire:<colour>                          a flame: glows, and is seen through (v8)
    sand:beach  sand:wet                   photographed sand (DS3's sandbox; wet, darker:
                                           WK2's riverbanks)
    ground:grass  ground:pebbles           photographed turf and a pebbled riverbed (WK2)
    ground:asphalt  ground:paving          a road's asphalt, a pavement's concrete slabs and
    ground:concrete                        its kerbstones (DI3's model street, v14)

The Isaac backend draws them as materials (``engines/isaac/finishes.py``);
everything that reads colours -- the mock engine, reports, predicates about
colour -- sees the finish's colour through ``engines.base.color_rgb``.  This
module imports nothing from Isaac.
"""

from __future__ import annotations

from typing import Tuple

RGB = Tuple[float, float, float]

KINDS = ("wood", "paint", "plastic", "ceramic", "fabric", "metal", "paper", "glass", "flame", "fire",
         "sand", "ground")
WOOD_RGB = {"oak": (0.62, 0.45, 0.28), "ash": (0.80, 0.68, 0.52), "plywood": (0.76, 0.62, 0.45),
            "walnut": (0.36, 0.24, 0.16), "cherry": (0.55, 0.30, 0.20), "ebony": (0.17, 0.12, 0.10),
            "stain_red": (0.72, 0.13, 0.10), "stain_yellow": (0.80, 0.60, 0.14),
            "stain_green": (0.24, 0.52, 0.22), "stain_blue": (0.14, 0.30, 0.64)}
SAND_RGB = {"beach": (0.80, 0.70, 0.54), "wet": (0.52, 0.42, 0.30)}
#: ground covers, photographed (v10: WK2's diorama): turf, a pebbled riverbed;
#: (v14: DI3's street) asphalt, a pavement's slabs, kerbstones
GROUND_RGB = {"grass": (0.30, 0.46, 0.18), "pebbles": (0.52, 0.49, 0.44),
              "asphalt": (0.26, 0.23, 0.21), "paving": (0.50, 0.45, 0.40),
              "concrete": (0.46, 0.42, 0.35)}
METAL_RGB = {"gold": (0.83, 0.66, 0.22), "brass": (0.78, 0.62, 0.32), "steel": (0.62, 0.63, 0.64),
             "chrome": (0.80, 0.81, 0.83), "nickel": (0.78, 0.77, 0.74),
             "aluminium": (0.75, 0.76, 0.77), "copper": (0.72, 0.45, 0.3), "black": (0.08, 0.08, 0.09)}


def is_finish(color) -> bool:
    return isinstance(color, str) and color.split(":", 1)[0] in KINDS and ":" in color


def split(finish: str) -> Tuple[str, str]:
    kind, _, value = finish.partition(":")
    return kind, value


def finish_rgb(finish: str) -> RGB:
    """The colour a finish is seen as, for everything that is not a renderer."""
    from .base import COLORS

    kind, value = split(finish)
    if kind == "wood":
        return WOOD_RGB.get(value, WOOD_RGB["oak"])
    if kind == "sand":
        return SAND_RGB.get(value, SAND_RGB["beach"])
    if kind == "ground":
        return GROUND_RGB.get(value, GROUND_RGB["grass"])
    if kind == "metal":
        return METAL_RGB.get(value, METAL_RGB["steel"])
    if kind == "paper":
        return COLORS["white"]
    return COLORS.get(value, COLORS["white"])


def linear(rgb: RGB) -> RGB:
    """An sRGB colour (the palette's, as it should look) in the renderer's
    linear space.  Handed over as is, the palette rendered washed out -- an
    orange xylophone bar next to a yellow one read as two yellows (v2)."""
    return tuple(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb)


__all__ = ["KINDS", "is_finish", "split", "finish_rgb", "linear"]
