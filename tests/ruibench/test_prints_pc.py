"""CR4's look: the printed faces of its computer parts, and where they lie.

The look is drawn only -- every collider of the task is what it was before
the prints came (v1.12) -- so what is tested here is that each print is drawn
to the face it goes on, lies on it, and that nothing drawn stands where the
card has to go.
"""

import math
import subprocess
import sys

import pytest

from ruibench.core.geometry import rotate
from ruibench.engines import parts as P
from ruibench.engines import prints, prints_pc
from ruibench.tasks.impl import cr4_graphics_card as CR4

SPECS = ("mainboard:", "gpu_back:", "gpu_shroud:", "gpu_fins:", "dimm:", "fins:", "atx:", "carton:top",
         "carton:side", "carton_end:label", "carton_end:plain", "foam_pad:")


def _bodies(level="easy", seed=0):
    return {b.name: b for b in CR4.build_scene(level, seed).bodies}


def _prints(body):
    return [p for p in body.attrs["parts"] if p[0] == "print"]


def test_pc_prints_are_named_with_the_others_and_drawn_to_their_sizes():
    from PIL import Image

    assert set(prints_pc.DESIGNS) == set(prints_pc.SIZES) == {s.split(":")[0] for s in SPECS}
    for spec in SPECS:
        design = spec.split(":")[0]
        assert prints.DESIGNS[design] is prints_pc.DESIGNS[design]
        w, h = prints.SIZES[design]
        img = Image.open(prints.image(spec))
        assert img.mode == "RGB" and img.size[0] / img.size[1] == pytest.approx(w / h, abs=0.01), spec
        assert min(img.size) >= 64 and img.size[0] / w >= 5.9                    # six pixels a millimetre at least


def test_pc_prints_are_the_same_every_time_they_are_drawn():
    for design, arg in (("gpu_shroud", ""), ("carton_end", "label"), ("dimm", "")):
        a, b = prints_pc.DESIGNS[design](arg), prints_pc.DESIGNS[design](arg)
        assert a.tobytes() == b.tobytes(), design


def test_pc_prints_show_what_they_are():
    from PIL import Image

    def mean(spec, box=None):
        img = Image.open(prints.image(spec))
        if box:
            w, h = img.size
            img = img.crop((int(box[0] * w), int(box[1] * h), int(box[2] * w), int(box[3] * h)))
        px = list(img.resize((16, 16)).getdata())
        return tuple(sum(c[k] for c in px) / len(px) for k in range(3))

    r, g, b = mean("mainboard:")
    assert g > 1.5 * r and g > 1.3 * b and g < 130                               # a dark green board
    assert max(mean("gpu_back:")) < 70 and max(mean("gpu_shroud:", (0.0, 0.0, 0.2, 0.2))) < 70      # black
    r, g, b = mean("carton:side")
    assert r > g > b and 120 < r < 215                                           # kraft brown
    r, g, b = mean("foam_pad:")
    assert r > 215 and r > b > g > 140                                           # pale pink, not magenta
    # the fan's hub label is the bright thing in the middle of its well
    hub = mean("gpu_shroud:", (0.42, 0.44, 0.46, 0.56))
    assert min(hub) > 110


def test_pc_prints_bring_neither_numpy_nor_pil_into_the_core():
    code = ("import sys; import ruibench.engines.prints as p; "
            "assert 'mainboard' in p.DESIGNS; "
            "sys.exit(1 if ('numpy' in sys.modules or 'PIL' in sys.modules) else 0)")
    assert subprocess.run([sys.executable, "-c", code]).returncode == 0


def test_cr4_the_looks_names_leave_the_references_alone():
    """The look's constants were added to a file full of the reference's: the
    first name given to a print's stand-off was ``LIFT``, the reference's own
    for how high it lifts the card -- which then carried it 0.2 mm over the
    pad instead of 10 cm (and still got it in: only a run side by side with
    the old look showed it, Isaac, 10/5)."""
    assert CR4.PROUD < 0.001
    assert (CR4.GRIP_IN, CR4.COME_FROM, CR4.LIFT, CR4.CLEAR_OF_BOX, CR4.HOVER) == (0.013, 0.07, 0.10, 0.07, 0.05)


# -- where they lie -------------------------------------------------------------------
def _faces(body):
    """(normal, the face's two half extents, its middle) of every box part's six faces."""
    out = []
    for kind, centre, half, rot, _, _ in body.attrs["parts"]:
        if kind != "box" or tuple(rot) != (1.0, 0.0, 0.0, 0.0):
            continue
        for axis in range(3):
            for sign in (1, -1):
                normal = tuple(sign if k == axis else 0 for k in range(3))
                middle = tuple(c + sign * half[k] if k == axis else c for k, c in enumerate(centre))
                rest = sorted(half[k] for k in range(3) if k != axis)
                out.append((normal, rest, middle))
    return out


@pytest.mark.parametrize("level", ("easy", "hard"))
def test_cr4_every_flat_print_lies_on_a_face_of_its_own_size(level):
    seen = 0
    for body in _bodies(level).values():
        faces = _faces(body)
        for _, centre, (hx, hy, wrap), rot, spec, collides in _prints(body):
            assert collides is False and spec.startswith("paper:")
            if wrap:
                continue
            normal = tuple(round(v) for v in rotate(rot, (0.0, 0.0, 1.0)))
            mine = sorted((hx, hy))
            on = [f for f in faces if f[0] == normal
                  and all(abs(a - b) < 0.0006 for a, b in zip(f[1], mine))
                  and math.dist(f[2], centre) < 0.0006]
            assert on, (body.name, spec, centre)
            seen += 1
    assert seen == 13                 # 3 on the card, 6 on the carton and its pad, 4 on the board


def test_cr4_the_fins_go_once_round_the_heatsink():
    board = _bodies()[CR4.BOARD]
    [(_, centre, (hx, hy, wrap), rot, spec, _)] = [p for p in _prints(board) if p[2][2]]
    assert spec == "paper:fins:" and 2 * hy == pytest.approx(2 * math.pi * wrap)
    assert 2 * hx == pytest.approx(CR4.SINK_H, abs=1e-4) and wrap < CR4.COOLER_R
    # the band's middle is on the heatsink's side, its axis the cooler's
    z0 = -(CR4.BOARD_TOP + CR4.SLOT_H) / 2
    assert centre == pytest.approx((CR4.COOLER_AT[0] + wrap, CR4.COOLER_AT[1], z0 + CR4.BOARD_TOP + CR4.SINK_H / 2))


@pytest.mark.parametrize("level", ("easy", "hard"))
def test_cr4_the_look_is_only_drawn(level):
    bodies = _bodies(level)
    solid = {name: [p for p in b.attrs["parts"] if p[5] is not False] for name, b in bodies.items()}
    # what a hand, the card and the slot meet is what it was: the card's slab and its tab; the carton and
    # its pad; the board, the slot's seven pieces, the cooler, two sockets, a module, the ports, the power
    # connector; the seat
    assert {name: len(parts) for name, parts in solid.items()} == {CR4.CARD: 2, CR4.BOX: 2, CR4.BOARD: 14,
                                                                  CR4.SEAT: 1}
    assert all(kind in ("box", "cylinder") for parts in solid.values() for kind, *_ in parts)
    # the cooler's fan: its blades, pitched, inside the cylinder that is the cooler
    z0 = -(CR4.BOARD_TOP + CR4.SLOT_H) / 2
    up = z0 + CR4.BOARD_TOP
    cx, cy = CR4.COOLER_AT
    blades = [p for p in bodies[CR4.BOARD].attrs["parts"]
              if p[0] == "box" and p[5] is False and p[4] == "plastic:dark"]
    assert len(blades) == CR4.BLADES
    for blade in blades:
        for x, y, z in P.part_corners(blade):
            assert math.hypot(x - cx, y - cy) < CR4.COOLER_R and up + CR4.SINK_H < z < up + CR4.COOLER_H


def test_cr4_nothing_drawn_stands_where_a_seated_card_is():
    """Beside the slot, for the card's whole length, nothing that is only drawn
    is higher than the slot: the card's body comes down to the slot's top."""
    board = _bodies()[CR4.BOARD]
    z0 = -(CR4.BOARD_TOP + CR4.SLOT_H) / 2
    slot_top = z0 + CR4.BOARD_TOP + CR4.SLOT_H
    sx = CR4.SLOT_AT[0]
    for part in board.attrs["parts"]:
        if part[5] is not False or part[0] == "print":
            continue
        corners = list(P.part_corners(part))
        if max(z for _, _, z in corners) <= slot_top + 1e-9:
            continue
        xs = [x for x, _, _ in corners]
        assert max(xs) < sx - CR4.CARD_T - 0.003 or min(xs) > sx + CR4.CARD_T + 0.003, part
