"""CR4: the card the other way round.  The reference fits it with one face toward the cooler side;
the scorer asks for the gold edge in the groove, whichever way the card faces (``_twin``, which the
reference never sets).

It does not work: on Isaac the card's edge gets into the slot but is never pressed home (0 of 4,
45-62 replies against the reference's 16).  Kept as a record of what was tried."""

from __future__ import annotations

from ruibench.tasks.impl.cr4_graphics_card import CardPolicy
from ruikit.diagnostics.alter_reference import Way, adopt


class OtherFace(CardPolicy):
    name = "card_other_face"

    def reset(self) -> None:
        super().reset()
        self._twin = 1


WAY = Way("CR4", "cr4_other_face", lambda inst: adopt(OtherFace, inst), "the card turned about",
          isaac="fails: 0/4, never pressed home")
