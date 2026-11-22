"""Seed discipline.

One ``world_seed`` per episode determines the initial state, the event times
and any branch state (design doc details §1.3).  Sub-streams are derived by
name so that adding a new consumer never shifts the numbers an existing one
draws -- which would silently invalidate a frozen split.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass
from typing import Sequence, TypeVar

T = TypeVar("T")


def derive_seed(world_seed: int, *parts: object) -> int:
    """Deterministically derive a 64-bit sub-seed from ``world_seed`` and a label."""
    key = "|".join([str(world_seed), *(str(p) for p in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big")


@dataclass
class SeedBundle:
    """The named random streams of a single episode.

    Every stream is independent, so ``bundle.stream("events")`` yields the same
    numbers whether or not anything drew from ``bundle.stream("layout")``.
    """

    world_seed: int

    def stream(self, name: str) -> random.Random:
        return random.Random(derive_seed(self.world_seed, name))

    def choice(self, name: str, options: Sequence[T]) -> T:
        return self.stream(name).choice(list(options))

    def uniform(self, name: str, lo: float, hi: float) -> float:
        return self.stream(name).uniform(lo, hi)

    def jitter(self, name: str, base, spread: float):
        """Jitter each component of a vector by +-spread."""
        rng = self.stream(name)
        return tuple(v + rng.uniform(-spread, spread) for v in base)
