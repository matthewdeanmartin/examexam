# utils/secure_random.py
from __future__ import annotations

import secrets
from collections.abc import Sequence
from typing import Any


class SecureRandom:
    """
    Drop-in replacement for random.Random, using secrets for security.

    Methods mirror a subset of random.Random that people actually use.
    """

    def __init__(self, seed: Any | None = None) -> None:
        # secrets doesn't support seeding — seed is ignored.
        self._seed = seed

    def choice(self, seq: Sequence[Any]) -> Any:
        return secrets.choice(seq)

    def sample(self, population: Sequence[Any], k: int) -> list[Any]:
        if k > len(population):
            raise ValueError("Sample larger than population")
        # Use SystemRandom.sample for efficiency and correctness
        return secrets.SystemRandom().sample(population, k)

    def randint(self, a: int, b: int) -> int:
        return secrets.randbelow(b - a + 1) + a

    def randbelow(self, n: int) -> int:
        return secrets.randbelow(n)

    def token_hex(self, nbytes: int = 32) -> str:
        return secrets.token_hex(nbytes)
