"""Deterministic deck implementation."""

from __future__ import annotations

import random
from typing import List, Optional

from poker_engine.cards import Card, RANKS, SUITS


class Deck:
    """A standard 52-card deck with optional deterministic shuffle."""

    def __init__(self, seed: Optional[int] = None, auto_shuffle: bool = True):
        self.seed = seed
        self.rng = random.Random(seed)
        self.cards: List[Card] = [Card(rank, suit) for suit in SUITS for rank in RANKS]
        if auto_shuffle:
            self.shuffle()

    def shuffle(self) -> None:
        self.rng.shuffle(self.cards)

    def deal(self, count: int = 1) -> List[Card]:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count > len(self.cards):
            raise ValueError("Cannot deal more cards than remain in the deck")
        dealt = self.cards[:count]
        del self.cards[:count]
        return dealt

    def burn(self) -> Card:
        return self.deal(1)[0]

    def remaining(self) -> int:
        return len(self.cards)

