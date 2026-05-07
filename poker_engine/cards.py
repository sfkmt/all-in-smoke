"""Card primitives for Texas Hold'em."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


RANKS = "23456789TJQKA"
SUITS = "cdhs"
RANK_VALUES = {rank: index + 2 for index, rank in enumerate(RANKS)}
RANK_LABELS_JA = {
    "A": "A",
    "K": "K",
    "Q": "Q",
    "J": "J",
    "T": "10",
    "9": "9",
    "8": "8",
    "7": "7",
    "6": "6",
    "5": "5",
    "4": "4",
    "3": "3",
    "2": "2",
}
SUIT_LABELS_JA = {
    "c": "クラブ",
    "d": "ダイヤ",
    "h": "ハート",
    "s": "スペード",
}


@dataclass(frozen=True)
class Card:
    """A single playing card."""

    rank: str
    suit: str

    def __post_init__(self) -> None:
        rank = self.rank.upper()
        suit = self.suit.lower()
        if rank not in RANKS:
            raise ValueError(f"Invalid card rank: {self.rank!r}")
        if suit not in SUITS:
            raise ValueError(f"Invalid card suit: {self.suit!r}")
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    @classmethod
    def from_str(cls, text: str) -> "Card":
        value = str(text).strip()
        if len(value) != 2:
            raise ValueError(f"Card code must have length 2: {text!r}")
        return cls(value[0], value[1])

    @property
    def rank_value(self) -> int:
        return RANK_VALUES[self.rank]

    def __str__(self) -> str:
        return f"{self.rank}{self.suit}"

    def to_dict(self) -> dict:
        return {"rank": self.rank, "suit": self.suit, "code": str(self)}


def parse_cards(values: Iterable[str]) -> List[Card]:
    return [Card.from_str(value) for value in values]


def card_to_japanese(value: str) -> str:
    """Return a natural Japanese label for a two-character poker card code."""
    card = Card.from_str(value)
    return "{0}{1}".format(SUIT_LABELS_JA[card.suit], RANK_LABELS_JA[card.rank])


def cards_to_japanese(values: Iterable[str]) -> List[str]:
    """Return Japanese labels for card codes, preserving order."""
    return [card_to_japanese(value) for value in values]


def cards_text_japanese(values: Iterable[str]) -> str:
    """Return a compact Japanese text label for a list of card codes."""
    labels = cards_to_japanese(values)
    return "・".join(labels)
