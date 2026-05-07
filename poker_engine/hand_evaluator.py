"""Five-card and seven-card poker hand evaluation."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from typing import Iterable, Optional, Tuple

from poker_engine.cards import Card


CATEGORY_NAMES = {
    8: "straight_flush",
    7: "four_of_a_kind",
    6: "full_house",
    5: "flush",
    4: "straight",
    3: "three_of_a_kind",
    2: "two_pair",
    1: "one_pair",
    0: "high_card",
}


@dataclass(frozen=True, order=True)
class HandRank:
    category: int
    tiebreakers: Tuple[int, ...]
    name: str = field(compare=False)
    cards: Tuple[str, ...] = field(default_factory=tuple, compare=False)

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "name": self.name,
            "tiebreakers": list(self.tiebreakers),
            "cards": list(self.cards),
        }


def _straight_high(rank_values: Iterable[int]) -> Optional[int]:
    unique = set(rank_values)
    if 14 in unique:
        unique.add(1)
    for high in range(14, 4, -1):
        if {high, high - 1, high - 2, high - 3, high - 4}.issubset(unique):
            return high
    return None


def evaluate_five(cards: Iterable[Card]) -> HandRank:
    card_list = list(cards)
    if len(card_list) != 5:
        raise ValueError("evaluate_five requires exactly five cards")

    rank_values = [card.rank_value for card in card_list]
    rank_counts = Counter(rank_values)
    flush = len({card.suit for card in card_list}) == 1
    straight = _straight_high(rank_values)
    sorted_ranks = tuple(sorted(rank_values, reverse=True))
    cards_tuple = tuple(str(card) for card in card_list)

    if straight and flush:
        return HandRank(8, (straight,), CATEGORY_NAMES[8], cards_tuple)

    grouped = sorted(rank_counts.items(), key=lambda item: (item[1], item[0]), reverse=True)
    if grouped[0][1] == 4:
        quad = grouped[0][0]
        kicker = max(rank for rank, count in grouped if count == 1)
        return HandRank(7, (quad, kicker), CATEGORY_NAMES[7], cards_tuple)
    if grouped[0][1] == 3 and grouped[1][1] == 2:
        return HandRank(6, (grouped[0][0], grouped[1][0]), CATEGORY_NAMES[6], cards_tuple)
    if flush:
        return HandRank(5, sorted_ranks, CATEGORY_NAMES[5], cards_tuple)
    if straight:
        return HandRank(4, (straight,), CATEGORY_NAMES[4], cards_tuple)
    if grouped[0][1] == 3:
        trip = grouped[0][0]
        kickers = tuple(sorted((rank for rank, count in grouped if count == 1), reverse=True))
        return HandRank(3, (trip, *kickers), CATEGORY_NAMES[3], cards_tuple)

    pairs = sorted((rank for rank, count in grouped if count == 2), reverse=True)
    if len(pairs) == 2:
        kicker = max(rank for rank, count in grouped if count == 1)
        return HandRank(2, (pairs[0], pairs[1], kicker), CATEGORY_NAMES[2], cards_tuple)
    if len(pairs) == 1:
        kickers = tuple(sorted((rank for rank, count in grouped if count == 1), reverse=True))
        return HandRank(1, (pairs[0], *kickers), CATEGORY_NAMES[1], cards_tuple)
    return HandRank(0, sorted_ranks, CATEGORY_NAMES[0], cards_tuple)


def evaluate_seven(cards: Iterable[Card]) -> HandRank:
    card_list = list(cards)
    if len(card_list) < 5:
        raise ValueError("evaluate_seven requires at least five cards")
    return max(evaluate_five(combo) for combo in combinations(card_list, 5))

