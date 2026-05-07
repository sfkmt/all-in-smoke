"""Main pot and side pot construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from poker_engine.table import PlayerState


@dataclass(frozen=True)
class SidePot:
    amount: int
    eligible_seats: List[int]
    contributing_seats: List[int]

    def to_dict(self) -> dict:
        return {
            "amount": self.amount,
            "eligible_seats": list(self.eligible_seats),
            "contributing_seats": list(self.contributing_seats),
        }


def build_side_pots(players: Iterable[PlayerState]) -> List[SidePot]:
    player_list = list(players)
    levels = sorted({player.committed for player in player_list if player.committed > 0})
    pots: List[SidePot] = []
    previous = 0
    for level in levels:
        contributors = [player for player in player_list if player.committed >= level]
        amount = (level - previous) * len(contributors)
        eligible = [player.seat_id for player in contributors if not player.folded]
        if amount > 0 and eligible:
            pots.append(
                SidePot(
                    amount=amount,
                    eligible_seats=sorted(eligible),
                    contributing_seats=sorted(player.seat_id for player in contributors),
                )
            )
        previous = level
    return pots

