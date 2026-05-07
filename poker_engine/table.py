"""Table and hand state models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from poker_engine.cards import Card
from poker_engine.deck import Deck


PREFLOP = "preflop"
FLOP = "flop"
TURN = "turn"
RIVER = "river"
SHOWDOWN = "showdown"


@dataclass
class PlayerState:
    seat_id: int
    stack: int
    hole_cards: List[Card] = field(default_factory=list)
    current_bet: int = 0
    committed: int = 0
    folded: bool = False
    all_in: bool = False

    def contribute(self, amount: int) -> int:
        if amount < 0:
            raise ValueError("Contribution cannot be negative")
        actual = min(int(amount), self.stack)
        self.stack -= actual
        self.current_bet += actual
        self.committed += actual
        if self.stack == 0:
            self.all_in = True
        return actual

    @property
    def can_act(self) -> bool:
        return not self.folded and not self.all_in

    def to_public_dict(self) -> Dict:
        return {
            "seat_id": self.seat_id,
            "stack": self.stack,
            "current_bet": self.current_bet,
            "committed": self.committed,
            "folded": self.folded,
            "all_in": self.all_in,
        }


@dataclass
class HandState:
    hand_id: int
    players: List[PlayerState]
    deck: Deck
    button_seat: int
    small_blind: int
    big_blind: int
    street: str = PREFLOP
    board: List[Card] = field(default_factory=list)
    action_history: List[Dict] = field(default_factory=list)
    acted_seats: Set[int] = field(default_factory=set)
    current_bet: int = 0
    last_raise_amount: int = 0
    action_seat: Optional[int] = None
    completed: bool = False
    winners: List[int] = field(default_factory=list)
    payouts: Dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.players.sort(key=lambda player: player.seat_id)

    def player(self, seat_id: int) -> PlayerState:
        for player in self.players:
            if player.seat_id == seat_id:
                return player
        raise KeyError(f"Unknown seat id: {seat_id}")

    @property
    def pot(self) -> int:
        return sum(player.committed for player in self.players)

    def contenders(self) -> List[PlayerState]:
        return [player for player in self.players if not player.folded]

    def actionable_players(self) -> List[PlayerState]:
        return [player for player in self.players if player.can_act]

    def reset_street_bets(self) -> None:
        for player in self.players:
            player.current_bet = 0
        self.current_bet = 0
        self.last_raise_amount = self.big_blind
        self.acted_seats.clear()

    def to_public_dict(self) -> Dict:
        return {
            "hand_id": self.hand_id,
            "street": self.street,
            "button_seat": self.button_seat,
            "small_blind": self.small_blind,
            "big_blind": self.big_blind,
            "board": [str(card) for card in self.board],
            "pot": self.pot,
            "current_bet": self.current_bet,
            "action_seat": self.action_seat,
            "players": [player.to_public_dict() for player in self.players],
        }

