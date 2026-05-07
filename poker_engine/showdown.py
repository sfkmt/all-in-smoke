"""Showdown and settlement helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from poker_engine.betting import deal_remaining_board
from poker_engine.hand_evaluator import HandRank, evaluate_seven
from poker_engine.pots import build_side_pots
from poker_engine.table import SHOWDOWN, HandState


@dataclass
class ShowdownResult:
    winners: List[int]
    payouts: Dict[int, int]
    hand_ranks: Dict[int, HandRank] = field(default_factory=dict)
    side_pots: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "winners": list(self.winners),
            "payouts": dict(self.payouts),
            "hand_ranks": {str(seat): rank.to_dict() for seat, rank in self.hand_ranks.items()},
            "side_pots": list(self.side_pots),
        }


def settle_fold_win(state: HandState) -> ShowdownResult:
    contenders = state.contenders()
    if len(contenders) != 1:
        raise ValueError("settle_fold_win requires exactly one contender")
    winner = contenders[0]
    payout = state.pot
    winner.stack += payout
    state.completed = True
    state.winners = [winner.seat_id]
    state.payouts = {winner.seat_id: payout}
    state.action_seat = None
    return ShowdownResult(winners=[winner.seat_id], payouts={winner.seat_id: payout})


def settle_showdown(state: HandState) -> ShowdownResult:
    if len(state.contenders()) == 1:
        return settle_fold_win(state)
    if len(state.board) < 5:
        deal_remaining_board(state)
    state.street = SHOWDOWN

    side_pots = build_side_pots(state.players)
    ranks = {
        player.seat_id: evaluate_seven([*player.hole_cards, *state.board])
        for player in state.contenders()
    }
    payouts: Dict[int, int] = {}
    for pot in side_pots:
        eligible_ranks = {seat: ranks[seat] for seat in pot.eligible_seats if seat in ranks}
        if not eligible_ranks:
            continue
        best = max(eligible_ranks.values())
        winners = sorted(seat for seat, rank in eligible_ranks.items() if rank == best)
        share, remainder = divmod(pot.amount, len(winners))
        for index, seat in enumerate(winners):
            payouts[seat] = payouts.get(seat, 0) + share + (1 if index < remainder else 0)

    for seat, payout in payouts.items():
        state.player(seat).stack += payout
    state.completed = True
    state.winners = sorted(payouts, key=lambda seat: (-payouts[seat], seat))
    state.payouts = dict(payouts)
    state.action_seat = None
    return ShowdownResult(
        winners=list(state.winners),
        payouts=payouts,
        hand_ranks=ranks,
        side_pots=[pot.to_dict() for pot in side_pots],
    )

