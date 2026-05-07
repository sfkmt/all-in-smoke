"""Betting round and hand progression helpers."""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from poker_engine.actions import ALL_IN, BET, CALL, CHECK, FOLD, RAISE, LegalAction, PokerAction
from poker_engine.deck import Deck
from poker_engine.table import FLOP, PREFLOP, RIVER, SHOWDOWN, TURN, HandState, PlayerState


class IllegalActionError(ValueError):
    """Raised when an action is not legal for the current hand state."""


def _sorted_seats(state: HandState) -> List[int]:
    return sorted(player.seat_id for player in state.players)


def _next_seat(state: HandState, after_seat: int, candidates: Optional[Iterable[int]] = None) -> Optional[int]:
    seats = _sorted_seats(state)
    candidate_set = set(candidates if candidates is not None else seats)
    if not seats or not candidate_set:
        return None
    start_index = seats.index(after_seat) if after_seat in seats else -1
    for offset in range(1, len(seats) + 1):
        seat = seats[(start_index + offset) % len(seats)]
        if seat in candidate_set:
            return seat
    return None


def _blind_seats(state: HandState) -> tuple[int, int]:
    if len(state.players) == 2:
        big_blind = _next_seat(state, state.button_seat)
        if big_blind is None:
            raise ValueError("Cannot determine heads-up big blind")
        return state.button_seat, big_blind
    small_blind = _next_seat(state, state.button_seat)
    if small_blind is None:
        raise ValueError("Cannot determine small blind")
    big_blind = _next_seat(state, small_blind)
    if big_blind is None:
        raise ValueError("Cannot determine big blind")
    return small_blind, big_blind


def _next_actionable_seat(state: HandState, after_seat: int) -> Optional[int]:
    candidates = [
        player.seat_id
        for player in state.players
        if player.can_act
        and (player.current_bet < state.current_bet or player.seat_id not in state.acted_seats)
    ]
    return _next_seat(state, after_seat, candidates)


def _first_preflop_action_seat(state: HandState) -> Optional[int]:
    _, big_blind = _blind_seats(state)
    return _next_actionable_seat(state, big_blind)


def _first_postflop_action_seat(state: HandState) -> Optional[int]:
    return _next_actionable_seat(state, state.button_seat)


def _deal_hole_cards(state: HandState) -> None:
    first = _next_seat(state, state.button_seat)
    if first is None:
        raise ValueError("Cannot deal without players")
    seats = _sorted_seats(state)
    start = seats.index(first)
    order = seats[start:] + seats[:start]
    for _ in range(2):
        for seat in order:
            state.player(seat).hole_cards.extend(state.deck.deal(1))


def start_hand(
    hand_id: int,
    stacks: Dict[int, int],
    button_seat: int,
    small_blind: int,
    big_blind: int,
    seed: Optional[int] = None,
) -> HandState:
    players = [
        PlayerState(seat_id=int(seat_id), stack=int(stack))
        for seat_id, stack in sorted(stacks.items())
        if int(stack) > 0
    ]
    if len(players) < 2:
        raise ValueError("At least two players with chips are required")
    state = HandState(
        hand_id=hand_id,
        players=players,
        deck=Deck(seed=seed),
        button_seat=int(button_seat),
        small_blind=int(small_blind),
        big_blind=int(big_blind),
        last_raise_amount=int(big_blind),
    )
    _deal_hole_cards(state)
    small_seat, big_seat = _blind_seats(state)
    state.player(small_seat).contribute(state.small_blind)
    state.player(big_seat).contribute(state.big_blind)
    state.current_bet = max(player.current_bet for player in state.players)
    state.last_raise_amount = state.big_blind
    state.action_seat = _first_preflop_action_seat(state)
    return state


def legal_actions(state: HandState, seat_id: int) -> List[LegalAction]:
    if state.completed:
        return []
    player = state.player(seat_id)
    if not player.can_act:
        return []
    to_call = max(0, state.current_bet - player.current_bet)
    max_total = player.current_bet + player.stack
    actions: List[LegalAction] = []

    if to_call > 0:
        actions.append(LegalAction(FOLD, to_call=to_call))
        actions.append(LegalAction(CALL, to_call=to_call))
        min_raise_to = state.current_bet + max(state.last_raise_amount, state.big_blind)
        if max_total >= min_raise_to:
            actions.append(LegalAction(RAISE, min_amount=min_raise_to, max_amount=max_total, to_call=to_call))
        if player.stack > 0:
            actions.append(LegalAction(ALL_IN, min_amount=max_total, max_amount=max_total, to_call=to_call))
        return actions

    actions.append(LegalAction(CHECK, to_call=0))
    if player.stack <= 0:
        return actions
    if state.current_bet == 0:
        if max_total >= state.big_blind:
            actions.append(LegalAction(BET, min_amount=state.big_blind, max_amount=max_total, to_call=0))
        actions.append(LegalAction(ALL_IN, min_amount=max_total, max_amount=max_total, to_call=0))
        return actions
    min_raise_to = state.current_bet + max(state.last_raise_amount, state.big_blind)
    if max_total >= min_raise_to:
        actions.append(LegalAction(RAISE, min_amount=min_raise_to, max_amount=max_total, to_call=0))
    actions.append(LegalAction(ALL_IN, min_amount=max_total, max_amount=max_total, to_call=0))
    return actions


def _legal_by_name(state: HandState, seat_id: int) -> Dict[str, LegalAction]:
    return {action.action: action for action in legal_actions(state, seat_id)}


def _amount_in_range(action: PokerAction, legal: LegalAction) -> int:
    if action.amount is None:
        raise IllegalActionError(f"{action.action} requires an amount")
    amount = int(action.amount)
    if legal.min_amount is not None and amount < legal.min_amount:
        raise IllegalActionError(f"{action.action} amount {amount} below minimum {legal.min_amount}")
    if legal.max_amount is not None and amount > legal.max_amount:
        raise IllegalActionError(f"{action.action} amount {amount} exceeds maximum {legal.max_amount}")
    return amount


def betting_round_complete(state: HandState) -> bool:
    if len(state.contenders()) <= 1:
        return True
    actionable = state.actionable_players()
    if not actionable:
        return True
    return all(
        player.current_bet == state.current_bet and player.seat_id in state.acted_seats
        for player in actionable
    )


def apply_action(state: HandState, seat_id: int, action: PokerAction) -> Dict:
    if state.action_seat is not None and seat_id != state.action_seat:
        raise IllegalActionError(f"Seat {seat_id} cannot act now; expected seat {state.action_seat}")
    player = state.player(seat_id)
    legal = _legal_by_name(state, seat_id)
    if action.action not in legal:
        raise IllegalActionError(f"Action {action.action!r} is not legal for seat {seat_id}")

    to_call_before = max(0, state.current_bet - player.current_bet)
    previous_bet = state.current_bet
    contributed = 0

    if action.action == FOLD:
        player.folded = True
        state.acted_seats.add(seat_id)
    elif action.action == CHECK:
        state.acted_seats.add(seat_id)
    elif action.action == CALL:
        contributed = player.contribute(to_call_before)
        state.acted_seats.add(seat_id)
    elif action.action in {BET, RAISE}:
        amount = _amount_in_range(action, legal[action.action])
        contributed = player.contribute(amount - player.current_bet)
        state.current_bet = max(state.current_bet, player.current_bet)
        raise_amount = state.current_bet - previous_bet
        state.last_raise_amount = max(raise_amount, state.big_blind)
        state.acted_seats = {seat_id}
    elif action.action == ALL_IN:
        contributed = player.contribute(player.stack)
        if player.current_bet > state.current_bet:
            state.current_bet = player.current_bet
            raise_amount = state.current_bet - previous_bet
            if raise_amount >= state.last_raise_amount:
                state.last_raise_amount = raise_amount
                state.acted_seats = {seat_id}
            else:
                state.acted_seats.add(seat_id)
        else:
            state.acted_seats.add(seat_id)

    record = {
        "hand_id": state.hand_id,
        "street": state.street,
        "seat_id": seat_id,
        "action": action.action,
        "amount": player.current_bet,
        "contributed": contributed,
        "to_call_before": to_call_before,
        "pot_after": state.pot,
        "stack_after": player.stack,
    }
    state.action_history.append(record)

    if len(state.contenders()) <= 1:
        state.action_seat = None
    elif betting_round_complete(state):
        state.action_seat = None
    else:
        state.action_seat = _next_actionable_seat(state, seat_id)
    return record


def advance_street(state: HandState) -> str:
    if not betting_round_complete(state):
        raise ValueError("Cannot advance street before betting round is complete")
    if len(state.contenders()) <= 1:
        state.action_seat = None
        return state.street

    if state.street == PREFLOP:
        state.deck.burn()
        state.board.extend(state.deck.deal(3))
        state.street = FLOP
    elif state.street == FLOP:
        state.deck.burn()
        state.board.extend(state.deck.deal(1))
        state.street = TURN
    elif state.street == TURN:
        state.deck.burn()
        state.board.extend(state.deck.deal(1))
        state.street = RIVER
    elif state.street == RIVER:
        state.street = SHOWDOWN
        state.action_seat = None
        return state.street
    else:
        return state.street

    state.reset_street_bets()
    state.action_seat = _first_postflop_action_seat(state)
    if betting_round_complete(state):
        state.action_seat = None
    return state.street


def deal_remaining_board(state: HandState) -> None:
    while len(state.board) < 5:
        state.deck.burn()
        state.board.extend(state.deck.deal(1 if state.board else 3))
    state.street = SHOWDOWN
    state.action_seat = None

