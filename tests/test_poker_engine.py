"""Tests for the deterministic Texas Hold'em engine."""

import unittest

from poker_engine import (
    Deck,
    IllegalActionError,
    PokerAction,
    advance_street,
    apply_action,
    build_side_pots,
    card_to_japanese,
    cards_text_japanese,
    evaluate_five,
    evaluate_seven,
    legal_actions,
    parse_cards,
    settle_fold_win,
    settle_showdown,
    start_hand,
)
from poker_engine.table import PlayerState


class PokerEngineTests(unittest.TestCase):
    def test_card_codes_have_japanese_labels(self):
        self.assertEqual(card_to_japanese("As"), "スペードA")
        self.assertEqual(card_to_japanese("Td"), "ダイヤ10")
        self.assertEqual(cards_text_japanese(["Ah", "Kd"]), "ハートA・ダイヤK")

    def test_deck_has_52_unique_seed_reproducible_cards(self):
        deck = Deck(seed=7, auto_shuffle=False)
        self.assertEqual(len(deck.cards), 52)
        self.assertEqual(len({str(card) for card in deck.cards}), 52)
        self.assertEqual(
            [str(card) for card in Deck(seed=42).deal(10)],
            [str(card) for card in Deck(seed=42).deal(10)],
        )

    def test_hand_evaluator_orders_common_hands(self):
        straight_flush = evaluate_five(parse_cards(["Ah", "Kh", "Qh", "Jh", "Th"]))
        quads = evaluate_five(parse_cards(["9h", "9d", "9s", "9c", "2d"]))
        wheel = evaluate_five(parse_cards(["Ah", "2d", "3s", "4c", "5h"]))
        full_house = evaluate_seven(parse_cards(["Ah", "Ad", "As", "Kc", "Kd", "2h", "3s"]))

        self.assertGreater(straight_flush, quads)
        self.assertEqual(wheel.name, "straight")
        self.assertEqual(wheel.tiebreakers, (5,))
        self.assertEqual(full_house.name, "full_house")
        self.assertEqual(full_house.tiebreakers, (14, 13))

    def test_start_hand_posts_blinds_and_sets_action(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )

        self.assertEqual(state.player(1).current_bet, 10)
        self.assertEqual(state.player(2).current_bet, 20)
        self.assertEqual(state.action_seat, 0)
        self.assertEqual(len(state.player(0).hole_cards), 2)

    def test_heads_up_button_is_small_blind_and_acts_first_preflop(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 500, 1: 500},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=2,
        )

        self.assertEqual(state.player(0).current_bet, 10)
        self.assertEqual(state.player(1).current_bet, 20)
        self.assertEqual(state.action_seat, 0)

    def test_raise_requires_minimum_total_amount(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        actions = {action.action: action for action in legal_actions(state, 0)}

        self.assertEqual(actions["call"].to_call, 20)
        self.assertEqual(actions["raise"].min_amount, 40)
        with self.assertRaises(IllegalActionError):
            apply_action(state, 0, PokerAction("raise", amount=39))

    def test_betting_round_advances_to_flop(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )

        apply_action(state, 0, PokerAction("call"))
        apply_action(state, 1, PokerAction("call"))
        apply_action(state, 2, PokerAction("check"))

        self.assertIsNone(state.action_seat)
        self.assertEqual(advance_street(state), "flop")
        self.assertEqual(len(state.board), 3)
        self.assertEqual(state.action_seat, 1)

    def test_side_pots_for_all_in(self):
        players = [
            PlayerState(seat_id=0, stack=0, committed=50, all_in=True),
            PlayerState(seat_id=1, stack=0, committed=100, all_in=True),
            PlayerState(seat_id=2, stack=100, committed=100),
        ]

        pots = build_side_pots(players)

        self.assertEqual([pot.amount for pot in pots], [150, 100])
        self.assertEqual(pots[0].eligible_seats, [0, 1, 2])
        self.assertEqual(pots[1].eligible_seats, [1, 2])

    def test_folded_hand_awards_pot_to_last_contender(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )

        apply_action(state, 0, PokerAction("fold"))
        apply_action(state, 1, PokerAction("fold"))
        result = settle_fold_win(state)

        self.assertEqual(result.winners, [2])
        self.assertEqual(result.payouts, {2: 30})
        self.assertTrue(state.completed)

    def test_showdown_splits_side_pots_by_eligible_winners(self):
        state = start_hand(
            hand_id=2,
            stacks={0: 50, 1: 100, 2: 100},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        state.player(0).hole_cards = parse_cards(["Ah", "Ad"])
        state.player(1).hole_cards = parse_cards(["Kh", "Kd"])
        state.player(2).hole_cards = parse_cards(["Qh", "Qd"])
        state.board = parse_cards(["2c", "7s", "9h", "Jd", "3c"])
        state.street = "showdown"
        for player in state.players:
            player.current_bet = 0
            player.committed = 0
            player.folded = False
            player.all_in = True
            player.stack = 0
        state.player(0).committed = 50
        state.player(1).committed = 100
        state.player(2).committed = 100

        result = settle_showdown(state)

        self.assertEqual(result.payouts, {0: 150, 1: 100})
        self.assertEqual(state.player(0).stack, 150)
        self.assertEqual(state.player(1).stack, 100)
        self.assertEqual(state.player(2).stack, 0)

    def test_complete_hand_reaches_showdown_after_checked_streets(self):
        state = start_hand(
            hand_id=3,
            stacks={0: 1000, 1: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=3,
        )

        apply_action(state, 0, PokerAction("call"))
        apply_action(state, 1, PokerAction("check"))
        self.assertEqual(advance_street(state), "flop")
        apply_action(state, 1, PokerAction("check"))
        apply_action(state, 0, PokerAction("check"))
        self.assertEqual(advance_street(state), "turn")
        apply_action(state, 1, PokerAction("check"))
        apply_action(state, 0, PokerAction("check"))
        self.assertEqual(advance_street(state), "river")
        apply_action(state, 1, PokerAction("check"))
        apply_action(state, 0, PokerAction("check"))
        self.assertEqual(advance_street(state), "showdown")

        result = settle_showdown(state)

        self.assertTrue(state.completed)
        self.assertEqual(sum(result.payouts.values()), 40)


if __name__ == "__main__":
    unittest.main()
