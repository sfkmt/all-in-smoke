"""Tests for agent observations and the simulation runner."""

import io
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

from poker_agents import AggressiveAgent, BaseAgent, CallingAgent, RandomAgent, TightAgent
from poker_agents.base import AgentDecision, Observation
from poker_engine import PokerAction, apply_action, start_hand
from poker_simulation import (
    JsonlLogger,
    build_observation,
    resolve_action,
    run_hand,
    run_tournament,
)


class ScriptedSequence(BaseAgent):
    """Test helper: returns a pre-loaded list of AgentDecisions."""

    def __init__(self, agent_id: str, decisions):
        super().__init__(agent_id)
        self.decisions = list(decisions)
        self.calls = 0
        self.observations = []

    def decide_action(self, observation: Observation) -> AgentDecision:
        self.observations.append(observation)
        if self.calls < len(self.decisions):
            decision = self.decisions[self.calls]
        else:
            decision = AgentDecision(action="check")
        self.calls += 1
        return decision


class ObservationTests(unittest.TestCase):
    def test_observation_hides_other_hole_cards_and_deck(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        observation = build_observation(state, 0)
        payload = observation.to_dict()

        self.assertNotIn("deck", payload)
        self.assertNotIn("seed", payload)
        self.assertEqual(len(observation.hole_cards), 2)
        own = set(observation.hole_cards)
        others = []
        for seat in (1, 2):
            others.extend(str(card) for card in state.player(seat).hole_cards)
        self.assertTrue(own.isdisjoint(others))
        self.assertIn("call", observation.legal_action_names())

    def test_observation_exposes_amount_bounds_for_raise(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        observation = build_observation(state, 0)

        self.assertEqual(observation.to_call, 20)
        self.assertEqual(observation.min_raise_to, 40)
        self.assertEqual(observation.max_raise_to, 1000)


class SafeFallbackTests(unittest.TestCase):
    def test_illegal_action_falls_back_to_call(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        decision = AgentDecision(action="check")
        action, reason = resolve_action(state, 0, decision)

        self.assertEqual(action.action, "call")
        self.assertIsNotNone(reason)
        self.assertTrue(reason.startswith("illegal"))

    def test_malformed_action_falls_back(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        decision = AgentDecision(action="teleport")
        action, reason = resolve_action(state, 0, decision)

        self.assertEqual(action.action, "call")
        self.assertTrue(reason.startswith("malformed"))

    def test_amount_below_min_falls_back(self):
        state = start_hand(
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=1,
        )
        decision = AgentDecision(action="raise", amount=25)
        action, reason = resolve_action(state, 0, decision)

        self.assertEqual(action.action, "call")
        self.assertEqual(reason, "amount_below_min")


class RunHandTests(unittest.TestCase):
    def test_run_hand_completes_with_calling_agents(self):
        agents = {seat: CallingAgent(f"call-{seat}") for seat in (0, 1, 2)}
        logger = JsonlLogger()
        result = run_hand(
            agents,
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=5,
            logger=logger,
        )

        total_before = 3 * 1000
        total_after = sum(result.final_stacks.values())
        self.assertEqual(total_before, total_after)
        self.assertEqual(result.street, "showdown")
        self.assertIsNotNone(result.showdown)
        events = [event["event"] for event in logger.events]
        self.assertEqual(events[0], "hand_start")
        self.assertIn("hand_result", events)
        self.assertIn("action", events)
        self.assertEqual(events[-1], "session_snapshot")

    def test_run_hand_fold_settlement(self):
        decisions_by_seat = {
            0: [AgentDecision(action="fold")],
            1: [AgentDecision(action="fold")],
            2: [],
        }
        agents = {seat: ScriptedSequence(f"s{seat}", decisions_by_seat[seat]) for seat in (0, 1, 2)}
        result = run_hand(
            agents,
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=7,
        )

        self.assertEqual(result.winners, [2])
        self.assertEqual(result.payouts, {2: 30})
        self.assertIsNone(result.showdown)

    def test_all_in_hand_runs_to_showdown_without_further_actions(self):
        decisions_by_seat = {
            0: [AgentDecision(action="all_in")],
            1: [AgentDecision(action="call")],
        }
        agents = {seat: ScriptedSequence(f"s{seat}", decisions_by_seat[seat]) for seat in (0, 1)}
        result = run_hand(
            agents,
            hand_id=1,
            stacks={0: 200, 1: 200},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=11,
        )

        self.assertEqual(result.street, "showdown")
        self.assertEqual(sum(result.final_stacks.values()), 400)
        self.assertEqual(len(result.board), 5)

    def test_run_hand_logs_memory_reasoning_and_table_talk(self):
        talk_from_0 = {"to": "all", "text": "I sense weakness."}
        decisions_by_seat = {
            0: [
                AgentDecision(
                    action="raise",
                    amount=60,
                    inner_voice="pretending to bluff",
                    table_talk=talk_from_0,
                    psych={"tilt": 0.2, "mood": "provocative"},
                ),
                AgentDecision(action="check", inner_voice="slow it down"),
            ],
            1: [AgentDecision(action="call", inner_voice="trap set"), AgentDecision(action="check")],
            2: [AgentDecision(action="fold", inner_voice="too rich for me")],
        }
        agents = {
            seat: ScriptedSequence(f"s{seat}", decisions_by_seat[seat]) for seat in (0, 1, 2)
        }
        logger = JsonlLogger()
        run_hand(
            agents,
            hand_id=1,
            stacks={0: 1000, 1: 1000, 2: 1000},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=5,
            logger=logger,
        )
        mr_events = [e for e in logger.events if e["event"] == "memory_reasoning"]
        talk_events = [e for e in logger.events if e["event"] == "table_talk"]
        action_events = [e for e in logger.events if e["event"] == "action"]

        self.assertTrue(mr_events)
        self.assertEqual(len(mr_events), len(action_events))
        for action_ev, mr_ev in zip(action_events, mr_events):
            self.assertEqual(action_ev["step"], mr_ev["step"])
            self.assertEqual(action_ev["seat_id"], mr_ev["seat_id"])

        self.assertTrue(talk_events)
        first_talk = talk_events[0]
        self.assertEqual(first_talk["seat_id"], 0)
        self.assertEqual(first_talk["text"], talk_from_0["text"])

        # Later seats should observe seat 0's public table_talk when acting.
        later_obs = [
            obs for obs in agents[1].observations if obs.recent_table_talk
        ]
        self.assertTrue(later_obs, "seat 1 should see seat 0's table talk")
        self.assertEqual(later_obs[0].recent_table_talk[0]["text"], talk_from_0["text"])

    def test_run_hand_normalizes_card_language_before_logging(self):
        logger = JsonlLogger()
        agents = {
            0: ScriptedSequence(
                "s0",
                [
                    AgentDecision(
                        action="fold",
                        reasoning="A6sは弱い",
                        inner_voice="5クラブ7か",
                        table_talk={"to": "all", "text": "Tdを見る"},
                        psych={"reads": {"1": "As強め"}},
                    )
                ],
            ),
            1: ScriptedSequence("s1", [AgentDecision(action="check")]),
            2: ScriptedSequence("s2", [AgentDecision(action="check")]),
        }

        run_hand(
            agents,
            hand_id=1,
            stacks={0: 500, 1: 500, 2: 500},
            button_seat=0,
            small_blind=5,
            big_blind=10,
            seed=33,
            logger=logger,
        )

        action_event = next(e for e in logger.events if e["event"] == "action")
        decision = action_event["decision"]
        self.assertEqual(decision["reasoning"], "Aと6のスーテッドは弱い")
        self.assertEqual(decision["inner_voice"], "クラブ5と7か")
        self.assertEqual(decision["table_talk"]["text"], "ダイヤ10を見る")
        self.assertEqual(decision["psych"]["reads"]["1"], "スペードA強め")

    def test_six_seat_hand_completes(self):
        agents = {seat: RandomAgent(f"r{seat}", seed=seat + 1) for seat in range(6)}
        result = run_hand(
            agents,
            hand_id=1,
            stacks={seat: 500 for seat in range(6)},
            button_seat=0,
            small_blind=5,
            big_blind=10,
            seed=21,
        )

        self.assertEqual(sum(result.final_stacks.values()), 3000)

    def test_scripted_agents_emit_inner_voice_for_poker_phase(self):
        logger = JsonlLogger()
        agents = {
            0: TightAgent("tight"),
            1: CallingAgent("calling"),
            2: AggressiveAgent("aggro"),
        }
        run_hand(
            agents,
            hand_id=1,
            stacks={0: 500, 1: 500, 2: 500},
            button_seat=0,
            small_blind=5,
            big_blind=10,
            seed=33,
            logger=logger,
        )
        mr_events = [e for e in logger.events if e["event"] == "memory_reasoning"]

        self.assertTrue(mr_events)
        self.assertTrue(all(e.get("inner_voice") for e in mr_events))
        self.assertTrue(all(isinstance(e.get("psych"), dict) for e in mr_events))


class TournamentTests(unittest.TestCase):
    def test_tournament_conserves_chips_and_produces_standings(self):
        agents = {
            0: TightAgent("tight"),
            1: CallingAgent("calling"),
            2: AggressiveAgent("aggro"),
        }
        result = run_tournament(
            agents,
            starting_stacks={0: 500, 1: 500, 2: 500},
            num_hands=6,
            small_blind=5,
            big_blind=10,
            button_seat=0,
            seed=100,
        )

        self.assertEqual(sum(result.final_stacks.values()), 1500)
        self.assertEqual(len(result.standings), 3)
        self.assertEqual(
            sorted(row["seat_id"] for row in result.standings),
            [0, 1, 2],
        )

    def test_tournament_stops_when_only_one_player_has_chips(self):
        agents = {
            0: AggressiveAgent("aggro"),
            1: CallingAgent("call"),
        }
        result = run_tournament(
            agents,
            starting_stacks={0: 50, 1: 50},
            num_hands=100,
            small_blind=5,
            big_blind=10,
            button_seat=0,
            seed=2,
        )

        survivors = [seat for seat, chips in result.final_stacks.items() if chips > 0]
        self.assertEqual(len(survivors), 1)
        self.assertEqual(sum(result.final_stacks.values()), 100)

    def test_tournament_can_increase_blinds_by_hand_level(self):
        agents = {
            0: TightAgent("tight"),
            1: CallingAgent("calling"),
            2: AggressiveAgent("aggro"),
        }
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "tournament.jsonl"
            run_tournament(
                agents,
                starting_stacks={0: 1000, 1: 1000, 2: 1000},
                num_hands=5,
                small_blind=5,
                big_blind=10,
                blind_increase_every=2,
                blind_multiplier=2,
                max_big_blind=40,
                button_seat=0,
                seed=100,
                log_path=log_path,
            )

            hand_starts = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("event") == "hand_start"
            ]

        self.assertEqual([event["small_blind"] for event in hand_starts], [5, 5, 10, 10, 20])
        self.assertEqual([event["big_blind"] for event in hand_starts], [10, 10, 20, 20, 40])

    def test_tournament_can_use_fixed_heads_up_blinds(self):
        agents = {
            0: TightAgent("tight"),
            1: CallingAgent("calling"),
        }
        with TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "heads_up.jsonl"
            run_tournament(
                agents,
                starting_stacks={0: 1000, 1: 1000},
                num_hands=3,
                small_blind=5,
                big_blind=10,
                blind_increase_every=1,
                blind_multiplier=2,
                max_big_blind=80,
                freeze_blinds_when_heads_up=True,
                heads_up_small_blind=160,
                heads_up_big_blind=320,
                button_seat=0,
                seed=100,
                log_path=log_path,
            )

            hand_starts = [
                json.loads(line)
                for line in log_path.read_text(encoding="utf-8").splitlines()
                if json.loads(line).get("event") == "hand_start"
            ]

        self.assertTrue(hand_starts)
        self.assertTrue(all(event["small_blind"] == 160 for event in hand_starts))
        self.assertTrue(all(event["big_blind"] == 320 for event in hand_starts))


class LoggerTests(unittest.TestCase):
    def test_logger_emits_valid_jsonl(self):
        buffer = io.StringIO()
        logger = JsonlLogger(stream=buffer)
        agents = {seat: CallingAgent(f"c{seat}") for seat in (0, 1)}
        run_hand(
            agents,
            hand_id=1,
            stacks={0: 200, 1: 200},
            button_seat=0,
            small_blind=10,
            big_blind=20,
            seed=3,
            logger=logger,
        )

        lines = [line for line in buffer.getvalue().splitlines() if line]
        self.assertTrue(lines)
        for line in lines:
            parsed = json.loads(line)
            self.assertIn("event", parsed)
        self.assertEqual(json.loads(lines[0])["event"], "hand_start")
        events = [json.loads(line)["event"] for line in lines]
        self.assertIn("hand_result", events)
        self.assertEqual(events[-1], "session_snapshot")


if __name__ == "__main__":
    unittest.main()
