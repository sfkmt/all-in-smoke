"""Tests for ALL-IN SMOKE crisis transfer."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from live_fire_simulation import (
    LiveFireAgentState,
    _crisis_match_snapshot,
    run_live_fire_during_poker,
)
from poker_agents.manifest_loader import load_manifest
from smoke_simulation import (
    build_crisis_profiles,
    build_public_reputations,
    run_smoke_crisis,
)
from tools.run_all_in_smoke import run_all_in_smoke


ROOT = Path(__file__).resolve().parents[1]


class SmokeReputationTests(unittest.TestCase):
    def test_public_reputation_extracts_bluff_and_fold_signals(self):
        events = [
            {"event": "action", "seat_id": 0, "agent_id": "folder", "action": "fold"},
            {"event": "action", "seat_id": 0, "agent_id": "folder", "action": "call"},
            {"event": "action", "seat_id": 1, "agent_id": "talker", "action": "raise"},
            {"event": "table_talk", "seat_id": 1, "agent_id": "talker", "text": "降りた方がいいよ"},
            {"event": "hand_result", "winners": [1], "showdown": None},
            {
                "event": "session_snapshot",
                "seat_id": 1,
                "agent_id": "talker",
                "snapshot": {"tilt": 0.4, "rivalries": [{"kind": "bluffed_me"}]},
            },
        ]
        reps = build_public_reputations(events)

        self.assertGreater(reps[0].fold_ability, 0.0)
        self.assertGreater(reps[1].bluff_reputation, 0.3)
        self.assertEqual(reps[1].rivalry_kinds["bluffed_me"], 1)
        self.assertEqual(reps[1].tilt, 0.4)

    def test_manifest_crisis_profile_overrides_poker_fallbacks(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        reps = build_public_reputations([], seat_to_agent_id={spec.seat_id: spec.agent_id for spec in manifest.agents})
        profiles = build_crisis_profiles(manifest, reps)

        self.assertAlmostEqual(profiles[2].ability_gaps["help_seeking"], 0.34)
        self.assertAlmostEqual(profiles[3].ability_gaps["situational_awareness"], 0.68)


class SmokeCrisisTests(unittest.TestCase):
    def test_smoke_crisis_emits_transfer_and_summary_events(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "action", "seat_id": 1, "agent_id": "bluff-showman", "action": "raise"},
            {"event": "action", "seat_id": 1, "agent_id": "bluff-showman", "action": "raise"},
            {"event": "table_talk", "seat_id": 1, "agent_id": "bluff-showman", "text": "これは降ろせる"},
            {"event": "table_talk", "seat_id": 1, "agent_id": "bluff-showman", "text": "まだ煙じゃない"},
            {"event": "hand_result", "winners": [1], "showdown": None},
            {"event": "action", "seat_id": 0, "agent_id": "fold-master", "action": "fold"},
            {"event": "action", "seat_id": 0, "agent_id": "fold-master", "action": "call"},
        ]
        result = run_smoke_crisis(manifest, poker_events, seed=813)
        event_names = [event["event"] for event in result.events]

        self.assertIn("crisis_start", event_names)
        self.assertIn("reputation_transfer_input", event_names)
        self.assertIn("evacuation_decision", event_names)
        self.assertIn("crisis_summary", event_names)
        self.assertIn("exited_count", result.metrics)

    def test_all_in_smoke_runner_writes_poker_and_live_fire_logs(self):
        manifest_path = ROOT / "configs" / "all_in_smoke_demo.yaml"
        manifest = load_manifest(manifest_path)
        with tempfile.TemporaryDirectory() as tmp:
            summaries = run_all_in_smoke(
                manifest,
                manifest_path=manifest_path,
                out_dir=Path(tmp),
            )
            self.assertEqual(len(summaries), 1)
            self.assertTrue(Path(summaries[0]["poker_log"]).exists())
            self.assertTrue(Path(summaries[0]["live_fire_log"]).exists())
            self.assertTrue(Path(summaries[0]["full_replay_log"]).exists())
            self.assertIn("live_fire_metrics", summaries[0])

    def test_live_fire_tracks_chip_temptation_without_forcing_an_action(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = []
        step = 1
        for hand_id in range(1, 5):
            poker_events.append({"event": "hand_start", "hand_id": hand_id, "seats": [0, 1]})
            for seat_id, agent_id in [
                (0, "fold-master"),
                (1, "bluff-showman"),
                (2, "calling-station"),
                (3, "volatile-random"),
                (4, "quiet-helper"),
                (5, "suspicious-checker"),
            ]:
                poker_events.append(
                    {
                        "event": "action",
                        "step": step,
                        "hand_id": hand_id,
                        "seat_id": seat_id,
                        "agent_id": agent_id,
                        "action": "call",
                        "stack_after": 1000 - step * 12 if seat_id == 3 else 1000,
                        "pot_after": 120 + step * 30,
                    }
                )
                step += 1

        events = run_live_fire_during_poker(manifest, poker_events, fire_start_hand=2)
        event_names = [event["event"] for event in events]
        statuses = [
            state["status"]
            for event in events
            if event["event"] == "live_fire_tick"
            for state in event["agent_states"].values()
        ]
        states = [
            state
            for event in events
            if event["event"] == "live_fire_tick"
            for state in event["agent_states"].values()
        ]

        self.assertIn("live_fire_start", event_names)
        self.assertIn("live_fire_tick", event_names)
        self.assertIn("stood_up", statuses)
        self.assertTrue(
            any(status in {"clinging_to_stack", "tempted_by_chips"} for status in statuses)
        )
        self.assertTrue(any(float(state.get("chip_temptation", 0.0)) >= 0.58 for state in states))
        prohibited_prefix = "s" + "teal"
        prohibited_field = "s" + "tolen_chips"
        self.assertFalse(any(status.startswith(prohibited_prefix) for status in statuses))
        self.assertTrue(all(prohibited_field not in state for state in states))

    def test_crisis_match_fatal_overrides_chip_lead(self):
        states = {
            0: LiveFireAgentState(seat_id=0, agent_id="chip-leader", status="fatal"),
            1: LiveFireAgentState(seat_id=1, agent_id="short-stack", status="playing"),
        }

        match = _crisis_match_snapshot(
            [0, 1],
            states,
            {0: 5000, 1: 100},
            poker_match={
                "resolved": True,
                "winner_seat": 0,
                "loser_seat": 1,
                "winner_reason": "opponent_busted",
            },
        )

        self.assertEqual(match["winner_seat"], 1)
        self.assertEqual(match["loser_seat"], 0)
        self.assertEqual(match["winner_reason"], "opponent_fatal")
        self.assertEqual(match["overall_winner_seat"], 1)
        self.assertEqual(match["fatal_losers"], [0])

    def test_crisis_match_first_to_stand_forfeits_stack(self):
        states = {
            0: LiveFireAgentState(
                seat_id=0,
                agent_id="first-out",
                status="stood_up",
                stood_up_step=12,
            ),
            1: LiveFireAgentState(seat_id=1, agent_id="still-seated", status="playing"),
        }

        match = _crisis_match_snapshot([0, 1], states, {0: 900, 1: 1100})

        self.assertEqual(match["winner_seat"], 1)
        self.assertEqual(match["loser_seat"], 0)
        self.assertEqual(match["winner_reason"], "opponent_stood_up_first")
        self.assertEqual(
            match["forfeitures"],
            [{"from_seat": 0, "to_seat": 1, "stack": 900, "claimed_stack": 2000}],
        )

    def test_live_fire_emits_crisis_match_pressure_and_result(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1], "stacks": {0: 1200, 1: 800}},
            {
                "event": "action",
                "step": 1,
                "hand_id": 1,
                "seat_id": 0,
                "agent_id": "fold-master",
                "action": "call",
                "stack_after": 1040,
                "pot_after": 320,
            },
            {
                "event": "action",
                "step": 2,
                "hand_id": 1,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "action": "check",
                "stack_after": 640,
                "pot_after": 320,
            },
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_when="tournament_heads_up",
            fire_duration_ticks=8,
        )
        ticks = [event for event in events if event["event"] == "live_fire_tick"]
        result = next(event for event in events if event["event"] == "crisis_match_result")

        self.assertTrue(events[0]["crisis_match_rule"]["active"])
        self.assertTrue(all(tick["crisis_match"]["active"] for tick in ticks))
        self.assertTrue(
            any(
                "crisis_match_forfeit_pressure" in tick["agent_states"]["0"]["dynamic_state"]
                for tick in ticks
            )
        )
        self.assertEqual(result["crisis_match"]["seats"], [0, 1])
        self.assertIn("fatal_overrides_chips", result["crisis_match"]["rule"])

    def test_non_match_members_escape_after_bust_without_forfeit_pressure(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1], "stacks": {0: 1200, 1: 800}},
            {
                "event": "action",
                "step": 1,
                "hand_id": 1,
                "seat_id": 0,
                "agent_id": "fold-master",
                "action": "call",
                "stack_after": 1040,
                "pot_after": 320,
            },
            {
                "event": "action",
                "step": 2,
                "hand_id": 1,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "action": "check",
                "stack_after": 640,
                "pot_after": 320,
            },
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_when="tournament_heads_up",
            fire_duration_ticks=8,
        )
        ticks = [event for event in events if event["event"] == "live_fire_tick"]
        seat2_states = [tick["agent_states"]["2"] for tick in ticks]

        self.assertTrue(
            any(
                state["status"] == "stood_up" and state["motive"] == "stood_up_after_bust"
                for state in seat2_states
            )
        )
        self.assertTrue(
            all(
                "crisis_match_forfeit_pressure" not in state["dynamic_state"]
                for state in seat2_states
            )
        )

    def test_survival_panic_reduces_effective_forfeit_pressure_near_contact(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1], "stacks": {0: 1200, 1: 800}},
            {
                "event": "action",
                "step": 1,
                "hand_id": 1,
                "seat_id": 0,
                "agent_id": "fold-master",
                "action": "call",
                "stack_after": 1040,
                "pot_after": 320,
            },
            {
                "event": "action",
                "step": 2,
                "hand_id": 1,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "action": "check",
                "stack_after": 640,
                "pot_after": 320,
            },
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_when="tournament_heads_up",
            fire_duration_ticks=12,
        )
        samples = [
            tick["agent_states"][seat]["dynamic_state"]
            for tick in events
            if tick["event"] == "live_fire_tick"
            for seat in ("0", "1")
            if "crisis_match_forfeit_pressure" in tick["agent_states"][seat]["dynamic_state"]
        ]

        self.assertTrue(
            any(
                sample.get("crisis_survival_panic", 0.0) > 0.05
                and sample.get("effective_match_forfeit_pressure", 0.0)
                < sample["crisis_match_forfeit_pressure"]
                for sample in samples
            )
        )

    def test_live_fire_uses_poker_bust_when_no_crisis_loss_happened(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1], "stacks": {0: 1200, 1: 800}},
            {
                "event": "action",
                "step": 1,
                "hand_id": 1,
                "seat_id": 0,
                "agent_id": "fold-master",
                "action": "all_in",
                "stack_after": 0,
                "pot_after": 2400,
            },
            {
                "event": "action",
                "step": 2,
                "hand_id": 1,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "action": "call",
                "stack_after": 0,
                "pot_after": 2400,
            },
            {
                "event": "hand_result",
                "hand_id": 1,
                "winners": [1],
                "payouts": {1: 2400},
                "final_stacks": {0: 0, 1: 2000},
                "showdown": {"winner": 1},
            },
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_when="tournament_heads_up",
            fire_duration_ticks=2,
        )
        result = next(event for event in events if event["event"] == "crisis_match_result")
        match = result["crisis_match"]

        self.assertEqual(match["poker_match"]["winner_seat"], 1)
        self.assertEqual(match["poker_match"]["loser_seat"], 0)
        self.assertEqual(match["winner_reason"], "poker_resolved_during_fire")
        self.assertEqual(match["overall_winner_seat"], 1)

    def test_live_fire_accumulates_poker_feedback_into_dynamic_state(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1], "stacks": {0: 1000, 1: 1000}},
            {
                "event": "action",
                "step": 1,
                "hand_id": 1,
                "seat_id": 0,
                "agent_id": "fold-master",
                "action": "raise",
                "stack_after": 900,
                "pot_after": 130,
                "to_call_before": 10,
            },
            {
                "event": "action",
                "step": 2,
                "hand_id": 1,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "action": "fold",
                "stack_after": 970,
                "pot_after": 130,
                "to_call_before": 90,
            },
            {
                "event": "hand_result",
                "hand_id": 1,
                "winners": [0],
                "payouts": {0: 130, 1: 0},
                "final_stacks": {0: 1130, 1: 970},
                "showdown": None,
            },
            {
                "event": "session_snapshot",
                "hand_id": 1,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "snapshot": {
                    "tilt": 0.62,
                    "recent_outcomes": [{"hand_id": 1, "delta": -30, "showdown": False, "note": "小さく負け"}],
                    "rivalries": [{"kind": "bluffed_me", "opponent": "fold-master"}],
                },
            },
            {"event": "hand_start", "hand_id": 2, "seats": [0, 1], "stacks": {0: 1130, 1: 970}},
            {
                "event": "action",
                "step": 3,
                "hand_id": 2,
                "seat_id": 0,
                "agent_id": "fold-master",
                "action": "call",
                "stack_after": 1120,
                "pot_after": 20,
                "to_call_before": 10,
            },
            {
                "event": "action",
                "step": 4,
                "hand_id": 2,
                "seat_id": 1,
                "agent_id": "bluff-showman",
                "action": "call",
                "stack_after": 960,
                "pot_after": 40,
                "to_call_before": 10,
            },
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_hand=2,
            fire_duration_ticks=8,
        )
        first_tick = next(event for event in events if event["event"] == "live_fire_tick")
        seat0 = first_tick["agent_states"]["0"]["dynamic_state"]
        seat1 = first_tick["agent_states"]["1"]["dynamic_state"]
        seat0_state = first_tick["agent_states"]["0"]

        self.assertGreater(seat0["chip_attachment"], 0.0)
        self.assertGreater(seat1["loss_chasing"], 0.2)
        self.assertGreater(seat1["rivalry_pressure"], 0.2)
        self.assertTrue(seat0_state["inner_voice"])
        self.assertIn("fire_read", seat0_state["psych"])

    def test_live_fire_can_start_after_hand_becomes_heads_up(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1, 2, 3]},
            {"event": "action", "step": 1, "hand_id": 1, "seat_id": 0, "agent_id": "fold-master", "action": "fold", "stack_after": 1000, "pot_after": 30},
            {"event": "action", "step": 2, "hand_id": 1, "seat_id": 1, "agent_id": "bluff-showman", "action": "call", "stack_after": 960, "pot_after": 80},
            {"event": "action", "step": 3, "hand_id": 1, "seat_id": 2, "agent_id": "calling-station", "action": "fold", "stack_after": 980, "pot_after": 80},
            {"event": "action", "step": 4, "hand_id": 1, "seat_id": 3, "agent_id": "volatile-random", "action": "bet", "stack_after": 900, "pot_after": 180},
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_when="heads_up",
            fire_duration_ticks=6,
        )
        post_poker_ticks = [
            event for event in events
            if event["event"] == "live_fire_tick" and event.get("post_poker")
        ]

        self.assertEqual(events[0]["event"], "live_fire_start")
        self.assertEqual(events[0]["step"], 4)
        self.assertEqual(events[0]["start_condition"], "heads_up")
        self.assertTrue(post_poker_ticks)
        self.assertEqual(events[-1]["pressure"], 1.0)

    def test_live_fire_can_start_when_tournament_reaches_two_players(self):
        manifest = load_manifest(ROOT / "configs" / "all_in_smoke_demo.yaml")
        poker_events = [
            {"event": "hand_start", "hand_id": 1, "seats": [0, 1, 2]},
            {"event": "action", "step": 1, "hand_id": 1, "seat_id": 0, "agent_id": "fold-master", "action": "fold", "stack_after": 1000, "pot_after": 30},
            {"event": "hand_start", "hand_id": 2, "seats": [0, 1]},
            {"event": "action", "step": 2, "hand_id": 2, "seat_id": 0, "agent_id": "fold-master", "action": "call", "stack_after": 990, "pot_after": 20},
            {"event": "action", "step": 3, "hand_id": 2, "seat_id": 1, "agent_id": "bluff-showman", "action": "check", "stack_after": 1010, "pot_after": 20},
        ]

        events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_when="tournament_heads_up",
            fire_duration_ticks=4,
        )

        self.assertEqual(events[0]["event"], "live_fire_start")
        self.assertEqual(events[0]["hand_id"], 2)
        self.assertEqual(events[0]["step"], 2)
        self.assertEqual(events[0]["start_condition"], "tournament_heads_up")


if __name__ == "__main__":
    unittest.main()
