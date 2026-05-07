"""Tests for the bring-your-own-agent layer (manifest, endpoint, tools)."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent

from poker_agents import (
    AggressiveAgent,
    CallingAgent,
    EndpointAgent,
    ManifestError,
    TightAgent,
    load_manifest,
    parse_manifest,
)
from poker_agents.base import AgentDecision, Observation
from tools.poker_run_tournament import run_competition
from tools.poker_validate_agent import validate_agent


class _EchoHandler(BaseHTTPRequestHandler):
    """Test server that returns a preset decision for every POST."""

    decision_payload = {"action": "call", "reasoning": "test"}

    def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = json.dumps(self.decision_payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002 - silence stderr
        return


class _BadJsonHandler(_EchoHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(length)
        body = b"not json at all"
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class ManifestLoaderTests(unittest.TestCase):
    def test_parse_manifest_builds_scripted_agents(self):
        document = {
            "agents": [
                {"agent_id": "t0", "seat_id": 0, "type": "scripted", "class": "TightAgent"},
                {"agent_id": "c1", "seat_id": 1, "type": "scripted", "class": "CallingAgent"},
            ],
            "tournament": {"starting_stack": 500, "num_hands": 5, "seeds": [1, 2]},
        }
        manifest = parse_manifest(document)
        agents = manifest.build_agents()

        self.assertIsInstance(agents[0], TightAgent)
        self.assertIsInstance(agents[1], CallingAgent)
        self.assertEqual(manifest.tournament.seeds, [1, 2])
        self.assertEqual(manifest.tournament.starting_stack, 500)

    def test_parse_manifest_reads_blind_escalation(self):
        document = {
            "agents": [
                {"agent_id": "t0", "seat_id": 0, "type": "scripted", "class": "TightAgent"},
                {"agent_id": "c1", "seat_id": 1, "type": "scripted", "class": "CallingAgent"},
            ],
            "tournament": {
                "starting_stack": 1000,
                "num_hands": 80,
                "small_blind": 5,
                "big_blind": 10,
                "blind_increase_every": 8,
                "blind_multiplier": 2,
                "max_big_blind": 160,
                "freeze_blinds_when_heads_up": True,
                "heads_up_small_blind": 160,
                "heads_up_big_blind": 320,
            },
        }
        manifest = parse_manifest(document)

        self.assertEqual(manifest.tournament.blind_increase_every, 8)
        self.assertEqual(manifest.tournament.blind_multiplier, 2.0)
        self.assertEqual(manifest.tournament.max_big_blind, 160)
        self.assertTrue(manifest.tournament.freeze_blinds_when_heads_up)
        self.assertEqual(manifest.tournament.heads_up_small_blind, 160)
        self.assertEqual(manifest.tournament.heads_up_big_blind, 320)

    def test_duplicate_seat_rejected(self):
        document = {
            "agents": [
                {"agent_id": "a", "seat_id": 0, "type": "scripted", "class": "TightAgent"},
                {"agent_id": "b", "seat_id": 0, "type": "scripted", "class": "CallingAgent"},
            ]
        }
        with self.assertRaises(ManifestError):
            parse_manifest(document)

    def test_unknown_class_rejected_on_build(self):
        document = {
            "agents": [
                {"agent_id": "x", "seat_id": 0, "type": "scripted", "class": "GhostAgent"},
                {"agent_id": "y", "seat_id": 1, "type": "scripted", "class": "TightAgent"},
            ]
        }
        manifest = parse_manifest(document)
        with self.assertRaises(ManifestError):
            manifest.build_agents()

    def test_load_manifest_from_yaml_file(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.yaml"
            path.write_text(
                dedent(
                    """
                    agents:
                      - agent_id: tight-0
                        seat_id: 0
                        type: scripted
                        class: TightAgent
                      - agent_id: aggro-1
                        seat_id: 1
                        type: scripted
                        class: AggressiveAgent
                    tournament:
                      starting_stack: 400
                      num_hands: 3
                      seeds: [7]
                    """
                ).strip()
            )
            manifest = load_manifest(path)
        self.assertEqual([spec.agent_id for spec in manifest.agents], ["tight-0", "aggro-1"])
        self.assertEqual(manifest.tournament.seeds, [7])


class EndpointAgentTests(unittest.TestCase):
    def _serve(self, handler_cls):
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def test_roundtrip_parses_decision(self):
        server, thread = self._serve(_EchoHandler)
        try:
            port = server.server_address[1]
            agent = EndpointAgent("http", endpoint=f"http://127.0.0.1:{port}/decide", timeout=2.0)
            observation = Observation(
                hand_id=1,
                street="preflop",
                seat_id=0,
                hole_cards=["Ah", "Kd"],
                board=[],
                pot=30,
                to_call=20,
                current_bet=20,
                min_raise_to=40,
                max_raise_to=1000,
                legal_actions=[
                    {"action": "fold", "min_amount": None, "max_amount": None, "to_call": 20},
                    {"action": "call", "min_amount": None, "max_amount": None, "to_call": 20},
                ],
                stacks={0: 1000, 1: 1000},
                committed={0: 0, 1: 20},
                folded_seats=[],
                all_in_seats=[],
                button_seat=0,
                big_blind=20,
                small_blind=10,
            )
            decision = agent.decide_action(observation)
            self.assertEqual(decision.action, "call")
            self.assertEqual(decision.reasoning, "test")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_invalid_json_falls_back_to_fold(self):
        server, thread = self._serve(_BadJsonHandler)
        try:
            port = server.server_address[1]
            agent = EndpointAgent("bad", endpoint=f"http://127.0.0.1:{port}/decide", timeout=2.0)
            observation = Observation(
                hand_id=1,
                street="preflop",
                seat_id=0,
                hole_cards=["Ah", "Kd"],
                board=[],
                pot=0,
                to_call=0,
                current_bet=0,
                min_raise_to=None,
                max_raise_to=None,
                legal_actions=[],
                stacks={0: 1000},
                committed={0: 0},
                folded_seats=[],
                all_in_seats=[],
                button_seat=0,
                big_blind=20,
                small_blind=10,
            )
            decision = agent.decide_action(observation)
            self.assertEqual(decision.action, "fold")
            self.assertIn("invalid json", decision.reasoning or "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)

    def test_unreachable_endpoint_folds(self):
        agent = EndpointAgent(
            "offline",
            endpoint="http://127.0.0.1:1/unreachable",
            timeout=0.5,
        )
        observation = Observation(
            hand_id=1,
            street="preflop",
            seat_id=0,
            hole_cards=[],
            board=[],
            pot=0,
            to_call=0,
            current_bet=0,
            min_raise_to=None,
            max_raise_to=None,
            legal_actions=[],
            stacks={},
            committed={},
            folded_seats=[],
            all_in_seats=[],
            button_seat=0,
            big_blind=20,
            small_blind=10,
        )
        decision = agent.decide_action(observation)
        self.assertEqual(decision.action, "fold")


class ValidationHarnessTests(unittest.TestCase):
    def test_scripted_agent_passes_validation(self):
        agent = AggressiveAgent("aggro")
        report = validate_agent(agent)
        self.assertTrue(report.passed, msg=report.to_dict())

    def test_invalid_agent_fails_validation(self):
        class AlwaysCheckAgent(CallingAgent):
            def decide_action(self, observation: Observation) -> AgentDecision:
                return AgentDecision(action="check")

        agent = AlwaysCheckAgent("bad")
        report = validate_agent(agent)
        self.assertFalse(report.passed)
        fails = [scenario for scenario in report.scenarios if not scenario.passed]
        self.assertTrue(fails)


class CompetitionRunnerTests(unittest.TestCase):
    def test_run_competition_aggregates_profits_across_seeds(self):
        manifest = parse_manifest(
            {
                "agents": [
                    {"agent_id": "t0", "seat_id": 0, "type": "scripted", "class": "TightAgent"},
                    {
                        "agent_id": "c1",
                        "seat_id": 1,
                        "type": "scripted",
                        "class": "CallingAgent",
                    },
                    {
                        "agent_id": "a2",
                        "seat_id": 2,
                        "type": "scripted",
                        "class": "AggressiveAgent",
                    },
                ],
                "tournament": {
                    "starting_stack": 300,
                    "num_hands": 4,
                    "small_blind": 5,
                    "big_blind": 10,
                    "seeds": [1, 2],
                },
            }
        )
        rows = run_competition(manifest)
        total_profit = sum(row.total_profit for row in rows)

        self.assertEqual(len(rows), 3)
        self.assertEqual(total_profit, 0)
        self.assertEqual(sum(row.wins for row in rows), 2)


if __name__ == "__main__":
    unittest.main()
