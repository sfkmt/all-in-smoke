"""Tests for the OpenRouter-backed poker agent."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from poker_agents import OpenRouterAgent, parse_manifest
from poker_agents.base import Observation


CANNED_DECISION = {
    "action": "call",
    "amount": None,
    "confidence": 0.66,
    "reasoning": "A6sでコールする",
    "memory": None,
    "inner_voice": "スペAならまだ戦える。",
    "psych": {"tilt": 0.1, "confidence_on_hand": 0.7, "reads": {"1": "As警戒"}, "mood": "冷静"},
    "table_talk": {"to": "all", "text": "Tdも見る。"},
}


def _fake_observation() -> Observation:
    return Observation(
        hand_id=1,
        street="preflop",
        seat_id=0,
        hole_cards=["As", "6d"],
        board=[],
        pot=30,
        to_call=20,
        current_bet=20,
        min_raise_to=40,
        max_raise_to=1000,
        legal_actions=[
            {"action": "fold", "min_amount": None, "max_amount": None, "to_call": 20},
            {"action": "call", "min_amount": None, "max_amount": None, "to_call": 20},
            {"action": "raise", "min_amount": 40, "max_amount": 1000, "to_call": 20},
        ],
        stacks={0: 1000, 1: 1000},
        committed={0: 0, 1: 20},
        folded_seats=[],
        all_in_seats=[],
        button_seat=0,
        big_blind=20,
        small_blind=10,
    )


class _OpenRouterStubHandler(BaseHTTPRequestHandler):
    content = json.dumps(CANNED_DECISION, ensure_ascii=False)
    received_requests = []
    received_auth = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        type(self).received_requests.append(json.loads(raw))
        type(self).received_auth.append(self.headers.get("Authorization"))
        body = json.dumps(
            {
                "choices": [
                    {"message": {"role": "assistant", "content": self.content}},
                ],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15},
            },
            ensure_ascii=False,
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


class OpenRouterAgentTests(unittest.TestCase):
    def _serve(self, handler_cls):
        handler_cls.received_requests = []
        handler_cls.received_auth = []
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _shutdown(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_parses_openrouter_decision_and_sends_json_request(self):
        server, thread = self._serve(_OpenRouterStubHandler)
        try:
            port = server.server_address[1]
            agent = OpenRouterAgent(
                "grok",
                endpoint=f"http://127.0.0.1:{port}/api/v1/chat/completions",
                api_key_env="OPENROUTER_TEST_KEY",
                timeout=2.0,
                persona="needler",
            )
            with patch.dict("os.environ", {"OPENROUTER_TEST_KEY": "or-test"}):
                decision = agent.decide_action(_fake_observation())

            self.assertEqual(decision.action, "call")
            self.assertEqual(decision.inner_voice, "スペードAならまだ戦える。")
            self.assertEqual(decision.reasoning, "Aと6のスーテッドでコールする")
            self.assertEqual((decision.table_talk or {}).get("text"), "ダイヤ10も見る。")
            self.assertEqual(((decision.psych or {}).get("reads") or {}).get("1"), "スペードA警戒")
            self.assertEqual((_OpenRouterStubHandler.received_auth[0]), "Bearer or-test")
            request = _OpenRouterStubHandler.received_requests[0]
            self.assertEqual(request["model"], "x-ai/grok-4.1-fast")
            self.assertEqual(request["response_format"], {"type": "json_object"})
            self.assertEqual(request["reasoning"], {"enabled": False})
            user_content = json.loads(request["messages"][1]["content"])
            self.assertEqual(user_content["hole_cards_ja"], ["スペードA", "ダイヤ6"])
        finally:
            self._shutdown(server, thread)

    def test_missing_api_key_folds(self):
        agent = OpenRouterAgent("grok", api_key_env="OPENROUTER_TEST_KEY")
        with patch.dict("os.environ", {}, clear=True):
            decision = agent.decide_action(_fake_observation())
        self.assertEqual(decision.action, "fold")
        self.assertIn("api key missing", decision.reasoning or "")


class OpenRouterManifestTests(unittest.TestCase):
    def test_manifest_builds_openrouter_agent(self):
        manifest = parse_manifest(
            {
                "agents": [
                    {
                        "agent_id": "grok-0",
                        "seat_id": 0,
                        "type": "openrouter",
                        "model": "x-ai/grok-4.1-fast",
                        "api_key_env": "OPENROUTER_API_KEY",
                        "timeout": 20,
                        "temperature": 0.2,
                        "think": False,
                        "persona": "stoic",
                    },
                    {
                        "agent_id": "tight-1",
                        "seat_id": 1,
                        "type": "scripted",
                        "class": "TightAgent",
                    },
                ]
            }
        )
        agent = manifest.build_agents()[0]
        self.assertIsInstance(agent, OpenRouterAgent)
        self.assertEqual(agent.model, "x-ai/grok-4.1-fast")
        self.assertEqual(agent.api_key_env, "OPENROUTER_API_KEY")
        self.assertFalse(agent.reasoning_enabled)
        self.assertFalse(agent.table_talk_allowed)


if __name__ == "__main__":
    unittest.main()
