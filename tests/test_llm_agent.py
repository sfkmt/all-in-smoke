"""Tests for the Ollama-backed LLM agent."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from poker_agents import LlmAgent, parse_manifest
from poker_agents.base import Observation


CANNED_DECISION = {
    "action": "call",
    "amount": None,
    "confidence": 0.7,
    "reasoning": "A6sでコール",
    "memory": "opp raised preflop",
    "inner_voice": "スペAなら戦える",
    "psych": {"tilt": 0.1, "confidence_on_hand": 0.6, "reads": {"1": "As強め"}, "mood": "calm"},
    "table_talk": {"to": "all", "text": "Tdを見る"},
}


def _fake_observation() -> Observation:
    return Observation(
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


class _OllamaStubHandler(BaseHTTPRequestHandler):
    content = json.dumps(CANNED_DECISION)
    received_requests: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        type(self).received_requests.append(json.loads(raw))
        body = json.dumps(
            {
                "model": "qwen3.5:9b",
                "message": {"role": "assistant", "content": self.content},
                "done": True,
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


class _MalformedContentHandler(_OllamaStubHandler):
    content = "this is not json"


class LlmAgentTests(unittest.TestCase):
    def _serve(self, handler_cls):
        handler_cls.received_requests = []
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _shutdown(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_parses_canned_decision_and_sends_json_format(self):
        server, thread = self._serve(_OllamaStubHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "qwen",
                model="qwen3.5:9b",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
            )
            decision = agent.decide_action(_fake_observation())

            self.assertEqual(decision.action, "call")
            self.assertAlmostEqual(decision.confidence or 0.0, 0.7)
            self.assertEqual(decision.reasoning, "Aと6のスーテッドでコール")
            self.assertEqual(decision.inner_voice, "スペードAなら戦える")
            self.assertEqual(((decision.psych or {}).get("reads") or {}).get("1"), "スペードA強め")

            request = _OllamaStubHandler.received_requests[0]
            self.assertEqual(request["model"], "qwen3.5:9b")
            self.assertEqual(request["format"], "json")
            self.assertFalse(request["stream"])
            self.assertIs(request["think"], False)
            self.assertEqual(request["messages"][0]["role"], "system")
            self.assertIn("As", request["messages"][0]["content"])
            self.assertIn("スペードA", request["messages"][0]["content"])
            self.assertIn("inner_voice is NOT a hand explanation", request["messages"][0]["content"])
            self.assertIn("fear, hesitation, attachment", request["messages"][0]["content"])
            self.assertIn("Avoid poker terms like top pair", request["messages"][0]["content"])
            user_content = json.loads(request["messages"][1]["content"])
            self.assertEqual(user_content["hole_cards"], ["Ah", "Kd"])
            self.assertEqual(user_content["hole_cards_ja"], ["ハートA", "ダイヤK"])
            self.assertEqual(user_content["hole_cards_text"], "ハートA・ダイヤK")
            self.assertIn("As=スペードA", user_content["card_reading_note"])
            self.assertIn("略称", user_content["card_reading_note"])
            self.assertNotIn("seed", user_content)
        finally:
            self._shutdown(server, thread)

    def test_persona_disallows_table_talk(self):
        server, thread = self._serve(_OllamaStubHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "qwen",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
                persona="analyst",
            )
            decision = agent.decide_action(_fake_observation())
            self.assertEqual(decision.inner_voice, "スペードAなら戦える")
            self.assertIsNone(decision.table_talk)
            self.assertEqual((decision.psych or {}).get("mood"), "calm")
            self.assertFalse(agent.table_talk_allowed)
            self.assertIn("Analyst", agent.system_prompt)
        finally:
            self._shutdown(server, thread)

    def test_persona_allows_table_talk(self):
        server, thread = self._serve(_OllamaStubHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "qwen",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
                persona="needler",
            )
            decision = agent.decide_action(_fake_observation())
            self.assertEqual((decision.table_talk or {}).get("text"), "ダイヤ10を見る")
            self.assertTrue(agent.table_talk_allowed)
        finally:
            self._shutdown(server, thread)

    def test_prompt_includes_recent_table_talk(self):
        server, thread = self._serve(_OllamaStubHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "qwen",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
            )
            obs = _fake_observation()
            obs_with_talk = Observation(**{**obs.__dict__, "recent_table_talk": [
                {"seat_id": 1, "hand_id": 1, "street": "preflop", "to": "all", "text": "sit down"},
            ]})
            agent.decide_action(obs_with_talk)
            request = _OllamaStubHandler.received_requests[0]
            user_content = json.loads(request["messages"][1]["content"])
            self.assertEqual(user_content["recent_table_talk"][0]["text"], "sit down")
            self.assertEqual(user_content["recent_table_talk"][0]["from_seat"], 1)
        finally:
            self._shutdown(server, thread)

    def test_prompt_includes_identity_and_voice_profile_without_archetype(self):
        server, thread = self._serve(_OllamaStubHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "Soma(42)",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
                table_talk_allowed=False,
                identity_context={
                    "display_name": "Soma(42)",
                    "full_name": "Takase Soma",
                    "age": 42,
                    "gender": "male",
                },
                voice_profile={
                    "source": "timeql_agentspoker_lack_v1",
                    "primary_lack": "loss",
                    "inner_voice_directives": ["過去の負けと手放しづらさが出やすい。"],
                },
            )
            decision = agent.decide_action(_fake_observation())
            request = _OllamaStubHandler.received_requests[0]
            system_content = request["messages"][0]["content"]
            user_content = json.loads(request["messages"][1]["content"])

            self.assertIsNone(decision.table_talk)
            self.assertIn("voice_profile", system_content)
            self.assertIn("Do not force", system_content)
            self.assertIn("emotional pressure", system_content)
            self.assertIn("Put cards, board, ranges, odds", system_content)
            self.assertNotIn("Analyst", system_content)
            self.assertEqual(user_content["identity_context"]["full_name"], "Takase Soma")
            self.assertEqual(user_content["identity_context"]["age"], 42)
            self.assertEqual(user_content["voice_profile"]["primary_lack"], "loss")
        finally:
            self._shutdown(server, thread)

    def test_think_true_forwarded_to_top_level(self):
        server, thread = self._serve(_OllamaStubHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "qwen",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
                think=True,
            )
            agent.decide_action(_fake_observation())
            request = _OllamaStubHandler.received_requests[0]
            self.assertIs(request["think"], True)
            self.assertNotIn("think", request["options"])
        finally:
            self._shutdown(server, thread)

    def test_malformed_content_folds(self):
        server, thread = self._serve(_MalformedContentHandler)
        try:
            port = server.server_address[1]
            agent = LlmAgent(
                "qwen",
                endpoint=f"http://127.0.0.1:{port}/api/chat",
                timeout=2.0,
            )
            decision = agent.decide_action(_fake_observation())
            self.assertEqual(decision.action, "fold")
            self.assertIn("invalid json", decision.reasoning or "")
        finally:
            self._shutdown(server, thread)

    def test_unreachable_ollama_folds(self):
        agent = LlmAgent(
            "qwen",
            endpoint="http://127.0.0.1:1/api/chat",
            timeout=0.5,
        )
        decision = agent.decide_action(_fake_observation())
        self.assertEqual(decision.action, "fold")
        self.assertIn("unreachable", decision.reasoning or "")


class LlmManifestTests(unittest.TestCase):
    def test_manifest_builds_llm_agent_with_defaults(self):
        manifest = parse_manifest(
            {
                "agents": [
                    {
                        "agent_id": "qwen-0",
                        "seat_id": 0,
                        "type": "llm",
                        "model": "qwen3.5:9b",
                        "endpoint": "http://localhost:11434/api/chat",
                        "timeout": 45.0,
                        "temperature": 0.5,
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
        agents = manifest.build_agents()
        llm = agents[0]

        self.assertIsInstance(llm, LlmAgent)
        self.assertEqual(llm.model, "qwen3.5:9b")
        self.assertEqual(llm.endpoint, "http://localhost:11434/api/chat")
        self.assertEqual(llm.timeout, 45.0)
        self.assertEqual(llm.temperature, 0.5)
        self.assertFalse(llm.think)

    def test_manifest_forwards_persona(self):
        manifest = parse_manifest(
            {
                "agents": [
                    {
                        "agent_id": "analyst-0",
                        "seat_id": 0,
                        "type": "llm",
                        "persona": "analyst",
                    },
                    {
                        "agent_id": "needler-1",
                        "seat_id": 1,
                        "type": "llm",
                        "persona": "needler",
                    },
                ]
            }
        )
        agents = manifest.build_agents()
        self.assertFalse(agents[0].table_talk_allowed)
        self.assertTrue(agents[1].table_talk_allowed)
        self.assertIn("Analyst", agents[0].system_prompt)
        self.assertIn("Needler", agents[1].system_prompt)

    def test_manifest_forwards_think_flag(self):
        manifest = parse_manifest(
            {
                "agents": [
                    {
                        "agent_id": "qwen-0",
                        "seat_id": 0,
                        "type": "llm",
                        "think": True,
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
        llm = manifest.build_agents()[0]
        self.assertTrue(llm.think)

    def test_manifest_forwards_table_talk_and_voice_context(self):
        manifest = parse_manifest(
            {
                "agents": [
                    {
                        "agent_id": "Ren(31)",
                        "seat_id": 0,
                        "type": "llm",
                        "table_talk_allowed": False,
                        "full_name": "Mizuno Ren",
                        "gender": "male",
                        "age": 31,
                        "voice_profile": {
                            "source": "timeql_agentspoker_lack_v1",
                            "primary_lack": "protection",
                        },
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
        llm = manifest.build_agents()[0]
        self.assertFalse(llm.table_talk_allowed)
        self.assertEqual(llm.agent_context["identity_context"]["full_name"], "Mizuno Ren")
        self.assertEqual(llm.agent_context["identity_context"]["age"], 31)
        self.assertEqual(llm.agent_context["voice_profile"]["primary_lack"], "protection")


if __name__ == "__main__":
    unittest.main()
