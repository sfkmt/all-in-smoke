"""Tests for the god-view commentator pipeline (no live LLM dependency)."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory

from poker_agents.commentator import (
    build_payload_for_step,
    call_commentator,
    hand_category,
    preflop_label,
)
from tools.poker_commentator import (
    board_for_street,
    generate_commentary,
    index_events,
    parse_jsonl,
    write_jsonl,
)


class EquityHelpers(unittest.TestCase):
    def test_preflop_label_pairs_and_suited(self):
        self.assertEqual(preflop_label(["Ah", "Ad"]), "AA")
        self.assertEqual(preflop_label(["Ah", "Kh"]), "AKs")
        self.assertEqual(preflop_label(["Ah", "Kd"]), "AKo")
        self.assertEqual(preflop_label(["7c", "2d"]), "72o")  # canonical high-low order

    def test_hand_category_none_preflop(self):
        self.assertIsNone(hand_category(["Ah", "Kd"], []))

    def test_hand_category_post_flop(self):
        cat = hand_category(["Ah", "Ad"], ["Ac", "5d", "9h"])
        self.assertIsNotNone(cat)
        self.assertEqual(cat["name"], "three_of_a_kind")
        self.assertEqual(cat["category"], 3)


class PayloadConstruction(unittest.TestCase):
    def _hand_start(self):
        return {
            "hand_id": 1,
            "hole_cards": {0: ["Ah", "Kd"], 1: ["Qc", "Qs"], 2: ["7h", "2c"]},
            "stacks": {0: 1000, 1: 1000, 2: 1000},
            "button_seat": 0, "small_blind": 5, "big_blind": 10,
        }

    def test_god_view_includes_all_hole_cards_and_categories(self):
        action = {
            "step": 5, "hand_id": 1, "seat_id": 1, "agent_id": "needler-1",
            "street": "flop", "action": "raise", "amount": 80,
            "to_call_before": 0, "stack_after": 920, "pot_after": 110,
            "decision": {"inner_voice": "set だ、強気で", "reasoning": "set", "psych": {"mood": "calm"}},
        }
        prior_actions = [
            {"step": 1, "seat_id": 0, "agent_id": "tight-0", "street": "preflop", "action": "raise", "amount": 30},
            {"step": 2, "seat_id": 1, "agent_id": "needler-1", "street": "preflop", "action": "call", "amount": 30},
            {"step": 3, "seat_id": 2, "agent_id": "showman-2", "street": "preflop", "action": "fold", "amount": None},
        ]
        prior_reasonings = {
            1: {"step": 1, "seat_id": 0, "agent_id": "tight-0", "inner_voice": "AK は強い", "psych": {"mood": "calm"}},
            2: {"step": 2, "seat_id": 1, "agent_id": "needler-1", "inner_voice": "QQ で trap", "psych": {"mood": "calm"}},
        }
        payload = build_payload_for_step(
            action_event=action,
            hand_start=self._hand_start(),
            prior_actions=prior_actions,
            prior_reasonings=prior_reasonings,
            seat_to_agent_id={0: "tight-0", 1: "needler-1", 2: "showman-2"},
            pot_after=110,
            board_now=["Qd", "5c", "2s"],
        )
        contender_seats = sorted(c["seat"] for c in payload["contenders_god_view"])
        # seat 2 folded → excluded
        self.assertEqual(contender_seats, [0, 1])
        seat1 = next(c for c in payload["contenders_god_view"] if c["seat"] == 1)
        # QQ on Q52 → set of queens
        self.assertEqual(seat1["current_hand_category"]["name"], "three_of_a_kind")
        self.assertEqual(seat1["preflop_label"], "QQ")
        # current action surface
        self.assertEqual(payload["current_action"]["agent_id"], "needler-1")
        self.assertEqual(payload["current_action"]["inner_voice"], "set だ、強気で")
        # inner voices included
        ids = [v["agent_id"] for v in payload["recent_inner_voices"]]
        self.assertIn("tight-0", ids)
        self.assertIn("needler-1", ids)


class StreetBoardSlicing(unittest.TestCase):
    def test_board_for_street(self):
        full = ["2c", "3d", "4h", "5s", "7c"]
        self.assertEqual(board_for_street("preflop", full), [])
        self.assertEqual(board_for_street("flop", full), full[:3])
        self.assertEqual(board_for_street("turn", full), full[:4])
        self.assertEqual(board_for_street("river", full), full)
        self.assertEqual(board_for_street("showdown", full), full)


# ---------------------------------------------------------------------------
# End-to-end with stub Ollama
# ---------------------------------------------------------------------------

class _StubHandler(BaseHTTPRequestHandler):
    received: list = []
    canned: str = "今、needler のレイズきたー！QQ でセット完成、これは強い！"

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        type(self).received.append(json.loads(raw))
        body = json.dumps(
            {"model": "stub", "message": {"role": "assistant", "content": self.canned}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


class CommentatorEndToEnd(unittest.TestCase):
    def _serve(self):
        _StubHandler.received = []
        server = HTTPServer(("127.0.0.1", 0), _StubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _shutdown(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_call_commentator_returns_text(self):
        server, thread = self._serve()
        try:
            text = call_commentator(
                {"hello": "world"},
                endpoint=f"http://127.0.0.1:{server.server_address[1]}/api/chat",
                timeout=2.0,
            )
            self.assertEqual(text, _StubHandler.canned)
            self.assertEqual(len(_StubHandler.received), 1)
        finally:
            self._shutdown(server, thread)

    def test_generate_commentary_writes_one_event_per_action(self):
        server, thread = self._serve()
        try:
            events = [
                {"event": "hand_start", "hand_id": 1,
                 "hole_cards": {0: ["Ah", "Kd"], 1: ["Qc", "Qs"]},
                 "stacks": {0: 1000, 1: 1000}, "button_seat": 0,
                 "small_blind": 5, "big_blind": 10},
                {"event": "action", "step": 1, "hand_id": 1, "seat_id": 0,
                 "agent_id": "tight-0", "street": "preflop", "action": "raise",
                 "amount": 30, "pot_after": 35, "to_call_before": 5, "stack_after": 970,
                 "decision": {"inner_voice": "AK 強気"}},
                {"event": "action", "step": 2, "hand_id": 1, "seat_id": 1,
                 "agent_id": "needler-1", "street": "preflop", "action": "call",
                 "amount": None, "pot_after": 65, "to_call_before": 30, "stack_after": 970,
                 "decision": {"inner_voice": "QQ trap"}},
                {"event": "hand_result", "hand_id": 1, "board": ["Qd", "5c", "2s", "9h", "Jd"]},
            ]
            commentary = generate_commentary(
                events,
                model="stub",
                endpoint=f"http://127.0.0.1:{server.server_address[1]}/api/chat",
                timeout=2.0,
                temperature=0.5,
                progress=False,
            )
            self.assertEqual(len(commentary), 2)
            self.assertEqual([c["step"] for c in commentary], [1, 2])
            for c in commentary:
                self.assertEqual(c["event"], "commentary")
                self.assertEqual(c["text"], _StubHandler.canned)
        finally:
            self._shutdown(server, thread)

    def test_jsonl_roundtrip(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "c.jsonl"
            write_jsonl(path, [{"event": "commentary", "step": 1, "hand_id": 1, "text": "yo"}])
            parsed = parse_jsonl(path)
            self.assertEqual(parsed[0]["text"], "yo")


class IndexEvents(unittest.TestCase):
    def test_index_events_groups_by_kind(self):
        events = [
            {"event": "hand_start", "hand_id": 1, "hole_cards": {}},
            {"event": "action", "step": 2, "hand_id": 1, "seat_id": 0, "agent_id": "a"},
            {"event": "action", "step": 1, "hand_id": 1, "seat_id": 1, "agent_id": "b"},
            {"event": "memory_reasoning", "step": 1, "seat_id": 1, "inner_voice": "x"},
            {"event": "hand_result", "hand_id": 1, "board": ["Ah"]},
        ]
        starts, boards, actions, reasonings = index_events(events)
        self.assertIn(1, starts)
        self.assertEqual(boards[1], ["Ah"])
        self.assertEqual([a["step"] for a in actions], [1, 2])
        self.assertIn(1, reasonings)


if __name__ == "__main__":
    unittest.main()
