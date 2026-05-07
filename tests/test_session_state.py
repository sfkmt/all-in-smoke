"""Tests for SessionState (long-term memory + session-wide tilt)."""

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from poker_agents import AggressiveAgent, CallingAgent, LlmAgent, TightAgent
from poker_agents.base import Observation
from poker_agents.session_state import SessionState
from poker_simulation import run_tournament


# Mirrors poker_engine.hand_evaluator.CATEGORY_NAMES (int → label).
_CATEGORY = {
    "high_card": 0, "one_pair": 1, "two_pair": 2, "three_of_a_kind": 3,
    "straight": 4, "flush": 5, "full_house": 6, "four_of_a_kind": 7,
    "straight_flush": 8,
}


def _rank_dict(name: str) -> dict:
    return {"category": _CATEGORY[name], "name": name, "tiebreakers": [], "cards": []}


def _showdown_result_factory(
    *,
    hand_id: int,
    own_seat: int,
    own_payout: int,
    opp_seat: int,
    opp_payout: int,
    own_committed: int,
    opp_committed: int,
    own_category: str = "two_pair",
    opp_category: str = "three_of_a_kind",
):
    return {
        "hand_id": hand_id,
        "winners": [opp_seat] if opp_payout > own_payout else [own_seat],
        "payouts": {own_seat: own_payout, opp_seat: opp_payout},
        "final_stacks": {own_seat: 1000 - own_committed + own_payout, opp_seat: 1000 - opp_committed + opp_payout},
        "board": ["2c", "3d", "4h", "5s", "7c"],
        "street": "showdown",
        "action_log": [
            {"seat_id": own_seat, "street": "preflop", "action": "raise", "amount": 30, "contributed": 30},
            {"seat_id": opp_seat, "street": "preflop", "action": "call", "amount": 30, "contributed": 30},
            {"seat_id": own_seat, "street": "river", "action": "bet", "amount": own_committed - 30, "contributed": own_committed - 30},
            {"seat_id": opp_seat, "street": "river", "action": "raise", "amount": opp_committed - 30, "contributed": opp_committed - 30},
        ],
        "showdown": {
            "hand_ranks": {
                str(own_seat): _rank_dict(own_category),
                str(opp_seat): _rank_dict(opp_category),
            },
        },
    }


class SessionStateTilt(unittest.TestCase):
    def test_tilt_starts_zero(self):
        s = SessionState(agent_id="me", own_seat=0)
        self.assertEqual(s.tilt_factor(), 0.0)
        self.assertEqual(s.mood_label(), "落ち着いている")

    def test_tilt_rises_after_loss_and_decays(self):
        s = SessionState(agent_id="me", own_seat=0)
        seat_to_id = {0: "me", 1: "needler"}

        result = _showdown_result_factory(
            hand_id=1, own_seat=0, opp_seat=1,
            own_committed=200, opp_committed=200,
            own_payout=0, opp_payout=400,
            own_category="two_pair", opp_category="three_of_a_kind",
        )
        s.ingest_hand_result(result, seat_to_agent_id=seat_to_id, big_blind=10)
        tilt_after_loss = s.tilt_factor()
        self.assertGreater(tilt_after_loss, 0.3, "loss + bad-beat bonus should push tilt up")

        # Subsequent break-even hand should decay tilt without adding loss.
        breakeven = _showdown_result_factory(
            hand_id=2, own_seat=0, opp_seat=1,
            own_committed=10, opp_committed=10,
            own_payout=10, opp_payout=10,
            own_category="high_card", opp_category="high_card",
        )
        s.ingest_hand_result(breakeven, seat_to_agent_id=seat_to_id, big_blind=10)
        self.assertLess(s.tilt_factor(), tilt_after_loss)

    def test_tilt_eases_after_big_win(self):
        s = SessionState(agent_id="me", own_seat=0, tilt=0.5)
        seat_to_id = {0: "me", 1: "needler"}
        win = _showdown_result_factory(
            hand_id=1, own_seat=0, opp_seat=1,
            own_committed=100, opp_committed=200,
            own_payout=300, opp_payout=0,
            own_category="full_house", opp_category="two_pair",
        )
        s.ingest_hand_result(win, seat_to_agent_id=seat_to_id, big_blind=10)
        self.assertLess(s.tilt_factor(), 0.5)


class SessionStateRivalries(unittest.TestCase):
    def test_showdown_loss_creates_shown_strong_note(self):
        s = SessionState(agent_id="me", own_seat=0)
        seat_to_id = {0: "me", 1: "needler"}
        result = _showdown_result_factory(
            hand_id=1, own_seat=0, opp_seat=1,
            own_committed=200, opp_committed=200,
            own_payout=0, opp_payout=400,
            own_category="one_pair", opp_category="three_of_a_kind",
        )
        s.ingest_hand_result(result, seat_to_agent_id=seat_to_id, big_blind=10)
        notes = s.rivalries["needler"]
        self.assertEqual(notes[-1].kind, "shown_strong")
        self.assertIn("three_of_a_kind", notes[-1].text)

    def test_fold_win_creates_bluff_suspect_note(self):
        s = SessionState(agent_id="me", own_seat=0)
        seat_to_id = {0: "me", 1: "showman"}
        fold_win = {
            "hand_id": 5,
            "winners": [1],
            "payouts": {0: 0, 1: 60},
            "final_stacks": {0: 970, 1: 1030},
            "board": ["2c", "3d", "4h", "Kh"],
            "street": "river",
            "action_log": [
                {"seat_id": 0, "street": "preflop", "action": "call", "amount": 10, "contributed": 10},
                {"seat_id": 1, "street": "preflop", "action": "raise", "amount": 30, "contributed": 30},
                {"seat_id": 0, "street": "preflop", "action": "call", "amount": 30, "contributed": 20},
                {"seat_id": 1, "street": "river", "action": "bet", "amount": 100, "contributed": 100},
                {"seat_id": 0, "street": "river", "action": "fold", "amount": 0, "contributed": 0},
            ],
            "showdown": None,
        }
        s.ingest_hand_result(fold_win, seat_to_agent_id=seat_to_id, big_blind=10)
        notes = s.rivalries["showman"]
        self.assertEqual(notes[-1].kind, "bluffed_me")
        self.assertIn("river", notes[-1].text)

    def test_prompt_block_hides_raw_tilt_number(self):
        s = SessionState(agent_id="me", own_seat=0, tilt=0.7)
        s.recent_outcomes.append(__import__("poker_agents.session_state", fromlist=["Outcome"]).Outcome(
            hand_id=1, delta=-100, showdown=False, note="大きく負け",
        ))
        block = s.prompt_block()
        self.assertIsNotNone(block)
        rendered = json.dumps(block, ensure_ascii=False)
        self.assertNotIn("0.7", rendered)
        self.assertNotIn("tilt", rendered)
        self.assertIn("ティルト", rendered)  # mood label leaks vibe, not number

    def test_prompt_block_returns_none_when_fresh(self):
        s = SessionState(agent_id="me", own_seat=0)
        self.assertIsNone(s.prompt_block())


# ---------------------------------------------------------------------------
# LlmAgent integration: session_context appears in user prompt after one hand
# ---------------------------------------------------------------------------

CANNED_DECISION = {
    "action": "call",
    "amount": None,
    "confidence": 0.5,
    "reasoning": "stub",
    "memory": None,
    "inner_voice": "stub",
    "psych": None,
    "table_talk": None,
}


class _StubHandler(BaseHTTPRequestHandler):
    received_requests: list = []

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)
        type(self).received_requests.append(json.loads(raw))
        body = json.dumps(
            {"model": "stub", "message": {"role": "assistant", "content": json.dumps(CANNED_DECISION)}}
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # noqa: A002
        return


def _fake_obs(seat: int = 0) -> Observation:
    return Observation(
        hand_id=2, street="preflop", seat_id=seat, hole_cards=["Ah", "Kd"], board=[],
        pot=30, to_call=20, current_bet=20, min_raise_to=40, max_raise_to=1000,
        legal_actions=[
            {"action": "fold", "min_amount": None, "max_amount": None, "to_call": 20},
            {"action": "call", "min_amount": None, "max_amount": None, "to_call": 20},
        ],
        stacks={0: 1000, 1: 1000}, committed={0: 0, 1: 20},
        folded_seats=[], all_in_seats=[],
        button_seat=0, big_blind=20, small_blind=10,
    )


class LlmAgentSessionContext(unittest.TestCase):
    def _serve(self):
        _StubHandler.received_requests = []
        server = HTTPServer(("127.0.0.1", 0), _StubHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        return server, thread

    def _shutdown(self, server, thread):
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)

    def test_session_context_absent_when_fresh(self):
        server, thread = self._serve()
        try:
            agent = LlmAgent(
                "qwen", endpoint=f"http://127.0.0.1:{server.server_address[1]}/api/chat",
                timeout=2.0,
            )
            agent.decide_action(_fake_obs())
            user_content = json.loads(_StubHandler.received_requests[0]["messages"][1]["content"])
            self.assertNotIn("session_context", user_content)
        finally:
            self._shutdown(server, thread)

    def test_session_context_appears_after_hand_end(self):
        server, thread = self._serve()
        try:
            agent = LlmAgent(
                "qwen", endpoint=f"http://127.0.0.1:{server.server_address[1]}/api/chat",
                timeout=2.0, persona="needler",
            )
            # Simulate one prior hand: assign seat then ingest a fold-win loss.
            agent.on_hand_start(_fake_obs(seat=0))
            agent.on_hand_end({
                "hand_id": 1,
                "winners": [1], "payouts": {0: 0, 1: 80},
                "final_stacks": {0: 940, 1: 1060},
                "board": ["2c", "3d", "4h", "Kh"],
                "street": "river",
                "action_log": [
                    {"seat_id": 1, "street": "river", "action": "bet", "amount": 60, "contributed": 60},
                    {"seat_id": 0, "street": "river", "action": "fold", "amount": 0, "contributed": 60},
                ],
                "showdown": None,
                "seat_to_agent_id": {0: "qwen", 1: "rival-bot"},
                "big_blind": 20,
            })
            agent.decide_action(_fake_obs(seat=0))
            user_content = json.loads(_StubHandler.received_requests[0]["messages"][1]["content"])
            self.assertIn("session_context", user_content)
            ctx = user_content["session_context"]
            self.assertEqual(len(ctx["rivalries"]), 1)
            self.assertEqual(ctx["rivalries"][0]["opponent"], "rival-bot")
            self.assertEqual(ctx["rivalries"][0]["kind"], "bluffed_me")
        finally:
            self._shutdown(server, thread)


# ---------------------------------------------------------------------------
# Scripted agents react to tilt
# ---------------------------------------------------------------------------

class ScriptedAgentTiltReactions(unittest.TestCase):
    def test_aggressive_agent_increases_raise_size_when_tilted(self):
        agent = AggressiveAgent("aggro")
        obs = Observation(
            hand_id=1, street="flop", seat_id=0, hole_cards=["7h", "2d"],
            board=["Kc", "Qd", "Js"],
            pot=100, to_call=0, current_bet=0,
            min_raise_to=20, max_raise_to=400,
            legal_actions=[
                {"action": "check", "min_amount": None, "max_amount": None, "to_call": 0},
                {"action": "bet", "min_amount": 20, "max_amount": 400, "to_call": 0},
            ],
            stacks={0: 400, 1: 400}, committed={0: 0, 1: 0},
            folded_seats=[], all_in_seats=[],
            button_seat=0, big_blind=10, small_blind=5,
        )
        calm = agent.decide_action(obs)
        self.assertEqual(calm.amount, 20)

        agent.session.tilt = 1.0
        steamed = agent.decide_action(obs)
        self.assertEqual(steamed.amount, 400)
        self.assertIn("tilted", steamed.reasoning)

    def test_tight_agent_loosens_when_tilted(self):
        agent = TightAgent("nit")
        obs = Observation(
            hand_id=1, street="preflop", seat_id=0,
            hole_cards=["7h", "2d"],  # trash, would normally fold
            board=[],
            pot=30, to_call=20, current_bet=20,
            min_raise_to=40, max_raise_to=400,
            legal_actions=[
                {"action": "fold", "min_amount": None, "max_amount": None, "to_call": 20},
                {"action": "call", "min_amount": None, "max_amount": None, "to_call": 20},
                {"action": "raise", "min_amount": 40, "max_amount": 400, "to_call": 20},
            ],
            stacks={0: 400, 1: 400}, committed={0: 0, 1: 20},
            folded_seats=[], all_in_seats=[],
            button_seat=0, big_blind=20, small_blind=10,
        )
        calm = agent.decide_action(obs)
        self.assertEqual(calm.action, "fold")

        agent.session.tilt = 0.6
        steamed = agent.decide_action(obs)
        self.assertIn(steamed.action, {"call", "raise"})
        self.assertIn("tilted", steamed.reasoning or "")


# ---------------------------------------------------------------------------
# End-to-end: tournament loop populates session memory across hands
# ---------------------------------------------------------------------------

class SessionSnapshotLogging(unittest.TestCase):
    def test_session_snapshot_events_land_in_jsonl(self):
        import io
        from poker_simulation import JsonlLogger, run_hand

        agents = {0: TightAgent("tight"), 1: AggressiveAgent("aggro"), 2: CallingAgent("call")}
        buffer = io.StringIO()
        logger = JsonlLogger(stream=buffer)
        run_hand(
            agents,
            hand_id=1,
            stacks={0: 200, 1: 200, 2: 200},
            button_seat=0,
            small_blind=5, big_blind=10,
            seed=7, logger=logger,
        )
        snaps = [
            json.loads(line) for line in buffer.getvalue().splitlines()
            if json.loads(line).get("event") == "session_snapshot"
        ]
        self.assertEqual(len(snaps), 3)
        seats = sorted(s["seat_id"] for s in snaps)
        self.assertEqual(seats, [0, 1, 2])
        for s in snaps:
            snap = s["snapshot"]
            self.assertIn("mood", snap)
            self.assertIn("tilt", snap)
            self.assertIn("rivalries", snap)
            self.assertIn("recent_outcomes", snap)


class SessionPersistsAcrossTournament(unittest.TestCase):
    def test_agents_carry_session_across_hands(self):
        agents = {0: TightAgent("tight"), 1: AggressiveAgent("aggro"), 2: CallingAgent("call")}
        run_tournament(
            agents,
            starting_stacks={0: 300, 1: 300, 2: 300},
            num_hands=8, small_blind=5, big_blind=10,
            button_seat=0, seed=11,
        )
        # At least one agent should have played a few hands and accumulated state.
        played = [a.session.hands_played for a in agents.values()]
        self.assertTrue(any(n >= 4 for n in played))
        # Own seat must be remembered after on_hand_start.
        for seat, agent in agents.items():
            if agent.session.hands_played > 0:
                self.assertEqual(agent.session.own_seat, seat)


if __name__ == "__main__":
    unittest.main()
