"""Local LLM agent backed by Ollama's chat API (default: Qwen 3.5 9B)."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from poker_agents.base import AgentDecision, BaseAgent, Observation
from poker_agents.card_language import normalize_decision_card_language
from poker_agents.personas import AGENT_CONTEXT_RULES, PERSONAS, SCHEMA_RULES, get_persona
from poker_agents.session_state import SessionState
from poker_agents.voice_profile import build_agent_context
from poker_engine.cards import cards_text_japanese, cards_to_japanese


DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_TIMEOUT = 30.0
DEFAULT_TEMPERATURE = 0.6
ACTION_HISTORY_WINDOW = 12
TABLE_TALK_WINDOW = 16


DEFAULT_SYSTEM_PROMPT = (
    "あなたはテキサスホールデムをマインドスポーツとして戦う日本語話者のポーカー AI です。"
    + AGENT_CONTEXT_RULES
    + SCHEMA_RULES
)


def _format_observation(
    observation: Observation,
    session: Optional[SessionState] = None,
    agent_context: Optional[Dict[str, Any]] = None,
) -> str:
    history = observation.action_history[-ACTION_HISTORY_WINDOW:]
    compact_history = [
        {
            "seat": entry.get("seat_id"),
            "street": entry.get("street"),
            "action": entry.get("action"),
            "amount": entry.get("amount"),
        }
        for entry in history
    ]
    talk = observation.recent_table_talk[-TABLE_TALK_WINDOW:]
    compact_talk = [
        {
            "from_seat": entry.get("seat_id"),
            "street": entry.get("street"),
            "to": entry.get("to"),
            "text": entry.get("text"),
        }
        for entry in talk
    ]
    payload = {
        "hand_id": observation.hand_id,
        "street": observation.street,
        "seat_id": observation.seat_id,
        "hole_cards": observation.hole_cards,
        "hole_cards_ja": cards_to_japanese(observation.hole_cards),
        "hole_cards_text": cards_text_japanese(observation.hole_cards),
        "board": observation.board,
        "board_ja": cards_to_japanese(observation.board),
        "board_text": cards_text_japanese(observation.board),
        "card_reading_note": (
            "カードを日本語で言う時は *_ja / *_text を使う。"
            "raw code を発音風にしない。例: As=スペードA, Ac=クラブA, "
            "Ah=ハートA, Ad=ダイヤA, Td=ダイヤ10。"
            "スペAのような略称やA6sのような英数字コンボは使わない。"
        ),
        "pot": observation.pot,
        "to_call": observation.to_call,
        "current_bet": observation.current_bet,
        "min_raise_to": observation.min_raise_to,
        "max_raise_to": observation.max_raise_to,
        "stacks": observation.stacks,
        "committed": observation.committed,
        "folded_seats": observation.folded_seats,
        "all_in_seats": observation.all_in_seats,
        "button_seat": observation.button_seat,
        "big_blind": observation.big_blind,
        "small_blind": observation.small_blind,
        "legal_actions": observation.legal_actions,
        "recent_actions": compact_history,
        "recent_table_talk": compact_talk,
    }
    if isinstance(agent_context, dict):
        identity_context = agent_context.get("identity_context")
        voice_profile = agent_context.get("voice_profile")
        if isinstance(identity_context, dict):
            payload["identity_context"] = identity_context
        if isinstance(voice_profile, dict):
            payload["voice_profile"] = voice_profile
    if session is not None:
        block = session.prompt_block()
        if block is not None:
            payload["session_context"] = block
    return json.dumps(payload, sort_keys=True)


class LlmAgent(BaseAgent):
    """Delegates decisions to an Ollama-hosted chat model.

    Uses Ollama's `format: "json"` option to constrain output. Any transport,
    timeout, or parse failure folds with a reasoning string; the simulation
    runner's safe-fallback then downgrades to call/check as appropriate.
    """

    def __init__(
        self,
        agent_id: str,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = None,
        persona: Optional[str] = None,
        table_talk_allowed: Optional[bool] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        voice_profile: Optional[Dict[str, Any]] = None,
        think: bool = False,
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id)
        self.model = model
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        if persona is not None:
            persona_obj = get_persona(persona)
            self.persona = persona_obj
            self.system_prompt = system_prompt or persona_obj.system_prompt
            self.table_talk_allowed = persona_obj.table_talk_allowed
        else:
            self.persona = None
            self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
            self.table_talk_allowed = True
        if table_talk_allowed is not None:
            self.table_talk_allowed = bool(table_talk_allowed)
        self.agent_context = build_agent_context(
            agent_id=agent_id,
            metadata={},
            identity_context=identity_context,
            voice_profile=voice_profile,
        )
        # Ollama's `think` is a top-level request field, not an option.
        # Default off: reasoning-on-by-default models (e.g. qwen3.5) emit
        # thousands of thinking tokens per decision and blow the timeout.
        self.think = bool(think)
        self.extra_options = dict(extra_options or {})

    def _build_messages(self, observation: Observation) -> List[Dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _format_observation(
                    observation,
                    self.session,
                    agent_context=self.agent_context,
                ),
            },
        ]

    def _request_body(self, observation: Observation) -> bytes:
        options: Dict[str, Any] = {"temperature": self.temperature}
        options.update(self.extra_options)
        payload = {
            "model": self.model,
            "messages": self._build_messages(observation),
            "stream": False,
            "format": "json",
            "think": self.think,
            "options": options,
        }
        return json.dumps(payload).encode("utf-8")

    def decide_action(self, observation: Observation) -> AgentDecision:
        request = urllib.request.Request(
            self.endpoint,
            data=self._request_body(observation),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            return AgentDecision(action="fold", reasoning=f"ollama unreachable: {reason}")
        except TimeoutError as exc:
            return AgentDecision(action="fold", reasoning=f"ollama timeout: {exc}")

        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as exc:
            return AgentDecision(action="fold", reasoning=f"ollama invalid envelope: {exc}")

        content = _extract_content(envelope)
        if not content:
            return AgentDecision(action="fold", reasoning="ollama empty content")

        try:
            decision_payload = json.loads(content)
        except json.JSONDecodeError as exc:
            return AgentDecision(action="fold", reasoning=f"llm invalid json: {exc}")
        if not isinstance(decision_payload, dict):
            return AgentDecision(action="fold", reasoning="llm json is not object")
        decision = AgentDecision.from_mapping(decision_payload)
        normalize_decision_card_language(decision)
        if not self.table_talk_allowed:
            decision.table_talk = None
        return decision


def _extract_content(envelope: Dict[str, Any]) -> Optional[str]:
    message = envelope.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    response = envelope.get("response")
    if isinstance(response, str):
        return response
    return None
