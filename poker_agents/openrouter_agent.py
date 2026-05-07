"""OpenRouter-backed poker agent using the OpenAI-compatible chat API."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from poker_agents.base import AgentDecision, BaseAgent, Observation
from poker_agents.card_language import normalize_decision_card_language
from poker_agents.llm_agent import DEFAULT_SYSTEM_PROMPT, _format_observation
from poker_agents.personas import get_persona
from poker_agents.session_state import SessionState
from poker_agents.voice_profile import build_agent_context


DEFAULT_MODEL = "x-ai/grok-4.1-fast"
DEFAULT_ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
DEFAULT_TIMEOUT = 45.0
DEFAULT_TEMPERATURE = 0.3
DEFAULT_API_KEY_ENV = "OPENROUTER_API_KEY"


class OpenRouterAgent(BaseAgent):
    """Delegates decisions to OpenRouter's chat completions endpoint."""

    def __init__(
        self,
        agent_id: str,
        *,
        model: str = DEFAULT_MODEL,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key_env: str = DEFAULT_API_KEY_ENV,
        timeout: float = DEFAULT_TIMEOUT,
        temperature: float = DEFAULT_TEMPERATURE,
        system_prompt: Optional[str] = None,
        persona: Optional[str] = None,
        table_talk_allowed: Optional[bool] = None,
        identity_context: Optional[Dict[str, Any]] = None,
        voice_profile: Optional[Dict[str, Any]] = None,
        reasoning_enabled: bool = False,
        max_completion_tokens: int = 700,
        extra_options: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(agent_id)
        self.model = model
        self.endpoint = endpoint
        self.api_key_env = api_key_env
        self.timeout = float(timeout)
        self.temperature = float(temperature)
        self.reasoning_enabled = bool(reasoning_enabled)
        self.max_completion_tokens = int(max_completion_tokens)
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
        self.extra_options = dict(extra_options or {})

    def _build_messages(self, observation: Observation) -> List[Dict[str, str]]:
        session = self.session if isinstance(self.session, SessionState) else None
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": _format_observation(
                    observation,
                    session,
                    agent_context=self.agent_context,
                ),
            },
        ]

    def _request_body(self, observation: Observation) -> bytes:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": self._build_messages(observation),
            "temperature": self.temperature,
            "stream": False,
            "response_format": {"type": "json_object"},
            "reasoning": {"enabled": self.reasoning_enabled},
            "max_completion_tokens": self.max_completion_tokens,
        }
        payload.update(self.extra_options)
        return json.dumps(payload).encode("utf-8")

    def decide_action(self, observation: Observation) -> AgentDecision:
        api_key = os.environ.get(self.api_key_env)
        if not api_key:
            return AgentDecision(
                action="fold",
                reasoning="openrouter api key missing: {0}".format(self.api_key_env),
            )
        request = urllib.request.Request(
            self.endpoint,
            data=self._request_body(observation),
            headers={
                "Authorization": "Bearer {0}".format(api_key),
                "Content-Type": "application/json",
                "HTTP-Referer": "https://local.all-in-smoke",
                "X-Title": "AgentsPoker Hackathon",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", exc)
            return AgentDecision(action="fold", reasoning="openrouter unreachable: {0}".format(reason))
        except TimeoutError as exc:
            return AgentDecision(action="fold", reasoning="openrouter timeout: {0}".format(exc))

        try:
            envelope = json.loads(body)
        except json.JSONDecodeError as exc:
            return AgentDecision(action="fold", reasoning="openrouter invalid envelope: {0}".format(exc))

        content = _extract_content(envelope)
        if not content:
            return AgentDecision(action="fold", reasoning="openrouter empty content")

        try:
            decision_payload = json.loads(content)
        except json.JSONDecodeError as exc:
            return AgentDecision(action="fold", reasoning="openrouter invalid json: {0}".format(exc))
        if not isinstance(decision_payload, dict):
            return AgentDecision(action="fold", reasoning="openrouter json is not object")
        decision = AgentDecision.from_mapping(decision_payload)
        normalize_decision_card_language(decision)
        if not self.table_talk_allowed:
            decision.table_talk = None
        return decision


def _extract_content(envelope: Dict[str, Any]) -> Optional[str]:
    choices = envelope.get("choices")
    if isinstance(choices, list) and choices:
        first = choices[0]
        if isinstance(first, dict):
            message = first.get("message")
            if isinstance(message, dict):
                content = message.get("content")
                if isinstance(content, str):
                    return content
    return None
