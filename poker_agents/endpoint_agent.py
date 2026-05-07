"""HTTP-backed agent that delegates decisions to an external endpoint."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

from poker_agents.base import AgentDecision, BaseAgent, Observation


class EndpointAgent(BaseAgent):
    """Posts the observation as JSON to `endpoint` and parses the response.

    On timeout / transport error / malformed JSON we return a fold decision
    tagged with the error reason; the simulation runner's safe-fallback
    layer then substitutes call/check/fold as appropriate.
    """

    def __init__(
        self,
        agent_id: str,
        endpoint: str,
        *,
        timeout: float = 5.0,
        headers: Optional[Dict[str, str]] = None,
    ):
        super().__init__(agent_id)
        self.endpoint = endpoint
        self.timeout = float(timeout)
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def decide_action(self, observation: Observation) -> AgentDecision:
        payload = json.dumps(observation.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=payload,
            headers=self.headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            return AgentDecision(
                action="fold",
                reasoning=f"endpoint error: {exc.reason if hasattr(exc, 'reason') else exc}",
            )
        except TimeoutError as exc:
            return AgentDecision(action="fold", reasoning=f"endpoint timeout: {exc}")

        try:
            parsed: Dict[str, Any] = json.loads(body)
        except json.JSONDecodeError as exc:
            return AgentDecision(action="fold", reasoning=f"invalid json: {exc}")
        if not isinstance(parsed, dict):
            return AgentDecision(action="fold", reasoning="response is not an object")
        return AgentDecision.from_mapping(parsed)
