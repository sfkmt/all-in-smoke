"""Base agent interface and observation/decision schemas."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Observation:
    """Information revealed to a single agent at decision time.

    Only public state plus the acting seat's own hole cards is included.
    Other players' hole cards, the undealt deck, and the RNG seed MUST NOT
    appear here.
    """

    hand_id: int
    street: str
    seat_id: int
    hole_cards: List[str]
    board: List[str]
    pot: int
    to_call: int
    current_bet: int
    min_raise_to: Optional[int]
    max_raise_to: Optional[int]
    legal_actions: List[Dict[str, Any]]
    stacks: Dict[int, int]
    committed: Dict[int, int]
    folded_seats: List[int]
    all_in_seats: List[int]
    button_seat: int
    big_blind: int
    small_blind: int
    action_history: List[Dict[str, Any]] = field(default_factory=list)
    # Public table talk seen so far in the current hand. Each entry shape:
    # {"seat_id": int, "hand_id": int, "street": str, "to": "all"|int, "text": str}
    recent_table_talk: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def legal_action_names(self) -> List[str]:
        return [entry["action"] for entry in self.legal_actions]

    def legal_action(self, name: str) -> Optional[Dict[str, Any]]:
        for entry in self.legal_actions:
            if entry["action"] == name:
                return entry
        return None


@dataclass
class AgentDecision:
    """Structured response from an agent.

    Only `action` is required. Everything else is optional metadata that the
    engine records verbatim for downstream analysis.
    """

    action: str
    amount: Optional[int] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    memory: Optional[str] = None
    inner_voice: Optional[str] = None
    table_talk: Optional[Dict[str, Any]] = None
    psych: Optional[Dict[str, Any]] = None

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "AgentDecision":
        amount = payload.get("amount")
        confidence = payload.get("confidence")
        table_talk = payload.get("table_talk")
        if not isinstance(table_talk, dict):
            table_talk = None
        psych = payload.get("psych")
        if not isinstance(psych, dict):
            psych = None
        return cls(
            action=str(payload.get("action", "")),
            amount=int(amount) if amount is not None else None,
            confidence=float(confidence) if confidence is not None else None,
            reasoning=payload.get("reasoning"),
            memory=payload.get("memory"),
            inner_voice=payload.get("inner_voice"),
            table_talk=table_talk,
            psych=psych,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "amount": self.amount,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "memory": self.memory,
            "inner_voice": self.inner_voice,
            "table_talk": self.table_talk,
            "psych": self.psych,
        }


class BaseAgent(ABC):
    """Abstract interface every agent must implement."""

    def __init__(self, agent_id: str):
        from poker_agents.session_state import SessionState  # local to avoid cycle

        self.agent_id = agent_id
        self.session = SessionState(agent_id=agent_id)

    @abstractmethod
    def decide_action(self, observation: Observation) -> AgentDecision:
        """Return the agent's chosen action for the given observation."""

    def on_hand_start(self, observation: Observation) -> None:
        """Default: stash the seat assignment so on_hand_end can use it.

        Subclasses that override should call super().on_hand_start(observation)
        unless they manage `session.own_seat` themselves.
        """
        self.session.own_seat = observation.seat_id

    def on_hand_end(self, result: Dict[str, Any]) -> None:
        """Default: ingest the hand result into session memory + tilt.

        `result` is `HandResult.to_dict()` augmented by the simulator with
        `seat_to_agent_id` and `big_blind`. Subclasses that override should
        call super().on_hand_end(result) to keep memory current.
        """
        seat_to_agent_id = {
            int(seat): str(name)
            for seat, name in (result.get("seat_to_agent_id") or {}).items()
        }
        big_blind = int(result.get("big_blind") or 0) or 1
        self.session.ingest_hand_result(
            result,
            seat_to_agent_id=seat_to_agent_id,
            big_blind=big_blind,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(agent_id={self.agent_id!r})"
