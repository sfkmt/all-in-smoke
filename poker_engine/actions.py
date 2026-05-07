"""Action schemas."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


FOLD = "fold"
CHECK = "check"
CALL = "call"
BET = "bet"
RAISE = "raise"
ALL_IN = "all_in"
ACTION_TYPES = {FOLD, CHECK, CALL, BET, RAISE, ALL_IN}


@dataclass(frozen=True)
class PokerAction:
    """An agent-proposed action.

    For bet/raise, amount is the player's total current street bet after action.
    """

    action: str
    amount: Optional[int] = None

    def __post_init__(self) -> None:
        normalized = str(self.action).strip().lower().replace("-", "_")
        if normalized == "allin":
            normalized = ALL_IN
        if normalized not in ACTION_TYPES:
            raise ValueError(f"Unknown poker action: {self.action!r}")
        object.__setattr__(self, "action", normalized)
        if self.amount is not None:
            object.__setattr__(self, "amount", int(self.amount))

    @classmethod
    def from_mapping(cls, payload: Dict[str, Any]) -> "PokerAction":
        return cls(action=str(payload.get("action", "")), amount=payload.get("amount"))

    def to_dict(self) -> Dict[str, Any]:
        return {"action": self.action, "amount": self.amount}


@dataclass(frozen=True)
class LegalAction:
    action: str
    min_amount: Optional[int] = None
    max_amount: Optional[int] = None
    to_call: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "min_amount": self.min_amount,
            "max_amount": self.max_amount,
            "to_call": self.to_call,
        }

