"""Deterministic baseline agents useful as sparring partners."""

from __future__ import annotations

import random
from typing import Optional

from poker_agents.base import AgentDecision, BaseAgent, Observation
from poker_engine.cards import cards_text_japanese


PREMIUM_PAIRS = {"AA", "KK", "QQ", "JJ", "TT"}
STRONG_BROADWAY = {"AK", "AQ", "KQ"}


def _stack_pressure(observation: Observation) -> float:
    own_stack = max(1, int(observation.stacks.get(observation.seat_id, 0) or 0))
    return min(1.0, float(observation.to_call or 0) / own_stack)


def _psych(mood: str, observation: Observation, confidence: float, *, tilt: float = 0.0) -> dict:
    return {
        "mood": mood,
        "tilt": float(tilt or 0.0),
        "confidence_on_hand": float(confidence),
        "pot_pressure": round(_stack_pressure(observation), 3),
    }


def _voice_for_action(style: str, observation: Observation, action: str, *, premium: bool = False) -> str:
    cards = cards_text_japanese(observation.hole_cards) if observation.hole_cards else "手札"
    to_call = int(observation.to_call or 0)
    if action in {"raise", "bet", "all_in"}:
        if style == "random":
            return f"{cards}。根拠は薄いが、ここで動けば卓の流れが変わるかもしれない。"
        if style == "aggressive":
            return f"{cards}。相手に考える時間を渡さない。こちらから圧をかける。"
        if premium:
            return f"{cards} は戦える。ここは価値を取りにいく。"
        return f"{cards}。少し熱いが、降りるより主導権を取りたい。"
    if action == "call":
        if style == "calling":
            return f"{cards}。強い確信はないが、{to_call} ならまだ見届けられる。"
        if premium:
            return f"{cards} は捨てきれない。相手の次の動きを見る。"
        return f"{cards}。迷うが、この額ならまだ残れる。"
    if action == "check":
        if premium:
            return f"{cards} は悪くない。無料なら次のカードを見たい。"
        return f"{cards}。無理に膨らませず、場の反応を見る。"
    if action == "fold":
        if to_call > 0:
            return f"{cards} で {to_call} は重い。ここで降りられるかが勝負だ。"
        return f"{cards}。今は勝負する理由が足りない。"
    return f"{cards}。この場面は安全に処理する。"


def _rank_code(card: str) -> str:
    return card[0].upper()


def _is_premium_hole(hole_cards) -> bool:
    if len(hole_cards) < 2:
        return False
    r1, r2 = sorted((_rank_code(hole_cards[0]), _rank_code(hole_cards[1])), reverse=True)
    combo = r1 + r2
    if r1 == r2 and combo in PREMIUM_PAIRS:
        return True
    return combo in STRONG_BROADWAY


class RandomAgent(BaseAgent):
    """Picks a uniformly random legal action within legal amount bounds."""

    def __init__(self, agent_id: str, seed: Optional[int] = None):
        super().__init__(agent_id)
        self.rng = random.Random(seed)

    def decide_action(self, observation: Observation) -> AgentDecision:
        if not observation.legal_actions:
            return AgentDecision(
                action="fold",
                reasoning="no legal actions",
                inner_voice="動ける選択肢がない。ここは手を引くしかない。",
                psych=_psych("blocked", observation, 0.2),
            )
        choice = self.rng.choice(observation.legal_actions)
        amount = None
        if choice["min_amount"] is not None and choice["max_amount"] is not None:
            amount = self.rng.randint(choice["min_amount"], choice["max_amount"])
        action = choice["action"]
        return AgentDecision(
            action=action,
            amount=amount,
            confidence=0.5,
            reasoning="random",
            inner_voice=_voice_for_action("random", observation, action),
            psych=_psych("swingy", observation, 0.5),
        )


class CallingAgent(BaseAgent):
    """Calling station: checks when free, calls when facing a bet."""

    def decide_action(self, observation: Observation) -> AgentDecision:
        names = observation.legal_action_names()
        if "check" in names:
            return AgentDecision(
                action="check",
                confidence=0.5,
                reasoning="calling station checks",
                inner_voice=_voice_for_action("calling", observation, "check"),
                psych=_psych("curious", observation, 0.5),
            )
        if "call" in names:
            return AgentDecision(
                action="call",
                confidence=0.5,
                reasoning="calling station calls",
                inner_voice=_voice_for_action("calling", observation, "call"),
                psych=_psych("curious", observation, 0.5),
            )
        return AgentDecision(
            action="fold",
            confidence=0.5,
            reasoning="cannot call",
            inner_voice=_voice_for_action("calling", observation, "fold"),
            psych=_psych("boxed_in", observation, 0.5),
        )


class TightAgent(BaseAgent):
    """Plays only premium holdings; folds to any bet otherwise.

    When session tilt crosses 0.5 the agent abandons its premium-only filter
    and plays every holding as if it were premium — the classic "steamed
    nit goes maniac" behaviour.
    """

    TILT_LOOSE_THRESHOLD = 0.5

    def decide_action(self, observation: Observation) -> AgentDecision:
        names = observation.legal_action_names()
        tilt = self.session.tilt_factor()
        on_tilt = tilt >= self.TILT_LOOSE_THRESHOLD
        premium = _is_premium_hole(observation.hole_cards) or on_tilt

        if premium:
            raise_entry = observation.legal_action("raise") or observation.legal_action("bet")
            if raise_entry and raise_entry["min_amount"] is not None:
                tag = "tilted loose raise" if on_tilt else "premium hand value raise"
                return AgentDecision(
                    action=raise_entry["action"],
                    amount=raise_entry["min_amount"],
                    confidence=0.8,
                    reasoning=tag,
                    inner_voice=_voice_for_action("tight", observation, raise_entry["action"], premium=True),
                    psych=_psych("tilted" if on_tilt else "controlled", observation, 0.8, tilt=tilt),
                )
            if "call" in names:
                tag = "tilted loose call" if on_tilt else "premium hand flat call"
                return AgentDecision(
                    action="call",
                    confidence=0.7,
                    reasoning=tag,
                    inner_voice=_voice_for_action("tight", observation, "call", premium=True),
                    psych=_psych("tilted" if on_tilt else "controlled", observation, 0.7, tilt=tilt),
                )
            if "check" in names:
                return AgentDecision(
                    action="check",
                    confidence=0.6,
                    reasoning="premium hand free card",
                    inner_voice=_voice_for_action("tight", observation, "check", premium=True),
                    psych=_psych("controlled", observation, 0.6),
                )

        if "check" in names:
            return AgentDecision(
                action="check",
                confidence=0.6,
                reasoning="tight check",
                inner_voice=_voice_for_action("tight", observation, "check"),
                psych=_psych("guarded", observation, 0.6),
            )
        return AgentDecision(
            action="fold",
            confidence=0.9,
            reasoning="tight fold",
            inner_voice=_voice_for_action("tight", observation, "fold"),
            psych=_psych("disciplined", observation, 0.9),
        )


class AggressiveAgent(BaseAgent):
    """Bets or raises whenever legal, otherwise calls.

    Raise size scales with session tilt: calm = min, fully tilted = max.
    """

    def decide_action(self, observation: Observation) -> AgentDecision:
        names = observation.legal_action_names()
        tilt = self.session.tilt_factor()
        for aggressive in ("raise", "bet"):
            entry = observation.legal_action(aggressive)
            if entry and entry["min_amount"] is not None:
                lo = int(entry["min_amount"])
                hi = int(entry["max_amount"]) if entry.get("max_amount") is not None else lo
                amount = lo + int(round((hi - lo) * tilt))
                tag = "aggressive %s (tilted)" % aggressive if tilt >= 0.35 else f"aggressive {aggressive}"
                return AgentDecision(
                    action=aggressive,
                    amount=amount,
                    confidence=0.65,
                    reasoning=tag,
                    inner_voice=_voice_for_action("aggressive", observation, aggressive),
                    psych=_psych("heated" if tilt >= 0.35 else "pressing", observation, 0.65, tilt=tilt),
                )
        if "call" in names:
            return AgentDecision(
                action="call",
                confidence=0.55,
                reasoning="aggressive call",
                inner_voice=_voice_for_action("aggressive", observation, "call"),
                psych=_psych("pressing", observation, 0.55),
            )
        if "check" in names:
            return AgentDecision(
                action="check",
                confidence=0.55,
                reasoning="aggressive check",
                inner_voice=_voice_for_action("aggressive", observation, "check"),
                psych=_psych("pressing", observation, 0.55),
            )
        return AgentDecision(
            action="fold",
            confidence=0.5,
            reasoning="forced fold",
            inner_voice=_voice_for_action("aggressive", observation, "fold"),
            psych=_psych("frustrated", observation, 0.5),
        )
