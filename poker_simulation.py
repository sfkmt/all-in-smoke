"""Hand and tournament orchestration for agent-only Texas Hold'em."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, TextIO

from poker_agents.base import AgentDecision, BaseAgent, Observation
from poker_agents.card_language import normalize_decision_card_language
from poker_engine import (
    IllegalActionError,
    PokerAction,
    advance_street,
    apply_action,
    legal_actions,
    settle_fold_win,
    settle_showdown,
    start_hand,
)
from poker_engine.actions import BET, CALL, CHECK, FOLD, RAISE
from poker_engine.table import SHOWDOWN, HandState


def build_observation(
    state: HandState,
    seat_id: int,
    *,
    recent_table_talk: Optional[Sequence[Dict[str, Any]]] = None,
) -> Observation:
    """Return the information the engine is willing to show `seat_id`.

    Must not include other players' hole cards, the undealt deck, or the
    RNG seed. Callers pass the result to `BaseAgent.decide_action`.
    """
    player = state.player(seat_id)
    to_call = max(0, state.current_bet - player.current_bet)
    legal = [entry.to_dict() for entry in legal_actions(state, seat_id)]
    min_raise_to = None
    max_raise_to = None
    for entry in legal:
        if entry["action"] in {BET, RAISE}:
            min_raise_to = entry["min_amount"]
            max_raise_to = entry["max_amount"]
            break
    return Observation(
        hand_id=state.hand_id,
        street=state.street,
        seat_id=seat_id,
        hole_cards=[str(card) for card in player.hole_cards],
        board=[str(card) for card in state.board],
        pot=state.pot,
        to_call=to_call,
        current_bet=state.current_bet,
        min_raise_to=min_raise_to,
        max_raise_to=max_raise_to,
        legal_actions=legal,
        stacks={other.seat_id: other.stack for other in state.players},
        committed={other.seat_id: other.committed for other in state.players},
        folded_seats=[other.seat_id for other in state.players if other.folded],
        all_in_seats=[other.seat_id for other in state.players if other.all_in],
        button_seat=state.button_seat,
        big_blind=state.big_blind,
        small_blind=state.small_blind,
        action_history=list(state.action_history),
        recent_table_talk=list(recent_table_talk or []),
    )


def _safe_fallback(observation: Observation, reason: str) -> PokerAction:
    names = observation.legal_action_names()
    if CALL in names:
        return PokerAction(CALL)
    if CHECK in names:
        return PokerAction(CHECK)
    return PokerAction(FOLD)


def resolve_action(
    state: HandState,
    seat_id: int,
    decision: AgentDecision,
) -> tuple[PokerAction, Optional[str]]:
    """Translate an AgentDecision into a legal PokerAction.

    Returns `(action, fallback_reason)`. `fallback_reason` is set when the
    engine rejects the agent's choice and a safe default is used instead.
    """
    observation = build_observation(state, seat_id)
    try:
        action = PokerAction(
            action=decision.action,
            amount=_resolve_amount(decision, observation),
        )
    except ValueError as exc:
        fallback = _safe_fallback(observation, str(exc))
        return fallback, f"malformed: {exc}"

    if action.action not in observation.legal_action_names():
        fallback = _safe_fallback(observation, f"illegal action {action.action}")
        return fallback, f"illegal: {action.action}"

    entry = observation.legal_action(action.action)
    if entry and entry["min_amount"] is not None and action.amount is None:
        fallback = _safe_fallback(observation, f"{action.action} missing amount")
        return fallback, f"missing_amount: {action.action}"
    if entry and action.amount is not None:
        if entry["min_amount"] is not None and action.amount < entry["min_amount"]:
            fallback = _safe_fallback(observation, "amount below min")
            return fallback, "amount_below_min"
        if entry["max_amount"] is not None and action.amount > entry["max_amount"]:
            fallback = _safe_fallback(observation, "amount above max")
            return fallback, "amount_above_max"
    return action, None


def _resolve_amount(decision: AgentDecision, observation: Observation) -> Optional[int]:
    if decision.amount is not None:
        return int(decision.amount)
    entry = observation.legal_action(decision.action.strip().lower())
    if entry and entry["min_amount"] is not None:
        return int(entry["min_amount"])
    return None


@dataclass
class HandResult:
    hand_id: int
    winners: List[int]
    payouts: Dict[int, int]
    final_stacks: Dict[int, int]
    board: List[str]
    street: str
    action_log: List[Dict[str, Any]] = field(default_factory=list)
    showdown: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hand_id": self.hand_id,
            "winners": list(self.winners),
            "payouts": dict(self.payouts),
            "final_stacks": dict(self.final_stacks),
            "board": list(self.board),
            "street": self.street,
            "action_log": list(self.action_log),
            "showdown": self.showdown,
        }


class JsonlLogger:
    """Writes jsonl events to one main stream plus optional per-event side streams.

    A monotonic `step` counter is bumped for every decision point (`action`,
    `memory_reasoning`, `table_talk`); callers should call `next_step()` once
    per decision and stamp all three events with that value so they line up.
    """

    def __init__(
        self,
        stream: Optional[TextIO] = None,
        *,
        capture: bool = True,
        side_streams: Optional[Mapping[str, TextIO]] = None,
    ):
        self.stream = stream
        self.capture = capture
        self.side_streams: Dict[str, TextIO] = dict(side_streams or {})
        self.events: List[Dict[str, Any]] = []
        self.step: int = 0

    def next_step(self) -> int:
        self.step += 1
        return self.step

    def log(self, event_type: str, payload: Mapping[str, Any]) -> None:
        record = {"event": event_type, **dict(payload)}
        if self.capture:
            self.events.append(record)
        line = json.dumps(record, sort_keys=True) + "\n"
        if self.stream is not None:
            self.stream.write(line)
            self.stream.flush()
        side = self.side_streams.get(event_type)
        if side is not None:
            side.write(line)
            side.flush()


def _agent_for(agents: Mapping[int, BaseAgent], seat_id: int) -> BaseAgent:
    if seat_id not in agents:
        raise KeyError(f"No agent registered for seat {seat_id}")
    return agents[seat_id]


def _agent_metadata(agent: BaseAgent) -> Dict[str, Any]:
    return {
        "agent_id": agent.agent_id,
        "gender": getattr(agent, "gender", None),
        "full_name": getattr(agent, "full_name", None),
    }


def _finalize(state: HandState) -> Dict[str, Any]:
    if len(state.contenders()) <= 1:
        result = settle_fold_win(state)
        return {"result": result.to_dict(), "showdown": False}
    result = settle_showdown(state)
    return {"result": result.to_dict(), "showdown": True}


def _blind_level_for_hand(hand_index: int, blind_increase_every: Optional[int]) -> int:
    if blind_increase_every is None or blind_increase_every <= 0:
        return 0
    return max(0, int(hand_index) // int(blind_increase_every))


def _blinds_for_hand(
    *,
    hand_index: int,
    small_blind: int,
    big_blind: int,
    blind_increase_every: Optional[int] = None,
    blind_multiplier: float = 1.0,
    max_big_blind: Optional[int] = None,
) -> tuple[int, int]:
    if small_blind <= 0 or big_blind <= 0:
        raise ValueError("small_blind and big_blind must be positive")
    if small_blind >= big_blind:
        raise ValueError("small_blind must be smaller than big_blind")
    if blind_multiplier <= 0:
        raise ValueError("blind_multiplier must be positive")
    if max_big_blind is not None and max_big_blind <= 0:
        raise ValueError("max_big_blind must be positive")

    level = _blind_level_for_hand(hand_index, blind_increase_every)
    if level == 0 or blind_multiplier == 1.0:
        current_small = int(small_blind)
        current_big = int(big_blind)
    else:
        scale = float(blind_multiplier) ** level
        current_small = max(1, int(round(float(small_blind) * scale)))
        current_big = max(2, int(round(float(big_blind) * scale)))

    if max_big_blind is not None and current_big > max_big_blind:
        current_big = int(max_big_blind)
        ratio = float(small_blind) / float(big_blind)
        current_small = max(1, int(round(current_big * ratio)))

    if current_small >= current_big:
        current_small = max(1, current_big // 2)
    return current_small, current_big


def _heads_up_blinds(
    *,
    small_blind: Optional[int],
    big_blind: Optional[int],
    fallback_small_blind: int,
    fallback_big_blind: int,
) -> tuple[int, int]:
    if big_blind is None:
        current_big = int(fallback_big_blind)
    else:
        current_big = int(big_blind)
    if small_blind is None:
        current_small = max(1, current_big // 2)
    else:
        current_small = int(small_blind)
    if current_small <= 0 or current_big <= 0:
        raise ValueError("heads-up blinds must be positive")
    if current_small >= current_big:
        raise ValueError("heads-up small blind must be smaller than heads-up big blind")
    return current_small, current_big


def run_hand(
    agents: Mapping[int, BaseAgent],
    *,
    hand_id: int,
    stacks: Mapping[int, int],
    button_seat: int,
    small_blind: int,
    big_blind: int,
    seed: Optional[int] = None,
    logger: Optional[JsonlLogger] = None,
) -> HandResult:
    """Play a single hand to completion, returning the result."""
    state = start_hand(
        hand_id=hand_id,
        stacks=dict(stacks),
        button_seat=button_seat,
        small_blind=small_blind,
        big_blind=big_blind,
        seed=seed,
    )

    if logger is not None:
        logger.log(
            "hand_start",
            {
                "hand_id": hand_id,
                "button_seat": button_seat,
                "small_blind": small_blind,
                "big_blind": big_blind,
                "stacks": dict(stacks),
                "seats": [player.seat_id for player in state.players],
                "agents": {
                    int(seat): _agent_metadata(agent)
                    for seat, agent in agents.items()
                },
                # Observer-mode dump: lets a replay viewer show every seat's
                # hole cards. Agents never receive this field through Observation.
                "hole_cards": {
                    player.seat_id: [str(card) for card in player.hole_cards]
                    for player in state.players
                },
            },
        )

    action_log: List[Dict[str, Any]] = []
    talk_history: List[Dict[str, Any]] = []

    for seat_id, agent in agents.items():
        try:
            agent.on_hand_start(build_observation(state, seat_id))
        except Exception:  # noqa: BLE001 - hook failures must not abort the hand
            pass

    while not state.completed:
        if state.action_seat is None:
            if len(state.contenders()) <= 1 or state.street == SHOWDOWN:
                break
            advance_street(state)
            continue

        seat_id = state.action_seat
        agent = _agent_for(agents, seat_id)
        observation = build_observation(
            state, seat_id, recent_table_talk=talk_history
        )
        try:
            decision = agent.decide_action(observation)
        except Exception as exc:  # noqa: BLE001
            decision = AgentDecision(action="fold", reasoning=f"agent error: {exc}")
        normalize_decision_card_language(decision)
        action, fallback = resolve_action(state, seat_id, decision)
        try:
            record = apply_action(state, seat_id, action)
        except IllegalActionError:
            record = apply_action(state, seat_id, PokerAction(FOLD))
            fallback = fallback or "engine_rejected"

        step = logger.next_step() if logger is not None else 0
        log_entry = {
            "step": step,
            "hand_id": hand_id,
            "seat_id": seat_id,
            "agent_id": agent.agent_id,
            "gender": getattr(agent, "gender", None),
            "full_name": getattr(agent, "full_name", None),
            "street": record["street"],
            "action": record["action"],
            "amount": record["amount"],
            "contributed": record["contributed"],
            "to_call_before": record["to_call_before"],
            "pot_after": record["pot_after"],
            "stack_after": record["stack_after"],
            "decision": decision.to_dict(),
            "fallback": fallback,
        }
        action_log.append(log_entry)
        if logger is not None:
            logger.log("action", log_entry)
            logger.log(
                "memory_reasoning",
                {
                    "step": step,
                    "hand_id": hand_id,
                    "seat_id": seat_id,
                    "agent_id": agent.agent_id,
                    "gender": getattr(agent, "gender", None),
                    "full_name": getattr(agent, "full_name", None),
                    "street": record["street"],
                    "inner_voice": decision.inner_voice,
                    "reasoning": decision.reasoning,
                    "memory": decision.memory,
                    "psych": decision.psych,
                    "confidence": decision.confidence,
                },
            )
            talk = decision.table_talk
            if isinstance(talk, dict) and talk.get("text"):
                talk_entry = {
                    "step": step,
                    "hand_id": hand_id,
                    "street": record["street"],
                    "seat_id": seat_id,
                    "agent_id": agent.agent_id,
                    "gender": getattr(agent, "gender", None),
                    "full_name": getattr(agent, "full_name", None),
                    "to": talk.get("to", "all"),
                    "text": talk.get("text"),
                }
                talk_history.append(talk_entry)
                logger.log("table_talk", talk_entry)

    settlement = _finalize(state)
    final_stacks = {player.seat_id: player.stack for player in state.players}
    result = HandResult(
        hand_id=hand_id,
        winners=list(state.winners),
        payouts=dict(state.payouts),
        final_stacks=final_stacks,
        board=[str(card) for card in state.board],
        street=state.street,
        action_log=action_log,
        showdown=settlement["result"] if settlement["showdown"] else None,
    )

    end_payload = {
        **result.to_dict(),
        "seat_to_agent_id": {
            int(seat): agent.agent_id for seat, agent in agents.items()
        },
        "big_blind": big_blind,
    }
    for agent in agents.values():
        try:
            agent.on_hand_end(end_payload)
        except Exception:  # noqa: BLE001
            pass

    if logger is not None:
        logger.log("hand_result", result.to_dict())
        for seat, agent in agents.items():
            session = getattr(agent, "session", None)
            if session is None:
                continue
            try:
                snap = session.snapshot()
            except Exception:  # noqa: BLE001
                continue
            logger.log(
                "session_snapshot",
                {
                    "hand_id": hand_id,
                    "seat_id": int(seat),
                    "agent_id": agent.agent_id,
                    "gender": getattr(agent, "gender", None),
                    "full_name": getattr(agent, "full_name", None),
                    "snapshot": snap,
                },
            )

    return result


def _rotate_button(active_seats: Sequence[int], current_button: int) -> int:
    if not active_seats:
        raise ValueError("No active seats to rotate button to")
    ordered = sorted(active_seats)
    if current_button in ordered:
        index = ordered.index(current_button)
        return ordered[(index + 1) % len(ordered)]
    for seat in ordered:
        if seat > current_button:
            return seat
    return ordered[0]


@dataclass
class TournamentResult:
    final_stacks: Dict[int, int]
    hand_results: List[HandResult] = field(default_factory=list)
    standings: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "final_stacks": dict(self.final_stacks),
            "hand_results": [hand.to_dict() for hand in self.hand_results],
            "standings": list(self.standings),
        }


def run_tournament(
    agents: Mapping[int, BaseAgent],
    *,
    starting_stacks: Mapping[int, int],
    num_hands: int,
    small_blind: int,
    big_blind: int,
    blind_increase_every: Optional[int] = None,
    blind_multiplier: float = 1.0,
    max_big_blind: Optional[int] = None,
    freeze_blinds_when_heads_up: bool = False,
    heads_up_small_blind: Optional[int] = None,
    heads_up_big_blind: Optional[int] = None,
    button_seat: Optional[int] = None,
    seed: Optional[int] = None,
    log_path: Optional[Path] = None,
    memory_reasoning_path: Optional[Path] = None,
    messages_path: Optional[Path] = None,
) -> TournamentResult:
    """Play up to `num_hands` hands rotating the button, dropping busted players."""
    stacks: Dict[int, int] = {int(seat): int(chips) for seat, chips in starting_stacks.items()}
    if button_seat is None:
        button_seat = min(stacks)

    hand_results: List[HandResult] = []
    stream = None
    if log_path is not None:
        stream = open(log_path, "w", encoding="utf-8")
    side_streams: Dict[str, TextIO] = {}
    mr_stream = None
    msg_stream = None
    if memory_reasoning_path is not None:
        mr_stream = open(memory_reasoning_path, "w", encoding="utf-8")
        side_streams["memory_reasoning"] = mr_stream
    if messages_path is not None:
        msg_stream = open(messages_path, "w", encoding="utf-8")
        side_streams["table_talk"] = msg_stream
    logger = JsonlLogger(stream=stream, side_streams=side_streams)

    try:
        for hand_index in range(num_hands):
            active_seats = [seat for seat, chips in stacks.items() if chips > 0]
            if len(active_seats) < 2:
                break
            if button_seat not in active_seats:
                button_seat = _rotate_button(active_seats, button_seat)
            active_agents = {seat: agents[seat] for seat in active_seats}
            active_stacks = {seat: stacks[seat] for seat in active_seats}
            hand_seed = None if seed is None else seed + hand_index
            hand_small_blind, hand_big_blind = _blinds_for_hand(
                hand_index=hand_index,
                small_blind=small_blind,
                big_blind=big_blind,
                blind_increase_every=blind_increase_every,
                blind_multiplier=blind_multiplier,
                max_big_blind=max_big_blind,
            )
            if freeze_blinds_when_heads_up and len(active_seats) <= 2:
                hand_small_blind, hand_big_blind = _heads_up_blinds(
                    small_blind=heads_up_small_blind,
                    big_blind=heads_up_big_blind,
                    fallback_small_blind=hand_small_blind,
                    fallback_big_blind=hand_big_blind,
                )
            result = run_hand(
                active_agents,
                hand_id=hand_index + 1,
                stacks=active_stacks,
                button_seat=button_seat,
                small_blind=hand_small_blind,
                big_blind=hand_big_blind,
                seed=hand_seed,
                logger=logger,
            )
            hand_results.append(result)
            for seat, chips in result.final_stacks.items():
                stacks[seat] = chips
            button_seat = _rotate_button(
                [seat for seat, chips in stacks.items() if chips > 0] or active_seats,
                button_seat,
            )

        standings = sorted(
            (
                {
                    "seat_id": seat,
                    "agent_id": agents[seat].agent_id,
                    "gender": getattr(agents[seat], "gender", None),
                    "full_name": getattr(agents[seat], "full_name", None),
                    "stack": stacks[seat],
                }
                for seat in agents
            ),
            key=lambda row: (-row["stack"], row["seat_id"]),
        )
        logger.log("standings", {"standings": standings})
        return TournamentResult(
            final_stacks=dict(stacks),
            hand_results=hand_results,
            standings=standings,
        )
    finally:
        if stream is not None:
            stream.close()
        if mr_stream is not None:
            mr_stream.close()
        if msg_stream is not None:
            msg_stream.close()
