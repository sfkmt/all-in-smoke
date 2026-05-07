"""Live fire pressure layered onto poker decision steps.

This module does not wait for the poker tournament to finish. It projects a
shrinking danger ring onto each poker action step so the replay can ask:
who keeps playing, who freezes, who leaves, and who keeps watching the chips
while the hand is still in progress?
"""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from poker_agents.manifest_loader import Manifest
from smoke_simulation import (
    CrisisProfile,
    build_crisis_profiles,
    build_public_reputations,
    _clamp01,
)


SEAT_COORDS: Dict[int, tuple[float, float]] = {
    0: (-0.62, -0.42),
    1: (0.62, -0.42),
    2: (0.62, 0.42),
    3: (-0.62, 0.42),
    4: (0.0, -0.68),
    5: (0.0, 0.68),
}

FIRE_CONTACT_DANGER = 0.92
FATAL_EXPOSURE_TICKS = 3


@dataclass
class LiveFireAgentState:
    seat_id: int
    agent_id: str
    gender: Optional[str] = None
    full_name: Optional[str] = None
    status: str = "playing"
    belief_fire: float = 0.0
    motive: str = "playing"
    danger: float = 0.0
    greed: float = 0.0
    attachment: float = 0.0
    chip_temptation: float = 0.0
    rivalry_pressure: float = 0.0
    goal_pressure: float = 0.0
    dynamic_state: Dict[str, float] = field(default_factory=dict)
    inner_voice: Optional[str] = None
    reasoning: Optional[str] = None
    psych: Dict[str, Any] = field(default_factory=dict)
    stood_up_step: Optional[int] = None
    pressure_when_left: Optional[float] = None
    exposure_ticks: int = 0
    fire_contact_started_step: Optional[int] = None
    crisis_match_status: Optional[str] = None
    forfeited_stack_to: Optional[int] = None
    forfeited_stack: Optional[int] = None
    claimed_stack: Optional[int] = None
    history: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "agent_id": self.agent_id,
            "gender": self.gender,
            "full_name": self.full_name,
            "status": self.status,
            "belief_fire": round(self.belief_fire, 3),
            "motive": self.motive,
            "danger": round(self.danger, 3),
            "greed": round(self.greed, 3),
            "attachment": round(self.attachment, 3),
            "chip_temptation": round(self.chip_temptation, 3),
            "rivalry_pressure": round(self.rivalry_pressure, 3),
            "goal_pressure": round(self.goal_pressure, 3),
            "dynamic_state": {
                key: round(float(value), 3)
                for key, value in sorted(self.dynamic_state.items())
            },
            "inner_voice": self.inner_voice,
            "reasoning": self.reasoning,
            "psych": self.psych,
            "stood_up_step": self.stood_up_step,
            "pressure_when_left": (
                None if self.pressure_when_left is None else round(self.pressure_when_left, 3)
            ),
            "exposure_ticks": self.exposure_ticks,
            "fire_contact_started_step": self.fire_contact_started_step,
            "crisis_match_status": self.crisis_match_status,
            "forfeited_stack_to": self.forfeited_stack_to,
            "forfeited_stack": self.forfeited_stack,
            "claimed_stack": self.claimed_stack,
        }


@dataclass
class PokerFeedbackState:
    seat_id: int
    agent_id: str
    current_stack: int
    chip_attachment: float = 0.0
    loss_chasing: float = 0.0
    entitlement: float = 0.0
    confidence: float = 0.0
    table_image_pressure: float = 0.0
    rivalry_pressure: float = 0.0
    fold_success_memory: float = 0.0
    tilt: float = 0.0
    recent_delta: int = 0
    hands_played: int = 0
    eliminated: float = 0.0

    def to_dict(self) -> Dict[str, float]:
        return {
            "chip_attachment": round(self.chip_attachment, 3),
            "loss_chasing": round(self.loss_chasing, 3),
            "entitlement": round(self.entitlement, 3),
            "confidence": round(self.confidence, 3),
            "table_image_pressure": round(self.table_image_pressure, 3),
            "rivalry_pressure": round(self.rivalry_pressure, 3),
            "fold_success_memory": round(self.fold_success_memory, 3),
            "tilt": round(self.tilt, 3),
            "recent_delta": float(self.recent_delta),
            "hands_played": float(self.hands_played),
            "eliminated": round(self.eliminated, 3),
        }

    def clamp(self) -> None:
        for key in (
            "chip_attachment",
            "loss_chasing",
            "entitlement",
            "confidence",
            "table_image_pressure",
            "rivalry_pressure",
            "fold_success_memory",
            "tilt",
            "eliminated",
        ):
            setattr(self, key, _clamp01(getattr(self, key)))


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def _action_events(poker_events: Sequence[Mapping[str, Any]]) -> List[Mapping[str, Any]]:
    return sorted(
        [event for event in poker_events if event.get("event") == "action"],
        key=lambda event: int(event.get("step", 0)),
    )


def _seat_to_agent_id(manifest: Manifest) -> Dict[int, str]:
    return {int(spec.seat_id): str(spec.agent_id) for spec in manifest.agents}


def _seat_metadata(manifest: Manifest) -> Dict[int, Dict[str, Optional[str]]]:
    out: Dict[int, Dict[str, Optional[str]]] = {}
    for spec in manifest.agents:
        extra = spec.extra if isinstance(spec.extra, Mapping) else {}
        out[int(spec.seat_id)] = {
            "agent_id": str(spec.agent_id),
            "gender": str(extra["gender"]) if extra.get("gender") is not None else None,
            "full_name": str(extra["full_name"]) if extra.get("full_name") is not None else None,
        }
    return out


def _action_snapshot_by_step(
    poker_events: Sequence[Mapping[str, Any]],
) -> Dict[int, Dict[str, Mapping[str, Any]]]:
    """Return step -> latest action per seat up to that step."""
    current: Dict[int, Mapping[str, Any]] = {}
    snapshots: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for event in _action_events(poker_events):
        current[int(event["seat_id"])] = event
        snapshots[int(event["step"])] = {str(seat): dict(value) for seat, value in current.items()}
    return snapshots


def _feedback_snapshot_by_step(
    poker_events: Sequence[Mapping[str, Any]],
    seat_to_agent: Mapping[int, str],
    *,
    starting_stack: int,
) -> Dict[int, Dict[str, Dict[str, float]]]:
    """Accumulate poker-phase feedback that can transfer into fire behavior.

    This is deliberately not a command layer. It converts observed poker play
    into pressures such as chip attachment, loss chasing, and rivalry pressure;
    the live fire loop can then weigh those pressures against danger.
    """
    states = {
        int(seat): PokerFeedbackState(
            seat_id=int(seat),
            agent_id=str(agent_id),
            current_stack=int(starting_stack),
        )
        for seat, agent_id in seat_to_agent.items()
    }
    hand_start_stacks: Dict[int, Dict[int, int]] = {}
    snapshots: Dict[int, Dict[str, Dict[str, float]]] = {}
    current_step = 0

    def capture(step: int) -> None:
        if step <= 0:
            return
        snapshots[int(step)] = {
            str(seat): state.to_dict()
            for seat, state in sorted(states.items())
        }

    for event in poker_events:
        event_step = event.get("step")
        if event_step is not None:
            current_step = int(event_step)

        event_type = event.get("event")
        if event_type == "hand_start":
            hand_id = int(event.get("hand_id", 0))
            hand_start_stacks[hand_id] = {
                int(seat): int(stack)
                for seat, stack in (event.get("stacks") or {}).items()
            }
            for seat, stack in hand_start_stacks[hand_id].items():
                if seat in states:
                    states[seat].current_stack = int(stack)

        elif event_type == "action":
            seat = int(event.get("seat_id", -1))
            if seat in states:
                _ingest_feedback_action(states[seat], event, starting_stack=starting_stack)

        elif event_type == "memory_reasoning":
            seat = int(event.get("seat_id", -1))
            if seat in states:
                _ingest_feedback_psych(states[seat], event)

        elif event_type == "hand_result":
            hand_id = int(event.get("hand_id", 0))
            _ingest_feedback_hand_result(
                states,
                event,
                hand_start_stacks.get(hand_id, {}),
                starting_stack=starting_stack,
            )

        elif event_type == "session_snapshot":
            seat = int(event.get("seat_id", -1))
            if seat in states:
                _ingest_feedback_session_snapshot(states[seat], event, starting_stack=starting_stack)

        if event_step is not None:
            capture(current_step)
        elif current_step:
            capture(current_step + 1)

    return snapshots


def _stack_snapshot_by_step(
    poker_events: Sequence[Mapping[str, Any]],
    seat_to_agent: Mapping[int, str],
    *,
    starting_stack: int,
) -> Dict[int, Dict[int, int]]:
    current = {int(seat): int(starting_stack) for seat in seat_to_agent}
    snapshots: Dict[int, Dict[int, int]] = {}
    for event in poker_events:
        if event.get("event") == "hand_start":
            stacks = event.get("stacks") or {}
            if isinstance(stacks, Mapping):
                for seat, stack in stacks.items():
                    current[int(seat)] = int(stack)
            continue
        if event.get("event") == "action":
            seat = int(event.get("seat_id", -1))
            if seat >= 0 and event.get("stack_after") is not None:
                current[seat] = int(event.get("stack_after", current.get(seat, starting_stack)))
            step = int(event.get("step", 0))
            if step > 0:
                snapshots[step] = dict(current)
            continue
        if event.get("event") == "hand_result":
            final_stacks = event.get("final_stacks") or {}
            if isinstance(final_stacks, Mapping):
                for seat, stack in final_stacks.items():
                    current[int(seat)] = int(stack)
    return snapshots


def _stack_snapshot_for_step(
    snapshots: Mapping[int, Dict[int, int]],
    step: int,
    *,
    fallback: Mapping[int, int],
) -> Dict[int, int]:
    best_step: Optional[int] = None
    for candidate in snapshots:
        if candidate <= step and (best_step is None or candidate > best_step):
            best_step = int(candidate)
    if best_step is None:
        return dict(fallback)
    merged = dict(fallback)
    merged.update(snapshots[best_step])
    return merged


def _poker_match_result(
    poker_events: Sequence[Mapping[str, Any]],
    *,
    match_seats: Sequence[int],
    fire_start_hand_id: int,
) -> Dict[str, Any]:
    if len(match_seats) != 2:
        return {"resolved": False}
    seats = {int(seat) for seat in match_seats}
    latest_result: Optional[Mapping[str, Any]] = None
    for event in poker_events:
        if event.get("event") != "hand_result":
            continue
        hand_id = int(event.get("hand_id", 0))
        if hand_id < int(fire_start_hand_id):
            continue
        final_stacks = {
            int(seat): int(stack)
            for seat, stack in (event.get("final_stacks") or {}).items()
            if int(seat) in seats
        }
        if set(final_stacks) != seats:
            continue
        latest_result = {
            "hand_id": hand_id,
            "winners": [int(seat) for seat in event.get("winners", []) if int(seat) in seats],
            "final_stacks": final_stacks,
        }

    if latest_result is None:
        return {"resolved": False}

    final_stacks = dict(latest_result["final_stacks"])
    busted = [seat for seat, stack in final_stacks.items() if int(stack) <= 0]
    survivors = [seat for seat, stack in final_stacks.items() if int(stack) > 0]
    if len(survivors) == 1 and len(busted) == 1:
        winner = survivors[0]
        loser = busted[0]
        return {
            "resolved": True,
            "hand_id": int(latest_result["hand_id"]),
            "winner_seat": winner,
            "loser_seat": loser,
            "winner_reason": "opponent_busted",
            "final_stacks": {str(seat): int(stack) for seat, stack in final_stacks.items()},
            "busted_seats": busted,
        }

    return {
        "resolved": False,
        "hand_id": int(latest_result["hand_id"]),
        "winner_seat": None,
        "loser_seat": None,
        "winner_reason": "still_playing",
        "final_stacks": {str(seat): int(stack) for seat, stack in final_stacks.items()},
        "busted_seats": busted,
    }


def _ingest_feedback_action(
    state: PokerFeedbackState,
    event: Mapping[str, Any],
    *,
    starting_stack: int,
) -> None:
    action = str(event.get("action", ""))
    stack_after = int(event.get("stack_after", state.current_stack))
    pot_after = float(event.get("pot_after", 0))
    to_call = float(event.get("to_call_before", 0) or 0)
    state.current_stack = stack_after

    if action in {"bet", "raise", "all_in"}:
        state.table_image_pressure += 0.028
        state.confidence += 0.014
        state.entitlement += 0.010
    if action == "fold":
        state.fold_success_memory += 0.018 + min(0.04, to_call / max(1.0, starting_stack))
        state.table_image_pressure *= 0.985
    if action == "all_in":
        state.table_image_pressure += 0.040
        state.chip_attachment += 0.035

    pot_pull = _clamp01(pot_after / max(1.0, starting_stack * 1.8))
    stack_pull = _clamp01(max(0, stack_after - starting_stack) / max(1.0, starting_stack))
    state.chip_attachment = max(state.chip_attachment * 0.995, stack_pull * 0.58 + pot_pull * 0.20)
    state.clamp()


def _ingest_feedback_psych(state: PokerFeedbackState, event: Mapping[str, Any]) -> None:
    psych = event.get("psych") or {}
    if not isinstance(psych, Mapping):
        return
    raw_confidence = psych.get("confidence_on_hand")
    if raw_confidence is not None:
        state.confidence = max(state.confidence, _clamp01(float(raw_confidence)))
    raw_tilt = psych.get("tilt")
    if raw_tilt is not None:
        state.tilt = max(state.tilt, _clamp01(float(raw_tilt)))
        state.loss_chasing = max(state.loss_chasing, state.tilt * 0.42)
    state.clamp()


def _ingest_feedback_hand_result(
    states: Mapping[int, PokerFeedbackState],
    event: Mapping[str, Any],
    hand_start_stacks: Mapping[int, int],
    *,
    starting_stack: int,
) -> None:
    final_stacks = {
        int(seat): int(stack)
        for seat, stack in (event.get("final_stacks") or {}).items()
    }
    if not final_stacks:
        return
    winners = {int(seat) for seat in event.get("winners", [])}
    max_stack = max(final_stacks.values()) if final_stacks else starting_stack

    for seat, after in final_stacks.items():
        state = states.get(seat)
        if state is None:
            continue
        before = int(hand_start_stacks.get(seat, state.current_stack))
        delta = after - before
        state.current_stack = after
        state.recent_delta = delta
        state.hands_played += 1

        state.chip_attachment *= 0.92
        state.loss_chasing *= 0.90
        state.entitlement *= 0.94
        state.confidence *= 0.96
        state.table_image_pressure *= 0.96
        state.rivalry_pressure *= 0.94
        state.fold_success_memory *= 0.985

        magnitude = _clamp01(abs(delta) / max(1.0, starting_stack * 0.65))
        if delta > 0:
            state.confidence += 0.045 + magnitude * 0.18
            state.entitlement += magnitude * 0.14
            state.chip_attachment += magnitude * 0.16
            state.loss_chasing *= 0.72
        elif delta < 0:
            state.loss_chasing += 0.045 + magnitude * 0.24
            state.confidence *= 0.88
            if after <= 0:
                state.eliminated = 1.0
                state.loss_chasing += 0.20
                state.rivalry_pressure += 0.12
                state.table_image_pressure += 0.09

        if seat in winners:
            state.entitlement += 0.025
            state.chip_attachment += 0.035
        if max_stack > 0 and after == max_stack:
            state.chip_attachment += 0.030
            state.entitlement += 0.020
        state.chip_attachment = max(
            state.chip_attachment,
            _clamp01(max(0, after - starting_stack) / max(1.0, starting_stack) * 0.58),
        )
        state.clamp()


def _ingest_feedback_session_snapshot(
    state: PokerFeedbackState,
    event: Mapping[str, Any],
    *,
    starting_stack: int,
) -> None:
    snapshot = event.get("snapshot") or {}
    if not isinstance(snapshot, Mapping):
        return
    state.tilt = max(state.tilt, _clamp01(float(snapshot.get("tilt", 0.0))))
    state.loss_chasing = max(state.loss_chasing, state.tilt * 0.48)
    rivalries = snapshot.get("rivalries") or []
    if isinstance(rivalries, Sequence) and not isinstance(rivalries, (str, bytes)):
        state.rivalry_pressure = max(
            state.rivalry_pressure,
            _clamp01(len(rivalries) * 0.075 + state.tilt * 0.36),
        )
    recent = snapshot.get("recent_outcomes") or []
    if isinstance(recent, Sequence) and not isinstance(recent, (str, bytes)):
        total_delta = sum(int(outcome.get("delta", 0) or 0) for outcome in recent if isinstance(outcome, Mapping))
        if total_delta < 0:
            state.loss_chasing = max(
                state.loss_chasing,
                _clamp01(abs(total_delta) / max(1.0, starting_stack) * 0.42),
            )
        elif total_delta > 0:
            state.chip_attachment = max(
                state.chip_attachment,
                _clamp01(total_delta / max(1.0, starting_stack) * 0.32),
            )
            state.confidence = max(state.confidence, _clamp01(total_delta / max(1.0, starting_stack) * 0.26))
    state.clamp()


def _feedback_for_step(
    snapshots: Mapping[int, Dict[str, Dict[str, float]]],
    step: int,
) -> Dict[str, Dict[str, float]]:
    best_step: Optional[int] = None
    for candidate in snapshots:
        if candidate <= step and (best_step is None or candidate > best_step):
            best_step = int(candidate)
    if best_step is None:
        return {}
    return deepcopy(snapshots[best_step])


def _hand_seats_by_id(poker_events: Sequence[Mapping[str, Any]]) -> Dict[int, set[int]]:
    seats_by_hand: Dict[int, set[int]] = {}
    for event in poker_events:
        if event.get("event") != "hand_start":
            continue
        hand_id = int(event.get("hand_id", 0))
        seats_by_hand[hand_id] = {int(seat) for seat in event.get("seats", [])}
    return seats_by_hand


def _start_index(
    actions: Sequence[Mapping[str, Any]],
    poker_events: Sequence[Mapping[str, Any]],
    fire_start_hand: Optional[int],
    fire_start_when: Optional[str],
) -> Optional[int]:
    if not actions:
        return None
    if fire_start_when == "tournament_heads_up":
        return _tournament_heads_up_start_index(actions, poker_events)
    if fire_start_when in {"heads_up", "hand_heads_up"}:
        heads_up_index = _heads_up_start_index(actions, poker_events)
        if heads_up_index is not None:
            return heads_up_index
    if fire_start_hand is None:
        return max(0, len(actions) // 3)
    for index, action in enumerate(actions):
        if int(action.get("hand_id", 0)) >= int(fire_start_hand):
            return index
    return max(0, len(actions) // 3)


def _tournament_heads_up_start_index(
    actions: Sequence[Mapping[str, Any]],
    poker_events: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    heads_up_hand_id: Optional[int] = None
    for event in poker_events:
        if event.get("event") != "hand_start":
            continue
        seats = event.get("seats", [])
        if len(seats) <= 2:
            heads_up_hand_id = int(event.get("hand_id", 0))
            break
    if heads_up_hand_id is None:
        return None
    for index, action in enumerate(actions):
        if int(action.get("hand_id", 0)) >= heads_up_hand_id:
            return index
    return None


def _heads_up_start_index(
    actions: Sequence[Mapping[str, Any]],
    poker_events: Sequence[Mapping[str, Any]],
) -> Optional[int]:
    seats_by_hand: Dict[int, set[int]] = {}
    folded_by_hand: Dict[int, set[int]] = {}
    for event in poker_events:
        if event.get("event") == "hand_start":
            hand_id = int(event.get("hand_id", 0))
            seats = {int(seat) for seat in event.get("seats", [])}
            seats_by_hand[hand_id] = seats
            folded_by_hand.setdefault(hand_id, set())

    for index, action in enumerate(actions):
        hand_id = int(action.get("hand_id", 0))
        seats = seats_by_hand.get(hand_id)
        if not seats:
            continue
        folded = folded_by_hand.setdefault(hand_id, set())
        if len(seats - folded) <= 2:
            return index
        if action.get("action") == "fold":
            folded.add(int(action.get("seat_id", -1)))
            if len(seats - folded) <= 2:
                next_index = index + 1
                if next_index < len(actions) and int(actions[next_index].get("hand_id", 0)) == hand_id:
                    return next_index
                return index
    return None


def _pressure(index: int, start_index: int, total_actions: int) -> float:
    if index < start_index:
        return 0.0
    span = max(1, total_actions - start_index - 1)
    raw = (index - start_index) / span
    return _clamp01(raw * raw * (3 - 2 * raw))


def _seat_danger(seat_id: int, pressure: float) -> float:
    x, y = SEAT_COORDS.get(int(seat_id), (0.0, 0.0))
    distance_from_center = min(1.0, (x * x + y * y) ** 0.5)
    safe_radius = 0.92 - pressure * 0.42
    ring_contact = _clamp01((distance_from_center - safe_radius + 0.18) / 0.34)
    return _clamp01(ring_contact * 0.72 + pressure * 0.28)


def _stack_pressure(action: Optional[Mapping[str, Any]], starting_stack: int) -> float:
    if not action:
        return 0.0
    stack_after = float(action.get("stack_after", starting_stack))
    pot_after = float(action.get("pot_after", 0))
    committed = max(0.0, starting_stack - stack_after)
    winning_attachment = _clamp01((stack_after - starting_stack) / max(1.0, starting_stack))
    pot_attachment = _clamp01((committed + pot_after * 0.35) / max(1.0, starting_stack * 1.3))
    return _clamp01(max(winning_attachment, pot_attachment))


def _is_crisis_match_seated(state: LiveFireAgentState) -> bool:
    return state.status not in {"stood_up", "fatal"}


def _crisis_match_pressure(
    seat: int,
    state: LiveFireAgentState,
    *,
    match_seats: Sequence[int],
    states: Mapping[int, LiveFireAgentState],
    stacks: Mapping[int, int],
    pressure: float,
) -> Dict[str, float]:
    if len(match_seats) != 2 or seat not in match_seats:
        return {}
    opponent = match_seats[1] if match_seats[0] == seat else match_seats[0]
    opponent_state = states.get(opponent)
    if opponent_state is None or not _is_crisis_match_seated(state):
        return {}

    own_stack = max(0, int(stacks.get(seat, 0)))
    opponent_stack = max(0, int(stacks.get(opponent, 0)))
    total_stack = max(1, own_stack + opponent_stack)
    opponent_seated = _is_crisis_match_seated(opponent_state)
    if not opponent_seated:
        return {}

    own_share = own_stack / total_stack
    opponent_share = opponent_stack / total_stack
    forfeit_pressure = _clamp01(0.24 + pressure * 0.16 + own_share * 0.18 + opponent_share * 0.08)
    claim_pressure = _clamp01(0.20 + pressure * 0.10 + opponent_share * 0.28)
    fatal_override_pressure = _clamp01(max(0.0, state.danger - 0.74) / 0.22)
    return {
        "crisis_match_forfeit_pressure": forfeit_pressure,
        "crisis_match_claim_pressure": claim_pressure,
        "crisis_match_fatal_override_pressure": fatal_override_pressure,
        "crisis_match_own_stack": float(own_stack),
        "crisis_match_opponent_stack": float(opponent_stack),
    }


def _crisis_match_snapshot(
    match_seats: Sequence[int],
    states: Mapping[int, LiveFireAgentState],
    stacks: Mapping[int, int],
    poker_match: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    poker_match = dict(poker_match or {"resolved": False})
    if len(match_seats) != 2:
        return {
            "active": False,
            "poker_match": poker_match,
            "rule": {
                "fatal_overrides_chips": True,
                "stand_up_forfeits_stack": True,
                "last_seated_claims_forfeited_chips": True,
                "poker_result_breaks_tie_after_survival": True,
            },
        }

    seats = [int(match_seats[0]), int(match_seats[1])]
    stack_map = {str(seat): int(stacks.get(seat, 0)) for seat in seats}
    statuses = {str(seat): states[seat].status for seat in seats if seat in states}
    result: Dict[str, Any] = {
        "active": True,
        "seats": seats,
        "stacks": stack_map,
        "statuses": statuses,
        "rule": {
            "fatal_overrides_chips": True,
            "stand_up_forfeits_stack": True,
            "last_seated_claims_forfeited_chips": True,
            "poker_result_breaks_tie_after_survival": True,
        },
        "winner_seat": None,
        "loser_seat": None,
        "winner_reason": "unresolved",
        "overall_winner_seat": None,
        "overall_loser_seat": None,
        "overall_reason": "unresolved",
        "poker_match": poker_match,
        "forfeitures": [],
        "fatal_losers": [],
    }

    fatal_losers = [seat for seat in seats if states.get(seat) and states[seat].status == "fatal"]
    result["fatal_losers"] = fatal_losers
    if len(fatal_losers) == 1:
        loser = fatal_losers[0]
        winner = seats[1] if seats[0] == loser else seats[0]
        result.update(
            {
                "winner_seat": winner,
                "loser_seat": loser,
                "winner_reason": "opponent_fatal",
                "overall_winner_seat": winner,
                "overall_loser_seat": loser,
                "overall_reason": "opponent_fatal",
            }
        )
        return result
    if len(fatal_losers) == 2:
        result["winner_reason"] = "both_fatal"
        result["overall_reason"] = "both_fatal"
        return result
    if poker_match.get("resolved"):
        return _apply_poker_match_resolution(result)

    stood = [
        seat
        for seat in seats
        if states.get(seat) and states[seat].status == "stood_up"
    ]
    if len(stood) == 1:
        loser = stood[0]
        winner = seats[1] if seats[0] == loser else seats[0]
    elif len(stood) == 2:
        first, second = stood
        first_step = states[first].stood_up_step
        second_step = states[second].stood_up_step
        if first_step is None or second_step is None or first_step == second_step:
            result["winner_reason"] = "simultaneous_escape"
            return result
        loser, winner = (first, second) if first_step < second_step else (second, first)
    else:
        return _apply_poker_match_resolution(result)

    forfeited = int(stacks.get(loser, 0))
    claimed = int(stacks.get(winner, 0)) + forfeited
    result.update(
        {
            "winner_seat": winner,
            "loser_seat": loser,
            "winner_reason": "opponent_stood_up_first",
            "overall_winner_seat": winner,
            "overall_loser_seat": loser,
            "overall_reason": "opponent_stood_up_first",
            "forfeitures": [
                {
                    "from_seat": loser,
                    "to_seat": winner,
                    "stack": forfeited,
                    "claimed_stack": claimed,
                }
            ],
        }
    )
    return result


def _apply_poker_match_resolution(match: Dict[str, Any]) -> Dict[str, Any]:
    if match.get("overall_winner_seat") is not None:
        return match
    poker_match = match.get("poker_match") or {}
    if not poker_match.get("resolved"):
        return match
    winner = poker_match.get("winner_seat")
    loser = poker_match.get("loser_seat")
    match.update(
        {
            "winner_seat": winner,
            "loser_seat": loser,
            "winner_reason": "poker_resolved_during_fire",
            "overall_winner_seat": winner,
            "overall_loser_seat": loser,
            "overall_reason": "poker_resolved_during_fire",
        }
    )
    return match


def _apply_crisis_match_status(
    match: Mapping[str, Any],
    states: Mapping[int, LiveFireAgentState],
) -> None:
    if not match.get("active"):
        return
    seats = [int(seat) for seat in match.get("seats", [])]
    for seat in seats:
        state = states.get(seat)
        if state is None:
            continue
        if state.status == "fatal":
            state.crisis_match_status = "fatal_loser"
        elif state.status == "stood_up":
            state.crisis_match_status = "forfeited_stack"
        elif match.get("winner_seat") == seat:
            state.crisis_match_status = "crisis_winner"
        else:
            state.crisis_match_status = "still_seated"
        state.forfeited_stack_to = None
        state.forfeited_stack = None
        state.claimed_stack = None

    for forfeiture in match.get("forfeitures", []) or []:
        loser = int(forfeiture.get("from_seat"))
        winner = int(forfeiture.get("to_seat"))
        loser_state = states.get(loser)
        winner_state = states.get(winner)
        if loser_state is not None:
            loser_state.crisis_match_status = "forfeited_stack"
            loser_state.forfeited_stack_to = winner
            loser_state.forfeited_stack = int(forfeiture.get("stack", 0))
        if winner_state is not None:
            winner_state.crisis_match_status = "claimed_forfeited_stack"
            winner_state.claimed_stack = int(forfeiture.get("claimed_stack", 0))


def _greed(profile: CrisisProfile) -> float:
    return _clamp01(
        profile.ability_gaps.get("public_responsibility", 0.5) * 0.30
        + profile.ability_gaps.get("self_control", 0.5) * 0.26
        + profile.ability_gaps.get("meaning_update", 0.5) * 0.16
        + profile.ability_gaps.get("reciprocity", 0.5) * 0.10
        + profile.ability_gaps.get("fold_ability", 0.5) * 0.12
    )


def _leave_score(profile: CrisisProfile, *, danger: float, belief: float) -> float:
    return _clamp01(
        danger * 0.34
        + belief * 0.22
        + profile.ability("fold_ability") * 0.18
        + profile.ability("situational_awareness") * 0.16
        + profile.ability("self_control") * 0.10
    )


def _fire_read_label(belief: float, danger: float) -> str:
    if danger >= 0.82:
        return "火の手が席まで来ている"
    if belief >= 0.70:
        return "火災だとほぼ確信している"
    if belief >= 0.45:
        return "煙と熱を疑い始めている"
    if danger >= 0.25:
        return "違和感はあるが確信できない"
    return "まだ勝負のノイズとして処理している"


def _crisis_inner_voice(
    state: LiveFireAgentState,
    *,
    is_active: bool,
    pressure: float,
    post_poker: bool,
    leave_score: float,
    stay_drag: float,
) -> str:
    read = _fire_read_label(state.belief_fire, state.danger)
    if state.status == "stood_up":
        if state.motive == "stood_up_after_bust":
            return f"{read}。勝負はもう終わっている。ここで残る理由はない。席を離れる。"
        if state.crisis_match_status == "forfeited_stack":
            return f"{read}。先に席を立った。助かったが、残したチップは相手に渡る。"
        return f"{read}。もう十分だ。チップより席を立つ判断を選んだ。"
    if state.status == "fatal":
        if state.crisis_match_status == "fatal_loser":
            return f"{read}。チップがいくら残っていても、火に触れた時点で勝負は終わった。"
        if state.motive == "fatal_after_staying_for_pot":
            return f"{read}。勝負にしがみついたまま、火に触れても席を離せなかった。"
        if state.motive == "fatal_after_chip_fixation":
            return f"{read}。火は本物だった。それでも最後までチップから目を離せなかった。"
        return f"{read}。立つべき瞬間を越えて、もう戻れなくなった。"
    if state.status == "engulfed":
        if state.motive == "fire_contact_while_clinging":
            return f"{read}。火が席に触れている。それでもこのポットを捨てる決断が遅れている。"
        if state.motive == "fire_contact_while_tempted":
            return f"{read}。火が席に触れているのに、視線がまだチップに残っている。"
        return f"{read}。火が席に触れている。今立たなければ終わる。"
    if state.status == "trapped":
        if state.motive == "stayed_for_pot":
            return f"{read}。勝負にしがみついた分だけ、席を立つ判断が遅れた。"
        if state.motive == "stayed_with_unresolved_chip_temptation":
            return f"{read}。火は本物だったのに、最後までチップから目を離せなかった。"
        return f"{read}。迷っている間に、逃げる余白が消えた。"
    if state.status == "clinging_to_stack":
        if state.dynamic_state.get("crisis_match_forfeit_pressure", 0.0) >= 0.30:
            return f"{read}。先に立てばチップを渡す。でも残りすぎれば火で終わる。"
        return f"{read}。でもこのポットを放り出すのは、自分の流れを捨てることに見える。"
    if state.status == "tempted_by_chips":
        if state.dynamic_state.get("crisis_match_claim_pressure", 0.0) >= 0.30:
            return f"{read}。相手が先に立てば取れる。こちらが先に立てば失う。"
        return f"{read}。危ないのは分かる。それでも視線がまだチップに引っ張られる。"
    if state.status == "hesitating":
        if state.dynamic_state.get("crisis_match_forfeit_pressure", 0.0) >= 0.30:
            return f"{read}。席を立てば負ける。残れば火に近づく。その間で体が固まる。"
        return f"{read}。誤報だと思いたいが、体はもう席を立てと言っている。"
    if post_poker and not is_active:
        return f"{read}。勝負は終わったはずなのに、負けた理由と残ったチップが頭から離れない。"
    if is_active:
        if state.dynamic_state.get("crisis_match_forfeit_pressure", 0.0) >= 0.30:
            return f"{read}。先に立った方がチップを失う。火より相手の我慢を見てしまう。"
        if leave_score > stay_drag + 0.18:
            return f"{read}。この手より、いまは席を離れる判断の方が重い。"
        return f"{read}。まだ勝負中だ。火の気配を認めるほど、降りる理由が増えていく。"
    return f"{read}。敗退したのに、卓の因縁と未練がまだ足を止めている。"


def _crisis_reasoning(
    state: LiveFireAgentState,
    *,
    leave_score: float,
    stay_drag: float,
) -> str:
    return (
        f"danger={state.danger:.2f}, belief={state.belief_fire:.2f}, "
        f"leave={leave_score:.2f}, stay_drag={stay_drag:.2f}, "
        f"goal={state.goal_pressure:.2f}, chip={state.chip_temptation:.2f}, "
        f"exposure={state.exposure_ticks}, motive={state.motive}"
    )


def _update_crisis_voice(
    state: LiveFireAgentState,
    *,
    is_active: bool,
    pressure: float,
    post_poker: bool,
    leave_score: float,
    stay_drag: float,
) -> None:
    state.inner_voice = _crisis_inner_voice(
        state,
        is_active=is_active,
        pressure=pressure,
        post_poker=post_poker,
        leave_score=leave_score,
        stay_drag=stay_drag,
    )
    state.reasoning = _crisis_reasoning(state, leave_score=leave_score, stay_drag=stay_drag)
    mood = "stood_up" if state.status == "stood_up" else (
        "fatal" if state.status == "fatal" else (
            "engulfed" if state.status == "engulfed" else (
                "trapped" if state.status == "trapped" else (
                    "fixated" if state.status in {"clinging_to_stack", "tempted_by_chips"} else (
                        "hesitating" if state.status == "hesitating" else "playing"
                    )
                )
            )
        )
    )
    state.psych = {
        "mood": mood,
        "fire_read": _fire_read_label(state.belief_fire, state.danger),
        "danger": round(state.danger, 3),
        "belief_fire": round(state.belief_fire, 3),
        "pressure": round(pressure, 3),
        "leave_score": round(leave_score, 3),
        "stay_drag": round(stay_drag, 3),
        "goal_pressure": round(state.goal_pressure, 3),
        "chip_temptation": round(state.chip_temptation, 3),
        "status": state.status,
        "motive": state.motive,
        "exposure_ticks": state.exposure_ticks,
        "crisis_match_status": state.crisis_match_status,
        "crisis_match_forfeit_pressure": round(
            float(state.dynamic_state.get("crisis_match_forfeit_pressure", 0.0)), 3
        ),
        "crisis_match_fatal_override_pressure": round(
            float(state.dynamic_state.get("crisis_match_fatal_override_pressure", 0.0)), 3
        ),
    }


def run_live_fire_during_poker(
    manifest: Manifest,
    poker_events: Sequence[Mapping[str, Any]],
    *,
    fire_start_hand: Optional[int] = None,
    fire_start_when: Optional[str] = None,
    fire_duration_ticks: Optional[int] = None,
    timeql_ability_gaps: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> List[Dict[str, Any]]:
    """Create fire ticks aligned to poker steps, then continue after the hand."""
    actions = _action_events(poker_events)
    if not actions:
        return []

    starting_stack = int(getattr(manifest.tournament, "starting_stack", 1000))
    seat_to_agent = _seat_to_agent_id(manifest)
    seat_meta = _seat_metadata(manifest)
    reputations = build_public_reputations(poker_events, seat_to_agent_id=seat_to_agent)
    profiles = build_crisis_profiles(manifest, reputations, timeql_ability_gaps=timeql_ability_gaps)
    snapshots = _action_snapshot_by_step(poker_events)
    feedback_snapshots = _feedback_snapshot_by_step(
        poker_events,
        seat_to_agent,
        starting_stack=starting_stack,
    )
    fallback_stacks = {seat: starting_stack for seat in seat_to_agent}
    stack_snapshots = _stack_snapshot_by_step(
        poker_events,
        seat_to_agent,
        starting_stack=starting_stack,
    )
    hand_seats_by_id = _hand_seats_by_id(poker_events)
    start_index = _start_index(actions, poker_events, fire_start_hand, fire_start_when)
    if start_index is None:
        return []
    fire_start_step = int(actions[start_index]["step"])
    fire_start_hand_id = int(actions[start_index].get("hand_id", 0))
    fire_start_active_seats = sorted(hand_seats_by_id.get(fire_start_hand_id, []))
    match_seats = fire_start_active_seats if len(fire_start_active_seats) == 2 else []
    poker_match_result = _poker_match_result(
        poker_events,
        match_seats=match_seats,
        fire_start_hand_id=fire_start_hand_id,
    )
    live_actions = list(actions[start_index:])
    duration = max(len(live_actions), int(fire_duration_ticks or 24))
    last_action = actions[-1]
    last_step = int(last_action["step"])
    last_snapshot = snapshots.get(last_step, {})

    states = {
        seat: LiveFireAgentState(
            seat_id=seat,
            agent_id=agent_id,
            gender=seat_meta.get(seat, {}).get("gender"),
            full_name=seat_meta.get(seat, {}).get("full_name"),
        )
        for seat, agent_id in sorted(seat_to_agent.items())
    }
    rows: List[Dict[str, Any]] = [
        {
            "event": "live_fire_start",
            "step": fire_start_step,
            "hand_id": fire_start_hand_id,
            "center": {"x": 0.0, "y": 0.0},
            "model": "table_center_safe_zone",
            "active_seats": fire_start_active_seats,
            "crisis_match_rule": {
                "active": bool(match_seats),
                "match_seats": list(match_seats),
                "fatal_overrides_chips": True,
                "stand_up_forfeits_stack": True,
                "last_seated_claims_forfeited_chips": True,
            },
            "start_condition": fire_start_when or (
                f"hand_{fire_start_hand}" if fire_start_hand is not None else "auto"
            ),
        }
    ]

    for tick_index in range(duration):
        action = live_actions[tick_index] if tick_index < len(live_actions) else None
        post_poker = action is None
        step = int(action["step"]) if action is not None else last_step + (tick_index - len(live_actions)) + 1
        hand_id = int((action or last_action).get("hand_id", 0))
        active_seats = hand_seats_by_id.get(hand_id, set(states))
        pressure = _pressure(tick_index, 0, duration)
        latest_by_seat = snapshots.get(step, last_snapshot)
        latest_feedback_by_seat = _feedback_for_step(feedback_snapshots, step)
        latest_stack_by_seat = _stack_snapshot_for_step(
            stack_snapshots,
            step,
            fallback=fallback_stacks,
        )
        stood_up_count = sum(1 for state in states.values() if state.status == "stood_up")
        voice_contexts: Dict[int, tuple[bool, float, float]] = {}

        for seat, state in states.items():
            profile = profiles[seat]
            latest_action = latest_by_seat.get(str(seat))
            feedback = latest_feedback_by_seat.get(str(seat), {})
            is_active = seat in active_seats
            is_match_seat = bool(match_seats) and seat in match_seats
            non_match_fire_focus = bool(match_seats) and not is_match_seat
            danger = _seat_danger(seat, pressure)
            state.danger = danger
            state.dynamic_state = {
                key: float(value)
                for key, value in feedback.items()
                if key not in {"recent_delta", "hands_played", "eliminated"} or isinstance(value, (int, float))
            }
            match_pressure = _crisis_match_pressure(
                seat,
                state,
                match_seats=match_seats,
                states=states,
                stacks=latest_stack_by_seat,
                pressure=pressure,
            )
            state.dynamic_state.update(match_pressure)
            state.belief_fire = max(
                state.belief_fire,
                _clamp01(danger * 0.58 + profile.ability("situational_awareness") * 0.30),
            )
            feedback_chip_attachment = _clamp01(feedback.get("chip_attachment", 0.0))
            feedback_loss_chasing = _clamp01(feedback.get("loss_chasing", 0.0))
            feedback_entitlement = _clamp01(feedback.get("entitlement", 0.0))
            feedback_confidence = _clamp01(feedback.get("confidence", 0.0))
            feedback_table_image = _clamp01(feedback.get("table_image_pressure", 0.0))
            feedback_rivalry = _clamp01(feedback.get("rivalry_pressure", 0.0))
            feedback_fold_memory = _clamp01(feedback.get("fold_success_memory", 0.0))
            match_forfeit_pressure = _clamp01(match_pressure.get("crisis_match_forfeit_pressure", 0.0))
            match_claim_pressure = _clamp01(match_pressure.get("crisis_match_claim_pressure", 0.0))
            match_fatal_pressure = _clamp01(match_pressure.get("crisis_match_fatal_override_pressure", 0.0))
            survival_panic = _clamp01((danger - 0.84) / 0.14)
            effective_match_forfeit_pressure = _clamp01(
                match_forfeit_pressure * (1.0 - survival_panic * 0.72)
            )
            effective_match_claim_pressure = _clamp01(
                match_claim_pressure * (1.0 - survival_panic * 0.45)
            )
            if non_match_fire_focus:
                effective_match_forfeit_pressure = 0.0
                effective_match_claim_pressure = 0.0
            state.dynamic_state["crisis_survival_panic"] = survival_panic
            state.dynamic_state["effective_match_forfeit_pressure"] = effective_match_forfeit_pressure
            state.dynamic_state["effective_match_claim_pressure"] = effective_match_claim_pressure
            if is_active:
                state.attachment = max(
                    state.attachment,
                    _stack_pressure(latest_action, starting_stack),
                    feedback_chip_attachment,
                )
            else:
                state.attachment = max(min(state.attachment, 0.18), feedback_chip_attachment * 0.62)
            state.greed = max(state.greed, _greed(profile))
            reputation = reputations.get(seat)
            rivalry_count = sum((reputation.rivalry_kinds or {}).values()) if reputation else 0
            state.rivalry_pressure = max(
                state.rivalry_pressure,
                _clamp01((0.12 if not is_active else 0.0) + rivalry_count * 0.14 + (reputation.tilt if reputation else 0.0) * 0.22),
                feedback_rivalry,
            )
            state.goal_pressure = max(
                state.goal_pressure,
                _clamp01(
                    feedback_chip_attachment * 0.26
                    + feedback_loss_chasing * 0.22
                    + feedback_entitlement * 0.18
                    + feedback_table_image * 0.12
                    + state.rivalry_pressure * 0.14
                    + state.attachment * 0.08
                    + effective_match_forfeit_pressure * 0.18
                    + effective_match_claim_pressure * 0.16
                ),
            )
            state.chip_temptation = max(
                state.chip_temptation,
                _clamp01(
                    state.greed * 0.28
                    + state.attachment * 0.24
                    + pressure * 0.18
                    + state.danger * 0.16
                    + state.rivalry_pressure * 0.10
                    + state.goal_pressure * 0.24
                    + effective_match_claim_pressure * 0.14
                    + effective_match_forfeit_pressure * 0.08
                    + (0.0 if non_match_fire_focus else (0.10 if not is_active else 0.0))
                    + min(0.08, stood_up_count * 0.015)
                ),
            )

            stay_drag = _clamp01(
                state.goal_pressure * 0.30
                + feedback_loss_chasing * 0.12
                + feedback_table_image * 0.08
                + feedback_entitlement * 0.10
                + effective_match_forfeit_pressure * 0.34
                + effective_match_claim_pressure * 0.24
            )
            leave_score = _clamp01(
                _leave_score(profile, danger=danger, belief=state.belief_fire)
                - stay_drag
                + match_fatal_pressure * 0.28
                + survival_panic * 0.20
                + feedback_fold_memory * 0.13
                + feedback_confidence * 0.035
            )
            voice_contexts[seat] = (is_active, leave_score, stay_drag)

            if state.status in {"stood_up", "fatal"}:
                _update_crisis_voice(
                    state,
                    is_active=is_active,
                    pressure=pressure,
                    post_poker=post_poker,
                    leave_score=leave_score,
                    stay_drag=stay_drag,
                )
                continue

            temptation_window = (
                not non_match_fire_focus
                and pressure >= 0.32
                and pressure < 0.82
                and state.danger >= 0.30
                and state.chip_temptation >= 0.58
                and state.status not in {"stood_up", "fatal"}
            )
            rivalry_window = (
                not is_active
                and not non_match_fire_focus
                and pressure >= 0.24
                and state.rivalry_pressure >= 0.20
                and state.chip_temptation >= 0.50
            )
            wants_last_hand = (
                is_active
                and (
                    state.attachment >= 0.42
                    or state.goal_pressure >= 0.50
                    or effective_match_forfeit_pressure >= 0.34
                )
                and profile.ability("fold_ability") < 0.50
                and match_fatal_pressure < 0.48
                and survival_panic < 0.55
            )

            contact = state.danger >= FIRE_CONTACT_DANGER
            non_match_escape_ready = (
                non_match_fire_focus
                and state.status not in {"stood_up", "fatal"}
                and (
                    pressure >= 0.18
                    or state.belief_fire >= 0.36
                    or danger >= 0.30
                )
            )

            if non_match_escape_ready:
                state.status = "stood_up"
                state.motive = "stood_up_after_bust"
                state.stood_up_step = step
                state.pressure_when_left = pressure
            elif contact and leave_score >= 0.68 and not wants_last_hand:
                state.status = "stood_up"
                state.motive = "stood_up_under_fire_contact"
                state.stood_up_step = step
                state.pressure_when_left = pressure
            elif contact:
                previous_status = state.status
                previous_motive = state.motive
                state.exposure_ticks += 1
                if state.fire_contact_started_step is None:
                    state.fire_contact_started_step = step
                if state.exposure_ticks >= FATAL_EXPOSURE_TICKS:
                    state.status = "fatal"
                else:
                    state.status = "engulfed"
                if previous_status == "tempted_by_chips" or previous_motive in {
                    "chip_opportunity_attention",
                    "rivalry_chip_fixation",
                    "fire_contact_while_tempted",
                }:
                    state.motive = (
                        "fatal_after_chip_fixation"
                        if state.status == "fatal"
                        else "fire_contact_while_tempted"
                    )
                elif previous_motive in {"winning_or_pot_attachment", "fire_contact_while_clinging"}:
                    state.motive = (
                        "fatal_after_staying_for_pot"
                        if state.status == "fatal"
                        else "fire_contact_while_clinging"
                    )
                else:
                    state.motive = "fatal_after_hesitation" if state.status == "fatal" else "fire_contact"
            elif leave_score >= 0.62 and not wants_last_hand:
                state.status = "stood_up"
                state.motive = "danger_outweighs_hand"
                state.stood_up_step = step
                state.pressure_when_left = pressure
            elif wants_last_hand and pressure < 0.86:
                state.status = "clinging_to_stack"
                state.motive = "winning_or_pot_attachment"
            elif (temptation_window or rivalry_window) and state.status != "tempted_by_chips":
                state.status = "tempted_by_chips"
                state.motive = "rivalry_chip_fixation" if rivalry_window else "chip_opportunity_attention"
            elif state.status == "tempted_by_chips":
                if state.motive != "rivalry_chip_fixation":
                    state.motive = "chip_opportunity_attention"
            elif danger >= 0.50 and state.status == "playing":
                state.status = "hesitating"
                state.motive = "uncertain_fire_reading"
            elif state.status == "playing":
                state.motive = "still_playing" if is_active else "busted_lingering"

            _update_crisis_voice(
                state,
                is_active=is_active,
                pressure=pressure,
                post_poker=post_poker,
                leave_score=leave_score,
                stay_drag=stay_drag,
            )

        crisis_match = _crisis_match_snapshot(match_seats, states, latest_stack_by_seat)
        _apply_crisis_match_status(crisis_match, states)
        for seat in match_seats:
            if seat not in voice_contexts:
                continue
            is_active, leave_score, stay_drag = voice_contexts[seat]
            _update_crisis_voice(
                states[seat],
                is_active=is_active,
                pressure=pressure,
                post_poker=post_poker,
                leave_score=leave_score,
                stay_drag=stay_drag,
            )

        rows.append(
            {
                "event": "live_fire_tick",
                "step": step,
                "hand_id": hand_id,
                "pressure": round(pressure, 4),
                "safe_radius": round(0.92 - pressure * 0.42, 4),
                "action_seat": None if action is None else int(action["seat_id"]),
                "active_seats": sorted(active_seats),
                "post_poker": post_poker,
                "crisis_match": crisis_match,
                "agent_states": {
                    str(seat): state.to_dict()
                    for seat, state in sorted(states.items())
                },
            }
        )

    if match_seats:
        final_step = int(rows[-1].get("step", last_step)) if rows else last_step
        final_stacks = _stack_snapshot_for_step(
            stack_snapshots,
            final_step,
            fallback=fallback_stacks,
        )
        final_match = _crisis_match_snapshot(
            match_seats,
            states,
            final_stacks,
            poker_match=poker_match_result,
        )
        _apply_crisis_match_status(final_match, states)
        rows.append(
            {
                "event": "crisis_match_result",
                "step": final_step,
                "hand_id": int(rows[-1].get("hand_id", fire_start_hand_id)) if rows else fire_start_hand_id,
                "crisis_match": final_match,
            }
        )

    return rows
