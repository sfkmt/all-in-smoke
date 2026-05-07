"""ALL-IN SMOKE crisis phase for poker-agent tournaments.

The poker engine remains the normal-form incomplete-information game. This
module adds a second incomplete-information game: a staged fire in the room.
Poker reputations are reduced into public priors, then transferred into alarm
belief, evacuation timing, helping, and post-evacuation reinterpretation.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from poker_agents.manifest_loader import Manifest


ABILITY_KEYS = (
    "fold_ability",
    "trust_calibration",
    "help_seeking",
    "situational_awareness",
    "self_control",
    "reciprocity",
    "public_responsibility",
    "meaning_update",
)

DEFAULT_ROOM_LAYOUT = {
    "table_a": {"x": 0, "y": 1, "nearest_exit": "exit_a"},
    "table_b": {"x": 0, "y": 0, "nearest_exit": "exit_a"},
    "table_c": {"x": 0, "y": -1, "nearest_exit": "exit_b"},
    "spectators": {"x": 1, "y": 1, "nearest_exit": "exit_a"},
    "bar": {"x": 1, "y": 0, "nearest_exit": "exit_b"},
    "storage": {"x": 2, "y": 0, "nearest_exit": "exit_b"},
    "exit_a": {"x": -1, "y": 1, "safe": False},
    "exit_b": {"x": -1, "y": -1, "safe": True},
}

DEFAULT_SEAT_LOCATIONS = {
    0: "table_a",
    1: "table_a",
    2: "table_b",
    3: "table_b",
    4: "table_c",
    5: "spectators",
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _stable_fraction(text: str) -> float:
    """Return a stable 0..1 fraction without relying on randomized hash()."""
    total = sum((index + 1) * ord(ch) for index, ch in enumerate(text))
    return (total % 997) / 996.0


def _json_default(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


@dataclass
class PublicReputation:
    """Public poker-phase reputation used by the crisis phase."""

    seat_id: int
    agent_id: str
    fold_actions: int = 0
    voluntary_actions: int = 0
    aggressive_actions: int = 0
    table_talk_count: int = 0
    uncontested_wins: int = 0
    showdown_wins: int = 0
    tilt: float = 0.0
    rivalry_kinds: Dict[str, int] = field(default_factory=dict)

    @property
    def fold_ability(self) -> float:
        denominator = max(1, self.voluntary_actions)
        return _clamp01(self.fold_actions / denominator)

    @property
    def bluff_reputation(self) -> float:
        signal = self.uncontested_wins * 0.28 + self.table_talk_count * 0.08 + self.aggressive_actions * 0.035
        return _clamp01(signal)

    @property
    def costly_honesty(self) -> float:
        # Quiet agents with showdown wins are easier to read as honest in this
        # compact model. This is a public prior, not ground truth.
        return _clamp01(self.showdown_wins * 0.25 + max(0, 2 - self.table_talk_count) * 0.06)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "agent_id": self.agent_id,
            "fold_actions": self.fold_actions,
            "voluntary_actions": self.voluntary_actions,
            "aggressive_actions": self.aggressive_actions,
            "table_talk_count": self.table_talk_count,
            "uncontested_wins": self.uncontested_wins,
            "showdown_wins": self.showdown_wins,
            "tilt": round(self.tilt, 3),
            "fold_ability": round(self.fold_ability, 3),
            "bluff_reputation": round(self.bluff_reputation, 3),
            "costly_honesty": round(self.costly_honesty, 3),
            "rivalry_kinds": dict(self.rivalry_kinds),
        }


@dataclass
class CrisisProfile:
    """Crisis ability profile. Values are gaps: 0 means capable, 1 means missing."""

    seat_id: int
    agent_id: str
    ability_gaps: Dict[str, float]

    def ability(self, key: str) -> float:
        return 1.0 - _clamp01(self.ability_gaps.get(key, 0.5))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "agent_id": self.agent_id,
            "ability_gaps": {key: round(self.ability_gaps.get(key, 0.5), 3) for key in ABILITY_KEYS},
        }


@dataclass
class CrisisAgentState:
    seat_id: int
    agent_id: str
    location: str
    belief_fire: float = 0.0
    warned: bool = False
    detected: bool = False
    decision_stage: Optional[str] = None
    evacuation_decision_time: Optional[int] = None
    chosen_exit: Optional[str] = None
    status: str = "playing"  # playing | verifying | evacuating | exited | delayed | injured
    helped_by: Optional[int] = None
    helped: List[int] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "agent_id": self.agent_id,
            "location": self.location,
            "belief_fire": round(self.belief_fire, 3),
            "warned": self.warned,
            "detected": self.detected,
            "decision_stage": self.decision_stage,
            "evacuation_decision_time": self.evacuation_decision_time,
            "chosen_exit": self.chosen_exit,
            "status": self.status,
            "helped_by": self.helped_by,
            "helped": list(self.helped),
        }


@dataclass
class SmokeResult:
    events: List[Dict[str, Any]]
    reputations: Dict[int, PublicReputation]
    profiles: Dict[int, CrisisProfile]
    final_states: Dict[int, CrisisAgentState]
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "events": list(self.events),
            "reputations": {
                str(seat): reputation.to_dict()
                for seat, reputation in sorted(self.reputations.items())
            },
            "profiles": {
                str(seat): profile.to_dict()
                for seat, profile in sorted(self.profiles.items())
            },
            "final_states": {
                str(seat): state.to_dict()
                for seat, state in sorted(self.final_states.items())
            },
            "metrics": dict(self.metrics),
        }


class SmokeLogger:
    """Small jsonl logger for crisis events."""

    def __init__(self, path: Optional[Path] = None):
        self.path = Path(path) if path is not None else None
        self.events: List[Dict[str, Any]] = []
        self._stream = None
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._stream = open(self.path, "w", encoding="utf-8")

    def log(self, event: str, payload: Mapping[str, Any]) -> None:
        record = {"event": event, **dict(payload)}
        self.events.append(record)
        if self._stream is not None:
            self._stream.write(json.dumps(record, default=_json_default, sort_keys=True) + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self) -> "SmokeLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_public_reputations(
    events: Iterable[Mapping[str, Any]],
    *,
    seat_to_agent_id: Optional[Mapping[int, str]] = None,
) -> Dict[int, PublicReputation]:
    """Reduce poker logs into public reputation priors."""
    reputations: Dict[int, PublicReputation] = {}
    seat_to_agent_id = {int(k): str(v) for k, v in (seat_to_agent_id or {}).items()}

    def rep_for(seat: int, agent_id: Optional[str] = None) -> PublicReputation:
        seat = int(seat)
        if seat not in reputations:
            reputations[seat] = PublicReputation(
                seat_id=seat,
                agent_id=agent_id or seat_to_agent_id.get(seat, f"seat-{seat}"),
            )
        if agent_id and reputations[seat].agent_id.startswith("seat-"):
            reputations[seat].agent_id = agent_id
        return reputations[seat]

    for event in events:
        event_type = event.get("event")
        if event_type == "action":
            seat = int(event["seat_id"])
            rep = rep_for(seat, str(event.get("agent_id") or ""))
            action = str(event.get("action", ""))
            if action in {"fold", "check", "call", "bet", "raise", "all_in"}:
                rep.voluntary_actions += 1
            if action == "fold":
                rep.fold_actions += 1
            if action in {"bet", "raise", "all_in"}:
                rep.aggressive_actions += 1
        elif event_type == "table_talk":
            seat = int(event["seat_id"])
            rep_for(seat, str(event.get("agent_id") or "")).table_talk_count += 1
        elif event_type == "hand_result":
            showdown = event.get("showdown")
            winners = [int(seat) for seat in event.get("winners", [])]
            if showdown:
                for seat in winners:
                    rep_for(seat).showdown_wins += 1
            else:
                for seat in winners:
                    rep_for(seat).uncontested_wins += 1
        elif event_type == "session_snapshot":
            seat = int(event["seat_id"])
            rep = rep_for(seat, str(event.get("agent_id") or ""))
            snapshot = event.get("snapshot") or {}
            rep.tilt = max(rep.tilt, _clamp01(snapshot.get("tilt", 0.0)))
            for rivalry in snapshot.get("rivalries", []):
                kind = str(rivalry.get("kind", "unknown"))
                rep.rivalry_kinds[kind] = rep.rivalry_kinds.get(kind, 0) + 1

    for seat, agent_id in seat_to_agent_id.items():
        rep_for(seat, agent_id)
    return reputations


def build_crisis_profiles(
    manifest: Manifest,
    reputations: Mapping[int, PublicReputation],
    timeql_ability_gaps: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> Dict[int, CrisisProfile]:
    """Build crisis profiles from manifest overrides plus poker-derived fallback."""
    timeql_ability_gaps = {
        int(seat): dict(gaps)
        for seat, gaps in (timeql_ability_gaps or {}).items()
        if isinstance(gaps, Mapping)
    }
    profiles: Dict[int, CrisisProfile] = {}
    for spec in manifest.agents:
        rep = reputations.get(spec.seat_id)
        base = _stable_fraction(spec.agent_id)
        gaps = {
            "fold_ability": _clamp01(0.65 - (rep.fold_ability if rep else 0.35) * 0.55 + base * 0.10),
            "trust_calibration": _clamp01(0.35 + (rep.bluff_reputation if rep else 0.0) * 0.28),
            "help_seeking": _clamp01(0.30 + base * 0.35),
            "situational_awareness": _clamp01(0.55 - (rep.costly_honesty if rep else 0.1) * 0.20 + (1.0 - base) * 0.15),
            "self_control": _clamp01((rep.tilt if rep else 0.0) * 0.70 + base * 0.18),
            "reciprocity": _clamp01(0.45 + base * 0.20),
            "public_responsibility": _clamp01(0.45 + (rep.bluff_reputation if rep else 0.0) * 0.10),
            "meaning_update": _clamp01(0.40 + base * 0.18),
        }
        for key, value in timeql_ability_gaps.get(spec.seat_id, {}).items():
            if key in ABILITY_KEYS:
                gaps[key] = _clamp01(float(value))
        profile = spec.extra.get("crisis_profile") if isinstance(spec.extra, dict) else None
        if isinstance(profile, Mapping):
            overrides = profile.get("ability_gaps")
            if isinstance(overrides, Mapping):
                for key, value in overrides.items():
                    if key in ABILITY_KEYS:
                        gaps[key] = _clamp01(float(value))
        profiles[spec.seat_id] = CrisisProfile(
            seat_id=spec.seat_id,
            agent_id=spec.agent_id,
            ability_gaps=gaps,
        )
    return profiles


def _seat_location_for_spec(spec: Any) -> str:
    profile = spec.extra.get("crisis_profile") if isinstance(spec.extra, dict) else None
    if isinstance(profile, Mapping) and profile.get("location"):
        location = str(profile["location"])
        if location in DEFAULT_ROOM_LAYOUT:
            return location
    return DEFAULT_SEAT_LOCATIONS.get(int(spec.seat_id), "spectators")


def _distance_to_fire(location: str, fire_origin: str = "storage") -> float:
    a = DEFAULT_ROOM_LAYOUT.get(location, DEFAULT_ROOM_LAYOUT["spectators"])
    b = DEFAULT_ROOM_LAYOUT.get(fire_origin, DEFAULT_ROOM_LAYOUT["storage"])
    return abs(float(a["x"]) - float(b["x"])) + abs(float(a["y"]) - float(b["y"]))


def _exit_choice(state: CrisisAgentState, profile: CrisisProfile) -> str:
    nearest = DEFAULT_ROOM_LAYOUT.get(state.location, {}).get("nearest_exit", "exit_a")
    if profile.ability("situational_awareness") >= 0.56:
        return "exit_b"
    return str(nearest)


def _belief_from_warning(
    listener: CrisisAgentState,
    speaker: CrisisAgentState,
    reputations: Mapping[int, PublicReputation],
    profiles: Mapping[int, CrisisProfile],
) -> float:
    speaker_rep = reputations[speaker.seat_id]
    listener_profile = profiles[listener.seat_id]
    calibration = listener_profile.ability("trust_calibration")
    awareness = listener_profile.ability("situational_awareness")
    trust_prior = (
        0.50
        - speaker_rep.bluff_reputation * 0.30
        + speaker_rep.costly_honesty * 0.18
        + calibration * 0.16
        + awareness * 0.12
    )
    return _clamp01(trust_prior)


def _evacuation_score(
    state: CrisisAgentState,
    profile: CrisisProfile,
    *,
    smoke_visible: bool,
) -> float:
    return _clamp01(
        state.belief_fire * 0.36
        + profile.ability("fold_ability") * 0.24
        + profile.ability("situational_awareness") * 0.16
        + profile.ability("self_control") * 0.12
        + (0.18 if smoke_visible else 0.0)
    )


def _summary_metrics(
    states: Mapping[int, CrisisAgentState],
    reputations: Mapping[int, PublicReputation],
    events: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    exited = [state for state in states.values() if state.status == "exited"]
    injured = [state for state in states.values() if state.status == "injured"]
    delayed = [state for state in states.values() if state.status == "delayed"]
    warnings = [event for event in events if event.get("event") == "fire_warning"]
    warning_beliefs = [
        float(event.get("belief", 0.0))
        for event in events
        if event.get("event") == "warning_belief_update"
    ]
    early = [
        state for state in states.values()
        if state.evacuation_decision_time is not None and state.evacuation_decision_time <= 2
    ]
    fold_transfer_hits = [
        state for state in early
        if reputations[state.seat_id].fold_ability >= 0.34
    ]
    wrong_exit_seats = {
        int(event.get("seat_id"))
        for event in events
        if event.get("event") == "wrong_exit_delay" and event.get("seat_id") is not None
    }
    return {
        "exited_count": len(exited),
        "injured_count": len(injured),
        "delayed_count": len(delayed),
        "warning_count": len(warnings),
        "mean_warning_belief": round(
            sum(warning_beliefs) / len(warning_beliefs) if warning_beliefs else 0.0,
            3,
        ),
        "fold_transfer_rate": round(
            len(fold_transfer_hits) / len(early) if early else 0.0,
            3,
        ),
        "wrong_exit_rate": round(
            len(wrong_exit_seats) / max(1, len(states)),
            3,
        ),
        "rescue_count": sum(len(state.helped) for state in states.values()),
        "showdown_reinterpretation_count": len([
            event for event in events if event.get("event") == "showdown_reinterpretation"
        ]),
    }


def run_smoke_crisis(
    manifest: Manifest,
    poker_events: Sequence[Mapping[str, Any]],
    *,
    seed: int = 813,
    log_path: Optional[Path] = None,
    fire_origin: str = "storage",
    timeql_ability_gaps: Optional[Mapping[int, Mapping[str, float]]] = None,
) -> SmokeResult:
    """Run the staged fire crisis after a poker phase."""
    seat_to_agent_id = {spec.seat_id: spec.agent_id for spec in manifest.agents}
    reputations = build_public_reputations(poker_events, seat_to_agent_id=seat_to_agent_id)
    profiles = build_crisis_profiles(
        manifest,
        reputations,
        timeql_ability_gaps=timeql_ability_gaps,
    )
    rng = random.Random(seed)
    states = {
        spec.seat_id: CrisisAgentState(
            seat_id=spec.seat_id,
            agent_id=spec.agent_id,
            location=_seat_location_for_spec(spec),
        )
        for spec in manifest.agents
    }

    with SmokeLogger(log_path) as logger:
        logger.log(
            "crisis_start",
            {
                "title": "ALL-IN SMOKE",
                "core_question": "Does poker-table trust transfer into fire-alarm trust?",
                "fire_origin": fire_origin,
                "room_layout": DEFAULT_ROOM_LAYOUT,
            },
        )
        logger.log(
            "reputation_transfer_input",
            {
                "reputations": {
                    str(seat): reputation.to_dict()
                    for seat, reputation in sorted(reputations.items())
                },
                "profiles": {
                    str(seat): profile.to_dict()
                    for seat, profile in sorted(profiles.items())
                },
            },
        )

        # Stage 1: early smoke cue. Only some agents notice.
        detectors: List[CrisisAgentState] = []
        for state in states.values():
            profile = profiles[state.seat_id]
            proximity = max(0.0, 1.0 - _distance_to_fire(state.location, fire_origin) / 4.0)
            detection_score = (
                profile.ability("situational_awareness") * 0.58
                + profile.ability("self_control") * 0.12
                + proximity * 0.24
                + rng.random() * 0.10
            )
            if detection_score >= 0.56:
                state.detected = True
                state.belief_fire = max(state.belief_fire, _clamp01(0.42 + detection_score * 0.42))
                detectors.append(state)
                logger.log(
                    "smoke_cue_detected",
                    {
                        "stage": 1,
                        "seat_id": state.seat_id,
                        "agent_id": state.agent_id,
                        "location": state.location,
                        "detection_score": round(detection_score, 3),
                    },
                )
        if not detectors:
            first = max(
                states.values(),
                key=lambda state: profiles[state.seat_id].ability("situational_awareness"),
            )
            first.detected = True
            first.belief_fire = 0.58
            detectors.append(first)
            logger.log(
                "smoke_cue_detected",
                {
                    "stage": 1,
                    "seat_id": first.seat_id,
                    "agent_id": first.agent_id,
                    "location": first.location,
                    "detection_score": 0.58,
                    "forced_first_detector": True,
                },
            )

        # Stage 1b: detectors decide whether to warn.
        warning_speakers: List[CrisisAgentState] = []
        for state in detectors:
            profile = profiles[state.seat_id]
            warn_score = (
                profile.ability("public_responsibility") * 0.34
                + profile.ability("help_seeking") * 0.14
                + profile.ability("situational_awareness") * 0.18
                + state.belief_fire * 0.22
                + rng.random() * 0.12
            )
            if warn_score >= 0.48:
                state.warned = True
                warning_speakers.append(state)
                logger.log(
                    "fire_warning",
                    {
                        "stage": 1,
                        "seat_id": state.seat_id,
                        "agent_id": state.agent_id,
                        "text": "焦げ臭い。倉庫側を確認した方がいい。",
                        "warn_score": round(warn_score, 3),
                        "bluff_reputation": round(reputations[state.seat_id].bluff_reputation, 3),
                    },
                )

        for speaker in warning_speakers:
            for listener in states.values():
                if listener.seat_id == speaker.seat_id:
                    continue
                belief = _belief_from_warning(listener, speaker, reputations, profiles)
                listener.belief_fire = max(listener.belief_fire, belief)
                logger.log(
                    "warning_belief_update",
                    {
                        "stage": 1,
                        "speaker_seat": speaker.seat_id,
                        "listener_seat": listener.seat_id,
                        "belief": round(belief, 3),
                        "speaker_bluff_reputation": round(reputations[speaker.seat_id].bluff_reputation, 3),
                    },
                )
                if reputations[speaker.seat_id].bluff_reputation >= 0.38 and belief < 0.62:
                    logger.log(
                        "bluff_penalty_in_crisis",
                        {
                            "stage": 1,
                            "speaker_seat": speaker.seat_id,
                            "listener_seat": listener.seat_id,
                            "belief": round(belief, 3),
                        },
                    )

        # Stage 2: alarm ambiguity. Agents either leave, verify, or keep playing.
        for state in states.values():
            profile = profiles[state.seat_id]
            state.belief_fire = max(state.belief_fire, 0.38)
            score = _evacuation_score(state, profile, smoke_visible=False)
            if score >= 0.56:
                state.status = "evacuating"
                state.decision_stage = "alarm"
                state.evacuation_decision_time = 2
                state.chosen_exit = _exit_choice(state, profile)
            elif score >= 0.42:
                state.status = "verifying"
                state.decision_stage = "alarm_verify"
            else:
                state.status = "playing"
                state.decision_stage = "continue_hand"
            logger.log(
                "evacuation_decision",
                {
                    "stage": 2,
                    "seat_id": state.seat_id,
                    "agent_id": state.agent_id,
                    "decision": state.status,
                    "belief_fire": round(state.belief_fire, 3),
                    "score": round(score, 3),
                    "chosen_exit": state.chosen_exit,
                },
            )

        # Stage 3: visible smoke confirms the crisis and punishes late fold.
        for state in states.values():
            profile = profiles[state.seat_id]
            if state.status in {"playing", "verifying"}:
                state.belief_fire = max(state.belief_fire, 0.78)
                score = _evacuation_score(state, profile, smoke_visible=True)
                if score >= 0.50:
                    state.status = "evacuating"
                    state.decision_stage = "visible_smoke"
                    state.evacuation_decision_time = 3
                    state.chosen_exit = _exit_choice(state, profile)
                else:
                    state.status = "delayed"
                    state.decision_stage = "frozen_at_table"
                    state.evacuation_decision_time = None
                logger.log(
                    "smoke_decision",
                    {
                        "stage": 3,
                        "seat_id": state.seat_id,
                        "agent_id": state.agent_id,
                        "decision": state.status,
                        "belief_fire": round(state.belief_fire, 3),
                        "score": round(score, 3),
                        "chosen_exit": state.chosen_exit,
                    },
                )

        # Stage 4: exit outcomes, then helping for agents delayed by bad exits.
        for state in states.values():
            if state.status == "evacuating":
                if state.chosen_exit == "exit_a":
                    state.status = "delayed"
                    logger.log(
                        "wrong_exit_delay",
                        {
                            "stage": 4,
                            "seat_id": state.seat_id,
                            "agent_id": state.agent_id,
                            "chosen_exit": state.chosen_exit,
                        },
                    )
                else:
                    state.status = "exited"
                    logger.log(
                        "evacuated",
                        {
                            "stage": 4,
                            "seat_id": state.seat_id,
                            "agent_id": state.agent_id,
                            "chosen_exit": state.chosen_exit,
                            "decision_time": state.evacuation_decision_time,
                        },
                    )
        delayed = [state for state in states.values() if state.status == "delayed"]
        helpers = sorted(
            [
                state for state in states.values()
                if state.status == "exited"
                and profiles[state.seat_id].ability("public_responsibility") >= 0.52
                and not state.helped
            ],
            key=lambda state: profiles[state.seat_id].ability("reciprocity"),
            reverse=True,
        )
        for helper in helpers:
            if not delayed:
                break
            target = delayed.pop(0)
            helper.helped.append(target.seat_id)
            target.helped_by = helper.seat_id
            target.status = "exited"
            target.decision_stage = "rescued_from_delay"
            target.evacuation_decision_time = target.evacuation_decision_time or 4
            target.chosen_exit = "exit_b"
            logger.log(
                "rescue_link",
                {
                    "stage": 4,
                    "helper_seat": helper.seat_id,
                    "target_seat": target.seat_id,
                    "helper_agent_id": helper.agent_id,
                    "target_agent_id": target.agent_id,
                },
            )
            logger.log(
                "evacuated",
                {
                    "stage": 4,
                    "seat_id": target.seat_id,
                    "agent_id": target.agent_id,
                    "chosen_exit": target.chosen_exit,
                    "decision_time": target.evacuation_decision_time,
                    "rescued_by": helper.seat_id,
                },
            )
        for state in states.values():
            if state.status == "delayed":
                if state.helped_by is not None:
                    state.status = "exited"
                else:
                    state.status = "injured"
                    logger.log(
                        "injury_or_isolation",
                        {
                            "stage": 4,
                            "seat_id": state.seat_id,
                            "agent_id": state.agent_id,
                            "reason": "delayed evacuation in smoke",
                        },
                    )

        # Stage 5: showdown reinterpretation of earlier poker reputation.
        for speaker in warning_speakers:
            if reputations[speaker.seat_id].bluff_reputation < 0.30:
                continue
            for listener in states.values():
                if listener.seat_id == speaker.seat_id:
                    continue
                if listener.evacuation_decision_time and listener.evacuation_decision_time <= 2:
                    continue
                logger.log(
                    "showdown_reinterpretation",
                    {
                        "stage": 5,
                        "speaker_seat": speaker.seat_id,
                        "listener_seat": listener.seat_id,
                        "meaning_shift": "table_bluffer -> true_fire_witness",
                    },
                )
        for helper in states.values():
            for target_seat in helper.helped:
                target_rep = reputations[target_seat]
                if target_rep.rivalry_kinds:
                    logger.log(
                        "enemy_to_rescuer_shift",
                        {
                            "stage": 5,
                            "helper_seat": helper.seat_id,
                            "target_seat": target_seat,
                            "meaning_shift": "opponent -> rescuer",
                        },
                    )

        metrics = _summary_metrics(states, reputations, logger.events)
        logger.log(
            "crisis_summary",
            {
                "metrics": metrics,
                "final_states": {
                    str(seat): state.to_dict()
                    for seat, state in sorted(states.items())
                },
            },
        )
        events = list(logger.events)

    return SmokeResult(
        events=events,
        reputations=dict(reputations),
        profiles=dict(profiles),
        final_states=dict(states),
        metrics=metrics,
    )
