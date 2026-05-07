"""Convert compiled TimeQL lack/body profiles into ALL-IN SMOKE ability gaps."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from poker_agents.voice_profile import build_identity_context, build_timeql_voice_profile


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

LACK_KEYS = (
    "belonging",
    "recognition",
    "loss",
    "trust",
    "protection",
    "freedom",
    "atonement",
    "connection",
)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _num(mapping: Mapping[str, Any], key: str, default: float = 0.5) -> float:
    value = mapping.get(key, default)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _lack_scores(profile: Mapping[str, Any]) -> Dict[str, float]:
    latents = profile.get("humanlm_latents")
    if not isinstance(latents, Mapping):
        return {}
    lack_profile = latents.get("agentspoker_lack")
    if not isinstance(lack_profile, Mapping):
        return {}
    raw_scores = lack_profile.get("lack_scores")
    if not isinstance(raw_scores, Mapping):
        return {}
    return {
        str(key): _clamp01(float(value))
        for key, value in raw_scores.items()
        if isinstance(value, (int, float))
    }


def _activation(profile: Mapping[str, Any]) -> float:
    traits = profile.get("simulation_traits")
    if isinstance(traits, Mapping) and isinstance(traits.get("activation_pressure"), (int, float)):
        return _clamp01(float(traits["activation_pressure"]))
    latents = profile.get("humanlm_latents")
    if isinstance(latents, Mapping):
        agentspoker_lack = latents.get("agentspoker_lack")
        if isinstance(agentspoker_lack, Mapping) and isinstance(agentspoker_lack.get("activation_pressure"), (int, float)):
            return _clamp01(float(agentspoker_lack["activation_pressure"]))
    return 0.5


def _precomputed_ability_gaps(profile: Mapping[str, Any]) -> Dict[str, float]:
    contrast = profile.get("agentspoker_lack_contrast")
    if not isinstance(contrast, Mapping):
        return {}
    raw_gaps = contrast.get("ability_gaps")
    if not isinstance(raw_gaps, Mapping):
        return {}
    return {
        key: round(_clamp01(float(raw_gaps[key])), 3)
        for key in ABILITY_KEYS
        if isinstance(raw_gaps.get(key), (int, float))
    }


def _compute_timeql_profile_to_ability_gaps(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Map a compiled TimeQL profile into crisis ability gaps.

    The output values are gaps, not abilities: 0.0 means the capability is
    present, 1.0 means it is strongly missing. This keeps the result compatible
    with `crisis_profile.ability_gaps` in the ALL-IN SMOKE manifest.
    """
    body = profile.get("body_vector") if isinstance(profile.get("body_vector"), Mapping) else {}
    traits = profile.get("simulation_traits") if isinstance(profile.get("simulation_traits"), Mapping) else {}
    lack = _lack_scores(profile)
    activation = _activation(profile)

    stability = _num(body, "stability")
    impulsivity = _num(body, "impulsivity")
    risk_sensitivity = _num(body, "risk_sensitivity")
    social_permeability = _num(body, "social_permeability")
    verification_need = _num(body, "verification_need")
    care_drive = _num(body, "care_drive")
    adaptability = _num(body, "adaptability")
    emotional_reactivity = _num(body, "emotional_reactivity")
    recovery_rate = _num(body, "recovery_rate")
    timing_sensitivity = _num(body, "timing_sensitivity")

    social_trust = _num(traits, "social_trust")
    self_verification = _num(traits, "self_verification")
    prosociality = _num(traits, "prosociality")
    urgency_bias = _num(traits, "urgency_bias")
    message_clarity = _num(traits, "message_clarity")
    reaction_delay = _num(traits, "reaction_delay_base", 4.0)
    panic_threshold = _num(traits, "panic_threshold")

    belonging = lack.get("belonging", 0.5)
    recognition = lack.get("recognition", 0.5)
    loss = lack.get("loss", 0.5)
    trust_lack = lack.get("trust", 0.5)
    protection = lack.get("protection", 0.5)
    freedom = lack.get("freedom", 0.5)
    atonement = lack.get("atonement", 0.5)
    connection = lack.get("connection", 0.5)

    trust_extreme = abs(social_trust - 0.5) * 2.0
    delay_gap = _clamp01((reaction_delay - 1.0) / 8.0)

    return {
        "fold_ability": round(_clamp01(
            impulsivity * 0.24
            + (1.0 - stability) * 0.22
            + recognition * 0.12
            + freedom * 0.12
            + urgency_bias * 0.12
            + activation * 0.10
            - self_verification * 0.10
        ), 3),
        "trust_calibration": round(_clamp01(
            trust_lack * 0.30
            + trust_extreme * 0.24
            + verification_need * 0.16
            + loss * 0.12
            + (1.0 - message_clarity) * 0.10
        ), 3),
        "help_seeking": round(_clamp01(
            (1.0 - social_permeability) * 0.26
            + (1.0 - message_clarity) * 0.22
            + connection * 0.16
            + belonging * 0.12
            + panic_threshold * 0.10
            - social_trust * 0.10
        ), 3),
        "situational_awareness": round(_clamp01(
            (1.0 - timing_sensitivity) * 0.30
            + delay_gap * 0.22
            + (1.0 - risk_sensitivity) * 0.18
            + activation * 0.10
            + (1.0 - self_verification) * 0.10
        ), 3),
        "self_control": round(_clamp01(
            impulsivity * 0.28
            + emotional_reactivity * 0.24
            + (1.0 - recovery_rate) * 0.22
            + urgency_bias * 0.12
            + activation * 0.10
        ), 3),
        "reciprocity": round(_clamp01(
            (1.0 - care_drive) * 0.24
            + (1.0 - prosociality) * 0.24
            + (1.0 - atonement) * 0.16
            + recognition * 0.12
            + (1.0 - connection) * 0.10
        ), 3),
        "public_responsibility": round(_clamp01(
            (1.0 - prosociality) * 0.28
            + (1.0 - care_drive) * 0.22
            + (1.0 - protection) * 0.16
            + recognition * 0.10
            + freedom * 0.08
            - atonement * 0.08
        ), 3),
        "meaning_update": round(_clamp01(
            (1.0 - adaptability) * 0.26
            + (1.0 - recovery_rate) * 0.18
            + trust_lack * 0.16
            + loss * 0.12
            + activation * 0.10
            + (1.0 - social_permeability) * 0.10
        ), 3),
    }


def convert_timeql_profile_to_ability_gaps(profile: Mapping[str, Any]) -> Dict[str, float]:
    """Return ALL-IN SMOKE ability gaps for a raw or contrasted TimeQL profile."""
    gaps = _compute_timeql_profile_to_ability_gaps(profile)
    gaps.update(_precomputed_ability_gaps(profile))
    return gaps


def _mean_by_key(rows: Sequence[Mapping[str, float]], keys: Iterable[str]) -> Dict[str, float]:
    means: Dict[str, float] = {}
    count = max(1, len(rows))
    for key in keys:
        means[key] = sum(float(row.get(key, 0.5)) for row in rows) / count
    return means


def _ranked(mapping: Mapping[str, float]) -> List[Tuple[str, float]]:
    return sorted(
        ((str(key), float(value)) for key, value in mapping.items()),
        key=lambda item: (-item[1], item[0]),
    )


def _contrast_value(
    value: float,
    *,
    table_mean: float,
    contrast_strength: float,
    baseline_pull: float,
) -> float:
    return _clamp01(
        0.5
        + (float(value) - float(table_mean)) * float(contrast_strength)
        + (float(value) - 0.5) * float(baseline_pull)
    )


def _contrast_lack_scores(
    lack_scores: Mapping[str, float],
    means: Mapping[str, float],
    *,
    contrast_strength: float,
    baseline_pull: float,
    primary_floor: float,
    secondary_floor: float,
    tertiary_floor: float,
    low_cap: float,
) -> Dict[str, float]:
    contrasted = {
        key: _contrast_value(
            float(lack_scores.get(key, 0.5)),
            table_mean=float(means.get(key, 0.5)),
            contrast_strength=contrast_strength,
            baseline_pull=baseline_pull,
        )
        for key in LACK_KEYS
    }
    ranking = _ranked(contrasted)
    floors = [primary_floor, secondary_floor, tertiary_floor]
    for index, floor in enumerate(floors):
        if index < len(ranking):
            key = ranking[index][0]
            contrasted[key] = max(contrasted[key], floor)
    for key, _ in ranking[-2:]:
        contrasted[key] = min(contrasted[key], low_cap)
    return {key: round(_clamp01(value), 3) for key, value in contrasted.items()}


def _contrast_ability_gaps(
    gaps: Mapping[str, float],
    means: Mapping[str, float],
    *,
    contrast_strength: float,
    baseline_pull: float,
    top_boost: float,
    second_boost: float,
    bottom_cut: float,
) -> Dict[str, float]:
    contrasted = {
        key: _contrast_value(
            float(gaps.get(key, 0.5)),
            table_mean=float(means.get(key, 0.5)),
            contrast_strength=contrast_strength,
            baseline_pull=baseline_pull,
        )
        for key in ABILITY_KEYS
    }
    ranking = _ranked(contrasted)
    if ranking:
        contrasted[ranking[0][0]] = min(0.92, contrasted[ranking[0][0]] + top_boost)
    if len(ranking) > 1:
        contrasted[ranking[1][0]] = min(0.88, contrasted[ranking[1][0]] + second_boost)
    for key, _ in ranking[-2:]:
        contrasted[key] = max(0.08, contrasted[key] - bottom_cut)
    return {key: round(_clamp01(value), 3) for key, value in contrasted.items()}


def _ensure_lack_profile(profile: Dict[str, Any]) -> Dict[str, Any]:
    latents = profile.setdefault("humanlm_latents", {})
    if not isinstance(latents, dict):
        latents = {}
        profile["humanlm_latents"] = latents
    lack_profile = latents.get("agentspoker_lack")
    if isinstance(lack_profile, dict):
        return lack_profile
    lack_profile = {}
    latents["agentspoker_lack"] = lack_profile
    return lack_profile


def _set_contrasted_lack_scores(profile: Dict[str, Any], lack_scores: Mapping[str, float]) -> None:
    lack_profile = _ensure_lack_profile(profile)
    ranking = _ranked(lack_scores)
    lack_profile["lack_scores"] = dict(lack_scores)
    if ranking:
        lack_profile["primary_lack"] = ranking[0][0]
        lack_profile["secondary_lacks"] = [
            {"lack": key, "score": round(value, 3)}
            for key, value in ranking[1:4]
        ]
        lack_profile["lack_severity"] = round(sum(value for _, value in ranking[:3]) / min(3, len(ranking)), 3)
        traits = profile.setdefault("simulation_traits", {})
        if isinstance(traits, dict):
            traits["primary_lack"] = ranking[0][0]
            traits["lack_severity"] = lack_profile["lack_severity"]


def compile_lack_contrast_profiles(
    profiles: Sequence[Mapping[str, Any]],
    *,
    lack_contrast_strength: float = 3.0,
    ability_contrast_strength: float = 3.0,
    lack_baseline_pull: float = 0.35,
    ability_baseline_pull: float = 0.25,
) -> List[Dict[str, Any]]:
    """Build AgentsPoker-specific profiles with table-relative lack contrast.

    Raw TimeQL output is intentionally preserved by callers. This function
    returns derived profiles that exaggerate within-table differences and embeds
    precomputed `ability_gaps` for ALL-IN SMOKE.
    """
    copied = [deepcopy(dict(profile)) for profile in profiles if isinstance(profile, Mapping)]
    raw_lack_rows = [
        {key: _lack_scores(profile).get(key, 0.5) for key in LACK_KEYS}
        for profile in copied
    ]
    lack_means = _mean_by_key(raw_lack_rows, LACK_KEYS)

    lack_adjusted_gaps: List[Dict[str, float]] = []
    raw_gap_rows: List[Dict[str, float]] = []
    for profile, raw_lack in zip(copied, raw_lack_rows):
        raw_gaps = _compute_timeql_profile_to_ability_gaps(profile)
        contrasted_lack = _contrast_lack_scores(
            raw_lack,
            lack_means,
            contrast_strength=lack_contrast_strength,
            baseline_pull=lack_baseline_pull,
            primary_floor=0.76,
            secondary_floor=0.66,
            tertiary_floor=0.58,
            low_cap=0.34,
        )
        _set_contrasted_lack_scores(profile, contrasted_lack)
        adjusted_gaps = _compute_timeql_profile_to_ability_gaps(profile)
        raw_gap_rows.append(raw_gaps)
        lack_adjusted_gaps.append(adjusted_gaps)
        query_meta = profile.get("query_meta") if isinstance(profile.get("query_meta"), Mapping) else {}
        profile["agentspoker_lack_contrast"] = {
            "version": 1,
            "source": "table_relative_v1_agentspoker" if isinstance(query_meta.get("timeql_v1_agentspoker"), Mapping) else "table_relative_agentspoker",
            "lack_contrast_strength": float(lack_contrast_strength),
            "ability_contrast_strength": float(ability_contrast_strength),
            "raw_lack_scores": raw_lack,
            "contrasted_lack_scores": contrasted_lack,
            "raw_ability_gaps": raw_gaps,
            "lack_adjusted_ability_gaps": adjusted_gaps,
        }

    gap_means = _mean_by_key(lack_adjusted_gaps, ABILITY_KEYS)
    for profile, adjusted_gaps in zip(copied, lack_adjusted_gaps):
        ability_gaps = _contrast_ability_gaps(
            adjusted_gaps,
            gap_means,
            contrast_strength=ability_contrast_strength,
            baseline_pull=ability_baseline_pull,
            top_boost=0.08,
            second_boost=0.04,
            bottom_cut=0.05,
        )
        ranking = _ranked(ability_gaps)
        contrast = profile["agentspoker_lack_contrast"]
        contrast["ability_gaps"] = ability_gaps
        contrast["dominant_crisis_gaps"] = [
            {"gap": key, "score": round(value, 3)}
            for key, value in ranking[:3]
        ]
    return copied


def load_timeql_ability_gap_overrides(path: Path, seat_by_agent_id: Optional[Mapping[int, int]] = None) -> Dict[int, Dict[str, float]]:
    """Load compiled profiles and return seat_id -> ability_gaps overrides."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        profiles = list(raw.values())
    elif isinstance(raw, list):
        profiles = raw
    else:
        raise ValueError("TimeQL profile file must contain a list or mapping.")

    seat_by_agent_id = {int(k): int(v) for k, v in (seat_by_agent_id or {}).items()}
    overrides: Dict[int, Dict[str, float]] = {}
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            continue
        identity = profile.get("identity_seed") if isinstance(profile.get("identity_seed"), Mapping) else {}
        raw_agent_id = identity.get("agent_id", index)
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError):
            agent_id = index
        seat_id = seat_by_agent_id.get(agent_id, agent_id)
        overrides[int(seat_id)] = convert_timeql_profile_to_ability_gaps(profile)
    return overrides


def load_timeql_voice_context_overrides(path: Path, seat_by_agent_id: Optional[Mapping[int, int]] = None) -> Dict[int, Dict[str, Any]]:
    """Load compiled profiles and return seat_id -> LLM identity/voice context."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, Mapping):
        profiles = list(raw.values())
    elif isinstance(raw, list):
        profiles = raw
    else:
        raise ValueError("TimeQL profile file must contain a list or mapping.")

    seat_by_agent_id = {int(k): int(v) for k, v in (seat_by_agent_id or {}).items()}
    overrides: Dict[int, Dict[str, Any]] = {}
    for index, profile in enumerate(profiles):
        if not isinstance(profile, Mapping):
            continue
        identity = profile.get("identity_seed") if isinstance(profile.get("identity_seed"), Mapping) else {}
        raw_agent_id = identity.get("agent_id", index)
        try:
            agent_id = int(raw_agent_id)
        except (TypeError, ValueError):
            agent_id = index
        seat_id = seat_by_agent_id.get(agent_id, agent_id)
        display_agent_id = str(identity.get("name") or agent_id)
        overrides[int(seat_id)] = {
            "identity_context": build_identity_context(
                agent_id=display_agent_id,
                metadata={},
                timeql_profile=profile,
            ),
            "voice_profile": build_timeql_voice_profile(profile),
        }
    return overrides
