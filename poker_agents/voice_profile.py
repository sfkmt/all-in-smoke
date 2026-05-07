"""Build LLM voice context from identity metadata and TimeQL profiles."""

from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, List, Mapping, Optional, Tuple


LACK_VOICE_NOTES = {
    "belonging": "場から外れる感覚に敏感。心の声では、孤立・席の空気・自分の居場所への反応が漏れやすい。",
    "recognition": "評価や面子に敏感。心の声では、見られ方・格好悪さ・勝って証明したい気持ちが混じりやすい。",
    "loss": "失ったものや取り返しへの反応が強い。心の声では、過去の負け・残ったチップ・手放しづらさが出やすい。",
    "trust": "信用と証拠に敏感。心の声では、相手の意図・本物かどうか・裏を取る必要が出やすい。",
    "protection": "安全と守りに敏感。心の声では、危険の見積もり・守るべき線・被害を避ける判断が出やすい。",
    "freedom": "縛られる感覚に敏感。心の声では、窮屈さ・押し返したい衝動・流れを変えたい欲が出やすい。",
    "atonement": "責任や帳尻に敏感。心の声では、借りを返す・ミスを修正する・筋を通す意識が出やすい。",
    "connection": "声や接点に敏感。心の声では、誰かの反応・会話の糸口・つながりの途切れが気になりやすい。",
}

GAP_VOICE_NOTES = {
    "fold_ability": "降りる判断に摩擦が出る。手放す悔しさ、まだ残れる理由探し、損切りへの抵抗が漏れやすい。",
    "trust_calibration": "相手の発言や場の異変をすぐには信じ切れない。確証を求める独白が入りやすい。",
    "help_seeking": "助けを求める・声をかける動きが遅れやすい。内側で処理してしまう。",
    "situational_awareness": "卓外の変化への切り替えが遅れやすい。違和感を勝負のノイズとして処理しがち。",
    "self_control": "衝動が漏れやすい。短い高揚、苛立ち、急な方針転換が心の声に出る。",
    "reciprocity": "他者への返報より、自分の勝負や損得が先に立ちやすい。",
    "public_responsibility": "全体への警告や場の責任を引き受けるまでに迷いが出やすい。",
    "meaning_update": "一度置いた読みを更新しづらい。新情報を見ても、前の解釈に戻りやすい。",
}

BODY_STYLE_NOTES = (
    ("communication_directness", 0.58, "言い切りが増える。短く直接的な語尾を使う。"),
    ("verification_need", 0.58, "根拠確認の言葉が増える。断定前に一拍置く。"),
    ("emotional_reactivity", 0.58, "感情の温度が漏れる。ただし大げさなキャラ口調にはしない。"),
    ("impulsivity", 0.58, "反応が速く、短い衝動語が混じる。"),
    ("timing_sensitivity", 0.58, "タイミングや間合いへの言及が増える。"),
    ("stability", 0.58, "声が落ち着く。状況を一文で整理してから動く。"),
)


def build_identity_context(
    *,
    agent_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
    timeql_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Return identity context for prompt injection.

    Age and gender are exposed as self-context, not as stereotype drivers.
    """
    metadata = metadata or {}
    identity = {}
    if isinstance(timeql_profile, Mapping):
        seed = timeql_profile.get("identity_seed")
        if isinstance(seed, Mapping):
            identity.update({str(k): v for k, v in seed.items() if v is not None})
    for key in (
        "display_name",
        "name",
        "full_name",
        "family_name",
        "given_name",
        "gender",
        "age",
        "birth_date",
        "datetime",
        "location",
        "timezone",
    ):
        value = metadata.get(key)
        if value is not None and key not in identity:
            identity[key] = value
    display_name = str(identity.get("display_name") or identity.get("name") or agent_id)
    identity["agent_id"] = str(agent_id)
    identity["display_name"] = display_name
    if "age" not in identity:
        parsed_age = _parse_age(display_name)
        if parsed_age is None:
            parsed_age = _age_from_birth_date(identity.get("birth_date"))
        if parsed_age is not None:
            identity["age"] = parsed_age
    if "given_name" not in identity:
        identity["given_name"] = _given_name(identity)
    return _compact(identity)


def build_timeql_voice_profile(profile: Mapping[str, Any]) -> Dict[str, Any]:
    """Build a compact, prompt-friendly voice profile from a TimeQL profile."""
    identity = profile.get("identity_seed") if isinstance(profile.get("identity_seed"), Mapping) else {}
    body = profile.get("body_vector") if isinstance(profile.get("body_vector"), Mapping) else {}
    traits = profile.get("simulation_traits") if isinstance(profile.get("simulation_traits"), Mapping) else {}
    latents = profile.get("humanlm_latents") if isinstance(profile.get("humanlm_latents"), Mapping) else {}
    lack_profile = latents.get("agentspoker_lack") if isinstance(latents.get("agentspoker_lack"), Mapping) else {}
    contrast = profile.get("agentspoker_lack_contrast") if isinstance(profile.get("agentspoker_lack_contrast"), Mapping) else {}

    lack_scores = _numeric_mapping(
        contrast.get("contrasted_lack_scores")
        if isinstance(contrast.get("contrasted_lack_scores"), Mapping)
        else lack_profile.get("lack_scores")
    )
    ability_gaps = _numeric_mapping(
        contrast.get("ability_gaps")
        if isinstance(contrast.get("ability_gaps"), Mapping)
        else {}
    )
    top_lacks = _ranked(lack_scores)[:3]
    top_gaps = _ranked(ability_gaps)[:3]
    primary_lack = str(lack_profile.get("primary_lack") or (top_lacks[0][0] if top_lacks else ""))

    directives: List[str] = []
    for lack, score in top_lacks:
        note = LACK_VOICE_NOTES.get(lack)
        if note and score >= 0.5:
            directives.append(note)
    for gap, score in top_gaps:
        note = GAP_VOICE_NOTES.get(gap)
        if note and score >= 0.48:
            directives.append(note)
    directives.extend(_body_style_directives(body))
    if not directives:
        directives.append("心の声は現在の判断、直前の勝敗、場の異変に即して短く出す。固定キャラ口調に寄せない。")

    return _compact(
        {
            "source": "timeql_agentspoker_lack_v1",
            "identity_name": identity.get("name"),
            "primary_lack": primary_lack,
            "top_lacks": [
                {"lack": key, "score": round(value, 3)}
                for key, value in top_lacks
            ],
            "dominant_ability_gaps": [
                {"gap": key, "score": round(value, 3)}
                for key, value in top_gaps
            ],
            "state_bias": {
                "activation_state": traits.get("activation_state") or lack_profile.get("activation_state"),
                "activation_pressure": _round_optional(traits.get("activation_pressure") or lack_profile.get("activation_pressure")),
                "self_control_base": _round_optional(traits.get("self_control_base")),
                "warning_trust_baseline": _round_optional(traits.get("warning_trust_baseline")),
                "danger_verification_need": _round_optional(traits.get("danger_verification_need")),
                "chip_attachment_base": _round_optional(traits.get("chip_attachment_base")),
                "loss_chasing_base": _round_optional(traits.get("loss_chasing_base")),
            },
            "style_axes": {
                "directness": _round_optional(body.get("communication_directness")),
                "verification_need": _round_optional(body.get("verification_need")),
                "emotional_reactivity": _round_optional(body.get("emotional_reactivity")),
                "impulsivity": _round_optional(body.get("impulsivity")),
                "stability": _round_optional(body.get("stability")),
                "timing_sensitivity": _round_optional(body.get("timing_sensitivity")),
            },
            "inner_voice_directives": directives[:6],
        }
    )


def build_agent_context(
    *,
    agent_id: str,
    metadata: Optional[Mapping[str, Any]] = None,
    identity_context: Optional[Mapping[str, Any]] = None,
    voice_profile: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    identity = build_identity_context(agent_id=agent_id, metadata=metadata)
    if isinstance(identity_context, Mapping):
        identity.update({str(k): v for k, v in identity_context.items() if v is not None})
    context = {"identity_context": _compact(identity)}
    if isinstance(voice_profile, Mapping):
        context["voice_profile"] = _compact(dict(voice_profile))
    return context


def _body_style_directives(body: Mapping[str, Any]) -> List[str]:
    notes: List[str] = []
    for key, threshold, note in BODY_STYLE_NOTES:
        value = body.get(key)
        if isinstance(value, (int, float)) and float(value) >= threshold:
            notes.append(note)
    if isinstance(body.get("communication_directness"), (int, float)) and float(body["communication_directness"]) <= 0.42:
        notes.append("言葉を少し迂回させる。言い切りすぎず、内側で測る感じを残す。")
    if isinstance(body.get("recovery_rate"), (int, float)) and float(body["recovery_rate"]) <= 0.42:
        notes.append("直前の損失や違和感を引きずりやすい。")
    return notes


def _numeric_mapping(value: Any) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        return {}
    result = {}
    for key, raw in value.items():
        if isinstance(raw, bool):
            continue
        if isinstance(raw, (int, float)):
            result[str(key)] = max(0.0, min(1.0, float(raw)))
    return result


def _ranked(mapping: Mapping[str, float]) -> List[Tuple[str, float]]:
    return sorted(mapping.items(), key=lambda item: (-item[1], item[0]))


def _compact(value: Any) -> Any:
    if isinstance(value, Mapping):
        compacted = {}
        for key, item in value.items():
            next_value = _compact(item)
            if next_value is not None and next_value != {} and next_value != []:
                compacted[str(key)] = next_value
        return compacted
    if isinstance(value, list):
        compacted_list = []
        for item in value:
            next_value = _compact(item)
            if next_value is not None:
                compacted_list.append(next_value)
        return compacted_list
    return value


def _round_optional(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return round(float(value), 3)
    return None


def _parse_age(display_name: str) -> Optional[int]:
    match = re.search(r"\((\d{1,3})\)", str(display_name))
    if not match:
        return None
    return int(match.group(1))


def _age_from_birth_date(value: Any, *, today: Optional[date] = None) -> Optional[int]:
    if not value:
        return None
    today = today or date.today()
    try:
        year, month, day = (int(part) for part in str(value)[:10].split("-"))
    except (TypeError, ValueError):
        return None
    age = today.year - year
    if (today.month, today.day) < (month, day):
        age -= 1
    return age


def _given_name(identity: Mapping[str, Any]) -> Optional[str]:
    full_name = identity.get("full_name")
    if isinstance(full_name, str) and full_name.strip():
        parts = full_name.split()
        if parts:
            return parts[-1]
    display = identity.get("display_name") or identity.get("name")
    if isinstance(display, str) and display.strip():
        return re.sub(r"\(\d{1,3}\)", "", display).strip() or None
    return None
