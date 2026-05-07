"""Normalize card references in agent-facing Japanese text."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from poker_agents.base import AgentDecision
from poker_engine.cards import RANK_LABELS_JA, SUIT_LABELS_JA


_RANK = r"([2-9TJQKA])"
_RANKS = set("23456789TJQKA")
_SUIT_ABBREVIATIONS = {
    "スペ": "スペード",
    "ダイ": "ダイヤ",
    "クラ": "クラブ",
    "ハー": "ハート",
    "ダ": "ダイヤ",
    "ク": "クラブ",
    "ハ": "ハート",
}


def normalize_card_language(text: Optional[str]) -> Optional[str]:
    """Rewrite poker shorthand into natural Japanese card wording."""
    if not isinstance(text, str):
        return text
    out = text
    out = out.replace("オフスーツ", "オフスート")
    out = out.replace("オフスートスート", "オフスート")
    out = out.replace("オフスートオフスート", "オフスート")
    out = out.replace("スートオフ", "オフスート")
    out = out.replace("フロッシュドロー", "フラッシュドロー")
    out = re.sub(r"(?<![A-Za-z0-9]){0}{0}s(?![A-Za-z0-9])".format(_RANK), _replace_suited_combo, out)
    out = re.sub(r"(?<![A-Za-z0-9]){0}{0}o(?![A-Za-z0-9])".format(_RANK), _replace_offsuit_combo, out)
    out = re.sub(
        r"(?<![A-Za-z0-9]){0}{0}(オフ|オフスート|スーテッド)(?![A-Za-z0-9])".format(_RANK),
        _replace_rank_combo_with_suffix,
        out,
    )
    out = re.sub(
        r"(スーテッド|オフスート)([A2-9TJQK]|10)と([A2-9TJQK]|10)",
        _replace_suffix_rank_combo,
        out,
    )
    out = re.sub(
        r"オフスートの([A2-9TJQK]|10)と([A2-9TJQK]|10)",
        _replace_offsuit_prefix_combo,
        out,
    )
    out = re.sub(
        r"([A2-9TJQK]|10)(スペード|ダイヤ|クラブ|ハート)([A2-9TJQK]|10)",
        _replace_rank_suit_rank,
        out,
    )
    out = re.sub(
        r"([A2-9TJQK]|10)(スペード|ダイヤ|クラブ|ハート)",
        _replace_rank_suit,
        out,
    )
    out = re.sub(
        r"(?<![A-Za-z0-9]){0}([cdhs]){0}([cdhs])(?![A-Za-z0-9])".format(_RANK),
        _replace_two_single_cards,
        out,
    )
    out = re.sub(
        r"(?<![A-Za-z0-9]){0}{0}([cdhs])(?![A-Za-z0-9])".format(_RANK),
        _replace_same_suit_code_combo,
        out,
    )
    out = re.sub(r"(?<![A-Za-z0-9]){0}([cdhs])(?![A-Za-z0-9])".format(_RANK), _replace_single_card, out)
    out = re.sub(
        r"(スペード|ダイヤ|クラブ|ハート)([A2-9TJQK]|10)-([A2-9TJQK]|10)",
        _replace_suit_hyphen_combo,
        out,
    )
    out = re.sub(
        r"(?<![A-Za-z0-9]){0}-{0}(オフ|オフスート|スーテッド)?(?![A-Za-z0-9])".format(_RANK),
        _replace_hyphen_combo,
        out,
    )
    out = re.sub(
        r"(スペード|ダイヤ|クラブ|ハート)([A2-9TJQK]|10)([A2-9TJQK]|10)(の)?(スーテッド|オフスート)?",
        _replace_suit_rank_combo,
        out,
    )
    out = re.sub(
        r"(スペード|ダイヤ|クラブ|ハート)([A2-9TJQK]|10)と([A2-9TJQK]|10)の\1スーテッド",
        _replace_redundant_suit_combo,
        out,
    )
    out = re.sub(
        r"(?<![A-Za-z0-9]){0}{0}(スペード|ダイヤ|クラブ|ハート)(?![A-Za-z0-9])".format(_RANK),
        _replace_same_suit_combo,
        out,
    )
    out = re.sub(r"(?<![A-Za-z0-9]){0}{0}(?![A-Za-z0-9])".format(_RANK), _replace_rank_combo, out)
    for short, full in _SUIT_ABBREVIATIONS.items():
        out = re.sub(r"{0}([A2-9TJQK]|10)".format(short), r"{0}\1".format(full), out)
    out = out.replace("オフスートスート", "オフスート")
    out = out.replace("オフスートオフスート", "オフスート")
    return out


def normalize_decision_card_language(decision: AgentDecision) -> AgentDecision:
    """Normalize card wording in all user-facing decision text fields."""
    decision.reasoning = normalize_card_language(decision.reasoning)
    decision.memory = normalize_card_language(decision.memory)
    decision.inner_voice = normalize_card_language(decision.inner_voice)
    if isinstance(decision.table_talk, dict):
        text = decision.table_talk.get("text")
        if isinstance(text, str):
            decision.table_talk["text"] = normalize_card_language(text)
    if isinstance(decision.psych, dict):
        _normalize_mapping_text(decision.psych)
    return decision


def _replace_suited_combo(match: re.Match) -> str:
    return "{0}と{1}のスーテッド".format(
        _rank_label(match.group(1)),
        _rank_label(match.group(2)),
    )


def _replace_offsuit_combo(match: re.Match) -> str:
    return "{0}と{1}のオフスート".format(
        _rank_label(match.group(1)),
        _rank_label(match.group(2)),
    )


def _replace_single_card(match: re.Match) -> str:
    return "{0}{1}".format(SUIT_LABELS_JA[match.group(2)], _rank_label(match.group(1)))


def _replace_two_single_cards(match: re.Match) -> str:
    return "{0}{1}・{2}{3}".format(
        SUIT_LABELS_JA[match.group(2)],
        _rank_label(match.group(1)),
        SUIT_LABELS_JA[match.group(4)],
        _rank_label(match.group(3)),
    )


def _replace_rank_suit(match: re.Match) -> str:
    return "{0}{1}".format(match.group(2), _rank_label(match.group(1)))


def _replace_rank_suit_rank(match: re.Match) -> str:
    return "{0}{1}と{2}".format(
        match.group(2),
        _rank_label(match.group(1)),
        _rank_label(match.group(3)),
    )


def _replace_rank_combo(match: re.Match) -> str:
    return "{0}と{1}".format(_rank_label(match.group(1)), _rank_label(match.group(2)))


def _replace_rank_combo_with_suffix(match: re.Match) -> str:
    suffix = "オフスート" if match.group(3) == "オフ" else match.group(3)
    return "{0}と{1}の{2}".format(
        _rank_label(match.group(1)),
        _rank_label(match.group(2)),
        suffix,
    )


def _replace_suffix_rank_combo(match: re.Match) -> str:
    return "{0}と{1}の{2}".format(
        _rank_label(match.group(2)),
        _rank_label(match.group(3)),
        match.group(1),
    )


def _replace_offsuit_prefix_combo(match: re.Match) -> str:
    return "{0}と{1}のオフスート".format(
        _rank_label(match.group(1)),
        _rank_label(match.group(2)),
    )


def _replace_same_suit_code_combo(match: re.Match) -> str:
    return "{0}と{1}の{2}スーテッド".format(
        _rank_label(match.group(1)),
        _rank_label(match.group(2)),
        SUIT_LABELS_JA[match.group(3)],
    )


def _replace_hyphen_combo(match: re.Match) -> str:
    suffix = match.group(3)
    base = "{0}と{1}".format(_rank_label(match.group(1)), _rank_label(match.group(2)))
    if not suffix:
        return base
    suffix = "オフスート" if suffix == "オフ" else suffix
    return "{0}の{1}".format(base, suffix)


def _replace_suit_hyphen_combo(match: re.Match) -> str:
    return "{0}と{1}の{2}スーテッド".format(
        _rank_label(match.group(2)),
        _rank_label(match.group(3)),
        match.group(1),
    )


def _replace_same_suit_combo(match: re.Match) -> str:
    return "{0}と{1}の{2}スーテッド".format(
        _rank_label(match.group(1)),
        _rank_label(match.group(2)),
        match.group(3),
    )


def _replace_suit_rank_combo(match: re.Match) -> str:
    suffix = match.group(5)
    if suffix:
        return "{0}{1}と{2}の{3}".format(
            match.group(1),
            _rank_label(match.group(2)),
            _rank_label(match.group(3)),
            suffix,
        )
    return "{0}{1}と{2}".format(
        match.group(1),
        _rank_label(match.group(2)),
        _rank_label(match.group(3)),
    )


def _replace_redundant_suit_combo(match: re.Match) -> str:
    return "{0}と{1}の{2}スーテッド".format(
        _rank_label(match.group(2)),
        _rank_label(match.group(3)),
        match.group(1),
    )


def _rank_label(rank: str) -> str:
    value = rank.upper()
    if value not in _RANKS:
        return rank
    return RANK_LABELS_JA[value]


def _normalize_mapping_text(mapping: Dict[str, Any]) -> None:
    for key, value in list(mapping.items()):
        if isinstance(value, str):
            mapping[key] = normalize_card_language(value)
        elif isinstance(value, dict):
            _normalize_mapping_text(value)
