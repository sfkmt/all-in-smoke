"""Normalize commentator text for Japanese TTS engines.

Cloud TTS (ElevenLabs, VOICEVOX, etc.) reading commentary aloud will mangle
poker shorthand if fed raw — `AKs` becomes "エーケーエス", `9Jo` becomes
random letters, `stoic-1` gets read as "ストイックマイナスイチ". The fix
is a structured pre-processing pass that turns every known short-form into
its spoken Japanese form before the audio request goes out.

Scope (in order of effect):

1. **Card combos** — pairs (`QQ`), suited (`AKs`), offsuit (`9Jo`).
2. **Single cards** — rank+suit (`Ah`, `7d`, `Tc`).
3. **Hyphenated rank lists** — `9-J`, `5-8-K` (TTS often reads each
   character separately and inserts long pauses).
4. **Agent IDs** — `<persona>-<seat>` → `<カタカナ>N番` for the six
   personas this project ships.
5. **Percent notation** — `70%` → `70パーセント`.

Order matters: longer patterns (3-char combos) run before 2-char patterns
to avoid mid-token overlap. All replacements are idempotent.
"""

from __future__ import annotations

import re
from typing import Dict


_RANK_JP: Dict[str, str] = {
    "A": "エース", "K": "キング", "Q": "クイーン", "J": "ジャック",
    "T": "10", "9": "9", "8": "8", "7": "7", "6": "6", "5": "5",
    "4": "4", "3": "3", "2": "2",
}

_SUIT_JP: Dict[str, str] = {
    "h": "ハート", "d": "ダイヤ", "c": "クラブ", "s": "スペード",
}

# Stable persona → katakana map. Add new personas here if the project grows.
_PERSONA_JP: Dict[str, str] = {
    "analyst": "アナリスト",
    "stoic": "ストイック",
    "needler": "ニードラー",
    "showman": "ショーマン",
    "gambler": "ギャンブラー",
    "veteran": "ベテラン",
    "qwen": "クウェン",
}


def _rank(c: str) -> str:
    return _RANK_JP.get(c.upper(), c)


def _replace_combo_suited(m: re.Match) -> str:
    return f"{_rank(m.group(1))}{_rank(m.group(2))} スーテッド"


def _replace_combo_offsuit(m: re.Match) -> str:
    return f"{_rank(m.group(1))}{_rank(m.group(2))} オフ"


def _replace_pair(m: re.Match) -> str:
    rank = _rank(m.group(1))
    return f"{rank}{rank}"


def _replace_single_card(m: re.Match) -> str:
    return f"{_rank(m.group(1))} {_SUIT_JP[m.group(2)]}"


def _replace_hyphen_ranks(m: re.Match) -> str:
    return " ".join(_rank(c) for c in m.group(0).split("-"))


def _replace_agent_id(m: re.Match) -> str:
    persona = m.group(1).lower()
    seat = m.group(2)
    if persona not in _PERSONA_JP:
        return m.group(0)
    return f"{_PERSONA_JP[persona]}{seat}番"


def _replace_percent(m: re.Match) -> str:
    return f"{m.group(1)}パーセント"


# Compile all patterns once. Order is significant — see module docstring.
_PATTERNS = [
    # 1. Card combos (3 chars: rank+rank+s/o). Word-boundary pinned so it
    #    doesn't bite mid-Japanese-katakana.
    (re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA])([2-9TJQKA])s(?![A-Za-z0-9])"), _replace_combo_suited),
    (re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA])([2-9TJQKA])o(?![A-Za-z0-9])"), _replace_combo_offsuit),

    # 2. Pair (2 chars, same rank). Run AFTER combos so QQs (if it appeared)
    #    isn't pre-eaten.
    (re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA])\1(?![A-Za-z0-9])"), _replace_pair),

    # 3. Single card (rank + lowercase suit). Must run after pair so `Ah`
    #    isn't shadowed by an `AA`-style pattern.
    (re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA])([hdcs])(?![A-Za-z0-9])"), _replace_single_card),

    # 4. Hyphenated rank lists (2-5 ranks). After single-card so we don't
    #    join `Ah-Kd` (already-suited cards).
    (re.compile(r"(?<![A-Za-z0-9])([2-9TJQKA])(?:-[2-9TJQKA]){1,4}(?![A-Za-z0-9])"), _replace_hyphen_ranks),

    # 5. Agent IDs.
    (re.compile(r"\b([A-Za-z]+)-(\d+)\b"), _replace_agent_id),

    # 6. Percent.
    (re.compile(r"(\d+)\s*%"), _replace_percent),
]


def normalize_for_tts(text: str) -> str:
    """Return `text` rewritten so a Japanese TTS speaks poker shorthand correctly."""
    out = text
    for pattern, repl in _PATTERNS:
        out = pattern.sub(repl, out)
    return out
