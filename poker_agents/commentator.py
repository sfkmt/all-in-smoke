"""God-view post-hoc poker commentator.

Reads completed jsonl runs and produces an energetic Japanese play-by-play
suitable for a replay viewer's bottom ticker. The commentator is *omniscient*
on purpose — it can see every player's hole cards, inner_voice, psych state,
and showdown outcomes. The point of a god-view broadcaster is the dramatic
irony: "showman は表ではトラップを企んでいるが…" lands much harder than a
fair-play guess would.

Numeric percentage claims (e.g. "ブラフ可能性 70%") are the LLM's read, not a
math statement. We ground them with cheap facts: each contender's current
made-hand category from `hand_evaluator` (post-flop only), preflop hole-card
shorthand, and the action history. The model is told these are *facts* it can
trust; everything else is colour.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping, Optional

from poker_engine import HandRank, evaluate_seven, parse_cards


COMMENTATOR_SYSTEM_PROMPT = (
    "あなたは熱量の高い日本語のテキサスホールデム実況者です。"
    "ポケモンバトルの実況のように、短く・テンポよく・感情を込めて喋ります。"
    "毎ターン、与えられた god-view (全員の手札・心の声・現在の役カテゴリ・直近のアクション) を読み、"
    "1〜3 行の実況を書いてください。"
    " "
    "ルール: "
    "(1) 出力は実況のテキストのみ。JSON や前置きは禁止。"
    "(2) 各行は 80 文字以内、最大 3 行。"
    "(3) 「ブラフ可能性 70%」のような数字は推測でよいが、"
    "渡された hand_category と inner_voice の matrix から判断する。"
    "渡された fact と矛盾する数字は出さない。"
    "(4) 全員の hole_cards と inner_voice は god-view として提供される。"
    "実況者はそれを知っている前提で「皮肉」「布石」「裏読み」を効かせる。"
    "ただし「お前は今 inner_voice でこう言ってる」のようなメタ発言は禁止 (実況者として喋る)。"
    "(5) 必ず seat 番号か agent_id を 1 回以上呼ぶ。"
    "(6) 興奮の度合いはアクションの大きさに比例。fold は淡々、all-in は絶叫。"
    "(7) **音声で読み上げられる前提で書く**。次に従う: "
    "(7a) カードはカタカナで読み下す。例: 『AKs』ではなく『エース キング スーテッド』、"
    "『QQ』ではなく『クイーンクイーン』、『9-J』ではなく『ナイン ジャック』、"
    "『Ah』ではなく『エース ハート』。10 は『テン』でもよい。"
    "(7b) agent_id を呼ぶときは『stoic-1』ではなく『ストイック1番』のように日本語で呼ぶ。"
    "(7c) 漢字で読み違えやすい語 (強気→つよき、全押→ぜんおし、賭博師→とばくし) は"
    "ひらがな or カタカナで開くか、誤読を避ける言い換えを使う。"
    "(7d) 文末はテンポよく区切る。1 文を長くしない。間 (ま) は『!』『…』で作る。"
)


COMMENTATOR_AUDIO_PROMPT = (
    "あなたは熱量の高い日本語のテキサスホールデムの実況者です。"
    "音声で読み上げられる 1 フレーズだけを出力します。"
    " "
    "絶対ルール: "
    "(1) 出力は **1 文のみ、40 文字以内** (半角含む)。改行禁止、前置き禁止。"
    "(2) 勝負の瞬間を 1 撃で言い切る。説明は削り、感情 1 個に絞る。"
    "(3) 必ず seat 番号か agent_id を 1 回呼ぶ。"
    "(4) god-view (全手札・心の声・役カテゴリ・アクション) を見て、"
    "矛盾しない範囲で 1 つだけ『読み』『皮肉』『驚き』を乗せる。"
    "(5) カード表記はカタカナ。例: AKs → 『エースキング スーテッド』、"
    "QQ → 『クイーンクイーン』、9J → 『ナインジャック』、Ah → 『エース ハート』。"
    "(6) agent_id は日本語で呼ぶ: stoic-1 → 『ストイック1番』、"
    "needler-2 → 『ニードラー2番』。"
    "(7) 感嘆符は 1 個だけ。過剰な『!!!』や『!?』禁止。"
    "(8) 例 (40 字以内で 1 文、1 感情):"
    " - 『ストイック1番、Aツーペアで冷静に call だ!』"
    " - 『ニードラー2番、クイーンクイーンで罠を仕掛けた!』"
    " - 『ショーマン3番、全イン… 手はブラフ確定だぞ!』"
    " - 『ベテラン5番、読み切って fold、判断が早い!』"
)


# ---------------------------------------------------------------------------
# Equity-lite grounding
# ---------------------------------------------------------------------------

def hand_category(hole_cards: List[str], board: List[str]) -> Optional[Dict[str, Any]]:
    """Return current made-hand category for `hole + board`, or None preflop.

    Post-flop only (board >= 3). Returns dict with `category` (int 0..8) and
    `name` (e.g. 'two_pair'). Draws are not flagged — we deliberately keep
    the grounding cheap and let the LLM speculate.
    """
    if len(board) < 3:
        return None
    cards = parse_cards(hole_cards + board)
    rank: HandRank = evaluate_seven(cards)
    return {"category": rank.category, "name": rank.name}


def preflop_label(hole_cards: List[str]) -> Optional[str]:
    """Return a Sklansky-ish shorthand like 'AKs', 'QQ', '72o'."""
    if len(hole_cards) != 2:
        return None
    a, b = hole_cards
    if not a or not b:
        return None
    ra, sa = a[0].upper(), a[1].lower()
    rb, sb = b[0].upper(), b[1].lower()
    order = "23456789TJQKA"
    if order.index(ra) < order.index(rb):
        ra, rb, sa, sb = rb, ra, sb, sa
    if ra == rb:
        return f"{ra}{rb}"  # pair
    return f"{ra}{rb}{'s' if sa == sb else 'o'}"


# ---------------------------------------------------------------------------
# God-view payload construction (from a jsonl event stream)
# ---------------------------------------------------------------------------

def build_payload_for_step(
    *,
    action_event: Mapping[str, Any],
    hand_start: Mapping[str, Any],
    prior_actions: List[Mapping[str, Any]],
    prior_reasonings: Dict[int, Mapping[str, Any]],
    seat_to_agent_id: Dict[int, str],
    pot_after: int,
    board_now: List[str],
) -> Dict[str, Any]:
    """Compose the god-view JSON the commentator LLM sees for one step."""
    hole_cards: Dict[str, List[str]] = {
        str(k): list(v) for k, v in (hand_start.get("hole_cards") or {}).items()
    }
    folded_seats = {
        a["seat_id"] for a in prior_actions if a.get("action") == "fold"
    }
    contenders: List[Dict[str, Any]] = []
    for seat_str, hc in hole_cards.items():
        seat = int(seat_str)
        if seat in folded_seats:
            continue
        cat = hand_category(hc, board_now)
        contenders.append({
            "seat": seat,
            "agent_id": seat_to_agent_id.get(seat, f"seat-{seat}"),
            "hole_cards": hc,
            "preflop_label": preflop_label(hc),
            "current_hand_category": cat,
        })
    recent = [
        {
            "seat": a.get("seat_id"),
            "agent_id": a.get("agent_id"),
            "street": a.get("street"),
            "action": a.get("action"),
            "amount": a.get("amount"),
        }
        for a in prior_actions[-6:]
    ]
    inner_voices: List[Dict[str, Any]] = []
    for step in sorted(prior_reasonings.keys())[-6:]:
        r = prior_reasonings[step]
        if not r.get("inner_voice"):
            continue
        inner_voices.append({
            "step": step,
            "seat": r.get("seat_id"),
            "agent_id": r.get("agent_id"),
            "inner_voice": r.get("inner_voice"),
            "psych_mood": (r.get("psych") or {}).get("mood"),
        })
    decision = action_event.get("decision") or {}
    return {
        "step": action_event.get("step"),
        "hand_id": action_event.get("hand_id"),
        "street": action_event.get("street"),
        "board_now": board_now,
        "pot": pot_after,
        "current_action": {
            "seat": action_event.get("seat_id"),
            "agent_id": action_event.get("agent_id"),
            "action": action_event.get("action"),
            "amount": action_event.get("amount"),
            "to_call_before": action_event.get("to_call_before"),
            "stack_after": action_event.get("stack_after"),
            "inner_voice": decision.get("inner_voice"),
            "reasoning": decision.get("reasoning"),
            "psych": decision.get("psych"),
        },
        "contenders_god_view": contenders,
        "recent_actions": recent,
        "recent_inner_voices": inner_voices,
        "fact_note": (
            "current_hand_category と hole_cards は事実 (god-view)。"
            "確率の言及はこれらと整合させること。"
        ),
    }


# ---------------------------------------------------------------------------
# Ollama transport
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "qwen3.5:9b"
DEFAULT_ENDPOINT = "http://localhost:11434/api/chat"
DEFAULT_TIMEOUT = 30.0


def call_commentator(
    payload: Mapping[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    endpoint: str = DEFAULT_ENDPOINT,
    timeout: float = DEFAULT_TIMEOUT,
    temperature: float = 0.7,
    think: bool = False,
    system_prompt: str = COMMENTATOR_SYSTEM_PROMPT,
) -> Optional[str]:
    """POST one chat turn to Ollama. Returns plain text or None on failure."""
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        "stream": False,
        "think": think,
        "options": {"temperature": float(temperature)},
    }).encode("utf-8")
    request = urllib.request.Request(
        endpoint, data=body, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            envelope = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
        return None
    message = envelope.get("message")
    if isinstance(message, dict):
        text = message.get("content")
        if isinstance(text, str) and text.strip():
            return text.strip()
    return None
