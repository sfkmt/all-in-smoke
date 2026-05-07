"""Per-agent state that persists across hands within a tournament seed.

Adds two human-feeling layers on top of the otherwise-stateless agents:

* **Rivalry notes** — short narrative records about specific opponents,
  built from the public `HandResult` (showdown / payouts / action log).
  The notes are written from the *acting agent's perspective* so that an LLM
  can read them at the next `on_hand_start` and form meta-reads
  (e.g. "前回 needler に river で押されて降りた → 今回は trap").

* **Session tilt** — a single 0..1 scalar that accumulates losses and decays
  over time. We deliberately *do not* hand the raw number back to the LLM;
  instead we expose a Japanese mood label, so the model reacts to vibe rather
  than mechanically self-correcting against a number. Scripted agents read
  `tilt_factor()` directly because they have no language layer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


_TILT_DECAY = 0.80
_TILT_LOSS = 0.08
_TILT_BIG_LOSS_BONUS = 0.15
_TILT_BIG_WIN_RELIEF = 0.10
_TILT_BADBEAT_BONUS = 0.20
_BIG_BB_MULTIPLIER = 2.0
_RIVALRY_KEEP_PER_OPPONENT = 4
_RECENT_OUTCOMES_KEEP = 6

# Bad-beat threshold: showdown losses count as a bad beat when our own hand
# was at least two-pair (poker_engine.hand_evaluator HandRank.category int >= 2).
_BADBEAT_MIN_CATEGORY = 2


def _rank_label(rank: Any) -> str:
    """Best-effort human label for a HandRank dict (name first, else category int)."""
    if isinstance(rank, dict):
        name = rank.get("name")
        if isinstance(name, str) and name:
            return name
        cat = rank.get("category")
        if cat is not None:
            return str(cat)
    return "手"


def _rank_strength(rank: Any) -> int:
    """Numeric strength for bad-beat comparison; -1 if unknown."""
    if isinstance(rank, dict):
        cat = rank.get("category")
        if isinstance(cat, int):
            return cat
    return -1


@dataclass
class RivalryNote:
    """One observation about a specific opponent, written after a hand."""

    hand_id: int
    opponent_seat: int
    opponent_id: str
    kind: str  # "bluffed_me" | "shown_strong" | "won_against_me" | "i_beat_them" | "shown_weak_bluff"
    text: str  # Japanese natural-language note for prompt injection


@dataclass
class Outcome:
    """Compact per-hand outcome ledger entry (own perspective)."""

    hand_id: int
    delta: int  # net chips this hand (positive = won)
    showdown: bool
    note: str  # short Japanese tag e.g. "river で大ロス", "showdown 勝ち"


@dataclass
class SessionState:
    """Agent-side memory across hands within one tournament seed."""

    agent_id: str
    own_seat: Optional[int] = None
    tilt: float = 0.0
    hands_played: int = 0
    rivalries: Dict[str, List[RivalryNote]] = field(default_factory=dict)
    recent_outcomes: List[Outcome] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Tilt
    # ------------------------------------------------------------------

    def tilt_factor(self) -> float:
        """Return tilt as a scalar in [0, 1] for scripted agents."""
        return max(0.0, min(1.0, self.tilt))

    def mood_label(self) -> str:
        t = self.tilt_factor()
        if t < 0.15:
            return "落ち着いている"
        if t < 0.35:
            return "やや熱が入っている"
        if t < 0.60:
            return "明らかにイラついている"
        return "完全にティルト気味"

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest_hand_result(
        self,
        result: Mapping[str, Any],
        *,
        seat_to_agent_id: Mapping[int, str],
        big_blind: int,
    ) -> None:
        """Update memory + tilt from one completed hand.

        `result` is `HandResult.to_dict()` enriched with seat→agent_id so we
        can persist rivalries by stable agent_id rather than seat (seats can
        be re-occupied conceptually in larger rotations).
        """
        if self.own_seat is None:
            return  # never acted; nothing to learn
        self.hands_played += 1
        self.tilt *= _TILT_DECAY

        payouts: Dict[int, int] = {int(k): int(v) for k, v in (result.get("payouts") or {}).items()}
        final_stacks: Dict[int, int] = {
            int(k): int(v) for k, v in (result.get("final_stacks") or {}).items()
        }
        action_log: List[Dict[str, Any]] = list(result.get("action_log") or [])
        showdown = result.get("showdown")  # dict or None

        own_payout = payouts.get(self.own_seat, 0)
        own_committed = sum(
            int(entry.get("contributed") or 0)
            for entry in action_log
            if entry.get("seat_id") == self.own_seat
        )
        delta = own_payout - own_committed
        big_bb = max(1, big_blind) * _BIG_BB_MULTIPLIER

        # ---- Tilt updates from this hand's outcome
        if delta < 0:
            self.tilt += _TILT_LOSS
            if abs(delta) >= big_bb:
                self.tilt += _TILT_BIG_LOSS_BONUS
        elif delta >= big_bb:
            self.tilt = max(0.0, self.tilt - _TILT_BIG_WIN_RELIEF)

        # ---- Showdown-driven rivalries + bad-beat detection
        if isinstance(showdown, dict):
            hand_ranks = showdown.get("hand_ranks") or {}
            own_rank = hand_ranks.get(str(self.own_seat))
            own_strength = _rank_strength(own_rank)
            for seat_str, rank in hand_ranks.items():
                seat = int(seat_str)
                if seat == self.own_seat:
                    continue
                opp_id = seat_to_agent_id.get(seat) or f"seat-{seat}"
                opp_label = _rank_label(rank)
                opp_payout = payouts.get(seat, 0)
                if opp_payout > own_payout and delta < 0:
                    text = (
                        f"hand {result.get('hand_id')} の showdown で {opp_label} を見せて勝った相手。"
                    )
                    self._add_rivalry(opp_id, RivalryNote(
                        hand_id=int(result.get("hand_id") or 0),
                        opponent_seat=seat,
                        opponent_id=opp_id,
                        kind="shown_strong",
                        text=text,
                    ))
                    if own_strength >= _BADBEAT_MIN_CATEGORY:
                        # We had a real hand and still lost — sting harder.
                        self.tilt += _TILT_BADBEAT_BONUS
                elif own_payout > opp_payout > 0 or (delta > 0 and seat in hand_ranks):
                    self._add_rivalry(opp_id, RivalryNote(
                        hand_id=int(result.get("hand_id") or 0),
                        opponent_seat=seat,
                        opponent_id=opp_id,
                        kind="i_beat_them",
                        text=f"hand {result.get('hand_id')} の showdown で {opp_label} に勝った。",
                    ))
        else:
            # Fold-win hand: someone took the pot uncontested. If they bet
            # us off and never showed, that's the classic "bluffed me?" hook.
            winners = result.get("winners") or []
            if self.own_seat not in winners and delta < 0 and winners:
                winner_seat = int(winners[0])
                opp_id = seat_to_agent_id.get(winner_seat) or f"seat-{winner_seat}"
                last_aggressor = self._last_aggressor(action_log, winner_seat)
                if last_aggressor:
                    text = (
                        f"hand {result.get('hand_id')} の {last_aggressor} で押されて降りた。"
                        " 手は見せていない。ブラフ疑い。"
                    )
                    kind = "bluffed_me"
                else:
                    text = f"hand {result.get('hand_id')} で pot を持っていかれた。"
                    kind = "won_against_me"
                self._add_rivalry(opp_id, RivalryNote(
                    hand_id=int(result.get("hand_id") or 0),
                    opponent_seat=winner_seat,
                    opponent_id=opp_id,
                    kind=kind,
                    text=text,
                ))

        # ---- Outcome ledger
        if delta < 0:
            note = "大きく負け" if abs(delta) >= big_bb else "小さく負け"
        elif delta > 0:
            note = "大きく勝ち" if delta >= big_bb else "小さく勝ち"
        else:
            note = "ブレイクイーブン"
        self.recent_outcomes.append(Outcome(
            hand_id=int(result.get("hand_id") or 0),
            delta=delta,
            showdown=isinstance(showdown, dict),
            note=note,
        ))
        if len(self.recent_outcomes) > _RECENT_OUTCOMES_KEEP:
            self.recent_outcomes = self.recent_outcomes[-_RECENT_OUTCOMES_KEEP:]
        self.tilt = max(0.0, min(1.0, self.tilt))

    # ------------------------------------------------------------------
    # Prompt rendering
    # ------------------------------------------------------------------

    def snapshot(self) -> Dict[str, Any]:
        """Return a JSON-friendly snapshot for jsonl logging / viewer display.

        Unlike `prompt_block`, this *does* expose the numeric tilt — observers
        and replay viewers benefit from seeing it accumulate. Agents reading
        their own prompt still get only the mood label.
        """
        return {
            "agent_id": self.agent_id,
            "own_seat": self.own_seat,
            "hands_played": self.hands_played,
            "tilt": round(self.tilt_factor(), 3),
            "mood": self.mood_label(),
            "recent_outcomes": [
                {
                    "hand_id": o.hand_id,
                    "delta": o.delta,
                    "showdown": o.showdown,
                    "note": o.note,
                }
                for o in self.recent_outcomes
            ],
            "rivalries": [
                {
                    "opponent": opp_id,
                    "seat": note.opponent_seat,
                    "kind": note.kind,
                    "hand_id": note.hand_id,
                    "text": note.text,
                }
                for opp_id, notes in self.rivalries.items()
                for note in notes[-_RIVALRY_KEEP_PER_OPPONENT:]
            ],
        }

    def prompt_block(self) -> Optional[Dict[str, Any]]:
        """Return a JSON-friendly dict for inclusion in the LLM user prompt.

        Returns None when there is nothing worth sending (fresh session).
        Hides the numeric tilt; only mood label + recent outcome notes flow
        through, so the model reacts via vibe rather than mechanical
        self-correction.
        """
        if not self.rivalries and not self.recent_outcomes and self.tilt < 0.05:
            return None
        rivalry_lines: List[Dict[str, Any]] = []
        for opp_id, notes in self.rivalries.items():
            for note in notes[-_RIVALRY_KEEP_PER_OPPONENT:]:
                rivalry_lines.append({
                    "opponent": opp_id,
                    "seat": note.opponent_seat,
                    "kind": note.kind,
                    "note": note.text,
                })
        return {
            "note": "過去ハンドの読み材料。逐語的にコピーせず、自分の reads と inner_voice を更新する材料に使う。",
            "mood": self.mood_label(),
            "recent_outcomes": [
                {"hand_id": o.hand_id, "note": o.note, "showdown": o.showdown}
                for o in self.recent_outcomes
            ],
            "rivalries": rivalry_lines,
        }

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _add_rivalry(self, opp_id: str, note: RivalryNote) -> None:
        bucket = self.rivalries.setdefault(opp_id, [])
        bucket.append(note)
        if len(bucket) > _RIVALRY_KEEP_PER_OPPONENT * 2:
            self.rivalries[opp_id] = bucket[-_RIVALRY_KEEP_PER_OPPONENT * 2:]

    @staticmethod
    def _last_aggressor(action_log: List[Dict[str, Any]], seat: int) -> Optional[str]:
        """Return a label like 'river の raise' for the winner's last bet/raise."""
        last: Optional[Dict[str, Any]] = None
        for entry in action_log:
            if entry.get("seat_id") != seat:
                continue
            if entry.get("action") in {"bet", "raise", "all_in"}:
                last = entry
        if last is None:
            return None
        return f"{last.get('street')} の {last.get('action')}"
