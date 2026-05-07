"""Generate Pokemon-battle-style commentary for a completed run jsonl.

Usage:
  PYTHONPATH=. python tools/poker_commentator.py run.seed1.jsonl \
      --out run.seed1.commentary.jsonl

Produces one `commentary` event per `action` event in the source, stamped
with the same `step` so the viewer can join them. The commentator is
omniscient (god-view): it sees every player's hole cards, inner_voices, and
the current made-hand category for each contender.

Steps that fail to elicit a response (network down, model timeout, empty
output) are skipped — the viewer simply has no commentary at that step.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Tuple

from poker_agents.commentator import (
    COMMENTATOR_AUDIO_PROMPT,
    COMMENTATOR_SYSTEM_PROMPT,
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    DEFAULT_TIMEOUT,
    build_payload_for_step,
    call_commentator,
)


STREET_TO_BOARD_LEN = {"preflop": 0, "flop": 3, "turn": 4, "river": 5, "showdown": 5}


def parse_jsonl(path: Path) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                events.append(json.loads(raw))
            except json.JSONDecodeError:
                continue
    return events


def index_events(events: List[Mapping[str, Any]]) -> Tuple[
    Dict[int, Dict[str, Any]],          # hand_id -> hand_start
    Dict[int, List[str]],               # hand_id -> final_board
    List[Mapping[str, Any]],            # action events in step order
    Dict[int, Mapping[str, Any]],       # step -> memory_reasoning
]:
    hand_starts: Dict[int, Dict[str, Any]] = {}
    final_boards: Dict[int, List[str]] = {}
    actions: List[Mapping[str, Any]] = []
    reasonings: Dict[int, Mapping[str, Any]] = {}
    for ev in events:
        et = ev.get("event")
        if et == "hand_start":
            hand_starts[int(ev["hand_id"])] = dict(ev)
        elif et == "hand_result":
            final_boards[int(ev["hand_id"])] = list(ev.get("board") or [])
        elif et == "action":
            actions.append(ev)
        elif et == "memory_reasoning":
            reasonings[int(ev["step"])] = ev
    actions.sort(key=lambda e: int(e.get("step", 0)))
    return hand_starts, final_boards, actions, reasonings


def board_for_street(street: str, final_board: List[str]) -> List[str]:
    n = STREET_TO_BOARD_LEN.get(street, 0)
    return list(final_board[:n])


def seat_to_agent_id_from_actions(actions: List[Mapping[str, Any]]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for a in actions:
        seat = a.get("seat_id")
        agent = a.get("agent_id")
        if seat is None or agent is None:
            continue
        seat_int = int(seat)
        mapping.setdefault(seat_int, str(agent))
    return mapping


def generate_commentary(
    events: List[Mapping[str, Any]],
    *,
    model: str,
    endpoint: str,
    timeout: float,
    temperature: float,
    max_steps: Optional[int] = None,
    progress: bool = False,
    system_prompt: Optional[str] = None,
) -> List[Dict[str, Any]]:
    hand_starts, final_boards, actions, reasonings = index_events(events)
    seat_to_agent_id = seat_to_agent_id_from_actions(actions)

    output: List[Dict[str, Any]] = []
    prior_actions_by_hand: Dict[int, List[Mapping[str, Any]]] = {}
    skipped = 0

    for index, action in enumerate(actions):
        if max_steps is not None and index >= max_steps:
            break
        hand_id = int(action["hand_id"])
        hand_start = hand_starts.get(hand_id)
        final_board = final_boards.get(hand_id, [])
        if hand_start is None:
            continue
        prior = prior_actions_by_hand.setdefault(hand_id, [])
        board_now = board_for_street(action.get("street", "preflop"), final_board)
        prior_reasonings_upto = {
            step: r for step, r in reasonings.items() if step <= int(action["step"])
        }
        payload = build_payload_for_step(
            action_event=action,
            hand_start=hand_start,
            prior_actions=list(prior),
            prior_reasonings=prior_reasonings_upto,
            seat_to_agent_id=seat_to_agent_id,
            pot_after=int(action.get("pot_after", 0)),
            board_now=board_now,
        )
        prior.append(action)

        kwargs = dict(
            model=model,
            endpoint=endpoint,
            timeout=timeout,
            temperature=temperature,
        )
        if system_prompt is not None:
            kwargs["system_prompt"] = system_prompt
        text = call_commentator(payload, **kwargs)
        if text is None:
            skipped += 1
            if progress:
                print(f"  step {action['step']}: skipped", file=sys.stderr)
            continue
        output.append({
            "event": "commentary",
            "step": int(action["step"]),
            "hand_id": hand_id,
            "text": text,
        })
        if progress:
            preview = text.replace("\n", " ⏎ ")[:80]
            print(f"  step {action['step']}: {preview}", file=sys.stderr)

    if progress:
        total = len(output) + skipped
        print(f"done — {len(output)}/{total} commentary lines (skipped {skipped})", file=sys.stderr)
    return output


def write_jsonl(path: Path, records: List[Mapping[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source run jsonl (e.g. run.seed1.jsonl)")
    parser.add_argument(
        "--out", type=Path, default=None,
        help="Output commentary jsonl. Defaults to <input-stem>.commentary.jsonl",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-steps", type=int, default=None,
                        help="Limit number of action steps to commentate (smoke test)")
    parser.add_argument("--audio-mode", action="store_true",
                        help="Use 1-sentence/40-char prompt optimized for TTS playback")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-step progress")
    args = parser.parse_args(argv)

    out_path = args.out or args.input.with_name(args.input.stem + ".commentary.jsonl")
    events = parse_jsonl(args.input)
    if not events:
        print(f"no events parsed from {args.input}", file=sys.stderr)
        return 1
    commentary = generate_commentary(
        events,
        model=args.model,
        endpoint=args.endpoint,
        timeout=args.timeout,
        temperature=args.temperature,
        max_steps=args.max_steps,
        progress=not args.quiet,
        system_prompt=COMMENTATOR_AUDIO_PROMPT if args.audio_mode else None,
    )
    write_jsonl(out_path, commentary)
    print(f"wrote {len(commentary)} commentary events to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
