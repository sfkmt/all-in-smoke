"""Run a competition from a manifest across multiple seeds and emit a leaderboard.

Run: `python -m tools.poker_run_tournament manifest.yaml --logs out/run.jsonl`.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from poker_agents.manifest_loader import Manifest, load_manifest
from poker_simulation import run_tournament


@dataclass
class LeaderboardRow:
    seat_id: int
    agent_id: str
    total_profit: int
    per_seed_stacks: Dict[int, int] = field(default_factory=dict)
    wins: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seat_id": self.seat_id,
            "agent_id": self.agent_id,
            "total_profit": self.total_profit,
            "wins": self.wins,
            "per_seed_stacks": dict(self.per_seed_stacks),
        }


def _per_seed_path(base: Optional[Path], seed: int) -> Optional[Path]:
    if base is None:
        return None
    return base.with_name(f"{base.stem}.seed{seed}{base.suffix}")


def run_competition(
    manifest: Manifest,
    *,
    log_path: Optional[Path] = None,
    memory_reasoning_path: Optional[Path] = None,
    messages_path: Optional[Path] = None,
) -> List[LeaderboardRow]:
    tournament = manifest.tournament
    starting_stacks = {spec.seat_id: tournament.starting_stack for spec in manifest.agents}
    profit: Dict[int, int] = defaultdict(int)
    wins: Dict[int, int] = defaultdict(int)
    per_seed_stacks: Dict[int, Dict[int, int]] = defaultdict(dict)

    for seed in tournament.seeds:
        agents = manifest.build_agents()
        seed_log = _per_seed_path(log_path, seed)
        seed_mr = _per_seed_path(memory_reasoning_path, seed)
        seed_msg = _per_seed_path(messages_path, seed)
        result = run_tournament(
            agents,
            starting_stacks=starting_stacks,
            num_hands=tournament.num_hands,
            small_blind=tournament.small_blind,
            big_blind=tournament.big_blind,
            blind_increase_every=tournament.blind_increase_every,
            blind_multiplier=tournament.blind_multiplier,
            max_big_blind=tournament.max_big_blind,
            freeze_blinds_when_heads_up=tournament.freeze_blinds_when_heads_up,
            heads_up_small_blind=tournament.heads_up_small_blind,
            heads_up_big_blind=tournament.heads_up_big_blind,
            button_seat=tournament.button_seat,
            seed=seed,
            log_path=seed_log,
            memory_reasoning_path=seed_mr,
            messages_path=seed_msg,
        )
        for seat_id, final_stack in result.final_stacks.items():
            profit[seat_id] += final_stack - tournament.starting_stack
            per_seed_stacks[seat_id][seed] = final_stack
        winner_seat = max(result.final_stacks, key=lambda seat: result.final_stacks[seat])
        wins[winner_seat] += 1

    rows = [
        LeaderboardRow(
            seat_id=spec.seat_id,
            agent_id=spec.agent_id,
            total_profit=profit[spec.seat_id],
            wins=wins[spec.seat_id],
            per_seed_stacks=per_seed_stacks[spec.seat_id],
        )
        for spec in manifest.agents
    ]
    rows.sort(key=lambda row: (-row.total_profit, -row.wins, row.seat_id))
    return rows


def _format_table(rows: List[LeaderboardRow]) -> str:
    lines = ["rank  agent_id            seat  profit   wins"]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"{rank:>4}  {row.agent_id:<18}  {row.seat_id:>4}  {row.total_profit:>+6}  {row.wins:>4}"
        )
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="agent manifest YAML")
    parser.add_argument(
        "--logs",
        type=Path,
        default=None,
        help="base path for per-seed jsonl logs (e.g. out/run.jsonl)",
    )
    parser.add_argument(
        "--memory-log",
        type=Path,
        default=None,
        help="base path for per-seed memory_reasoning jsonl (inner voice + psych)",
    )
    parser.add_argument(
        "--messages-log",
        type=Path,
        default=None,
        help="base path for per-seed table_talk jsonl (public messages)",
    )
    parser.add_argument("--json", action="store_true", help="emit leaderboard as JSON")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    for path in (args.logs, args.memory_log, args.messages_log):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
    rows = run_competition(
        manifest,
        log_path=args.logs,
        memory_reasoning_path=args.memory_log,
        messages_path=args.messages_log,
    )

    if args.json:
        print(json.dumps([row.to_dict() for row in rows], indent=2))
    else:
        print(_format_table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
