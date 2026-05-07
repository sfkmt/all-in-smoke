"""Run poker hands with ALL-IN SMOKE live fire pressure.

Example:
  python -m tools.run_all_in_smoke configs/all_in_smoke_demo.yaml --out-dir out/all_in_smoke
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from poker_agents.manifest_loader import Manifest, load_manifest
from live_fire_simulation import read_jsonl, run_live_fire_during_poker, write_jsonl
from poker_simulation import run_tournament
from smoke_timeql_converter import (
    load_timeql_ability_gap_overrides,
    load_timeql_voice_context_overrides,
)


def _load_crisis_config(path: Path) -> Dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(document, dict):
        return {}
    crisis = document.get("crisis")
    return crisis if isinstance(crisis, dict) else {}


def run_all_in_smoke(
    manifest: Manifest,
    *,
    manifest_path: Path,
    out_dir: Path,
) -> List[Dict[str, Any]]:
    """Run each tournament seed and attach live fire ticks to poker action steps."""
    crisis_config = _load_crisis_config(manifest_path)
    timeql_ability_gaps = _load_timeql_ability_gaps(
        crisis_config,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    _attach_timeql_voice_contexts(
        manifest,
        _load_timeql_voice_contexts(
            crisis_config,
            manifest=manifest,
            manifest_path=manifest_path,
        ),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    tournament = manifest.tournament
    starting_stacks = {spec.seat_id: tournament.starting_stack for spec in manifest.agents}
    summaries: List[Dict[str, Any]] = []

    for seed in tournament.seeds:
        agents = manifest.build_agents()
        poker_log = out_dir / f"all_in_smoke.seed{seed}.poker.jsonl"
        memory_log = out_dir / f"all_in_smoke.seed{seed}.memory_reasoning.jsonl"
        messages_log = out_dir / f"all_in_smoke.seed{seed}.messages.jsonl"
        live_fire_log = out_dir / f"all_in_smoke.seed{seed}.live_fire.jsonl"
        full_replay_log = out_dir / f"all_in_smoke.seed{seed}.full_replay.jsonl"
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
            log_path=poker_log,
            memory_reasoning_path=memory_log,
            messages_path=messages_log,
        )
        poker_events = read_jsonl(poker_log)
        live_fire_events = run_live_fire_during_poker(
            manifest,
            poker_events,
            fire_start_hand=crisis_config.get("fire_start_hand"),
            fire_start_when=crisis_config.get("fire_start_when"),
            fire_duration_ticks=crisis_config.get("fire_duration_ticks"),
            timeql_ability_gaps=timeql_ability_gaps,
        )
        write_jsonl(live_fire_log, live_fire_events)
        write_jsonl(full_replay_log, [*poker_events, *live_fire_events])
        summary = {
            "seed": seed,
            "poker_log": str(poker_log),
            "live_fire_log": str(live_fire_log),
            "full_replay_log": str(full_replay_log),
            "final_stacks": result.final_stacks,
            "standings": result.standings,
            "live_fire_metrics": _live_fire_metrics(live_fire_events),
        }
        summaries.append(summary)

    summary_path = out_dir / "all_in_smoke.summary.json"
    summary_path.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    return summaries


def _live_fire_metrics(events: List[Dict[str, Any]]) -> Dict[str, Any]:
    final_tick = next((event for event in reversed(events) if event.get("event") == "live_fire_tick"), None)
    crisis_match_result = next(
        (
            event.get("crisis_match")
            for event in reversed(events)
            if event.get("event") == "crisis_match_result"
        ),
        None,
    )
    ever_tempted = set()
    ever_clinging = set()
    max_goal_pressure = 0.0
    max_loss_chasing = 0.0
    for event in events:
        if event.get("event") != "live_fire_tick":
            continue
        for seat, state in (event.get("agent_states") or {}).items():
            if state.get("status") == "tempted_by_chips":
                ever_tempted.add(int(seat))
            if state.get("status") == "clinging_to_stack" or state.get("motive") == "winning_or_pot_attachment":
                ever_clinging.add(int(seat))
            max_goal_pressure = max(max_goal_pressure, float(state.get("goal_pressure", 0.0) or 0.0))
            dynamic = state.get("dynamic_state") or {}
            if isinstance(dynamic, dict):
                max_loss_chasing = max(max_loss_chasing, float(dynamic.get("loss_chasing", 0.0) or 0.0))
    if not final_tick:
        return {
            "tick_count": 0,
            "stood_up_count": 0,
            "engulfed_count": 0,
            "fatal_count": 0,
            "chip_temptation_count": 0,
            "clinging_count": 0,
            "max_goal_pressure": 0.0,
            "max_loss_chasing": 0.0,
            "crisis_match": crisis_match_result,
        }
    states = list((final_tick.get("agent_states") or {}).values())
    return {
        "tick_count": sum(1 for event in events if event.get("event") == "live_fire_tick"),
        "stood_up_count": sum(1 for state in states if state.get("status") == "stood_up"),
        "engulfed_count": sum(1 for state in states if state.get("status") == "engulfed"),
        "fatal_count": sum(1 for state in states if state.get("status") == "fatal"),
        "chip_temptation_count": len(ever_tempted),
        "clinging_count": len(ever_clinging),
        "max_goal_pressure": round(max_goal_pressure, 3),
        "max_loss_chasing": round(max_loss_chasing, 3),
        "crisis_match": crisis_match_result,
    }


def _resolve_optional_path(raw_path: Any, manifest_path: Path) -> Optional[Path]:
    if not raw_path:
        return None
    path = Path(str(raw_path))
    if path.is_absolute():
        return path
    bases = [
        manifest_path.parent,
        manifest_path.parent.parent,
        Path.cwd(),
    ]
    for base in bases:
        candidate = base / path
        if candidate.exists():
            return candidate
    return manifest_path.parent / path


def _load_timeql_ability_gaps(
    crisis_config: Dict[str, Any],
    *,
    manifest: Manifest,
    manifest_path: Path,
) -> Dict[int, Dict[str, float]]:
    path = _resolve_optional_path(crisis_config.get("timeql_profiles_path"), manifest_path)
    if path is None:
        return {}
    if not path.exists():
        print(
            "warning: TimeQL profiles not found at {path}; using manifest/fallback crisis profiles".format(
                path=path
            ),
            file=sys.stderr,
        )
        return {}
    seat_by_agent_id = _seat_by_timeql_agent_id(manifest)
    return load_timeql_ability_gap_overrides(path, seat_by_agent_id=seat_by_agent_id)


def _load_timeql_voice_contexts(
    crisis_config: Dict[str, Any],
    *,
    manifest: Manifest,
    manifest_path: Path,
) -> Dict[int, Dict[str, Any]]:
    path = _resolve_optional_path(crisis_config.get("timeql_profiles_path"), manifest_path)
    if path is None or not path.exists():
        return {}
    seat_by_agent_id = _seat_by_timeql_agent_id(manifest)
    return load_timeql_voice_context_overrides(path, seat_by_agent_id=seat_by_agent_id)


def _attach_timeql_voice_contexts(manifest: Manifest, contexts: Dict[int, Dict[str, Any]]) -> None:
    for spec in manifest.agents:
        context = contexts.get(spec.seat_id)
        if not isinstance(context, dict):
            continue
        identity = context.get("identity_context")
        if isinstance(identity, dict):
            spec.extra.setdefault("identity_context", identity)
            for key in ("full_name", "gender", "birth_date", "datetime", "location", "age", "display_name"):
                if key in identity:
                    spec.extra.setdefault(key, identity[key])
        voice_profile = context.get("voice_profile")
        if isinstance(voice_profile, dict):
            spec.extra.setdefault("voice_profile", voice_profile)


def _seat_by_timeql_agent_id(manifest: Manifest) -> Dict[int, int]:
    seat_by_agent_id = {}
    for spec in manifest.agents:
        raw_timeql_agent_id = spec.extra.get("timeql_agent_id") if isinstance(spec.extra, dict) else None
        if raw_timeql_agent_id is None:
            raw_timeql_agent_id = spec.seat_id
        seat_by_agent_id[int(raw_timeql_agent_id)] = spec.seat_id
    return seat_by_agent_id


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="agent manifest YAML")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("out/all_in_smoke"),
        help="directory for poker and live fire jsonl logs",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON summary")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    summaries = run_all_in_smoke(manifest, manifest_path=args.manifest, out_dir=args.out_dir)
    if args.json:
        print(json.dumps(summaries, ensure_ascii=False, indent=2))
    else:
        for summary in summaries:
            metrics = summary["live_fire_metrics"]
            print(
                "seed {seed}: ticks={tick_count} stood_up={stood_up_count} "
                "engulfed={engulfed_count} fatal={fatal_count} tempted={chip_temptation_count} "
                "goal_peak={max_goal_pressure} "
                "live_fire_log={live_fire_log}".format(
                    seed=summary["seed"],
                    live_fire_log=summary["live_fire_log"],
                    **metrics,
                )
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
