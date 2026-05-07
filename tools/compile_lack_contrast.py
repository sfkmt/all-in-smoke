"""Compile table-relative lack contrast profiles for AgentsPoker."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from smoke_timeql_converter import compile_lack_contrast_profiles  # noqa: E402


def _load_profiles(path: Path) -> List[Mapping[str, Any]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(raw, list):
        return [profile for profile in raw if isinstance(profile, Mapping)]
    if isinstance(raw, Mapping):
        return [profile for profile in raw.values() if isinstance(profile, Mapping)]
    raise ValueError("input profile file must contain a list or mapping")


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary_row(profile: Mapping[str, Any]) -> Dict[str, Any]:
    identity = profile.get("identity_seed") if isinstance(profile.get("identity_seed"), Mapping) else {}
    latents = profile.get("humanlm_latents") if isinstance(profile.get("humanlm_latents"), Mapping) else {}
    lack_profile = latents.get("agentspoker_lack") if isinstance(latents.get("agentspoker_lack"), Mapping) else {}
    contrast = profile.get("agentspoker_lack_contrast")
    if not isinstance(contrast, Mapping):
        contrast = {}
    return {
        "agent_id": identity.get("agent_id"),
        "name": identity.get("name"),
        "primary_lack": lack_profile.get("primary_lack"),
        "dominant_crisis_gaps": contrast.get("dominant_crisis_gaps", []),
    }


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("profiles", type=Path, help="raw TimeQL compiled profile JSON")
    parser.add_argument("--out", type=Path, required=True, help="derived contrast profile JSON")
    parser.add_argument("--lack-strength", type=float, default=3.0)
    parser.add_argument("--ability-strength", type=float, default=3.0)
    parser.add_argument("--json", action="store_true", help="print machine-readable summary")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    profiles = _load_profiles(args.profiles)
    contrasted = compile_lack_contrast_profiles(
        profiles,
        lack_contrast_strength=args.lack_strength,
        ability_contrast_strength=args.ability_strength,
    )
    _write_json(args.out, contrasted)
    summary = {
        "input": str(args.profiles),
        "out": str(args.out),
        "profiles": len(contrasted),
        "lack_strength": args.lack_strength,
        "ability_strength": args.ability_strength,
        "rows": [_summary_row(profile) for profile in contrasted],
    }
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        for row in summary["rows"]:
            gaps = ", ".join(
                f"{item['gap']}={item['score']}"
                for item in row.get("dominant_crisis_gaps", [])
                if isinstance(item, Mapping)
            )
            print(f"{row['name']}: primary_lack={row['primary_lack']} {gaps}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
