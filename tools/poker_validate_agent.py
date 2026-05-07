"""Exercise a bring-your-own agent against the contract and report pass/fail.

Run: `python -m tools.poker_validate_agent path/to/manifest.yaml --agent-id my-agent`.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from poker_agents.base import AgentDecision, BaseAgent, Observation
from poker_agents.manifest_loader import AgentSpec, load_manifest
from poker_engine import PokerAction, apply_action, start_hand
from poker_simulation import build_observation


@dataclass
class ScenarioReport:
    name: str
    passed: bool
    latency_ms: float
    decision: Optional[Dict[str, Any]] = None
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "passed": self.passed,
            "latency_ms": round(self.latency_ms, 2),
            "decision": self.decision,
            "errors": list(self.errors),
        }


@dataclass
class ValidationReport:
    agent_id: str
    passed: bool
    scenarios: List[ScenarioReport] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "passed": self.passed,
            "scenarios": [scenario.to_dict() for scenario in self.scenarios],
        }


def _preflop_facing_bb() -> Observation:
    state = start_hand(
        hand_id=1,
        stacks={0: 1000, 1: 1000, 2: 1000},
        button_seat=0,
        small_blind=10,
        big_blind=20,
        seed=1,
    )
    return build_observation(state, state.action_seat)


def _flop_no_bet() -> Observation:
    state = start_hand(
        hand_id=2,
        stacks={0: 1000, 1: 1000, 2: 1000},
        button_seat=0,
        small_blind=10,
        big_blind=20,
        seed=2,
    )
    apply_action(state, 0, PokerAction("call"))
    apply_action(state, 1, PokerAction("call"))
    apply_action(state, 2, PokerAction("check"))
    from poker_engine import advance_street

    advance_street(state)
    return build_observation(state, state.action_seat)


def _turn_facing_bet() -> Observation:
    state = start_hand(
        hand_id=3,
        stacks={0: 1000, 1: 1000},
        button_seat=0,
        small_blind=10,
        big_blind=20,
        seed=3,
    )
    apply_action(state, 0, PokerAction("call"))
    apply_action(state, 1, PokerAction("check"))
    from poker_engine import advance_street

    advance_street(state)
    apply_action(state, 1, PokerAction("check"))
    apply_action(state, 0, PokerAction("check"))
    advance_street(state)
    apply_action(state, 1, PokerAction("bet", amount=60))
    return build_observation(state, state.action_seat)


def _short_stack_allin_decision() -> Observation:
    state = start_hand(
        hand_id=4,
        stacks={0: 40, 1: 1000},
        button_seat=0,
        small_blind=10,
        big_blind=20,
        seed=4,
    )
    return build_observation(state, state.action_seat)


def default_scenarios() -> List[tuple[str, Observation]]:
    return [
        ("preflop_facing_bb", _preflop_facing_bb()),
        ("flop_no_bet", _flop_no_bet()),
        ("turn_facing_bet", _turn_facing_bet()),
        ("short_stack_decision", _short_stack_allin_decision()),
    ]


def _validate_decision(observation: Observation, decision: AgentDecision) -> List[str]:
    errors: List[str] = []
    legal_names = observation.legal_action_names()
    if decision.action not in legal_names:
        errors.append(f"action {decision.action!r} not in legal {legal_names}")
        return errors
    entry = observation.legal_action(decision.action)
    if entry is None:
        errors.append(f"no legal entry for {decision.action!r}")
        return errors
    if entry["min_amount"] is not None:
        if decision.amount is None:
            errors.append(f"{decision.action} requires amount, got None")
            return errors
        if decision.amount < entry["min_amount"]:
            errors.append(f"amount {decision.amount} below min {entry['min_amount']}")
        if entry["max_amount"] is not None and decision.amount > entry["max_amount"]:
            errors.append(f"amount {decision.amount} above max {entry['max_amount']}")
    return errors


def validate_agent(
    agent: BaseAgent,
    *,
    scenarios: Optional[List[tuple[str, Observation]]] = None,
    max_latency_ms: float = 5000.0,
) -> ValidationReport:
    scenarios = scenarios if scenarios is not None else default_scenarios()
    results: List[ScenarioReport] = []
    for name, observation in scenarios:
        errors: List[str] = []
        started = time.perf_counter()
        decision_payload: Optional[Dict[str, Any]] = None
        try:
            decision = agent.decide_action(observation)
        except Exception as exc:  # noqa: BLE001
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            results.append(
                ScenarioReport(
                    name=name,
                    passed=False,
                    latency_ms=elapsed_ms,
                    errors=[f"agent raised {type(exc).__name__}: {exc}"],
                )
            )
            continue
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        decision_payload = decision.to_dict()
        errors = _validate_decision(observation, decision)
        if elapsed_ms > max_latency_ms:
            errors.append(f"latency {elapsed_ms:.1f}ms exceeds {max_latency_ms:.0f}ms")
        results.append(
            ScenarioReport(
                name=name,
                passed=not errors,
                latency_ms=elapsed_ms,
                decision=decision_payload,
                errors=errors,
            )
        )
    return ValidationReport(
        agent_id=agent.agent_id,
        passed=all(scenario.passed for scenario in results),
        scenarios=results,
    )


def _select_spec(manifest_path: Path, agent_id: Optional[str]) -> AgentSpec:
    manifest = load_manifest(manifest_path)
    if agent_id is None:
        if len(manifest.agents) != 1:
            raise SystemExit(
                "manifest contains multiple agents; pass --agent-id to pick one"
            )
        return manifest.agents[0]
    for spec in manifest.agents:
        if spec.agent_id == agent_id:
            return spec
    raise SystemExit(f"agent_id {agent_id!r} not found in manifest")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifest", type=Path, help="path to agent manifest YAML")
    parser.add_argument("--agent-id", help="specific agent to validate")
    parser.add_argument(
        "--max-latency-ms",
        type=float,
        default=5000.0,
        help="per-scenario latency budget (default: 5000ms)",
    )
    parser.add_argument(
        "--json", action="store_true", help="emit report as JSON instead of text"
    )
    args = parser.parse_args(argv)

    spec = _select_spec(args.manifest, args.agent_id)
    agent = spec.build()
    report = validate_agent(agent, max_latency_ms=args.max_latency_ms)

    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        status = "PASS" if report.passed else "FAIL"
        print(f"agent {report.agent_id}: {status}")
        for scenario in report.scenarios:
            mark = "ok" if scenario.passed else "FAIL"
            print(f"  [{mark}] {scenario.name} ({scenario.latency_ms:.1f}ms)")
            for error in scenario.errors:
                print(f"        - {error}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
