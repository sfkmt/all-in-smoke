"""Load agent manifests describing who sits at which seat."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import yaml

from poker_agents.base import BaseAgent
from poker_agents.endpoint_agent import EndpointAgent
from poker_agents.llm_agent import (
    DEFAULT_ENDPOINT as LLM_DEFAULT_ENDPOINT,
    DEFAULT_MODEL as LLM_DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT as LLM_DEFAULT_TIMEOUT,
    LlmAgent,
)
from poker_agents.openrouter_agent import (
    DEFAULT_API_KEY_ENV as OPENROUTER_DEFAULT_API_KEY_ENV,
    DEFAULT_ENDPOINT as OPENROUTER_DEFAULT_ENDPOINT,
    DEFAULT_MODEL as OPENROUTER_DEFAULT_MODEL,
    DEFAULT_TEMPERATURE as OPENROUTER_DEFAULT_TEMPERATURE,
    DEFAULT_TIMEOUT as OPENROUTER_DEFAULT_TIMEOUT,
    OpenRouterAgent,
)
from poker_agents.scripted_agents import (
    AggressiveAgent,
    CallingAgent,
    RandomAgent,
    TightAgent,
)
from poker_agents.voice_profile import build_identity_context


SCRIPTED_CLASSES: Dict[str, type] = {
    "RandomAgent": RandomAgent,
    "CallingAgent": CallingAgent,
    "TightAgent": TightAgent,
    "AggressiveAgent": AggressiveAgent,
}


class ManifestError(ValueError):
    """Raised when a manifest document is malformed or inconsistent."""


@dataclass
class AgentSpec:
    agent_id: str
    seat_id: int
    type: str
    class_name: Optional[str] = None
    endpoint: Optional[str] = None
    timeout: Optional[float] = None
    seed: Optional[int] = None
    model: Optional[str] = None
    temperature: Optional[float] = None
    system_prompt: Optional[str] = None
    persona: Optional[str] = None
    think: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def build(self) -> BaseAgent:
        if self.type == "scripted":
            if self.class_name is None:
                raise ManifestError(f"scripted agent {self.agent_id} requires `class`")
            cls = SCRIPTED_CLASSES.get(self.class_name)
            if cls is None:
                raise ManifestError(
                    f"unknown scripted class {self.class_name!r}; "
                    f"expected one of {sorted(SCRIPTED_CLASSES)}"
                )
            kwargs: Dict[str, Any] = {}
            if cls is RandomAgent:
                kwargs["seed"] = self.seed
            agent = cls(agent_id=self.agent_id, **kwargs)
            _attach_agent_metadata(agent, self)
            return agent
        if self.type == "endpoint":
            if not self.endpoint:
                raise ManifestError(f"endpoint agent {self.agent_id} requires `endpoint`")
            agent = EndpointAgent(
                agent_id=self.agent_id,
                endpoint=self.endpoint,
                timeout=self.timeout if self.timeout is not None else 5.0,
            )
            _attach_agent_metadata(agent, self)
            return agent
        if self.type == "llm":
            kwargs: Dict[str, Any] = dict(
                agent_id=self.agent_id,
                model=self.model or LLM_DEFAULT_MODEL,
                endpoint=self.endpoint or LLM_DEFAULT_ENDPOINT,
                timeout=self.timeout if self.timeout is not None else LLM_DEFAULT_TIMEOUT,
                temperature=(
                    self.temperature if self.temperature is not None else DEFAULT_TEMPERATURE
                ),
            )
            if self.system_prompt is not None:
                kwargs["system_prompt"] = self.system_prompt
            elif self.persona is None:
                kwargs["system_prompt"] = DEFAULT_SYSTEM_PROMPT
            if self.persona is not None:
                kwargs["persona"] = self.persona
            _add_context_kwargs(kwargs, self)
            if self.think is not None:
                kwargs["think"] = self.think
            agent = LlmAgent(**kwargs)
            _attach_agent_metadata(agent, self)
            return agent
        if self.type == "openrouter":
            kwargs = dict(
                agent_id=self.agent_id,
                model=self.model or OPENROUTER_DEFAULT_MODEL,
                endpoint=self.endpoint or OPENROUTER_DEFAULT_ENDPOINT,
                api_key_env=str(self.extra.get("api_key_env", OPENROUTER_DEFAULT_API_KEY_ENV)),
                timeout=self.timeout if self.timeout is not None else OPENROUTER_DEFAULT_TIMEOUT,
                temperature=(
                    self.temperature if self.temperature is not None else OPENROUTER_DEFAULT_TEMPERATURE
                ),
                reasoning_enabled=bool(self.think) if self.think is not None else False,
                max_completion_tokens=int(self.extra.get("max_completion_tokens", 700)),
            )
            if self.system_prompt is not None:
                kwargs["system_prompt"] = self.system_prompt
            elif self.persona is None:
                kwargs["system_prompt"] = DEFAULT_SYSTEM_PROMPT
            if self.persona is not None:
                kwargs["persona"] = self.persona
            _add_context_kwargs(kwargs, self)
            agent = OpenRouterAgent(**kwargs)
            _attach_agent_metadata(agent, self)
            return agent
        raise ManifestError(f"unknown agent type {self.type!r} for {self.agent_id}")


@dataclass
class TournamentConfig:
    starting_stack: int = 1000
    num_hands: int = 20
    small_blind: int = 5
    big_blind: int = 10
    blind_increase_every: Optional[int] = None
    blind_multiplier: float = 1.0
    max_big_blind: Optional[int] = None
    freeze_blinds_when_heads_up: bool = False
    heads_up_small_blind: Optional[int] = None
    heads_up_big_blind: Optional[int] = None
    seeds: List[int] = field(default_factory=lambda: [1])
    button_seat: Optional[int] = None


@dataclass
class Manifest:
    agents: List[AgentSpec]
    tournament: TournamentConfig

    def seat_ids(self) -> List[int]:
        return [spec.seat_id for spec in self.agents]

    def build_agents(self) -> Dict[int, BaseAgent]:
        agents: Dict[int, BaseAgent] = {}
        for spec in self.agents:
            if spec.seat_id in agents:
                raise ManifestError(f"duplicate seat_id {spec.seat_id}")
            agents[spec.seat_id] = spec.build()
        return agents


def _attach_agent_metadata(agent: BaseAgent, spec: AgentSpec) -> None:
    """Attach optional manifest metadata used by logs and replay viewers."""
    metadata = dict(spec.extra or {})
    gender = metadata.get("gender")
    full_name = metadata.get("full_name")
    if gender is not None:
        setattr(agent, "gender", str(gender))
    if full_name is not None:
        setattr(agent, "full_name", str(full_name))
    setattr(agent, "manifest_metadata", metadata)


def _add_context_kwargs(kwargs: Dict[str, Any], spec: AgentSpec) -> None:
    """Forward identity and TimeQL voice metadata into LLM agents."""
    metadata = dict(spec.extra or {})
    identity_context = metadata.get("identity_context")
    if not isinstance(identity_context, Mapping):
        identity_context = build_identity_context(agent_id=spec.agent_id, metadata=metadata)
    voice_profile = metadata.get("voice_profile")
    kwargs["identity_context"] = dict(identity_context)
    if isinstance(voice_profile, Mapping):
        kwargs["voice_profile"] = dict(voice_profile)
    table_talk_allowed = metadata.get("table_talk_allowed")
    if isinstance(table_talk_allowed, bool):
        kwargs["table_talk_allowed"] = table_talk_allowed


def _parse_agent(entry: Mapping[str, Any]) -> AgentSpec:
    required = {"agent_id", "seat_id", "type"}
    missing = required - set(entry)
    if missing:
        raise ManifestError(f"agent entry missing fields: {sorted(missing)}")
    reserved = {
        "agent_id",
        "seat_id",
        "type",
        "class",
        "endpoint",
        "timeout",
        "seed",
        "model",
        "temperature",
        "system_prompt",
        "persona",
        "think",
    }
    return AgentSpec(
        agent_id=str(entry["agent_id"]),
        seat_id=int(entry["seat_id"]),
        type=str(entry["type"]),
        class_name=entry.get("class"),
        endpoint=entry.get("endpoint"),
        timeout=float(entry["timeout"]) if "timeout" in entry else None,
        seed=int(entry["seed"]) if entry.get("seed") is not None else None,
        model=entry.get("model"),
        temperature=(
            float(entry["temperature"]) if entry.get("temperature") is not None else None
        ),
        system_prompt=entry.get("system_prompt"),
        persona=entry.get("persona"),
        think=bool(entry["think"]) if "think" in entry else None,
        extra={key: value for key, value in entry.items() if key not in reserved},
    )


def _parse_tournament(entry: Optional[Mapping[str, Any]]) -> TournamentConfig:
    if entry is None:
        return TournamentConfig()
    seeds_raw = entry.get("seeds", [1])
    if isinstance(seeds_raw, int):
        seeds = [int(seeds_raw)]
    else:
        seeds = [int(value) for value in seeds_raw]
    return TournamentConfig(
        starting_stack=int(entry.get("starting_stack", 1000)),
        num_hands=int(entry.get("num_hands", 20)),
        small_blind=int(entry.get("small_blind", 5)),
        big_blind=int(entry.get("big_blind", 10)),
        blind_increase_every=(
            int(entry["blind_increase_every"])
            if entry.get("blind_increase_every") is not None
            else None
        ),
        blind_multiplier=float(entry.get("blind_multiplier", 1.0)),
        max_big_blind=(
            int(entry["max_big_blind"])
            if entry.get("max_big_blind") is not None
            else None
        ),
        freeze_blinds_when_heads_up=bool(entry.get("freeze_blinds_when_heads_up", False)),
        heads_up_small_blind=(
            int(entry["heads_up_small_blind"])
            if entry.get("heads_up_small_blind") is not None
            else None
        ),
        heads_up_big_blind=(
            int(entry["heads_up_big_blind"])
            if entry.get("heads_up_big_blind") is not None
            else None
        ),
        seeds=seeds,
        button_seat=int(entry["button_seat"]) if entry.get("button_seat") is not None else None,
    )


def parse_manifest(document: Mapping[str, Any]) -> Manifest:
    agents_raw = document.get("agents")
    if not agents_raw:
        raise ManifestError("manifest must define at least one agent under `agents`")
    agents = [_parse_agent(entry) for entry in agents_raw]
    seat_ids = [spec.seat_id for spec in agents]
    if len(set(seat_ids)) != len(seat_ids):
        raise ManifestError(f"duplicate seat_id in manifest: {seat_ids}")
    if len(agents) < 2:
        raise ManifestError("manifest requires at least two agents")
    tournament = _parse_tournament(document.get("tournament"))
    return Manifest(agents=agents, tournament=tournament)


def load_manifest(path: Path) -> Manifest:
    text = Path(path).read_text(encoding="utf-8")
    document = yaml.safe_load(text)
    if not isinstance(document, Mapping):
        raise ManifestError(f"manifest root must be a mapping, got {type(document).__name__}")
    return parse_manifest(document)
