# AgentsPoker TimeQL Usage Policy

This file fixes the TimeQL contract before running AgentsPoker simulations.

## Current Phase

AgentsPoker currently consumes TimeQL only as per-person poker/crisis priors for
ALL-IN SMOKE behavior.

- Source: TimeQL v1 AgentsPoker bundle (`api_family: v1_agentspoker`)
- Endpoints: natal, jyotish, tekitenzui
- Output: `personas/agentspoker_timeql_compiled_profiles.json`
- Derived output: `personas/agentspoker_timeql_lack_contrast_profiles.json`
- Body compiler: `timeql_integration/agentspoker_compiler.py`
- Contrast compiler: `all-in-smoke/tools/compile_lack_contrast.py`
- Runtime consumer: `all-in-smoke/tools/run_all_in_smoke.py`
- Runtime effect: contrasted lack profiles provide crisis `ability_gaps`

Manifest-level `crisis_profile.ability_gaps` still has final priority for
explicit scenario tuning. TimeQL fills the body-derived prior layer; the
scenario manifest can override it when the story needs a specific behavior.

The compiled profile must not contain Festival-specific context or fields such
as `festival_context`, `festival_moment`, `festival_crowd_pressure`,
`festival_ritual_openness`, `festival_closing_pull`, or `festival_lack`.
AgentsPoker-specific lack output lives under `humanlm_latents.agentspoker_lack`.

The contrast file is intentionally not a TimeQL artifact. It is an
AgentsPoker-specific dramatization layer:

- raw TimeQL profiles stay preserved
- lack scores are contrasted against the current six-person table
- dominant lack categories are sharpened
- final ALL-IN SMOKE ability gaps are precomputed and embedded under
  `agentspoker_lack_contrast.ability_gaps`

## Relation

Relation profiles are not consumed by the current ALL-IN SMOKE runtime.

They should stay disabled until a runtime mapping is added. The planned use is
pair-level pressure only:

- trust or distrust in warnings
- rivalry and table-image pressure
- rescue/help likelihood
- table-talk credibility

Relation must not directly change poker actions, chip stacks, or fatal state.
It should influence only intermediate social pressure fields that are visible in
logs.

## Time

Time profiles are not consumed by the current ALL-IN SMOKE runtime.

They should stay disabled until temporal windows are mapped into explicit,
logged pressure modifiers. The planned use is scenario-time pressure only:

- reaction delay shifts
- attention/situational-awareness shifts
- urgency or panic threshold shifts
- per-window changes such as `applies_from_step` / `applies_to_step`

Time must not silently mutate base persona traits. Any time effect should appear
as a named runtime modifier in the live-fire log.

## API Family

Current AgentsPoker persona materialization is v1 AgentsPoker based.

The repository also supports TimeQL v2 body, relation, and time artifacts, but
AgentsPoker should not use v2 relation/time until the runtime mapping above is
implemented and tested.

## Public Repository Policy

TimeQL is integrated through an environment variable, not through committed
credentials. Use a TimeQL Free plan API key locally as `TIMEQL_API_KEY` when
materializing persona profiles.

Public source should contain only anonymized sample manifests and code. Do not
commit generated TimeQL persona profiles, private seed manifests, run logs,
replay JSONL files, MP4 exports, or API keys.
