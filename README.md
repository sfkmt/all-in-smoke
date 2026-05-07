# ALL-IN SMOKE

Poker-phase incomplete information and disaster pressure now run at the same
time. While the hand is still being played, a red danger ring shrinks toward
the center of the table. Each agent decides whether to keep playing, cling to a
winning stack, linger because the chips still matter, or stand up before the
danger reaches them. If fire reaches a still-seated agent and contact continues
for several ticks, the state escalates from `engulfed` to `fatal`.

## Quick Start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run the fully local scripted demo:

```bash
python -m tools.run_all_in_smoke configs/all_in_smoke_demo.yaml --out-dir out/all_in_smoke_demo --json
```

Outputs:

- `out/all_in_smoke_demo/all_in_smoke.seed13.poker.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.seed13.live_fire.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.seed13.full_replay.jsonl`
- `out/all_in_smoke_demo/all_in_smoke.summary.json`

Open the browser replay:

```bash
python -m http.server 8765 --bind 127.0.0.1
```

Then open:

```text
http://127.0.0.1:8765/visualization/viewer.html?file=/out/all_in_smoke_demo/all_in_smoke.seed13.full_replay.jsonl
```

The viewer can also load `.jsonl` files manually with the file picker.

## OpenRouter Demo

The OpenRouter config uses `api_key_env: OPENROUTER_API_KEY`; do not put API
keys in YAML files.

```bash
export OPENROUTER_API_KEY=...
python -m tools.run_all_in_smoke configs/all_in_smoke_openrouter_grok_full.yaml --out-dir out/openrouter_demo --json
```

Then open:

```text
http://127.0.0.1:8765/visualization/viewer.html?file=/out/openrouter_demo/all_in_smoke.seed13.full_replay.jsonl
```

## MP4 Export

MP4 export requires Playwright Chromium and FFmpeg.

```bash
python -m pip install playwright
python -m playwright install chromium
# macOS: brew install ffmpeg
```

Keep the HTTP server above running, then export:

```bash
python -m tools.export_replay_video \
  out/all_in_smoke_demo/all_in_smoke.seed13.full_replay.jsonl \
  --out out/all_in_smoke_demo/all_in_smoke.seed13.full_replay.mp4 \
  --viewer-url http://127.0.0.1:8765/visualization/viewer.html \
  --seconds-per-step 0.45 \
  --width 1920 \
  --height 1080 \
  --fps 30
```

Key live fire events:

- `live_fire_start`
- `live_fire_tick`

Important live statuses:

- `clinging_to_stack`
- `tempted_by_chips`
- `stood_up`
- `engulfed`
- `fatal`

Each `live_fire_tick` also includes a per-seat `dynamic_state`. It is
accumulated from poker play before the fire tick, not authored as a direct
crisis command:

- `chip_attachment`
- `loss_chasing`
- `entitlement`
- `confidence`
- `table_image_pressure`
- `rivalry_pressure`
- `fold_success_memory`

## Layer 3

`configs/all_in_smoke_demo.yaml` supports manual capability gaps under
`crisis_profile.ability_gaps`.

It also supports TimeQL-derived gaps:

```yaml
crisis:
  timeql_profiles_path: "../personas/agentspoker_timeql_lack_contrast_profiles.json"
```

TimeQL integration is optional. For local experiments, create a TimeQL Free
plan API key and expose it via an environment variable before materializing
profiles:

```bash
export TIMEQL_API_KEY=...
```

Do not commit `TIMEQL_API_KEY`, generated persona profiles, run logs, or replay
videos. Public configs in this repository use anonymized sample agents; private
persona manifests and compiled TimeQL outputs are intentionally gitignored.

`smoke_timeql_converter.py` maps compiled TimeQL body/lack profiles into:

- `fold_ability`
- `trust_calibration`
- `help_seeking`
- `situational_awareness`
- `self_control`
- `reciprocity`
- `public_responsibility`
- `meaning_update`

Manifest values override TimeQL values, and TimeQL values override the
poker-derived fallback.

## Test

From this directory:

```bash
python -m pytest -q
```
