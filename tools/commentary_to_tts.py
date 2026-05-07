"""Synthesize per-step audio for a commentary jsonl via ElevenLabs.

Takes the commentary stream produced by tools/poker_commentator.py, runs
each line through the TTS normalizer, and POSTs to ElevenLabs's text-to-
speech endpoint. Produces one MP3 per commentary step plus a manifest
(step → file → duration in seconds) so the muxing tool can place each
clip on the video timeline.

Usage:
    export ELEVENLABS_API_KEY=sk_xxx
    PYTHONPATH=. python tools/commentary_to_tts.py \
        out/run_6p_long.seed1.commentary.jsonl \
        --voice-id tGhb4uYSV8sWBI31DYU8 \
        --out-dir out/audio_6p_long

Failures (network, 4xx, empty) skip the step rather than abort the run —
the muxer simply has silence for that frame.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from poker_agents.tts_normalizer import normalize_for_tts


DEFAULT_MODEL = "eleven_multilingual_v2"
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.4,
    "similarity_boost": 0.75,
    "style": 0.45,
    "speed": 1.15,
    "use_speaker_boost": True,
}
DEFAULT_TIMEOUT = 60.0


def synthesize(
    *,
    text: str,
    voice_id: str,
    api_key: str,
    model_id: str = DEFAULT_MODEL,
    voice_settings: Optional[Mapping[str, Any]] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Optional[bytes]:
    """POST to ElevenLabs TTS. Return MP3 bytes or None on failure."""
    body = json.dumps({
        "text": text,
        "model_id": model_id,
        "voice_settings": dict(voice_settings or DEFAULT_VOICE_SETTINGS),
    }).encode("utf-8")
    request = urllib.request.Request(
        f"https://api.elevenlabs.io/v1/text-to-speech/{voice_id}",
        data=body,
        headers={"xi-api-key": api_key, "Content-Type": "application/json", "Accept": "audio/mpeg"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")[:200]
        print(f"  HTTP {exc.code}: {body}", file=sys.stderr)
        return None
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"  network error: {exc}", file=sys.stderr)
        return None


def probe_duration(path: Path) -> float:
    """Return MP3 duration in seconds via ffprobe (0.0 on failure)."""
    try:
        out = subprocess.check_output(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            text=True, timeout=10,
        )
        return float(out.strip())
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return 0.0


def parse_commentary(path: Path) -> List[Dict[str, Any]]:
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
    return [e for e in events if e.get("event") == "commentary"]


def run(
    commentary_path: Path,
    *,
    out_dir: Path,
    voice_id: str,
    api_key: str,
    model_id: str,
    voice_settings: Mapping[str, Any],
    timeout: float,
    sleep_between: float,
    max_steps: Optional[int] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    events = parse_commentary(commentary_path)
    if max_steps is not None:
        events = events[:max_steps]
    print(f"[tts] {len(events)} commentary events → {out_dir}")

    manifest: List[Dict[str, Any]] = []
    skipped = 0
    started = time.perf_counter()
    for index, ev in enumerate(events):
        step = int(ev["step"])
        raw_text = str(ev.get("text", "")).strip()
        # Collapse blank-line padding so the LLM's whitespace habits don't
        # waste TTS budget on silence.
        raw_text = "\n".join(line for line in raw_text.splitlines() if line.strip())
        if not raw_text:
            skipped += 1
            continue
        text = normalize_for_tts(raw_text)
        mp3_path = out_dir / f"step_{step:04d}.mp3"
        audio = synthesize(
            text=text, voice_id=voice_id, api_key=api_key,
            model_id=model_id, voice_settings=voice_settings, timeout=timeout,
        )
        if audio is None:
            skipped += 1
            print(f"  step {step}: skipped (synth failed)", file=sys.stderr)
            continue
        mp3_path.write_bytes(audio)
        duration = probe_duration(mp3_path)
        manifest.append({
            "step": step,
            "hand_id": ev.get("hand_id"),
            "file": mp3_path.name,
            "duration": round(duration, 3),
            "char_count": len(text),
            "text": text,
        })
        elapsed = time.perf_counter() - started
        print(f"  [{index+1}/{len(events)}] step {step}: {duration:.2f}s, "
              f"{len(text)} chars, elapsed {elapsed:.1f}s")
        if sleep_between > 0:
            time.sleep(sleep_between)

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps({
        "voice_id": voice_id,
        "model_id": model_id,
        "voice_settings": dict(voice_settings),
        "source": str(commentary_path),
        "clips": manifest,
        "skipped": skipped,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    total_chars = sum(c["char_count"] for c in manifest)
    print(f"[tts] done — {len(manifest)} clips, {skipped} skipped, "
          f"{total_chars} chars total, manifest at {manifest_path}")
    return manifest_path


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("commentary", type=Path, help="Source commentary jsonl")
    parser.add_argument("--out-dir", type=Path, required=True, help="Where to write step_NNNN.mp3 + manifest.json")
    parser.add_argument("--voice-id", required=True)
    parser.add_argument("--model-id", default=DEFAULT_MODEL)
    parser.add_argument("--stability", type=float, default=DEFAULT_VOICE_SETTINGS["stability"])
    parser.add_argument("--similarity-boost", type=float, default=DEFAULT_VOICE_SETTINGS["similarity_boost"])
    parser.add_argument("--style", type=float, default=DEFAULT_VOICE_SETTINGS["style"])
    parser.add_argument("--speed", type=float, default=DEFAULT_VOICE_SETTINGS["speed"],
                        help="Speech speed 0.7-1.2 (default 1.15 for rushing tempo)")
    parser.add_argument("--no-speaker-boost", action="store_true")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--sleep-between", type=float, default=0.0,
                        help="Seconds to wait between API calls (rate limiting)")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args(argv)

    api_key = os.environ.get("ELEVENLABS_API_KEY")
    if not api_key:
        print("ELEVENLABS_API_KEY env var not set", file=sys.stderr)
        return 1
    voice_settings = {
        "stability": args.stability,
        "similarity_boost": args.similarity_boost,
        "style": args.style,
        "speed": args.speed,
        "use_speaker_boost": not args.no_speaker_boost,
    }
    run(
        args.commentary,
        out_dir=args.out_dir,
        voice_id=args.voice_id,
        api_key=api_key,
        model_id=args.model_id,
        voice_settings=voice_settings,
        timeout=args.timeout,
        sleep_between=args.sleep_between,
        max_steps=args.max_steps,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
