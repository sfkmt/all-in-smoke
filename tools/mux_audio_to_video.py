"""Overlay per-step TTS clips onto a rendered replay video.

Takes a replay mp4 (from tools/export_replay_video.py) and a TTS manifest
(from tools/commentary_to_tts.py) and produces a new mp4 with commentary
audio mixed onto the video timeline.

Two timing modes:

  --timing <path>          (preferred) read frame start times from the
                           timing.json the exporter wrote alongside the
                           video. Each clip plays at the exact second its
                           frame appears, so audio and visual stay locked.

  --seconds-per-step S     (legacy) place clip i at i*S. Use only when the
                           video was exported with the original fixed-rate
                           pipeline.

Usage (recommended):
    PYTHONPATH=. python tools/mux_audio_to_video.py \
        --video out/replay.mp4 \
        --manifest out/audio/manifest.json \
        --timing out/replay.timing.json \
        --out out/replay_voiced.mp4
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from poker_agents.replay_timing import from_jsonable, step_to_start_time


def load_manifest(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def _delays_from_timing(
    clips: List[Dict[str, Any]], timing_path: Path,
) -> List[int]:
    """Return per-clip adelay (ms) using the exporter's frame timings.

    Clips whose step doesn't appear in the timing file (e.g. truncated
    --max-steps render) are dropped — caller filters by index.
    """
    timing_doc = json.loads(timing_path.read_text(encoding="utf-8"))
    timings = from_jsonable(timing_doc.get("frames", []))
    starts = step_to_start_time(timings)
    delays_ms: List[int] = []
    for clip in clips:
        step = int(clip["step"])
        if step not in starts:
            delays_ms.append(-1)
            continue
        delays_ms.append(int(round(starts[step] * 1000)))
    return delays_ms


def _delays_from_spf(
    clips: List[Dict[str, Any]], seconds_per_step: float,
) -> List[int]:
    """Legacy: clip i fires at (step_i - first_step) * spf."""
    first_step = int(clips[0]["step"])
    return [
        int(round((int(c["step"]) - first_step) * seconds_per_step * 1000))
        for c in clips
    ]


def build_filter_graph(
    delays_ms: List[int],
    volume: float,
) -> str:
    """Build an ffmpeg -filter_complex string that adelays each input then amixes."""
    parts: List[str] = []
    labels: List[str] = []
    # Input 0 is the video; audio clips start at input index 1.
    for i, delay_ms in enumerate(delays_ms):
        input_idx = i + 1
        # adelay takes per-channel delays; provide 2 for stereo safety.
        parts.append(
            f"[{input_idx}:a]adelay={delay_ms}|{delay_ms},volume={volume}[a{i}]"
        )
        labels.append(f"[a{i}]")
    if not parts:
        raise ValueError("no clips to mux")
    mix = "".join(labels) + f"amix=inputs={len(delays_ms)}:duration=longest:normalize=0[aout]"
    return ";".join(parts) + ";" + mix


def run_ffmpeg(
    *,
    video: Path,
    clip_paths: List[Path],
    filter_graph: str,
    out_path: Path,
) -> None:
    cmd = ["ffmpeg", "-y", "-i", str(video)]
    for clip in clip_paths:
        cmd.extend(["-i", str(clip)])
    cmd.extend([
        "-filter_complex", filter_graph,
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_path),
    ])
    print("[ffmpeg]", " ".join(cmd[:6]), "... (+", len(clip_paths), "audio inputs)")
    subprocess.run(cmd, check=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--timing", type=Path, default=None,
        help="timing.json from export_replay_video.py (preferred). "
             "When given, --seconds-per-step is ignored.",
    )
    parser.add_argument("--seconds-per-step", type=float, default=2.0,
                        help="Legacy fixed-rate timing. Ignored when --timing is given.")
    parser.add_argument("--volume", type=float, default=1.0,
                        help="Per-clip gain multiplier (1.0 = unchanged)")
    args = parser.parse_args(argv)

    manifest = load_manifest(args.manifest)
    clips = sorted(manifest.get("clips", []), key=lambda c: int(c["step"]))
    if not clips:
        print("manifest has no clips", file=sys.stderr)
        return 1
    manifest_dir = args.manifest.parent

    if args.timing is not None:
        if not args.timing.exists():
            print(f"timing file not found: {args.timing}", file=sys.stderr)
            return 1
        raw_delays = _delays_from_timing(clips, args.timing)
        kept = [(c, d) for c, d in zip(clips, raw_delays) if d >= 0]
        dropped = len(clips) - len(kept)
        if dropped:
            print(f"[mux] dropped {dropped} clip(s) whose step isn't in the timing file")
        if not kept:
            print("no clips matched the timing file", file=sys.stderr)
            return 1
        clips = [c for c, _ in kept]
        delays_ms = [d for _, d in kept]
        print(f"[mux] timing-driven: {len(clips)} clips, "
              f"first delay {delays_ms[0]}ms, last {delays_ms[-1]}ms")
    else:
        delays_ms = _delays_from_spf(clips, args.seconds_per_step)
        print(f"[mux] fixed-rate: spf={args.seconds_per_step}, {len(clips)} clips")

    clip_paths = [manifest_dir / c["file"] for c in clips]
    for p in clip_paths:
        if not p.exists():
            print(f"missing clip file: {p}", file=sys.stderr)
            return 1

    args.out.parent.mkdir(parents=True, exist_ok=True)
    filter_graph = build_filter_graph(delays_ms, args.volume)
    run_ffmpeg(video=args.video, clip_paths=clip_paths, filter_graph=filter_graph, out_path=args.out)
    print(f"[ok] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
