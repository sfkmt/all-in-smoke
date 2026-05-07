"""Export the poker replay viewer to an mp4 via headless Chromium + ffmpeg.

Requires the viewer to be reachable over HTTP (run `python -m http.server`
from the repo root) and `run.seed*.jsonl` sitting somewhere Playwright can
read. Emits one PNG per step into a temp dir then stitches with ffmpeg.

Usage (fixed-rate, original):
    PYTHONPATH=. python tools/export_replay_video.py out/run.seed1.jsonl \
        --out out/replay.mp4 --seconds-per-step 0.8

Usage (TTS-driven variable durations — each frame held for its commentary
length, with a small inter-clip gap; emits a sibling timing.json the muxer
reads to place audio at the matching offset):
    PYTHONPATH=. python tools/export_replay_video.py out/run.seed1.jsonl \
        --out out/replay.mp4 \
        --extra out/run.seed1.commentary.jsonl \
        --audio-manifest out/audio/manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Sequence

from poker_agents.replay_timing import (
    FrameTiming,
    audio_durations_from_manifest,
    build_frame_timings,
    to_concat_lines,
    to_jsonable,
    total_duration,
)


def _write_concat_list(
    list_path: Path,
    frames_dir: Path,
    timings: Sequence[FrameTiming],
) -> None:
    """Write an ffmpeg concat-demuxer file with per-frame durations."""
    if not timings:
        raise ValueError("no timings to write")
    lines = to_concat_lines(
        timings,
        lambda t: (frames_dir / f"frame_{t.frame_index:04d}.png").as_posix(),
    )
    list_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def export(
    run_path: Path,
    out_path: Path,
    viewer_url: str,
    seconds_per_step: float,
    width: int,
    height: int,
    fps: int,
    settle_ms: int,
    headful: bool,
    keep_frames: bool,
    extra_paths: list,
    max_steps: Optional[int] = None,
    audio_manifest_path: Optional[Path] = None,
    timing_out_path: Optional[Path] = None,
    frame_gap: float = 0.25,
    default_frame_seconds: float = 1.2,
) -> None:
    from playwright.sync_api import sync_playwright

    if not run_path.exists():
        raise SystemExit(f"run log not found: {run_path}")
    file_args = [str(run_path)] + [str(p) for p in extra_paths if p.exists()]

    audio_durations = {}
    if audio_manifest_path is not None:
        if not audio_manifest_path.exists():
            raise SystemExit(f"audio manifest not found: {audio_manifest_path}")
        manifest = json.loads(audio_manifest_path.read_text(encoding="utf-8"))
        audio_durations = audio_durations_from_manifest(manifest)
        print(f"[timing] {len(audio_durations)} steps have audio durations from {audio_manifest_path}")

    frames_dir = Path(tempfile.mkdtemp(prefix="pokerframes_"))
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=not headful)
            ctx = browser.new_context(
                viewport={"width": width, "height": height},
                device_scale_factor=1,
            )
            page = ctx.new_page()
            page.goto(viewer_url, wait_until="load")
            page.wait_for_selector("#file-picker")
            page.set_input_files("#file-picker", file_args)
            page.wait_for_function(
                "window.state && window.state.currentSeed !== null",
                timeout=15000,
            )
            total = int(
                page.evaluate("state.seeds.get(state.currentSeed).timeline.length")
            )
            if max_steps is not None and max_steps > 0:
                total = min(total, max_steps)
            frame_steps: List[int] = page.evaluate(
                "state.seeds.get(state.currentSeed).timeline"
                ".slice(0, %d).map(f => f.step)" % total
            )
            print(f"[export] steps={total}  viewport={width}x{height}  out={out_path}")
            print(f"[export] inputs={file_args}")

            for i in range(total):
                page.evaluate(f"renderFrame({i})")
                # Let the bubble-pop animation settle and layout stabilise.
                page.wait_for_timeout(settle_ms)
                page.screenshot(
                    path=str(frames_dir / f"frame_{i:04d}.png"),
                    full_page=False,
                    animations="disabled",
                )
                if (i + 1) % 10 == 0 or i + 1 == total:
                    print(f"  rendered {i+1}/{total}")
            browser.close()

        if audio_manifest_path is not None:
            timings = build_frame_timings(
                frame_steps=frame_steps,
                audio_durations=audio_durations,
                gap=frame_gap,
                default_frame_seconds=default_frame_seconds,
            )
            voiced = sum(1 for t in timings if t.has_audio)
            print(
                f"[timing] {voiced}/{len(timings)} frames voiced, "
                f"total video = {total_duration(timings):.2f}s"
            )
            list_path = frames_dir / "concat.txt"
            _write_concat_list(list_path, frames_dir, timings)
            cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(list_path),
                "-fps_mode", "cfr",
                "-r", str(fps),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "20",
                str(out_path),
            ]
            print("[ffmpeg]", " ".join(cmd))
            subprocess.run(cmd, check=True)
            timing_path = timing_out_path or out_path.with_suffix(".timing.json")
            timing_path.write_text(json.dumps({
                "video": str(out_path),
                "audio_manifest": str(audio_manifest_path),
                "frame_gap": frame_gap,
                "default_frame_seconds": default_frame_seconds,
                "total_seconds": total_duration(timings),
                "frames": to_jsonable(timings),
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[ok] wrote {out_path}")
            print(f"[ok] wrote timing → {timing_path}")
        else:
            input_fps = 1.0 / seconds_per_step
            # pad makes width/height even for yuv420p; -vsync vfr avoids duplicate frame spam
            cmd = [
                "ffmpeg", "-y",
                "-framerate", f"{input_fps:.6f}",
                "-i", str(frames_dir / "frame_%04d.png"),
                "-r", str(fps),
                "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-preset", "medium",
                "-crf", "20",
                str(out_path),
            ]
            print("[ffmpeg]", " ".join(cmd))
            subprocess.run(cmd, check=True)
            print(f"[ok] wrote {out_path}")
    finally:
        if keep_frames:
            print(f"[frames kept at] {frames_dir}")
        else:
            shutil.rmtree(frames_dir, ignore_errors=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_jsonl", type=Path, help="path to run.seed*.jsonl")
    parser.add_argument("-o", "--out", type=Path, default=Path("replay.mp4"))
    parser.add_argument(
        "--viewer-url",
        default="http://127.0.0.1:8765/visualization/viewer.html",
        help="URL of the running viewer (python -m http.server ...)",
    )
    parser.add_argument("--seconds-per-step", type=float, default=0.8,
                        help="Used only when --audio-manifest is not given")
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--settle-ms",
        type=int,
        default=180,
        help="delay per step after calling renderFrame before screenshotting",
    )
    parser.add_argument(
        "--headful", action="store_true", help="show the browser window while rendering"
    )
    parser.add_argument(
        "--keep-frames",
        action="store_true",
        help="preserve the temp PNG frames for debugging",
    )
    parser.add_argument(
        "--extra",
        type=Path,
        action="append",
        default=[],
        help="additional jsonl to load alongside the run (e.g. commentary). Repeatable.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="cap the number of rendered timeline frames (smoke / preview)",
    )
    parser.add_argument(
        "--audio-manifest",
        type=Path,
        default=None,
        help=("TTS manifest.json from commentary_to_tts.py. When supplied, each "
              "frame is held for the matching audio length (+gap) instead of a "
              "fixed --seconds-per-step. Also writes a sibling timing.json the "
              "muxer reads."),
    )
    parser.add_argument(
        "--timing-out",
        type=Path,
        default=None,
        help="Where to write the timing.json (defaults to <out>.timing.json)",
    )
    parser.add_argument(
        "--frame-gap",
        type=float,
        default=0.25,
        help="Silence appended after each voiced frame (seconds, audio-manifest mode only)",
    )
    parser.add_argument(
        "--default-frame-seconds",
        type=float,
        default=1.2,
        help="Hold time for frames with no audio (audio-manifest mode only)",
    )
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    export(
        args.run_jsonl.resolve(),
        args.out.resolve(),
        args.viewer_url,
        args.seconds_per_step,
        args.width,
        args.height,
        args.fps,
        args.settle_ms,
        args.headful,
        args.keep_frames,
        [p.resolve() for p in args.extra],
        args.max_steps,
        audio_manifest_path=args.audio_manifest.resolve() if args.audio_manifest else None,
        timing_out_path=args.timing_out.resolve() if args.timing_out else None,
        frame_gap=args.frame_gap,
        default_frame_seconds=args.default_frame_seconds,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
