"""Cut a vertical (9:16) highlight reel from a voiced replay video.

Takes a long landscape replay (e.g. out/replay_6p_short_voiced.mp4) plus the
commentary jsonl, picks N time-range segments, and produces a portrait mp4
suitable for Reels/Shorts/TikTok with the commentary text burned in as
large center-bottom subtitles (so it's readable on a phone even after the
horizontal crop drops the in-viewer ticker).

The crop pulls a 608×1080 vertical slice of the source landscape (centered
on the felt) and scales to 1080×1920. Audio is preserved from the source.

Usage:
    PYTHONPATH=. python tools/make_short_reel.py \
        --video out/replay_6p_short_voiced.mp4 \
        --commentary out/run_6p_long.audio.commentary.jsonl \
        --out out/replay_6p_highlights.mp4 \
        --seconds-per-step 6.0 \
        --segments 60:90:'Hand 1 開幕バスト' 162:186:'Hand 2 連続オールイン' 468:498:'Hand 13 最終決戦'
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple


def parse_commentary(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if not raw:
                continue
            try:
                ev = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if ev.get("event") == "commentary":
                out.append(ev)
    return out


def fmt_srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    ms = int(round((seconds - int(seconds)) * 1000))
    if ms == 1000:
        s += 1
        ms = 0
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def collect_segment_lines(
    commentary: List[Dict[str, Any]],
    seconds_per_step: float,
    *,
    segment_start: float,
    segment_end: float,
) -> List[Tuple[float, float, str]]:
    """Return [(local_start, local_end, text), ...] clipped to a segment window."""
    items: List[Tuple[float, float, str]] = []
    for ev in commentary:
        step = int(ev["step"])
        text = str(ev.get("text", "")).strip()
        if not text:
            continue
        text = "\n".join(line for line in text.splitlines() if line.strip())
        start = (step - 1) * seconds_per_step
        end = step * seconds_per_step
        if end <= segment_start or start >= segment_end:
            continue
        local_start = max(0.0, start - segment_start)
        local_end = min(segment_end - segment_start, end - segment_start)
        if local_end <= local_start:
            continue
        items.append((local_start, local_end, text))
    items.sort(key=lambda t: t[0])
    return items


def parse_segment(arg: str) -> Tuple[float, float, str]:
    parts = arg.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(f"bad segment {arg!r}; expected START:END[:LABEL]")
    start = float(parts[0])
    end = float(parts[1])
    label = parts[2] if len(parts) >= 3 else ""
    if end <= start:
        raise argparse.ArgumentTypeError(f"segment end must exceed start: {arg!r}")
    return start, end, label


def _wrap_japanese(text: str, font: "Any", max_px: int) -> List[str]:
    """Wrap Japanese text at character boundaries to fit within `max_px`."""
    out: List[str] = []
    for raw in text.split("\n"):
        current = ""
        for ch in raw:
            candidate = current + ch
            bbox = font.getbbox(candidate)
            if bbox[2] - bbox[0] > max_px and current:
                out.append(current)
                current = ch
            else:
                current = candidate
        if current:
            out.append(current)
    return out


def render_subtitle_png(
    text: str,
    *,
    out_path: Path,
    frame_size: Tuple[int, int],
    font_path: str,
    fontsize: int,
    margin_v: int,
    padding: int = 24,
    max_width_frac: float = 0.92,
) -> None:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGBA", frame_size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype(font_path, fontsize)
    max_text_width = int(frame_size[0] * max_width_frac) - 2 * padding
    lines = _wrap_japanese(text, font, max_text_width)
    if not lines:
        img.save(out_path)
        return
    line_gap = 10
    line_heights = [font.getbbox(ln)[3] - font.getbbox(ln)[1] for ln in lines]
    total_text_h = sum(line_heights) + line_gap * (len(lines) - 1)
    # Use widest line to size the box; centered horizontally.
    text_widths = [font.getbbox(ln)[2] - font.getbbox(ln)[0] for ln in lines]
    max_line_w = max(text_widths)
    box_w = max_line_w + 2 * padding
    box_h = total_text_h + 2 * padding
    box_x = (frame_size[0] - box_w) // 2
    box_y = frame_size[1] - box_h - margin_v
    draw.rectangle([box_x, box_y, box_x + box_w, box_y + box_h], fill=(0, 0, 0, 184))
    y = box_y + padding
    for ln, lh, lw in zip(lines, line_heights, text_widths):
        x = (frame_size[0] - lw) // 2
        # Subtle text shadow for legibility over bright video cells
        draw.text((x + 2, y + 2), ln, font=font, fill=(0, 0, 0, 160))
        draw.text((x, y), ln, font=font, fill=(255, 255, 255, 255))
        y += lh + line_gap
    img.save(out_path)


def render_segment(
    *,
    source: Path,
    start: float,
    duration: float,
    crop: str,
    lines: List[Tuple[float, float, str]],
    png_dir: Path,
    font_path: str,
    fontsize: int,
    margin_v: int,
    frame_size: Tuple[int, int],
    out_path: Path,
) -> None:
    # Generate one transparent PNG per subtitle line.
    png_paths: List[Path] = []
    for i, (_s, _e, text) in enumerate(lines):
        png_path = png_dir / f"sub_{i:03d}.png"
        render_subtitle_png(
            text, out_path=png_path, frame_size=frame_size,
            font_path=font_path, fontsize=fontsize, margin_v=margin_v,
        )
        png_paths.append(png_path)

    # Build filter_complex: crop/scale the video, then overlay each sub in turn.
    filter_parts: List[str] = [f"[0:v]{crop}[base0]"]
    last_label = "base0"
    for i, (s, e, _) in enumerate(lines):
        next_label = f"v{i}"
        filter_parts.append(
            f"[{last_label}][{i+1}:v]overlay=x=0:y=0:enable='between(t,{s:.3f},{e:.3f})'[{next_label}]"
        )
        last_label = next_label
    fc = ";".join(filter_parts)

    cmd = ["ffmpeg", "-y", "-ss", f"{start:.3f}", "-t", f"{duration:.3f}", "-i", str(source)]
    for p in png_paths:
        cmd.extend(["-i", str(p)])
    cmd.extend([
        "-filter_complex", fc,
        "-map", f"[{last_label}]",
        "-map", "0:a",
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-pix_fmt", "yuv420p",
        str(out_path),
    ])
    print(f"  [seg] {start:.0f}s+{duration:.0f}s ({len(lines)} subs) → {out_path.name}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("---- ffmpeg stderr tail ----", file=sys.stderr)
        print("\n".join(result.stderr.splitlines()[-20:]), file=sys.stderr)
        raise subprocess.CalledProcessError(result.returncode, cmd)


def concat_segments(segment_paths: List[Path], out_path: Path) -> None:
    """Concat encoded segments with the demuxer (fast, lossless, identical params)."""
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        for p in segment_paths:
            fh.write(f"file '{p.resolve()}'\n")
        listpath = Path(fh.name)
    try:
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", str(listpath),
            "-c", "copy",
            str(out_path),
        ]
        print(f"[concat] {len(segment_paths)} segments → {out_path.name}")
        subprocess.run(cmd, check=True)
    finally:
        listpath.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--video", type=Path, required=True, help="Source landscape voiced mp4")
    parser.add_argument("--commentary", type=Path, required=True, help="Commentary jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seconds-per-step", type=float, default=6.0,
                        help="Step duration the source video was rendered with")
    parser.add_argument("--segments", nargs="+", type=parse_segment, required=True,
                        help="START:END[:LABEL] in seconds, repeated")
    parser.add_argument("--crop", default="crop=608:1080:436:0,scale=1080:1920",
                        help="ffmpeg vf crop+scale chain to produce 9:16")
    parser.add_argument("--font-path", default="/System/Library/Fonts/ヒラギノ角ゴシック W7.ttc",
                        help="Path to a Japanese-capable ttf/ttc/otf for drawtext")
    parser.add_argument("--fontsize", type=int, default=44)
    parser.add_argument("--margin-v", type=int, default=160,
                        help="Pixels from bottom of frame to subtitle box")
    parser.add_argument("--frame-width", type=int, default=1080)
    parser.add_argument("--frame-height", type=int, default=1920)
    parser.add_argument("--keep-tmp", action="store_true")
    args = parser.parse_args(argv)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    commentary = parse_commentary(args.commentary)
    if not commentary:
        print("no commentary events parsed", file=sys.stderr)
        return 1

    tmp_dir = Path(tempfile.mkdtemp(prefix="reel_"))
    segment_paths: List[Path] = []
    try:
        for i, (start, end, label) in enumerate(args.segments):
            duration = end - start
            lines = collect_segment_lines(
                commentary, args.seconds_per_step,
                segment_start=start, segment_end=end,
            )
            png_dir = tmp_dir / f"seg_{i:02d}_png"
            png_dir.mkdir(exist_ok=True)
            print(f"[seg {i}] '{label}' {start:.0f}-{end:.0f}s ({len(lines)} lines)")
            seg_path = tmp_dir / f"seg_{i:02d}.mp4"
            render_segment(
                source=args.video, start=start, duration=duration,
                crop=args.crop, lines=lines,
                png_dir=png_dir, font_path=args.font_path,
                fontsize=args.fontsize, margin_v=args.margin_v,
                frame_size=(args.frame_width, args.frame_height),
                out_path=seg_path,
            )
            segment_paths.append(seg_path)
        concat_segments(segment_paths, args.out)
        print(f"[ok] wrote {args.out}")
    finally:
        if not args.keep_tmp:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)
        else:
            print(f"[tmp kept at] {tmp_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
