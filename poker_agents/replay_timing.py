"""Per-frame timing helper for the replay video pipeline.

The exporter renders one frame per timeline entry; each entry has a `step`.
When TTS audio exists for a step we want the frame to be held for the audio
duration (plus a small inter-clip gap so words don't bleed). When no audio
exists we fall back to a default short duration.

The same timing is consumed by the mux tool to place each audio clip at the
exact start time of its frame, so audio and video stay locked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class FrameTiming:
    frame_index: int
    step: int
    duration: float          # seconds the frame is held on screen
    start_time: float        # seconds from the start of the video
    has_audio: bool
    audio_duration: float    # 0.0 when has_audio is False


def build_frame_timings(
    *,
    frame_steps: Sequence[int],
    audio_durations: Mapping[int, float],
    gap: float = 0.25,
    default_frame_seconds: float = 1.2,
    min_frame_seconds: float = 0.6,
) -> List[FrameTiming]:
    """Compute per-frame durations and cumulative start times.

    Args:
      frame_steps: step value for each frame in render order.
      audio_durations: {step: tts duration in seconds}; missing steps are
          treated as no-audio.
      gap: silence (seconds) appended after a voiced frame so the next clip
          doesn't start the instant the prior one ends.
      default_frame_seconds: hold-time for a frame with no audio.
      min_frame_seconds: floor applied to voiced frames whose audio is
          oddly short (e.g. failed-then-truncated clip).
    """
    if gap < 0 or default_frame_seconds <= 0 or min_frame_seconds <= 0:
        raise ValueError("gap >= 0 and durations must be positive")

    timings: List[FrameTiming] = []
    cursor = 0.0
    for idx, step in enumerate(frame_steps):
        audio = audio_durations.get(int(step))
        has_audio = audio is not None and audio > 0
        if has_audio:
            duration = max(min_frame_seconds, float(audio)) + gap
        else:
            duration = default_frame_seconds
        timings.append(FrameTiming(
            frame_index=idx,
            step=int(step),
            duration=round(duration, 3),
            start_time=round(cursor, 3),
            has_audio=has_audio,
            audio_duration=round(float(audio), 3) if has_audio else 0.0,
        ))
        cursor += duration
    return timings


def total_duration(timings: Iterable[FrameTiming]) -> float:
    last_end = 0.0
    for t in timings:
        last_end = t.start_time + t.duration
    return round(last_end, 3)


def step_to_start_time(timings: Iterable[FrameTiming]) -> Dict[int, float]:
    """Map each step to the moment its frame appears on the timeline.

    If a step appears more than once (shouldn't happen in normal runs), the
    earliest occurrence wins — that's where the audio belongs.
    """
    out: Dict[int, float] = {}
    for t in timings:
        out.setdefault(t.step, t.start_time)
    return out


def to_concat_lines(timings: Sequence["FrameTiming"], frame_path_for: callable) -> List[str]:
    """Build ffmpeg concat-demuxer lines, one duration per file (last included).

    `frame_path_for(timing)` returns the absolute file path string. ffmpeg 8
    honours the trailing `duration` directive and does not need the
    repeat-last-file trick that older builds required.
    """
    lines: List[str] = []
    for t in timings:
        lines.append(f"file '{frame_path_for(t)}'")
        lines.append(f"duration {t.duration:.3f}")
    return lines


def to_jsonable(timings: Iterable[FrameTiming]) -> List[Dict[str, object]]:
    return [
        {
            "frame_index": t.frame_index,
            "step": t.step,
            "duration": t.duration,
            "start_time": t.start_time,
            "has_audio": t.has_audio,
            "audio_duration": t.audio_duration,
        }
        for t in timings
    ]


def from_jsonable(records: Iterable[Mapping[str, object]]) -> List[FrameTiming]:
    out: List[FrameTiming] = []
    for r in records:
        out.append(FrameTiming(
            frame_index=int(r["frame_index"]),
            step=int(r["step"]),
            duration=float(r["duration"]),
            start_time=float(r["start_time"]),
            has_audio=bool(r["has_audio"]),
            audio_duration=float(r.get("audio_duration", 0.0) or 0.0),
        ))
    return out


def audio_durations_from_manifest(manifest: Mapping[str, object]) -> Dict[int, float]:
    """Pull {step: duration} out of a commentary_to_tts.py manifest."""
    clips = manifest.get("clips") or []
    out: Dict[int, float] = {}
    for clip in clips:
        try:
            step = int(clip["step"])  # type: ignore[index]
            duration = float(clip.get("duration", 0.0))  # type: ignore[union-attr]
        except (KeyError, TypeError, ValueError):
            continue
        if duration > 0:
            out[step] = duration
    return out
