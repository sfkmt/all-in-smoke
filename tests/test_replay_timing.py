from poker_agents.replay_timing import (
    audio_durations_from_manifest,
    build_frame_timings,
    from_jsonable,
    step_to_start_time,
    to_jsonable,
    total_duration,
)


def test_voiced_frame_uses_audio_duration_plus_gap():
    timings = build_frame_timings(
        frame_steps=[1],
        audio_durations={1: 2.4},
        gap=0.25,
        default_frame_seconds=1.2,
    )
    assert timings[0].duration == 2.65
    assert timings[0].start_time == 0.0
    assert timings[0].has_audio is True
    assert timings[0].audio_duration == 2.4


def test_missing_audio_falls_back_to_default():
    timings = build_frame_timings(
        frame_steps=[5],
        audio_durations={},
        default_frame_seconds=1.5,
    )
    assert timings[0].duration == 1.5
    assert timings[0].has_audio is False
    assert timings[0].audio_duration == 0.0


def test_short_audio_floored_to_min():
    timings = build_frame_timings(
        frame_steps=[1],
        audio_durations={1: 0.1},
        gap=0.25,
        min_frame_seconds=0.6,
    )
    assert timings[0].duration == 0.85  # max(0.6, 0.1) + 0.25


def test_cumulative_start_times():
    timings = build_frame_timings(
        frame_steps=[1, 2, 3, 4],
        audio_durations={1: 2.0, 3: 4.0},
        gap=0.25,
        default_frame_seconds=1.0,
    )
    durations = [t.duration for t in timings]
    starts = [t.start_time for t in timings]
    assert durations == [2.25, 1.0, 4.25, 1.0]
    assert starts == [0.0, 2.25, 3.25, 7.5]
    assert total_duration(timings) == 8.5


def test_step_to_start_time_keeps_first_occurrence():
    timings = build_frame_timings(
        frame_steps=[1, 1, 2],
        audio_durations={1: 1.0, 2: 1.0},
        gap=0.0,
        default_frame_seconds=1.0,
        min_frame_seconds=0.5,
    )
    mapping = step_to_start_time(timings)
    assert mapping[1] == 0.0
    assert mapping[2] == 2.0


def test_jsonable_round_trip():
    timings = build_frame_timings(
        frame_steps=[1, 2],
        audio_durations={1: 1.5},
        gap=0.2,
        default_frame_seconds=1.0,
    )
    restored = from_jsonable(to_jsonable(timings))
    assert restored == timings


def test_audio_durations_from_manifest_skips_zero_and_bad():
    manifest = {
        "clips": [
            {"step": 1, "duration": 2.5},
            {"step": 2, "duration": 0.0},
            {"step": 3},                     # missing duration
            {"duration": 1.0},               # missing step
            {"step": "x", "duration": 1.0},  # bad step
            {"step": 4, "duration": 1.7},
        ]
    }
    out = audio_durations_from_manifest(manifest)
    assert out == {1: 2.5, 4: 1.7}


def test_negative_gap_rejected():
    try:
        build_frame_timings(frame_steps=[1], audio_durations={}, gap=-0.1)
    except ValueError:
        return
    raise AssertionError("expected ValueError for negative gap")
