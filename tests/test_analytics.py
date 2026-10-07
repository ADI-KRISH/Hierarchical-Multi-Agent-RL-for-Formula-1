import itertools

import pytest

from f1rl.analytics import (
    lap_sections,
    profile_times_s,
    section_timings,
    summarize_laps,
)
from f1rl.envs.track import Segment, Track, technical_track


def test_lap_sections_alternate_and_cover_the_whole_lap() -> None:
    track = technical_track()
    sections = lap_sections(track, corner_curvature_per_m=0.002, min_corner_m=10.0)

    assert [s.name for s in sections if s.is_corner] == ["T1", "T2", "T3", "T4", "T5"]
    assert sections[0].start_m == 0.0
    assert sections[-1].end_m == pytest.approx(track.total_length_m)
    for before, after in itertools.pairwise(sections):
        assert before.end_m == pytest.approx(after.start_m)
        assert before.is_corner != after.is_corner
    assert sections[1].min_radius_m == pytest.approx(15.0)  # the hairpin


def test_lap_sections_treat_short_kinks_as_straight() -> None:
    track = Track(
        segments=(
            Segment(100.0, 0.0),
            Segment(3.0, 0.05),  # a noise kink, not a corner
            Segment(100.0, 0.0),
            Segment(50.0, 0.02),
        )
    )
    sections = lap_sections(track, corner_curvature_per_m=0.002, min_corner_m=10.0)
    assert [s.name for s in sections] == ["S1", "T1"]
    assert sections[0].end_m == pytest.approx(203.0)


def test_profile_times_at_constant_speed() -> None:
    assert profile_times_s([10.0, 10.0, 10.0], ds_m=5.0) == [0.0, 0.5, 1.0]


def test_section_deltas_add_up_to_the_lap_time_gap() -> None:
    track = Track(segments=(Segment(100.0, 0.0), Segment(50.0, 0.02)))
    sections = lap_sections(track, 0.002, 10.0)
    distance = [0.0, 50.0, 100.0, 150.0]
    ref_time, run_time = [0.0, 1.0, 2.0, 3.0], [0.0, 1.2, 2.5, 3.9]
    speed = [50.0, 50.0, 50.0, 50.0]

    timings = section_timings(
        sections, distance, run_time, speed, distance, ref_time, speed
    )

    assert sum(t.delta_s for t in timings) == pytest.approx(3.9 - 3.0)
    assert timings[1].delta_s == pytest.approx((3.9 - 2.5) - (3.0 - 2.0))


def test_summarize_laps_ignores_unfinished_laps() -> None:
    summary = summarize_laps([30.0, None, 32.0], off_track=1)
    assert summary.completed == 2
    assert summary.completion_rate == pytest.approx(2 / 3)
    assert summary.mean_s == pytest.approx(31.0)
    assert summary.best_s == 30.0
    assert summary.worst_s == 32.0


def test_summarize_laps_with_no_finishes() -> None:
    summary = summarize_laps([None, None], off_track=2)
    assert summary.mean_s is None
    assert summary.completion_rate == 0.0
