import math
from collections.abc import Callable

import pytest

from f1rl.envs.track import (
    SYNTHETIC_TRACKS,
    Segment,
    Track,
    curvature_at,
    example_track,
    max_abs_curvature_between,
    track_xy,
    width_at,
)


def test_total_length_sums_segments() -> None:
    track = Track(segments=(Segment(100.0, 0.0), Segment(50.0, 0.02)))
    assert track.total_length_m == 150.0


def test_segment_width_defaults_but_is_overridable() -> None:
    assert Segment(100.0, 0.0).width_m > 0.0
    assert Segment(100.0, 0.0, width_m=20.0).width_m == 20.0


def test_width_at_picks_the_containing_segments_width() -> None:
    track = Track(
        segments=(
            Segment(100.0, 0.0, width_m=15.0),
            Segment(50.0, 0.02, width_m=8.0),
        )
    )
    assert width_at(track, 50.0) == 15.0
    assert width_at(track, 120.0) == 8.0


def test_curvature_at_picks_the_containing_segment() -> None:
    track = Track(segments=(Segment(100.0, 0.0), Segment(50.0, 0.02)))
    assert curvature_at(track, 0.0) == 0.0
    assert curvature_at(track, 99.0) == 0.0
    assert curvature_at(track, 120.0) == 0.02


def test_curvature_at_wraps_past_the_finish_line() -> None:
    track = Track(segments=(Segment(100.0, 0.0), Segment(50.0, 0.02)))
    assert curvature_at(track, 150.0) == curvature_at(track, 0.0)
    assert curvature_at(track, 200.0) == curvature_at(track, 50.0)


def test_example_track_is_a_closed_positive_length_lap() -> None:
    track = example_track()
    assert track.total_length_m > 0
    assert len(track.segments) > 0


def test_max_abs_curvature_between_sees_a_corner_inside_the_window() -> None:
    track = Track(
        segments=(Segment(100.0, 0.0), Segment(10.0, -0.05), Segment(100.0, 0.0))
    )
    assert max_abs_curvature_between(track, 0.0, 99.0) == 0.0
    assert max_abs_curvature_between(track, 50.0, 150.0) == 0.05  # unsigned
    assert max_abs_curvature_between(track, 111.0, 150.0) == 0.0


def test_max_abs_curvature_between_wraps_past_the_finish_line() -> None:
    track = Track(segments=(Segment(10.0, 0.02), Segment(100.0, 0.0)))
    assert max_abs_curvature_between(track, 105.0, 115.0) == 0.02
    assert max_abs_curvature_between(track, 225.0, 235.0) == 0.02  # second lap


@pytest.mark.parametrize("make_track", list(SYNTHETIC_TRACKS.values()))
def test_synthetic_tracks_close_into_a_loop(make_track: Callable[[], Track]) -> None:
    """Integrating each lap's curvature must bring the centerline back to the
    start -- otherwise the geometry isn't a lap, and its map would be wrong.
    """
    track = make_track()
    x, y = track_xy(track)[-1]
    assert math.hypot(x, y) < 1e-6
    heading_turned = sum(s.length_m * s.curvature_per_m for s in track.segments)
    assert heading_turned == pytest.approx(2 * math.pi)


def test_track_xy_close_loop_shears_out_a_closing_gap() -> None:
    # Three quarters of a circle plus a straight: not a closed lap.
    track = Track(segments=(Segment(150.0, 0.01), Segment(100.0, 0.0)))
    open_end = track_xy(track)[-1]
    assert math.hypot(*open_end) > 1.0
    closed = track_xy(track, close_loop=True)
    assert math.hypot(*closed[-1]) < 1e-9
    assert closed[0] == (0.0, 0.0)
