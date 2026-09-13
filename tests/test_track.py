from f1rl.envs.track import Segment, Track, curvature_at, example_track


def test_total_length_sums_segments() -> None:
    track = Track(segments=(Segment(100.0, 0.0), Segment(50.0, 0.02)))
    assert track.total_length_m == 150.0


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
