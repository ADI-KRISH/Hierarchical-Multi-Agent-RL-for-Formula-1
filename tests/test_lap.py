import math

import pytest

from f1rl.config import EXAMPLE_CAR, GRAVITY_M_S2
from f1rl.envs.track import Segment, Track, curvature_at, example_track
from f1rl.models.car import max_corner_speed_ms
from f1rl.models.lap import (
    distance_to_braking_point_m,
    lap_time_s,
    max_safe_speed_ms,
    speed_profile_ms,
    standing_start_lap_time_s,
    standing_start_profile_ms,
)


def test_speed_profile_stays_within_car_limits() -> None:
    track = example_track()
    speeds = speed_profile_ms(EXAMPLE_CAR, track)
    assert len(speeds) > len(track.segments)
    assert all(0.0 < v <= EXAMPLE_CAR.max_speed_ms for v in speeds)


def test_speed_profile_respects_the_corner_grip_limit_throughout_the_corner() -> None:
    """Regression: a corner's grip cap must hold at every station inside it, not
    just at the segment boundary -- otherwise the car could accelerate past its
    lateral-grip limit in the middle of the corner.
    """
    track = example_track()
    speeds = speed_profile_ms(EXAMPLE_CAR, track, step_m=1.0)
    corner_cap = max_corner_speed_ms(EXAMPLE_CAR, curvature_per_m=1 / 100)
    ds = track.total_length_m / len(speeds)
    for k, v in enumerate(speeds):
        if curvature_at(track, k * ds) != 0.0:
            assert v <= corner_cap + 1e-6


def test_lap_time_is_physically_sane() -> None:
    """Roadmap phase-1 done criterion: tens of seconds to a couple minutes."""
    lap_time = lap_time_s(EXAMPLE_CAR, example_track())
    assert 5.0 < lap_time < 180.0


def test_more_iterations_do_not_increase_lap_time() -> None:
    """Extra converging passes can only tighten the speed profile, never loosen it."""
    track = example_track()
    coarse = lap_time_s(EXAMPLE_CAR, track, iterations=1)
    converged = lap_time_s(EXAMPLE_CAR, track, iterations=5)
    assert converged <= coarse


def test_standing_start_is_slower_than_a_flying_lap() -> None:
    track = example_track()
    flying = lap_time_s(EXAMPLE_CAR, track)
    standing = standing_start_lap_time_s(EXAMPLE_CAR, track)
    assert flying < standing < flying + 10.0


def test_standing_start_profile_launches_from_rest_within_limits() -> None:
    track = example_track()
    speeds = standing_start_profile_ms(EXAMPLE_CAR, track)
    flying = speed_profile_ms(EXAMPLE_CAR, track)
    assert speeds[0] == 0.0
    assert len(speeds) == len(flying) + 1
    assert all(s <= f + 1e-9 for s, f in zip(speeds[1:-1], flying[1:], strict=True))


def test_standing_start_matches_closed_form_on_a_short_straight() -> None:
    """Never reaching top speed, a standing lap of a straight is t = sqrt(2d/a)."""
    track = Track(segments=(Segment(length_m=100.0, curvature_per_m=0.0),))
    accel = EXAMPLE_CAR.max_accel_g * GRAVITY_M_S2
    expected = math.sqrt(2 * 100.0 / accel)
    assert standing_start_lap_time_s(EXAMPLE_CAR, track, step_m=0.5) == pytest.approx(
        expected, rel=1e-3
    )


def test_max_safe_speed_is_top_speed_far_from_any_corner() -> None:
    track = Track(segments=(Segment(5000.0, 0.0), Segment(100.0, 0.01)))
    assert max_safe_speed_ms(EXAMPLE_CAR, track, 100.0) == EXAMPLE_CAR.max_speed_ms


def test_max_safe_speed_is_the_corner_limit_inside_a_corner() -> None:
    track = Track(segments=(Segment(5000.0, 0.0), Segment(500.0, 1 / 50)))
    limit = max_corner_speed_ms(EXAMPLE_CAR, 1 / 50)
    assert max_safe_speed_ms(EXAMPLE_CAR, track, 5100.0) == pytest.approx(limit)


def test_max_safe_speed_follows_the_braking_curve_before_a_corner() -> None:
    """d metres before a corner, the car may be doing sqrt(v_c^2 + 2 a d)."""
    track = Track(segments=(Segment(5000.0, 0.0), Segment(500.0, 1 / 50)))
    limit = max_corner_speed_ms(EXAMPLE_CAR, 1 / 50)
    braking = EXAMPLE_CAR.max_braking_g * GRAVITY_M_S2
    for d in (5.0, 30.0, 60.0):
        expected = min(math.sqrt(limit**2 + 2 * braking * d), EXAMPLE_CAR.max_speed_ms)
        safe = max_safe_speed_ms(EXAMPLE_CAR, track, 5000.0 - d)
        assert safe == pytest.approx(expected)


def test_max_safe_speed_sees_a_corner_across_the_finish_line() -> None:
    track = Track(segments=(Segment(50.0, 1 / 30), Segment(1000.0, 0.0)))
    limit = max_corner_speed_ms(EXAMPLE_CAR, 1 / 30)
    assert max_safe_speed_ms(EXAMPLE_CAR, track, 1049.0) < EXAMPLE_CAR.max_speed_ms
    assert max_safe_speed_ms(EXAMPLE_CAR, track, 1049.99) == pytest.approx(
        limit, rel=0.01
    )


def test_distance_to_braking_point_counts_down_to_zero_at_the_braking_curve() -> None:
    track = Track(segments=(Segment(5000.0, 0.0), Segment(500.0, 1 / 50)))
    corner = max_corner_speed_ms(EXAMPLE_CAR, 1 / 50)
    speed = 80.0
    braking = (speed**2 - corner**2) / (2 * EXAMPLE_CAR.max_braking_g * GRAVITY_M_S2)
    for room in (250.0, 100.0, 0.0, -20.0):
        at = 5000.0 - braking - room
        got = distance_to_braking_point_m(EXAMPLE_CAR, track, at, speed, 300.0)
        assert got == pytest.approx(room)
    # Already slower than the corner needs: nothing to brake for, capped at horizon.
    assert distance_to_braking_point_m(EXAMPLE_CAR, track, 4990.0, 10.0, 300.0) == 300.0
