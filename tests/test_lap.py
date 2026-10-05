import math

import pytest

from f1rl.config import EXAMPLE_CAR, GRAVITY_M_S2
from f1rl.envs.track import Segment, Track, curvature_at, example_track
from f1rl.models.car import max_corner_speed_ms
from f1rl.models.lap import (
    lap_time_s,
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
