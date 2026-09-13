import math
from dataclasses import replace

import pytest

from f1rl.config import EXAMPLE_CAR
from f1rl.models.car import (
    accel_limited_speed_ms,
    brake_limited_speed_ms,
    max_corner_speed_ms,
)


def test_straight_has_no_corner_speed_limit() -> None:
    assert max_corner_speed_ms(EXAMPLE_CAR, 0.0) == EXAMPLE_CAR.max_speed_ms


def test_tighter_corners_are_slower() -> None:
    tight = max_corner_speed_ms(EXAMPLE_CAR, curvature_per_m=1 / 50)
    wide = max_corner_speed_ms(EXAMPLE_CAR, curvature_per_m=1 / 200)
    assert tight < wide


def test_corner_speed_is_capped_at_top_speed() -> None:
    # Huge radius -> physics would allow more than the car can actually do.
    v = max_corner_speed_ms(EXAMPLE_CAR, curvature_per_m=1 / 100_000)
    assert v == pytest.approx(EXAMPLE_CAR.max_speed_ms)


def test_accel_limited_speed_increases_with_distance_and_caps_at_top_speed() -> None:
    v_short = accel_limited_speed_ms(EXAMPLE_CAR, entry_speed_ms=20.0, distance_m=10.0)
    v_long = accel_limited_speed_ms(
        EXAMPLE_CAR, entry_speed_ms=20.0, distance_m=10_000.0
    )
    assert 20.0 < v_short < v_long
    assert v_long == EXAMPLE_CAR.max_speed_ms


def test_brake_and_accel_limits_are_symmetric_under_equal_g() -> None:
    # Same g both ways -> braking from v1 to v2 over distance is the mirror of
    # accelerating from v2 to v1 over that same distance.
    car = replace(EXAMPLE_CAR, max_accel_g=2.0, max_braking_g=2.0)
    accelerated = accel_limited_speed_ms(car, entry_speed_ms=30.0, distance_m=40.0)
    braked = brake_limited_speed_ms(car, exit_speed_ms=30.0, distance_m=40.0)
    assert math.isclose(accelerated, braked)
