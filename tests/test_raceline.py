import math

import numpy as np
import pytest

from f1rl.config import EXAMPLE_CAR, GRAVITY_M_S2
from f1rl.models.raceline import (
    _reach_v2,
    gg_lap_time_s,
    gg_speed_profile,
    line_xy,
    min_curvature_line,
    optimal_line,
)
from racetrack_fixtures import stadium_track

MARGIN_M = 1.75


def test_reach_with_no_cornering_is_plain_kinematics() -> None:
    assert _reach_v2(c=100.0, d=50.0, q=0.0) == pytest.approx(150.0)


def test_reach_solves_its_defining_equation() -> None:
    c, d, q = 900.0, 400.0, 0.001
    x = _reach_v2(c, d, q)
    assert x == pytest.approx(c + d * math.sqrt(1 - (x * q) ** 2), rel=1e-9)


def test_cornering_leaves_less_grip_for_accelerating() -> None:
    assert _reach_v2(900.0, 400.0, 0.001) < _reach_v2(900.0, 400.0, 0.0)


def test_constant_curvature_runs_at_the_corner_limit() -> None:
    kappa = np.full(200, 1 / 100.0)
    v = gg_speed_profile(EXAMPLE_CAR, 5.0, kappa)
    limit = math.sqrt(EXAMPLE_CAR.max_lateral_g * GRAVITY_M_S2 * 100.0)
    assert v == pytest.approx(np.full(200, limit), rel=1e-6)


def test_profile_respects_the_friction_ellipse() -> None:
    """Speed never changes faster than the grip left over from cornering allows."""
    track = stadium_track()
    v = gg_speed_profile(EXAMPLE_CAR, track.ds_m, track.kappa)
    ay = EXAMPLE_CAR.max_lateral_g * GRAVITY_M_S2
    for i in range(track.n):
        j = (i + 1) % track.n
        q = (abs(track.kappa[i]) + abs(track.kappa[j])) / 2 / ay
        rate = (v[j] ** 2 - v[i] ** 2) / (2 * track.ds_m)  # m/s^2, + = accelerating
        limit = (
            EXAMPLE_CAR.max_accel_g if rate > 0 else EXAMPLE_CAR.max_braking_g
        ) * GRAVITY_M_S2
        spare = math.sqrt(max(0.0, 1 - (max(v[i], v[j]) ** 2 * q) ** 2))
        assert abs(rate) <= limit * spare + 1e-3


def test_optimal_line_stays_inside_the_edges() -> None:
    track = stadium_track()
    alpha = min_curvature_line(track, MARGIN_M)
    assert np.all(alpha <= track.w_left - MARGIN_M + 1e-6)
    assert np.all(alpha >= -(track.w_right - MARGIN_M) - 1e-6)


def test_optimal_line_uses_the_width_and_beats_the_centreline() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    assert float(np.ptp(line.alpha)) > 3.0  # it swings across the track
    assert line.lap_time_s < line.centre_lap_time_s


def test_a_track_narrower_than_the_car_is_rejected() -> None:
    with pytest.raises(ValueError, match="narrower"):
        min_curvature_line(stadium_track(width_m=3.0), MARGIN_M)


def test_line_xy_offsets_to_the_left() -> None:
    track = stadium_track()
    x, y = line_xy(track, np.full(track.n, 2.0))
    assert np.allclose(np.hypot(x - track.x, y - track.y), 2.0)


def test_lap_time_of_a_flat_out_circle_is_circumference_over_speed() -> None:
    n = 400
    kappa = np.full(n, 1 / 100.0)
    ds = 2 * math.pi * 100.0 / n
    v = math.sqrt(EXAMPLE_CAR.max_lateral_g * GRAVITY_M_S2 * 100.0)
    assert gg_lap_time_s(EXAMPLE_CAR, ds, kappa) == pytest.approx(
        2 * math.pi * 100.0 / v, rel=1e-6
    )
