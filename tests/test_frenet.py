import math

import pytest

from f1rl.config import EXAMPLE_CAR, GRAVITY_M_S2
from f1rl.models.frenet import FrenetState, frenet_step, map_action

G = GRAVITY_M_S2


def _straight(_s: float) -> float:
    return 0.0


def test_full_throttle_and_full_brake_use_their_own_limits() -> None:
    assert map_action(EXAMPLE_CAR, 1.0, 0.0)[0] == pytest.approx(1.5 * G)
    assert map_action(EXAMPLE_CAR, -1.0, 0.0)[0] == pytest.approx(-5.0 * G)


def test_full_lock_is_the_lateral_limit_and_left_is_positive() -> None:
    assert map_action(EXAMPLE_CAR, 0.0, 1.0)[1] == pytest.approx(5.0 * G)
    assert map_action(EXAMPLE_CAR, 0.0, -1.0)[1] == pytest.approx(-5.0 * G)


@pytest.mark.parametrize("u_long", [-1.0, -0.6, 0.0, 0.6, 1.0])
@pytest.mark.parametrize("u_lat", [-1.0, -0.4, 0.0, 0.4, 1.0])
def test_the_mapped_command_never_leaves_the_friction_ellipse(
    u_long: float, u_lat: float
) -> None:
    a_x, a_y = map_action(EXAMPLE_CAR, u_long, u_lat)
    ax_max = (EXAMPLE_CAR.max_accel_g if a_x >= 0 else EXAMPLE_CAR.max_braking_g) * G
    ay_max = EXAMPLE_CAR.max_lateral_g * G
    assert (a_x / ax_max) ** 2 + (a_y / ay_max) ** 2 <= 1 + 1e-9


def test_out_of_range_commands_are_clipped() -> None:
    assert map_action(EXAMPLE_CAR, 5.0, 0.0) == map_action(EXAMPLE_CAR, 1.0, 0.0)


def test_a_straight_car_advances_at_its_speed() -> None:
    state = FrenetState(0.0, 0.0, 0.0, 50.0)
    for _ in range(50):
        state = frenet_step(EXAMPLE_CAR, state, 0.0, 0.0, _straight, 0.02, 10.0)
    assert state.s_m == pytest.approx(50.0)
    assert state.n_m == pytest.approx(0.0, abs=1e-9)


def test_steering_left_moves_the_car_left() -> None:
    state = FrenetState(0.0, 0.0, 0.0, 50.0)
    for _ in range(25):
        state = frenet_step(EXAMPLE_CAR, state, 0.0, 10.0, _straight, 0.02, 10.0)
    assert state.n_m > 0 and state.psi_rad > 0


def test_following_a_constant_corner_exactly_keeps_the_car_on_the_centreline() -> None:
    kappa, v = 0.01, 40.0
    state = FrenetState(0.0, 0.0, 0.0, v)
    for _ in range(int(2 * math.pi / kappa / v / 0.02)):  # one full revolution
        state = frenet_step(
            EXAMPLE_CAR, state, 0.0, kappa * v * v, lambda _s: kappa, 0.02, 10.0
        )
    assert abs(state.n_m) < 0.01 and abs(state.psi_rad) < 1e-3


def test_the_inside_of_a_corner_is_shorter() -> None:
    """At the same speed a car on the inside of a bend covers more centreline per
    second than one on the outside: the whole point of a racing line."""
    kappa = 0.01
    inside = FrenetState(0.0, 4.0, 0.0, 40.0)
    outside = FrenetState(0.0, -4.0, 0.0, 40.0)
    s_in = frenet_step(EXAMPLE_CAR, inside, 0.0, 0.0, lambda _s: kappa, 0.02, 10.0).s_m
    s_out = frenet_step(
        EXAMPLE_CAR, outside, 0.0, 0.0, lambda _s: kappa, 0.02, 10.0
    ).s_m
    assert s_in > s_out


def test_speed_is_capped_and_never_negative() -> None:
    state = FrenetState(0.0, 0.0, 0.0, 96.9)
    state = frenet_step(EXAMPLE_CAR, state, 100.0, 0.0, _straight, 0.02, 10.0)
    assert state.speed_ms == EXAMPLE_CAR.max_speed_ms
    slow = FrenetState(0.0, 0.0, 0.0, 1.0)
    slow = frenet_step(EXAMPLE_CAR, slow, -1e3, 0.0, _straight, 0.02, 10.0)
    assert slow.speed_ms == 0.0
