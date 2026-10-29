"""Frenet-frame point-mass car: where the car is relative to the track, and how a
steering command changes it. Pure functions, no I/O, no globals.

State: ``s`` distance along the centreline, ``n`` lateral offset (positive = left),
``psi`` heading relative to the centreline's tangent, ``v`` speed. With centreline
curvature ``kappa`` (positive = left), the standard kinematics are::

    ds/dt  = v cos(psi) / (1 - n kappa)
    dn/dt  = v sin(psi)
    dpsi/dt = a_y / v - kappa ds/dt          (a_y: centripetal accel, left positive)
    dv/dt  = a_x - drag v^2           (drag: `CarParams.drag_per_m`, 0 = none)

The offset ``n`` now changes the corner the car drives: sitting on the inside of a
bend shortens it (``ds/dt`` is faster along the centreline) and opens its radius
-- which is the whole racing line.

Grip is one friction ellipse shared by the two axes: `map_action` shrinks any
(throttle/brake, steer) command onto what the tyres can deliver.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from f1rl.config import GRAVITY_M_S2, CarParams


@dataclass(frozen=True)
class FrenetState:
    s_m: float
    n_m: float
    psi_rad: float
    speed_ms: float


def map_action(car: CarParams, u_long: float, u_lat: float) -> tuple[float, float]:
    """Command in [-1, 1]^2 -> (a_x, a_y) in m/s^2 inside the friction ellipse.

    ``u_long`` > 0 is throttle (up to ``max_accel_g``), < 0 brake (``max_braking_g``);
    ``u_lat`` > 0 asks for a left turn (up to ``max_lateral_g``). A command outside
    the unit circle is shrunk radially onto it, so "as hard as possible" always means
    the limit and never past it: ``(a_x/ax_max)^2 + (a_y/ay_max)^2 <= 1``.
    """
    u_long = min(1.0, max(-1.0, u_long))
    u_lat = min(1.0, max(-1.0, u_lat))
    norm = math.hypot(u_long, u_lat)
    if norm > 1.0:
        u_long, u_lat = u_long / norm, u_lat / norm
    ax_max = (car.max_accel_g if u_long >= 0 else car.max_braking_g) * GRAVITY_M_S2
    return u_long * ax_max, u_lat * car.max_lateral_g * GRAVITY_M_S2


def _rates(
    state: FrenetState,
    a_x: float,
    a_y: float,
    kappa: float,
    speed_floor_ms: float,
    drag_per_m: float = 0.0,
) -> tuple[float, float, float, float]:
    """(ds/dt, dn/dt, dpsi/dt, dv/dt) at ``state`` -- see the module docstring."""
    v = state.speed_ms
    # Inside a tight bend n * kappa -> 1; keep the divisor away from zero.
    ds = v * math.cos(state.psi_rad) / max(0.1, 1.0 - state.n_m * kappa)
    return (
        ds,
        v * math.sin(state.psi_rad),
        a_y / max(v, speed_floor_ms) - kappa * ds,
        a_x - drag_per_m * v * v,
    )


def frenet_step(
    car: CarParams,
    state: FrenetState,
    a_x: float,
    a_y: float,
    kappa_at: Callable[[float], float],
    dt_s: float,
    speed_floor_ms: float,
) -> FrenetState:
    """Advance the car ``dt_s`` under centripetal acceleration ``a_y`` and tangential
    ``a_x`` (held constant over the step). ``kappa_at(s)`` is the centreline's
    curvature. ``speed_floor_ms`` stops the turn rate ``a_y / v`` blowing up near rest.

    Midpoint (second-order) integration: a first-order Euler step drifts a car that is
    exactly following a line by about a metre per lap at 50 Hz, which is more than
    the margin a racing line leaves.
    """
    drag = car.drag_per_m
    k1 = _rates(state, a_x, a_y, kappa_at(state.s_m), speed_floor_ms, drag)
    half = FrenetState(
        state.s_m + 0.5 * dt_s * k1[0],
        state.n_m + 0.5 * dt_s * k1[1],
        state.psi_rad + 0.5 * dt_s * k1[2],
        state.speed_ms + 0.5 * dt_s * k1[3],
    )
    k2 = _rates(half, a_x, a_y, kappa_at(half.s_m), speed_floor_ms, drag)
    v = min(car.max_speed_ms, max(0.0, state.speed_ms + dt_s * k2[3]))
    return FrenetState(
        state.s_m + dt_s * k2[0],
        state.n_m + dt_s * k2[1],
        state.psi_rad + dt_s * k2[2],
        v,
    )
