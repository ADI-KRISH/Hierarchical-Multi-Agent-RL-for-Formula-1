"""Point-mass car physics: pure functions, no I/O, no globals.

Lateral and longitudinal grip are treated as independent axes (max_lateral_g /
max_accel_g / max_braking_g) rather than a combined friction circle -- a
simplification appropriate for phase 1's sanity check; combined-grip modeling
is future work.
"""

import math

from f1rl.config import GRAVITY_M_S2, CarParams


def max_corner_speed_ms(car: CarParams, curvature_per_m: float) -> float:
    """Max speed a corner of this curvature allows: v = sqrt(lateral_g * g * radius)."""
    if curvature_per_m == 0.0:
        return car.max_speed_ms
    radius_m = 1.0 / abs(curvature_per_m)
    v = math.sqrt(car.max_lateral_g * GRAVITY_M_S2 * radius_m)
    return min(v, car.max_speed_ms)


def accel_limited_speed_ms(
    car: CarParams, entry_speed_ms: float, distance_m: float
) -> float:
    """Speed reachable after accelerating at max_accel_g over distance_m."""
    v_squared = entry_speed_ms**2 + 2 * car.max_accel_g * GRAVITY_M_S2 * distance_m
    return min(math.sqrt(v_squared), car.max_speed_ms)


def brake_limited_speed_ms(
    car: CarParams, exit_speed_ms: float, distance_m: float
) -> float:
    """Speed distance_m before exit_speed_ms is reached, braking at max_braking_g."""
    v_squared = exit_speed_ms**2 + 2 * car.max_braking_g * GRAVITY_M_S2 * distance_m
    return math.sqrt(v_squared)
