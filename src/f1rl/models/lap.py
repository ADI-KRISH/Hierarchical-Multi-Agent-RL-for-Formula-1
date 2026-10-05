"""The "drive at the limit" policy: turns a track + car into a lap time.

Pure functions, no I/O. Fine-grained forward/backward speed-profile sweep:
sample the lap every `step_m`, cap each station at its local cornering-grip
limit, then sweep forward under max acceleration and backward under max
braking. One node per curvature-segment isn't fine enough -- the grip cap
would only bite at a segment's edge, letting the car accelerate straight
through the middle of a corner past its own lateral-grip limit. A few
iterations converge the closed loop (the last station's exit speed feeds
the first).
"""

import itertools
import math

from f1rl.config import CarParams
from f1rl.envs.track import Track, curvature_at
from f1rl.models.car import (
    accel_limited_speed_ms,
    brake_limited_speed_ms,
    max_corner_speed_ms,
)


def speed_profile_ms(
    car: CarParams, track: Track, step_m: float = 2.0, iterations: int = 3
) -> list[float]:
    """Speed (m/s) at evenly spaced stations, `step_m` apart, around the lap."""
    n = math.ceil(track.total_length_m / step_m)
    ds = track.total_length_m / n  # even spacing that divides the lap exactly
    speeds = [max_corner_speed_ms(car, curvature_at(track, k * ds)) for k in range(n)]

    for _ in range(iterations):
        for k in range(n):
            reachable = accel_limited_speed_ms(car, speeds[k - 1], ds)
            speeds[k] = min(speeds[k], reachable)
        for k in reversed(range(n)):
            allowed = brake_limited_speed_ms(car, speeds[(k + 1) % n], ds)
            speeds[k] = min(speeds[k], allowed)

    return speeds


def lap_time_s(
    car: CarParams, track: Track, step_m: float = 2.0, iterations: int = 3
) -> float:
    """Total lap time (s): sum of station_spacing / average(entry, exit speed)."""
    speeds = speed_profile_ms(car, track, step_m, iterations)
    n = len(speeds)
    ds = track.total_length_m / n
    total = 0.0
    for k in range(n):
        avg_speed = (speeds[k] + speeds[(k + 1) % n]) / 2
        total += ds / avg_speed
    return total


def standing_start_profile_ms(
    car: CarParams, track: Track, step_m: float = 2.0, iterations: int = 3
) -> list[float]:
    """Like `speed_profile_ms`, but launching from rest at the start line.

    One extra entry closes the lap: index ``n`` is the speed crossing the finish
    line, taken from the flying-lap profile (nothing slows the car for it).
    """
    flying = speed_profile_ms(car, track, step_m, iterations)
    n = len(flying)
    ds = track.total_length_m / n
    speeds = [0.0, *flying[1:], flying[0]]
    for k in range(1, n + 1):
        speeds[k] = min(speeds[k], accel_limited_speed_ms(car, speeds[k - 1], ds))
    return speeds


def standing_start_lap_time_s(
    car: CarParams, track: Track, step_m: float = 2.0, iterations: int = 3
) -> float:
    """Lap time (s) from a standing start at the limit -- what `DriverEnv`
    episodes are timed against, since each one launches from rest.
    """
    speeds = standing_start_profile_ms(car, track, step_m, iterations)
    ds = track.total_length_m / (len(speeds) - 1)
    return sum(ds / ((a + b) / 2) for a, b in itertools.pairwise(speeds))


if __name__ == "__main__":
    from f1rl.config import EXAMPLE_CAR
    from f1rl.envs.track import example_track

    print(f"Sample lap time: {lap_time_s(EXAMPLE_CAR, example_track()):.2f}s")
