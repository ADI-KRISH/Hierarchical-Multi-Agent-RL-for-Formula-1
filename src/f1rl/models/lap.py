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

import bisect
import itertools
import math

from f1rl.config import GRAVITY_M_S2, CarParams
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


def max_safe_speed_ms(car: CarParams, track: Track, distance_m: float) -> float:
    """Fastest the car can be going at `distance_m` and still brake down to every
    corner's grip limit ahead of it -- i.e. it hasn't yet passed a braking point.

    Looks one full-speed stopping distance ahead; nothing farther can constrain
    the car. Exact for the segment model: each segment's limit applies from its
    start (or from here, for the segment the car is in).
    """
    horizon_m = car.max_speed_ms**2 / (2 * car.max_braking_g * GRAVITY_M_S2)
    lap_m = track.total_length_m
    starts = track.segment_starts_m
    n = len(track.segments)
    position_m = distance_m % lap_m
    index = bisect.bisect_right(starts, position_m) - 1
    ahead_m = 0.0  # distance from here to the start of the segment at `index`
    safe = car.max_speed_ms
    while ahead_m <= horizon_m:
        segment = track.segments[index % n]
        corner_ms = max_corner_speed_ms(car, segment.curvature_per_m)
        safe = min(safe, brake_limited_speed_ms(car, corner_ms, ahead_m))
        segment_end_m = starts[index % n] + segment.length_m + lap_m * (index // n)
        ahead_m = segment_end_m - position_m
        index += 1
    return safe


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
