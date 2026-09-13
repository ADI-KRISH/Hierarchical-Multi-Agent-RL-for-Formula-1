"""The "drive at the limit" policy: turns a track + car into a lap time.

Pure functions, no I/O. Three-pass speed-profile algorithm: cap each segment at
its cornering-grip limit, then sweep forward under max acceleration and
backward under max braking. A few iterations converge the closed loop (a
segment's exit speed feeds the next segment's entry, wrapping past the start).
"""

from f1rl.config import CarParams
from f1rl.envs.track import Track
from f1rl.models.car import (
    accel_limited_speed_ms,
    brake_limited_speed_ms,
    max_corner_speed_ms,
)


def speed_profile_ms(car: CarParams, track: Track, iterations: int = 3) -> list[float]:
    """Speed (m/s) at the start of each segment, driving the lap at the limit."""
    n = len(track.segments)
    speeds = [max_corner_speed_ms(car, s.curvature_per_m) for s in track.segments]

    for _ in range(iterations):
        for i in range(n):
            prev = (i - 1) % n
            reachable = accel_limited_speed_ms(
                car, speeds[prev], track.segments[prev].length_m
            )
            speeds[i] = min(speeds[i], reachable)
        for i in reversed(range(n)):
            nxt = (i + 1) % n
            length = track.segments[i].length_m
            allowed = brake_limited_speed_ms(car, speeds[nxt], length)
            speeds[i] = min(speeds[i], allowed)

    return speeds


def lap_time_s(car: CarParams, track: Track, iterations: int = 3) -> float:
    """Total lap time (s). Per segment: time = length / average(entry, exit speed)."""
    speeds = speed_profile_ms(car, track, iterations)
    n = len(speeds)
    total = 0.0
    for i, segment in enumerate(track.segments):
        avg_speed = (speeds[i] + speeds[(i + 1) % n]) / 2
        total += segment.length_m / avg_speed
    return total


if __name__ == "__main__":
    from f1rl.config import EXAMPLE_CAR
    from f1rl.envs.track import example_track

    print(f"Sample lap time: {lap_time_s(EXAMPLE_CAR, example_track()):.2f}s")
