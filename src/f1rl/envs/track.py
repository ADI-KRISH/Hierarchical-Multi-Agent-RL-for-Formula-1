"""Track geometry: a lap as a sequence of constant-curvature, constant-width
stretches.

Phase 1 needs this to compute lap times for a fixed policy; phase 2's
`DriverEnv` reads distance-along-track, curvature, and width from it for the
observation space and track-limit check.
"""

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass
from functools import cached_property

from f1rl.config import DEFAULT_TRACK_WIDTH_M


@dataclass(frozen=True)
class Segment:
    """A constant-curvature, constant-width stretch of track.

    ``curvature_per_m`` is 1/radius (signed by turn direction, positive = left);
    0.0 is a straight. ``width_m`` is per segment because real circuits narrow at
    some corners.
    """

    length_m: float
    curvature_per_m: float
    width_m: float = DEFAULT_TRACK_WIDTH_M


@dataclass(frozen=True)
class Track:
    """A closed lap: an ordered tuple of segments."""

    segments: tuple[Segment, ...]

    @cached_property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)

    @cached_property
    def segment_starts_m(self) -> tuple[float, ...]:
        """Distance from the start line at which each segment begins."""
        starts = [0.0]
        for segment in self.segments[:-1]:
            starts.append(starts[-1] + segment.length_m)
        return tuple(starts)


def _segment_at(track: Track, distance_m: float) -> Segment:
    position_m = distance_m % track.total_length_m
    index = bisect.bisect_right(track.segment_starts_m, position_m) - 1
    return track.segments[index]


def curvature_at(track: Track, distance_m: float) -> float:
    """Curvature (1/m) at ``distance_m`` around the lap, wrapping past the start."""
    return _segment_at(track, distance_m).curvature_per_m


def width_at(track: Track, distance_m: float) -> float:
    """Track width (m) at ``distance_m`` around the lap, wrapping past the start."""
    return _segment_at(track, distance_m).width_m


def max_abs_curvature_between(track: Track, start_m: float, end_m: float) -> float:
    """Tightest curvature (largest |1/m|) anywhere in [start_m, end_m) of the lap.

    Covers every segment the window touches, wrapping past the start line, so a
    short corner can't hide between two point samples.
    """
    n = len(track.segments)
    lap_m = track.total_length_m
    index = bisect.bisect_right(track.segment_starts_m, start_m % lap_m) - 1
    position_m = start_m - start_m % lap_m + track.segment_starts_m[index]
    tightest = 0.0
    while position_m < end_m:
        segment = track.segments[index % n]
        tightest = max(tightest, abs(segment.curvature_per_m))
        position_m += segment.length_m
        index += 1
    return tightest


def track_xy(track: Track, step_m: float = 2.0) -> list[tuple[float, float]]:
    """Centerline (x, y) points at most `step_m` apart, rebuilt from curvature.

    Starts at the origin heading along +x, and walks each segment exactly (a
    constant-curvature arc), so a well-formed closed lap ends back at the origin.
    Only the map plot needs this; the physics works in distance-along-track.
    """
    x = y = heading = 0.0
    points = [(x, y)]
    for segment in track.segments:
        n = max(1, math.ceil(segment.length_m / step_m))
        ds = segment.length_m / n
        for _ in range(n):
            # Chord of a constant-curvature arc points along its midpoint heading.
            turn = segment.curvature_per_m * ds
            chord = (
                ds if turn == 0.0 else 2 * math.sin(turn / 2) / segment.curvature_per_m
            )
            x += chord * math.cos(heading + turn / 2)
            y += chord * math.sin(heading + turn / 2)
            heading += turn
            points.append((x, y))
    return points


def example_track() -> Track:
    """A synthetic oval: two straights and two 180-degree corners, narrowing at
    the corners the way real circuits do.

    Not a real circuit -- a minimal parameterized lap for sanity-checking the
    physics model. Geometry is a free design choice for the simulator, not a
    physical constant, so it needs no citation.
    """
    straight = Segment(length_m=800.0, curvature_per_m=0.0, width_m=15.0)
    corner_radius_m = 100.0
    corner = Segment(
        length_m=math.pi * corner_radius_m,  # semicircle
        curvature_per_m=1.0 / corner_radius_m,
        width_m=10.0,
    )
    return Track(segments=(straight, corner, straight, corner))


def technical_track() -> Track:
    """A synthetic closed lap with mixed corners: a slow 15 m hairpin, a fast
    sweeper, a chicane, and a long main straight.

    Like `example_track`, a free design choice rather than a real circuit. It is
    four left-hand 90-degree corners (radii 15/60/120/30 m) joined by straights,
    with straight lengths chosen so the lap closes exactly, plus an S-shaped
    chicane on the back straight whose net heading change and sideways shift are
    both zero.
    """
    quarter = math.pi / 2

    def corner(radius_m: float, width_m: float = DEFAULT_TRACK_WIDTH_M) -> Segment:
        return Segment(quarter * radius_m, 1.0 / radius_m, width_m)

    chicane_radius_m = 25.0
    chicane_angle = math.pi / 6  # 30 degrees each way; spans 4 r sin(30) = 50 m
    chicane = (
        Segment(chicane_angle * chicane_radius_m, 1.0 / chicane_radius_m, 10.0),
        Segment(2 * chicane_angle * chicane_radius_m, -1.0 / chicane_radius_m, 10.0),
        Segment(chicane_angle * chicane_radius_m, 1.0 / chicane_radius_m, 10.0),
    )
    # Closure: x: 1000 + 15 - 60 - (400 + 50 + 415) - 120 + 30 = 0
    #          y: 15 + 375 + 60 - 120 - 300 - 30 = 0
    return Track(
        segments=(
            Segment(1000.0, 0.0, 15.0),  # main straight
            corner(15.0, 10.0),  # hairpin
            Segment(375.0, 0.0),
            corner(60.0),
            Segment(400.0, 0.0),
            *chicane,
            Segment(415.0, 0.0),
            corner(120.0),  # fast sweeper
            Segment(300.0, 0.0),
            corner(30.0),
        )
    )


#: Synthetic tracks by name, for CLIs and reports. Real circuits come from
#: `envs.circuits.load_circuit` instead (they need FastF1 data).
SYNTHETIC_TRACKS: dict[str, Callable[[], Track]] = {
    "oval": example_track,
    "technical": technical_track,
}
