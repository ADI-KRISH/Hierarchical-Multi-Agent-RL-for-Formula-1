"""Track geometry: a lap as a sequence of constant-curvature, constant-width
stretches.

Phase 1 needs this to compute lap times for a fixed policy; phase 2's
`DriverEnv` reads distance-along-track, curvature, and width from it for the
observation space and track-limit check.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A constant-curvature, constant-width stretch of track.

    ``curvature_per_m`` is 1/radius (signed by turn direction); 0.0 is a straight.
    ``width_m`` is a design choice for the synthetic track, not a cited spec --
    real circuits narrow at some corners, which is why it's per segment.
    """

    length_m: float
    curvature_per_m: float
    width_m: float = 12.0


@dataclass(frozen=True)
class Track:
    """A closed lap: an ordered tuple of segments."""

    segments: tuple[Segment, ...]

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)


def _segment_at(track: Track, distance_m: float) -> Segment:
    remaining = distance_m % track.total_length_m
    for segment in track.segments:
        if remaining < segment.length_m:
            return segment
        remaining -= segment.length_m
    return track.segments[-1]  # floating-point edge at the seam


def curvature_at(track: Track, distance_m: float) -> float:
    """Curvature (1/m) at ``distance_m`` around the lap, wrapping past the start."""
    return _segment_at(track, distance_m).curvature_per_m


def width_at(track: Track, distance_m: float) -> float:
    """Track width (m) at ``distance_m`` around the lap, wrapping past the start."""
    return _segment_at(track, distance_m).width_m


def example_track() -> Track:
    """A synthetic oval: two straights and two 180-degree corners, narrowing at
    the corners the way real circuits do.

    Not a real circuit -- a minimal parameterized lap for sanity-checking the
    physics model ahead of phase 2's `DriverEnv`. Geometry is a free design
    choice for the simulator, not a physical constant, so it needs no citation.
    """
    straight = Segment(length_m=800.0, curvature_per_m=0.0, width_m=15.0)
    corner_radius_m = 100.0
    corner = Segment(
        length_m=math.pi * corner_radius_m,  # semicircle
        curvature_per_m=1.0 / corner_radius_m,
        width_m=10.0,
    )
    return Track(segments=(straight, corner, straight, corner))
