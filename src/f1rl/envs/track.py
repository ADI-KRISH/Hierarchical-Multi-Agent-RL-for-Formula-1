"""Track geometry: a lap as a sequence of constant-curvature segments.

Phase 1 needs this to compute lap times for a fixed policy; phase 2's
`DriverEnv` will read distance-along-track and curvature from it for the
observation space.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    """A constant-curvature stretch of track.

    ``curvature_per_m`` is 1/radius (signed by turn direction); 0.0 is a straight.
    """

    length_m: float
    curvature_per_m: float


@dataclass(frozen=True)
class Track:
    """A closed lap: an ordered tuple of segments, plus a constant track width.

    ``width_m`` is one value for the whole lap -- real circuits narrow at some
    corners, but that's a refinement the MVP's track-limit check doesn't need.
    """

    segments: tuple[Segment, ...]
    width_m: float = 12.0  # design choice for the synthetic track, not a cited spec.

    @property
    def total_length_m(self) -> float:
        return sum(s.length_m for s in self.segments)


def curvature_at(track: Track, distance_m: float) -> float:
    """Curvature (1/m) at ``distance_m`` around the lap, wrapping past the start."""
    remaining = distance_m % track.total_length_m
    for segment in track.segments:
        if remaining < segment.length_m:
            return segment.curvature_per_m
        remaining -= segment.length_m
    return track.segments[-1].curvature_per_m  # floating-point edge at the seam


def example_track() -> Track:
    """A synthetic oval: two straights and two 180-degree corners.

    Not a real circuit -- a minimal parameterized lap for sanity-checking the
    physics model ahead of phase 2's `DriverEnv`. Geometry is a free design
    choice for the simulator, not a physical constant, so it needs no citation.
    """
    straight = Segment(length_m=800.0, curvature_per_m=0.0)
    corner_radius_m = 100.0
    corner = Segment(
        length_m=math.pi * corner_radius_m,  # semicircle
        curvature_per_m=1.0 / corner_radius_m,
    )
    return Track(segments=(straight, corner, straight, corner))
