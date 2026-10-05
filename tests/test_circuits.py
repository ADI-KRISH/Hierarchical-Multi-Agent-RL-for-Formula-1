import math

import numpy as np
import pytest

from f1rl.envs.circuits import segments_from_position
from f1rl.envs.track import Segment, Track, curvature_at


def test_segments_from_position_recovers_a_circles_curvature() -> None:
    """A perfect circle of radius r has curvature 1/r everywhere -- the pure
    resampling/curvature pipeline should recover that from a raw point trace,
    no network involved.
    """
    radius = 50.0
    angles = np.linspace(0.0, 2 * math.pi, 400, endpoint=False)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    distance = radius * angles

    stations = segments_from_position(x, y, distance, step_m=2.0)
    segments = tuple(
        Segment(length_m=length, curvature_per_m=c) for length, c in stations
    )
    track = Track(segments=segments)

    expected = 1.0 / radius
    curvatures = [curvature_at(track, k * 2.0) for k in range(len(stations))]
    interior = curvatures[4:-4]  # skip stations touching the wrap-smoothing seam
    assert all(abs(abs(c) - expected) < expected * 0.05 for c in interior)


def test_segments_from_position_is_flat_on_a_straight_line() -> None:
    distance = np.linspace(0.0, 200.0, 100)
    x = distance.copy()
    y = np.zeros_like(distance)

    stations = segments_from_position(x, y, distance, step_m=5.0)

    # An open line isn't a lap: the periodic treatment closes it with a U-turn,
    # which (after two smoothing passes) bends the stations near the seam.
    interior = stations[6:-6]
    assert all(abs(c) < 1e-6 for _length, c in interior)


def test_segments_from_position_covers_the_whole_lap() -> None:
    """Regression: the stations must add up to the full lap distance -- the
    closing stretch from the last sample back to the first is part of the lap.
    """
    radius = 50.0
    angles = np.linspace(0.0, 2 * math.pi, 401)  # last point meets the first
    distance = radius * angles

    stations = segments_from_position(
        radius * np.cos(angles), radius * np.sin(angles), distance, step_m=2.0
    )

    assert sum(length for length, _c in stations) == pytest.approx(distance[-1])
