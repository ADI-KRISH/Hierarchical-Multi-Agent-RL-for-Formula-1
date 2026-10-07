import math

import numpy as np
import pytest

from f1rl.config import EARTH_MEAN_RADIUS_M
from f1rl.envs.circuits import (
    REAL_CIRCUITS,
    segments_from_lonlat,
    segments_from_position,
)
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


def test_segments_from_lonlat_recovers_a_circle_and_its_official_length() -> None:
    """A 300 m-radius circle drawn as 60 sparse GPS points (like the dataset's
    hand-digitised centerlines) comes back as curvature ~1/300 and is rescaled to
    the stated official length."""
    radius_m, lat0 = 300.0, 50.0
    angles = np.linspace(0.0, 2 * math.pi, 60, endpoint=False)
    deg_per_m = 180 / (math.pi * EARTH_MEAN_RADIUS_M)
    lat = lat0 + radius_m * np.sin(angles) * deg_per_m
    lon = 5.0 + radius_m * np.cos(angles) * deg_per_m / math.cos(math.radians(lat0))

    official = 2 * math.pi * radius_m * 1.01  # 1% longer, as if the map were off
    stations = segments_from_lonlat(lon, lat, official, step_m=5.0)

    assert sum(length for length, _c in stations) == pytest.approx(official)
    curvatures = np.array([c for _length, c in stations])
    expected = 1 / (radius_m * 1.01)
    assert np.allclose(np.abs(curvatures), expected, rtol=0.03)
    total_turn = float(np.sum(curvatures) * stations[0][0])
    assert abs(total_turn) == pytest.approx(2 * math.pi, rel=0.01)  # one full loop


def test_real_circuit_names_resolve_to_dataset_ids() -> None:
    assert set(REAL_CIRCUITS) == {
        "zandvoort",
        "spa",
        "suzuka",
        "monaco",
        "sepang",
        "interlagos",
    }
