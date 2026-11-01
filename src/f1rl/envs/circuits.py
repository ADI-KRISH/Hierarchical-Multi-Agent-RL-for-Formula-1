"""Real circuit geometry, cached under data/. Two sources:

1. **Circuit centerlines** (`load_real_circuit`): the open f1-circuits dataset
   (github.com/bacinger/f1-circuits, MIT licence), pinned to one commit. Each
   circuit is a GPS polyline (lon/lat) plus its official lap length. Works for
   circuits that are no longer raced (e.g. Sepang), which FastF1 cannot cover:
   its position telemetry only exists from 2018 on.
2. **FastF1 telemetry** (`load_circuit`), described below.

FastF1 reconstructs each lap's car position (X, Y, cumulative Distance) from
official timing-and-GPS data -- real circuit shape, not a guess (CLAUDE.md:
"don't invent F1 numbers"). We take one clean lap's position trace and turn it
into a curvature-per-distance profile, resampled into the same constant-
curvature `Segment` list `Track` already uses, at few-metre resolution -- fine
enough that no per-corner arc-fitting is needed.

FastF1's X/Y position channels are in 1/10 metre, not metres -- verified
against `Distance` (metres): the raw X/Y polyline's summed segment length is
~10x the lap's Distance range. Divided out in `_fetch_lap_position` below.
"""

import json
import math
import re
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import splev, splprep

from f1rl.config import (
    CIRCUIT_CURVATURE_WINDOW_M,
    DEFAULT_TRACK_WIDTH_M,
    EARTH_MEAN_RADIUS_M,
)
from f1rl.envs.track import Segment, Track

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
CIRCUIT_CACHE_DIR = DATA_DIR / "circuits"
FASTF1_CACHE_DIR = DATA_DIR / "fastf1_cache"

_POSITION_UNITS_PER_M = 10.0  # see module docstring


def load_circuit(
    year: int,
    grand_prix: str,
    session: str = "Q",
    step_m: float = 5.0,
    width_m: float = DEFAULT_TRACK_WIDTH_M,
) -> Track:
    """Real track geometry for `grand_prix` in `year`, from FastF1's fastest lap.

    Cached under data/circuits/ after the first call, which needs network
    access; later calls for the same (year, grand_prix, session, step_m) just
    read the cache.
    """
    cache_path = (
        CIRCUIT_CACHE_DIR / f"{year}_{_slug(grand_prix)}_{session}_{step_m:g}m.json"
    )
    if cache_path.exists():
        stations = json.loads(cache_path.read_text())["stations"]
    else:
        x, y, distance = _fetch_lap_position(year, grand_prix, session)
        stations = segments_from_position(x, y, distance, step_m)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "source": f"FastF1 {year} {grand_prix} {session}, fastest lap",
                    "step_m": step_m,
                    "stations": stations,
                }
            )
        )
    return Track(
        segments=tuple(
            Segment(length_m=length, curvature_per_m=curvature, width_m=width_m)
            for length, curvature in stations
        )
    )


def segments_from_position(
    x: np.ndarray, y: np.ndarray, distance: np.ndarray, step_m: float
) -> list[list[float]]:
    """Resample a closed-lap (x, y) trace by arc length and compute curvature.

    Pure function (no network): given any distance-parameterized closed-loop
    path, returns [[length_m, curvature_per_m], ...] stations `step_m` apart
    whose lengths sum to the full lap. The trace's last point is taken to meet
    its first, so the path is treated as periodic throughout. A real position
    trace is noisy, so the resampled path is smoothed -- wrap-around -- before
    differentiating twice (position -> heading -> curvature), which is what
    actually needs it.
    """
    n = max(8, round((distance[-1] - distance[0]) / step_m))
    ds = (distance[-1] - distance[0]) / n
    grid = distance[0] + ds * np.arange(n)  # n stations; the lap closes the loop
    gx = _smooth_periodic(np.interp(grid, distance, x))
    gy = _smooth_periodic(np.interp(grid, distance, y))
    heading = np.arctan2(np.roll(gy, -1) - gy, np.roll(gx, -1) - gx)
    turn = np.angle(np.exp(1j * (np.roll(heading, -1) - heading)))  # wrap to +-pi
    curvature = _smooth_periodic(turn / ds)
    return [[ds, float(c)] for c in curvature]


def _smooth_periodic(values: np.ndarray, window: int = 5) -> np.ndarray:
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="wrap")
    return np.convolve(padded, kernel, mode="valid")


def _fetch_lap_position(
    year: int, grand_prix: str, session: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import fastf1  # local import: heavy and network-capable, only on a cache miss

    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    ev = fastf1.get_session(year, grand_prix, session)
    ev.load(telemetry=True, laps=True, weather=False, messages=False)
    lap = ev.laps.pick_fastest()
    tel = lap.get_telemetry().dropna(subset=["Distance", "X", "Y"])
    tel = tel.drop_duplicates(subset="Distance").sort_values("Distance")
    x = tel["X"].to_numpy() / _POSITION_UNITS_PER_M
    y = tel["Y"].to_numpy() / _POSITION_UNITS_PER_M
    distance = tel["Distance"].to_numpy()
    return x, y, distance


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


#: Pinned commit of github.com/bacinger/f1-circuits (MIT licence), so circuit
#: geometry never changes under a run.
GEO_DATASET_COMMIT = "394d8fbe70ef2c0b0c8d23ff7bee61fa09606055"
GEO_DATASET_URL = (
    "https://raw.githubusercontent.com/bacinger/f1-circuits/"
    f"{GEO_DATASET_COMMIT}/f1-circuits.geojson"
)

#: Bump when `segments_from_lonlat` changes, so cached circuits are rebuilt.
GEO_PROCESSING_VERSION = 2

#: Track name -> dataset circuit id.
REAL_CIRCUITS: dict[str, str] = {
    "zandvoort": "nl-1948",
    "spa": "be-1925",
    "suzuka": "jp-1962",
    "monaco": "mc-1929",
    "sepang": "my-1999",
    "interlagos": "br-1940",
}


def load_real_circuit(
    name: str, step_m: float = 5.0, width_m: float = DEFAULT_TRACK_WIDTH_M
) -> Track:
    """Real geometry for a circuit in `REAL_CIRCUITS`, from its GPS centerline.

    The first call downloads the pinned dataset (network); afterwards it reads
    the cache under data/circuits/.
    """
    circuit_id = REAL_CIRCUITS[name]
    cache_path = (
        CIRCUIT_CACHE_DIR / f"geo{GEO_PROCESSING_VERSION}_{circuit_id}_{step_m:g}m.json"
    )
    if cache_path.exists():
        stations = json.loads(cache_path.read_text())["stations"]
    else:
        feature = _geo_feature(circuit_id)
        lon, lat = np.array(feature["geometry"]["coordinates"]).T
        props = feature["properties"]
        stations = segments_from_lonlat(lon, lat, float(props["length"]), step_m)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {
                    "source": f"bacinger/f1-circuits@{GEO_DATASET_COMMIT[:8]} "
                    f"{circuit_id} ({props['Name']})",
                    "official_length_m": props["length"],
                    "step_m": step_m,
                    "stations": stations,
                }
            )
        )
    return Track(
        segments=tuple(
            Segment(length_m=length, curvature_per_m=curvature, width_m=width_m)
            for length, curvature in stations
        )
    )


def segments_from_lonlat(
    lon: np.ndarray,
    lat: np.ndarray,
    official_length_m: float,
    step_m: float,
    window_m: float = CIRCUIT_CURVATURE_WINDOW_M,
) -> list[list[float]]:
    """A closed GPS centerline -> [[length_m, curvature_per_m], ...] stations.

    Pure function. Projects lon/lat to local metres (equirectangular about the
    centroid -- exact enough over a few km), fits a periodic interpolating cubic
    spline, takes curvature from the spline's exact derivatives, rescales to the
    official lap length, resamples every `step_m`, and averages the curvature
    over `window_m` along the track (wrapping round the lap). An interpolating
    spline, not a smoothing one: with points this sparse, a smoothing spline
    drops knots and distorts even a perfect circle by ~10%.
    """
    points = np.column_stack([lon, lat])
    if np.allclose(points[0], points[-1]):
        points = points[:-1]  # closed rings repeat the first point
    lat0 = math.radians(float(points[:, 1].mean()))
    x = EARTH_MEAN_RADIUS_M * np.radians(points[:, 0] - points[:, 0].mean())
    x = x * math.cos(lat0)
    y = EARTH_MEAN_RADIUS_M * np.radians(points[:, 1] - points[:, 1].mean())

    tck, _u = splprep([np.r_[x, x[0]], np.r_[y, y[0]]], s=0, per=1)
    u = np.linspace(0.0, 1.0, 50 * len(x), endpoint=False)
    dx, dy = splev(u, tck, der=1)
    ddx, ddy = splev(u, tck, der=2)
    speed = np.hypot(dx, dy)  # metres per unit spline parameter
    curvature = (dx * ddy - dy * ddx) / speed**3

    scale = official_length_m / float(np.sum(speed) / len(u))
    arc_m = (np.cumsum(speed) - speed[0]) / len(u) * scale
    n = max(8, round(official_length_m / step_m))
    ds = official_length_m / n
    grid = ds * (np.arange(n) + 0.5)  # each station's curvature at its midpoint
    station_curvature = np.interp(grid, arc_m, curvature / scale)
    window = max(1, round(window_m / ds)) | 1  # odd, so the average is centred
    station_curvature = _smooth_periodic(station_curvature, window)
    return [[ds, float(c)] for c in station_curvature]


def _geo_feature(circuit_id: str) -> dict[str, Any]:
    dataset = CIRCUIT_CACHE_DIR / f"f1-circuits-{GEO_DATASET_COMMIT[:8]}.geojson"
    if not dataset.exists():
        dataset.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(GEO_DATASET_URL, timeout=60) as response:
            dataset.write_bytes(response.read())
    features = json.loads(dataset.read_text())["features"]
    return next(f for f in features if f["properties"]["id"] == circuit_id)
