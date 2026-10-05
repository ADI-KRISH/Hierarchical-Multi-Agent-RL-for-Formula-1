"""Real circuit geometry via FastF1 telemetry, cached under data/.

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
import re
from pathlib import Path

import numpy as np

from f1rl.config import DEFAULT_TRACK_WIDTH_M
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
