"""`RaceTrack`: a closed lap as a uniformly spaced centreline with real widths.

The Frenet-frame env (`RaceEnv`), the optimal-line solver and the lap sim all work on
this: station ``i`` sits ``i * ds_m`` along the centreline and carries its position,
heading, signed curvature (positive = left) and the distance to the left and right
track edge. Pure data plus pure functions, no I/O.

Source of the real tracks: TUMFTM/racetrack-database (LGPL-3.0 data, downloaded at
runtime and cached under ``data/``, never vendored) -- see `load_tum_track`.
"""

import json
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.interpolate import splev, splprep

from f1rl.config import CIRCUIT_CURVATURE_WINDOW_M
from f1rl.envs.track import Segment, Track

Array = NDArray[np.float64]

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
TUM_CACHE_DIR = DATA_DIR / "tum"

#: Pinned commit of github.com/TUMFTM/racetrack-database (LGPL-3.0 data), so track
#: geometry never changes under a run.
TUM_COMMIT = "e59595d1f3573b30d1ded6a08984935b957688e0"
TUM_URL = f"https://raw.githubusercontent.com/TUMFTM/racetrack-database/{TUM_COMMIT}"

#: The 25 tracks of the database (file stems).
TUM_TRACKS: tuple[str, ...] = (
    "Austin",
    "BrandsHatch",
    "Budapest",
    "Catalunya",
    "Hockenheim",
    "IMS",
    "Melbourne",
    "MexicoCity",
    "Montreal",
    "Monza",
    "MoscowRaceway",
    "Norisring",
    "Nuerburgring",
    "Oschersleben",
    "Sakhir",
    "SaoPaulo",
    "Sepang",
    "Shanghai",
    "Silverstone",
    "Sochi",
    "Spa",
    "Spielberg",
    "Suzuka",
    "YasMarina",
    "Zandvoort",
)

#: Never trained on, only evaluated: a mix of fast, technical and long circuits
#: (docs/racing_line_research.md, step 1).
TUM_HELD_OUT: tuple[str, ...] = (
    "Suzuka",
    "Spa",
    "SaoPaulo",
    "Silverstone",
    "YasMarina",
)

#: Used to pick the best checkpoint during training -- never trained on, and kept
#: apart from `TUM_HELD_OUT` so choosing a model never touches the final test tracks.
TUM_VALIDATION: tuple[str, ...] = ("Montreal", "Hockenheim", "Catalunya")

#: Everything else: the tracks the driver trains on.
TUM_TRAIN: tuple[str, ...] = tuple(
    t for t in TUM_TRACKS if t not in TUM_HELD_OUT and t not in TUM_VALIDATION
)

#: Bump when `racetrack_from_xy` changes, so cached tracks are rebuilt.
RACETRACK_VERSION = 4


@dataclass(frozen=True, eq=False)
class RaceTrack:
    """A closed lap on ``n`` stations spaced ``ds_m`` apart along the centreline."""

    name: str
    ds_m: float
    x: Array
    y: Array
    heading: Array  # rad, direction of travel
    kappa: Array  # 1/m, signed (positive = left turn)
    w_left: Array  # m, centreline to left edge
    w_right: Array  # m, centreline to right edge

    @property
    def n(self) -> int:
        return len(self.x)

    @property
    def length_m(self) -> float:
        return self.ds_m * self.n

    def to_track(self) -> Track:
        """The same lap as the v1 `Track` (centreline only), one segment per station.

        Lets the v1 baseline driver and limit-lap code run on real widths.
        """
        return Track(
            tuple(
                Segment(self.ds_m, float(k), float(wl + wr))
                for k, wl, wr in zip(self.kappa, self.w_left, self.w_right, strict=True)
            )
        )


def racetrack_from_xy(
    name: str,
    x: Array,
    y: Array,
    w_right: Array,
    w_left: Array,
    step_m: float = 5.0,
    window_m: float = CIRCUIT_CURVATURE_WINDOW_M,
) -> RaceTrack:
    """Resample a closed centreline (+ widths per point) onto a uniform grid.

    Fits a periodic interpolating cubic spline through the points, takes heading
    and curvature from its exact derivatives, resamples every ``step_m`` of arc
    length, and averages curvature over ``window_m`` (wrapping round the lap) to
    drop digitisation wiggles. Widths are interpolated by fraction of the lap.
    """
    pts = np.column_stack([x, y])
    if np.allclose(pts[0], pts[-1]):
        pts, w_right, w_left = pts[:-1], w_right[:-1], w_left[:-1]
    px, py = pts[:, 0], pts[:, 1]
    tck, _u = splprep([np.r_[px, px[0]], np.r_[py, py[0]]], s=0, per=1)
    u = np.linspace(0.0, 1.0, 20 * len(px), endpoint=False)
    dx, dy = splev(u, tck, der=1)
    speed = np.hypot(dx, dy)
    length_m = float(np.sum(speed) / len(u))
    arc_m = (np.cumsum(speed) - speed[0]) / len(u)

    n = max(8, round(length_m / step_m))
    ds = length_m / n
    u_i = np.interp(ds * np.arange(n), arc_m, u)
    sx, sy = splev(u_i, tck)
    dx, dy = splev(u_i, tck, der=1)
    heading = np.arctan2(dy, dx)
    # Curvature as the central heading difference, not the spline's point
    # derivative: sampled every `ds` the point value aliases the spline's wiggles,
    # while this telescopes -- the sum of kappa * ds is the heading change -- and
    # is centred on the station, which is what the Frenet kinematics rely on.
    turn = np.angle(np.exp(1j * (np.roll(heading, -1) - np.roll(heading, 1))))
    kappa = turn / (2 * ds)
    window = max(1, round(window_m / ds)) | 1
    kappa = _smooth_periodic(kappa, window)

    # Where each input point sits along the lap, to interpolate its widths.
    chord = np.hypot(np.diff(np.r_[px, px[0]]), np.diff(np.r_[py, py[0]]))
    frac_in = np.r_[0.0, np.cumsum(chord)] / chord.sum()
    frac_out = np.arange(n) / n
    w_r = np.interp(frac_out, frac_in, np.r_[w_right, w_right[0]])
    w_l = np.interp(frac_out, frac_in, np.r_[w_left, w_left[0]])
    return RaceTrack(
        name=name,
        ds_m=ds,
        x=np.asarray(sx, dtype=np.float64),
        y=np.asarray(sy, dtype=np.float64),
        heading=heading,
        kappa=kappa,
        w_left=w_l,
        w_right=w_r,
    )


def _smooth_periodic(values: Array, window: int) -> Array:
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="wrap")
    return np.convolve(padded, kernel, mode="valid")


def _tum_file(kind: str, name: str) -> Path:
    path = TUM_CACHE_DIR / TUM_COMMIT[:8] / kind / f"{name}.csv"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(f"{TUM_URL}/{kind}/{name}.csv", timeout=60) as r:
            path.write_bytes(r.read())
    return path


def load_tum_raceline(name: str) -> Array:
    """TUM's own minimum-curvature racing line for ``name``: (n, 2) x, y in metres.

    Used as a benchmark for our optimiser, never as training input.
    """
    return np.loadtxt(_tum_file("racelines", name), delimiter=",", comments="#")


def load_tum_track(name: str, step_m: float = 5.0) -> RaceTrack:
    """A `RaceTrack` for one of `TUM_TRACKS`, cached under data/tum/ after the
    first call (which needs network access)."""
    if name not in TUM_TRACKS:
        raise KeyError(f"unknown TUM track {name!r}; known: {', '.join(TUM_TRACKS)}")
    cache = TUM_CACHE_DIR / f"v{RACETRACK_VERSION}_{TUM_COMMIT[:8]}_{name}_{step_m:g}m"
    cache = cache.with_suffix(".json")
    if cache.exists():
        d = json.loads(cache.read_text())
        return RaceTrack(
            name=name,
            ds_m=d["ds_m"],
            **{k: np.array(d[k]) for k in _ARRAYS},
        )
    data = np.loadtxt(_tum_file("tracks", name), delimiter=",", comments="#")
    # No curvature smoothing: the Frenet kinematics need `kappa` to be the true turn
    # rate of `heading`, or the car's heading drifts inside every corner.
    track = racetrack_from_xy(
        name, data[:, 0], data[:, 1], data[:, 2], data[:, 3], step_m, window_m=0.0
    )
    cache.write_text(
        json.dumps(
            {
                "source": f"TUMFTM/racetrack-database@{TUM_COMMIT[:8]} {name}",
                "ds_m": track.ds_m,
                **{k: getattr(track, k).tolist() for k in _ARRAYS},
            }
        )
    )
    return track


_ARRAYS = ("x", "y", "heading", "kappa", "w_left", "w_right")
