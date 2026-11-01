"""The optimal racing line and the lap time it allows. Pure functions, no I/O.

Two pieces, both standard (docs/racing_line_research.md, sections 2.2 and 4):

1. `min_curvature_line`: the line as one lateral shift ``alpha`` per centreline
   station, bounded by the track edges less the car's half-width, chosen to minimise
   the squared curvature of the driven path -- a bounded linear least-squares
   problem, iterated because the point spacing changes as the line moves.
2. `gg_lap`: the limit lap along any closed path. Speed is capped by the cornering
   grip at every station, then swept forward under acceleration and backward under
   braking -- but, unlike `models.lap`, throttle/brake and cornering share ONE
   friction ellipse (``(a_x/a_x_max)^2 + (a_y/a_y_max)^2 <= 1``), which is the grip
   rule `RaceEnv` enforces, so the env and its reference agree.
"""

import math
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from numpy.typing import NDArray
from scipy.sparse.linalg import spsolve

from f1rl.config import GRAVITY_M_S2, LINE_CURVATURE_WINDOW_STATIONS, CarParams
from f1rl.envs.racetrack import RaceTrack, racetrack_from_xy

Array = NDArray[np.float64]


def min_curvature_line(
    track: RaceTrack,
    margin_m: float,
    iterations: int = 4,
    smoothness: float = 0.0,
    damping: float = 1.0,
    min_radius_frac: float = 0.6,
) -> Array:
    """Lateral shift alpha (m, positive = left) of the minimum-curvature line.

    Each station's shift is bounded to ``[-(w_right - margin), w_left - margin]``
    (``margin_m`` = the car's half-width plus any safety gap). The path curvature
    is approximated by the second difference of the path points over their local
    spacing; the spacing is re-measured from the previous iterate. ``smoothness`` > 0
    adds that weight on the curvature of the shift itself (``alpha''``), which keeps
    the line from kinking where both edges bind in a tight hairpin.

    ``min_radius_frac``: the line may not cut inside a corner to a radius below this
    fraction of the centreline's own. Left free, the optimiser hugs the inside of a
    tight hairpin (a 7 m centreline radius gave a 2 m line radius), which no car can
    follow and which the lap sim would price as a near-standstill.
    """
    n = track.n
    nx, ny = -np.sin(track.heading), np.cos(track.heading)
    lo = -(track.w_right - margin_m)
    hi = track.w_left - margin_m
    inside = (1.0 - min_radius_frac) / np.maximum(np.abs(track.kappa), 1e-9)
    hi = np.where(
        track.kappa > 0, np.minimum(hi, inside), hi
    )  # left bend: inside = left
    lo = np.where(
        track.kappa < 0, np.maximum(lo, -inside), lo
    )  # right bend: inside = right
    if np.any(lo > hi):
        raise ValueError("track narrower than the car at some station")

    ones = np.ones(n)
    d2 = sp.diags([ones, -2 * ones, ones], [-1, 0, 1], shape=(n, n), format="lil")
    d2[0, n - 1] = 1.0  # periodic: the lap closes
    d2[n - 1, 0] = 1.0
    d2 = d2.tocsr()

    alpha = np.zeros(n)
    for _ in range(iterations):
        px, py = track.x + alpha * nx, track.y + alpha * ny
        gap = np.hypot(np.roll(px, -1) - px, np.roll(py, -1) - py)
        h = (gap + np.roll(gap, 1)) / 2  # local point spacing of the current line
        w = sp.diags(1.0 / h**2)
        rows = [w @ d2 @ sp.diags(nx), w @ d2 @ sp.diags(ny)]
        rhs = [-(w @ (d2 @ track.x)), -(w @ (d2 @ track.y))]
        if smoothness:
            rows.append(smoothness * d2 / track.ds_m**2)
            rhs.append(np.zeros(n))
        a = sp.vstack(rows).tocsr()
        b = -np.concatenate([-r for r in rhs]) if False else np.concatenate(rhs)
        solved = _bounded_least_squares(a, b, lo, hi)
        alpha = alpha + damping * (solved - alpha)
    return alpha


def _bounded_least_squares(
    a: sp.csr_matrix, b: Array, lo: Array, hi: Array, max_rounds: int = 200
) -> Array:
    """argmin |a x - b|^2 subject to lo <= x <= hi, by an active-set method on the
    (banded) normal equations: solve the free variables, pin any that leave their
    bounds, release pinned ones the objective wants to move back inside. Orders of
    magnitude faster than a general bounded solver on these ~1000-variable bands.
    """
    h = (a.T @ a).tocsc()
    g = np.asarray(a.T @ b)
    n = len(g)
    x = np.clip(np.zeros(n), lo, hi)
    pinned = np.zeros(n, dtype=bool)
    for _ in range(max_rounds):
        free = ~pinned
        rhs = g[free] - h[free][:, pinned] @ x[pinned]
        x[free] = spsolve(h[free][:, free].tocsc(), rhs)
        over = (x > hi) | (x < lo)
        if over.any():
            x = np.clip(x, lo, hi)
            pinned |= over
            continue
        grad = h @ x - g  # >0 at the upper bound means "wants to go back in"
        release = pinned & (((x >= hi) & (grad > 1e-9)) | ((x <= lo) & (grad < -1e-9)))
        if not release.any():
            break
        pinned &= ~release
    return x


def line_xy(track: RaceTrack, alpha: Array) -> tuple[Array, Array]:
    """World (x, y) of the line ``alpha`` metres left of the centreline."""
    return (
        track.x - alpha * np.sin(track.heading),
        track.y + alpha * np.cos(track.heading),
    )


def path_geometry(x: Array, y: Array, step_m: float = 5.0) -> RaceTrack:
    """Spacing and curvature of any closed polyline, processed the same way as a
    centreline (so a line of ours and a line from elsewhere compare fairly)."""
    zeros = np.zeros(len(x))
    return racetrack_from_xy("path", x, y, zeros, zeros, step_m)


def _reach_v2(c: float, d: float, q: float) -> float:
    """Squared speed at the far end of an interval, on the friction ellipse.

    The longitudinal acceleration available falls as the lateral demand rises, and
    the lateral demand depends on the speed reached, so the end speed solves
    ``x = c + d * sqrt(1 - (x q)^2)`` with ``c`` the near-end speed squared,
    ``d = 2 * spacing * max_longitudinal_accel`` and ``q = |kappa| / a_y_max``.
    Squaring gives a quadratic whose positive root is returned in closed form.
    Evaluating the ellipse at the slower end instead lets a profile brake harder
    than the grip it is also using to corner.
    """
    q2 = q * q
    root = math.sqrt(max(0.0, 1.0 + q2 * (d * d - c * c)))
    return (c + d * root) / (1.0 + d * d * q2)


def gg_speed_profile(
    car: CarParams, ds: float | Array, kappa: Array, iterations: int = 6
) -> Array:
    """Limit speed (m/s) at every station of a closed path with spacing ``ds``
    (scalar or per station) and signed curvature ``kappa``, on one friction ellipse.
    """
    n = len(kappa)
    spacing = np.broadcast_to(np.asarray(ds, dtype=np.float64), (n,))
    ay_max = car.max_lateral_g * GRAVITY_M_S2
    ax_acc = car.max_accel_g * GRAVITY_M_S2
    ax_brk = car.max_braking_g * GRAVITY_M_S2
    k = np.abs(kappa)
    # Curvature over each interval: the mean of its two end stations.
    q = (k + np.roll(k, -1)) / 2 / ay_max
    with np.errstate(divide="ignore"):
        v_lim = np.minimum(car.max_speed_ms, np.sqrt(ay_max / np.maximum(k, 1e-9)))
    v = v_lim.copy()

    for _ in range(iterations):
        before = v.copy()
        for i in range(n):
            j = (i + 1) % n
            reach = _reach_v2(v[i] ** 2, 2 * spacing[i] * ax_acc, q[i])
            v[j] = min(v[j], math.sqrt(reach))
        for i in reversed(range(n)):
            j = (i + 1) % n
            reach = _reach_v2(v[j] ** 2, 2 * spacing[i] * ax_brk, q[i])
            v[i] = min(v[i], math.sqrt(reach))
        if np.max(np.abs(v - before)) < 1e-6:
            break
    return v


def gg_lap_time_s(car: CarParams, ds: float | Array, kappa: Array) -> float:
    """Flying-lap time (s) at the limit along a closed path (see `gg_speed_profile`)."""
    v = gg_speed_profile(car, ds, kappa)
    spacing = np.broadcast_to(np.asarray(ds, dtype=np.float64), (len(kappa),))
    return float(np.sum(2 * spacing / (v + np.roll(v, -1))))


@dataclass(frozen=True, eq=False)
class OptimalLine:
    """The minimum-curvature line on a track and what it allows."""

    alpha: Array  # m, lateral shift left of the centreline, per station
    x: Array
    y: Array
    kappa: Array  # path curvature at each station of the line
    ds: Array  # path spacing between station i and i+1
    speed: Array  # m/s, limit speed at each station
    lap_time_s: float
    centre_lap_time_s: float  # same physics, driving the centreline


def optimal_line(track: RaceTrack, car: CarParams, margin_m: float) -> OptimalLine:
    """Minimum-curvature line for ``track`` and its limit lap on the g-g ellipse.

    ``margin_m`` is how far the car's centre must stay from either edge.
    """
    alpha = min_curvature_line(track, margin_m)
    x, y = line_xy(track, alpha)
    # Curvature and spacing of the line itself, per centreline station: the line is
    # re-measured on the same stations so station i of the line is station i of the
    # track (the env and the reference share one index).
    gap = np.hypot(np.roll(x, -1) - x, np.roll(y, -1) - y)
    kappa = smooth_curvature(menger_curvature(x, y), track.ds_m)
    speed = gg_speed_profile(car, gap, kappa)
    lap = float(np.sum(2 * gap / (speed + np.roll(speed, -1))))
    centre = gg_lap_time_s(car, track.ds_m, track.kappa)
    return OptimalLine(alpha, x, y, kappa, gap, speed, lap, centre)


def menger_curvature(x: Array, y: Array) -> Array:
    """Signed curvature of the circle through each point and its two neighbours."""
    ax, ay = x - np.roll(x, 1), y - np.roll(y, 1)
    bx, by = np.roll(x, -1) - x, np.roll(y, -1) - y
    cross = ax * by - ay * bx
    la, lb = np.hypot(ax, ay), np.hypot(bx, by)
    lc = np.hypot(ax + bx, ay + by)
    return np.asarray(2 * cross / (la * lb * lc))


def smooth_curvature(kappa: Array, ds_m: float) -> Array:
    """Average ``kappa`` over `LINE_CURVATURE_WINDOW_STATIONS` stations (wrapping round
    the lap): the curvature the speed profile sees. Only enough to remove point-to-
    point noise in a polyline; a longer window (25 m) flattens real apex peaks by
    ~20% and makes the limit lap 1-2% faster than any car could actually follow."""
    del ds_m  # the window is in stations: the line is sampled on the track's grid
    return _smooth(kappa, LINE_CURVATURE_WINDOW_STATIONS)


def _smooth(values: Array, window: int) -> Array:
    kernel = np.ones(window) / window
    padded = np.pad(values, window // 2, mode="wrap")
    return np.convolve(padded, kernel, mode="valid")
