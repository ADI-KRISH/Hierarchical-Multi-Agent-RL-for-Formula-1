"""A small synthetic closed track for offline tests: a 14 m wide stadium."""

import numpy as np

from f1rl.envs.racetrack import RaceTrack, racetrack_from_xy

STRADDLE_RADIUS_M = 100.0
STRAIGHT_M = 600.0


def stadium_track(width_m: float = 14.0, step_m: float = 5.0) -> RaceTrack:
    """Two 600 m straights joined by two 100 m-radius semicircles (~1.8 km),
    driven counter-clockwise (left-handers)."""
    r, length = STRADDLE_RADIUS_M, STRAIGHT_M
    n_arc = round(np.pi * r / step_m)
    n_line = round(length / step_m)
    a = np.linspace(-np.pi / 2, np.pi / 2, n_arc, endpoint=False)
    b = np.linspace(np.pi / 2, 3 * np.pi / 2, n_arc, endpoint=False)
    x = np.concatenate(
        [
            np.linspace(0, length, n_line, endpoint=False),
            length + r * np.cos(a),
            np.linspace(length, 0, n_line, endpoint=False),
            r * np.cos(b),
        ]
    )
    y = np.concatenate(
        [
            np.full(n_line, -r),
            r * np.sin(a),
            np.full(n_line, r),
            r * np.sin(b),
        ]
    )
    half = np.full(len(x), width_m / 2)
    return racetrack_from_xy("stadium", x, y, half, half, step_m, window_m=0.0)
