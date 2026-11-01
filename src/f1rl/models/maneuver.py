"""A manoeuvre is a plan for where the car sits across the track: a smooth lateral
shift from the racing line as a function of distance, plus how hard it brakes.

Pure functions. ``Maneuver.points`` are ``(metres after start_s, shift left of the
racing line)`` knots, joined by cosine easing and held at the last value after the
final knot. Overtaking moves (dive-bomb, around the outside, switchback, dummy) and
defending moves (cover, inside line) are just different knot lists
(`agents.tactics`).
"""

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class Maneuver:
    start_s: float
    points: tuple[tuple[float, float], ...]
    # Braking is planned at this fraction of the brake limit while the manoeuvre runs
    # (None = the driver's normal). A dive-bomb brakes later: a higher fraction.
    brake_fraction: float | None = None

    @property
    def end_s(self) -> float:
        return self.start_s + self.points[-1][0]


def shift_at(m: Maneuver, s_m: Array) -> Array:
    """Shift (m, left positive) at the unwrapped distances ``s_m``."""
    xs = np.array([p[0] for p in m.points], dtype=np.float64)
    ys = np.array([p[1] for p in m.points], dtype=np.float64)
    d = np.asarray(s_m, dtype=np.float64) - m.start_s
    if len(xs) == 1:
        return np.full_like(d, ys[0])
    k = np.clip(np.searchsorted(xs, d, side="right") - 1, 0, len(xs) - 2)
    span = np.maximum(xs[k + 1] - xs[k], 1e-9)
    t = np.clip((d - xs[k]) / span, 0.0, 1.0)
    ease = 0.5 * (1.0 - np.cos(math.pi * t))
    return np.asarray(ys[k] + (ys[k + 1] - ys[k]) * ease)


def hold(shift_m: float) -> Maneuver:
    """No manoeuvre: stay ``shift_m`` off the racing line."""
    return Maneuver(0.0, ((0.0, shift_m),))
