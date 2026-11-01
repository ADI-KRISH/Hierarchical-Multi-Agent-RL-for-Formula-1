"""A scripted driver that follows an `OptimalLine` -- the check that env v2 can
actually be driven at the reference pace, and a ready comparison for the RL agent.

Steering is the line's own curvature (feed-forward) plus a Stanley-style correction
on the heading and cross-track error; speed is the line's limit-speed profile
(feed-forward acceleration) plus a proportional correction. Like the v1 baseline it
reads exact state from `info`, which is more than the RL policy observes.
"""

import math
from dataclasses import replace

import numpy as np
from numpy.typing import NDArray

from f1rl.config import GRAVITY_M_S2, CarParams, FollowerParams
from f1rl.envs.racetrack import RaceTrack
from f1rl.models.raceline import (
    OptimalLine,
    gg_speed_profile,
    menger_curvature,
)

Array = NDArray[np.float64]
ActType = NDArray[np.float32]


class LineFollower:
    def __init__(
        self,
        car: CarParams,
        track: RaceTrack,
        line: OptimalLine,
        params: FollowerParams | None = None,
    ) -> None:
        self.car = car
        self.track = track
        self.line = line
        self.params = params or FollowerParams()
        # Direction of travel of the line itself at each station, centred like the
        # track's heading (a forward chord would be half a station out of step: a
        # heading bias of ~0.03 rad in a tight corner).
        self._line_heading = np.arctan2(
            np.roll(line.y, -1) - np.roll(line.y, 1),
            np.roll(line.x, -1) - np.roll(line.x, 1),
        )
        # The line's own turn rate, unsmoothed and centred on each station like the
        # env's track curvature: steering it exactly reproduces the line. The
        # smoothed `line.kappa` the speed profile uses flattens apex peaks.
        self._line_kappa = menger_curvature(line.x, line.y)
        # Target speeds from a car with `grip_fraction` of the real grip on every
        # axis. The reference profile uses 100%, leaving nothing to steer back onto
        # the line with: a correction then steals grip from braking and the car
        # can't shed speed in a corner. A speed scale doesn't fix that; reserve does.
        f = self.params.grip_fraction
        derated = replace(
            car,
            max_lateral_g=car.max_lateral_g * f,
            max_accel_g=car.max_accel_g * (self.params.accel_fraction or f),
            max_braking_g=car.max_braking_g * (self.params.braking_fraction or f),
        )
        # From the unsmoothed curvature, the same the steering follows: at a sharp
        # kink the 3-station average the reference uses is a third lower than what
        # the car must actually turn, leaving no grip to brake with there.
        self._speed = gg_speed_profile(derated, line.ds, self._line_kappa)

    def _at(self, values: Array, s_m: float, angle: bool = False) -> float:
        """``values`` at distance ``s_m``: linear between stations, periodic.
        ``angle`` interpolates through the +-pi seam."""
        track = self.track
        pos = (s_m % track.length_m) / track.ds_m
        i = int(pos)
        f = pos - i
        a, b = float(values[i % track.n]), float(values[(i + 1) % track.n])
        if angle:
            b = a + _wrap(b - a)
        return a * (1.0 - f) + b * f

    def speed_at(self, s_m: float) -> float:
        """The target speed (m/s) at distance ``s_m``."""
        return self._at(self._speed, s_m)

    def max_curvature(self, s_m: float, ahead_m: float) -> float:
        """The tightest the line turns (1/m) in the next ``ahead_m`` metres."""
        i = int((s_m % self.track.length_m) / self.track.ds_m)
        k = max(1, round(ahead_m / self.track.ds_m))
        idx = (i + np.arange(k)) % self.track.n
        return float(np.max(np.abs(self._line_kappa[idx])))

    def line_state(self, s_m: float) -> tuple[float, float, float]:
        """(lateral offset, heading relative to the track, target speed) of a car
        sitting exactly on the line at ``s_m``."""
        psi = _wrap(
            self._at(self._line_heading, s_m, angle=True)
            - self._at(self.track.heading, s_m, angle=True)
        )
        return self._at(self.line.alpha, s_m), psi, self.speed_at(s_m)

    def act(self, info: dict[str, float], shift_m: float = 0.0) -> ActType:
        p, track = self.params, self.track
        s, n, psi, v = (
            info["distance_m"],
            info["lateral_offset_m"],
            info["heading_rad"],
            info["speed_ms"],
        )
        cross_track = (
            n - self._at(self.line.alpha, s) - shift_m
        )  # shift: aim off the line
        heading_error = _wrap(
            self._at(track.heading, s, angle=True)
            + psi
            - self._at(self._line_heading, s, angle=True)
        )
        # Feed forward the line's own turn rate (at the middle of the hold: the action
        # stays fixed for a whole decision, over which a corner's curvature changes),
        # and add a critically damped PD correction on the lateral and heading
        # error, whose bandwidth does not shrink with speed the way Stanley's does.
        ahead_m = 0.5 * p.decision_s * v
        w = p.bandwidth_rad_s
        # A path ``shift_m`` to the left of the line turns with curvature k/(1 - d k):
        # tighter on the inside of a bend, easier on the outside.
        k_mid = self._at(self._line_kappa, s + ahead_m)
        a_y = (
            k_mid / max(0.25, 1.0 - shift_m * k_mid) * v * v
            - w * w * cross_track
            - 2.0 * p.damping * w * v * heading_error
        )
        u_lat = float(np.clip(a_y / (self.car.max_lateral_g * GRAVITY_M_S2), -1, 1))

        # Feed forward the profile's own accel/braking, taken mid-hold like the
        # steering (braking that starts up to a whole decision late can't be made
        # up), then correct the speed error: a pure P controller tracking a braking
        # ramp lags it by decel * tau.
        i = int(((s + ahead_m) % track.length_m) / track.ds_m) % track.n

        # On the shifted path the corner speeds scale with the root of the radius: lower
        # on the inside; never raised on the outside.
        def cap(k: float) -> float:
            return min(1.0, math.sqrt(max(0.25, 1.0 - shift_m * k)))

        v_k = float(self._speed[(i + 1) % track.n]) * cap(
            float(self._line_kappa[(i + 1) % track.n])
        )
        v_0 = float(self._speed[i]) * cap(float(self._line_kappa[i]))
        a_ref = (v_k * v_k - v_0 * v_0) / (2.0 * float(self.line.ds[i]))
        v_i = self._at(self._speed, s) * cap(self._at(self._line_kappa, s))
        a_want = a_ref + (v_i - v) / p.speed_time_constant_s
        peak = self.car.max_accel_g if a_want >= 0 else self.car.max_braking_g
        u_long = a_want / (peak * GRAVITY_M_S2)
        room = math.sqrt(max(0.0, 1.0 - u_lat * u_lat))  # steering gets grip first
        u_long = float(np.clip(u_long, -room, room))
        return np.array([u_long, u_lat], dtype=np.float32)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
