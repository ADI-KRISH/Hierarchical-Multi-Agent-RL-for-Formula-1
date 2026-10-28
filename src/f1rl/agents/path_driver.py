"""A scripted driver that follows the racing line shifted by a `Maneuver`.

`LineFollower` only knows the racing line; asked to run a metre or two off it, its
speeds are wrong for the new, tighter or wider corner and it runs out of grip. Here
the shifted path is built first (world coordinates, curvature, heading), the corner
speeds and braking points are planned on THAT path over the next ~250 m, and the same
feed-forward + PD tracker follows it. Reads exact state from ``info`` like the other
scripted drivers.
"""

import math

import numpy as np
from numpy.typing import NDArray

from f1rl.agents.line_follower import ActType, LineFollower
from f1rl.config import GRAVITY_M_S2, CarParams, FollowerParams
from f1rl.envs.racetrack import RaceTrack
from f1rl.models.maneuver import Maneuver, shift_at
from f1rl.models.raceline import OptimalLine, _reach_v2, menger_curvature

Array = NDArray[np.float64]

#: Stations behind the car kept in the plan (for centred curvature/heading).
BEHIND = 2
#: Off the racing line the car tracks its path less tightly (the path is bent and
#: the line no longer tells it the speed), so corner speeds are planned for this
#: much less lateral grip per metre of shift.
SHIFT_DERATE_PER_M = 0.02


class PathDriver:
    def __init__(
        self,
        car: CarParams,
        track: RaceTrack,
        line: OptimalLine,
        room_left: Array,
        room_right: Array,
        params: FollowerParams | None = None,
        horizon_m: float = 250.0,
        edge_inset_m: float = 0.2,
    ) -> None:
        self.car = car
        self.track = track
        self.line = line
        self.params = params or FollowerParams()
        self.base = LineFollower(car, track, line, self.params)
        self.room_l = room_left - edge_inset_m
        self.room_r = room_right - edge_inset_m
        self._h = round(horizon_m / track.ds_m)
        f = self.params.grip_fraction
        self._ay = car.max_lateral_g * GRAVITY_M_S2 * f
        self._ax_brk = (
            car.max_braking_g * GRAVITY_M_S2 * (self.params.braking_fraction or f)
        )

    def speed_at(self, s_m: float) -> float:
        """The racing line's target speed (m/s) at distance ``s_m``."""
        return self.base.speed_at(s_m)

    def line_speed(self) -> Array:
        """The racing line's target speed (m/s) at every station."""
        return self.base._speed

    def line_kappa(self) -> Array:
        """The racing line's curvature (1/m) at every station."""
        return self.base._line_kappa

    def plan(
        self,
        s_m: float,
        m: Maneuver,
        brake_fraction: float | None = None,
        extra_speed_ms: float = 0.0,
    ) -> tuple[Array, Array, Array, Array, Array]:
        """The path over the horizon: (unwrapped s, lateral offset n, heading
        relative to the track, curvature, target speed) per station."""
        track, n_st = self.track, self.track.n
        i0 = int(s_m / track.ds_m) - BEHIND
        k = np.arange(i0, i0 + self._h + BEHIND + 2)
        s = (k * track.ds_m).astype(np.float64)
        idx = k % n_st
        alpha = self.line.alpha[idx]
        # Keep inside the track WITHOUT kinking the path: clip the shift to the room
        # left beside the line, smooth it, and clip again (a path pinned to a
        # wandering edge is a slalom the car cannot drive at speed).
        up, down = self.room_l[idx] - alpha, -self.room_r[idx] - alpha
        sh = np.clip(shift_at(m, s), down, up)
        sh = np.clip(
            np.convolve(np.pad(sh, 4, mode="edge"), np.ones(9) / 9, "valid"), down, up
        )
        n = alpha + sh
        theta = track.heading[idx]
        x = track.x[idx] - n * np.sin(theta)
        y = track.y[idx] + n * np.cos(theta)
        # Centred differences: heading and curvature of the path itself.
        head = np.arctan2(y[2:] - y[:-2], x[2:] - x[:-2])
        psi = np.angle(np.exp(1j * (head - theta[1:-1])))
        kap = menger_curvature(x[1:-1], y[1:-1])
        kap[0], kap[-1] = kap[1], kap[-2]  # the window's ends wrapped round
        seg = np.hypot(np.diff(x[1:-1]), np.diff(y[1:-1]))
        shift = np.abs(n[1:-1] - self.line.alpha[idx[1:-1]])
        v = self._speed_plan(
            kap,
            seg,
            idx[1:-1].astype(np.float64),
            brake_fraction,
            shift,
            extra_speed_ms,
        )
        return s[1:-1], n[1:-1], np.asarray(psi, dtype=np.float64), kap, v

    def _speed_plan(
        self,
        kap: Array,
        seg: Array,
        idx: Array,
        brake_fraction: float | None,
        shift: Array,
        extra_speed_ms: float,
    ) -> Array:
        ay = self._ay * (1.0 - SHIFT_DERATE_PER_M * shift)
        v = np.minimum(
            self.car.max_speed_ms + extra_speed_ms,
            np.sqrt(ay / np.maximum(np.abs(kap), 1e-9)),
        )
        # Past the horizon the path is back on the line: end at its (derated) speed.
        v[-1] = min(v[-1], self.base.speed_at(idx[-1] * self.track.ds_m))
        brake = self._ax_brk
        if brake_fraction is not None:
            brake = self.car.max_braking_g * GRAVITY_M_S2 * brake_fraction
        q = (np.abs(kap[:-1]) + np.abs(kap[1:])) / 2 / ay[:-1]
        for i in range(len(v) - 2, -1, -1):
            reach = _reach_v2(v[i + 1] ** 2, 2.0 * seg[i] * brake, q[i])
            v[i] = min(v[i], math.sqrt(reach))
        return v

    def act(
        self, info: dict[str, float], m: Maneuver, extra_speed_ms: float = 0.0
    ) -> ActType:
        p, track = self.params, self.track
        s, n, psi, v = (
            info["distance_m"],
            info["lateral_offset_m"],
            info["heading_rad"],
            info["speed_ms"],
        )
        ps, pn, ppsi, pk, pv = self.plan(s, m, m.brake_fraction, extra_speed_ms)
        ahead_m = 0.5 * p.decision_s * v
        cross_track = n - float(np.interp(s, ps, pn))
        heading_error = _wrap(psi - float(np.interp(s, ps, ppsi)))
        w = p.bandwidth_rad_s
        a_y = (
            float(np.interp(s + ahead_m, ps, pk)) * v * v
            - w * w * cross_track
            - 2.0 * p.damping * w * v * heading_error
        )
        u_lat = float(np.clip(a_y / (self.car.max_lateral_g * GRAVITY_M_S2), -1, 1))
        v_here = float(np.interp(s, ps, pv))
        v_mid = float(np.interp(s + ahead_m, ps, pv))
        v_next = float(np.interp(s + ahead_m + track.ds_m, ps, pv))
        a_ref = (v_next**2 - v_mid**2) / (2.0 * track.ds_m)
        a_want = a_ref + (v_here - v) / p.speed_time_constant_s
        peak = self.car.max_accel_g if a_want >= 0 else self.car.max_braking_g
        u_long = a_want / (peak * GRAVITY_M_S2)
        room = math.sqrt(max(0.0, 1.0 - u_lat * u_lat))
        u_long = float(np.clip(u_long, -room, room))
        return np.array([u_long, u_lat], dtype=np.float32)


def _wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))
