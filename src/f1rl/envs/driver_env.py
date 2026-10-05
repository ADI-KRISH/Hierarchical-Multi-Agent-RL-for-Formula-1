"""`DriverEnv`: a Gymnasium wrapper around the point-mass car/track physics.

Longitudinal control (throttle/brake) plus a steering input that spends
whatever lateral grip the current corner isn't already using -- there is no
full 2D heading/position model, steering only moves the car's offset from the
racing line within the grip budget `models/car.py` already defines.
"""

import itertools
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from f1rl.config import GRAVITY_M_S2, CarParams, DriverEnvParams, SimParams
from f1rl.envs.track import (
    Track,
    curvature_at,
    max_abs_curvature_between,
    width_at,
)
from f1rl.models.car import max_corner_speed_ms

ObsType = NDArray[np.float32]
ActType = NDArray[np.float32]


class DriverEnv(gym.Env[ObsType, ActType]):
    """One car lapping `track`, controlled by throttle/brake and steering.

    Observation (Box, shape (4 + len(lookahead_m),)), every term in [0, 1]:
    [speed / max_speed, corner speed limit here / max_speed, the lowest corner
    speed limit in each lookahead window / max_speed, fraction of the lap
    completed, lateral offset / half track width]. The windows run between
    consecutive `lookahead_m` distances (0-25 m, 25-50 m, ... by default), and
    report the slowest point anywhere inside, so a short hairpin can't slip
    between two samples. Speed limits come from `max_corner_speed_ms`, so "how
    slow must I be there" is read off directly, not inferred from curvature.
    Action (Box, shape (2,)): [throttle/brake, steer], each in [-1, 1] (negative
    throttle = braking; negative steer = toward the racing line).
    Reward: distance covered this step as a fraction of the lap (a full lap pays
    1.0), minus `time_penalty_per_s` per simulated second -- so a faster lap
    scores higher -- and a fixed penalty plus episode end on running off track.

    Track limits and steering: cornering consumes lateral grip
    (`speed^2 * curvature / g`); whatever exceeds the car's `max_lateral_g` is a
    forced drift toward the outside edge the driver can't steer out of.
    Whatever grip the corner leaves unused lets the steer action turn the car's
    velocity up to `max_heading_ratio` off the track direction, moving the offset
    back toward (or away from) the racing line. Lateral speed therefore scales
    with forward speed: a stopped car can't move sideways. The offset is
    measured outward from the racing line and floors at 0 -- the model has no
    inside edge, since grip only ever pushes a car wide. Off-track is the offset
    exceeding half the current segment's width.

    `info` carries the car state each step (`distance_m`, `speed_ms`,
    `lateral_offset_m`, `elapsed_s`), how the episode ended (`off_track`,
    `lap_completed`), and on completing the lap, `lap_time_s` interpolated to
    the moment the car crossed the line.
    """

    def __init__(
        self,
        car: CarParams,
        track: Track,
        sim: SimParams,
        env_params: DriverEnvParams | None = None,
    ) -> None:
        self.car = car
        self.track = track
        self.sim = sim
        self.env_params = env_params or DriverEnvParams()

        n_obs = 4 + len(self.env_params.lookahead_m)
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(n_obs,), dtype=np.float32
        )

        self._seeded = False
        self._speed_ms = 0.0
        self._distance_m = 0.0
        self._lateral_offset_m = 0.0
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        # Seed from `sim.seed` on the first reset only, so later unseeded resets
        # continue the same RNG stream (each episode differs, all reproducible)
        # instead of replaying episode one forever.
        if seed is None and not self._seeded:
            seed = self.sim.seed
        super().reset(seed=seed)
        self._seeded = True

        self._speed_ms = 0.0
        self._distance_m = 0.0
        self._lateral_offset_m = float(
            self.np_random.uniform(0.0, self.env_params.start_offset_max_m)
        )
        self._step_count = 0
        return self._observation(), self._info(off_track=False, lap_completed=False)

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        command = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        dt = self.sim.dt_s

        peak_g = self.car.max_accel_g if command >= 0 else self.car.max_braking_g
        accel_g = command * peak_g
        new_speed = float(
            np.clip(
                self._speed_ms + accel_g * GRAVITY_M_S2 * dt,
                0.0,
                self.car.max_speed_ms,
            )
        )
        distance_step = (self._speed_ms + new_speed) / 2 * dt
        distance_before = self._distance_m
        self._speed_ms = new_speed
        self._distance_m += distance_step
        self._step_count += 1

        curvature_here = curvature_at(self.track, self._distance_m)
        lateral_g_demand = self._speed_ms**2 * abs(curvature_here) / GRAVITY_M_S2
        excess_g = max(0.0, lateral_g_demand - self.car.max_lateral_g)
        unused_grip_fraction = max(0.0, 1.0 - lateral_g_demand / self.car.max_lateral_g)

        forced_drift_m_s = excess_g * self.env_params.drift_gain_m_s_per_g
        steer_m_s = (
            steer
            * unused_grip_fraction
            * self._speed_ms
            * self.env_params.max_heading_ratio
        )
        self._lateral_offset_m = max(
            0.0, self._lateral_offset_m + (forced_drift_m_s + steer_m_s) * dt
        )

        width_here = width_at(self.track, self._distance_m)
        off_track = self._lateral_offset_m > width_here / 2

        lap_length_m = self.track.total_length_m
        reward = distance_step / lap_length_m - self.env_params.time_penalty_per_s * dt
        if off_track:
            reward -= self.env_params.off_track_penalty

        lap_completed = self._distance_m >= lap_length_m
        terminated = off_track or lap_completed
        truncated = self._step_count >= self.sim.max_episode_steps

        info = self._info(off_track=off_track, lap_completed=lap_completed)
        if lap_completed:
            # Interpolate within the final step to when the car crossed the line.
            fraction = (lap_length_m - distance_before) / distance_step
            info["lap_time_s"] = (self._step_count - 1 + fraction) * dt
        return self._observation(), reward, terminated, truncated, info

    def _info(self, *, off_track: bool, lap_completed: bool) -> dict[str, Any]:
        return {
            "distance_m": self._distance_m,
            "speed_ms": self._speed_ms,
            "lateral_offset_m": self._lateral_offset_m,
            "elapsed_s": self._step_count * self.sim.dt_s,
            "off_track": off_track,
            "lap_completed": lap_completed,
        }

    def _speed_limit_fraction(self, curvature_per_m: float) -> float:
        return max_corner_speed_ms(self.car, curvature_per_m) / self.car.max_speed_ms

    def _lookahead_limits(self, position_m: float) -> list[float]:
        edges = (0.0, *self.env_params.lookahead_m)
        return [
            self._speed_limit_fraction(
                max_abs_curvature_between(
                    self.track, position_m + near_m, position_m + far_m
                )
            )
            for near_m, far_m in itertools.pairwise(edges)
        ]

    def _observation(self) -> ObsType:
        position_m = self._distance_m % self.track.total_length_m
        width_here = width_at(self.track, position_m)
        offset_fraction = self._lateral_offset_m / (width_here / 2)
        return np.array(
            [
                self._speed_ms / self.car.max_speed_ms,
                self._speed_limit_fraction(curvature_at(self.track, position_m)),
                *self._lookahead_limits(position_m),
                position_m / self.track.total_length_m,
                min(offset_fraction, 1.0),
            ],
            dtype=np.float32,
        )
