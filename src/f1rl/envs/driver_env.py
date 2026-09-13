"""`DriverEnv`: a Gymnasium wrapper around the point-mass car/track physics.

Longitudinal control (throttle/brake) plus a steering input that spends
whatever lateral grip the current corner isn't already using -- there is no
full 2D heading/position model, steering only moves the car's offset from the
racing line within the grip budget `models/car.py` already defines.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from f1rl.config import GRAVITY_M_S2, CarParams, DriverEnvParams, SimParams
from f1rl.envs.track import Track, curvature_at, width_at

ObsType = NDArray[np.float32]
ActType = NDArray[np.float32]


class DriverEnv(gym.Env[ObsType, ActType]):
    """One car lapping `track`, controlled by throttle/brake and steering.

    Observation (Box, shape (5,)): [speed / max_speed, curvature now, curvature
    `lookahead_m` ahead, fraction of the lap completed, lateral offset / half
    track width] -- the curvature and offset terms are normalized and clipped to
    [-1, 1] / [0, 1].
    Action (Box, shape (2,)): [throttle/brake, steer], each in [-1, 1] (negative
    throttle = braking; negative steer = toward the racing line).
    Reward: distance covered this step, normalized by lap length (captures both
    progress and speed); a fixed penalty and episode end if the car runs off the
    track edge.

    Track limits and steering: cornering consumes lateral grip
    (`speed^2 * curvature / g`); whatever exceeds the car's `max_lateral_g` is a
    forced drift toward the edge the driver can't steer out of. Whatever grip is
    *not* consumed by the corner is a budget the steer action spends to move the
    offset back toward (or away from) the racing line -- so steering only works
    within the grip physics already allows. Off-track is the offset exceeding
    half the current segment's width.
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

        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, -1.0, -1.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32,
        )

        self._speed_ms = 0.0
        self._distance_m = 0.0
        self._lateral_offset_m = 0.0
        self._step_count = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed if seed is not None else self.sim.seed)
        self._speed_ms = 0.0
        self._distance_m = 0.0
        self._lateral_offset_m = 0.0
        self._step_count = 0
        return self._observation(), {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        command = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))

        peak_g = self.car.max_accel_g if command >= 0 else self.car.max_braking_g
        accel_g = command * peak_g
        new_speed = float(
            np.clip(
                self._speed_ms + accel_g * GRAVITY_M_S2 * self.sim.dt_s,
                0.0,
                self.car.max_speed_ms,
            )
        )
        distance_step = (self._speed_ms + new_speed) / 2 * self.sim.dt_s
        self._speed_ms = new_speed
        self._distance_m += distance_step
        self._step_count += 1

        curvature_here = curvature_at(self.track, self._distance_m)
        lateral_g_demand = self._speed_ms**2 * abs(curvature_here) / GRAVITY_M_S2
        excess_g = max(0.0, lateral_g_demand - self.car.max_lateral_g)
        remaining_budget_g = max(0.0, self.car.max_lateral_g - lateral_g_demand)

        forced_drift_m_s = excess_g * self.env_params.drift_gain_m_s_per_g
        steer_m_s = steer * remaining_budget_g * self.env_params.steer_gain_m_s_per_g
        self._lateral_offset_m = max(
            0.0, self._lateral_offset_m + (forced_drift_m_s + steer_m_s) * self.sim.dt_s
        )

        width_here = width_at(self.track, self._distance_m)
        off_track = self._lateral_offset_m > width_here / 2

        reward = distance_step / self.track.total_length_m
        if off_track:
            reward -= self.env_params.off_track_penalty

        lap_completed = self._distance_m >= self.track.total_length_m
        terminated = off_track or lap_completed
        truncated = self._step_count >= self.sim.max_episode_steps

        return self._observation(), reward, terminated, truncated, {}

    def _observation(self) -> ObsType:
        position_m = self._distance_m % self.track.total_length_m
        curvature_now = curvature_at(self.track, position_m)
        lookahead_m = position_m + self.env_params.lookahead_m
        curvature_ahead = curvature_at(self.track, lookahead_m)
        norm = self.env_params.curvature_norm_per_m
        width_here = width_at(self.track, position_m)
        offset_fraction = self._lateral_offset_m / (width_here / 2)
        return np.array(
            [
                self._speed_ms / self.car.max_speed_ms,
                np.clip(curvature_now / norm, -1.0, 1.0),
                np.clip(curvature_ahead / norm, -1.0, 1.0),
                position_m / self.track.total_length_m,
                np.clip(offset_fraction, 0.0, 1.0),
            ],
            dtype=np.float32,
        )
