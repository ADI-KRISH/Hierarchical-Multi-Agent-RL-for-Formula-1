"""`DriverEnv`: a Gymnasium wrapper around the point-mass car/track physics.

The car has no steering degree of freedom -- `models/car.py` and `models/lap.py`
already pin it to the track's racing line, so control is purely longitudinal
(throttle/brake) each step.
"""

from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from f1rl.config import GRAVITY_M_S2, CarParams, DriverEnvParams, SimParams
from f1rl.envs.track import Track, curvature_at

ObsType = NDArray[np.float32]
ActType = NDArray[np.float32]


class DriverEnv(gym.Env[ObsType, ActType]):
    """One car lapping `track`, controlled by throttle/brake each step.

    Observation (Box, shape (5,)): [speed / max_speed, curvature now, curvature
    `lookahead_m` ahead, fraction of the lap completed, lateral offset / half
    track width] -- the curvature and offset terms are normalized and clipped to
    [-1, 1] / [0, 1].
    Action (Box, shape (1,)): throttle/brake in [-1, 1] (negative = braking).
    Reward: distance covered this step, normalized by lap length (captures both
    progress and speed); a fixed penalty and episode end if the car runs off the
    track edge.

    Track limits: the car has no steering, so lateral position isn't directly
    controlled -- it's a consequence of cornering too fast. Whenever the current
    speed demands more lateral g than the car's grip (`max_lateral_g`), the car
    drifts toward the outside of the corner at a rate proportional to the grip
    deficit; whenever it's within grip, the offset recovers back toward the
    racing line. Off-track is the offset exceeding half the track width.
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
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
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
        if excess_g > 0.0:
            self._lateral_offset_m += (
                excess_g * self.env_params.drift_gain_m_s_per_g * self.sim.dt_s
            )
        else:
            self._lateral_offset_m = max(
                0.0,
                self._lateral_offset_m
                - self.env_params.recovery_rate_m_s * self.sim.dt_s,
            )
        off_track = self._lateral_offset_m > self.track.width_m / 2

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
        offset_fraction = self._lateral_offset_m / (self.track.width_m / 2)
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
