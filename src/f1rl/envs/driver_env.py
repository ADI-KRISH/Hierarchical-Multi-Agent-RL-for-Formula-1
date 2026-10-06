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
from f1rl.models.lap import (
    distance_to_braking_point_m,
    max_safe_speed_ms,
    standing_start_lap_time_s,
)

ObsType = NDArray[np.float32]
ActType = NDArray[np.float32]


class DriverEnv(gym.Env[ObsType, ActType]):
    """One car lapping `track`, controlled by throttle/brake and steering.

    Observation (Box, shape (6 + len(lookahead_m),)), every term in [0, 1]
    except the signed braking margin:
    [speed / max_speed, corner speed limit here / max_speed, the highest speed
    from which every corner ahead can still be made (`max_safe_speed_ms`, the
    braking-marker boards) / max_speed, the braking margin (safe speed - speed)
    / `margin_obs_scale_ms` clipped to [-1, 1], the lowest corner
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
    scores higher -- minus `overspeed_penalty_per_s` per second spent past a
    braking point (scaled by how far over), and a fixed penalty plus episode end
    on running off track or stalling (below `stall_speed_ms` for
    `stall_timeout_s`: a car stopped on track is retired). Each action is held
    for `action_repeat` physics steps; the step's reward is their sum.

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
    `stalled`, `lap_completed`), and on completing the lap, `lap_time_s` interpolated to
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

        n_obs = 6 + len(self.env_params.lookahead_m)
        if self.env_params.brake_point_horizon_m:
            n_obs += 1
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        low = np.zeros(n_obs, dtype=np.float32)
        low[3] = -1.0  # braking margin is signed
        if self.env_params.brake_point_horizon_m:
            low[-1] = -1.0  # so is the braking-point countdown (last term)
        self.observation_space = gym.spaces.Box(
            low=low, high=1.0, shape=(n_obs,), dtype=np.float32
        )

        self._time_penalty_per_s = self.env_params.time_penalty_per_s
        if self.env_params.time_penalty_per_limit_lap:
            limit_s = standing_start_lap_time_s(car, track)
            self._time_penalty_per_s += (
                self.env_params.time_penalty_per_limit_lap / limit_s
            )
        self._seeded = False
        self._speed_ms = 0.0
        self._distance_m = 0.0
        self._lateral_offset_m = 0.0
        self._step_count = 0
        self._stalled_s = 0.0

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
        self._stalled_s = 0.0
        info = self._info(off_track=False, lap_completed=False)
        info["stalled"] = False
        return self._observation(), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        command = float(np.clip(action[0], -1.0, 1.0))
        steer = float(np.clip(action[1], -1.0, 1.0))
        total = 0.0
        for _ in range(self.env_params.action_repeat):
            reward, terminated, truncated, info = self._physics_step(command, steer)
            total += reward
            if terminated or truncated:
                break
        return self._observation(), total, terminated, truncated, info

    def _physics_step(
        self, command: float, steer: float
    ) -> tuple[float, bool, bool, dict[str, Any]]:
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
        if self._speed_ms < self.env_params.stall_speed_ms:
            self._stalled_s += dt
        else:
            self._stalled_s = 0.0
        stalled = self._stalled_s >= self.env_params.stall_timeout_s

        lap_length_m = self.track.total_length_m
        reward = distance_step / lap_length_m - self._time_penalty_per_s * dt
        safe_ms = max_safe_speed_ms(self.car, self.track, self._distance_m)
        overspeed = max(0.0, self._speed_ms - safe_ms) / self.car.max_speed_ms
        reward -= self.env_params.overspeed_penalty_per_s * overspeed * dt
        if self.env_params.speed_use_reward_per_s:
            speed_use = min(self._speed_ms, safe_ms) / safe_ms
            reward += self.env_params.speed_use_reward_per_s * speed_use * dt
        if off_track or stalled:
            reward -= self.env_params.off_track_penalty

        lap_completed = self._distance_m >= lap_length_m
        terminated = off_track or stalled or lap_completed
        truncated = self._step_count >= self.sim.max_episode_steps

        info = self._info(off_track=off_track, lap_completed=lap_completed)
        info["stalled"] = stalled
        if lap_completed:
            # Interpolate within the final step to when the car crossed the line.
            fraction = (lap_length_m - distance_before) / distance_step
            info["lap_time_s"] = (self._step_count - 1 + fraction) * dt
        return reward, terminated, truncated, info

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
        safe_ms = max_safe_speed_ms(self.car, self.track, position_m)
        width_here = width_at(self.track, position_m)
        offset_fraction = self._lateral_offset_m / (width_here / 2)
        obs = np.array(
            [
                self._speed_ms / self.car.max_speed_ms,
                self._speed_limit_fraction(curvature_at(self.track, position_m)),
                safe_ms / self.car.max_speed_ms,
                np.clip(
                    (safe_ms - self._speed_ms) / self.env_params.margin_obs_scale_ms,
                    -1.0,
                    1.0,
                ),
                *self._lookahead_limits(position_m),
                position_m / self.track.total_length_m,
                min(offset_fraction, 1.0),
            ],
            dtype=np.float32,
        )
        horizon_m = self.env_params.brake_point_horizon_m
        if horizon_m:
            room_m = distance_to_braking_point_m(
                self.car, self.track, position_m, self._speed_ms, horizon_m
            )
            countdown = np.float32(np.clip(room_m / horizon_m, -1.0, 1.0))
            obs = np.append(obs, countdown)
        return obs
