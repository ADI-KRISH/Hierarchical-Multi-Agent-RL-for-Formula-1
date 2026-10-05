import math
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1rl.config import EXAMPLE_CAR, GRAVITY_M_S2, DriverEnvParams, SimParams
from f1rl.envs.driver_env import DriverEnv
from f1rl.envs.track import Segment, Track, example_track, technical_track
from f1rl.models.car import max_corner_speed_ms

FULL_THROTTLE_NO_STEER = np.array([1.0, 0.0], dtype=np.float32)


def _env() -> DriverEnv:
    return DriverEnv(EXAMPLE_CAR, example_track(), SimParams(seed=0))


def test_passes_gymnasium_env_checker() -> None:
    check_env(_env(), skip_render_check=True)


def test_random_agent_completes_an_episode_without_crashing() -> None:
    env = _env()
    obs, _info = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(env.sim.max_episode_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    else:
        raise AssertionError("episode never terminated or truncated")


def test_full_throttle_goes_off_track_at_the_corner() -> None:
    """A naive full-throttle policy reaches the example track's straight-line top
    speed well before its corner's much lower grip limit -- it must be flagged
    off-track there, not sail through at an unphysical speed.
    """
    env = _env()
    env.reset(seed=0)
    terminated = truncated = False
    final_reward = 0.0
    while not (terminated or truncated):
        _, final_reward, terminated, truncated, _ = env.step(FULL_THROTTLE_NO_STEER)

    assert terminated is True
    assert truncated is False
    assert final_reward < 0.0


def _drive(env: DriverEnv, action: np.ndarray) -> tuple[float, dict[str, Any]]:
    """Hold `action` until the episode ends; return (total reward, final info)."""
    total = 0.0
    terminated = truncated = False
    info: dict[str, Any] = {}
    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(action)
        total += reward
    return total, info


def test_zero_curvature_never_drifts_off_track() -> None:
    """A pure straight demands no lateral g at any speed, so even full throttle
    the whole way round must leave the lateral offset at 0.
    """
    straight_loop = Track(segments=(Segment(length_m=1000.0, curvature_per_m=0.0),))
    no_start_offset = DriverEnvParams(start_offset_max_m=0.0)
    env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0), no_start_offset)
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        _, _reward, terminated, truncated, info = env.step(FULL_THROTTLE_NO_STEER)
        assert info["lateral_offset_m"] == 0.0

    assert info["lap_completed"] is True  # ended by finishing, not by going off


def test_steering_recovers_offset_using_leftover_grip_budget() -> None:
    """Full inward steering should shrink an existing offset when the corner
    isn't using up all the car's lateral grip -- the leftover budget is what
    steering draws on to get back to the racing line.
    """
    straight_loop = Track(segments=(Segment(length_m=10_000.0, curvature_per_m=0.0),))
    env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0))
    _obs, info = env.reset(seed=0)
    before = info["lateral_offset_m"]
    assert before > 0.0  # default start offset is drawn from (0, 1] m

    for _ in range(50):  # get rolling: steering needs forward speed
        env.step(np.array([1.0, 0.0], dtype=np.float32))
    for _ in range(50):
        _, _reward, _terminated, _truncated, info = env.step(
            np.array([0.0, -1.0], dtype=np.float32)
        )

    assert info["lateral_offset_m"] < before


def test_a_stationary_car_cannot_move_sideways() -> None:
    """Regression: steering used to move the offset at a rate independent of
    forward speed, so a parked car could slide 5 m sideways in 0.2 s.
    """
    env = _env()
    _obs, info = env.reset(seed=0)
    before = info["lateral_offset_m"]

    for _ in range(50):
        _, _reward, _terminated, _truncated, info = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )

    assert info["speed_ms"] == 0.0
    assert info["lateral_offset_m"] == before


def test_full_outward_steering_alone_can_run_off_track() -> None:
    """Even with no cornering demand at all, steering hard toward the edge while
    moving should push the offset past a narrow track's half-width.
    """
    narrow_loop = Track(
        segments=(Segment(length_m=10_000.0, curvature_per_m=0.0, width_m=3.0),)
    )
    env = DriverEnv(EXAMPLE_CAR, narrow_loop, SimParams(seed=0))
    env.reset(seed=0)

    _total, info = _drive(env, np.array([1.0, 1.0], dtype=np.float32))

    assert info["off_track"] is True


def test_a_faster_lap_earns_more_reward() -> None:
    """Regression: progress alone pays 1.0 per lap at any pace, so a creeping car
    used to score exactly as well as a fast one. The time penalty must break that.
    """
    straight_loop = Track(segments=(Segment(length_m=1000.0, curvature_per_m=0.0),))
    fast_env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0))
    slow_env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0))
    fast_env.reset(seed=0)
    slow_env.reset(seed=0)

    fast_return, fast_info = _drive(fast_env, FULL_THROTTLE_NO_STEER)
    slow_return, slow_info = _drive(slow_env, np.array([0.1, 0.0], dtype=np.float32))

    assert fast_info["lap_completed"] and slow_info["lap_completed"]
    assert fast_info["lap_time_s"] < slow_info["lap_time_s"]
    assert fast_return > slow_return


def test_lap_time_is_interpolated_to_the_line() -> None:
    """Constant full throttle from rest on a straight: lap time must match the
    closed-form d = a t^2 / 2, to well inside one 0.02 s step.
    """
    length_m = 100.0  # short enough that top speed is never reached
    straight_loop = Track(segments=(Segment(length_m=length_m, curvature_per_m=0.0),))
    env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0))
    env.reset(seed=0)

    _total, info = _drive(env, FULL_THROTTLE_NO_STEER)

    accel = EXAMPLE_CAR.max_accel_g * GRAVITY_M_S2
    assert info["lap_time_s"] == pytest.approx(
        math.sqrt(2 * length_m / accel), abs=1e-3
    )


def test_unseeded_resets_continue_the_seeded_stream() -> None:
    """Regression: an unseeded reset used to reseed from `sim.seed` every time,
    replaying the first episode forever. Episodes must now differ from each other
    yet replay exactly for the same seed.
    """
    starts = []
    for _run in range(2):
        env = _env()
        offsets = [env.reset()[1]["lateral_offset_m"] for _ in range(3)]
        starts.append(offsets)

    assert len(set(starts[0])) == 3  # each episode starts differently
    assert starts[0] == starts[1]  # ...and the whole sequence is reproducible


def test_observation_sees_a_slow_corner_far_enough_ahead_to_brake() -> None:
    """The farthest lookahead must show a hairpin's speed limit before the car is
    inside its braking distance from top speed.
    """
    env = DriverEnv(EXAMPLE_CAR, technical_track(), SimParams(seed=0))
    env.reset(seed=0)
    hairpin_entry_m = 1000.0  # main straight ends here
    hairpin_limit_ms = max_corner_speed_ms(EXAMPLE_CAR, 1 / 15)
    braking_m = (EXAMPLE_CAR.max_speed_ms**2 - hairpin_limit_ms**2) / (
        2 * EXAMPLE_CAR.max_braking_g * GRAVITY_M_S2
    )
    env._distance_m = hairpin_entry_m - braking_m - 1.0  # just before braking

    farthest_lookahead = env._observation()[-3]

    assert farthest_lookahead < 0.5
