from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1rl.agents.line_follower import LineFollower
from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.race_env import RaceEnv
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

PARAMS = RaceEnvParams()
MARGIN_M = PARAMS.car_half_width_m + PARAMS.line_safety_m


def _env(mode: str = "random", seed: int = 0) -> RaceEnv:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    return RaceEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=seed, max_episode_steps=6000),
        PARAMS,
        line,
        mode,
    )


@pytest.mark.parametrize("mode", ["random", "lap"])
def test_passes_gymnasium_env_checker(mode: str) -> None:
    check_env(_env(mode), skip_render_check=True)


def test_random_agent_never_crashes_the_env() -> None:
    env = _env()
    obs, _ = env.reset(seed=0)
    for _ in range(300):
        obs, reward, terminated, truncated, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs) and np.all(np.isfinite(obs))
        assert np.isfinite(reward)
        if terminated or truncated:
            obs, _ = env.reset()


def test_a_bad_mode_is_rejected() -> None:
    with pytest.raises(ValueError, match="mode"):
        _env("race")


def test_resets_are_reproducible_from_a_seed() -> None:
    a, _ = _env(seed=3).reset(seed=3)
    b, _ = _env(seed=3).reset(seed=3)
    assert np.array_equal(a, b)


def test_random_starts_cover_the_lap() -> None:
    env = _env()
    starts = []
    for k in range(40):
        _, info = env.reset(seed=k)
        starts.append(info["distance_m"])
    assert max(starts) - min(starts) > 0.8 * env.track.length_m


def test_driving_into_the_wall_ends_the_episode_with_a_penalty() -> None:
    env = _env("lap")
    env.reset(seed=0)
    reward, terminated = 0.0, False
    info: dict[str, Any] = {}
    for _ in range(200):
        _, reward, terminated, _, info = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )
        if terminated:
            break
    assert terminated and info["off_track"]
    assert reward < -PARAMS.fail_penalty / 2


def test_reward_is_distance_advanced() -> None:
    env = _env("lap")
    env.reset(seed=0)
    _, reward, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float32))
    assert reward == pytest.approx(
        info["advanced_m"] / PARAMS.progress_scale_m, rel=0.05
    )


def test_the_observation_has_no_lap_progress_feature() -> None:
    """Shifting the car to another station with identical local geometry must not
    change the observation, or a policy could memorise a track by where it is."""
    env = _env("lap")
    env.reset(seed=0)
    state = env._state
    # Station 0 and station n/2 are both on a straight (the same geometry) ...
    env._state = state.__class__(0.0, 0.0, 0.0, 50.0)
    first = env._observation()
    env._state = state.__class__(env.track.length_m / 2, 0.0, 0.0, 50.0)
    second = env._observation()
    # ... apart from where the straights end, the two see the same road; a lap
    # counter would make every element differ. Most of the vector must agree.
    assert np.mean(np.isclose(first, second, atol=1e-3)) > 0.5


@pytest.mark.slow
def test_scripted_driver_laps_the_optimal_line_within_a_few_percent() -> None:
    env = _env("lap")
    track, line = env.track, env.reference
    assert line is not None
    driver = LineFollower(EXAMPLE_CAR, track, line)
    _, info = env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(driver.act(info))
        done = terminated or truncated
    assert info["lap_completed"]
    assert info["lap_time_s"] < line.lap_time_s * 1.04
    assert info["lap_time_s"] > line.lap_time_s * 0.99  # the reference is a ceiling


def test_reference_features_are_zero_error_when_the_car_is_on_the_line() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    params = RaceEnvParams(reference_features=True)
    env = RaceEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, mode="lap")
    plain = _env("lap")
    obs, _ = env.reset(seed=0)
    k = len(params.reference_points_m)
    assert obs.shape[0] == plain.observation_space.shape[0] + 2 * k + 1  # type: ignore[index]
    assert obs[-(2 * k + 1)] == pytest.approx(0.0, abs=0.02)  # lateral error now
    assert obs[-1] == pytest.approx(0.0, abs=0.02)  # heading error


def test_reference_features_need_a_reference_line() -> None:
    with pytest.raises(ValueError, match="reference"):
        RaceEnv(
            EXAMPLE_CAR,
            stadium_track(),
            SimParams(seed=0),
            RaceEnvParams(reference_features=True),
        )


def test_residual_env_with_zero_correction_drives_the_follower_lap() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    params = RaceEnvParams(residual_scale=0.2)
    env = RaceEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=0, max_episode_steps=6000),
        params,
        line,
        "lap",
    )
    check_env(env, skip_render_check=True)
    _, info = env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        done = terminated or truncated
    assert info["lap_completed"]
    assert line.lap_time_s * 0.99 < info["lap_time_s"] < line.lap_time_s * 1.04


def test_residual_needs_a_reference_line() -> None:
    with pytest.raises(ValueError, match="reference"):
        RaceEnv(
            EXAMPLE_CAR,
            stadium_track(),
            SimParams(seed=0),
            RaceEnvParams(residual_scale=0.2),
        )


def _tire_lap(push: float, wear: float = 0.0) -> dict[str, Any]:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    params = RaceEnvParams(residual_scale=0.2, tires=True, wear_cost_per_unit=1000.0)
    env = RaceEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=0, max_episode_steps=6000),
        params,
        line,
        "lap",
    )
    check_env(env, skip_render_check=True)
    _, info = env.reset(seed=0, options={"push": push, "wear": wear})
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        done = terminated or truncated
    return info


def test_conserving_is_slower_and_wears_less_than_pushing() -> None:
    push, conserve = _tire_lap(1.0), _tire_lap(0.0)
    assert push["lap_completed"] and conserve["lap_completed"]
    assert conserve["lap_time_s"] > push["lap_time_s"]
    assert conserve["wear"] < push["wear"]
    assert push["wear"] > 0.0


def test_worn_tyres_lap_slower() -> None:
    new, worn = _tire_lap(1.0, 0.0), _tire_lap(0.5, 0.5)
    assert worn["lap_time_s"] > new["lap_time_s"] or not worn["lap_completed"]


def _opp_env(mode: str = "lap") -> RaceEnv:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    params = RaceEnvParams(
        residual_scale=0.2,
        tires=True,
        opponents=("slow", "attacker", "defender", "equal"),
    )
    return RaceEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=0, max_episode_steps=6000),
        params,
        line,
        mode,
    )


@pytest.mark.parametrize("mode", ["random", "lap"])
def test_env_with_opponents_passes_the_checker(mode: str) -> None:
    check_env(_opp_env(mode), skip_render_check=True)


def test_following_a_slower_car_closely_costs_grip() -> None:
    env = _opp_env()
    _, info = env.reset(seed=0, options={"opponent": "slow", "gap_m": 20.0})
    grips = []
    for _ in range(200):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        grips.append(env._grip)
        if terminated or truncated:
            break
    assert min(grips) < 1.0  # sat in its wake at some point
    assert info["passes"] == 0  # the plain follower does not pass (or avoid) anyone


def test_an_attacker_starts_behind_and_stays_clear_of_the_car() -> None:
    env = _opp_env()
    obs, _ = env.reset(seed=0, options={"opponent": "attacker", "gap_m": 40.0})
    assert obs[-4] == 1.0 and obs[-3] < 0  # present, and it is behind us
    for _ in range(150):
        _, _, terminated, truncated, info = env.step(np.zeros(2, dtype=np.float32))
        if terminated or truncated:
            break
    assert not info["contact"]  # opponents yield rather than hit the car
