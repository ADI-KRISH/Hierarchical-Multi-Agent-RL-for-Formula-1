"""Early-stop rule, rivals' pace inputs, overtake mode and the extra SC cases."""

from dataclasses import replace

import numpy as np
import pytest

from f1rl.agents.strategy import baseline_policy
from f1rl.config import EXAMPLE_CAR, ErsParams, NeutralParams, StrategyParams
from f1rl.envs.strategy_env import (
    ERS_OBS_SIZE,
    OBS_SIZE,
    PACE_OBS_SIZE,
    RaceTrackInfo,
    StrategyEnv,
)
from f1rl.models.race import build_lap_table

PARAMS = StrategyParams(
    grip_step=0.05, fuel_points=3, n_cars=6, car_pace_spread=0.0, rivals="baseline"
)
LAPS = 30


@pytest.fixture(scope="module")
def tracks() -> dict[str, RaceTrackInfo]:
    ds = 5.0
    n_turn = round(np.pi * 160 / ds)
    k = np.concatenate([np.zeros(100), np.full(n_turn, 1 / 160)] * 2)
    table = build_lap_table(EXAMPLE_CAR, ds, k.astype(np.float64), PARAMS)
    return {"oval": RaceTrackInfo("oval", float(ds * len(k)), table)}


def test_no_early_stop_by_choice(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, replace(PARAMS, min_stop_fraction=0.3))
    env.reset(seed=0, options={"laps": LAPS})
    env.step(np.array([2, 2]))
    for _ in range(7):
        env.step(np.array([3, 2]))  # laps 1-7 < 9: refused
    assert env.cars[0].stops == 0
    while env.lap < 9:
        env.step(np.array([0, 2]))
    env.step(np.array([3, 2]))
    assert env.cars[0].stops == 1


def test_early_stop_allowed_under_safety_car(tracks: dict[str, RaceTrackInfo]) -> None:
    nu = NeutralParams(sc_chance=1.0, lap_range=(0.1, 0.1))
    env = StrategyEnv(tracks, replace(PARAMS, min_stop_fraction=0.5, neutral=nu))
    env.reset(seed=0, options={"laps": LAPS, "neutral": "sc"})
    first = env.neutral.index("sc")
    env.step(np.array([2, 2]))
    while env.lap < first:
        env.step(np.array([0, 2]))
    env.step(np.array([3, 2]))
    assert env.cars[0].stops == 1


def test_pace_and_battery_inputs(tracks: dict[str, RaceTrackInfo]) -> None:
    params = replace(PARAMS, pace_obs=True, ers=ErsParams(enabled=True))
    env = StrategyEnv(tracks, params)
    assert env.observation_space.shape == (OBS_SIZE + PACE_OBS_SIZE + ERS_OBS_SIZE,)
    assert list(env.action_space.nvec) == [4, 5, 2]
    obs, _ = env.reset(seed=1, options={"laps": LAPS})
    env.step(np.array([2, 2, 0]))
    obs, *_ = env.step(np.array([0, 2, 1]))  # overtake mode: battery used
    assert env.battery == pytest.approx(1.0 - 0.5 + 0.1)
    obs, *_ = env.step(np.array([0, 2, 1]))
    assert env.battery == pytest.approx(0.2)  # 0.6 >= 0.5: used again
    obs, *_ = env.step(np.array([0, 2, 1]))  # 0.2: not enough charge, no use
    assert env.battery == pytest.approx(0.3) and not env.history[-1]["ers"]
    assert env.observation_space.contains(obs)


def test_two_safety_cars_and_red_flag(tracks: dict[str, RaceTrackInfo]) -> None:
    nu = NeutralParams(sc_chance=1.0, max_periods=2, red_chance=0.5)
    env = StrategyEnv(tracks, replace(PARAMS, neutral=nu))
    env.reset(seed=2, options={"laps": LAPS, "neutral": "sc2"})
    runs = sum(
        1 for k in range(LAPS) if env.neutral[k] and (k == 0 or not env.neutral[k - 1])
    )
    assert 1 <= runs <= 2 and set(filter(None, env.neutral)) == {"sc"}
    env.reset(seed=2, options={"laps": LAPS, "neutral": "red"})
    red = env.neutral.index("red")
    obs, _ = env.reset(seed=2, options={"laps": LAPS, "neutral": "red"})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(baseline_policy(env, obs))
    # Every car, ours too, changed tyres for free on the red-flag lap.
    assert all(red in env.stop_laps[c] for c in range(env.params.n_cars))


def test_no_late_stop_and_no_wets_on_dry(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, replace(PARAMS, late_stop_laps=3))
    env.reset(seed=0, options={"laps": LAPS})
    env.step(np.array([2, 2]))
    while env.lap < LAPS - 3:
        env.step(np.array([0, 2]))
    for _ in range(3):
        env.step(np.array([3, 2]))  # the last 3 laps: refused
    assert env.cars[0].stops == 0
    wet = replace(PARAMS, wet_tyres=True, wet_tyre_min_wetness=0.05)
    env = StrategyEnv(tracks, wet)
    env.reset(seed=0, options={"laps": LAPS, "rain": False})
    env.step(np.array([2, 2]))
    env.step(np.array([env.tyres.index("wet") + 1, 2]))
    assert env.cars[0].stops == 0 and env.cars[0].compound != "wet"
    env.step(np.array([env.tyres.index("hard") + 1, 2]))
    assert env.cars[0].stops == 1
