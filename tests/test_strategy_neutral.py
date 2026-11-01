"""Safety car and VSC periods in `StrategyEnv` (`NeutralParams`)."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from f1rl.agents.strategy_baseline import FixedPlan
from f1rl.config import (
    EXAMPLE_CAR,
    NeutralParams,
    StrategyBaselineParams,
    StrategyParams,
)
from f1rl.envs.strategy_env import (
    NEUTRAL_OBS_SIZE,
    OBS_SIZE,
    RaceTrackInfo,
    StrategyEnv,
)
from f1rl.models.race import CarState, build_lap_table

PARAMS = StrategyParams(
    grip_step=0.05,
    fuel_points=3,
    n_cars=6,
    car_pace_spread=0.0,
    rivals="baseline",
    neutral=NeutralParams(sc_chance=0.5, vsc_chance=0.5, lap_range=(0.3, 0.3)),
)
LAPS = 30


@pytest.fixture(scope="module")
def tracks() -> dict[str, RaceTrackInfo]:
    ds = 5.0
    n_turn = round(np.pi * 160 / ds)
    k = np.concatenate([np.zeros(100), np.full(n_turn, 1 / 160)] * 2)
    table = build_lap_table(EXAMPLE_CAR, ds, k.astype(np.float64), PARAMS)
    return {"oval": RaceTrackInfo("oval", float(ds * len(k)), table)}


def drive(env: StrategyEnv, seed: int, kind: str, action: Any) -> list[list[float]]:
    """Gaps to the leader after every lap."""
    obs, _ = env.reset(seed=seed, options={"laps": LAPS, "neutral": kind})
    gaps = []
    done = False
    while not done:
        obs, _, done, _, _ = env.step(np.array(action(env)))
        assert env.observation_space.contains(obs)
        lead = env.cars[env.order[0]].time_s
        gaps.append([env.cars[c].time_s - lead for c in env.order])
    return gaps


def test_observation_and_period(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    assert env.observation_space.shape == (OBS_SIZE + NEUTRAL_OBS_SIZE,)
    env.reset(seed=0, options={"laps": LAPS, "neutral": "sc"})
    laps = [k for k, n in enumerate(env.neutral) if n]
    assert laps[0] == 9 and 3 <= len(laps) <= 5
    assert all(env.neutral[k] == "sc" for k in laps)
    off = StrategyEnv(tracks, replace(PARAMS, neutral=NeutralParams()))
    obs, info = off.reset(seed=0, options={"laps": LAPS})
    assert obs.shape == (OBS_SIZE,) and info["neutral"] is None


def test_safety_car_closes_the_field(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    stay = lambda e: [3, 2] if e.lap == 20 else [0, 2]  # noqa: E731
    sc = drive(env, 4, "sc", stay)
    vsc = drive(env, 4, "vsc", stay)
    first = env.neutral.index("vsc")
    assert sc[first + 2][-1] < vsc[first + 2][-1]  # SC bunched the field, VSC did not
    # Nobody passes on a neutralised lap: the order only changes by stops.
    env.reset(seed=4, options={"laps": LAPS, "neutral": "vsc"})
    for _ in range(first):
        env.step(np.array([0, 2]))
    before = list(env.order)
    env.step(np.array([0, 2]))
    stopped = [c for c in range(1, env.params.n_cars) if env.cars[c].stops]
    assert [c for c in env.order if c not in stopped] == [
        c for c in before if c not in stopped
    ]


def test_cheap_stop_and_rule_takes_it() -> None:
    plan = FixedPlan(StrategyBaselineParams(stop_fractions=(0.45,)))
    car = CarState(0.0, "medium", 0.3, 10, 50.0, frozenset({"medium"}))
    # 10 laps before the planned stop (lap 45 of 100), inside the 15% window.
    assert plan.plan(PARAMS, car, 35, 100).pit_to is None
    assert plan.plan(PARAMS, car, 35, 100, neutral=True).pit_to == "hard"
    assert plan.plan(PARAMS, car, 20, 100, neutral=True).pit_to is None


def test_min_stint_blocks_back_to_back_stops(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, replace(PARAMS, min_stint_laps=5))
    env.reset(seed=0, options={"laps": LAPS, "neutral": "none"})
    env.step(np.array([2, 2]))  # grid: mediums
    for _ in range(4):
        env.step(np.array([3, 2]))  # ask for hards every lap
    assert env.cars[0].stops == 1  # only the first call was taken
    for _ in range(4):
        env.step(np.array([0, 2]))
    env.step(np.array([1, 2]))  # 5 laps on that set: allowed again
    assert env.cars[0].stops == 2
