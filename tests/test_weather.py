"""Rain model (`f1rl.models.weather`) and wet races in `StrategyEnv`."""

from dataclasses import replace

import numpy as np
import pytest

from f1rl.agents.strategy import baseline_policy, race
from f1rl.config import EXAMPLE_CAR, StrategyParams, TireParams, WeatherParams
from f1rl.envs.strategy_env import (
    OBS_SIZE,
    WEATHER_OBS_SIZE,
    RaceTrackInfo,
    StrategyEnv,
)
from f1rl.models.race import CarState, LapPlan, build_lap_table, car_lap
from f1rl.models.weather import (
    race_weather,
    rule_tyre_class,
    wet_grip,
    wet_wear,
)

W = WeatherParams()
PARAMS = StrategyParams(
    grip_step=0.05,
    grip_min=0.35,
    fuel_points=3,
    n_cars=6,
    wet_tyres=True,
    weather=replace(W, rain_chance=0.5),
)


def best(w: float) -> str:
    return max(("soft", "intermediate", "wet"), key=lambda c: wet_grip(W, c, w))


def test_each_tyre_has_its_window() -> None:
    assert best(0.0) == best(0.2) == "soft"
    assert best(0.45) == "intermediate"
    assert best(0.9) == best(1.0) == "wet"
    # Crossovers sit near the rule's thresholds.
    assert abs(wet_grip(W, "medium", 0.3) - wet_grip(W, "intermediate", 0.3)) < 0.03
    assert abs(wet_grip(W, "wet", 0.7) - wet_grip(W, "intermediate", 0.7)) < 0.03
    assert wet_grip(W, "hard", 0.0) == 1.0


def test_wet_tyres_overheat_when_dry() -> None:
    assert wet_wear(W, "medium", 0.0) == 1.0
    assert wet_wear(W, "intermediate", 0.0) == pytest.approx(W.overheat_wear)
    assert wet_wear(W, "intermediate", 0.5) == 1.0
    assert wet_wear(W, "wet", 0.3) > wet_wear(W, "wet", 0.7) == 1.0


def test_shower_wets_then_dries_and_is_seeded() -> None:
    a = race_weather(W, np.random.default_rng(3), 60, rain=True)
    b = race_weather(W, np.random.default_rng(3), 60, rain=True)
    assert a == b and a.wet
    peak = int(np.argmax(a.wetness))
    assert a.wetness[peak] > 0.2
    assert a.wetness[-1] < a.wetness[peak]
    assert all(0.0 <= w <= 1.0 for w in a.wetness)
    dry = race_weather(W, np.random.default_rng(3), 60, rain=False)
    assert not dry.wet and max(dry.wetness) == 0.0
    assert 0.0 <= a.forecast(W, 10, 6) <= 1.0


def test_rule_switches_with_hysteresis() -> None:
    assert rule_tyre_class(W, "medium", 0.1) == "slick"
    assert rule_tyre_class(W, "medium", 0.4) == "intermediate"
    assert rule_tyre_class(W, "intermediate", 0.8) == "wet"
    assert rule_tyre_class(W, "intermediate", 0.28) == "intermediate"  # holds
    assert rule_tyre_class(W, "medium", 0.28) == "slick"  # holds
    assert rule_tyre_class(W, "intermediate", 0.1) == "slick"


def test_slicks_are_slow_in_the_wet() -> None:
    ds, n = 5.0, 400
    k = np.concatenate([np.zeros(n // 2), np.full(n // 2, 1 / 150)])
    table = build_lap_table(EXAMPLE_CAR, ds, k, PARAMS)
    car = CarState(0.0, "medium", 0.0, 0, 50.0, frozenset({"medium"}))
    tire = TireParams()
    dry, _ = car_lap(PARAMS, tire, table, car, LapPlan(0.5), 1.0, False, 0.0)
    wet, _ = car_lap(PARAMS, tire, table, car, LapPlan(0.5), 1.0, False, 0.8)
    inter_car = replace(car, compound="intermediate")
    inter, _ = car_lap(PARAMS, tire, table, inter_car, LapPlan(0.5), 1.0, False, 0.8)
    assert dry < inter < wet


@pytest.fixture(scope="module")
def tracks() -> dict[str, RaceTrackInfo]:
    ds = 5.0
    n_turn = round(np.pi * 160 / ds)
    k = np.concatenate([np.zeros(100), np.full(n_turn, 1 / 160)] * 2)
    table = build_lap_table(EXAMPLE_CAR, ds, k, PARAMS)
    return {"oval": RaceTrackInfo("oval", float(ds * len(k)), table)}


def test_wet_env_spaces_and_rain_race(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    obs, info = env.reset(seed=0, options={"laps": 40, "rain": True})
    assert obs.shape == (OBS_SIZE + WEATHER_OBS_SIZE,)
    assert env.action_space.nvec[0] == 6
    assert info["rain"]
    res = race(env, baseline_policy, seed=1, track="oval", rain=True)
    assert res["legal"]
    used = {h["pit"] for h in res["history"] if h["pit"]}
    assert used & {"intermediate", "wet"}, "the rule-based plan reacts to rain"
    # Rivals went to wet-weather tyres too.
    assert any(c.compounds_used & {"intermediate", "wet"} for c in env.cars[1:])


def test_wet_tyre_waives_two_compound_rule(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    env.reset(seed=0, options={"laps": 6, "rain": False})
    for k in range(6):
        env.step(np.array([4 if k == 2 else 0, 2]))  # one set of inters
    assert env._info()["legal"]


def test_dry_env_unchanged_without_wet_tyres(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, replace(PARAMS, wet_tyres=False))
    obs, info = env.reset(seed=0, options={"laps": 20})
    assert obs.shape == (OBS_SIZE,) and not info["rain"]
