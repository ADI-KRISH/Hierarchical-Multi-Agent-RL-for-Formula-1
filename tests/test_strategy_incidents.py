"""Our car's spins and punctures in `StrategyEnv` (`StrategyIncidentParams`)."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from f1rl.config import EXAMPLE_CAR, StrategyIncidentParams, StrategyParams
from f1rl.envs.strategy_env import (
    INCIDENT_OBS_SIZE,
    OBS_SIZE,
    RaceTrackInfo,
    StrategyEnv,
)
from f1rl.models.race import build_lap_table

PARAMS = StrategyParams(grip_step=0.05, fuel_points=3, n_cars=6, car_pace_spread=0.0)
LAPS = 30


@pytest.fixture(scope="module")
def tracks() -> dict[str, RaceTrackInfo]:
    ds = 5.0
    n_turn = round(np.pi * 160 / ds)
    k = np.concatenate([np.zeros(100), np.full(n_turn, 1 / 160)] * 2)
    table = build_lap_table(EXAMPLE_CAR, ds, k.astype(np.float64), PARAMS)
    return {"oval": RaceTrackInfo("oval", float(ds * len(k)), table)}


def laps(
    env: StrategyEnv, seed: int, incident: str | None, action: Any
) -> list[dict[str, Any]]:
    env.reset(seed=seed, options={"laps": LAPS, "incident": incident})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(np.array(action(env)))
        assert env.observation_space.contains(obs)
    return env.history


def stay(env: StrategyEnv) -> list[int]:
    # Medium start, stay out, normal push; one stop for hards at lap 15.
    return [3, 2] if env.lap == 15 else [0, 2]


def test_off_by_default_keeps_the_observation(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    obs, info = env.reset(seed=0, options={"laps": LAPS})
    assert obs.shape == (OBS_SIZE,) and info["incident"] is None


def test_spin_costs_time_and_tyre(tracks: dict[str, RaceTrackInfo]) -> None:
    inc = StrategyIncidentParams(
        chance=1.0, spin_loss_range=(10.0, 10.0), spin_wear=0.1
    )
    env = StrategyEnv(tracks, replace(PARAMS, incidents=inc))
    clean = laps(env, 3, "none", stay)
    spun = laps(env, 3, "spin", stay)
    assert env.observation_space.shape == (OBS_SIZE + INCIDENT_OBS_SIZE,)
    lap = env.incident[0] if env.incident else -1
    assert spun[lap]["lap_s"] >= clean[lap]["lap_s"] + 10.0 - 1e-6
    assert spun[lap]["wear"] > clean[lap]["wear"] + 0.09
    assert env._incident_obs()[0] == 1.0


def test_puncture_forces_a_stop(tracks: dict[str, RaceTrackInfo]) -> None:
    inc = StrategyIncidentParams(
        chance=1.0, lap_range=(0.3, 0.3), forced_compound="soft"
    )
    env = StrategyEnv(tracks, replace(PARAMS, incidents=inc))
    hist = laps(env, 1, "puncture", lambda e: [0, 2])  # never asks for a stop
    lap = env.incident[0] if env.incident else -1
    assert hist[lap + 1]["pit"] == "soft"  # forced on the next lap
    assert sum(h["pit"] is not None for h in hist) == 1


def test_same_seed_same_incident(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(
        tracks, replace(PARAMS, incidents=StrategyIncidentParams(chance=1.0))
    )
    a = laps(env, 7, None, stay)
    first = env.incident
    b = laps(env, 7, None, stay)
    assert a == b and env.incident == first and first is not None
