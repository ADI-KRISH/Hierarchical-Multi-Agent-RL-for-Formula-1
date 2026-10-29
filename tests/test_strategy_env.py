"""`StrategyEnv` on a synthetic oval (no track download)."""

from dataclasses import replace
from typing import Any

import numpy as np
import pytest

from f1rl.agents.strategy import baseline_policy, compare, race, random_policy
from f1rl.config import EXAMPLE_CAR, StrategyParams
from f1rl.envs.strategy_env import OBS_SIZE, RaceTrackInfo, StrategyEnv
from f1rl.models.race import build_lap_table, race_laps

PARAMS = StrategyParams(grip_step=0.05, fuel_points=3, n_cars=6)
LAPS = 30


def oval() -> tuple[float, np.typing.NDArray[np.float64]]:
    """A 2 km closed path: two 500 m straights and two 160 m-radius half circles."""
    ds = 5.0
    n_turn = round(np.pi * 160 / ds)
    k = np.concatenate([np.zeros(100), np.full(n_turn, 1 / 160)] * 2)
    return ds, k.astype(np.float64)


@pytest.fixture(scope="module")
def tracks() -> dict[str, RaceTrackInfo]:
    ds, k = oval()
    table = build_lap_table(EXAMPLE_CAR, ds, k, PARAMS)
    return {"oval": RaceTrackInfo("oval", float(ds * len(k)), table)}


def run(env: StrategyEnv, actions: Any, seed: int = 0) -> list[Any]:
    obs, _ = env.reset(seed=seed, options={"laps": LAPS})
    steps = []
    done = False
    while not done:
        a = actions(env) if callable(actions) else actions
        obs, r, done, trunc, info = env.step(np.array(a))
        assert env.observation_space.contains(obs)
        assert not trunc
        steps.append((r, info))
    return steps


def test_spaces_and_race_length(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    obs, info = env.reset(seed=0)
    assert obs.shape == (OBS_SIZE,)
    assert info["laps"] == tracks["oval"].laps == race_laps(2010.0) == 152
    obs, info = env.reset(seed=0, options={"laps": LAPS})
    assert len(run(env, [0, 2])) == LAPS


def test_same_seed_same_race(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    a = run(env, [0, 4], seed=5)
    b = run(env, [0, 4], seed=5)
    assert a == b


def test_never_stopping_is_disqualified(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    steps = run(env, [0, 2])
    final_r, info = steps[-1]
    assert not info["legal"]
    assert info["stops"] == 0
    assert final_r < -PARAMS.reward.rule_penalty / 2


def test_grid_choice_is_free_and_stop_is_legal(
    tracks: dict[str, RaceTrackInfo],
) -> None:
    env = StrategyEnv(tracks, PARAMS)

    def plan(e: StrategyEnv) -> list[int]:
        if e.lap == 0:
            return [1, 2]  # start on softs
        return [3, 2] if e.lap == 10 else [0, 2]

    steps = run(env, plan)
    assert steps[0][1]["stops"] == 0
    assert steps[0][1]["compound"] == "soft"
    assert steps[10][1]["stops"] == 1
    assert steps[-1][1]["compound"] == "hard"
    assert steps[-1][1]["legal"]


def test_grid_slot_option(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    _, info = env.reset(seed=1, options={"grid": 1, "laps": LAPS})
    assert info["position"] == 1
    _, info = env.reset(seed=1, options={"grid": PARAMS.n_cars, "laps": LAPS})
    assert info["position"] == PARAMS.n_cars


def test_baseline_races_legally_and_beats_random(
    tracks: dict[str, RaceTrackInfo],
) -> None:
    env = StrategyEnv(tracks, PARAMS)
    res = race(env, baseline_policy, seed=0, track="oval")
    assert res["legal"]
    assert res["stops"] >= 1
    assert len(res["history"]) == tracks["oval"].laps
    out = compare(
        env, {"baseline": baseline_policy, "random": random_policy(0)}, ("oval",), 4, 0
    )
    assert out["baseline"]["legal"] == 1.0
    assert out["baseline"]["race_time_s"] < out["random"]["race_time_s"]


def test_race_craft_skill_gains_places(tracks: dict[str, RaceTrackInfo]) -> None:
    places = []
    for skill in (0.0, 2.0):
        env = StrategyEnv(
            tracks, replace(PARAMS, attack_skill_s=skill, defend_skill_s=skill)
        )
        places.append(
            np.mean(
                [race(env, baseline_policy, s, "oval")["position"] for s in range(4)]
            )
        )
    assert places[1] <= places[0]
