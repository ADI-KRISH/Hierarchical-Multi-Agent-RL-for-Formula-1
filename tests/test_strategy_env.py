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


def test_undercut_and_overcut_are_counted(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    env.reset(seed=0, options={"laps": LAPS})
    base = [1, 0, 2, 3, 4, 5]
    swapped = [0, 1, 2, 3, 4, 5]
    # We stop on lap 10, car 1 (just ahead) on lap 12; we come out ahead.
    env.orders = [base] * 11 + [base] * 2 + [swapped] * (LAPS - 12)
    env.stop_laps = [[10], [12], [], [], [], []]
    assert env.battle_moves()["undercuts"] == 1
    # Same flip, but car 1 stopped first: an overcut.
    env.stop_laps = [[12], [10], [], [], [], []]
    assert env.battle_moves()["overcuts"] == 1
    # Stops too far apart do not count.
    env.stop_laps = [[10], [20], [], [], [], []]
    assert env.battle_moves() == {
        "undercuts": 0,
        "overcuts": 0,
        "undercut_on_us": 0,
        "overcut_on_us": 0,
    }


def test_rejoin_features(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    env.reset(seed=0, options={"laps": LAPS, "grid": 1})
    place, ahead, behind = env._rejoin()
    # From pole with the field 0.3 s apart, a 22 s stop drops us to the back.
    assert place == 1.0
    assert behind == 1.0
    assert 0.0 < ahead < 1.0


def test_baseline_rivals_and_win_bonus(tracks: dict[str, RaceTrackInfo]) -> None:
    p = replace(PARAMS, rivals="baseline", car_pace_spread=0.0)
    env = StrategyEnv(tracks, p)
    env.reset(seed=0, options={"laps": LAPS})
    assert all(plan.params.stints == ("medium", "hard") for plan in env.plans)
    stops = {plan.params.stop_fractions for plan in env.plans[1:]}
    assert len(stops) > 1  # jittered, not all on the same lap
    with pytest.raises(ValueError):
        StrategyEnv(tracks, replace(PARAMS, rivals="nope")).reset(seed=0)
    # From pole with no stop lost: the win bonus is paid at the flag.
    won = replace(p, reward=replace(p.reward, win_bonus=100.0))
    env = StrategyEnv(tracks, won)
    env.reset(seed=0, options={"laps": 3, "grid": 1})
    rewards = [env.step(np.array(a))[1] for a in ([2, 2], [3, 2], [0, 2])]
    info = env._info()
    if info["position"] == 1 and info["legal"]:
        assert rewards[-1] > 50.0


def test_tyre_inputs(tracks: dict[str, RaceTrackInfo]) -> None:
    env = StrategyEnv(tracks, PARAMS)
    env.reset(seed=0, options={"laps": LAPS})
    fresh = env._life_laps("hard", 0.0)
    assert fresh > env._life_laps("hard", 0.5) > 0.0
    assert env._life_laps("soft", 0.0) < env._life_laps("medium", 0.0) < fresh
    me = env.cars[0]
    assert env._tyre_lap(me, "soft", 0.0) < env._tyre_lap(me, "hard", 0.0)
    assert env._tyre_lap(me, "medium", 0.6) > env._tyre_lap(me, "medium", 0.0)
    assert env._tyre_gap(None) == 0.0
