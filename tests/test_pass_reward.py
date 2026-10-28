from dataclasses import replace

import numpy as np
import pytest

from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.opponents import wrap_gap
from f1rl.envs.race_env import RaceEnv
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

FLAT_OUT = np.array([1.0, 0.0], dtype=np.float32)


def _env(**kw: object) -> RaceEnv:
    params = RaceEnvParams(
        tires=True,
        opponents=("slow",),
        safety_filter=True,
        **kw,  # type: ignore[arg-type]
    )
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return RaceEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def _start_behind(env: RaceEnv, gap_m: float, opp_speed: float) -> None:
    env.reset(seed=0, options={"start_s": 0.0, "gap_s": 0.5, "opponent": "slow"})
    assert env._opp is not None
    # On the first straight (s = 0..600 m), side by side in different lanes.
    env._state = me = replace(env._state, s_m=50.0, n_m=-3.0, psi_rad=0.0)
    env._opp.state = replace(me, s_m=me.s_m + gap_m, n_m=3.0, speed_ms=opp_speed)
    env._ahead = env._held = False


def _drive(env: RaceEnv, lane_m: float = -3.0) -> np.ndarray:
    """Flat out, holding a lane (a simple steering law)."""
    me = env._state
    steer = np.clip(-0.15 * (me.n_m - lane_m) - 1.5 * me.psi_rad, -1.0, 1.0)
    return np.array([1.0, steer], dtype=np.float32)


def _gap(env: RaceEnv) -> float:
    assert env._opp is not None
    return wrap_gap(env._state.s_m - env._opp.state.s_m, env._lap_m)


def test_a_pass_only_counts_once_it_has_stuck() -> None:
    env = _env(pass_hold_m=100.0, overtake_bonus=100.0)
    _start_behind(env, 15.0, 40.0)  # a much slower car in the other lane, ahead
    swapped = None
    for k in range(60):
        _, _, terminated, _, info = env.step(_drive(env))
        assert not terminated
        if info["swaps"] and swapped is None:
            swapped = k
            assert info["passes"] == 0  # ahead, but not yet for 100 m
        if info["passes"]:
            break
    assert swapped is not None and info["passes"] == 1 and k > swapped


def test_gt_sophy_reward_pays_metres_gained_on_a_close_car() -> None:
    plain, sophy = _env(), _env(pass_reward_per_m=1.0)
    for env in (plain, sophy):
        _start_behind(env, 15.0, 40.0)
    before = _gap(sophy)
    _, r_plain, *_ = plain.step(FLAT_OUT)
    _, r_sophy, *_ = sophy.step(FLAT_OUT)
    gained = _gap(sophy) - before
    assert gained > 0.0
    assert r_sophy - r_plain == pytest.approx(gained, rel=1e-6)


def test_gt_sophy_reward_ignores_a_car_far_away() -> None:
    plain, sophy = _env(), _env(pass_reward_per_m=1.0)
    for env in (plain, sophy):
        _start_behind(env, 150.0, 40.0)  # 150 m up the road: outside the window
    _, r_plain, *_ = plain.step(FLAT_OUT)
    _, r_sophy, *_ = sophy.step(FLAT_OUT)
    assert r_sophy == pytest.approx(r_plain)
