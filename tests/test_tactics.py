import dataclasses
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1rl.agents.path_driver import PathDriver
from f1rl.agents.tactics import (
    DIVE_BRAKE_FRACTION,
    LANE_MOVES,
    MOVES,
    corners_ahead,
    make_move,
    room_beside,
)
from f1rl.config import EXAMPLE_CAR, OPPONENT_KINDS, RaceEnvParams, SimParams
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.maneuver import Maneuver, hold, shift_at
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

PARAMS = RaceEnvParams(tactical=True, tires=True, opponents=("none", *OPPONENT_KINDS))
MARGIN_M = PARAMS.car_half_width_m + PARAMS.line_safety_m


def _env(mode: str = "lap") -> TacticalEnv:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    return TacticalEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=0, max_episode_steps=6000),
        PARAMS,
        line,
        mode,
    )


def test_a_manoeuvre_eases_between_knots_and_holds_the_last() -> None:
    m = Maneuver(100.0, ((0.0, 0.0), (50.0, 2.0), (100.0, -1.0)))
    s = np.array([50.0, 100.0, 125.0, 150.0, 200.0, 400.0])
    got = shift_at(m, s)
    assert got[0] == 0.0 and got[1] == 0.0  # before the start: the first knot
    assert got[2] == pytest.approx(1.0)  # halfway up the cosine ease
    assert got[3] == pytest.approx(2.0)
    assert got[4] == pytest.approx(-1.0) and got[5] == pytest.approx(-1.0)
    assert hold(1.5).end_s == 0.0
    assert shift_at(hold(1.5), np.array([3.0]))[0] == 1.5


def test_the_path_driver_laps_the_line_better_than_the_old_follower() -> None:
    env = _env()
    env.reset(seed=0)
    driver = env._driver()
    assert isinstance(driver, PathDriver)
    _, info = env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(MOVES.index("line"))
        done = terminated or truncated
    assert info["lap_completed"]
    assert env.reference is not None
    assert info["lap_time_s"] < env.reference.lap_time_s * 1.03


def test_corners_ahead_lists_the_braking_point_before_the_apex() -> None:
    env = _env()
    env.reset(seed=0)
    corners = [c for s in range(0, 1800, 100) for c in corners_ahead(env._driver(), s)]
    assert corners
    assert all(c.brake_m <= c.apex_m for c in corners)
    assert all(c.side in (-1.0, 1.0) for c in corners)


@pytest.mark.parametrize("move", MOVES)
def test_every_move_can_be_driven_to_the_end_of_a_lap(move: str) -> None:
    env = _env()
    env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(MOVES.index(move))
        done = terminated or truncated
    assert not info["off_track"], move


def test_a_move_with_no_road_for_it_falls_back_to_the_line() -> None:
    env = _env()
    env.reset(seed=0)
    d = env._driver()
    m = make_move("divebomb", d, 0.0, 0.0, amplitude=0.1)  # nothing to shift into
    assert m.points[-1][1] == 0.0


@pytest.mark.parametrize("mode", ["random", "lap"])
def test_the_tactical_env_passes_the_gymnasium_checker(mode: str) -> None:
    check_env(_env(mode), skip_render_check=True)


@pytest.mark.parametrize("kind", list(OPPONENT_KINDS))
def test_every_opponent_kind_runs(kind: str) -> None:
    env = _env()
    env.reset(seed=0, options={"opponent": kind, "gap_m": 25.0})
    for _ in range(80):
        _, _, terminated, truncated, info = env.step(0)
        if terminated or truncated:
            break
    assert isinstance(info["contact"], bool)
    assert info["passes"] in (-1, 0, 1)


def test_the_overtake_assist_is_for_the_car_behind_on_a_straight() -> None:
    env = _env()
    env.reset(seed=0, options={"opponent": "equal", "gap_m": 30.0})
    assert env._opp is not None
    behind, ahead = env._state, env._opp.state
    straight = abs(env._kappa_at(behind.s_m)) < 1e-4
    boost = env._tow(behind, ahead)
    assert boost == (PARAMS.tow_boost_ms if straight else 0.0)
    assert env._tow(ahead, behind) == 0.0  # the leader gets none


def test_lane_moves_pick_an_offset_the_road_has_room_for() -> None:
    env = _env()
    env.reset(seed=0)
    d = env._driver()
    for name in LANE_MOVES:
        m = make_move(name, d, 300.0, 0.5, threat_shift=0.2)
        assert m.points[0] == (0.0, 0.5)  # from where the car is now
        want = m.points[-1][1]
        side = 1.0 if want >= 0 else -1.0
        assert abs(want) <= room_beside(d, 300.0, 120.0, side)
        if want:  # beside the other car, on the named side
            assert (want > 0.2) == name.startswith("left")
        assert (m.brake_fraction == DIVE_BRAKE_FRACTION) == name.endswith("_late")


def test_lane_moves_are_offered_only_when_turned_on() -> None:
    assert _env().action_space.n == len(MOVES)
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    params = dataclasses.replace(PARAMS, lane_moves=True)
    env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "lap")
    assert env.action_space.n == len(MOVES) + len(LANE_MOVES)
    env.reset(seed=0)
    for _ in range(40):  # left of where a car would be, late braking
        *_, terminated, truncated, info = env.step(env.moves.index("left3.0_late"))
        if terminated or truncated:
            break
    assert not info["off_track"]


def test_a_path_driver_with_its_steering_maxed_out_can_still_brake() -> None:
    from f1rl.config import FollowerParams
    from f1rl.models.maneuver import hold as hold_shift

    env = _env()
    env.reset(seed=0)
    base = env._driver()
    assert isinstance(base, PathDriver)
    # Pushed 3 m off its path, pointing away from it, and well over its speed.
    s = 400.0
    _, pn, _, _, pv = base.plan(s, hold_shift(0.0))
    info = {
        "distance_m": s,
        "lateral_offset_m": float(pn[0]) + 3.0,
        "heading_rad": 0.2,
        "speed_ms": float(pv[0]) + 20.0,
    }

    def act(keep: float) -> np.ndarray:
        d = PathDriver(
            base.car,
            base.track,
            base.line,
            base.room_l + 0.2,
            base.room_r + 0.2,
            FollowerParams(
                grip_fraction=base.params.grip_fraction, saturated_brake=keep
            ),
        )
        return d.act(info, hold_shift(0.0))

    old, new = act(0.0), act(0.6)
    assert abs(old[1]) == pytest.approx(1.0) and old[0] == pytest.approx(0.0)
    assert new[0] == pytest.approx(-0.6, abs=1e-6)  # brakes...
    assert abs(new[1]) == pytest.approx(0.8, abs=1e-6)  # ...giving up some steering


def test_aero_drag_cuts_the_drag_of_a_car_in_the_tow() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    off = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), PARAMS, line, "lap")
    assert off._drag(False) == 0.0 and off._drag(True) == 0.0
    params = dataclasses.replace(PARAMS, aero_drag=True, tow_drag_cut=0.12)
    env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "lap")
    k = env._drag(False)
    top = EXAMPLE_CAR.max_speed_ms
    assert k * top**2 == pytest.approx(EXAMPLE_CAR.max_accel_g * 9.80665, rel=1e-3)
    assert env._drag(True) == pytest.approx(0.88 * k)
    env.reset(seed=0)
    assert env._car_now.drag_per_m == pytest.approx(k)  # nobody ahead: full drag


def test_plan_features_add_the_next_corners_and_the_cars_ahead_moves() -> None:
    from f1rl.envs.tactical_env import PLAN_FEATURES

    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    base = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), PARAMS, line, "random")
    params = dataclasses.replace(PARAMS, plan_features=True)
    env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")
    n0 = base.observation_space.shape[0]  # type: ignore[index]
    assert env.observation_space.shape[0] == n0 + PLAN_FEATURES  # type: ignore[index]
    obs, _ = env.reset(seed=0, options={"opponent": "slow", "gap_m": 15.0})
    assert env.observation_space.contains(obs)
    for _ in range(6):
        obs, *_ = env.step(env.moves.index("line"))
    plan = obs[-PLAN_FEATURES:]
    assert plan[2] == 1.0  # no section battle: nothing counting down
    assert abs(plan[3]) < 2.0 and abs(plan[4]) < 2.0
    check_env(env, skip_render_check=True)


def _box_env(n: int) -> Any:
    """A do-nothing env with an ``n``-number observation and 3 moves."""
    import gymnasium as gym

    class Env(gym.Env):  # type: ignore[type-arg]
        observation_space = gym.spaces.Box(-1.0, 1.0, (n,), np.float32)
        action_space = gym.spaces.Discrete(3)

        def reset(self, *, seed: Any = None, options: Any = None) -> Any:
            super().reset(seed=seed)
            return np.zeros(n, np.float32), {}

        def step(self, action: Any) -> Any:
            return np.zeros(n, np.float32), 0.0, True, False, {}

    return Env()


def test_a_saved_policy_loads_into_a_grown_observation_unchanged(
    tmp_path: Path,
) -> None:
    import torch
    from stable_baselines3 import PPO

    from f1rl.agents.tactical import load_grown

    kw: dict[str, Any] = {"device": "cpu", "n_steps": 8, "batch_size": 8}
    small = PPO("MlpPolicy", _box_env(4), seed=0, **kw)
    small.save(tmp_path / "small.zip")
    big = PPO("MlpPolicy", _box_env(6), seed=1, **kw)
    grown = load_grown(big, str(tmp_path / "small.zip"))
    assert grown  # the first layers grew
    obs = torch.tensor([[0.3, -0.2, 0.5, 0.1]])
    wide = torch.cat([obs, torch.tensor([[0.9, -0.7]])], dim=1)
    with torch.no_grad():
        a = small.policy.get_distribution(obs).distribution.probs
        b = big.policy.get_distribution(wide).distribution.probs
    assert torch.allclose(a, b, atol=1e-6)


def test_a_trained_defender_sees_the_plain_layout_and_drives_the_car_ahead() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    params = dataclasses.replace(
        PARAMS, lane_moves=True, plan_features=True, opponents=("rival_ahead_slow",)
    )
    env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")
    base = len(env._observation())
    seen: list[int] = []

    def defender(obs: np.ndarray) -> int:
        seen.append(len(obs))
        return MOVES.index("cover")

    env.fixed_rival = env.rival = defender
    env.rival_plain_obs = True
    env.reset(seed=0, options={"opponent": "rival_ahead_slow", "gap_m": 30.0})
    for _ in range(10):
        env.step(env.moves.index("line"))
    assert seen and set(seen) == {base + len(MOVES) + 2}
    assert env._opp_move in ("cover", "line")  # its pick, or line if too late


def test_slower_opponents_can_brake_as_late_as_we_do() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    g = 9.80665 * EXAMPLE_CAR.max_braking_g
    for frac, want in ((0.0, 0.80), (0.95, 0.95)):
        params = dataclasses.replace(PARAMS, opponent_braking_fraction=frac)
        env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")
        env.reset(seed=0, options={"opponent": "slow", "gap_m": 30.0})
        assert env._opp is not None
        assert env._opp.driver._ax_brk == pytest.approx(want * g)
        assert env._opp.driver._ay < 0.81 * 9.80665 * EXAMPLE_CAR.max_lateral_g


def test_slower_opponents_can_accelerate_out_of_corners_as_we_do() -> None:
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, MARGIN_M)
    speeds = []
    for frac in (0.0, 0.95):
        params = dataclasses.replace(PARAMS, opponent_accel_fraction=frac)
        env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")
        env.reset(seed=0, options={"opponent": "slow", "gap_m": 30.0})
        assert env._opp is not None
        speeds.append(env._opp.driver.line_speed().copy())
    # Same corner speeds; quicker (or as quick) everywhere out of them.
    assert speeds[1].min() == pytest.approx(speeds[0].min(), rel=1e-3)
    assert (speeds[1] >= speeds[0] - 1e-6).all() and speeds[1].mean() > speeds[0].mean()
