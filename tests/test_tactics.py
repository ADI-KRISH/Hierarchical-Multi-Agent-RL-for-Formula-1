import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from f1rl.agents.path_driver import PathDriver
from f1rl.agents.tactics import MOVES, corners_ahead, make_move
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
