"""Sim fixes for on-track overtaking at race-like pace gaps
(configs/overtake_realpace.yaml): wake vs thrust, DRS alongside, the car ahead
keeps its pace."""

from dataclasses import replace

import pytest

from f1rl.config import EXAMPLE_CAR, OPPONENT_KINDS, RaceEnvParams, SimParams
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.frenet import FrenetState
from f1rl.models.raceline import optimal_line
from f1rl.models.safety import SafetyCheck, safe_command
from racetrack_fixtures import stadium_track

CHECK = SafetyCheck(
    lap_m=5000.0,
    car_length_m=5.5,
    car_width_m=2.0,
    margin_s_m=1.0,
    margin_n_m=0.3,
    horizon_s=1.5,
    max_horizon_s=2.0,
    dt_s=0.05,
    hold_s=0.1,
    lane_time_s=0.5,
    speed_floor_ms=10.0,
    steer=0.4,
    brake=0.8,
)


def _env(**overrides: object) -> TacticalEnv:
    params = RaceEnvParams(tactical=True, tires=True, opponents=("equal",))
    params = replace(params, **overrides)  # type: ignore[arg-type]
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def _in_wake(env: TacticalEnv) -> None:
    """Our car 10 m straight behind the other one (inside the wake)."""
    assert env._opp is not None
    me = env._state
    env._opp.state = replace(me, s_m=me.s_m + 10.0)
    env._update_grip()


def test_the_wake_cuts_thrust_only_in_earlier_runs() -> None:
    old = _env()
    old.reset(seed=0, options={"opponent": "equal", "gap_m": 30.0})
    _in_wake(old)
    new = _env(dirty_air_traction=False)
    new.reset(seed=0, options={"opponent": "equal", "gap_m": 30.0})
    _in_wake(new)
    loss = 1.0 - old.params.dirty_air_loss
    assert old._grip == pytest.approx(new._grip)  # cornering grip: both lose it
    assert old._car_now.max_lateral_g == pytest.approx(new._car_now.max_lateral_g)
    assert old._car_now.max_braking_g == pytest.approx(new._car_now.max_braking_g)
    assert new._car_now.max_accel_g == pytest.approx(old._car_now.max_accel_g / loss)


def test_drs_stays_open_alongside_until_a_car_length_past() -> None:
    for alongside, want in ((False, 0.0), (True, 6.0)):
        env = _env(tow_while_alongside=alongside, tow_boost_ms=6.0)
        env.reset(seed=0)
        env._kappa_at = lambda s: 0.0  # type: ignore[method-assign,assignment]
        rear = env._state
        level = replace(rear, s_m=rear.s_m + 2.0, n_m=rear.n_m + 3.0)
        assert env._tow(rear, level) == want
        assert env._tow(rear, replace(rear, s_m=rear.s_m + 30.0)) == 6.0
        assert env._tow(rear, replace(rear, s_m=rear.s_m - 8.0)) == 0.0  # past


def test_car_ahead_keeps_its_command_with_a_car_closing_behind() -> None:
    ahead = FrenetState(1007.0, 0.0, 0.0, 60.0)
    behind = FrenetState(1000.0, 0.0, 0.0, 80.0)  # 7 m back, closing at 20 m/s
    flat_out = (1.0, 0.0)

    def run(lead: FrenetState, check: SafetyCheck) -> tuple[float, float]:
        cmd, _ = safe_command(
            EXAMPLE_CAR,
            lead,
            flat_out,
            EXAMPLE_CAR,
            behind,
            (1.0, 0.0),
            lambda s: 0.0,
            lambda s: (6.0, 6.0),
            check,
        )
        return cmd

    # Earlier runs: the car ahead brakes and swerves for the car running into it.
    assert run(ahead, CHECK)[0] < 0.0
    # Now the car behind must avoid it (and is filtered first in RaceEnv).
    assert run(ahead, replace(CHECK, leader_keeps_pace=True)) == flat_out
    # Overlapping (the other car's nose beside it): still filtered as before.
    beside = replace(ahead, s_m=1004.0)
    assert run(beside, replace(CHECK, leader_keeps_pace=True)) == run(beside, CHECK)


def test_race_like_opponents_are_between_slow_and_equal() -> None:
    slow, equal = OPPONENT_KINDS["slow"][0], OPPONENT_KINDS["equal"][0]
    for kind in ("guard_close", "close", "mid"):
        grip, where = OPPONENT_KINDS[kind]
        assert slow < grip < equal and where == "ahead"


def test_lane_commit_runs_a_lane_move_until_the_attack_ends() -> None:
    from f1rl.agents.tactics import LANE_MOVES

    runs = []
    for commit in (False, True):
        env = _env(lane_moves=True, lane_commit=commit, safety_filter=True)
        env.reset(seed=0, options={"opponent": "equal", "gap_m": 12.0})
        assert env._opp is not None  # same speed: it stays within reach
        env._state = replace(env._state, speed_ms=env._opp.state.speed_ms)
        t0 = env._steps
        env.step(env.moves.index(LANE_MOVES[0]))
        runs.append(env._steps - t0)
    one = runs[0]
    assert runs[1] > one  # several decisions' worth in one step
    env = _env(lane_moves=True, lane_commit=True)
    env.reset(seed=0, options={"opponent": "equal", "gap_m": 12.0})
    assert env._lane_goes_on()
    env.params = replace(env.params, lane_commit_gap_m=5.0)
    assert not env._lane_goes_on()  # dropped too far back: given up
