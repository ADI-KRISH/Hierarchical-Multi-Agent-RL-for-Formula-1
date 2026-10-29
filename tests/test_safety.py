import dataclasses

import numpy as np
import pytest

from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.race_env import RaceEnv
from f1rl.models.frenet import FrenetState, map_action
from f1rl.models.raceline import optimal_line
from f1rl.models.safety import (
    SafetyCheck,
    backup,
    candidates,
    fallbacks,
    rollout,
    safe_command,
    separation,
    wrap,
)
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


def _straight(_s: float) -> float:
    return 0.0


def _wide(_s: float) -> tuple[float, float]:
    return 6.0, 6.0


def _filter(
    me: FrenetState, cmd: tuple[float, float], other: FrenetState
) -> tuple[tuple[float, float], bool]:
    return safe_command(
        EXAMPLE_CAR,
        me,
        cmd,
        EXAMPLE_CAR,
        other,
        (0.0, 0.0),
        _straight,
        _wide,
        CHECK,
    )


def test_gaps_fold_round_the_lap() -> None:
    assert wrap(4990.0, 5000.0) == pytest.approx(-10.0)
    assert wrap(10.0, 5000.0) == pytest.approx(10.0)


def test_cars_side_by_side_with_room_are_clear() -> None:
    a = FrenetState(100.0, -2.0, 0.0, 60.0)
    b = FrenetState(100.0, 2.0, 0.0, 60.0)
    assert separation(a, b, CHECK) > 0.0
    assert separation(a, FrenetState(103.0, -1.5, 0.0, 60.0), CHECK) < 0.0


def test_clean_racing_is_left_alone() -> None:
    me = FrenetState(100.0, 0.0, 0.0, 70.0)
    ahead = FrenetState(200.0, 0.0, 0.0, 70.0)  # 100 m clear at the same speed
    assert _filter(me, (1.0, 0.0), ahead) == ((1.0, 0.0), False)


def test_a_car_closing_fast_from_behind_is_made_to_brake() -> None:
    me = FrenetState(100.0, 0.0, 0.0, 80.0)
    # 30 m/s slower, 25 m up the road: still time to brake, so flat out is allowed
    assert _filter(me, (1.0, 0.0), FrenetState(125.0, 0.0, 0.0, 50.0))[1] is False
    ahead = FrenetState(119.0, 0.0, 0.0, 50.0)  # 19 m: braking must start now
    (u_long, _), hit = _filter(me, (1.0, 0.0), ahead)
    assert hit and u_long < 0.0
    worst, edge = rollout(
        EXAMPLE_CAR,
        me,
        map_action(EXAMPLE_CAR, u_long, 0.0),
        EXAMPLE_CAR,
        ahead,
        (0.0, 0.0),
        _straight,
        _wide,
        CHECK,
    )
    assert worst > 0.0 and edge > 0.0


def test_a_car_alongside_steering_in_is_turned_away() -> None:
    me = FrenetState(100.0, 1.2, 0.0, 60.0)
    beside = FrenetState(100.0, -1.2, 0.0, 60.0)  # just clear, to our right
    (_, steer), hit = _filter(me, (0.0, -1.0), beside)  # full lock into it
    assert hit and steer > -1.0


def test_the_car_behind_brakes_first_the_car_alongside_steers_first() -> None:
    me = FrenetState(100.0, 0.0, 0.0, 60.0)
    behind_first = candidates((0.5, 0.0), me, FrenetState(120.0, 0.0, 0.0, 50.0), CHECK)
    assert behind_first[0][0] < 0.0 and behind_first[0][1] == 0.0
    beside_first = candidates(
        (0.5, 0.0), me, FrenetState(101.0, -2.0, 0.0, 60.0), CHECK
    )
    assert beside_first[0] == (0.5, CHECK.steer)  # away from it: to the left


def test_braking_never_takes_the_grip_the_steering_needs() -> None:
    me = FrenetState(100.0, 0.0, 0.0, 60.0)
    for u_long, u_lat in candidates(
        (0.0, 0.9), me, FrenetState(120.0, 0.0, 0.0, 40.0), CHECK
    ):
        assert u_long**2 + u_lat**2 <= 1.0 + 1e-9


def test_a_car_squeezed_against_the_edge_backs_off_instead_of_going_off() -> None:
    me = FrenetState(100.0, -4.5, 0.0, 60.0)  # near the right edge
    beside = FrenetState(103.0, -2.2, 0.0, 60.0)  # half a car ahead, moving across
    other_accel = map_action(EXAMPLE_CAR, 0.0, -0.6)

    def narrow(_s: float) -> tuple[float, float]:
        return 5.0, 5.0

    cmd, hit = safe_command(
        EXAMPLE_CAR,
        me,
        (0.0, 0.0),
        EXAMPLE_CAR,
        beside,
        other_accel,
        _straight,
        narrow,
        CHECK,
    )
    assert hit and cmd[0] < 0.0  # it brakes: there is no room to its right
    assert cmd[1] >= -CHECK.steer  # and is not steered off the road


def test_a_car_braking_with_one_close_behind_first_tries_braking_less() -> None:
    me = FrenetState(110.0, 0.0, 0.0, 40.0)
    behind = FrenetState(100.0, 0.0, 0.0, 50.0)
    first = candidates((-1.0, 0.2), me, behind, CHECK)[0]
    assert first == (-0.5, 0.2)  # no brake-testing the car behind


def test_the_assumed_fallback_is_always_a_command_the_filter_can_pick() -> None:
    me = FrenetState(100.0, 1.0, 0.08, 60.0)  # heading across the track
    ahead = FrenetState(115.0, 0.0, 0.0, 55.0)
    cmds = fallbacks(EXAMPLE_CAR, me, (0.5, 0.0), ahead, 0.0, CHECK)
    assert len(cmds) == 1  # behind: keeping the lane means braking anyway
    u_long, u_lat = cmds[0]
    assert u_long < 0.0 and u_lat < 0.0  # brakes and turns back parallel
    assert u_long**2 + u_lat**2 <= 1.0 + 1e-9


def test_the_car_in_front_drives_to_its_plan_not_to_a_stop() -> None:
    me = FrenetState(100.0, 0.0, 0.0, 50.0)
    braking = -EXAMPLE_CAR.max_braking_g * 9.81
    a_x, _ = backup(EXAMPLE_CAR, me, braking, False, 0.0, CHECK)
    assert a_x == braking  # no plan known: it keeps its pedal

    a_x, _ = backup(EXAMPLE_CAR, me, braking, False, 0.0, CHECK, lambda s: 60.0)
    assert a_x > 0.0  # its plan is faster: it drives on

    def corner(s: float) -> float:
        return 50.0 - 0.8 * (s - 100.0)  # a braking zone

    a_x, _ = backup(EXAMPLE_CAR, me, 5.0, False, 0.0, CHECK, corner)
    assert a_x < -30.0  # it brakes for the corner as planned


def test_a_far_off_car_is_not_braked_for_running_wide() -> None:
    me = FrenetState(100.0, 3.0, 0.1, 60.0)  # heading for the left edge
    far = FrenetState(60.0, 0.0, 0.0, 60.0)  # 40 m behind
    cmd, hit = _filter(me, (1.0, 0.0), far)
    assert not hit and cmd == (1.0, 0.0)  # staying on the road is its own job


def test_with_no_room_on_the_road_a_car_runs_off_rather_than_crash() -> None:
    check = dataclasses.replace(CHECK, hold_s=0.5)
    me = FrenetState(100.0, 0.0, 0.0, 60.0)
    slow = FrenetState(130.0, 0.0, 0.0, 10.0)  # too close to stop for

    def narrow(_s: float) -> tuple[float, float]:
        return 1.5, 1.5

    def outcome(cmd: tuple[float, float]) -> tuple[float, float]:
        return rollout(
            EXAMPLE_CAR,
            me,
            map_action(EXAMPLE_CAR, *cmd),
            EXAMPLE_CAR,
            slow,
            (0.0, 0.0),
            _straight,
            narrow,
            check,
        )

    assert outcome((-1.0, 0.0))[0] < 0.0  # braking alone would hit it
    cmd, hit = safe_command(
        EXAMPLE_CAR,
        me,
        (1.0, 0.0),
        EXAMPLE_CAR,
        slow,
        (0.0, 0.0),
        _straight,
        narrow,
        check,
    )
    worst, edge = outcome(cmd)
    assert hit and worst > 0.0 and edge < 0.0  # it swerves off the road, clear


def _race_env(safety: bool) -> RaceEnv:
    params = RaceEnvParams(
        tires=True,
        opponents=("slow", "equal", "defender", "attacker"),
        opponent_gap_m=(8.0, 20.0),
        safety_filter=safety,
    )
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return RaceEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line)


def _contacts(env: RaceEnv, episodes: int) -> int:
    rng = np.random.default_rng(0)
    hits = 0
    for ep in range(episodes):
        env.reset(seed=ep)
        for _ in range(150):
            # a reckless driver: flat out, swerving at random
            action = np.array([1.0, rng.uniform(-0.6, 0.6)], dtype=np.float32)
            _, _, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break
        hits += info["contact"]
    return hits


@pytest.mark.slow
def test_with_the_filter_even_a_reckless_driver_never_hits_anyone() -> None:
    assert _contacts(_race_env(safety=False), 20) > 0  # the test can fail...
    assert _contacts(_race_env(safety=True), 20) == 0  # ...and the filter stops it


def test_lift_first_lifts_before_braking_hard() -> None:
    me = FrenetState(100.0, 0.0, 0.0, 75.0)
    ahead = FrenetState(110.0, 0.0, 0.0, 60.0)  # 10 m ahead, 15 m/s slower
    lift = dataclasses.replace(CHECK, lift_first=True)
    assert candidates((1.0, 0.0), me, ahead, lift)[:2] == [(0.0, 0.0), (-0.3, 0.0)]

    def run(check: SafetyCheck) -> tuple[tuple[float, float], bool]:
        return safe_command(
            EXAMPLE_CAR,
            me,
            (1.0, 0.0),
            EXAMPLE_CAR,
            ahead,
            (0.0, 0.0),
            _straight,
            _wide,
            check,
        )

    assert run(CHECK) == ((-0.8, 0.0), True)  # earlier runs: a hard brake
    assert run(lift) == ((-0.3, 0.0), True)  # a light brake is enough


def test_with_nothing_clear_a_car_takes_the_most_room_even_off_the_road() -> None:
    # Side by side, the other car squeezing in: no command keeps clear, and the
    # ones that keep furthest apart leave the road. A crash is worse than running
    # wide, so the filter takes the most room, not the command.
    check = dataclasses.replace(CHECK, hold_s=0.5)
    me = FrenetState(100.0, -0.9, 0.0, 50.0)
    other = FrenetState(101.0, 1.0, -0.1, 50.0)  # alongside, turning onto us

    def narrow(_s: float) -> tuple[float, float]:
        return 1.5, 1.5

    def judge(cmd: tuple[float, float]) -> tuple[float, float]:
        return rollout(
            EXAMPLE_CAR,
            me,
            map_action(EXAMPLE_CAR, *cmd),
            EXAMPLE_CAR,
            other,
            map_action(EXAMPLE_CAR, 0.0, -1.0),
            _straight,
            narrow,
            check,
        )

    command = (0.0, 0.0)
    clear = min(0.0, separation(me, other, check))
    tries = candidates(command, me, other, check)
    tries += fallbacks(EXAMPLE_CAR, me, command, other, 0.0, check)
    on_road = [judge(c)[0] for c in tries if judge(c)[1] > min(0.0, judge(command)[1])]
    assert judge(command)[0] < clear and all(sep < clear for sep in on_road)
    cmd, hit = safe_command(
        EXAMPLE_CAR,
        me,
        command,
        EXAMPLE_CAR,
        other,
        map_action(EXAMPLE_CAR, 0.0, -1.0),
        _straight,
        narrow,
        check,
    )
    assert hit
    assert judge(cmd)[0] > max([judge(command)[0], *on_road])
