"""World and learning fixes for overtaking (configs/tactical_v4.yaml)."""

from dataclasses import replace

import numpy as np

from f1rl.agents.tactics import MOVES
from f1rl.config import EXAMPLE_CAR, GRAVITY_M_S2, RaceEnvParams, SimParams
from f1rl.envs.opponents import FOLLOW_GAP_M, YIELD_BRAKE, follow_brake
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

BRAKE = EXAMPLE_CAR.max_braking_g * GRAVITY_M_S2


def _env(**overrides: object) -> TacticalEnv:
    params = RaceEnvParams(
        tactical=True,
        tires=True,
        opponents=("equal", "slow", "attacker_equal"),
        safety_filter=True,
    )
    params = replace(params, **overrides)  # type: ignore[arg-type]
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def test_hard_follow_brake_is_the_old_rule() -> None:
    assert follow_brake(0.5, 10.0, 3.0, BRAKE, soft=False) == -YIELD_BRAKE


def test_soft_follow_brakes_only_as_much_as_closing_needs() -> None:
    gentle = follow_brake(0.5, FOLLOW_GAP_M, 1.0, BRAKE, soft=True)
    harder = follow_brake(0.5, FOLLOW_GAP_M, 8.0, BRAKE, soft=True)
    assert -YIELD_BRAKE < gentle < 0.0
    assert harder < gentle
    # Not closing and not too close: the pedal is left alone.
    assert follow_brake(0.5, FOLLOW_GAP_M, 0.0, BRAKE, soft=True) == 0.5


def test_tow_adds_acceleration_on_a_straight() -> None:
    env = _env(tow_accel_gain=0.3)
    env.reset(seed=0, options={"opponent": "equal", "gap_m": 30.0})
    env._tow = lambda rear, front: 0.0  # type: ignore[method-assign]
    env._update_grip()
    plain = env._car_now.max_accel_g
    # Force the tow on and recompute.
    env._tow = lambda rear, front: env.params.tow_boost_ms  # type: ignore[method-assign]
    env._update_grip()
    assert env._car_now.max_accel_g > plain * 1.25


def test_opponent_features_extend_the_observation() -> None:
    base = _env().reset(seed=0)[0]
    more = _env(opponent_features=True).reset(seed=0)[0]
    assert more.shape[0] == base.shape[0] + 4


def test_a_committed_move_runs_as_one_step() -> None:
    env = _env(commit_moves=True)
    env.reset(seed=0, options={"opponent": "equal", "gap_m": 25.0})
    t0 = env._steps
    env.step(MOVES.index("outside"))
    long_step = env._steps - t0
    env2 = _env(commit_moves=True)
    env2.reset(seed=0, options={"opponent": "equal", "gap_m": 25.0})
    t0 = env2._steps
    env2.step(MOVES.index("line"))
    assert env2._steps - t0 == env2.option_steps * env2.params.action_repeat
    if env._move != "line":  # the stadium had room for the move
        assert long_step > env2._steps - t0


def test_without_yield_scripted_cars_still_never_touch() -> None:
    env = _env(opponent_yield=False, soft_follow=True, shield_in_tow=False)
    rng = np.random.default_rng(0)
    for ep in range(6):
        kind = ("equal", "slow", "attacker_equal")[ep % 3]
        env.reset(seed=ep, options={"opponent": kind, "gap_m": 12.0})
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(int(rng.integers(len(MOVES))))
            if terminated or truncated:
                break
        assert not info["contact"]
