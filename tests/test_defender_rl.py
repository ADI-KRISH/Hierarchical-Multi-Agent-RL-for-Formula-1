"""Learned-defender plumbing: restricted moves, defending rules, tactician rival."""

from dataclasses import replace

import numpy as np
import pytest

from f1rl.agents.defender_rl import defend_action
from f1rl.agents.tactics import MOVES
from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

DEFEND = ("line", "cover", "inside")


def _env(**overrides: object) -> TacticalEnv:
    params = RaceEnvParams(
        tactical=True,
        tires=True,
        opponents=("rival_tactician", "rival_tactician_fast"),
        safety_filter=True,
        opponent_yield=False,
        soft_follow=True,
        shield_in_tow=False,
        commit_moves=True,
        tactical_moves=DEFEND,
        defend_rules=True,
    )
    params = replace(params, **overrides)  # type: ignore[arg-type]
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def test_action_space_is_the_defending_moves() -> None:
    env = _env()
    assert env.action_space.n == 3  # type: ignore[attr-defined]
    obs, _ = env.reset(seed=0, options={"opponent": "rival_tactician", "gap_m": 25.0})
    env.step(DEFEND.index("cover"))
    assert env._move in DEFEND
    # The observation keeps the full move one-hot (same layout as the main driver).
    assert (
        obs.shape == _env(tactical_moves=(), defend_rules=False).reset(seed=0)[0].shape
    )


def test_unknown_moves_are_rejected() -> None:
    with pytest.raises(ValueError):
        _env(tactical_moves=("line", "teleport"))


def test_no_cover_across_a_close_nose() -> None:
    env = _env()
    env.reset(seed=0, options={"opponent": "rival_tactician", "gap_m": 8.0})
    assert not env._may_change([], "cover")
    assert env._may_change([], "line")


def test_tactician_drives_the_car_behind_and_nobody_crashes() -> None:
    env = _env()
    rng = np.random.default_rng(0)
    seen: set[str] = set()
    for ep in range(4):
        kind = ("rival_tactician", "rival_tactician_fast")[ep % 2]
        env.reset(seed=ep, options={"opponent": kind, "gap_m": 15.0})
        info: dict = {}
        for _ in range(40):
            _, _, term, trunc, info = env.step(int(rng.integers(3)))
            seen.add(env._opp_move)
            if term or trunc:
                break
        assert not info["contact"]
    assert seen & {"divebomb", "dummy", "outside", "switchback"}  # it attacks


def test_rule_based_defend_action_is_a_defending_move() -> None:
    env = _env()
    env.reset(seed=0, options={"opponent": "rival_tactician", "gap_m": 15.0})
    assert env.moves[defend_action(env)] in DEFEND
    assert all(m in MOVES for m in DEFEND)
