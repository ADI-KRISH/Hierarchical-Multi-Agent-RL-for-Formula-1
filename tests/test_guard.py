"""The defending sparring driver (`f1rl.agents.guard`, ``guard_*`` opponents)."""

from dataclasses import replace
from pathlib import Path

import numpy as np

from f1rl.agents.guard import Guard, inside_maneuver
from f1rl.agents.tactical import load_tactical_config
from f1rl.agents.tactics import MOVES, corners_ahead
from f1rl.config import (
    EXAMPLE_CAR,
    OPPONENT_KINDS,
    GuardParams,
    RaceEnvParams,
    SimParams,
)
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.frenet import FrenetState
from f1rl.models.maneuver import Maneuver, shift_at
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

GUARDS = ("guard_slow", "guard_mid", "guard_equal")
CONFIG = Path(__file__).resolve().parents[1] / "configs" / "guard_sparring.yaml"


def _env(**overrides: object) -> TacticalEnv:
    params = RaceEnvParams(
        tactical=True,
        tires=True,
        opponents=GUARDS,
        safety_filter=True,
        opponent_yield=False,
        soft_follow=True,
        shield_in_tow=False,
    )
    params = replace(params, **overrides)  # type: ignore[arg-type]
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def _straight_start(env: TacticalEnv) -> float:
    """A point on the stadium where the next corner's braking point is ~250 m on."""
    env.reset(seed=0, options={"opponent": "guard_equal", "gap_m": 20.0})
    d = env._opp.driver  # type: ignore[union-attr]
    for s in np.arange(0.0, env._lap_m, 5.0):
        c = corners_ahead(d, float(s))
        if c and 240.0 < c[0].brake_m < 260.0:
            return float(s)
    raise AssertionError("no straight found")


def _state(env: TacticalEnv, s: float, extra_n: float = 0.0) -> FrenetState:
    d = env._opp.driver  # type: ignore[union-attr]
    n, psi, v = d.base.line_state(s)
    return FrenetState(s, n + extra_n, psi, v)


def test_guard_kinds_are_ahead_and_get_a_guard() -> None:
    for kind in GUARDS:
        assert OPPONENT_KINDS[kind][1] == "ahead"
        env = _env()
        env.reset(seed=0, options={"opponent": kind, "gap_m": 20.0})
        assert env._opp is not None and env._opp.guard is not None
    env.reset(seed=0, options={"opponent": "guard_equal", "gap_m": 20.0})
    assert env._opp is not None
    assert env._opp.driver.params.grip_fraction == OPPONENT_KINDS["guard_equal"][0]


def test_inside_move_goes_to_the_inside_and_back() -> None:
    env = _env()
    s0 = _straight_start(env)
    d = env._opp.driver  # type: ignore[union-attr]
    cs = corners_ahead(d, s0)
    m = inside_maneuver(d, s0, 0.0, cs[0], None, GuardParams(max_shift_m=7.0))
    if m is None:  # the stadium line may already sit inside: then nothing to close
        return
    p = GuardParams()
    at_brake = float(shift_at(m, np.array([s0 + cs[0].brake_m - p.settle_m]))[0])
    assert at_brake * cs[0].side > 0  # toward the corner's inside, never outward
    assert abs(float(shift_at(m, np.array([m.end_s + 10.0]))[0])) < 1e-9


def test_one_move_per_section_and_none_when_alongside() -> None:
    env = _env()
    s0 = _straight_start(env)
    d = env._opp.driver  # type: ignore[union-attr]
    g = Guard(GuardParams(reaction_s=0.0))
    me = _state(env, s0)
    line = Maneuver(s0, ((0.0, 0.0),))
    lap = env._lap_m
    # A car right behind (a dive threat), lined up: one move, then no more.
    car = replace(_state(env, s0 - 15.0), speed_ms=me.speed_ms + 5.0)
    moves = [g.update(d, me, car, lap, line, 5.5) for _ in range(3)]
    assert sum(m is not None for m in moves) <= 1
    assert g.stats["moves"] <= 1
    # A fresh guard with the car alongside: it never starts a move toward it.
    g2 = Guard(GuardParams(reaction_s=0.0))
    beside = _state(env, s0 - 2.0, extra_n=2.5)
    m = g2.update(d, me, beside, lap, line, 5.5)
    assert g2.stats["moves"] == 0
    if m is not None:  # only ever a leave-room plan: never closer sideways
        for ds in (15.0, 30.0, 45.0, 60.0):
            s = s0 + ds
            n = d.base.line_state(s)[0] + float(shift_at(m, np.array([s]))[0])
            assert beside.n_m - n >= min(GuardParams().room_m, 2.5) - 1e-6


def test_no_move_in_the_braking_zone_or_across_a_close_nose() -> None:
    env = _env()
    s0 = _straight_start(env)
    d = env._opp.driver  # type: ignore[union-attr]
    lap = env._lap_m
    line = Maneuver(s0, ((0.0, 0.0),))
    brake_s = s0 + corners_ahead(d, s0)[0].brake_m
    g = Guard(GuardParams(reaction_s=0.0))
    me = _state(env, brake_s - 10.0)  # inside the no-move zone
    assert g.update(d, me, _state(env, brake_s - 30.0), lap, line, 5.5) is None
    g = Guard(GuardParams(reaction_s=0.0))
    me = _state(env, s0)
    close = _state(env, s0 - 9.0)  # within a car length + min_move_gap_m
    assert g.update(d, me, close, lap, line, 5.5) is None


def test_battles_against_a_guard_have_no_contact() -> None:
    env = _env(commit_moves=True)
    rng = np.random.default_rng(0)
    for ep in range(6):
        env.reset(seed=ep, options={"opponent": GUARDS[ep % 3], "gap_m": 15.0})
        info: dict = {}
        for _ in range(40):
            _, _, terminated, truncated, info = env.step(int(rng.integers(len(MOVES))))
            if terminated or truncated:
                break
        assert not info["contact"]
        assert env._opp is not None and env._opp.guard is not None


def test_config_guard_block_parses() -> None:
    params = load_tactical_config(CONFIG).env_params()
    assert isinstance(params.guard, GuardParams)
    assert set(GUARDS) <= set(params.opponents)
