"""Defend battles for the main driver: our car ahead, an attacker behind."""

import dataclasses
from pathlib import Path

import numpy as np

from f1rl.agents.tactical import (
    MultiTrackTactical,
    defend_options,
    load_tactical_config,
    section_battle,
    section_won,
)
from f1rl.config import EXAMPLE_CAR, SimParams
from f1rl.envs.opponents import wrap_gap
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from f1rl.sections import env_sections
from racetrack_fixtures import stadium_track

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "defend_sections.yaml"


def _env() -> TacticalEnv:
    params = load_tactical_config(CONFIG).env_params()
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def _info(**kw: object) -> dict[str, object]:
    base = {
        "section_won": True,
        "passes": 0,
        "contact": False,
        "off_track": False,
        "dirty_steps": 0,
        "defending": True,
    }
    return {**base, **kw}


def test_a_defence_is_held_without_a_pass_of_our_own() -> None:
    assert section_won(_info())
    assert not section_won(_info(passes=-1))
    assert not section_won(_info(section_won=False))
    assert not section_won(_info(dirty_steps=1))
    # An attack still needs a pass that stuck.
    assert not section_won(_info(defending=False))
    assert section_won(_info(defending=False, passes=1))


def test_defend_start_puts_the_attacker_behind_and_wears_our_tyres() -> None:
    env = _env()
    sec = next(s for s in env_sections(env) if s.kind == "braking")
    _, info = env.reset(
        seed=0, options=defend_options(sec, "rival_tactician", 10.0, 0.4)
    )
    assert info["defending"]
    assert env._opp is not None
    gap = wrap_gap(env._opp.state.s_m - env._state.s_m, env._lap_m)
    assert -11.0 < gap < -9.0
    assert abs(info["wear"] - 0.4) < 1e-9


def test_with_a_car_behind_only_defending_moves_are_taken() -> None:
    env = _env()
    sec = next(s for s in env_sections(env) if s.kind == "straight")
    env.reset(seed=0, options=defend_options(sec, "rival_tactician", 15.0))
    lane = next(m for m in env.moves if m.startswith("left"))
    env.step(env.moves.index(lane))
    assert env._move == "line"
    assert env.refused == 1


def test_a_frozen_main_driver_attacks_from_behind() -> None:
    env = _env()
    picks: list[int] = []

    def attacker(obs: np.ndarray) -> int:
        assert obs.shape == env.observation_space.shape  # full layout
        picks.append(1)
        return env.moves.index("pull_in")

    env.fixed_attacker = attacker
    sec = next(s for s in env_sections(env) if s.kind == "straight")
    r = section_battle(env, "line", sec, "rival_behind", 8.0)
    assert picks
    assert not r["contact"]


def test_defend_battles_never_crash() -> None:
    env = _env()
    for sec in [s for s in env_sections(env) if s.kind in ("straight", "braking")]:
        for kind in ("rival_tactician_fast", "attacker"):
            for policy in ("line", "tactician"):
                r = section_battle(env, policy, sec, kind, 8.0, 0.5)
                assert not r["contact"], (sec.name, kind, policy)


def test_training_mix_draws_defences() -> None:
    config = load_tactical_config(CONFIG)
    run = dataclasses.replace(
        config.run,
        section_defend_frac=1.0,
        section_attackers=["rival_tactician"],
    )
    env = _env()
    secs = {"stadium": env_sections(env)}
    multi = MultiTrackTactical({"stadium": env}, 0, sections=secs, run=run)
    for _ in range(3):
        _, info = multi.reset()
        assert info["defending"]
        assert env._opp is not None and env._opp.kind == "rival_tactician"


def test_driving_the_racing_line_ahead_is_never_a_chop() -> None:
    env = _env()
    assert env.params.chop_line_relative
    assert env._line_n(100.0) == env._driver().base.line_state(100.0)[0]
    for sec in [s for s in env_sections(env) if s.kind in ("straight", "braking")]:
        for kind in ("rival_tactician", "attacker"):
            r = section_battle(env, "line", sec, kind, 6.5)
            assert r["dirty_chop"] == 0, (sec.name, kind)


def test_defence_labels_keep_own_choice_when_nobody_is_behind() -> None:
    from f1rl.agents.tactical import DEFEND_MOVES, defence_label

    env = _env()
    sec = next(s for s in env_sections(env) if s.kind == "straight")
    env.reset(seed=0, options=defend_options(sec, "rival_tactician", 15.0))
    assert env.moves[defence_label(env, 3)] in DEFEND_MOVES
    env.reset(
        seed=0, options={"start_s": sec.start_s, "opponent": "slow", "gap_m": 15.0}
    )
    assert defence_label(env, 3) == 3
