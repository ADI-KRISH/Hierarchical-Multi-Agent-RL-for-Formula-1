"""Track sections (`f1rl.sections`) and section battles in `RaceEnv`."""

from pathlib import Path

import numpy as np

from f1rl.agents.tactical import (
    MultiTrackTactical,
    load_tactical_config,
    section_battle,
    section_won,
)
from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SectionParams, SimParams
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from f1rl.sections import KINDS, env_sections, section_options, track_sections
from racetrack_fixtures import stadium_track

CONFIG = Path(__file__).resolve().parents[1] / "configs" / "overtake_sections.yaml"


def _env() -> TacticalEnv:
    params = load_tactical_config(CONFIG).env_params()
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def _profile() -> tuple[np.ndarray, np.ndarray]:
    """A 2 km synthetic lap: two corners (speed dips) on a 5 m grid."""
    s = np.arange(400) * 5.0
    v = np.full(400, 90.0)
    kappa = np.zeros(400)
    for apex, side in ((500.0, 1.0), (1500.0, -1.0)):
        dip = np.clip(1.0 - np.abs(s - apex) / 200.0, 0.0, None)
        v -= 50.0 * dip
        kappa += side * 0.02 * dip
    return v, kappa


def test_sections_cover_each_corner_in_order() -> None:
    v, kappa = _profile()
    secs = track_sections("synthetic", v, kappa, 5.0, 100.0)
    assert [s.kind for s in secs] == list(KINDS) * 2
    for sec in secs:
        assert 0.0 <= sec.start_s < 2000.0
        assert sec.length_m >= SectionParams().min_length_m
    first, second = secs[:4], secs[4:]
    assert all(s.side == 1.0 for s in first) and all(s.side == -1.0 for s in second)
    assert abs(first[0].apex_s - 500.0) < 5.0
    # The straight ends at the braking point; the braking zone ends past the apex.
    straight, braking, corner_in, corner_out = first
    assert abs((straight.start_s + straight.length_m) % 2000 - straight.brake_s) < 1e-6
    assert braking.start_s + braking.length_m > braking.apex_s
    assert corner_in.start_s > braking.start_s
    # Corner battles start half alongside, on the inside (left here) or outside.
    assert corner_in.shift_m > 0 > corner_out.shift_m
    assert straight.shift_m == 0.0 and 0.0 < corner_in.gap_m < 5.5


def test_short_sections_are_dropped() -> None:
    v, kappa = _profile()
    secs = track_sections(
        "synthetic", v, kappa, 5.0, 100.0, SectionParams(min_length_m=900.0)
    )
    assert all(s.length_m >= 900.0 for s in secs)
    assert len(secs) < 8


def test_no_corners_no_sections() -> None:
    v = np.full(100, 90.0)
    assert track_sections("oval", v, np.zeros(100), 5.0, 100.0) == []


def test_section_battle_ends_when_behind_at_the_end() -> None:
    env = _env()
    secs = env_sections(env)
    assert secs
    sec = secs[0]
    # The line policy never makes a move: behind at the end, the battle is over.
    r = section_battle(env, "line", sec, "slow", 9.0)
    assert not r["contact"]
    assert not r["won"]
    progress = env._state.s_m - env._start_s
    assert progress < sec.length_m + 50.0


def test_section_start_is_no_faster_than_the_car_ahead() -> None:
    env = _env()
    for sec in env_sections(env):
        env.reset(seed=0, options=section_options(sec, "slow", 8.0))
        assert env._opp is not None
        assert env._state.speed_ms <= env._opp.state.speed_ms + 1e-9


def test_section_won_needs_a_stuck_clean_pass() -> None:
    base = {"section_won": True, "passes": 1, "contact": False, "off_track": False}
    assert section_won(base)
    assert not section_won({**base, "passes": 0})
    assert not section_won({**base, "section_won": False})
    assert not section_won({**base, "contact": True})


def test_without_section_nothing_changes() -> None:
    params = RaceEnvParams(tactical=True, tires=True, safety_filter=True)
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    env = TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")
    _, info = env.reset(seed=0, options={"opponent": "slow", "gap_m": 20.0})
    assert not info["section_done"] and env._section_m is None


def test_multitrack_draws_sections_and_tracks_wins() -> None:
    config = load_tactical_config(CONFIG)
    env = _env()
    secs = {"stadium": env_sections(env)}
    multi = MultiTrackTactical({"stadium": env}, 0, sections=secs, run=config.run)
    multi.reset()
    assert env._section_m is not None  # section_frac 1.0: every episode is one
    for _ in range(500):
        *_, term, trunc, _ = multi.step(0)
        if term or trunc:
            break
    assert multi.section_wins  # the result was recorded for that section
