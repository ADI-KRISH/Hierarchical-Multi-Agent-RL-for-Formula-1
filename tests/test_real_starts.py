import json
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import (
    EXAMPLE_CAR,
    OPPONENT_KINDS,
    RaceEnvParams,
    RealStartParams,
    SimParams,
)
from f1rl.envs.opponents import wrap_gap
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from f1rl.real_starts import by_track, load_real_starts, starts_from_race
from racetrack_fixtures import stadium_track

RACE: dict[str, Any] = {
    "year": 2021,
    "grand_prix": "Test Grand Prix",
    "tum_track": "Monza",
    "split": "train",
    "passes": [
        {
            "lap": 6,
            "attacker": "AAA",
            "defender": "BBB",
            "gap_before_s": 0.5,
            "attacker_tyre_life": 3.0,
            "defender_tyre_life": 20.0,  # 17 laps older: a clearly slower car
            "location_m": 900.0,
        },
        {
            "lap": 9,
            "attacker": "CCC",
            "defender": "DDD",
            "gap_before_s": 0.8,
            "attacker_tyre_life": 10.0,
            "defender_tyre_life": float("nan"),
            "location_m": 2000.0,
        },
        {"lap": 12, "attacker": "E", "defender": "F", "gap_before_s": 0.3},  # nowhere
    ],
    "defences": [
        {
            "first_lap": 20,
            "last_lap": 25,
            "attacker": "GGG",
            "defender": "HHH",
            "min_gap_s": 0.4,
            "outcome": "held",
        }
    ],
}


def test_each_pass_gives_an_attack_and_a_defence_before_the_spot() -> None:
    starts = starts_from_race(RACE, RealStartParams(lead_m=400.0, fresh_tyre_laps=8))
    first = [s for s in starts if "AAA>BBB" in s.source]
    assert {s.role for s in first} == {"attack", "defend"}
    assert all(s.start_s_m == 500.0 and s.gap_s == 0.5 for s in first)
    attack = next(s for s in first if s.role == "attack")
    defend = next(s for s in first if s.role == "defend")
    assert attack.opponent == "slow" and attack.real_outcome == "passed"
    assert defend.opponent == "attacker" and defend.real_outcome == "lost"
    second = next(s for s in starts if "CCC>DDD" in s.source and s.role == "attack")
    assert second.opponent == "defender"  # unknown tyre age: an equal-pace car


def test_a_pass_with_no_place_is_skipped_and_defences_start_at_the_line() -> None:
    starts = starts_from_race(RACE)
    assert not [s for s in starts if "E>F" in s.source]
    held = [s for s in starts if "HHH vs GGG" in s.source]
    assert {s.role for s in held} == {"attack", "defend"}
    assert all(s.start_s_m == 0.0 and s.real_outcome == "held" for s in held)
    assert all(s.opponent in OPPONENT_KINDS for s in starts)


def test_races_with_no_sim_track_give_nothing() -> None:
    assert starts_from_race({**RACE, "tum_track": None}) == []


def test_the_library_loads_in_file_order_and_splits_by_track(tmp_path: Path) -> None:
    (tmp_path / "b.json").write_text(json.dumps({**RACE, "split": "held_out"}))
    (tmp_path / "a.json").write_text(json.dumps(RACE))
    starts = load_real_starts(tmp_path)
    assert len(starts) == 2 * len(starts_from_race(RACE))
    assert starts[0].split == "train" and starts[-1].split == "held_out"
    assert set(by_track(starts, ["train"])) == {"Monza"}
    assert len(by_track(starts, ["train"])["Monza"]) == len(starts) // 2


def test_a_real_start_puts_both_cars_where_the_data_says() -> None:
    params = RaceEnvParams(
        tactical=True, tires=True, opponents=("slow",), safety_filter=True
    )
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    env = TacticalEnv(
        EXAMPLE_CAR, track, SimParams(seed=0), params, line, mode="random"
    )
    start = starts_from_race(RACE)[0]
    _, info = env.reset(seed=0, options={**start.options(), "push": 1.0})
    assert info["distance_m"] == np.float64(500.0)
    assert env._opp is not None and env._opp.kind == "slow"
    gap = wrap_gap(env._opp.state.s_m - env._state.s_m, env._lap_m)
    lo, hi = params.real_gap_m
    assert lo <= gap <= hi  # 0.5 s at the line's speed, kept in range
    for _ in range(40):
        _, _, terminated, truncated, info = env.step(2)  # divebomb, every time
        if terminated or truncated:
            break
    assert not info["contact"]
