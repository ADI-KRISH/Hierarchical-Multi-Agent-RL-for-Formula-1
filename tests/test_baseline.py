import json

import pytest

from f1rl.agents.baseline import (
    BaselineDriver,
    evaluate_baseline,
    resolve_track,
    run_episode,
)
from f1rl.config import EXAMPLE_CAR, BaselineParams, SimParams
from f1rl.envs.driver_env import DriverEnv
from f1rl.envs.track import SYNTHETIC_TRACKS
from f1rl.models.lap import standing_start_lap_time_s

SIM = SimParams(seed=0)


def _lap(track_name: str, params: BaselineParams, seed: int = 0) -> float | None:
    track = SYNTHETIC_TRACKS[track_name]()
    env = DriverEnv(EXAMPLE_CAR, track, SIM)
    driver = BaselineDriver(EXAMPLE_CAR, track, params, SIM.dt_s, seed)
    result = run_episode(env, driver, seed)
    assert result.completed, f"baseline failed to finish {track_name}"
    return result.lap_time_s


@pytest.mark.parametrize("track_name", list(SYNTHETIC_TRACKS))
def test_baseline_completes_a_lap_slower_than_the_limit(track_name: str) -> None:
    lap = _lap(track_name, BaselineParams())
    limit = standing_start_lap_time_s(EXAMPLE_CAR, SYNTHETIC_TRACKS[track_name]())
    assert lap is not None
    assert limit < lap < limit * 1.10  # cautious, but not absurdly slow


@pytest.mark.parametrize("track_name", list(SYNTHETIC_TRACKS))
def test_baseline_at_full_margins_drives_the_limit_lap(track_name: str) -> None:
    """With no caution and no jitter, the script follows the phase-1 limit
    profile -- so the env's physics and the profile agree on what's possible.
    """
    perfect = BaselineParams(
        corner_grip_margin=1.0, braking_margin=1.0, margin_jitter_std=0.0
    )
    lap = _lap(track_name, perfect)
    limit = standing_start_lap_time_s(EXAMPLE_CAR, SYNTHETIC_TRACKS[track_name]())
    assert lap == pytest.approx(limit, rel=0.005)


def test_baseline_is_reproducible_per_seed_and_varies_across_seeds() -> None:
    params = BaselineParams()
    assert _lap("oval", params, seed=3) == _lap("oval", params, seed=3)
    assert _lap("oval", params, seed=3) != _lap("oval", params, seed=4)


def test_margin_jitter_stays_within_floor_and_the_cars_real_grip() -> None:
    wild = BaselineParams(corner_grip_margin=0.99, margin_jitter_std=0.5)
    for seed in range(20):
        driver = BaselineDriver(
            EXAMPLE_CAR, SYNTHETIC_TRACKS["oval"](), wild, SIM.dt_s, seed
        )
        for margin in (driver.corner_grip_margin, driver.braking_margin):
            assert wild.margin_floor <= margin <= 1.0


def test_evaluate_baseline_results_are_json_ready() -> None:
    results = evaluate_baseline(
        "oval", SYNTHETIC_TRACKS["oval"](), EXAMPLE_CAR, episodes=2, seed=0
    )
    text = json.dumps(results)
    assert json.loads(text)["summary"]["completed"] == 2
    assert results["summary"]["gap_to_limit_pct"] > 0.0
    deltas = [s["delta_s"] for s in results["episodes"][0]["sections"]]
    lap = results["episodes"][0]["lap_time_s"]
    gap = lap - results["summary"]["limit_lap_time_s"]
    assert sum(deltas) == pytest.approx(gap, abs=1e-6)


def test_resolve_track_rejects_unknown_names() -> None:
    with pytest.raises(ValueError):
        resolve_track("not-a-track")
