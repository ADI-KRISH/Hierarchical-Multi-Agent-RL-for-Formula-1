"""Evaluation entry point: ``uv run python -m f1rl.eval --agent <model.zip>``.

Drives a saved racing-line agent (SAC, `RaceEnv`) around real TUM tracks and scores
each flying lap against what the same physics allows:

- the **optimal line**: the minimum-curvature line's limit lap, the ceiling;
- the **scripted driver**: `LineFollower` on that line, the best a hand-written
  controller manages at the same 10 Hz (what a learned driver can be compared with);
- the **centreline limit**: the best any driver could do without using the width;
- the **v1 baseline**: the cautious rule-based centreline driver (standing start).

The agent's laps are deterministic (a flying lap from the line's own start state),
so one lap per track is the whole story; ``--episodes`` only sets how many laps the
(jittered) v1 baseline averages.

Real-world lap records are deliberately not in this table: the sim car has no drag
and 5 g of grip at every speed, so it laps 10-15% faster than a real car, and a
record comparison would measure the simulator, not the driver.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from f1rl.agents.baseline import BaselineDriver, run_episode
from f1rl.agents.line_follower import LineFollower
from f1rl.agents.racer import drive_lap, load_racer_config, resolve_track_names
from f1rl.config import (
    EXAMPLE_CAR,
    BaselineParams,
    DriverEnvParams,
    RaceEnvParams,
    SimParams,
)
from f1rl.envs.driver_env import DriverEnv
from f1rl.envs.race_env import RaceEnv
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_VALIDATION, load_tum_track
from f1rl.models.raceline import optimal_line


def scripted_lap(env: RaceEnv, driver: LineFollower) -> float | None:
    _, info = env.reset(seed=0)
    done = False
    while not done:
        _, _, terminated, truncated, info = env.step(driver.act(info))
        done = terminated or truncated
    lap = info.get("lap_time_s")
    return float(lap) if info["lap_completed"] and lap is not None else None


def baseline_lap(track_name: str, episodes: int) -> float | None:
    """Mean v1 baseline lap on the track's centreline (standing start)."""
    track = load_tum_track(track_name).to_track()
    env = DriverEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=0, max_episode_steps=20000),
        DriverEnvParams(action_repeat=1),
    )
    laps = []
    for k in range(episodes):
        driver = BaselineDriver(
            EXAMPLE_CAR, track, BaselineParams(), dt_s=env.sim.dt_s, seed=k
        )
        result = run_episode(env, driver, seed=k, telemetry_every=10**9)
        if result.completed and result.lap_time_s is not None:
            laps.append(result.lap_time_s)
    return float(np.mean(laps)) if len(laps) == episodes else None


def evaluate(
    model: SAC, track_names: list[str], episodes: int, params: RaceEnvParams
) -> list[dict[str, Any]]:
    rows = []
    margin = params.car_half_width_m + params.line_safety_m
    for name in track_names:
        track = load_tum_track(name)
        line = optimal_line(track, EXAMPLE_CAR, margin)
        sim = SimParams(seed=0, max_episode_steps=10_000)
        env = RaceEnv(EXAMPLE_CAR, track, sim, params, line, mode="lap")
        agent = drive_lap(model, env, seed=0)
        scripted = scripted_lap(env, LineFollower(EXAMPLE_CAR, track, line))
        rows.append(
            {
                "track": name,
                "centreline_limit_s": line.centre_lap_time_s,
                "optimal_line_s": line.lap_time_s,
                "scripted_s": scripted,
                "baseline_s": baseline_lap(name, episodes),
                "agent_s": agent["lap_time_s"] if agent["completed"] else None,
                "agent_off_track": agent["off_track"],
            }
        )
    return rows


def _fmt(value: float | None, ref: float | None = None) -> str:
    if value is None:
        return "     --"
    text = f"{value:6.2f}"
    return text + (f" ({100 * (value / ref - 1):+5.1f}%)" if ref else "")


def print_table(rows: list[dict[str, Any]], title: str) -> None:
    print(f"\n{title}  (lap times in s; % is the gap to the optimal line)")
    print(
        f"{'track':14} {'centreline':>10} {'optimal':>8} "
        f"{'scripted':>16} {'v1 baseline':>16} {'RL agent':>16}"
    )
    for r in rows:
        ref = r["optimal_line_s"]
        print(
            f"{r['track']:14} {r['centreline_limit_s']:10.2f} {ref:8.2f} "
            f"{_fmt(r['scripted_s'], ref):>16} {_fmt(r['baseline_s'], ref):>16} "
            f"{_fmt(r['agent_s'], ref):>16}"
        )
    done = [r for r in rows if r["agent_s"] is not None]
    print(f"agent completed {len(done)}/{len(rows)} tracks", end="")
    if done:
        gaps = [100 * (r["agent_s"] / r["optimal_line_s"] - 1) for r in done]
        print(f"; mean gap to the optimal line {np.mean(gaps):+.2f}%")
    else:
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a saved racing-line agent against the optimal line."
    )
    parser.add_argument(
        "--agent", type=Path, required=True, help="Path to a saved SB3 SAC model."
    )
    parser.add_argument(
        "--tracks",
        default="tum_held_out",
        help="tum_held_out | tum_validation | tum_train | tum_all | comma list.",
    )
    parser.add_argument(
        "--episodes", type=int, default=5, help="Laps the v1 baseline averages."
    )
    parser.add_argument("--out", type=Path, default=None, help="Write results JSON.")
    args = parser.parse_args()

    names: list[str] = []
    for part in args.tracks.split(","):
        names += resolve_track_names(part.strip())
    model = SAC.load(args.agent, device="cpu")
    # The env must be built exactly as the agent was trained: use the run's config.
    config_path = args.agent.parent / "config.yaml"
    params = (
        load_racer_config(config_path).env_params()
        if config_path.exists()
        else RaceEnvParams()
    )
    rows = evaluate(model, names, args.episodes, params)
    held = set(TUM_HELD_OUT)
    valid = set(TUM_VALIDATION)
    title = (
        "Held-out tracks (never trained on)"
        if set(names) <= held
        else "Validation tracks"
        if set(names) <= valid
        else "Tracks"
    )
    print_table(rows, title)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
