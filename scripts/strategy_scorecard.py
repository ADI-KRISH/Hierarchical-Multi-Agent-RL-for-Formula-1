"""Held-out scorecard per race case for Strategy Agent models.

usage: uv run python scripts/strategy_scorecard.py <config.yaml> [model.zip ...]
           [--races N] [--json out.json]

Runs the rule-based plan and every given model on the same seeded races on the
five held-out circuits, once per case (clean, spin, puncture, safety car, VSC),
and prints wins, mean place, stops and race time. Read-only: never trains.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.agents.strategy import (
    SPLITS,
    baseline_policy,
    load_strategy_config,
    load_tracks,
    model_policy,
)
from f1rl.driver_profile import load_profile
from f1rl.envs.strategy_env import StrategyEnv

CASES: dict[str, dict[str, str]] = {
    "clean": {"incident": "none", "neutral": "none"},
    "spin": {"incident": "spin", "neutral": "none"},
    "puncture": {"incident": "puncture", "neutral": "none"},
    "sc": {"incident": "none", "neutral": "sc"},
    "vsc": {"incident": "none", "neutral": "vsc"},
}
SEED = 70_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("models", nargs="*", type=Path)
    parser.add_argument("--races", type=int, default=100)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    from stable_baselines3 import PPO

    config = load_strategy_config(args.config)
    params = config.strategy_params()
    profile = load_profile(Path(config.run.driver_profile))
    tracks = SPLITS["held_out"]
    env = StrategyEnv(
        load_tracks(tracks, params, profile),
        params,
        baseline=config.baseline_params(),
        profile=profile,
    )
    policies: dict[str, Any] = {"baseline": baseline_policy}
    for m in args.models:
        policies[str(m)] = model_policy(PPO.load(m, device="cpu"))

    out: dict[str, dict[str, dict[str, float]]] = {}
    for case, opts in CASES.items():
        out[case] = {}
        for name, policy in policies.items():
            place, stops, time_s = [], [], []
            for i in range(args.races):
                obs, _ = env.reset(
                    seed=SEED + i, options={"track": tracks[i % len(tracks)], **opts}
                )
                done = False
                while not done:
                    obs, _, done, _, info = env.step(policy(env, obs))
                place.append(info["position"])
                stops.append(info["stops"])
                time_s.append(info["race_time_s"])
            p = np.array(place)
            out[case][name] = {
                "wins": float((p == 1).mean()),
                "place": float(p.mean()),
                "stops": float(np.mean(stops)),
                "race_time_s": float(np.mean(time_s)),
                "race_times_s": [float(t) for t in time_s],
                "places": [int(x) for x in p],
            }
            print(
                f"{case:9s} {name[-45:]:45s} wins {100 * (p == 1).mean():5.1f}%  "
                f"place {p.mean():.2f}  stops {np.mean(stops):.2f}  "
                f"time {np.mean(time_s):.1f}",
                flush=True,
            )
    if args.json:
        args.json.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
