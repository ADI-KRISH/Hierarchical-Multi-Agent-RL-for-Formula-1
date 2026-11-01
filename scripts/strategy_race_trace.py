"""Lap-by-lap trace of one held-out race for the rule-based plan and a model.

usage: uv run python scripts/strategy_race_trace.py <config.yaml> <model.zip>
           [--case clean|spin|puncture|sc|vsc] [--race I] [--out trace.json]

Runs the same seeded race (the I-th race of `strategy_scorecard.py`'s held-out
set) once with the rule-based plan and once with the model, and writes our car's
lap history (place, lap time, compound, tyre wear, pit, safety car) for both.
Read-only: never trains.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from strategy_scorecard import CASES, SEED

from f1rl.agents.strategy import (
    SPLITS,
    baseline_policy,
    load_strategy_config,
    load_tracks,
    model_policy,
)
from f1rl.driver_profile import load_profile
from f1rl.envs.strategy_env import StrategyEnv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("--case", default="sc", choices=sorted(CASES))
    parser.add_argument("--race", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("race_trace.json"))
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
    policies: dict[str, Any] = {
        "baseline": baseline_policy,
        "agent": model_policy(PPO.load(args.model, device="cpu")),
    }
    track = tracks[args.race % len(tracks)]
    out: dict[str, Any] = {"track": track, "case": args.case, "race": args.race}
    for name, policy in policies.items():
        obs, _ = env.reset(
            seed=SEED + args.race, options={"track": track, **CASES[args.case]}
        )
        done = False
        while not done:
            obs, _, done, _, info = env.step(policy(env, obs))
        out[name] = {
            "history": [
                {k: v for k, v in h.items() if not isinstance(v, float) or v == v}
                for h in env.history
            ],
            "position": info["position"],
            "race_time_s": info["race_time_s"],
            "stops": info["stops"],
        }
        print(name, track, "P", info["position"], "stops", info["stops"])
    args.out.write_text(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
