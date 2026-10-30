"""PPO for `StrategyEnv` (the Strategy Agent), and races against the baseline.

Train: ``uv run python -m f1rl.train --config configs/strategy_ppo.yaml`` (a YAML
with ``strategy: true``). A run writes into ``runs/<name>/``: ``config.yaml``,
``meta.json`` (seed, git SHA), ``eval.csv`` and ``model_best.zip`` (best mean
finishing place on the VALIDATION tracks) plus ``model_final.zip``.

Compare: ``uv run python -m f1rl.agents.strategy --policy baseline --split
validation --races 10`` (or ``random``, or a model .zip). Every race is paired:
the same seed gives the same rivals, grid and track for each policy, so the
difference is the strategy alone.

The driver in these races is the stand-in (`models.race.LapTable`), not the
trained Driver Agent: results say how good the strategy is in this race model.
"""

import argparse
import csv
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from f1rl.agents.strategy_baseline import FixedPlan
from f1rl.config import (
    COMPOUNDS,
    EXAMPLE_CAR,
    CompoundParams,
    StrategyBaselineParams,
    StrategyParams,
    StrategyRewardParams,
)
from f1rl.driver_profile import DriverProfile, load_profile, save_profile
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRAIN, TUM_VALIDATION
from f1rl.envs.strategy_env import StrategyEnv, load_race_track

Policy = Callable[[StrategyEnv, Any], Any]

SPLITS: dict[str, tuple[str, ...]] = {
    "train": TUM_TRAIN,
    "validation": TUM_VALIDATION,
    "held_out": TUM_HELD_OUT,
}


@dataclass(frozen=True)
class StrategyRun:
    name: str
    seed: int
    total_timesteps: int
    n_envs: int = 8
    eval_every_steps: int = 20_000
    eval_races: int = 12  # per evaluation, spread over the validation tracks
    train_tracks: str = "train"
    eval_tracks: str = "validation"
    # Our driver, measured (`scripts/measure_driver.py`): JSON profile path.
    # Empty = the stand-in driver.
    driver_profile: str = ""


@dataclass(frozen=True)
class StrategyConfig:
    run: StrategyRun
    env: dict[str, Any] = field(default_factory=dict)
    baseline: dict[str, Any] = field(default_factory=dict)
    ppo: dict[str, Any] = field(default_factory=dict)

    def strategy_params(self, tracks: tuple[str, ...] = ()) -> StrategyParams:
        raw = dict(self.env)
        for name in COMPOUNDS:
            if name in raw:
                raw[name] = CompoundParams(**raw[name])
        if "reward" in raw:
            raw["reward"] = StrategyRewardParams(**raw["reward"])
        for key in ("push_levels",):
            if key in raw:
                raw[key] = tuple(raw[key])
        return StrategyParams(**{**raw, "tracks": tracks})

    def baseline_params(self) -> StrategyBaselineParams:
        raw = dict(self.baseline)
        for key in ("stints", "stop_fractions"):
            if key in raw:
                raw[key] = tuple(raw[key])
        return StrategyBaselineParams(**raw)


def load_strategy_config(path: Path) -> StrategyConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = set(raw) - {f.name for f in fields(StrategyConfig)} - {"strategy"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    config = StrategyConfig(
        run=StrategyRun(**raw["run"]),
        env=raw.get("env") or {},
        baseline=raw.get("baseline") or {},
        ppo=raw.get("ppo") or {},
    )
    config.strategy_params()
    config.baseline_params()
    return config


def load_tracks(
    names: tuple[str, ...],
    params: StrategyParams,
    profile: DriverProfile | None = None,
) -> dict[str, Any]:
    return {n: load_race_track(n, EXAMPLE_CAR, params, profile=profile) for n in names}


# --- policies ------------------------------------------------------------------------


def baseline_policy(env: StrategyEnv, obs: Any) -> Any:
    """Our car runs the rule-based plan (`StrategyEnv.baseline`)."""
    plan = FixedPlan(env.baseline)
    if env.lap == 0:
        return np.array(
            [COMPOUNDS.index(plan.start_compound) + 1, plan.params.push_index]
        )
    lap = plan.plan(env.params, env.cars[0], env.lap, env.total_laps)
    pit = 0 if lap.pit_to is None else COMPOUNDS.index(lap.pit_to) + 1
    return np.array([pit, plan.params.push_index])


def random_policy(seed: int) -> Policy:
    rng = np.random.default_rng(seed)

    def act(env: StrategyEnv, obs: Any) -> Any:
        # Mostly stay out: a uniform pit choice would stop three laps in four.
        pit = int(rng.integers(1, 4)) if rng.random() < 0.05 else 0
        return np.array([pit, int(rng.integers(len(env.params.push_levels)))])

    return act


def model_policy(model: Any) -> Policy:
    def act(env: StrategyEnv, obs: Any) -> Any:
        return model.predict(obs, deterministic=True)[0]

    return act


def race(env: StrategyEnv, policy: Policy, seed: int, track: str) -> dict[str, Any]:
    """One race; returns the final info plus the lap-by-lap history."""
    obs, info = env.reset(seed=seed, options={"track": track})
    done = False
    while not done:
        obs, _, done, _, info = env.step(policy(env, obs))
    return {**info, "history": list(env.history)}


def compare(
    env: StrategyEnv,
    policies: dict[str, Policy],
    tracks: tuple[str, ...],
    races: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired races (same seed and track for every policy): mean finishing place,
    mean race time, share of legal (two-compound) races and mean stops."""
    out: dict[str, dict[str, float]] = {}
    for name, policy in policies.items():
        results = [
            race(env, policy, seed + i, tracks[i % len(tracks)]) for i in range(races)
        ]
        out[name] = {
            "position": float(np.mean([r["position"] for r in results])),
            "race_time_s": float(np.mean([r["race_time_s"] for r in results])),
            "legal": float(np.mean([r["legal"] for r in results])),
            "stops": float(np.mean([r["stops"] for r in results])),
        }
    return out


# --- training ------------------------------------------------------------------------


def train(config: StrategyConfig, run_dir: Path) -> None:
    import stable_baselines3
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.env_util import make_vec_env

    from f1rl.agents.ppo import git_sha

    run = config.run
    train_names = SPLITS.get(run.train_tracks, (run.train_tracks,))
    eval_names = SPLITS.get(run.eval_tracks, (run.eval_tracks,))
    params = config.strategy_params(train_names)
    base = config.baseline_params()
    profile = load_profile(Path(run.driver_profile)) if run.driver_profile else None
    train_tracks = load_tracks(train_names, params, profile)
    eval_env = StrategyEnv(
        load_tracks(eval_names, params, profile), params, baseline=base, profile=profile
    )

    run_dir.mkdir(parents=True)
    if profile is not None:
        save_profile(profile, run_dir / "driver_profile.json")
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump({"strategy": True, **asdict(config)}), encoding="utf-8"
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "seed": run.seed,
                "git_sha": git_sha(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "stable_baselines3": stable_baselines3.__version__,
            },
            indent=2,
        )
    )

    vec = make_vec_env(
        lambda: StrategyEnv(train_tracks, params, baseline=base, profile=profile),
        n_envs=run.n_envs,
        seed=run.seed,
    )
    model = PPO("MlpPolicy", vec, seed=run.seed, verbose=0, **config.ppo)
    ref = compare(
        eval_env, {"baseline": baseline_policy}, eval_names, run.eval_races, 10_000
    )["baseline"]
    print(f"baseline on {run.eval_tracks}: {ref}")

    class Evaluate(BaseCallback):
        def __init__(self) -> None:
            super().__init__()
            self.best = float("inf")
            self.next_eval = run.eval_every_steps
            self.csv = run_dir / "eval.csv"
            with self.csv.open("w", newline="") as f:
                csv.writer(f).writerow(
                    [
                        "steps",
                        "position",
                        "race_time_s",
                        "legal",
                        "stops",
                        "baseline_position",
                        "baseline_race_time_s",
                    ]
                )

        def _on_step(self) -> bool:
            if self.num_timesteps < self.next_eval:
                return True
            self.next_eval += run.eval_every_steps
            res = compare(
                eval_env,
                {"model": model_policy(self.model)},
                eval_names,
                run.eval_races,
                10_000,
            )["model"]
            with self.csv.open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        self.num_timesteps,
                        res["position"],
                        res["race_time_s"],
                        res["legal"],
                        res["stops"],
                        ref["position"],
                        ref["race_time_s"],
                    ]
                )
            print(f"{self.num_timesteps:>9,}: {res}")
            # Illegal races are disqualified: rank them last.
            score = res["position"] + (1.0 - res["legal"]) * params.n_cars
            if score < self.best:
                self.best = score
                self.model.save(run_dir / "model_best.zip")
            return True

    model.learn(total_timesteps=run.total_timesteps, callback=Evaluate())
    model.save(run_dir / "model_final.zip")


# --- CLI -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Race strategies against each other.")
    parser.add_argument(
        "--policy",
        action="append",
        default=None,
        help="baseline, random or a model .zip (repeat to compare several).",
    )
    parser.add_argument("--split", default="validation", choices=sorted(SPLITS))
    parser.add_argument("--races", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-cars", type=int, default=None)
    parser.add_argument("--json", type=Path, default=None, help="Write results here.")
    parser.add_argument(
        "--profile", type=Path, default=None, help="Our driver, measured (JSON)."
    )
    args = parser.parse_args()

    tracks = SPLITS[args.split]
    params = StrategyParams(tracks=tracks)
    if args.n_cars:
        params = replace(params, n_cars=args.n_cars)
    profile = load_profile(args.profile) if args.profile else None
    env = StrategyEnv(load_tracks(tracks, params, profile), params, profile=profile)
    policies: dict[str, Policy] = {}
    for spec in args.policy or ["baseline", "random"]:
        if spec == "baseline":
            policies[spec] = baseline_policy
        elif spec == "random":
            policies[spec] = random_policy(args.seed)
        else:
            from stable_baselines3 import PPO

            policies[spec] = model_policy(PPO.load(spec))
    results = compare(env, policies, tracks, args.races, args.seed)
    for name, res in results.items():
        print(
            f"{name:>12}: place {res['position']:.2f}, race "
            f"{res['race_time_s']:.1f} s, legal {res['legal']:.0%}, "
            f"stops {res['stops']:.2f}"
        )
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
