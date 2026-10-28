"""Training entry point: ``uv run python -m f1rl.train --config <yaml>``.

Reads one experiment YAML from ``configs/`` and trains the driver, logging into
``runs/<run name>/``. A YAML with ``tactical: true`` is the race-craft driver
(`TacticalEnv`, `f1rl.agents.tactical`); one with a ``sac:`` section is the
racing-line driver (`RaceEnv`, `f1rl.agents.racer`); otherwise it is the v1
centreline PPO driver (`DriverEnv`, `f1rl.agents.ppo`).
"""

import argparse
import shutil
from dataclasses import replace
from pathlib import Path

import yaml

from f1rl.agents.baseline import resolve_track
from f1rl.agents.ppo import load_config, train


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL driver agent.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML under configs/ holding the hyperparameters.",
    )
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument(
        "--timesteps", type=int, default=None, help="Override run.total_timesteps."
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace an existing run folder."
    )
    args = parser.parse_args()

    raw = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if raw.get("tactical"):
        _train_tactical(args)
        return
    if "sac" in raw:
        _train_racer(args)
        return

    config = load_config(args.config)
    if args.timesteps is not None:
        config = replace(
            config, run=replace(config.run, total_timesteps=args.timesteps)
        )
    run_dir = args.runs / config.run.name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} exists -- pass --overwrite or rename the run.")
        shutil.rmtree(run_dir)

    tracks = {name: resolve_track(name) for name in config.run.tracks}
    where = ", ".join(f"{n} ({t.total_length_m:.0f} m)" for n, t in tracks.items())
    print(
        f"Training {config.run.name}: {config.run.total_timesteps:,} decisions on "
        f"{where}, seed {config.run.seed}"
    )
    train(config, tracks, run_dir)
    print(f"Done. Logs and models in {run_dir}/")


def _train_tactical(args: argparse.Namespace) -> None:
    from f1rl.agents.tactical import load_tactical_config
    from f1rl.agents.tactical import train as train_tactical

    config = load_tactical_config(args.config)
    if args.timesteps is not None:
        config = replace(
            config, run=replace(config.run, total_timesteps=args.timesteps)
        )
    run_dir = args.runs / config.run.name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} exists -- pass --overwrite or rename the run.")
        shutil.rmtree(run_dir)
    print(
        f"Training {config.run.name} (PPO, race-craft moves): "
        f"{config.run.total_timesteps:,} decisions, seed {config.run.seed}",
        flush=True,
    )
    train_tactical(config, run_dir)
    print(f"Done. Logs and models in {run_dir}/")


def _train_racer(args: argparse.Namespace) -> None:
    from f1rl.agents.racer import load_racer_config
    from f1rl.agents.racer import train as train_racer

    config = load_racer_config(args.config)
    if args.timesteps is not None:
        config = replace(
            config, run=replace(config.run, total_timesteps=args.timesteps)
        )
    run_dir = args.runs / config.run.name
    if run_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{run_dir} exists -- pass --overwrite or rename the run.")
        shutil.rmtree(run_dir)
    print(
        f"Training {config.run.name} (SAC, racing line): "
        f"{config.run.total_timesteps:,} decisions, seed {config.run.seed}",
        flush=True,
    )
    train_racer(config, run_dir)
    print(f"Done. Logs and models in {run_dir}/")


if __name__ == "__main__":
    main()
