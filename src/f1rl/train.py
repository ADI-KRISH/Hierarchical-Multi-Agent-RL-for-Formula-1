"""PPO training entry point: ``uv run python -m f1rl.train --config <yaml>``.

Reads one experiment YAML from ``configs/`` and trains the driver on its track,
logging into ``runs/<run name>/`` (see `f1rl.agents.ppo` for what's written).
"""

import argparse
import shutil
from pathlib import Path

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

    config = load_config(args.config)
    if args.timesteps is not None:
        from dataclasses import replace

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


if __name__ == "__main__":
    main()
