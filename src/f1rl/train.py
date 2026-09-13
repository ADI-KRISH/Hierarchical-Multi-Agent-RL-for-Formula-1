"""PPO training entry point: ``uv run python -m f1rl.train --config <yaml>``.

Placeholder -- training arrives in phase 4, once `DriverEnv` (phase 2) and the
rule-based baseline (phase 3) exist. The CLI surface is fixed here so the command
in CLAUDE.md stays stable.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the RL driver agent.")
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Experiment YAML under configs/ holding the hyperparameters.",
    )
    parser.parse_args()
    raise SystemExit("f1rl.train is not implemented yet (roadmap phase 4).")


if __name__ == "__main__":
    main()
