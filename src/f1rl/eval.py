"""Evaluation entry point: ``uv run python -m f1rl.eval --agent <path>``.

Placeholder -- scoring a saved agent against the rule-based baseline is phase 5.
"""

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score a saved agent against the rule-based baseline."
    )
    parser.add_argument(
        "--agent", type=Path, required=True, help="Path to a saved SB3 checkpoint."
    )
    parser.add_argument(
        "--episodes", type=int, default=20, help="Episodes to average over."
    )
    parser.parse_args()
    raise SystemExit("f1rl.eval is not implemented yet (roadmap phase 5).")


if __name__ == "__main__":
    main()
