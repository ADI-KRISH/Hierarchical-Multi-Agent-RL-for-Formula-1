# FormulaOneRL

A reinforcement learning driver agent that learns to lap a custom Formula 1 track
simulator faster than a rule-based baseline. Stable Baselines3 (PPO) on a custom
Gymnasium environment.

**Status:** phase 0 -- project skeleton. No simulation or training yet; see
`docs/roadmap.md` for the build plan and `docs/context.md` for the long-term vision.

## Quickstart

```bash
uv sync            # create the env from uv.lock
uv run pytest      # tests
uv run ruff check  # lint
uv run mypy src/   # type check
```
