# FormulaOneRL

A reinforcement learning driver agent that learns to lap a custom Formula 1 track
simulator faster than a rule-based baseline. Stable Baselines3 (PPO) on a custom
Gymnasium environment.

**Status:** phase 1 -- track + car physics. Point-mass car model and a
segment-based track give a lap time for a fixed "drive at the limit" policy;
no Gym env, baseline, or training yet. See `docs/roadmap.md` for the build plan
and `docs/context.md` for the long-term vision.

## Quickstart

```bash
uv sync            # create the env from uv.lock
uv run pytest      # tests
uv run ruff check  # lint
uv run mypy src/   # type check
```
