# FormulaOneRL

**Hierarchical Multi-Agent Reinforcement Learning for Formula 1 Driver and Race
Strategy.**

A reinforcement learning driver agent that learns to lap a custom Formula 1
track simulator faster than a rule-based baseline. Stable Baselines3 (PPO) on a
custom Gymnasium environment.

**Status:** phase 4 -- PPO drivers lap a synthetic track and six real circuits
(Zandvoort, Spa, Suzuka, Monaco, Sepang, Interlagos) cleanly, 1-4% behind a
rule-based baseline; `PROGRESS.md` explains why they are slower. See `PROGRESS.md` for what is done and what is left,
`docs/roadmap.md` for the build plan, and `docs/context.md` for the long-term
vision.

## Overview

Most F1 AI projects focus only on autonomous driving, but Formula 1 success
depends just as much on race strategy -- pit timing, tyre choice, fuel and ERS
management. The long-term goal here is a two-agent system that models both
halves of that job:

- **Driver Agent** (this MVP) -- real-time vehicle control: steering,
  throttle, braking, and racing-line optimization on a custom point-mass /
  bicycle-model car and track.
- **Race Strategist Agent** (future work) -- race-level tactical decisions
  (pit stops, tyre compound, fuel/ERS strategy) communicated to the driver as
  a high-level objective.

**This MVP builds the Driver Agent alone.** The Strategist and the two-agent
hierarchy are deliberately out of scope until the driver works end to end --
see `docs/context.md` for the full vision and `docs/roadmap.md` for the
phase-by-phase plan this repo is following.

## Why a custom simulator

Heavyweight simulators (TORCS, CARLA, Assetto Corsa) trade simulation speed
for physical fidelity. This project intentionally uses a lightweight custom
Gymnasium environment -- a point-mass car on a parameterized track, described
as a sequence of curvature + length segments -- because simulation speed
matters more than realism for RL sample efficiency. Physical constants (car
grip, mass, top speed) are cited from public F1 technical figures or
calibrated against FastF1 telemetry, not guessed.

## Tech stack

Python, Stable Baselines3 (PPO), Gymnasium, NumPy, PyTorch, Dash (dashboard,
later phases), FastF1 (telemetry calibration, later phases). Environment and
dependencies are managed with `uv` and pinned in `pyproject.toml` / `uv.lock`.

## Project layout

```
src/f1rl/envs/    Track model, DriverEnv (Gymnasium environment)
src/f1rl/models/  Car physics, tire, fuel -- pure functions, each unit tested
src/f1rl/agents/  SB3 training wrapper + rule-based baseline driver
src/f1rl/         train.py, eval.py, dashboard.py, config.py
configs/          One YAML per experiment; hyperparameters live here
docs/             context.md (long-term vision), roadmap.md (build plan)
tests/
```

## Quickstart

```bash
uv sync            # create the env from uv.lock
uv run pytest      # tests
uv run ruff check  # lint
uv run mypy src/   # type check

# Phase 3: run the rule-based baseline, then build the HTML report
uv run python -m f1rl.agents.baseline --track technical --episodes 20
uv run python -m f1rl.report          # -> runs/baseline/report.html

# Phase 4: train the PPO driver (~13 min on 4 CPU cores), then re-run the report
uv run python -m f1rl.train --config configs/driver_ppo.yaml
# One driver on six real circuits (~50 min); circuits download once to data/
uv run python -m f1rl.train --config configs/driver_ppo_circuits.yaml
```

Tracks: `oval`, `technical` (synthetic); `zandvoort`, `spa`, `suzuka`, `monaco`,
`sepang`, `interlagos` (real, from GPS centerlines of the open f1-circuits
dataset); or a FastF1 session as `<year>:<grand prix>`, e.g. `2023:Monaco`.

Other configs: `configs/driver_ppo_stable.yaml` is the best technical-track setup
found so far; `configs/experiments/` holds the diagnosis experiments behind the
"Why is the RL driver slower" table in `PROGRESS.md` (each file says what it tested
and what it found).

## Training on your own computer

Training needs no GPU: the policy is a small MLP and most time goes into
simulating the car, so a laptop CPU is fine. On 4 CPU cores a 2M-decision run takes
about 13 minutes and the six-circuit run (6M) about 50. Each run uses one core
(`torch_threads: 1`), so you can run one experiment per core in parallel. Local
training also keeps your models (`runs/` is not saved to git) and lets FastF1 reach
its servers.

1. Install [uv](https://docs.astral.sh/uv/):
   - Windows (PowerShell): `powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"`
   - macOS / Linux: `curl -LsSf https://astral.sh/uv/install.sh | sh`
2. Get the code, or update a copy you already have:
   ```bash
   # new copy
   git clone https://github.com/ADI-KRISH/Hierarchical-Multi-Agent-RL-for-Formula-1.git
   cd Hierarchical-Multi-Agent-RL-for-Formula-1

   # existing copy: save or stash any local edits first (git status), then
   git checkout main
   git pull origin main
   ```
   `runs/`, `data/` and `references/` are ignored by git, so pulling never touches them.
3. Install and check: `uv sync` (first time downloads PyTorch, a few GB), then
   `uv run pytest`.
4. Train and look at the results with the commands in Quickstart above; open
   `runs/baseline/report.html` in a browser. Runs are seeded, so results should match
   `PROGRESS.md` closely (different hardware can change the last decimals).

