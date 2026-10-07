# FormulaOneRL (MVP)

A single **Driver Agent** that learns to lap a custom Formula 1 track simulator
as fast as possible, beating a rule-based baseline. Reinforcement learning with
Stable Baselines3 on a custom Gymnasium environment.

**Scope discipline:** this MVP is ONE agent only. The two-agent hierarchy (a
Strategist that plans pit stops / tires / ERS) is FUTURE WORK — do not build it
unless a task explicitly says so. Full long-term vision lives in `docs/context.md`.
Current build plan lives in `docs/roadmap.md` — work it one phase at a time.

## Golden rules

- **Custom Gymnasium env only.** Do NOT add TORCS, CARLA, or Assetto Corsa.
  The car is a point-mass / bicycle model on a parameterized track. Simulation
  speed matters more than physical realism.
- **One env for now:** `DriverEnv` (per-step control). No `StrategyEnv` yet.
- **Determinism.** Every env, training run, and eval takes an explicit `seed`.
  No unseeded RNG and no wall-clock randomness anywhere in `src/`.
- **Don't invent F1 numbers.** Lap times, grip, and tire values come from a
  cited constant in `src/f1rl/config.py` or FastF1 data in `data/` — not guesses.
- Build one phase at a time. Get each phase running before starting the next.

## Environment & tooling

- **uv** manages the virtual env and dependencies. Use uv for everything — never
  call `pip install` directly, and don't hand-edit a requirements.txt.
- Dependencies live in `pyproject.toml`; `uv.lock` is committed for reproducibility.
- Add a dependency: `uv add <pkg>` (dev-only: `uv add --dev <pkg>`).
- Run any project command through uv so it uses the project env, e.g.
  `uv run pytest`. Don't activate the venv manually in instructions.

## Commands

- Install / sync env: `uv sync`
- Test: `uv run pytest` (mark slow sim/training tests `@pytest.mark.slow`)
- Lint + format: `uv run ruff check --fix . && uv run ruff format .`
- Type check: `uv run mypy src/`
- Train driver: `uv run python -m f1rl.train --config configs/driver_ppo.yaml`
- Evaluate vs baseline: `uv run python -m f1rl.eval --agent <path> --episodes 20`
- Baseline driver: `uv run python -m f1rl.agents.baseline --track technical --episodes 20`
- Report: `uv run python -m f1rl.report` (reads `runs/baseline/` + training runs, never runs a driver)
- Dashboard: `uv run python -m f1rl.dashboard` (reads logged runs, never trains)

Run `uv run ruff check`, `uv run mypy src/`, and `uv run pytest` before reporting
any task complete.

## Layout

- `src/f1rl/envs/` — `DriverEnv`, track model
- `src/f1rl/models/` — car physics, tire, fuel (pure functions, each with a test)
- `src/f1rl/agents/` — SB3 training wrapper + the rule-based baseline driver
- `src/f1rl/train.py`, `eval.py`, `dashboard.py`, `config.py`
- `configs/` — one YAML per experiment; hyperparameters live here, NOT in code
- `data/` — FastF1 caches / calibration constants (gitignored)
- `runs/` — training logs + checkpoints (gitignored)
- `tests/`, `docs/context.md`, `docs/roadmap.md`

## Conventions

- RL library is **Stable Baselines3**. Do not reach for Ray RLlib in the MVP.
- Physics/tire models are **pure functions** — state in, state out, no I/O, no
  globals. Every model gets a unit test.
- Reward logic lives in `DriverEnv`, tunable via config. When you change a
  reward, note the before/after in the commit — it invalidates earlier runs.
- New tunable value → add it to the config, don't hardcode it.
- Log each run's config + git SHA into its `runs/` folder so results reproduce.

## Gotchas

- Checkpoints (`runs/`) and FastF1 caches (`data/`) are large — gitignored,
  never commit them. But DO commit `uv.lock`.
- The dashboard is read-only over logged data. It must never start training.
- RL agents often fail to learn on the first try. When training looks flat,
  suspect the reward function or the observation scaling before the algorithm.