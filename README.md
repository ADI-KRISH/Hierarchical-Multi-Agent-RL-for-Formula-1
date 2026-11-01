# FormulaOneRL

**Hierarchical multi-agent reinforcement learning for Formula 1: a Driver Agent that
races and overtakes cleanly, and a Strategy Agent that calls pit stops, tyres and
push levels, both trained on 25 real circuits.**

FormulaOneRL is a custom Gymnasium racing simulator plus two PPO agents trained
with Stable-Baselines3:

* **Driver Agent** - every 0.5 s chooses a race-craft move (follow the line, pull
  out of the slipstream, go left/right of the car ahead, cover, take the inside,
  feint...). A planner drives the move at the grip limit, and a predictive safety
  shield makes contact impossible for every car.
* **Strategy Agent** - once a lap chooses whether to pit, which compound to fit
  and a push level, over full Grand Prix distances with safety cars, VSCs, rain,
  spins and punctures. It is trained on numbers measured from our own driver.

| Result (unseen circuits) | Our agent | Rule-based baseline |
|---|---|---|
| Contacts on 1,174 real 2021-2025 overtaking situations | **0** | 20 without the shield |
| Defences held, worn tyres (460) | **273** | 264 (hold the line), 257 (scripted defender) |
| Clean passes in section battles (350) | **49** | 29 (starting policy) |
| Race wins vs rule-based field, measured driver | **72%** | 10% |
| Race wins in the rain | **68%** | 0% |
| Race wins, one strategist for 8 race cases (1% faster car from P10): clean / spin / safety car / rain | **100% / 38.5% / 95.5% / 89.5%** | 99% / 13% / 93.5% / 66.5% |
| Same, VSC (the one case it loses) | 91% | **96%** |
| Planner lap pace | **+1.1%** off the limit lap | pure SAC RL: +17-20% |

## Documents

| Document | What it is |
|---|---|
| [`docs/report/FormulaOneRL_Report.md`](docs/report/FormulaOneRL_Report.md) (and `.docx`) | Academic report: method, experiments, ablations, references |
| [`docs/PROJECT_EXPLAINER.md`](docs/PROJECT_EXPLAINER.md) | The whole project in plain language: agents, scenarios, tools, features, problems faced |
| [`docs/WORK_SPLIT.md`](docs/WORK_SPLIT.md) | Equal three-member split of the work |
| [`docs/project_status.md`](docs/project_status.md) | Run-by-run results of the driver work |
| [`docs/strategy_agent_design.md`](docs/strategy_agent_design.md) | Strategy Agent design and results |
| [`docs/training_log.md`](docs/training_log.md) | SAC driver training runs |
| [`PROGRESS.md`](PROGRESS.md) | Phases 0-4 (first simulator and PPO lap-time driver) |
| [`docs/context.md`](docs/context.md), [`docs/roadmap.md`](docs/roadmap.md) | Original vision and build plan |

## How it works

```
                 ┌──────────────────────── Strategy level (once per lap) ────────────────────────┐
 race state ───► │ StrategyEnv (10 cars, tyres, fuel, SC/VSC, rain, incidents) ─► PPO strategist │ ─► pit / compound / push
 (46 inputs)     └───────────────▲───────────────────────────────────────────────────────────────┘
                                 │ measured driver profile (pace gap, pass & defend curves)
                 ┌───────────────┴──────── Driver level (every 0.5 s) ────────────────────────────┐
 car state  ───► │ TacticalEnv ─► PPO driver ─► move ─► PathDriver ─► safety shield ─► RaceEnv 50 Hz│
 (75 inputs)     └────────────────────────────────────────────────────────────────────────────────┘
```

* **Circuits:** 25 real circuits from the TUM racetrack database: 17 train, 3
  validation (Montreal, Hockenheim, Catalunya), 5 held-out test (Suzuka, Spa,
  Sao Paulo, Silverstone, Yas Marina).
* **Car:** Frenet-frame point mass with a friction ellipse (5 g lateral and braking,
  1.5 g traction, 97 m/s), tyre wear, slipstream, dirty air, aerodynamic drag.
* **Real data:** passes and defences from FastF1 (2021-2022) and OpenF1 (2023-2025)
  become start situations and a held-out test set.
* **Rewards:** driver - progress, a pass bonus paid only once the pass sticks for
  600 m, penalties for unsafe proximity and shield interventions; strategist -
  minus race time plus a win bonus.

## Quick start

Install [uv](https://docs.astral.sh/uv/), then:

```bash
git clone https://github.com/ADI-KRISH/Hierarchical-Multi-Agent-RL-for-Formula-1.git
cd Hierarchical-Multi-Agent-RL-for-Formula-1
uv sync                         # creates the environment from uv.lock
uv run pytest                   # unit tests
uv run ruff check . && uv run mypy src/
```

Circuits download once from the TUM racetrack database into `data/` (git-ignored).
Training needs no GPU; a laptop CPU is enough.

### Train

```bash
# Strategy Agent, final: one agent for every race case (4M laps)
uv run python -m f1rl.train --config configs/strategy_master.yaml

# Driver Agent, move-level PPO (needs the starting models named in the config)
uv run python -m f1rl.train --config configs/defend_sections_v2.yaml

# Pure-RL SAC driver baseline (throttle + steering)
uv run python -m f1rl.train --config configs/racer_sac.yaml

# First PPO lap-time driver on a synthetic track (~13 min)
uv run python -m f1rl.train --config configs/driver_ppo.yaml
```

Each run writes `runs/<name>/` with the resolved config, seed, git commit,
`eval.csv`, `model_best.zip` (chosen on validation circuits) and `model_final.zip`.

### Evaluate

```bash
# Strategy: held-out scorecard per race case (clean, spin, puncture, SC, VSC),
# same seeded races for the rule-based plan and each model
uv run python scripts/strategy_scorecard.py configs/strategy_master.yaml \
    runs/strategy_master/model_best.zip --races 200

# Driver: scenario scorecard on held-out circuits (add --real for real F1 starts)
uv run python -m f1rl.scenarios --policy runs/<run>/model_best.zip --tracks tum_held_out

# Driver: corner-by-corner battles (clean-racing rules)
uv run python scripts/section_gate.py --help

# SAC driver lap times on held-out circuits
uv run python -m f1rl.eval --agent runs/racer_sac/model_best.zip --tracks tum_held_out

# Report figures from the logged runs (read-only)
uv run python scripts/make_report_figures.py
```

## Repository layout

```
src/f1rl/
  envs/      DriverEnv, RaceEnv (50 Hz car), TacticalEnv (moves), StrategyEnv (laps),
             circuits, TUM tracks, opponents
  models/    car, Frenet frame, racing line, tyres, weather, race model, safety shield
  agents/    PPO/SAC wrappers, PathDriver, tactics, tactician, guard + RL defenders,
             strategist + rule-based plan
  train.py, eval.py, scenarios.py, sections.py, incidents.py, openf1.py,
  real_starts.py, driver_profile.py, config.py, report.py, dashboard.py
configs/     one YAML per experiment (hyperparameters live here, not in code)
scripts/     gates, oracles, demos, driver measurement, report figures
tests/       unit tests (pytest)
docs/        report, explainer, work split, status and design docs
```

`runs/` (checkpoints) and `data/` (circuit and FastF1/OpenF1 caches) are
git-ignored.

## Tech stack

Python 3.12 · Gymnasium · Stable-Baselines3 (PPO, SAC) · PyTorch · NumPy · SciPy ·
FastF1 · OpenF1 · TUM racetrack database · matplotlib · uv · pytest · ruff · mypy

## Reproducibility

Every environment, training run and evaluation takes an explicit seed; there is no
unseeded randomness in `src/`. Model selection uses only the validation circuits;
held-out circuits are used once, for the final scorecards. Values that are design
choices rather than F1 figures (tyre compounds, pit loss, slipstream, weather and
incident rates) are labelled as such in `src/f1rl/config.py`.

## Status and limitations

The simulator is a point-mass model with uncalibrated tyre and strategy constants;
rival strategies are rule-based and do not react; overtaking at realistic pace
gaps (under ~1 s per lap) remains rare; the driver does not yet take the
strategist's push mode as an input. See Section 5 of the report and
`docs/PROJECT_EXPLAINER.md` for the full list.
