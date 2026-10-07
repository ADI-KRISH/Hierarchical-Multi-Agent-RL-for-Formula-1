# FormulaOneRL — Build Roadmap (MVP)

Work these phases **in order, one per Claude Code session**. Do not start a phase
until the previous one's "Done when" box is genuinely true. After each phase:
run tests + lint, commit, then `/clear` before the next.

The MVP target: a single RL Driver Agent that laps a custom sim faster than a
rule-based baseline, with a dashboard showing it learn. The Strategist and the
two-agent hierarchy are explicitly out of scope here — they're Phase 8 (Future).

---

## Phase 0 — Project skeleton

Goal: an empty but runnable project, correct structure, nothing learns yet.

- Initialize with **uv** (`uv init`), package layout per CLAUDE.md.
- Add runtime deps with `uv add`: Stable Baselines3, Gymnasium, PyTorch, NumPy.
- Add dev deps with `uv add --dev`: pytest, ruff, mypy.
- `.gitignore` covering `runs/`, `data/`, `__pycache__`, `.venv/` (but NOT
  `uv.lock` — that gets committed).
- `src/f1rl/config.py` with a place for calibration constants.
- One trivial passing test.

**Prompt idea:** "Set up the project skeleton from CLAUDE.md using uv for env and
dependency management. No sim logic yet — just structure, packaging with
pyproject.toml + uv.lock, tooling, and one passing test."

**Done when:** `uv sync` works and `uv run pytest` is green.

---

## Phase 1 — Track + car physics (no RL)

Goal: compute a lap time for a fixed, hand-written driving policy.

- Track model: a lap as a sequence of segments (curvature + length).
- Car model: point-mass with max grip, accel, braking — as pure functions.
- A simple "drive at the limit" policy to sanity-check lap times.

**Prompt idea:** "Build the track model and point-mass car physics as pure
functions in `src/f1rl/models/` with unit tests. Compute a lap time for a fixed
policy. No Gym, no RL."

**Done when:** a sample lap time prints and looks physically sane (tens of
seconds to a couple minutes, not 0.1s or an hour). Every model has a test.

---

## Phase 2 — Wrap as a Gymnasium env

Goal: `DriverEnv` that a standard RL algorithm can plug into.

- Observation space (speed, track position, distance to next corner, etc.).
- Action space (steering, throttle, brake — continuous).
- Reward: progress + speed, penalties for going off-track. Keep it simple.
- Passes Gymnasium's env checker.

**Prompt idea:** "Wrap the car and track into a Gymnasium env `DriverEnv` per
docs/context.md. Include the env checker test. No training yet."

**Done when:** `gymnasium.utils.env_checker` passes and a random agent can step
through an episode without crashing.

---

## Phase 3 — Rule-based baseline

Goal: something for the RL agent to beat.

- A scripted driver: fixed racing line, fixed braking points.
- Runs in `DriverEnv`, reports a lap time and completion rate.

**Prompt idea:** "Write a rule-based baseline driver that runs in DriverEnv, and
print its average lap time over 20 runs. This is the number the RL agent must beat."

**Done when:** the baseline completes laps and you have its lap time written down.

---

## Phase 4 — Train the RL driver

Goal: a PPO agent that learns to drive — the heart of the project.

- SB3 PPO training script driven by `configs/driver_ppo.yaml`.
- Logs reward + lap time over training into `runs/`.
- Short run first to confirm the plumbing, then a longer run.

**Prompt idea:** "Write the SB3 PPO training script for DriverEnv with a config
file, logging reward and lap time to runs/. Start a short training run."

**Expect iteration.** The agent may not learn at first — this is normal and is
the real work. You and Claude Code adjust reward shaping, observation scaling,
and hyperparameters together across several runs. Watch the reward curve.

**Done when:** the trained agent completes clean laps and the reward curve
clearly trends up over training.

---

## Phase 4R — Racing line on many tracks (added 2026-10-07)

Why: the Phase 4 driver is locked to the track centreline, so it cannot learn a
racing line. On TUM's 25 real tracks the racing line is 9.2% faster than the
centreline in our own physics (`scripts/raceline_gain.py`). Full research, the
seven reference repos reviewed and the reasoning: `docs/racing_line_research.md`.
Work these steps in order, like phases.

### 4R.1 — More tracks with real widths
- Load TUMFTM/racetrack-database (25 tracks: centreline + left/right widths),
  downloaded and cached under `data/` (LGPL data, not vendored).
- Per-segment widths from the data. Hold out ~5 tracks for evaluation only.

**Done when:** all 25 load, lengths match the data, and a baseline lap runs on each.

### 4R.2 — Optimal racing line for every track
- Pure function: minimum-curvature line (lateral shift per station, bounded by the
  width) and its limit lap. Check against TUM's published racing lines.

**Done when:** our optimal lap times match the limit lap on TUM's lines within ~1%,
and the report shows the optimal line and its lap time per track.

### 4R.3 — Let the car steer (env v2)
- Frenet-frame kinematics: heading relative to the track and a steered path
  curvature, so the lateral offset changes the corner the car drives.
- One friction circle shared by steering and throttle/brake. Off track past the
  local half-width. Physics as pure functions with tests; keep v1 for comparison.
- Design in the hooks for driving styles (a style input, style-weighted reward
  terms), unused for now — see Phase 8.

**Done when:** a scripted driver following the 4R.2 optimal line laps within ~1%
of the optimal limit lap in env v2, and the env checker passes.

### 4R.4 — RL design from AM-RL
- Action mapping: actions scaled into what the tyres can deliver (cannot exceed grip).
- Observation: centreline and edge points ahead in the car's frame, speed, heading
  error, offset, yaw rate; no lap-progress feature.
- Random starts anywhere on the lap; reward = speed along the track or gate times.

### 4R.5 — Train across tracks with SAC/TD3
- SB3 SAC or TD3, envs dealt across the training tracks (+ generated tracks).
- Per-track evals and replay laps, as in Phase 4.

**Done when:** the driver completes every held-out track and is within a few
percent of the optimal-line limit there; replays show outside-apex-outside lines.

---

## Phase 5 — Beat the baseline

Goal: prove the agent is actually good.

- `eval.py` runs a saved agent for N episodes, reports mean/best lap time,
  completion rate, off-track count.
- Compare head-to-head with the Phase 3 baseline.

**Prompt idea:** "Write eval.py to score a saved agent over 20 episodes and print
a comparison table against the rule-based baseline."

**Done when:** the RL agent's lap time beats the baseline (or you understand and
can explain why not — itself a valid portfolio finding).

---

## Phase 6 — Dashboard

Goal: make it visible and demo-able.

- Dash app reading logged runs: lap-time-over-training curve, the car's path on
  the track, baseline-vs-agent comparison.
- Read-only. Never triggers training.

**Prompt idea:** "Build a read-only Dash dashboard reading runs/: training
lap-time curve, the car's racing line, and agent-vs-baseline comparison."

**Done when:** `python -m f1rl.dashboard` opens and shows the agent's learning
and its line around the track.

---

## Phase 7 — Polish for portfolio

Goal: something a recruiter can read and run.

- `README.md`: what it is, a GIF/screenshot, how to install and run, results.
- A short "Results" section with the baseline-vs-agent numbers.
- A "Future work" section describing the two-agent hierarchy (Phase 8) — this
  shows you understand where it goes without having to build it.

**Done when:** a stranger can clone the repo, follow the README, and see a
trained car lap the track.

---

## Phase 8 — Future work (NOT in the MVP)

Only after everything above runs and is committed. Each is a real project on its
own; do not start these to "finish faster."

- Driving styles: a style input (smooth to attacking) the driver is trained to
  follow, measured from telemetry (braking points, grip used, steering rate).
  Staging: `docs/racing_line_research.md` section 5.
- Opponent cars + overtaking and defending rewards (scripted opponents, then self-play).
- Tire degradation, fuel, ERS models feeding the observation.
- A **Strategy Agent** (discrete, once per lap) — the second half of the vision.
  Full design: `docs/strategy_agent_design.md`. Earliest realistic entry point is
  after Phase 5 (Driver Agent beats the baseline) or Phase 6 (dashboard) — do not
  start this without the user explicitly confirming first.
- Coupling the two agents (strategy mode fed into the driver's observation; the
  driving-style input is that channel).
- Stretch: self-play, FastF1 calibration, LLM race engineer, digital-twin track.