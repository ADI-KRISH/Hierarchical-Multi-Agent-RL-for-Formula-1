# Progress

Snapshot as of 2026-10-06. Phase definitions and "done when" criteria live in
`docs/roadmap.md`; this file only tracks where we are against them.

**Summary:** phases 0-4 are done. A PPO driver completes clean laps of the
technical track (44.18 s, 100% of eval laps) but is not yet faster than the
rule-based baseline (43.19 s mean). Phases 5-7 are not started.

## Done

### Phase 0 — Project skeleton
- uv-managed project, `pyproject.toml` + committed `uv.lock`.
- Package layout under `src/f1rl/`, with ruff, mypy and pytest wired up.
- `src/f1rl/config.py` holding the calibration constants.

### Phase 1 — Track + car physics
- `envs/track.py`: a lap as a sequence of constant-curvature segments.
- `models/car.py`: point-mass car (grip, acceleration, braking) as pure functions.
- `models/lap.py`: lap time for a fixed "drive at the limit" policy.
- Fix: the corner grip limit is enforced through the whole corner, not just at entry.

### Phase 2 — Gymnasium `DriverEnv`
- `envs/driver_env.py`: continuous-action env that passes Gymnasium's env checker.
- Progress reward with an off-track penalty and termination.

### Beyond the phase 2 spec
- **Real track limits:** off-track is now a lateral offset exceeding half the track
  width, replacing the old "speed above the corner grip cap" check. The offset is
  part of the observation.
- **Per-segment track width and steering:** the action is `[throttle/brake, steer]`.
  Lateral grip is a budget — cornering uses some, steering spends what is left.
- **Real circuit geometry:** `envs/circuits.py` builds a track from a FastF1 fastest-lap
  position trace and caches it under `data/circuits/`. Reconstructed lengths for
  Suzuka, Monaco and Zandvoort 2023 are within ~2% of the official lengths.
- **Strategy Agent design doc:** `docs/strategy_agent_design.md`. Design only, no code.

### Env review fixes (before phase 4)
- Steering moves the car sideways at a rate that scales with forward speed (a parked
  car used to slide 5 m in 0.2 s).
- Reward adds a per-second time penalty: progress alone paid 1.0 per lap at any pace.
- Observation reports corner speed limits in lookahead windows out to 150 m (was one
  curvature point 50 m ahead, which could skip a short hairpin).
- Unseeded resets continue the seeded RNG; `info` reports lap time and how an
  episode ended; track lookups are O(log n) (env ~9x faster).

### Phase 3 — Rule-based baseline
- `agents/baseline.py`: a cautious scripted driver (fixed plan at 90% corner grip,
  braking points as if brakes had 80%), seeded per-episode margin jitter.
- `analytics.py`: corner/straight sections, time lost per section, lap summaries.
- `report.py`: read-only HTML report over `runs/baseline/`.

Baseline lap times, 20 episodes, seed 0 (`EXAMPLE_CAR`, standing start):

| Track | Mean lap | Best | Limit lap | Gap to limit | Finished |
|---|---|---|---|---|---|
| oval (2.23 km) | 29.819 s ± 0.128 | 29.566 s | 29.162 s | +2.25% | 20/20 |
| technical (2.90 km) | 43.188 s ± 0.177 | 42.893 s | 41.931 s | +3.00% | 20/20 |

The RL agent has to beat the mean lap; the limit lap is the ceiling no driver in
this simulator can pass.

### Phase 4 — Train the RL driver
- `agents/ppo.py` + `train.py`: SB3 PPO driven by `configs/driver_ppo.yaml`. Each run
  logs to `runs/<name>/`: resolved config, git SHA, SB3 `progress.csv`, every
  training episode (`episodes.csv`), periodic deterministic evals on held-out seeds
  (`eval.csv`), `model_best.zip`, `model_final.zip`, and a telemetry lap of the best
  model (`best_lap.json`).
- `report.py` adds a training section: reward curve, eval lap time against the
  baseline and limit, and the agent's speed/throttle trace against both.

Result (`driver_ppo`, technical track, seed 0, 2M decisions at 10 Hz, ~13 min CPU):
first clean eval lap at 150k decisions (64.8 s), then steadily down to **44.18 s**,
still improving at the end. Baseline mean 43.19 s, limit 41.93 s. The agent lifts
and coasts on the shorter straights (280-285 km/h where the car can do 349), which
is where most of its remaining second goes.

What it took (each a config value; earlier runs kept under `runs/` locally):

| Run | Change | Outcome |
|---|---|---|
| 1 | 50 Hz decisions, off-track penalty only | Full throttle into T1 every time; braking a bit earlier still crashed, so no gradient toward braking |
| 2 | Graded overspeed penalty past braking points, safe-speed observation, 10 Hz decisions | Learned to brake for T1/T2, then parked before the chicane |
| 3 | γ 0.98 -> 0.995 | Parked at the hairpin instead: stopping cost almost nothing, crashing cost 1.0 |
| 4 | Stalling (< 2 m/s for 3 s) ends the episode like an off | No parking, but braking still imprecise |
| 5 | Braking margin (safe speed - speed) as its own observation, scaled to +-1 | Clean laps from 150k decisions on |

## Left

| Phase | What | State |
|---|---|---|
| 5 | `eval.py`: agent vs baseline comparison over N episodes | Not started — CLI stub |
| 6 | Read-only Dash dashboard over `runs/` | Not started — stub |
| 7 | README polish: results, GIF, future-work section | Not started |
| 8 | Future work: opponents, tire/fuel/ERS, Strategy Agent, two-agent coupling | Out of MVP scope; needs explicit go-ahead |

**Next up:** phase 5, `eval.py` head-to-head against the baseline. Likely levers to
close the 1 s gap: longer training (the curve hadn't flattened), and checking
whether the overspeed penalty makes it lift too early.

## Known gaps

- Track width is a design-choice default, not data-derived (FastF1 has no width channel).
- The FastF1 fetch has no automated test, to keep `uv run pytest` offline-safe; only the
  position-processing math is unit-tested.
- No tire, fuel or ERS models yet (phase 8).
- Baseline numbers above are on the synthetic tracks only. Real circuits work through
  `--track <year>:<grand prix>` but need FastF1 network access on first load.
- The baseline reads exact distance and offset from `info`, which is more than the
  RL policy's observation gives it. Fine for a reference lap time; worth knowing when
  comparing.
