# Progress

Snapshot as of 2026-10-06. Phase definitions and "done when" criteria live in
`docs/roadmap.md`; this file only tracks where we are against them.

**Summary:** phases 0-3 are done, plus extra env realism work beyond the phase 2
spec and a review pass over the env. Phases 4-7 are not started. Nothing has been
trained yet.

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

## Left

| Phase | What | State |
|---|---|---|
| 4 | SB3 PPO training, `configs/driver_ppo.yaml`, logging to `runs/` | Not started — `train.py` is a CLI stub, `configs/` is empty |
| 5 | `eval.py`: agent vs baseline comparison over N episodes | Not started — CLI stub |
| 6 | Read-only Dash dashboard over `runs/` | Not started — stub |
| 7 | README polish: results, GIF, future-work section | Not started |
| 8 | Future work: opponents, tire/fuel/ERS, Strategy Agent, two-agent coupling | Out of MVP scope; needs explicit go-ahead |

**Next up:** phase 4, PPO training on `DriverEnv`.

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
