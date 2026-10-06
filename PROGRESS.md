# Progress

Snapshot as of 2026-10-06. Phase definitions and "done when" criteria live in
`docs/roadmap.md`; this file only tracks where we are against them.

**Summary:** phases 0-4 are done, plus six real circuits. The best PPO driver laps
the technical track in 43.90 s, 0.7 s behind the rule-based baseline (43.19 s mean);
why it is slower is diagnosed below. One driver trained on all six real circuits
completes every lap, 0.8-3.9% behind the baseline. Phases 5-7 are not started.

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

### Real circuits (beyond the roadmap)
- `envs/circuits.py` loads Zandvoort, Spa, Suzuka, Monaco, Sepang and Interlagos from
  GPS centerlines in the open bacinger/f1-circuits dataset (MIT, pinned commit),
  rescaled to official lengths. Works offline after one download, and covers Sepang,
  which FastF1 can't (no telemetry before 2018). FastF1's host is blocked in the cloud
  sandbox, so this is the source used here.
- Limit laps run 61-92 s, roughly 10-15% quicker than real pole laps. That fits the
  model: 5 g of grip at every speed and no aerodynamic drag.
- `configs/driver_ppo_circuits.yaml`: one driver trained on all six at once (envs dealt
  round-robin), 6M decisions. Each eval lap is logged with telemetry, so the report
  can replay how the driving changed during training ("Watch it learn").

Baseline vs RL agent on the circuits (baseline: mean of 20 laps; agent: best eval lap):

| Circuit | Length | Limit lap | Baseline mean | RL agent (best model) | Agent vs baseline | Best eval lap during training |
|---|---|---|---|---|---|---|
| Zandvoort | 4.259 km | 65.79 s | 68.32 s | 69.60 s | +1.9% | 69.57 s |
| Spa | 7.004 km | 95.22 s | 98.06 s | 101.91 s | +3.9% | 101.91 s |
| Suzuka | 5.807 km | 84.37 s | 87.24 s | 89.55 s | +2.6% | 89.55 s |
| Monaco | 3.337 km | 63.24 s | 65.62 s | 66.16 s | +0.8% | 66.16 s |
| Sepang | 5.543 km | 81.89 s | 84.66 s | 86.63 s | +2.3% | 86.63 s |
| Interlagos | 4.309 km | 64.45 s | 66.83 s | 69.01 s | +3.3% | 69.01 s |

### Why is the RL driver slower than the baseline? (technical track)

Where the time goes: the agent is **faster than the baseline in every corner** (by
0.02-0.06 s each) and **loses all of it on the straights** (0.1-0.5 s per straight).
It lifts off long before its braking point and cruises: 6-9 s per lap of coasting while
it could still accelerate, against none for the baseline, and a median margin of 40-62
km/h below the safe speed against the baseline's 15. Its actual braking is excellent:
once braking, it tracks the safe-speed curve within 6-8 km/h.

What was tested (technical track, seed 0, 2M decisions unless noted):

| Hypothesis | Experiment | Best eval lap | Verdict |
|---|---|---|---|
| (original) | `driver_ppo` | 44.18 s | - |
| Undertrained | +450k more decisions from the 2M model | 44.18 s, flat | No |
| Speed pays too little | time penalty 0.01 -> 0.03 /s | 44.09 s | Learns ~2x faster, coasts less (9.4 -> 6.0 s), same plateau |
| Exploration noise makes it cautious | fine-tune with action std 0.2 -> 0.05 | 44.10 s, flat | No |
| Speed gain drowned by penalties | + reward for v / v_safe | 45.32 s at 1M (same curve as without) | No |
| Control rate (10 Hz) caps it | baseline driver at 10 Hz | 41.95 s possible | No: the env allows near-limit laps at 10 Hz |
| Braking point seen too late | + 300 m braking-point countdown observation | 44.80 s at 600k, then drifted | Learns fastest; PPO then degrades |
| PPO instability | countdown + larger rollouts, LR decay, KL limit | **43.90 s** | Best; smooth, no drift |

Conclusion: the environment allows ~42 s laps at the agent's own 10 Hz control rate,
and the agent has the information it needs. The limit is the optimiser. PPO settles
into a speed governor on straights (less throttle the faster it goes) because a few
metres of later lifting is worth ~0.003 reward per straight, while any overspeed or
crash costs ~1.0, so the safe local optimum is never left. Tuning cut the gap to the
baseline mean from 1.0 s to 0.7 s; it did not remove it. Next levers: an off-policy
algorithm (SAC, also in SB3, and listed for the driver in `docs/context.md`), or a
curriculum that starts episodes near braking points so late-braking is practised far
more often than once per lap.

### Racing-line research (2026-10-06)
`docs/racing_line_research.md` reviews seven open-source racing-line projects and
finds the bigger reason the driver is slow: **our env has no racing line**. The car is
locked to the centreline, so it can't use the track's width. On TUM's 25 real tracks,
driving TUM's minimum-curvature line instead of the centreline is **9.2% faster on
average** in our own physics (`scripts/raceline_gain.py`), about 6x the RL driver's
current gap to the baseline. The doc lays out a plan: TUM's tracks with real widths,
a minimum-curvature optimal reference, Frenet-frame steering (env v2), and AM-RL-style
action mapping, random starts and SAC/TD3 trained across tracks with held-out tests.

## Left

| Phase | What | State |
|---|---|---|
| 5 | `eval.py`: agent vs baseline comparison over N episodes | Not started — CLI stub |
| 6 | Read-only Dash dashboard over `runs/` | Not started — stub |
| 7 | README polish: results, GIF, future-work section | Not started |
| 8 | Future work: opponents, tire/fuel/ERS, Strategy Agent, two-agent coupling | Out of MVP scope; needs explicit go-ahead |

**Also next:** retrain the circuits driver with the settings that worked best on the
technical track (`configs/driver_ppo_stable.yaml`); the circuits run predates them.

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
