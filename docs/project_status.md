# Project status: done and to do (updated 2026-10-26)

Driver Agent for the F1 RL project (repo `ADI-KRISH/Hierarchical-Multi-Agent-RL-for-Formula-1`).
Run-by-run training numbers are in `docs/training_log.md`; the phase notes are in `PROGRESS.md`.

## Done

### Phase 4R: steering car, real tracks, SAC driver
- 25 real TUM tracks loaded (`src/f1rl/envs/racetrack.py`). Split: 17 train, 3 validation
  (Montreal, Hockenheim, Catalunya), 5 held-out (Suzuka, Spa, Sao Paulo, Silverstone, Yas Marina).
  Held-out tracks are never trained on and never used to pick a checkpoint.
- Optimal racing line per track: minimum-curvature line plus a g-g friction-ellipse lap
  simulator that gives the limit lap time (`src/f1rl/models/raceline.py`).
- Steering car in the Frenet frame (`src/f1rl/models/frenet.py`) and `RaceEnv`
  (`src/f1rl/envs/race_env.py`): one friction ellipse shared by throttle/brake and steering.
- Scripted `LineFollower` (`src/f1rl/agents/line_follower.py`): laps all 25 tracks at about +2.3% off
  the optimal line. This validates that the environment allows the pace.
- SAC training pipeline (`src/f1rl/agents/racer.py`, `train.py`, `eval.py`), seed 0:
  - `racer_sac`: pure RL, 28 observations, 2.0M decisions, 3 h 29 min.
  - `racer_hybrid`: pure RL plus 11 optimal-line features (39 observations), 1.0M decisions, 2 h 11 min.
  - Six experiment runs (finer lookahead, reward scale, 25 Hz, demonstrations, lower fail penalty).
    Total about 10.8 h of training.
- Results on the 5 held-out tracks: pure RL best checkpoint completes 4/5 at +19.2% off the optimal
  line; hybrid about +19-21% (5/5 completion at its 450k checkpoint in training eval). Neither beats
  the v1 PPO baseline by much; the scripted follower is far ahead at +2.3%.
- Reward: distance advanced / 10 m, minus 50 on failure (leave track, wrong way, below 10 m/s),
  minus 0.1 x change in pedal and steering commands. No lap-time or racing-line term.
- Tools and visuals: `scripts/compare_drivers.py` (animated comparison page, self-contained HTML),
  `scripts/plot_raceline.py` (optimal lines page), `docs/sac_diagram.svg` (how SAC works).
- Tests: 143 passing (`tests/test_racetrack.py`, `test_raceline.py`, `test_frenet.py`, `test_race_env.py`).
- Docs: `docs/training_log.md`, `docs/strategy_agent_design.md` (Strategy Agent plan), this file.

### Driver agent v3: tactical driver (2026-10-19, by user request: perfect the driver before any Strategy Agent)
- **Pace.** New `PathDriver` (`agents/path_driver.py`) plans corner speeds and braking points on the path
  it is actually driving, instead of reading them off the racing line. With no sideways move it laps
  the 24 usable TUM tracks at **+1.1% on average (worst +1.5%)** off the optimal line, all completed
  (the old `LineFollower`: +2.4%). Residual SAC on top of the old follower (`racer_residual`, 1M decisions)
  never beat the follower (+2.6-2.8%), so that route is dropped.
- **Race-craft moves** (`agents/tactics.py`): outside, divebomb, dummy (feint), switchback, cover,
  inside-defence, plus `line`. Each is a list of sideways-shift knots plus how late it brakes, built
  relative to the next corner. A grid over every corner of all 24 tracks (337 corners): each attacking move
  finishes without leaving the track 99.4% of the time (switchback 97%).
- **Opponents** (`envs/opponents.py`, kinds in `config.OPPONENT_KINDS`): backmarker, slow, equal,
  defender (covers once, takes the inside), attacker (divebombs). They give room to a car alongside and
  never move in a braking zone.
- **Tyres and modes**: `models/tire.py` (wear, grip loss), push level 0..1 sets the grip the driver plans for.
  Tyre, wake (4% grip) and overtake-assist (+6 m/s on straights within 80 m of the car ahead) constants are
  **design choices, not F1 figures**.
- **`TacticalEnv`** (`envs/tactical_env.py`): the action is a move, chosen every 0.5 s; a shield brakes
  rather than hit a car ahead. **`agents/tactician.py`** is the rule-based expert; **`agents/tactical.py`**
  trains PPO (behaviour-cloned from the tactician first); `configs/tactical.yaml`.
- **Scenario report**: `uv run python -m f1rl.scenarios --policy line|tactician|<model.zip>`.
  8 tracks (held-out + validation), flying laps:

  | scenario | no moves (`line`) | rule-based tactician |
  |---|---|---|
  | push (solo lap gap) | 8/8, +1.05% | 8/8, +1.05% |
  | conserve (solo lap gap, wear) | 8/8, +3.6%, wear 0.017 vs 0.019 | same |
  | pass backmarker | 0/8 | 6/8 |
  | pass slow (-8% corner speed) | 0/8 | 3/8 |
  | pass equal pace | 0/8 | 0/8 |
  | pass defender | 1/8 | 2/8 |
  | defend (half-worn tyres vs quicker attacker) | 6/8 | 6/8 |

  Passing on pace alone needs about a 10-15% corner-speed edge, because every move costs time.
- Real data check: FastF1 loads locally (Abu Dhabi 2021 laps load). A tyre-degradation fit from one race
  is not identifiable (tyre age and lap number are collinear; fuel burn mixes in), so tyre constants stay
  design choices until a multi-race fit is done.
- Tests: 173 passing.

### Step 3: real-race starts and the no-crash rule (2026-10-26)
- **Incident library** (`f1rl.incidents`, `f1rl.openf1`): real passes and defences per race in
  `data/incidents/` (FastF1 2021-2022, OpenF1 2023 on; 2018-2020 need a paid F1 TV login, skipped).
  Build with `uv run python -m f1rl.incidents --year <Y>`.
- **Real starts** (`f1rl.real_starts`): each real pass becomes an attack (start 400 m before the
  spot, the car ahead `slow` if its tyres were 8+ laps older, else `defender`) and a defence
  (`attacker` behind); each real defence becomes both, starting at the line. Pass locations line up
  with the TUM tracks' braking zones (checked at Monza, Spielberg, Sakhir, Catalunya), so lap
  distance from the line is used as is. OpenF1 races have no pass locations yet (no telemetry), so
  only their defences are used.
- **No-crash rule** (`models.safety`, `RaceEnvParams.safety_filter`): every car's command is checked
  ahead of time; a command is kept only if both cars could still fall back on the rules of the road
  (keep the lane, the car behind brakes) without touching. Otherwise the least intrusive safe command
  (brake and/or move away) replaces it, never one that leaves the road. The car behind is filtered
  first. Episodes never start in a crash nobody could avoid. Costs ~35% of env speed.
- **Training** (`configs/tactical_real.yaml`): 40% of episodes start from a real train-split
  situation; validation adds real starts; a checkpoint with any contact never counts as best.
- **Scorecard** (`uv run python -m f1rl.scenarios --policy <p> --tracks tum_held_out --real`), rule-based
  tactician on the 5 held-out tracks, 2021-2022 + Japan 2024 library (424 real starts):

  | real starts | no filter: won / contact / off | with filter: won / contact / off |
  |---|---|---|
  | attack vs equal-pace defender (170) | 6 / 6 / 1 | 5 / **0** / 4 |
  | attack vs slower car (42) | 12 / 6 / 1 | 12 / **0** / 3 |
  | defend vs quicker attacker (212) | 187 / 8 / 16 | 188 / **0** / 23 |

  Contacts go from 20 to 0; off-tracks rise from 18 to 30 (the filter's swerves leave the path driver
  less margin), still to fix. Passing an equal-pace defender stays rare (5/170): the sim needs a
  bigger edge to pass than real F1 does (see "pass equal" above).

### Step 3, run 1 and fixes (2026-10-27)
- **Run 1** (`tactical_real`, 1.5M decisions, laptop, ~1 h 50 min): defends well (real_defend 36/37) but
  stopped attacking (real_attack 7/43 after behaviour cloning, 0/43 from 200k on); 1-2 contacts in some
  evaluations. Cause: holding station scored almost the same as trying moves (Spielberg + Monza real
  starts: line 461, tactician 464 per episode), and entropy collapsed after behaviour cloning.
- **Reward change** (invalidates run 1): `overtake_bonus` 20 -> 100 (losing a place costs the same);
  PPO `ent_coef` 0.01 -> 0.03.
- **Safety filter fixes**: (1) the assumed fallback (keep the lane, brake if behind) is itself a
  candidate, so a command judged safe always leaves a safe command next time (fixed defend contacts at
  Monza's first chicane); (2) a car braking with another close behind first tries braking less (no
  brake-testing); (3) a command that is clear of the other car but leaves the road beside it is
  swapped for an on-road one that is just as clear. Check with random moves on 284 real starts
  (train + validation tracks): contacts 2 -> 1, off-tracks 26 -> 3.

### Step 3, run 2 (2026-10-27)
- **Run 2** (pass bonus 100, ent_coef 0.03, filter fixes 1-3): real_attack 37/43 from 200k on (run 1:
  0/43); real_defend 17-29/37 (run 1: 36/37). Best (0 contacts) at 1.1M: real_attack 37/43,
  real_defend 26/37, all pass_* scenarios 3/3. On all 150 validation defend starts the model holds 110,
  beating the scripted tactician (80) and line-holding (65).
- **Filter fix 4**: `safety_steer` 0.4 -> 0.8 and `safety_horizon_s` 1.5 -> 2.5 (max 2.0 -> 3.0). The
  scripted attacker alongside at corner entry (Catalunya T1, Montreal) could still turn in on a car
  holding its line (18 contacts in 150 validation defends). After: 0 contacts; random moves on 284
  real starts: contacts 1 -> 0, off-tracks 3 -> 0. Run 2's model with it on validation real starts:
  attack 59/103 (same), defend 43/47 (was 38), contacts 0 (was 1).

### Step 3 result: held-out scorecard (2026-10-27)
Run 2's `model_best.zip` (1.1M) on the 5 held-out tracks, full 2021-2025 library on the laptop (1,174
real starts), filter fix 4 on, vs the scripted tactician under the same filter:

| real starts | PPO model: won / contact / off | tactician: won / contact / off |
|---|---|---|
| attack vs equal-pace defender (495) | **434** / 0 / 0 | 168 / 0 / 0 |
| attack vs slower car (92) | **72** / 0 / 0 | 59 / 0 / 0 |
| defend vs quicker attacker (587) | **431** / 0 / 5 | 419 / 0 / 1 |

Standard scenarios (held-out, 5 tracks): all pass_* 5/5, 0 contacts; push 4/5 (1 off-track) at +3.1%
over the optimal lap, conserve 5/5 at +5.8%; defend 2/5. Still to do: lap pace under the tactical
layer (+3.1% vs the ~1% goal), the 5 off-tracks when defending, the defend scenario.

### Questions answered (no code)
- Scripted driver, training setup, SAC algorithm, reward function, what the HTML demo embeds.
- Which model for the driver (SAC pure RL now; scripted follower underneath the strategy stage).
- Which model for the Strategy Agent (MaskablePPO, small MLP, once-per-lap decisions).

## To do

### Immediate
- [ ] Push to GitHub on branch `racing-line-sac-driver`: `PROGRESS.md`, `docs`, `src`, `scripts`,
      `tests`, `configs`, plus force-added `runs/racer_sac`, `runs/racer_hybrid`, `runs/*.html`.
      Leave out `TrackA_Video2.gif`. Blocked earlier by the command classifier; run the commands with
      the `!` prefix or ask for a retry.
- [ ] Evaluate `runs/racer_hybrid/model_best.zip` on the 5 held-out tracks with `f1rl.eval`
      (only the training-time eval exists so far).
- [ ] Open `runs/driver_comparison.html` and `runs/optimal_racing_lines.html` in a real browser
      and check they render.
- [ ] Run `uv run ruff check`, `uv run mypy src/` and `uv run pytest` after any further change.
- [x] Train `configs/tactical_real.yaml` and score it with `f1rl.scenarios --real` on the held-out tracks.
- [ ] Bring off-tracks after a safety-filter swerve down (path driver recovery).
- [ ] Pass locations for OpenF1 (2023+) races, so their passes become real starts too.

### Break the 17-20% plateau (driver quality)
- [ ] Residual SAC: learn small corrections on top of the `LineFollower` (starts at +2.3%)
      instead of driving from scratch. Highest-value next step.
- [ ] Pick checkpoints by lap time over several seeds, not completion count (completion swings 0/3-3/3).
- [ ] Longer or varied-length episodes so the agent sees whole laps.

### Strategy Agent (needs an explicit go-ahead per CLAUDE.md)
- [ ] Opponent cars.
- [ ] Tyre, fuel and ERS models (cited constants only, no invented F1 numbers).
- [ ] Race-control layer (pit lane, rules).
- [ ] Lap-time lookup from the g-g simulator for tyre, fuel and mode settings, so strategy training is fast.
- [ ] Non-RL baseline strategy (rule-based or dynamic programming).
- [ ] MaskablePPO Strategy Agent, once per lap, sending a mode vector to the driver.
- [ ] Stage 3 (optional): joint fine-tuning.
