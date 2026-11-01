# FormulaOneRL: the whole project explained

This document explains, in plain language, what FormulaOneRL is, every agent in it,
how each one was trained, what we trained them on, the tools we used, the features
of the finished system, and the problems we ran into along the way and how we
solved (or did not solve) them. Numbers come from the project's logged runs; the
run-by-run detail is in [`docs/project_status.md`](project_status.md),
[`docs/training_log.md`](training_log.md) and
[`docs/strategy_agent_design.md`](strategy_agent_design.md). The academic write-up
is [`docs/report/FormulaOneRL_Report.md`](report/FormulaOneRL_Report.md).

---

## 1. The idea in one paragraph

A Formula 1 result depends on two people: the **driver**, who decides many times a
second how to drive, overtake and defend, and the **race strategist**, who decides
once a lap when to pit, which tyres to fit and how hard to push. FormulaOneRL
models both as reinforcement learning (RL) agents in our own fast simulator built
on 25 real circuits. The Driver Agent learns race-craft (when to attack, which
side, when to cover) on top of a car that already drives the racing line at the
grip limit, with a safety layer that makes contact impossible. The Strategy Agent
learns pit and tyre decisions over full races, using numbers measured from our own
driver, and learns to handle safety cars, rain, spins and punctures.

## 2. How the project grew (timeline)

| Stage | What we built | Outcome |
|---|---|---|
| Phase 0-1 | Project skeleton (uv, tests), a track made of curvature segments, a point-mass car | Car physics as pure functions with unit tests |
| Phase 2 | `DriverEnv`, the first Gymnasium environment | Passes Gymnasium's environment checker |
| Phase 3 | Rule-based baseline driver and lap analytics | Baseline 2.3-3.0% off the limit lap |
| Phase 4 | PPO driver (throttle + steering) | Clean laps on a test track and six real circuits, 1-4% behind the baseline |
| Phase 4R | Real TUM circuits (25), Frenet-frame steering car, optimal racing line, SAC driver | SAC plateaued 17-20% off the limit lap |
| Driver v3 | `PathDriver` planner (+1.1% off the limit) and race-craft moves; `TacticalEnv`; PPO picks moves | Driver can now overtake and defend |
| Step 3 | Real F1 overtakes (FastF1/OpenF1) as start states; no-crash safety shield | 0 contacts on 1,174 held-out real situations |
| Overtake v4 | Fixed the simulated world so passing is possible (tow, drag, shield in the tow) | Scripted passing went from ~8/36 to 25/36 |
| Sections | Corner-by-corner overtaking battles, strict clean-racing rules, direct lane moves | Learned driver matches the scripted tactician |
| Sparring partners | Scripted "guard" defenders and an RL-trained defender | 0 contacts in 480 guard battles |
| Defend run 2 | Main driver also trained to defend | Held-out defences 273 vs 264 (line) and passes 49 vs 29 |
| Real pace | Realistic 0.4-2 s/lap pace gaps, three sim fixes, committed lane moves | Overtaking at small gaps still rare |
| Strategy Agent | Lap-by-lap race model, `StrategyEnv`, PPO strategist on the measured driver | 72% wins vs 10% for the rule-based plan; rain 68% vs 0% |
| Strategy, every case | Fast car from P10, spins, punctures, SC/VSC, red flags, ERS overtake mode | 96% clean wins vs 91%, 29% after a spin vs 4.5% |

## 3. The agents

There are two learning agents (the Driver and the Strategist) plus several
supporting agents: scripted baselines, sparring opponents, and an RL-trained
defender.

### 3.1 Driver Agent (main RL agent, low level)

* **Job:** drive the car fast, overtake cleanly, defend cleanly, never crash.
* **Decides:** every 0.5 s, one of 17 moves:
  `line`, `outside`, `divebomb`, `dummy` (feint), `switchback`, `cover`, `inside`,
  `pull_in`, `pull_out`, and eight lane moves (`left`/`right` of the car ahead at
  3.0 m or 4.2 m, each with normal or late braking).
* **Sees (75 numbers):** speed, line offset, heading, corner speed limits in 12
  windows up to 250 m ahead, the optimal line ahead, tyre wear, push level, the
  nearest cars ahead and behind (gap, speed, side), the move it is running, the
  next two corners and metres left in the battle, and how the car ahead moved.
* **How the move is driven:** a scripted `PathDriver` plans speeds and braking points
  on the path the move asks for, so the car is always driven at the grip limit.
* **Safety:** every command from every car goes through the safety shield (below).
* **Algorithm:** PPO (Stable-Baselines3), 256x256 network, warm-started by copying
  the scripted tactician (behaviour cloning), then trained with RL.
* **Best model:** `runs/defend_sections_v2/model_best.zip` (laptop), 500k decisions.

### 3.2 Strategy Agent (main RL agent, high level)

* **Job:** finish a full Grand Prix as fast as possible and win.
* **Decides:** once a lap, `[pit: stay out / soft / medium / hard, push level 0-4]`
  (the first action picks the starting tyre). The "master" version adds wet tyres
  and an ERS overtake button.
* **Sees (46 numbers, 62 in the master version):** race progress, place, gaps,
  tyres, fuel, two-compound rule status, the lap-time gain of each fresh compound,
  tyre life vs laps left, pit window, where a stop would rejoin, rivals' tyres,
  safety-car/VSC flags, incident flags, and (rain configs) wetness and forecast.
* **Simulator:** `StrategyEnv`, one step per lap, 10 cars, our measured driver.
* **Algorithm:** PPO, 64x64 network (about 15k parameters).
* **Reward:** minus race time (0.1 per second) plus +20 for a win, -50 for breaking
  the two-compound rule. Earlier versions also paid places gained.

### 3.3 Supporting agents

| Agent | Type | Role |
|---|---|---|
| `BaselineDriver` (`agents/baseline.py`) | rule-based | cautious v1 driver, the first benchmark |
| `LineFollower` (`agents/line_follower.py`) | rule-based | follows the optimal line, +2.3% off the limit |
| `PathDriver` (`agents/path_driver.py`) | rule-based | plans speed on its own path, +1.1% off the limit; executes the RL driver's moves |
| `RacerAgent` SAC (`agents/racer.py`) | RL (SAC) | pure end-to-end throttle + steering driver; plateaued at +17-20% |
| Tactician (`agents/tactician.py`) | rule-based | scripted race-craft expert; baseline and teacher for behaviour cloning |
| Opponents (`envs/opponents.py`) | rule-based | backmarker, slow, equal, defender, attacker, close, mid |
| Guard defenders (`agents/guard.py`) | rule-based | one legal inside move per corner; sparring partner for overtaking |
| RL defender (`agents/defender_rl.py`) | RL (PPO) | learned defender (line / cover / inside); sparring partner |
| Frozen self (`rival_behind`) | RL (frozen copy) | our own overtaking driver used as the attacker when training defence |
| `FixedPlan` (`agents/strategy_baseline.py`) | rule-based | medium-to-hard one-stop at 45%; strategy baseline and rival strategies |

## 4. What we trained on (scenarios and examples)

### 4.1 Circuits

25 real circuits from the TUM racetrack database. Fixed split:
* **Train (17):** Austin, Brands Hatch, Budapest, Indianapolis (IMS), Melbourne, Mexico City, Monza, Moscow Raceway, Norisring, Nürburgring, Oschersleben, Sakhir, Sepang, Shanghai, Sochi, Spielberg, Zandvoort.
* **Validation (3):** Montreal, Hockenheim, Catalunya - used to pick the best checkpoint.
* **Held-out test (5):** Suzuka, Spa, Sao Paulo, Silverstone, Yas Marina - never used for training or selection.

### 4.2 Driver training scenarios

| Scenario | What happens | Used for |
|---|---|---|
| Solo flying lap, push | lap as fast as possible | pace |
| Solo flying lap, conserve | lap while saving tyres | tyre management |
| Pass backmarker / slow / equal / defender | catch and pass a car of each type | overtaking |
| Defend | half-worn tyres, quicker car behind | defending |
| **Real starts** | a real 2021-2025 F1 pass or defence, started 400 m before the real spot; car ahead is `slow` if its tyres were 8+ laps older, else `defender` | realistic attack/defence, held-out test |
| Section battles | one straight, braking zone, corner entry or exit, vs `slow` or `guard_slow` | corner-by-corner overtaking |
| Two-corner battles | battle runs until the next corner's exit | completing passes |
| Battles vs the RL defender | car ahead driven by the trained defender | harder overtaking |
| Defend sections | our car ahead at a straight or braking zone, tyres 0-70% worn, attacker 6.5-16 m behind | defending |
| Self-play pilots | both cars learn | tried, entropy collapsed |
| Real pace | opponents 0.4-2 s/lap slower | realistic overtaking |

Real-start library: 2021-2022 from FastF1, 2023-2025 from OpenF1. On the five
held-out circuits it holds 1,174 real situations (495 attacks vs equal-pace
defenders, 92 attacks vs slower cars, 587 defences).

### 4.3 Strategy training scenarios

| Config | Scenario | Held-out result (agent vs rule-based) |
|---|---|---|
| `strategy_ppo.yaml` | pilot with stand-in driver | place 4.20 vs 4.50 |
| `strategy_driver.yaml` | our measured driver, mixed rival plans | place 6.03 vs 6.80 |
| `strategy_win.yaml` | win against nine rule-based rivals | wins 72% vs 10%, podiums 95% vs 33% |
| `strategy_rain.yaml` | showers, intermediates, wets, forecast | wet races: wins 68% vs 0% |
| `strategy_fast.yaml` | fastest car (3%) from 10th | wins 88% vs 30% |
| `strategy_fast_skill.yaml` | 0.5% faster, better overtaker | wins 45% vs 15%, 11.4 passes/race |
| `strategy_fast_win.yaml` | 1% faster, win-only reward | 200/200 wins vs 91% |
| `strategy_recover.yaml` | a spin or puncture every race | place 6.9 vs 8.1 |
| `strategy_all.yaml` | every case: clean, spin, puncture, SC, VSC (1% car) | 1 seed, no min-stint rule: clean 96 vs 91%, spin 29 vs 4.5%, SC 94.5 vs 89.5%; 3 seeds with the rule: 96.7 / 29.8 / 78% |
| `strategy_all_dominant.yaml` | every case, 3% car | ~100% except puncture 56 vs 44% |
| `strategy_master.yaml` | adds rain, red flags, second SC, restarts, ERS, early-stop rule | in progress when this was written |

## 5. Tools and technologies

| Tool | Used for |
|---|---|
| Python 3.12 | everything |
| uv | virtual environment and locked dependencies (`uv.lock`) |
| Gymnasium | environment interface (`DriverEnv`, `RaceEnv`, `TacticalEnv`, `StrategyEnv`) |
| Stable-Baselines3 | PPO and SAC implementations |
| PyTorch | neural networks under SB3 |
| NumPy, SciPy | physics, racing line optimisation, curve fits |
| FastF1 | real 2021-2022 lap and position data |
| OpenF1 API | real 2023-2025 race data |
| TUM racetrack database | 25 real circuit centre lines and widths |
| bacinger/f1-circuits | early real circuit outlines (Phase 4) |
| PyYAML | experiment configs in `configs/` |
| matplotlib | report figures |
| pytest, ruff, mypy | tests, lint/format, strict type checks |
| Self-contained HTML pages | replays of laps, battles and races |
| Hardware | AMD Ryzen 7 7840HS laptop CPU (no GPU) and a 4-vCPU cloud container |

## 6. Features of the finished system

* Fast custom simulator: 50 Hz physics, friction-ellipse car, tyre wear, slipstream,
  dirty air, air drag, all seeded and reproducible.
* 25 real circuits with optimal racing lines and limit lap times.
* A planner that laps within 1.1% of the limit on its own.
* 17 race-craft moves, including direct "drive beside the car ahead" lane moves.
* A no-crash safety shield on every car; 0 contacts on 1,174 held-out real situations.
* Strict clean-racing rules: passes must stick (600 m / 1 km), no chopping, no
  squeezing, no tailgating.
* Real-race scenarios built from FastF1 and OpenF1 data.
* Scripted and RL-trained sparring opponents, including a frozen copy of our driver.
* Driver profile measurement (pace gap, pass and defend probability curves).
* Lap-by-lap Grand Prix simulator with compounds, fuel, pit loss, the two-compound
  rule, rain and forecast, safety car, VSC, red flags, spins, punctures and an ERS
  overtake mode.
* PPO Strategy Agent that learned undercuts, rain calls and incident recovery.
* Paired comparison tools (same seed, same race) and held-out scorecards.
* HTML replays: driver comparison, optimal lines, battle replays, race replays.
* One YAML config per experiment; every run logs its config, seed and git commit.
* 341 unit tests (all passing); ruff and strict mypy.

## 7. Problems we faced and how we handled them

### 7.1 Learning to drive

1. **The first PPO driver went flat out into Turn 1.** At 50 Hz with only an
   off-track penalty there was no signal towards braking. *Fix:* 10 Hz decisions,
   a graded over-speed penalty past braking points, and a "safe speed" observation.
2. **It then parked before the chicane.** With a discount of 0.98 (a 5 s horizon)
   stopping looked better than risking a crash. *Fix:* discount 0.995.
3. **It parked at the hairpin instead.** Stopping cost almost nothing. *Fix:* a stall
   (under 2 m/s for 3 s) ends the episode like an off-track.
4. **Braking was imprecise.** *Fix:* the braking margin (safe speed minus speed) as
   its own observation; clean laps from 150k decisions.
5. **The RL driver coasted on straights** and lost all its corner gains there: a
   cautious optimum.
6. **The SAC driver plateaued 17-20% off the limit.** Finer look-ahead, reward
   scaling, learning-rate decay, 25 Hz control, demonstrations, optimal-line
   features and residual learning on top of the line follower all failed to remove
   the plateau. *Fix (design change):* let a planner drive and let RL choose moves.

### 7.2 Real data

7. **No steering data, positions only ~4 Hz.** Real data cannot teach driving by
   imitation. *Fix:* use real passes as start situations and as a test set.
8. **FastF1 lost access to most years** (the live-timing server started requiring a
   login in October 2026). *Fix:* FastF1 for 2021-2022, OpenF1 for 2023-2025;
   2018-2020 skipped.
9. **The cloud container cannot reach F1 data hosts.** *Fix:* data pulls run on the
   laptop.
10. **A tyre-wear fit from one race was not identifiable** (tyre age and fuel burn
    move together). Tyre values remain labelled design choices.

### 7.3 Racing other cars

11. **Contacts.** *Fix:* the predictive safety shield; contacts 20 to 0 on 424
    held-out starts. Four rounds of shield fixes followed (fallback as a candidate,
    no brake-testing, no off-road swerve beside a car, wider steering and horizon).
12. **The driver stopped attacking (run 1).** A pass paid 20 while holding position
    paid almost the same. *Fix:* pass bonus 100 and more exploration (entropy 0.03).
13. **Reward hacking: cars swapped places back and forth.** The agent was paid for
    passes that did not last (434/495 "wins", only 7/49 under strict scoring).
    *Fix:* a pass pays only after it holds 600 m; losing a place costs the same;
    strict battle scoring.
14. **The simulated world made passing impossible.** The shield braked the car behind
    even in the slipstream, the tow only raised top speed, cars planned for dirty air
    from 75 m back, and scripted cars jumped aside. *Fix:* the v4 world (shield off in
    the tow, tow acceleration, 25 m wake, no yielding, track-limit margins).
15. **One section was too short to finish a pass.** *Fix:* two-corner battles and air
    drag.
16. **Slower cars braked 25-30 m early and were slow on straights,** which made passes
    too easy and unrealistic. *Fix:* opponents now brake and accelerate like us and
    lose time only in corners.
17. **The defender kept going off track.** The planner could not brake with steering
    at full lock, and the shield kept a crashing command when nothing was clear.
    *Fix:* braking while steering is saturated; take the safest option even off-road.
18. **The chop rule punished the defender for driving its own racing line.** *Fix:*
    a chop is only a step further off our own line towards the attacker.
19. **The RL defender collapsed to "never defend"** in its second run. We kept the
    first run's best model.
20. **Self-play did not learn**: entropy collapsed to about 0.3 nats and close cars
    crawled because of the shield. The shield was fixed; self-play remains future work.
21. **The driver profile was mis-measured:** opponents' planned grip leaked into our
    pace curve. Three more sim bugs found with it (DRS ended when noses overlapped,
    the shield held a car off its line, dirty air cut straight-line acceleration).
    *Fix:* measure opponent laps directly; opt-in flags for the fixes.
22. **Overtaking at realistic gaps (0.4-1.2 s/lap) is still rare** for both learned
    and scripted drivers. An oracle shows the ceiling is low with the current moves.

### 7.4 Strategy

23. **A win-only reward did not learn.** A random policy pits three laps in four and
    never wins. *Fix:* a "stay out" bias at initialisation.
24. **The incident agent did not learn with a win-only reward.** *Fix:* dense
    race-time reward.
25. **The agent stopped on two or three safety-car laps in a row.** *Fix:* minimum
    stint of 5 laps.
26. **The rain-trained agent was worse in dry races** (25% vs 72% wins). Use the dry
    agent in the dry for now.
27. **Rivals never react to an undercut,** which flatters early stops. Open.

### 7.5 Engineering

28. **Cloud container restarts when idle**, killing long background runs. Long runs
    moved to the laptop.
29. **`uv sync` stalled on CUDA wheels in the cloud** at times; a CPU-only scratch
    environment worked.
30. **The shield costs about 35% of simulation speed**, which limits experiments.

## 8. Main results at a glance

| Question | Answer |
|---|---|
| How fast is the planner? | +1.1% off the limit lap on 24 circuits |
| How fast is pure RL (SAC)? | +17-20% off the limit lap |
| Contacts on held-out real situations? | 0 of 1,174 for the trained driver |
| Defences held (held-out, worn tyres)? | 273 of 460 vs 264 line-holding, 257 scripted |
| Clean passes (held-out)? | 49 of 350 vs 29 for the starting model |
| Strategy wins vs rule-based (measured driver)? | 72% vs 10% |
| Strategy wins in rain? | 68% vs 0% |
| Strategy wins, 1% faster car: clean / spin / SC? | 3 seeds: 96.7 / 29.8 / 78% vs 91 / 4.5 / 89.5% (SC is a known weak spot) |
| Strategy wins, fastest car, win-only? | 200 of 200 |

## 9. Where things live

| Path | What |
|---|---|
| `src/f1rl/envs/` | `DriverEnv`, `RaceEnv`, `TacticalEnv`, `StrategyEnv`, tracks, opponents |
| `src/f1rl/models/` | car, Frenet frame, racing line, tyres, weather, race model, safety shield |
| `src/f1rl/agents/` | all agents listed in Section 3 |
| `src/f1rl/` | `train.py`, `eval.py`, `scenarios.py`, `sections.py`, `incidents.py`, `openf1.py`, `real_starts.py`, `driver_profile.py`, `config.py` |
| `configs/` | one YAML per experiment |
| `scripts/` | gates, oracles, demos, driver measurement, report figures |
| `tests/` | unit tests |
| `docs/` | this file, the report, status, training log, designs, work split |
