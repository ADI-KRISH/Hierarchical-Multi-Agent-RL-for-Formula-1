# Racing lines: what other projects do, and how we get there

Research notes, 2026-10-06. Goal: a driver that finds **the best racing line** and laps
**as fast as possible**, trained **across many tracks**. This doc answers why our
driver is slower today, walks through seven open-source projects one at a time, and
turns what they do differently into a plan.

Repos reviewed (shallow clones, latest commit at time of review):

| Repo | Commit | Licence | Kind |
|---|---|---|---|
| [TUMFTM/racetrack-database](https://github.com/TUMFTM/racetrack-database) | e59595d (2021-09) | LGPL-3.0 | Data: 25 real tracks with widths and racing lines |
| [vaul-ulaval/f1tenth-raceline-optim](https://github.com/vaul-ulaval/f1tenth-raceline-optim) | b78d133 (2023-12) | LGPL-3.0 | Optimal-control raceline (TUM fork) + GUI |
| [CL2-UWaterloo/Raceline-Optimization](https://github.com/CL2-UWaterloo/Raceline-Optimization) | 9290c5d (2023-12) | LGPL-3.0 | Optimal-control raceline (TUM fork) for F1TENTH |
| [agi-brain/Autonomous-Race-with-AM-RL](https://github.com/agi-brain/Autonomous-Race-with-AM-RL) | 001de54 (2024-06) | MIT | RL (TD3) with action mapping |
| [KGolemo/f1-racing-line-optimization](https://github.com/KGolemo/f1-racing-line-optimization) | aea9055 (2023-02) | none stated | RL (SAC) racing line on Monza |
| [Krishnanshu-Gupta/F1-Racing-Optimal-Path](https://github.com/Krishnanshu-Gupta/F1-Racing-Optimal-Path) | 69304ee (2024-11) | none stated | Neuroevolution + DQN on random tracks |
| [subinium/awesome-f1](https://github.com/subinium/awesome-f1) | 216e69f (2026-09) | none stated | Curated list of F1 data and tools |

Repos without a licence can be learned from but not copied from. The LGPL-3.0 data and
code can be used as libraries or data downloaded at runtime, which is how we already
treat the f1-circuits dataset.

---

## 1. Why our driver is slower

There are two separate reasons. The first is much bigger, and none of the tuning so
far could have touched it.

### 1.1 Our simulator has no racing line to find

In `DriverEnv` the car is a point on the track's **centreline**. Its sideways offset
from the centreline only counts toward going off track. It never changes the radius
of the corner the car is driving. So the agent can learn throttle and brake, but
outside-apex-outside (using the track's width to open a corner up) does nothing for it.
Every repo below that produces a racing line models the car's 2D path instead.

How much time that costs, measured in **our own physics** (`models/lap.py` limit lap,
`EXAMPLE_CAR`) on TUM's 25 tracks, driving the centreline vs TUM's minimum-curvature
racing line (`scripts/raceline_gain.py`, reproducible from the database in a minute):

| Track | Width | Limit lap, centreline | Limit lap, racing line | Gain |
|---|---|---|---|---|
| Austin | 13.9 m | 83.2 s | 72.9 s | -12.4% |
| Yas Marina | 12.9 m | 85.4 s | 74.9 s | -12.3% |
| Moscow Raceway | 11.7 m | 66.1 s | 58.2 s | -12.1% |
| Melbourne | 12.3 m | 74.8 s | 65.9 s | -11.8% |
| Oschersleben | 10.6 m | 55.9 s | 49.3 s | -11.7% |
| Sochi | 12.6 m | 83.1 s | 73.4 s | -11.6% |
| Silverstone | 13.8 m | 80.1 s | 71.1 s | -11.3% |
| Mexico City | 12.2 m | 64.7 s | 57.7 s | -10.9% |
| Hockenheim | 12.6 m | 64.8 s | 57.9 s | -10.7% |
| Sepang | 14.6 m | 78.8 s | 70.9 s | -10.1% |
| Sakhir | 13.4 m | 75.9 s | 68.5 s | -9.7% |
| Norisring | 15.9 m | 33.6 s | 30.4 s | -9.4% |
| Sao Paulo | 11.9 m | 60.2 s | 54.6 s | -9.3% |
| Brands Hatch | 9.2 m | 53.0 s | 48.2 s | -9.2% |
| Budapest | 10.0 m | 65.5 s | 59.6 s | -9.0% |
| Shanghai | 13.0 m | 78.2 s | 71.7 s | -8.4% |
| Catalunya | 11.2 m | 66.6 s | 61.0 s | -8.3% |
| Nuerburgring | 11.8 m | 73.0 s | 66.9 s | -8.3% |
| Zandvoort | 10.5 m | 62.0 s | 57.2 s | -7.9% |
| Suzuka | 9.8 m | 77.8 s | 71.7 s | -7.8% |
| Montreal | 9.7 m | 60.5 s | 56.0 s | -7.5% |
| Spielberg | 11.0 m | 56.7 s | 52.9 s | -6.8% |
| Monza | 9.4 m | 72.1 s | 67.4 s | -6.4% |
| Spa | 9.8 m | 90.8 s | 85.0 s | -6.4% |
| IMS (oval) | 15.3 m | 41.5 s | 41.2 s | -0.7% |

**Mean gain: 9.2%.** Our RL driver's gap to the rule-based baseline is 1.6% on the
technical track. The racing line is worth roughly six times that, and no change to
the reward or the algorithm can get it while the car is locked to the centreline.

### 1.2 Within the centreline model, PPO settles on a speed governor

This part is covered in `PROGRESS.md` ("Why is the RL driver slower than the
baseline?"). In short: the agent beats the baseline in every corner but backs off the
throttle on straights, the faster it goes. Braking 100 m later on one straight is
worth ~0.003 reward, while a crash or overspeed costs ~1.0, so PPO never leaves the
safe habit. Eight experiments ruled out undertraining, exploration noise and the
control rate; stronger time incentives, a braking-point countdown input and PPO
stabilisers narrowed the gap from 1.0 s to 0.7 s.

### 1.3 Why lap times sometimes got *worse* later in training

In the six-circuit run and the braking-marker run, eval lap times reached their best
partway through and then drifted back up (e.g. Monaco 66.2 s at 1.6M decisions, 69.6 s
at 2.7M). That is PPO instability with a constant learning rate: late updates keep
moving the policy after it has found a good region. Larger rollouts, a linear
learning-rate decay to zero and a KL limit removed the drift (`driver_ppo_stable`
improved monotonically to 43.90 s). `model_best.zip` already protects us from shipping
a drifted model, but every long run should use those stabilisers.

---

## 2. The repos, one by one

### 2.1 TUMFTM/racetrack-database: the data we were missing

**What it is.** Centrelines plus **left and right track widths** for 25 circuits
(21 F1 venues incl. Spa, Suzuka, Sepang, Sao Paulo, Zandvoort; plus DTM/IndyCar), and a
**minimum-curvature racing line** for each. Centrelines come from OpenStreetMap GPS,
smoothed; widths were extracted from satellite images; racing lines from TUM's
optimiser. Format: CSV `x_m, y_m, w_tr_right_m, w_tr_left_m` in metres, ~5 m spacing.

**What it does differently from us.** We have centrelines only (f1-circuits GeoJSON)
and a made-up 12 m width. TUM has measured widths (9.2-15.9 m, varying along each
track) and a reference racing line for every track.

**What to take.**
- Load all 25 tracks as a third circuit source (download and cache at runtime, like
  f1-circuits; LGPL data, not vendored). That takes us from 6 real circuits to ~27.
- Use the real per-point widths: `Segment.width_m` already supports per-segment width.
- Use TUM's racing line as a **benchmark**: the limit lap on that line is the new
  ceiling, and the RL driver's line can be scored by how close it gets.
- The README warns data quality varies by location, so spot-check each track (we
  already do a closure and length check).

### 2.2 CL2-UWaterloo/Raceline-Optimization: optimal control, the "answer key"

**What it is.** A fork of TUM's
[global_racetrajectory_optimization](https://github.com/TUMFTM/global_racetrajectory_optimization),
adapted for the F1TENTH 1:10 cars. Given a centreline and widths, it computes a racing
line by **optimisation, not learning**, in one of four ways:
- *shortest path*: minimise length;
- *minimum curvature*: minimise summed squared curvature as a quadratic programme
  (QP). The line is parametrised as `alpha_i`, a lateral shift along the normal at
  each centreline point, bounded by the width minus the car width. It can be iterated
  (IQP) to fix the curvature linearisation;
- *minimum time*: a full optimal-control problem (double-track vehicle model,
  Pacejka tyres, optional friction maps and powertrain) solved with CasADi + IPOPT;
- a velocity profile then comes from a forward/backward pass over a g-g-v diagram
  (the same idea as our `models/lap.py`, plus drag and speed-dependent limits).

It adds a `map_converter` notebook (occupancy-grid map -> centreline + widths) and
exports for a pure-pursuit controller. Ships tracks incl. `Spa_map.csv`, Berlin, Modena.

**What it does differently.** No learning at all. It gets the optimal line for the
model in seconds to minutes, per track. Its car model is much richer than ours.

**What to take.**
- The **`alpha` parametrisation**: a racing line = one lateral offset per station,
  bounded by the track width. It is the cleanest way to represent "a line" in our
  env, and the form a learned policy's output can be compared to.
- **Minimum-curvature QP as our optimal reference**. It is small (one variable per
  station, linear constraints) and can be solved with `scipy.optimize` / a QP solver
  in our own pure function, giving `optimal_line(track) -> alpha` and a
  "limit lap on the optimal line" for any track, including ones TUM doesn't cover.
- **Velocity from a g-g-v diagram**: our limit lap uses constant 5 g in every
  direction. A speed-dependent grip and drag model (only with cited numbers) is what
  makes "minimum curvature" and "minimum time" lines differ.

### 2.3 vaul-ulaval/f1tenth-raceline-optim: the same optimiser with a GUI

**What it is.** Another TUM fork for F1TENTH (Laval University), with the same four
optimisation modes and parameter files. Its additions are a PySide6 GUI (pick a map,
**generate a centreline from an occupancy-grid image** by distance transform +
skeletonisation, compare several racing lines on one map), a Dockerfile, and
example racelines for the ICRA 2023 F1TENTH track.

**What it does differently from CL2.** Tooling, not method: centreline extraction
from an image, and side-by-side **raceline comparison** in a viewer.

**What to take.**
- Distance-transform + skeleton centreline extraction is how we could add tracks for
  which only an image exists (and get widths for free, from the distance transform).
- The **compare-racelines view** is a good feature for our report: overlay the RL
  agent's line, the baseline's line and the optimal line on one map.

### 2.4 agi-brain/Autonomous-Race-with-AM-RL: the RL design closest to our goal

**What it is.** Code for the ISA Transactions 2024 paper *Learning autonomous race
driving with action mapping reinforcement learning*. A car learns to lap with **TD3**
(off-policy actor-critic). Key pieces:
- **Car model**: a bicycle model, dynamic longitudinally (motor power/torque, drag,
  rolling resistance) and kinematic laterally, RK4 at 100 Hz. The car steers, so the
  path, and with it the racing line, is the agent's to choose.
- **Action mapping (AM)**: the agent outputs a vector `(ux, uy)` in the unit square.
  A precomputed 200^4 lookup table (speed x steer x direction x amplitude) shrinks
  that vector onto the set of commands that keep the tyres inside the friction limit
  *at the current speed and steer*. The policy **cannot ask for an infeasible
  command**, so exploration doesn't waste episodes on grip violations, and the same
  policy transfers to a different friction level by swapping the map.
- **Observation**: speed, yaw rate, steer angle, lateral offset, heading error, plus
  **13 centreline points ahead in the car's frame** (5, 10, 20, 30, 40 m, then every
  20 m out to 200 m). Generic geometry, not features specific to one track.
- **Reward**: `speed * cos(heading error) / 10` every step (speed along the track),
  -100 on failure. Fail = off track, wrong way, grip exceeded, or speed < 6 m/s.
- **Random starts**: every episode starts at a random place on the track, random
  lateral offset, random speed 10-20 m/s. Every corner gets practised equally often,
  not just the first one.
- Result: a 36.9 s flying lap on their Track A, better lap times and success rates
  than RL without AM.

**What it does differently from us.** Steering changes the path; an action space that
can't exceed grip; generic lookahead geometry; random starts; off-policy TD3.

**What to take (most of it).**
- **Action mapping** answers our speed-governor problem directly: we saw PPO stay
  slow because over-the-limit actions are so costly. With a mapped action space, the
  agent's command "as hard as possible" means *the* limit, never beyond it. For us the
  map is analytic (friction circle: `a_x^2 + a_y^2 <= (mu g)^2`), so no lookup table.
- **Random starts** along the lap, with random speed: practises late braking at every
  corner, every episode. That is the curriculum `PROGRESS.md` suggested.
- **Car-frame lookahead points** as the observation: the natural input for a driver
  that must work on tracks it has never seen.
- **Off-policy TD3/SAC**: both are in Stable Baselines3, and AM-RL and KGolemo both
  used one. Replay makes them far more sample-efficient than PPO for continuous
  control.

### 2.5 KGolemo/f1-racing-line-optimization: RL on Monza with checkpoint gates

**What it is.** A pygame racer on a Monza layout traced from an image; **SAC** via Ray
RLlib (4 workers, 4,000 iterations).
- **Car**: arcade kinematics (heading, speed, steering rate that falls with speed).
  No grip model: the track walls are the only limit.
- **Observation**: 7 ray-cast distances to the walls (-60 to +60 degrees) + speed.
- **Reward ("laptime" mode)**: 121 **checkpoint gates** across the track. Crossing a
  gate pays `max(1, 100 - frames since last gate) / 100` (sooner = more); going
  backwards through one costs -1. Hitting a wall ends the episode.

**What it does differently.** Lidar-style perception rather than map features; a
time-to-next-gate reward; SAC; but one track and no tyre physics, so its line is
"fastest without touching walls", not grip-limited.

**What to take.**
- **Gate-time reward** as an alternative to our per-step progress reward: it pays
  directly for *time between gates*, which is exactly lap time, split into ~120 parts.
  Worth one A/B experiment against our current reward.
- **Ray-cast distances** as an optional observation (easy in Frenet coordinates:
  distance to the left/right edge along a few headings). Useful for generalisation.
- Warning sign: one track only, so the policy is free to memorise it. We should not.

### 2.6 Krishnanshu-Gupta/F1-Racing-Optimal-Path: random tracks and evolution

**What it is.** A pyglet simulator (built on Tomas Brezina's NeuralNetworkRacing) whose
cars learn on **randomly generated tile tracks**. Two learners:
- a **genetic algorithm over small neural networks** (elitism 10%, crossover 0.8,
  Gaussian mutation 0.2), with fitness from checkpoints reached and lap time;
- a **DQN** with discrete steering actions and epsilon-greedy exploration, plugged into
  the same evolutionary loop (the report says it was trained on a single random track
  to keep things simple, and only the neuroevolution agent optimises lap time).
- **Observation**: ray-cast sensor distances + speed.

**What it does differently.** Procedural tracks; population-based search; discrete
actions.

**What to take.**
- **Procedural track generation** for training variety: our `Track` is just segments,
  so random tracks (random corner radii, angles, straights, widths, closed by
  construction like `technical_track`) are cheap. Train on many, test on real
  circuits the agent has never seen: the honest test of "a driver", not "a lap".
- **Population-based training** is worth a later look for hyperparameters
  (e.g. reward weights), not as a replacement for gradient RL.
- Not worth copying: DQN's discrete actions (bad for fine throttle/steer control)
  and the single-track RL training.

### 2.7 subinium/awesome-f1: where to find more

A curated list (477 lines) of F1 resources. The relevant entries for us:
- **Track data**: f1-circuits (our current source), TUMFTM/racetrack-database (2.1).
- **Racing-line and lap-time tools**: TUMFTM global_racetrajectory_optimization (the
  parent of 2.2 and 2.3), and **fastest-lap** (juanmanzanero), an optimal-lap-time
  simulator with 3-DOF/6-DOF vehicle models, a second optimal-control reference.
- **Strategy**: TUMFTM race-simulation (pit-stop strategy with RL), relevant to the
  future Strategy Agent.
- **Telemetry**: FastF1 (already a dependency), TracingInsights archives (real speed
  traces to compare our laps against, and to calibrate grip and drag).

---

## 3. What everyone does differently from us, side by side

| | Us (today) | AM-RL | KGolemo | Krishnanshu | TUM / CL2 / Laval |
|---|---|---|---|---|---|
| Car can choose its line | **No** (centreline) | Yes (bicycle) | Yes (arcade) | Yes (arcade) | Yes (optimised `alpha`) |
| Grip limit | 5 g, any direction | Friction limit, mapped actions | None (walls) | None (walls) | g-g-v / tyre model |
| Track widths | 12 m constant | Fixed | From image | Random tiles | **Measured** |
| Tracks | 2 synthetic + 6 real | 1 | 1 (Monza) | Random | 25 real (+F1TENTH maps) |
| Observation | Speed, limits ahead, braking margin | **Car-frame lookahead points** | Ray casts | Ray casts | n/a |
| Reward | Progress - time - penalties | Speed along track | **Gate times** | Checkpoints / lap time | n/a (optimal control) |
| Algorithm | PPO | **TD3** + action mapping | **SAC** | GA + DQN | QP / IPOPT |
| Starts | Grid, standing | **Random place and speed** | Fixed | Fixed | n/a |

---

## 4. Plan: best racing line, fastest lap, many tracks

Each step is testable on its own and keeps the CLAUDE.md rules (custom Gymnasium env,
point-mass/bicycle model, pure-function physics with tests, cited numbers, SB3).

**Step 1: more tracks with real widths.** Add TUM's 25 tracks as a circuit source
(`load_tum_track(name)`, cached under `data/`), with per-station widths. Keep
f1-circuits for the six we have. Hold out ~5 tracks (e.g. Suzuka, Spa, Interlagos,
Silverstone, Yas Marina) that are **never trained on**, only evaluated.

**Step 2: an optimal reference for every track.** A pure function computing the
minimum-curvature line (`alpha` per station, QP with width bounds) and its limit lap.
Check it against TUM's own racing lines (expect matching lap times within ~1%). This
becomes the **new ceiling** in reports, alongside the centreline limit.

**Step 3: let the car choose its line (env v2).** Replace the 1D offset with Frenet-
frame kinematics, the standard way to put a car on a curved track:
`ds/dt = v cos(psi) / (1 - n kappa)`, `dn/dt = v sin(psi)`,
`dpsi/dt = v kappa_cmd - kappa ds/dt`, where `n` is the lateral offset, `psi` the
heading relative to the track and `kappa_cmd` the curvature the car steers. Grip:
`v^2 |kappa_cmd|` (lateral) and the throttle/brake command share one friction circle.
Off track when `|n|` exceeds the local half-width. Physics stays pure functions with
tests; keep the old env for comparison. This is the step that unlocks the ~9%.

**Step 4: borrow AM-RL's RL design.**
- *Action mapping*: actions are a direction and amplitude inside the friction circle,
  scaled to what's available at the current speed. The agent cannot exceed grip.
- *Observation*: car-frame lookahead points of the centreline and both edges
  (e.g. 10 points from 5 to 200 m), plus speed, heading error, offset and yaw rate.
  No lap-progress feature: it lets the policy memorise one track.
- *Random starts*: anywhere on the lap, any feasible offset and speed.
- *Reward*: speed along the track (AM-RL) or gate times (KGolemo); A/B both.

**Step 5: SAC/TD3 in Stable Baselines3, trained across tracks.** Envs dealt
round-robin over the training tracks (already supported), plus procedurally generated
tracks (Krishnanshu). Keep the PPO stabilisers if staying with PPO. Every run logs
per-track evals and replay laps (already in place).

**Step 6: measure what matters.**
- Lap time vs (a) the baseline, (b) the centreline limit, (c) the optimal-line limit,
  on training tracks **and on held-out tracks**.
- Line quality: mean distance from the optimal line; where the agent's line differs.
- Report: overlay agent, baseline and optimal lines on each map (Laval's compare view).

**Step 7 (optional): hybrid.** Give the agent the optimal line as an input and let it
learn a correction on top of it (residual RL). Usually the fastest route to beating
the optimiser's model-mismatch, but less of an "it found the line itself" result.

**What to expect.** Steps 1-2 are a few hours and immediately tell us how far today's
driver is from the real optimum on 25+ tracks. Step 3 is the big change (new env,
retraining everything). Steps 4-5 are where most experiment time goes. A realistic
target for the MVP is: completes every held-out track, within a few percent of the
optimal-line limit, and visibly takes outside-apex-outside lines in the replays.

**Open questions.**
- Car model fidelity: the optimal line differs between "minimum curvature" and
  "minimum time" mainly through speed-dependent grip and drag. Adding those needs
  cited values (FastF1 telemetry can calibrate them; FastF1 is blocked in the cloud
  sandbox but works locally).
- CLAUDE.md allows "point-mass / bicycle model"; Frenet point-mass with heading is
  within that. A full double-track tyre model (as in TUM's minimum-time solver) is not
  needed for the MVP.
