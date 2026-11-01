# Work split: three members, equal share

FormulaOneRL splits naturally into three equal parts along the system's own layers:
the **simulated world** (track, car, physics, safety), the **Driver Agent** (racing
and race-craft learning), and the **Strategy Agent** (race-level decisions) together
with evaluation and documentation. Each member owns one layer end to end: design,
code, tests, experiments and the matching sections of the report.

Replace "Member 1/2/3" with the team members' names.

## Summary

| | Member 1: Simulator and Safety | Member 2: Driver Agent | Member 3: Strategy Agent and Evaluation |
|---|---|---|---|
| Core question | "Is the world realistic, fast and safe?" | "Can the car race and overtake cleanly?" | "Can the team win the race?" |
| Main RL work | Reward signals for driving, SAC/PPO lap-time baselines | PPO move-level driver, behaviour cloning, curriculum, RL defender | PPO strategist, reward design, scenario curricula, seed study |
| Code owned (lines incl. tests) | ~6,200 | ~6,200 | ~5,200 + report figures and docs |
| Report sections | 3.2 Environment Design, 4.1 Setup (env, hardware), Limitations (simulator) | 3.3 RL Algorithm, 3.4 Novel components (driver), 4.3 driver results, 4.5 Ablation | Abstract, Introduction, Literature Review, 3.5 Pseudocode, 4.2 Performance, 4.4 Comparison, 4.6 Robustness, Conclusion |
| Share | 33.3% | 33.3% | 33.3% |

`src/f1rl/config.py` (all tunable values) is shared: each member adds and documents
the parameters for their own layer.

## Member 1: Simulator, physics and safety

**Goal:** a fast, seeded, realistic racing world on real circuits in which RL can
learn, plus the safety layer that makes contact impossible.

Tasks:
1. Project skeleton: uv environment, `pyproject.toml`, ruff/mypy/pytest set-up.
2. Track models: segment track (`envs/track.py`), real circuits from GPS outlines
   (`envs/circuits.py`) and the 25 TUM circuits with the 17/3/5 split
   (`envs/racetrack.py`).
3. Car physics as pure functions: point-mass car (`models/car.py`), limit-lap
   simulator (`models/lap.py`), Frenet-frame steering car (`models/frenet.py`).
4. Optimal racing line: minimum-curvature line and g-g lap time
   (`models/raceline.py`, `scripts/plot_raceline.py`, `scripts/raceline_gain.py`).
5. Tyre wear model (`models/tire.py`) and manoeuvre geometry (`models/maneuver.py`).
6. Gymnasium environments: `DriverEnv` (Phase 2) and `RaceEnv` (50 Hz, slipstream,
   dirty air, drag, track limits, clean-racing checks) (`envs/driver_env.py`,
   `envs/race_env.py`).
7. The predictive safety shield and its four rounds of fixes (`models/safety.py`).
8. Rule-based drivers used as baselines and executors: `BaselineDriver`,
   `LineFollower`, `PathDriver` (`agents/baseline.py`, `line_follower.py`,
   `path_driver.py`).
9. Lap analytics and the read-only report/dashboard (`analytics.py`, `report.py`,
   `dashboard.py`, `scripts/compare_drivers.py`).
10. Tests: `test_track`, `test_circuits`, `test_racetrack`, `test_car`, `test_lap`,
    `test_frenet`, `test_raceline`, `test_tire`, `test_driver_env`, `test_race_env`,
    `test_safety`, `test_baseline`, `test_analytics`, `test_report`.

Experiments owned: Phase 3 baseline laps; the early PPO lap-time driver (Phase 4)
and its reward fixes; the SAC driver runs (`racer_*` configs, Phase 4R); shield
contact/off-track checks.

## Member 2: Driver Agent (race-craft learning)

**Goal:** a driver that overtakes and defends cleanly on unseen circuits, trained
with RL on top of Member 1's world.

Tasks:
1. Race-craft move library: outside, divebomb, dummy, switchback, cover, inside,
   pull-out, lane moves (`agents/tactics.py`).
2. `TacticalEnv`: move-level action space, rules of the road, committed moves
   (`envs/tactical_env.py`).
3. Opponent cars and sparring partners: scripted opponents (`envs/opponents.py`),
   guard defenders (`agents/guard.py`), RL-trained defender (`agents/defender_rl.py`).
4. Rule-based tactician expert and behaviour-cloning warm start
   (`agents/tactician.py`, `agents/tactical.py`).
5. PPO driver training: real starts, section battles, two-corner battles, defence
   battles, frozen-self attacker, self-play pilots (`agents/tactical.py`,
   `agents/ppo.py`, `agents/racer.py`).
6. Track sections (straight / braking / corner entry / exit) (`sections.py`) and
   scenario scorecards (`scenarios.py`).
7. Gates, oracles and demos: `scripts/pass_gate.py`, `guard_gate.py`,
   `section_gate.py`, `section_oracle.py`, `defend_oracle.py`, `section_demo.py`.
8. Reward design for passes that stick, dirty-driving penalties and the fix for the
   place-swapping exploit.
9. Tests: `test_tactics`, `test_guard`, `test_defender_rl`, `test_defend_sections`,
   `test_sections`, `test_overtake_v4`, `test_pass_reward`, `test_self_play`,
   `test_realpace`, `test_ppo`.

Experiments owned: `tactical*`, `overtake_*`, `guard_sparring`, `defender_rl*`,
`defend_sections*` configs; the reward ablation (pass bonus 20 vs 100); held-out
driver scorecards.

## Member 3: Strategy Agent, real data and evaluation

**Goal:** a strategist that wins races against rule-based plans under every race
event, built on numbers measured from Member 2's driver; plus the real-data
pipeline, evaluation and write-up.

Tasks:
1. Real F1 data pipeline: FastF1 and OpenF1 incident library and real start states
   (`incidents.py`, `openf1.py`, `real_starts.py`).
2. Lap-by-lap race model: compounds, fuel, pit loss, passing odds, two-compound
   rule (`models/race.py`) and weather (`models/weather.py`).
3. `StrategyEnv`: observations, actions, rewards, SC/VSC, red flags, spins,
   punctures, rain, ERS overtake mode (`envs/strategy_env.py`).
4. Rule-based strategy baseline and rival plans (`agents/strategy_baseline.py`).
5. Driver profile: measure the trained driver's pace and pass/defend curves
   (`driver_profile.py`, `scripts/measure_driver.py`).
6. PPO strategist training and paired comparisons (`agents/strategy.py`,
   `train.py`, `eval.py`), all `strategy_*` configs.
7. Seed-stability study, learning curves and report figures
   (`scripts/make_report_figures.py`).
8. Documentation: report, project explainer, README, race replay page.
9. Tests: `test_race`, `test_strategy_env`, `test_strategy_incidents`,
   `test_strategy_neutral`, `test_strategy_cases`, `test_weather`,
   `test_driver_profile`, `test_incidents`, `test_openf1`, `test_real_starts`,
   `test_config`.

Experiments owned: every `strategy_*` run and the held-out strategy scorecards.

## Shared responsibilities (split equally)

* Weekly integration: each member runs `uv run ruff check`, `uv run mypy src/` and
  `uv run pytest` before merging.
* Code review: every change is reviewed by one of the other two members.
* Final presentation: each member presents their own layer (about one third of the
  slides) and one joint demo.

## Hand-offs between members

| From | To | What is handed over |
|---|---|---|
| Member 1 | Member 2 | `RaceEnv`, `PathDriver`, safety shield, racing lines |
| Member 1 | Member 3 | limit-lap tables per circuit (g-g simulator) |
| Member 3 | Member 2 | real start states from FastF1/OpenF1 |
| Member 2 | Member 3 | trained driver model for profile measurement |
| Member 3 | all | figures, scorecards and report drafts |
