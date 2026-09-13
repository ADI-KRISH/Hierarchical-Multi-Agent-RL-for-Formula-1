# FormulaOneRL — Project Context & Long-Term Vision

> This file holds the full, long-term vision for FormulaOneRL. The **current**
> build is the MVP (a single Driver Agent) — see `docs/roadmap.md` for what's
> actually being built now. Everything below Phase-MVP scope (the Strategy Agent,
> the hierarchy, the stretch goals) is the destination, not the current task.

---

## Project title

**FormulaOneRL: Hierarchical Multi-Agent Reinforcement Learning for Formula 1
Driver and Race Strategy**

## Overview

The long-term goal is a realistic Formula 1 racing AI built with Hierarchical
Multi-Agent Reinforcement Learning. Unlike single-agent RL racing projects, this
separates responsibilities into two cooperating agents:

1. **Driver Agent** — real-time vehicle control.
2. **Race Strategist Agent** — race-level tactical decisions.

Together they model the collaboration between an F1 driver and the race engineer
across a full Grand Prix, demonstrating hierarchical decision making, multi-agent
communication, long-horizon planning, and optimization under uncertainty.

**MVP note:** the MVP builds the Driver Agent alone. The Strategist and the
hierarchy are future work layered on once the driver works.

## Motivation

Most F1 AI projects focus only on autonomous driving, but F1 success depends
equally on strategy — pit timing, tire selection, fuel management, ERS
deployment, attack vs defend, weather, and safety-car response. This project
models both the driver and the strategist as cooperating RL agents.

## High-level architecture (full vision)

```
+-----------------------------------------------+
|      Formula 1 Race Simulation Environment     |
+-----------------------------------------------+
                      |
              Shared Race State
                      |
   +--------------+      +------------------+
   | Driver Agent | <--> | Strategy Agent   |
   +--------------+ comm +------------------+
          |
   Car control commands
          |
       Simulator
          |
     Race outcome
```

The Strategy Agent communicates high-level objectives; the Driver Agent
translates them into low-level driving. In the implementation, the strategist's
current mode is passed into the driver's observation — that vector *is* the
communication channel.

---

## Driver Agent

**Responsibilities:** steering, throttle, braking, racing-line optimization,
cornering, overtaking, defensive driving, DRS activation, vehicle stability.

**Observation space (candidates):** vehicle speed, steering angle, track
position, distance to apex, tire grip, tire wear, fuel load, ERS charge, opponent
positions, DRS availability, track limits, current lap, weather.

**Action space:**
- Continuous: steering angle, brake pressure, throttle percentage.
- Discrete: overtake, defend, enable DRS, use ERS, choose racing line.

**Reward — positive:** faster lap time, smooth racing line, successful overtakes,
clean corner exits, maintaining traction.
**Reward — negative:** collisions, off-track incidents, excessive tire wear,
unsafe overtakes, wheel spin, track-limit violations.

---

## Strategy Agent (future work)

Operates at a much lower frequency than the driver; manages strategy rather than
controlling the car directly.

**Responsibilities:** pit-stop timing, tire compound selection, fuel strategy,
ERS deployment policy, attack/defend modes, tire conservation, safety-car
strategy, weather adaptation, undercut/overcut decisions.

**Observation space (candidates):** current lap, remaining laps, tire
degradation, tire temperatures, fuel level, weather forecast, gap to competitors,
track position, pit window, safety-car / VSC status, battery charge, driver
aggression level.

**Action space (discrete):** pit this lap / stay out; soft / medium / hard tires;
push / normal / conserve; save fuel; deploy ERS aggressively / harvest; attack /
defend.

**Reward — positive:** better finishing position, lower total race time,
successful undercut, efficient tire usage, good fuel economy, championship points.
**Reward — negative:** poor pit timing, running out of tire life, fuel
starvation, wasted ERS, losing track position.

---

## Agent communication (future work)

The Strategy Agent periodically sends objectives to the Driver Agent, e.g.
Attack → later braking, higher ERS deployment, aggressive overtaking;
Conserve Tires → earlier braking, smooth steering, reduced wheel spin;
Fuel Save → lift and coast, reduced throttle.

---

## RL algorithms (candidates)

- Driver: PPO, SAC, TD3.
- Strategy: PPO, DQN, MAPPO, QMIX (future extension).

## Environment

Custom Gymnasium environment (point-mass / bicycle car model on a parameterized
track). Heavyweight sims (TORCS, CARLA, Assetto Corsa) are intentionally avoided
— simulation speed matters more than physical fidelity for RL sample efficiency.

## Race simulation components (full vision)

Tire degradation, tire temperature, fuel consumption, ERS battery, DRS zones, pit
lane timing, dynamic weather, track grip evolution, safety cars, virtual safety
cars, optional mechanical failures.

## Baseline models

- Rule-based driver: fixed racing line, fixed braking points.
- Rule-based strategy: fixed pit windows, static tire strategy, fixed ERS.

## Evaluation metrics

- Driver: average lap time, fastest lap, collision rate, off-track count, tire
  degradation, overtake success.
- Strategy: race time, finishing position, pit-stop efficiency, tire utilization,
  fuel usage, ERS efficiency.
- Overall: win rate, podium %, championship points, average finishing position.

## Visualization dashboard

Interactive dashboard: live race map, driver telemetry, tire wear, fuel, ERS, lap
times, position changes, pit-stop timeline, reward curves, training statistics.

## Tech stack

Python, PyTorch, Stable Baselines3 (Ray RLlib only for the multi-team / self-play
stretch), Gymnasium, NumPy, Pandas, Plotly, Dash, FastF1 for telemetry and
calibration. Environment and dependencies managed with uv.

## Stretch goals

Multi-team competition, opponent strategy prediction, hierarchical RL, curriculum
learning, self-play, Monte Carlo race simulation, Bayesian weather prediction,
digital twin of a real circuit, LLM-powered race engineer for natural-language
communication, offline RL on historical F1 telemetry, explainable-AI dashboard
showing why the strategist chose each decision.

## Expected outcome (full vision)

A complete simulated Grand Prix where the Driver Agent drives competitively, the
Strategy Agent wins races through intelligent decisions, and the two cooperate to
maximize performance — demonstrating RL, MARL, hierarchical RL, autonomous
systems, sequential decision making, simulation, and AI for motorsports, with
production-quality engineering, reproducible experiments, and an interactive
dashboard.