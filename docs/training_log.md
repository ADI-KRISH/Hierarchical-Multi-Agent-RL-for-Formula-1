# Training log: racing-line driver (Phase 4R), 2026-10-07

Every run of the steering-car driver (`RaceEnv`, SAC, TUM tracks), what changed, how long
it took and how it ended. Numbers come from each run's own `progress.csv` and `*.log`
(`runs/<name>/`, git-ignored unless force-added). Lap gaps are the mean over the tracks
completed, against that track's optimal-line limit lap in the same physics.

Hardware: AMD Ryzen 7 7840HS (8 cores / 16 threads), CPU only, no GPU. Several runs
shared the CPU for most of the day, so wall-clock times are longer than a run alone
(one run alone managed ~280-330 decisions/s; shared, 120-190/s).

Splits: train on 17 TUM tracks; pick the best checkpoint on 3 validation tracks
(Montreal, Hockenheim, Catalunya); judge on 5 held-out tracks (Suzuka, Spa, Sao Paulo,
Silverstone, Yas Marina) that are never trained on and never used to choose anything.

## Runs

| Run | What it changed | Decisions | Time | Outcome |
|---|---|---|---|---|
| `racer_sac` | **Final pure-RL recipe**: 12 lookahead windows, all rewards x10 (progress 10 m, fail penalty 50, smooth 0.1), LR 3e-4 -> 3e-5 | 2.0M, finished | 3 h 29 min | final model: held-out 3/5 complete at +17% (training eval); best checkpoint: 4/5 at +19.2% (`f1rl.eval`, table below) |
| `racer_hybrid` | Same recipe, plus the optimal line in the observation (hybrid) | 1.0M, finished | 2 h 11 min | held-out about +19-21%, no better than pure RL |
| `racer_sac_v1_first` | First recipe: 6 lookahead windows, base rewards, constant LR | 1.16M, stopped | 1 h 40 min | plateau about +18-20%, completions unstable |
| `racer_fine_x10` | 12 windows + rewards x10 + smoothness (no LR decay) | 680k, stopped | 1 h 20 min | learned faster than the first recipe (+18% at 300k vs +26%) |
| `racer_fine` | 12 windows + smoothness 0.01 only | 365k, stopped | 34 min | slower than the x10 variant |
| `racer_hybrid_risk` | Hybrid with fail penalty 50 -> 10 | 257k, stopped | 35 min | worse (+28% at 250k) |
| `racer_hz25` | 25 Hz decisions (action_repeat 2), gamma 0.998 | 282k, stopped | 35 min | no completed lap after 250k |
| `racer_demo` | 300k scripted-driver demonstrations in the replay buffer | 166k, stopped | 24 min (+ filling the buffer) | crashed on every track at 150k |

Earlier driver for reference: `driver_ppo_stable` (v1 centreline car, PPO) trained 2.0M
decisions in about 7 minutes.

Total for the Phase 4R runs above: about 10.8 hours.

## What the runs showed

- Every recipe plateaus at about 17-20% slower than the optimal line, while a hand-written
  controller (`LineFollower`) laps the same line 2-3% off. The environment allows the pace;
  the learned drivers do not find it.
- In training the final agent fails only 2-5% of episodes but averages ~60 m/s against
  ~70 m/s for the optimal line: it has settled on a cautious optimum (the same pattern as
  the PPO driver's speed governor).
- Things that helped early learning: finer lookahead windows, x10 reward scale.
- Things that did not remove the plateau: LR decay, 25 Hz control, demonstration pre-fill,
  a lower fail penalty, and giving the agent the optimal line.
- Completion rate on the 3 validation tracks swings between 0/3 and 3/3 at the same average
  lap gap, so the best checkpoint (chosen by completion first) is partly luck.

## Final numbers (held-out tracks, `racer_sac/model_best.zip`)

| Track | Optimal line | Scripted driver | v1 baseline | Learned driver |
|---|---|---|---|---|
| Suzuka | 76.24 s | 78.02 (+2.3%) | 86.72 (+13.7%) | 90.44 (+18.6%) |
| Spa | 89.57 | 91.53 (+2.2%) | 100.28 (+11.9%) | 107.55 (+20.1%) |
| Sao Paulo | 58.40 | 59.72 (+2.3%) | 69.45 (+18.9%) | 69.08 (+18.3%) |
| Silverstone | 75.95 | 77.67 (+2.3%) | 90.19 (+18.8%) | 91.00 (+19.8%) |
| Yas Marina | 83.46 | 85.96 (+3.0%) | 97.47 (+16.8%) | did not finish |

The v1 baseline is the cautious centreline driver from a standing start; the learned and
scripted laps are flying laps, so the comparison is only rough.

## Reproducing

```
uv run python -m f1rl.train --config configs/racer_sac.yaml        # pure RL, ~3.5 h shared CPU
uv run python -m f1rl.train --config configs/racer_hybrid.yaml     # hybrid
uv run python -m f1rl.eval --agent runs/racer_sac/model_best.zip --tracks tum_held_out
uv run python scripts/compare_drivers.py                           # interactive comparison page
```

Experiment configs are in `configs/experiments/`. All runs use seed 0.
