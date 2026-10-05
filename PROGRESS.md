# Progress

Snapshot as of 2026-10-06. Phase definitions and "done when" criteria live in
`docs/roadmap.md`; this file only tracks where we are against them.

**Summary:** phases 0-2 are done, plus extra env realism work beyond the phase 2
spec. Phases 3-7 are not started. Nothing has been trained yet.

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

## Left

| Phase | What | State |
|---|---|---|
| 3 | Rule-based baseline driver in `agents/`, with its lap time over 20 runs | Not started — `agents/` is empty |
| 4 | SB3 PPO training, `configs/driver_ppo.yaml`, logging to `runs/` | Not started — `train.py` is a CLI stub, `configs/` is empty |
| 5 | `eval.py`: agent vs baseline comparison over N episodes | Not started — CLI stub |
| 6 | Read-only Dash dashboard over `runs/` | Not started — stub |
| 7 | README polish: results, GIF, future-work section | Not started |
| 8 | Future work: opponents, tire/fuel/ERS, Strategy Agent, two-agent coupling | Out of MVP scope; needs explicit go-ahead |

**Next up:** phase 3, the rule-based baseline.

## Known gaps

- Track width is a design-choice default, not data-derived (FastF1 has no width channel).
- The FastF1 fetch has no automated test, to keep `uv run pytest` offline-safe; only the
  position-processing math is unit-tested.
- No tire, fuel or ERS models yet (phase 8).
