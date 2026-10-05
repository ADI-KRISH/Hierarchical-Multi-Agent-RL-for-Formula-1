"""Rule-based baseline driver (phase 3): the lap time the RL agent must beat.

A fixed script, no learning: before the lap it plans one speed profile -- the
phase-1 forward/backward sweep run on a derated car -- which fixes where it
brakes and how fast it takes each corner. It is deliberately cautious, like a
scripted bot rather than an optimal controller: corners are planned at
`corner_grip_margin` of the car's grip and braking points as if brakes had only
`braking_margin` of their capacity. Each episode jitters both margins a little
(seeded), the lap-to-lap variation of a real driver. It then follows the plan
in `DriverEnv`:

- throttle/brake: whatever command reaches the planned speed by the next step
  (clipped to the car's limits);
- steering: always back toward the racing line when the car has run wide.

It reads the car's state from `DriverEnv`'s `info` dict, which is privileged
information (exact distance and offset) an RL policy only sees through its
observation. That's fine for a baseline: it marks how well a sensible fixed
plan does, not what a policy with the same inputs could learn.
"""

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.analytics import (
    lap_sections,
    profile_times_s,
    section_timings,
    summarize_laps,
)
from f1rl.config import (
    EXAMPLE_CAR,
    GRAVITY_M_S2,
    AnalyticsParams,
    BaselineParams,
    CarParams,
    DriverEnvParams,
    SimParams,
)
from f1rl.envs.driver_env import ActType, DriverEnv
from f1rl.envs.track import SYNTHETIC_TRACKS, Track, track_xy, width_at
from f1rl.models.lap import lap_time_s, standing_start_profile_ms


def planned_profile_ms(
    car: CarParams,
    track: Track,
    corner_grip_margin: float,
    braking_margin: float,
    step_m: float = 2.0,
) -> list[float]:
    """The baseline's fixed plan: standing-start speed (m/s) at stations `step_m`
    apart (plus the finish line), for a car derated by the two margins.
    """
    derated = replace(
        car,
        max_lateral_g=car.max_lateral_g * corner_grip_margin,
        max_braking_g=car.max_braking_g * braking_margin,
    )
    return standing_start_profile_ms(derated, track, step_m)


@dataclass
class BaselineDriver:
    """Follows a fixed speed plan and steers back to the racing line."""

    car: CarParams
    track: Track
    params: BaselineParams
    dt_s: float
    seed: int
    step_m: float = 2.0
    corner_grip_margin: float = field(init=False)
    braking_margin: float = field(init=False)
    plan_ms: list[float] = field(init=False)

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        jitter = rng.normal(0.0, self.params.margin_jitter_std, size=2)
        low, high = self.params.margin_floor, 1.0
        self.corner_grip_margin = float(
            np.clip(self.params.corner_grip_margin + jitter[0], low, high)
        )
        self.braking_margin = float(
            np.clip(self.params.braking_margin + jitter[1], low, high)
        )
        self.plan_ms = planned_profile_ms(
            self.car,
            self.track,
            self.corner_grip_margin,
            self.braking_margin,
            self.step_m,
        )

    def target_speed_ms(self, distance_m: float) -> float:
        """Planned speed at `distance_m`, linearly interpolated between stations."""
        n = len(self.plan_ms) - 1
        ds = self.track.total_length_m / n
        position = min(max(distance_m, 0.0), self.track.total_length_m) / ds
        k = min(int(position), n - 1)
        frac = position - k
        return self.plan_ms[k] + frac * (self.plan_ms[k + 1] - self.plan_ms[k])

    def act(self, info: dict[str, Any]) -> ActType:
        speed = info["speed_ms"]
        # Aim at the plan where the car will be next step -- but at least one
        # station ahead, or a car at rest on the start line (planned speed 0)
        # would be told to stay put.
        ahead_m = max(speed * self.dt_s, self.step_m)
        target = self.target_speed_ms(info["distance_m"] + ahead_m)
        needed_g = (target - speed) / (self.dt_s * GRAVITY_M_S2)
        peak_g = self.car.max_accel_g if needed_g >= 0 else self.car.max_braking_g
        command = needed_g / peak_g
        steer = -1.0 if info["lateral_offset_m"] > 0.0 else 0.0
        return np.array([np.clip(command, -1.0, 1.0), steer], dtype=np.float32)


@dataclass(frozen=True)
class EpisodeResult:
    """One baseline episode: outcome plus decimated telemetry."""

    seed: int
    completed: bool
    off_track: bool
    lap_time_s: float | None
    total_reward: float
    off_track_at_m: float | None
    telemetry: dict[str, list[float]]


def run_episode(
    env: DriverEnv, driver: BaselineDriver, seed: int, telemetry_every: int = 5
) -> EpisodeResult:
    """Drive one episode; record telemetry every `telemetry_every` steps."""
    _obs, info = env.reset(seed=seed)
    channels = ("time_s", "distance_m", "speed_ms", "target_ms", "throttle", "steer")
    telemetry: dict[str, list[float]] = {
        name: [] for name in (*channels, "lateral_offset_m")
    }
    total_reward = 0.0
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action = driver.act(info)
        if step % telemetry_every == 0:
            target = driver.target_speed_ms(info["distance_m"])
            for name, value in zip(
                channels,
                (
                    info["elapsed_s"],
                    info["distance_m"],
                    info["speed_ms"],
                    target,
                    float(action[0]),
                    float(action[1]),
                ),
                strict=True,
            ):
                telemetry[name].append(round(value, 4))
            telemetry["lateral_offset_m"].append(round(info["lateral_offset_m"], 4))
        _obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

    return EpisodeResult(
        seed=seed,
        completed=bool(info["lap_completed"]),
        off_track=bool(info["off_track"]),
        lap_time_s=info.get("lap_time_s"),
        total_reward=total_reward,
        off_track_at_m=info["distance_m"] if info["off_track"] else None,
        telemetry=telemetry,
    )


def resolve_track(spec: str) -> Track:
    """A synthetic track by name, or a real circuit as ``<year>:<grand prix>``
    (FastF1 data; needs network access the first time, then reads the cache).
    """
    if spec in SYNTHETIC_TRACKS:
        return SYNTHETIC_TRACKS[spec]()
    year, sep, grand_prix = spec.partition(":")
    if not sep or not year.isdigit():
        names = ", ".join(SYNTHETIC_TRACKS)
        raise ValueError(f"track must be one of {names}, or <year>:<grand prix>")
    from f1rl.envs.circuits import load_circuit

    return load_circuit(int(year), grand_prix)


def evaluate_baseline(
    track_name: str,
    track: Track,
    car: CarParams,
    episodes: int,
    seed: int,
    baseline_params: BaselineParams | None = None,
    env_params: DriverEnvParams | None = None,
    analytics: AnalyticsParams | None = None,
) -> dict[str, Any]:
    """Run the baseline for `episodes` seeded laps and analyse them against the
    theoretical limit lap. Returns a JSON-ready results dict.
    """
    baseline_params = baseline_params or BaselineParams()
    env_params = env_params or DriverEnvParams()
    analytics = analytics or AnalyticsParams()
    sim = SimParams(seed=seed)
    env = DriverEnv(car, track, sim, env_params)

    runs = []
    for episode in range(episodes):
        episode_seed = seed + episode
        driver = BaselineDriver(car, track, baseline_params, sim.dt_s, episode_seed)
        result = run_episode(
            env, driver, episode_seed, analytics.telemetry_every_n_steps
        )
        runs.append((driver, result))

    limit_ms = standing_start_profile_ms(car, track)
    ds = track.total_length_m / (len(limit_ms) - 1)
    limit_distance = [k * ds for k in range(len(limit_ms))]
    limit_time = profile_times_s(limit_ms, ds)
    sections = lap_sections(
        track, analytics.corner_curvature_per_m, analytics.min_corner_m
    )

    episode_rows = []
    for driver, result in runs:
        row: dict[str, Any] = {
            "seed": result.seed,
            "completed": result.completed,
            "off_track": result.off_track,
            "off_track_at_m": result.off_track_at_m,
            "lap_time_s": result.lap_time_s,
            "total_reward": result.total_reward,
            "corner_grip_margin": driver.corner_grip_margin,
            "braking_margin": driver.braking_margin,
            "top_speed_ms": max(result.telemetry["speed_ms"]),
            "max_lateral_offset_m": max(result.telemetry["lateral_offset_m"]),
            "telemetry": result.telemetry,
        }
        if result.completed:
            # Close the trace at the line so section timing covers the whole lap.
            t = result.telemetry
            timings = section_timings(
                sections,
                [*t["distance_m"], track.total_length_m],
                [*t["time_s"], result.lap_time_s or 0.0],
                [*t["speed_ms"], t["speed_ms"][-1]],
                limit_distance,
                limit_time,
                limit_ms,
            )
            row["sections"] = [{**asdict(s), "delta_s": s.delta_s} for s in timings]
        episode_rows.append(row)

    summary = summarize_laps(
        [r.lap_time_s if r.completed else None for _d, r in runs],
        off_track=sum(r.off_track for _d, r in runs),
    )
    limit_lap_s = limit_time[-1]
    map_xy = track_xy(track, analytics.map_step_m)
    map_ds = track.total_length_m / (len(map_xy) - 1) if len(map_xy) > 1 else 0.0

    return {
        "track": track_name,
        "track_length_m": track.total_length_m,
        "created_by": "f1rl.agents.baseline",
        "git_sha": _git_sha(),
        "config": {
            "seed": seed,
            "episodes": episodes,
            "car": asdict(car),
            "sim": asdict(sim),
            "env": asdict(env_params),
            "baseline": asdict(baseline_params),
            "analytics": asdict(analytics),
        },
        "summary": {
            **asdict(summary),
            "completion_rate": summary.completion_rate,
            "limit_lap_time_s": limit_lap_s,
            "flying_lap_time_s": lap_time_s(car, track),
            "gap_to_limit_pct": (
                100 * (summary.mean_s / limit_lap_s - 1) if summary.mean_s else None
            ),
        },
        "limit_profile": {  # full resolution, finish line included
            "distance_m": [round(d, 3) for d in limit_distance],
            "time_s": [round(t, 4) for t in limit_time],
            "speed_ms": [round(v, 3) for v in limit_ms],
        },
        "sections": [asdict(s) for s in sections],
        "map": {
            "x_m": [round(x, 2) for x, _y in map_xy],
            "y_m": [round(y, 2) for _x, y in map_xy],
            "distance_m": [round(k * map_ds, 2) for k in range(len(map_xy))],
            "width_m": [width_at(track, k * map_ds) for k in range(len(map_xy))],
        },
        "episodes": episode_rows,
    }


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")


def _format_s(value: float | None) -> str:
    return "--" if value is None else f"{value:.3f}s"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the rule-based baseline driver and log its laps."
    )
    parser.add_argument(
        "--track",
        default="technical",
        help=f"{', '.join(SYNTHETIC_TRACKS)}, or <year>:<grand prix> (FastF1).",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("runs/baseline"))
    args = parser.parse_args()

    track = resolve_track(args.track)
    results = evaluate_baseline(
        args.track, track, EXAMPLE_CAR, args.episodes, args.seed
    )
    out_dir = args.out / re.sub(r"[^a-z0-9]+", "-", args.track.lower()).strip("-")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results))

    s = results["summary"]
    print(
        f"Baseline on {args.track} ({track.total_length_m:.0f} m), "
        f"{s['episodes']} episodes, seed {args.seed}"
    )
    print(
        f"  completed     {s['completed']}/{s['episodes']}  "
        f"(off track: {s['off_track']})"
    )
    print(f"  mean lap      {_format_s(s['mean_s'])}  +- {_format_s(s['std_s'])}")
    print(f"  best / worst  {_format_s(s['best_s'])} / {_format_s(s['worst_s'])}")
    print(
        f"  limit lap     {_format_s(s['limit_lap_time_s'])}  "
        f"(standing start, perfect driver)"
    )
    if s["gap_to_limit_pct"] is not None:
        print(f"  gap to limit  {s['gap_to_limit_pct']:+.2f}%")
    print(f"Wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
