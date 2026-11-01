"""SB3 PPO wrapper for `DriverEnv` (phase 4): config, env factory, lap logging.

Hyperparameters come from an experiment YAML under ``configs/`` -- nothing
tunable is hardcoded here. A run writes everything needed to reproduce and
inspect it into its ``runs/<name>/`` folder:

- ``config.yaml`` (the resolved config) and ``meta.json`` (git SHA, versions);
- ``progress.csv``: SB3's training stats per update;
- ``episodes.csv``: every training episode -- finished?, lap time, return;
- ``eval.csv``: periodic deterministic evaluation laps on held-out seeds;
- ``model_best.zip`` / ``model_final.zip``.
"""

import csv
import json
import platform
import subprocess
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import yaml
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from f1rl.config import EXAMPLE_CAR, CarParams, DriverEnvParams, SimParams
from f1rl.envs.driver_env import DriverEnv
from f1rl.envs.track import Track
from f1rl.models.lap import standing_start_lap_time_s


@dataclass(frozen=True)
class RunParams:
    """The `run:` section of an experiment YAML."""

    name: str
    seed: int
    track: str | list[str]  # one track, or several: envs are dealt round-robin
    total_timesteps: int
    n_envs: int
    eval_every_steps: int
    eval_episodes: int
    eval_seed: int  # eval laps use seeds eval_seed, eval_seed + 1, ...
    torch_threads: int = 1  # small MLPs train fastest (and reproducibly) on 1 thread
    # Continue from this saved model (its weights; the PPO settings come from
    # this config) instead of starting from scratch.
    init_from: str | None = None
    # Reset the policy's exploration noise (action std-dev) to this after
    # loading `init_from` -- e.g. to fine-tune a driver near the limit, where
    # wide noise turns well-placed braking into crashes.
    action_std: float | None = None
    # At every eval, one lap per track is saved to eval_laps.jsonl with telemetry
    # every this many decisions -- the record of how the driving changed.
    replay_every_decisions: int = 5

    @property
    def tracks(self) -> list[str]:
        return [self.track] if isinstance(self.track, str) else list(self.track)


@dataclass(frozen=True)
class TrainConfig:
    run: RunParams
    sim: dict[str, Any] = field(default_factory=dict)  # SimParams overrides
    env: dict[str, Any] = field(default_factory=dict)  # DriverEnvParams overrides
    ppo: dict[str, Any] = field(default_factory=dict)  # PPO(...) keyword args
    policy: dict[str, Any] = field(default_factory=dict)  # policy_kwargs

    def env_params(self) -> DriverEnvParams:
        overrides = dict(self.env)
        if "lookahead_m" in overrides:
            overrides["lookahead_m"] = tuple(overrides["lookahead_m"])
        return DriverEnvParams(**overrides)

    def sim_params(self, seed: int) -> SimParams:
        return SimParams(seed=seed, **self.sim)


def load_config(path: Path) -> TrainConfig:
    """Parse an experiment YAML; unknown keys fail loudly rather than being
    silently ignored."""
    raw = yaml.safe_load(path.read_text())
    sections = {f.name for f in fields(TrainConfig)}
    unknown = set(raw) - sections
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    config = TrainConfig(
        run=RunParams(**raw["run"]),
        sim=raw.get("sim") or {},
        env=raw.get("env") or {},
        ppo=raw.get("ppo") or {},
        policy=raw.get("policy") or {},
    )
    config.env_params()  # validate field names now, not mid-run
    config.sim_params(0)
    return config


def make_env_fn(
    car: CarParams,
    track: Track,
    sim: SimParams,
    env_params: DriverEnvParams,
    monitor: bool = True,
) -> Callable[[], gym.Env[Any, Any]]:
    def _make() -> gym.Env[Any, Any]:
        env: gym.Env[Any, Any] = DriverEnv(car, track, sim, env_params)
        return Monitor(env) if monitor else env

    return _make


def run_policy_lap(
    model: PPO,
    env: DriverEnv,
    seed: int,
    deterministic: bool = True,
    telemetry_every: int = 0,
) -> dict[str, Any]:
    """Drive one episode with `model`'s policy; optional decimated telemetry."""
    obs, info = env.reset(seed=seed)
    total = 0.0
    step = 0
    channels = ("time_s", "distance_m", "speed_ms", "lateral_offset_m")
    telemetry: dict[str, list[float]] = {
        c: [] for c in (*channels, "throttle", "steer")
    }
    terminated = truncated = False
    while not (terminated or truncated):
        action, _state = model.predict(obs, deterministic=deterministic)
        if telemetry_every and step % telemetry_every == 0:
            values = (
                info["elapsed_s"],
                info["distance_m"],
                info["speed_ms"],
                info["lateral_offset_m"],
            )
            for name, value in zip(channels, values, strict=True):
                telemetry[name].append(round(float(value), 4))
            clipped = np.clip(action, -1.0, 1.0)
            telemetry["throttle"].append(round(float(clipped[0]), 4))
            telemetry["steer"].append(round(float(clipped[1]), 4))
        obs, reward, terminated, truncated, info = env.step(action)
        total += float(reward)
        step += 1
    result: dict[str, Any] = {
        "seed": seed,
        "completed": bool(info["lap_completed"]),
        "off_track": bool(info["off_track"]),
        "stalled": bool(info.get("stalled", False)),
        "off_track_at_m": info["distance_m"] if info["off_track"] else None,
        "lap_time_s": info.get("lap_time_s"),
        "distance_m": info["distance_m"],
        "total_reward": total,
    }
    if telemetry_every:
        result["telemetry"] = telemetry
    return result


def eval_score(
    results: list[dict[str, Any]], limit_lap_s: dict[str, float] | None = None
) -> tuple[float, float, float]:
    """Sort key for "better model", higher is better: completion rate first, then
    faster laps, then mean reward -- which is what separates models before any
    of them finishes a lap (without it, early evals all tie). With several
    tracks, lap times are compared as a fraction of each track's limit lap
    (`limit_lap_s`, keyed by each result's "track"), so long tracks don't
    dominate."""
    done = [r for r in results if r["completed"]]
    rate = len(done) / len(results)
    if not done:
        pace = -float("inf")
    elif limit_lap_s:
        pace = -float(
            np.mean([r["lap_time_s"] / limit_lap_s[r["track"]] for r in done])
        )
    else:
        pace = -float(np.mean([r["lap_time_s"] for r in done]))
    return rate, pace, float(np.mean([r["total_reward"] for r in results]))


class LapLoggerCallback(BaseCallback):
    """Logs every finished training episode and runs periodic evaluation on
    every track, saving one replay lap per track per evaluation."""

    def __init__(
        self,
        run_dir: Path,
        eval_envs: dict[str, DriverEnv],
        env_tracks: list[str],
        eval_every_steps: int,
        eval_episodes: int,
        eval_seed: int,
        replay_every_decisions: int,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.eval_envs = eval_envs
        self.env_tracks = env_tracks  # track name of each training env
        self.eval_every_steps = eval_every_steps
        self.eval_episodes = eval_episodes
        self.eval_seed = eval_seed
        self.replay_every = replay_every_decisions
        self.limit_lap_s = {
            name: standing_start_lap_time_s(env.car, env.track)
            for name, env in eval_envs.items()
        }
        self.best_score: tuple[float, float, float] | None = None
        self._next_eval = 0
        self._returns: np.ndarray = np.zeros(0)
        self._episodes_file: Any = None
        self._episodes: Any = None
        self._recent: list[dict[str, Any]] = []

    def _on_training_start(self) -> None:
        self._returns = np.zeros(self.training_env.num_envs)
        self._episodes_file = (self.run_dir / "episodes.csv").open("w", newline="")
        self._episodes = csv.writer(self._episodes_file)
        self._episodes.writerow(
            [
                "timesteps",
                "env",
                "track",
                "completed",
                "off_track",
                "stalled",
                "lap_time_s",
                "distance_m",
                "total_reward",
            ]
        )
        with (self.run_dir / "eval.csv").open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "timesteps",
                    "track",
                    "completion_rate",
                    "mean_lap_s",
                    "best_lap_s",
                    "mean_reward",
                    "off_track",
                    "stalled",
                    "limit_lap_s",
                ]
            )
        (self.run_dir / "eval_laps.jsonl").write_text("")

    def _on_step(self) -> bool:
        self._returns += self.locals["rewards"]
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            row = {
                "completed": bool(info["lap_completed"]),
                "off_track": bool(info["off_track"]),
                "stalled": bool(info.get("stalled", False)),
                "lap_time_s": info.get("lap_time_s"),
                "distance_m": info["distance_m"],
                "total_reward": float(self._returns[i]),
            }
            self._returns[i] = 0.0
            self._recent.append(row)
            self._episodes.writerow(
                [
                    self.num_timesteps,
                    i,
                    self.env_tracks[i],
                    int(row["completed"]),
                    int(row["off_track"]),
                    int(row["stalled"]),
                    _fmt(row["lap_time_s"]),
                    f"{row['distance_m']:.1f}",
                    f"{row['total_reward']:.4f}",
                ]
            )
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every_steps
            self._evaluate()
        return True

    def _on_rollout_end(self) -> None:
        if not self._recent:
            return
        laps = [r["lap_time_s"] for r in self._recent if r["completed"]]
        self.logger.record("laps/train_completion", len(laps) / len(self._recent))
        self.logger.record(
            "laps/train_off_track", np.mean([r["off_track"] for r in self._recent])
        )
        self.logger.record(
            "laps/train_stalled", np.mean([r["stalled"] for r in self._recent])
        )
        if laps:
            self.logger.record("laps/train_mean_lap_s", float(np.mean(laps)))
        self._recent = []

    def _evaluate(self) -> None:
        assert isinstance(self.model, PPO)
        all_results = []
        summary = []
        for name, env in self.eval_envs.items():
            results = []
            for k in range(self.eval_episodes):
                every = self.replay_every if k == 0 else 0
                lap = run_policy_lap(
                    self.model, env, self.eval_seed + k, telemetry_every=every
                )
                lap["track"] = name
                if k == 0:
                    replay = {"timesteps": self.num_timesteps, **lap}
                    with (self.run_dir / "eval_laps.jsonl").open("a") as f:
                        f.write(json.dumps(replay) + "\n")
                    lap.pop("telemetry", None)
                results.append(lap)
            all_results.extend(results)
            laps = [r["lap_time_s"] for r in results if r["completed"]]
            rate = len(laps) / len(results)
            mean_lap = float(np.mean(laps)) if laps else None
            mean_reward = float(np.mean([r["total_reward"] for r in results]))
            with (self.run_dir / "eval.csv").open("a", newline="") as f:
                csv.writer(f).writerow(
                    [
                        self.num_timesteps,
                        name,
                        f"{rate:.3f}",
                        _fmt(mean_lap),
                        _fmt(min(laps) if laps else None),
                        f"{mean_reward:.4f}",
                        sum(r["off_track"] for r in results),
                        sum(r["stalled"] for r in results),
                        f"{self.limit_lap_s[name]:.4f}",
                    ]
                )
            self.logger.record(f"eval/{name}/completion_rate", rate)
            if mean_lap is not None:
                self.logger.record(f"eval/{name}/mean_lap_s", mean_lap)
            lap_text = f"{mean_lap:.2f}s" if mean_lap is not None else "--"
            summary.append(f"{name} {rate:.0%} {lap_text}")

        score = eval_score(all_results, self.limit_lap_s)
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score = score
            self.model.save(self.run_dir / "model_best.zip")
        print(
            f"[eval] {self.num_timesteps:>9,} steps  "
            + " | ".join(summary)
            + ("  *best" if improved else ""),
            flush=True,
        )

    def _on_training_end(self) -> None:
        self._evaluate()
        if self._episodes_file is not None:
            self._episodes_file.close()


def _fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.4f}"


def git_sha() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("-dirty" if dirty else "")


def train(
    config: TrainConfig,
    tracks: dict[str, Track],
    run_dir: Path,
    car: CarParams = EXAMPLE_CAR,
) -> PPO:
    """Train PPO per `config` on `tracks` (name -> Track), logging into
    `run_dir`. Training envs are dealt round-robin across the tracks."""
    run = config.run
    torch.set_num_threads(run.torch_threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": asdict(run),
                "sim": config.sim,
                "env": config.env,
                "ppo": config.ppo,
                "policy": config.policy,
            },
            sort_keys=False,
        )
    )
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "python": platform.python_version(),
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
                "gymnasium": gym.__version__,
                "car": asdict(car),
                "env_params": asdict(config.env_params()),
                "track_length_m": {n: t.total_length_m for n, t in tracks.items()},
            },
            indent=2,
        )
    )

    env_params = config.env_params()
    names = list(tracks)
    env_tracks = [names[i % len(names)] for i in range(run.n_envs)]
    vec_env = DummyVecEnv(
        [
            make_env_fn(car, tracks[name], config.sim_params(run.seed + i), env_params)
            for i, name in enumerate(env_tracks)
        ]
    )
    eval_envs = {
        name: DriverEnv(car, track, config.sim_params(run.eval_seed), env_params)
        for name, track in tracks.items()
    }

    policy_kwargs = dict(config.policy)
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = getattr(
            torch.nn, policy_kwargs["activation_fn"]
        )
    ppo_kwargs = dict(config.ppo)
    rate = ppo_kwargs.get("learning_rate")
    if isinstance(rate, dict):  # {start: .., end: ..}: linear decay over the run
        start, end = float(rate["start"]), float(rate["end"])
        ppo_kwargs["learning_rate"] = lambda progress: end + (start - end) * progress
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=run.seed,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
        **ppo_kwargs,
    )
    if run.init_from:
        model.set_parameters(run.init_from, device="cpu")
    if run.action_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(run.action_std)))
    model.set_logger(configure(str(run_dir), ["csv"]))
    callback = LapLoggerCallback(
        run_dir,
        eval_envs,
        env_tracks,
        run.eval_every_steps,
        run.eval_episodes,
        run.eval_seed,
        run.replay_every_decisions,
    )
    model.learn(total_timesteps=run.total_timesteps, callback=callback)
    model.save(run_dir / "model_final.zip")
    record_best_laps(run_dir, config, tracks, car)
    return model


def record_best_laps(
    run_dir: Path,
    config: TrainConfig,
    tracks: dict[str, Track],
    car: CarParams = EXAMPLE_CAR,
) -> dict[str, Any]:
    """Drive one deterministic eval lap per track with ``model_best.zip`` and save
    them, with 10 Hz telemetry, as ``best_lap.json`` -- so reports can show the
    agent's line from logs alone, without ever loading or running a model."""
    model = PPO.load(run_dir / "model_best.zip", device="cpu")
    env_params = config.env_params()
    every = max(1, round(0.1 / (config.sim_params(0).dt_s * env_params.action_repeat)))
    laps = {}
    for name, track in tracks.items():
        env = DriverEnv(car, track, config.sim_params(config.run.eval_seed), env_params)
        lap = run_policy_lap(model, env, config.run.eval_seed, telemetry_every=every)
        lap["track"] = name
        laps[name] = lap
    record = {"laps": laps}
    (run_dir / "best_lap.json").write_text(json.dumps(record))
    return record
