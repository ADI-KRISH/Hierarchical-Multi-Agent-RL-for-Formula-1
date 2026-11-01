"""SB3 SAC for `RaceEnv` (phase 4R.5): trains one driver across many real tracks and
scores it against each track's optimal racing line.

Everything tunable comes from an experiment YAML (see ``configs/racer_sac.yaml``).
A run writes into ``runs/<name>/``:

- ``config.yaml`` and ``meta.json`` (git SHA, versions, the optimal-line reference);
- ``progress.csv``: SB3's training stats;
- ``episodes.csv``: every training episode (track, return, distance, mean speed);
- ``eval.csv``: periodic deterministic flying laps on the validation, held-out and a
  sample of training tracks -- lap time, and the gap to that track's optimal line;
- ``eval_laps.jsonl``: one replay lap (with telemetry) per eval track per evaluation;
- ``model_best.zip`` (best on the VALIDATION tracks only) and ``model_final.zip``.

Track splits: the driver trains on `TUM_TRAIN`; `TUM_VALIDATION` picks the best
checkpoint; `TUM_HELD_OUT` is never trained on and never used to choose anything.
"""

import csv
import json
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
import yaml
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from f1rl.agents.line_follower import LineFollower
from f1rl.agents.ppo import git_sha
from f1rl.config import EXAMPLE_CAR, CarParams, RaceEnvParams, SimParams
from f1rl.envs.race_env import ActType, ObsType, RaceEnv
from f1rl.envs.racetrack import (
    TUM_HELD_OUT,
    TUM_TRACKS,
    TUM_TRAIN,
    TUM_VALIDATION,
    RaceTrack,
    load_tum_track,
)
from f1rl.models.raceline import OptimalLine, optimal_line


@dataclass(frozen=True)
class RacerRun:
    """The `run:` section of a racer experiment YAML."""

    name: str
    seed: int
    total_timesteps: int
    n_envs: int
    eval_every_steps: int
    eval_seed: int
    train_tracks: str | list[str] = "tum_train"
    validation_tracks: str | list[str] = "tum_validation"
    held_out_tracks: str | list[str] = "tum_held_out"
    # How many training tracks to also drive at each eval, for the learning curves.
    eval_train_tracks: int = 4
    torch_threads: int = 4
    replay_every_decisions: int = 5
    init_from: str | None = None
    # Demonstration transitions put in the replay buffer before learning starts:
    # the scripted optimal-line follower driving random-start episodes, with
    # Gaussian noise of this std on its commands so the buffer also holds
    # off-line states and how to recover from them. 0 = learn from scratch.
    demo_steps: int = 0
    demo_noise_std: float = 0.1
    # Extra scripted laps run on the validation tracks at every eval, each a dict:
    # ``name``, ``options`` (passed to `RaceEnv.reset`: push, wear, opponent, gap_m)
    # and ``min_passes`` (net overtakes needed to count: 1 = must pass the car ahead,
    # 0 = must not be passed). A scenario lap succeeds if it completes without
    # contact and meets ``min_passes``. Counted in the checkpoint score.
    eval_scenarios: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class RacerConfig:
    run: RacerRun
    sim: dict[str, Any] = field(default_factory=dict)  # SimParams overrides
    env: dict[str, Any] = field(default_factory=dict)  # RaceEnvParams overrides
    sac: dict[str, Any] = field(default_factory=dict)  # SAC(...) keyword args
    policy: dict[str, Any] = field(default_factory=dict)  # policy_kwargs

    def env_params(self) -> RaceEnvParams:
        overrides = dict(self.env)
        for key in (
            "start_speed_ms",
            "lookahead_windows_m",
            "edge_points_m",
            "reference_points_m",
            "opponents",
            "push_levels",
            "opponent_gap_m",
        ):
            if key in overrides:
                overrides[key] = tuple(overrides[key])
        return RaceEnvParams(**overrides)

    def sim_params(self, seed: int) -> SimParams:
        return SimParams(seed=seed, **self.sim)


def load_racer_config(path: Path) -> RacerConfig:
    """Parse a racer YAML; unknown keys fail loudly instead of being ignored."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = set(raw) - {f.name for f in fields(RacerConfig)}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    config = RacerConfig(
        run=RacerRun(**raw["run"]),
        sim=raw.get("sim") or {},
        env=raw.get("env") or {},
        sac=raw.get("sac") or {},
        policy=raw.get("policy") or {},
    )
    config.env_params()
    config.sim_params(0)
    return config


_TRACK_SETS: dict[str, tuple[str, ...]] = {
    "tum_all": TUM_TRACKS,
    "tum_train": TUM_TRAIN,
    "tum_validation": TUM_VALIDATION,
    "tum_held_out": TUM_HELD_OUT,
}


def resolve_track_names(spec: str | list[str]) -> list[str]:
    """A named set (``tum_train`` ...), one track, or a list of tracks."""
    if isinstance(spec, list):
        return list(spec)
    return list(_TRACK_SETS.get(spec, (spec,)))


@dataclass(frozen=True, eq=False)
class TrackKit:
    """A track with its optimal-line reference."""

    track: RaceTrack
    line: OptimalLine


def prepare_tracks(
    names: list[str], car: CarParams, params: RaceEnvParams
) -> dict[str, TrackKit]:
    margin = params.car_half_width_m + params.line_safety_m
    kits = {}
    for name in names:
        track = load_tum_track(name)
        kits[name] = TrackKit(track, optimal_line(track, car, margin))
    return kits


class MultiTrackEnv(gym.Env[ObsType, ActType]):
    """One env that drives a different track each episode.

    Every reset picks a track from the seeded RNG, so each of the training tracks is
    seen regardless of how many envs run in parallel, and runs reproduce.
    """

    def __init__(self, envs: dict[str, RaceEnv], seed: int) -> None:
        self.envs = envs
        self._names = list(envs)
        self._seed = seed
        self._current = self._names[0]
        self.action_space = envs[self._current].action_space
        self.observation_space = envs[self._current].observation_space
        self._first = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is None and self._first:
            seed = self._seed
        super().reset(seed=seed)
        self._first = False
        self._current = self._names[int(self.np_random.integers(len(self._names)))]
        inner_seed = int(self.np_random.integers(2**31 - 1))
        obs, info = self.envs[self._current].reset(seed=inner_seed)
        info["track"] = self._current
        return obs, info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.envs[self._current].step(action)
        info["track"] = self._current
        return obs, reward, terminated, truncated, info


def fill_with_demonstrations(
    model: SAC,
    envs: dict[str, RaceEnv],
    kits: dict[str, TrackKit],
    car: CarParams,
    steps: int,
    noise_std: float,
    seed: int,
) -> int:
    """Drive `steps` decisions with the scripted `LineFollower` (plus noise) on
    random-start episodes across `envs`, storing every transition in `model`'s
    replay buffer. Returns the number stored."""
    rng = np.random.default_rng(seed)
    names = list(envs)
    drivers = {n: LineFollower(car, kits[n].track, kits[n].line) for n in names}
    buffer = model.replay_buffer
    assert buffer is not None
    n = buffer.n_envs  # the buffer takes one transition per env at a time
    pending: list[tuple[Any, ...]] = []
    stored = 0
    while stored < steps:
        name = names[int(rng.integers(len(names)))]
        env, driver = envs[name], drivers[name]
        obs, info = env.reset(seed=int(rng.integers(2**31 - 1)))
        done = False
        while not done and stored < steps:
            action = driver.act(info)
            action = np.clip(
                action + rng.normal(0.0, noise_std, size=2), -1.0, 1.0
            ).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            pending.append(
                (obs, next_obs, action, reward, done, truncated and not terminated)
            )
            obs = next_obs
            stored += 1
            if len(pending) == n:
                o, no, a_, r, d, t = zip(*pending, strict=True)
                buffer.add(
                    np.array(o),
                    np.array(no),
                    np.array(a_),
                    np.array(r, dtype=np.float32),
                    np.array(d),
                    [{"TimeLimit.truncated": x} for x in t],
                )
                pending = []
    return stored


def drive_lap(
    model: Any,
    env: RaceEnv,
    seed: int,
    telemetry_every: int = 0,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One deterministic flying lap with `model`; optional decimated telemetry.
    ``options`` go to `RaceEnv.reset` (push, wear, opponent, gap_m)."""
    obs, info = env.reset(seed=seed, options=options)
    track = env.track
    channels = ("time_s", "distance_m", "speed_ms", "lateral_offset_m", "x", "y")
    telemetry: dict[str, list[float]] = {
        c: [] for c in (*channels, "throttle", "steer")
    }
    step = 0
    terminated = truncated = False
    while not (terminated or truncated):
        action, _ = model.predict(obs, deterministic=True)
        if telemetry_every and step % telemetry_every == 0:
            i = int(info["distance_m"] / track.ds_m) % track.n
            n = info["lateral_offset_m"]
            theta = float(track.heading[i])
            values = (
                info["elapsed_s"],
                info["distance_m"],
                info["speed_ms"],
                n,
                float(track.x[i]) - n * float(np.sin(theta)),
                float(track.y[i]) + n * float(np.cos(theta)),
            )
            for name, value in zip(channels, values, strict=True):
                telemetry[name].append(round(float(value), 3))
            clipped = np.clip(action, -1.0, 1.0)
            telemetry["throttle"].append(round(float(clipped[0]), 3))
            telemetry["steer"].append(round(float(clipped[1]), 3))
        obs, _reward, terminated, truncated, info = env.step(action)
        step += 1
    reference = env.reference
    result: dict[str, Any] = {
        "completed": bool(info["lap_completed"]),
        "off_track": bool(info["off_track"]),
        "lap_time_s": info.get("lap_time_s"),
        "distance_m": info["distance_m"],
        "optimal_lap_s": reference.lap_time_s if reference else None,
        "contact": bool(info.get("contact", False)),
        "passes": int(info.get("passes", 0)),
        "wear": float(info.get("wear", 0.0)),
    }
    if telemetry_every:
        result["telemetry"] = telemetry
    return result


def gap_percent(lap: dict[str, Any]) -> float | None:
    if not lap["completed"] or not lap["optimal_lap_s"]:
        return None
    return float(100.0 * (lap["lap_time_s"] / lap["optimal_lap_s"] - 1.0))


def scenario_success(lap: dict[str, Any], min_passes: int) -> bool:
    return bool(lap["completed"] and not lap["contact"] and lap["passes"] >= min_passes)


def selection_score(
    laps: list[dict[str, Any]], scenario_rate: float = 1.0
) -> tuple[float, float, float]:
    """Higher is better: completion rate, then the share of scenario laps (overtake,
    defend, push/conserve) that succeed, then a smaller mean gap to the optimal line
    (distance driven breaks ties while nothing completes)."""
    done = [gap for lap in laps if (gap := gap_percent(lap)) is not None]
    rate = len(done) / len(laps)
    if done:
        return rate, scenario_rate, -float(np.mean(done))
    return (
        rate,
        scenario_rate,
        float(np.mean([lap["distance_m"] for lap in laps])) - 1e9,
    )


class RacerCallback(BaseCallback):
    """Logs training episodes and runs the periodic evaluation."""

    def __init__(
        self,
        run_dir: Path,
        eval_kits: dict[str, dict[str, RaceEnv]],
        eval_every_steps: int,
        eval_seed: int,
        replay_every: int,
        scenarios: list[dict[str, Any]],
    ) -> None:
        super().__init__()
        self.scenarios = scenarios
        self.run_dir = run_dir
        self.eval_envs = eval_kits  # split -> track name -> lap-mode env
        self.eval_every = eval_every_steps
        self.eval_seed = eval_seed
        self.replay_every = replay_every
        self.best_score: tuple[float, float, float] | None = None
        self._next_eval = 0
        self._returns: np.ndarray = np.zeros(0)
        self._episode_steps: np.ndarray = np.zeros(0, dtype=int)
        self._file: Any = None
        self._writer: Any = None

    def _on_training_start(self) -> None:
        n = self.training_env.num_envs
        self._returns = np.zeros(n)
        self._episode_steps = np.zeros(n, dtype=int)
        self._file = (self.run_dir / "episodes.csv").open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            [
                "timesteps",
                "track",
                "steps",
                "failed",
                "return",
                "advanced_m",
                "mean_speed_ms",
            ]
        )
        with (self.run_dir / "eval.csv").open("w", newline="") as f:
            csv.writer(f).writerow(
                [
                    "timesteps",
                    "split",
                    "track",
                    "completed",
                    "off_track",
                    "lap_time_s",
                    "optimal_lap_s",
                    "gap_pct",
                ]
            )
        (self.run_dir / "eval_laps.jsonl").write_text("")

    def _on_step(self) -> bool:
        self._returns += self.locals["rewards"]
        self._episode_steps += 1
        for i, done in enumerate(self.locals["dones"]):
            if not done:
                continue
            info = self.locals["infos"][i]
            elapsed = max(info["elapsed_s"], 1e-9)
            self._writer.writerow(
                [
                    self.num_timesteps,
                    info.get("track", ""),
                    self._episode_steps[i],
                    int(not info.get("TimeLimit.truncated", False)),
                    f"{self._returns[i]:.3f}",
                    f"{info['progress_m']:.1f}",
                    f"{info['progress_m'] / elapsed:.2f}",
                ]
            )
            self._returns[i] = 0.0
            self._episode_steps[i] = 0
        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        assert isinstance(self.model, SAC)
        scored: list[dict[str, Any]] = []
        parts = []
        for split, envs in self.eval_envs.items():
            gaps = []
            for name, env in envs.items():
                lap = drive_lap(
                    self.model, env, self.eval_seed, telemetry_every=self.replay_every
                )
                gap = gap_percent(lap)
                with (self.run_dir / "eval_laps.jsonl").open("a") as f:
                    f.write(
                        json.dumps(
                            {
                                "timesteps": self.num_timesteps,
                                "split": split,
                                "track": name,
                                **lap,
                            }
                        )
                        + "\n"
                    )
                lap.pop("telemetry", None)
                with (self.run_dir / "eval.csv").open("a", newline="") as f:
                    csv.writer(f).writerow(
                        [
                            self.num_timesteps,
                            split,
                            name,
                            int(lap["completed"]),
                            int(lap["off_track"]),
                            ""
                            if lap["lap_time_s"] is None
                            else f"{lap['lap_time_s']:.3f}",
                            f"{lap['optimal_lap_s']:.3f}",
                            "" if gap is None else f"{gap:.2f}",
                        ]
                    )
                if split == "validation":
                    scored.append(lap)
                if gap is not None:
                    gaps.append(gap)
            n = len(envs)
            text = f"{len(gaps)}/{n}"
            if gaps:
                text += f" +{np.mean(gaps):.1f}%"
            parts.append(f"{split} {text}")
            self.logger.record(f"eval/{split}_completed", len(gaps) / n)
            if gaps:
                self.logger.record(f"eval/{split}_gap_pct", float(np.mean(gaps)))
        wins = []
        for sc in self.scenarios:
            for env in self.eval_envs["validation"].values():
                lap = drive_lap(self.model, env, self.eval_seed, options=sc["options"])
                wins.append(scenario_success(lap, int(sc.get("min_passes", 0))))
        if wins:
            parts.append(f"scenarios {sum(wins)}/{len(wins)}")
            self.logger.record("eval/scenario_rate", sum(wins) / len(wins))
        score = selection_score(scored, sum(wins) / len(wins) if wins else 1.0)
        improved = self.best_score is None or score > self.best_score
        if improved:
            self.best_score = score
            self.model.save(self.run_dir / "model_best.zip")
        print(
            f"[eval] {self.num_timesteps:>9,} steps  "
            + " | ".join(parts)
            + ("  *best" if improved else ""),
            flush=True,
        )

    def _on_training_end(self) -> None:
        self._evaluate()
        if self._file is not None:
            self._file.close()


def train(config: RacerConfig, run_dir: Path, car: CarParams = EXAMPLE_CAR) -> SAC:
    run = config.run
    torch.set_num_threads(run.torch_threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "run": asdict(run),
                "sim": config.sim,
                "env": config.env,
                "sac": config.sac,
                "policy": config.policy,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    params = config.env_params()
    train_names = resolve_track_names(run.train_tracks)
    val_names = resolve_track_names(run.validation_tracks)
    test_names = resolve_track_names(run.held_out_tracks)
    overlap = (set(train_names) & set(val_names)) | (set(train_names) & set(test_names))
    if overlap or set(val_names) & set(test_names):
        raise ValueError(f"train/validation/held-out tracks overlap: {sorted(overlap)}")

    kits = prepare_tracks(train_names + val_names + test_names, car, params)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "python": platform.python_version(),
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
                "gymnasium": gym.__version__,
                "car": asdict(car),
                "env_params": asdict(params),
                "splits": {
                    "train": train_names,
                    "validation": val_names,
                    "held_out": test_names,
                },
                "optimal_lap_s": {n: k.line.lap_time_s for n, k in kits.items()},
                "centreline_lap_s": {
                    n: k.line.centre_lap_time_s for n, k in kits.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def make_envs(names: list[str], mode: str, seed: int) -> dict[str, RaceEnv]:
        return {
            n: RaceEnv(
                car, kits[n].track, config.sim_params(seed), params, kits[n].line, mode
            )
            for n in names
        }

    def make_multi(i: int) -> Callable[[], MultiTrackEnv]:
        def _make() -> MultiTrackEnv:
            envs = make_envs(train_names, "random", run.seed + i)
            return MultiTrackEnv(envs, run.seed + i)

        return _make

    vec_env = DummyVecEnv([make_multi(i) for i in range(run.n_envs)])
    sample = train_names[:: max(1, len(train_names) // max(1, run.eval_train_tracks))]
    eval_envs = {
        "validation": make_envs(val_names, "lap", run.eval_seed),
        "held_out": make_envs(test_names, "lap", run.eval_seed),
        "train": make_envs(sample[: run.eval_train_tracks], "lap", run.eval_seed),
    }

    policy_kwargs = dict(config.policy)
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = getattr(
            torch.nn, policy_kwargs["activation_fn"]
        )
    sac_kwargs = dict(config.sac)
    rate = sac_kwargs.get("learning_rate")
    if isinstance(rate, dict):  # {start, end}: linear decay over the run
        start, end = float(rate["start"]), float(rate["end"])
        sac_kwargs["learning_rate"] = lambda progress: end + (start - end) * progress
    model = SAC(
        "MlpPolicy",
        vec_env,
        seed=run.seed,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
        **sac_kwargs,
    )
    if run.init_from:
        model.set_parameters(run.init_from, device="cpu")
    model.set_logger(configure(str(run_dir), ["csv"]))
    if run.demo_steps:
        stored = fill_with_demonstrations(
            model,
            make_envs(train_names, "random", run.seed + 10_000),
            kits,
            car,
            run.demo_steps,
            run.demo_noise_std,
            run.seed,
        )
        print(f"Stored {stored:,} demonstration transitions.", flush=True)
    callback = RacerCallback(
        run_dir,
        eval_envs,
        run.eval_every_steps,
        run.eval_seed,
        run.replay_every_decisions,
        run.eval_scenarios,
    )
    model.learn(total_timesteps=run.total_timesteps, callback=callback)
    model.save(run_dir / "model_final.zip")
    return model
