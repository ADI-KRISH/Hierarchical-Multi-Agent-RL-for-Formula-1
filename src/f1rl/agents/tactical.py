"""PPO for `TacticalEnv`: learns WHEN to attack or defend and WHICH move to make.

Everything tunable comes from an experiment YAML (``configs/tactical.yaml``). A run
writes into ``runs/<name>/``: ``config.yaml``, ``meta.json``, ``eval.csv`` (the
scenario suite on the validation tracks at every evaluation), ``model_best.zip``
(most scenarios won on the VALIDATION tracks; a contact-free evaluation always beats
one with contact) and ``model_final.zip``.

**Real starts** (`f1rl.real_starts`): with ``real_start_frac`` > 0, that share of
training episodes starts from a real pass or defence on the track being driven
(train-split races only), and the validation evaluation also replays the
validation tracks' real starts. Turn on ``safety_filter`` in ``env`` so no car ever
drives into another (`models.safety`); every evaluation reports contacts.

Optionally the policy first imitates the rule-based `tactician` (behaviour cloning),
so PPO starts from a driver that already passes backmarkers and improves from there.
Track splits are the usual ones: train on `TUM_TRAIN`, choose on `TUM_VALIDATION`,
judge on `TUM_HELD_OUT` (never trained on, never used to choose anything).
"""

import copy
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
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.vec_env import DummyVecEnv

from f1rl.agents.ppo import git_sha
from f1rl.agents.racer import prepare_tracks, resolve_track_names
from f1rl.agents.tactician import choose
from f1rl.agents.tactics import MOVES
from f1rl.config import EXAMPLE_CAR, CarParams, RaceEnvParams, SimParams
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.real_starts import ROLES, RealStart, by_track, load_real_starts
from f1rl.scenarios import SCENARIOS, run_one, run_real


@dataclass(frozen=True)
class TacticalRun:
    name: str
    seed: int
    total_timesteps: int
    n_envs: int
    eval_every_steps: int
    train_tracks: str | list[str] = "tum_train"
    validation_tracks: str | list[str] = "tum_validation"
    held_out_tracks: str | list[str] = "tum_held_out"
    torch_threads: int = 4
    # Behaviour cloning of the rule-based tactician before PPO: decisions to record
    # and passes over them. 0 = PPO from scratch.
    bc_decisions: int = 0
    bc_epochs: int = 10
    # Share of training episodes started from a real race situation (0 = none), and
    # where the incident library is (relative to the project, or absolute).
    real_start_frac: float = 0.0
    incident_dir: str = "data/incidents"
    # At most this many real starts per validation track in each evaluation (spread
    # evenly through the library), to keep evaluations quick.
    real_eval_per_track: int = 40
    # Self-play (GT Sophy's opponent population; Wurman et al., Nature 2022): this
    # share of training episodes races a ``rival_*`` car driven by a frozen copy of
    # the policy, taken every ``pool_every_steps`` (the last ``pool_size`` kept);
    # the rest race the scripted opponents. 0 = scripted only.
    rival_frac: float = 0.0
    pool_every_steps: int = 100_000
    pool_size: int = 10
    # Start from an earlier model's weights (a .zip path; "" = fresh, then the
    # behaviour cloning above if any).
    init_model: str = ""


@dataclass(frozen=True)
class TacticalConfig:
    run: TacticalRun
    sim: dict[str, Any] = field(default_factory=dict)
    env: dict[str, Any] = field(default_factory=dict)  # RaceEnvParams overrides
    ppo: dict[str, Any] = field(default_factory=dict)  # PPO(...) keyword args
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
            "real_gap_m",
            "pass_window_m",
        ):
            if key in overrides:
                overrides[key] = tuple(overrides[key])
        return RaceEnvParams(**{"tactical": True, "tires": True, **overrides})

    def sim_params(self, seed: int) -> SimParams:
        return SimParams(seed=seed, **self.sim)


def load_tactical_config(path: Path) -> TacticalConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    unknown = set(raw) - {f.name for f in fields(TacticalConfig)} - {"tactical"}
    if unknown:
        raise ValueError(f"unknown config sections: {sorted(unknown)}")
    config = TacticalConfig(
        run=TacticalRun(**raw["run"]),
        sim=raw.get("sim") or {},
        env=raw.get("env") or {},
        ppo=raw.get("ppo") or {},
        policy=raw.get("policy") or {},
    )
    config.env_params()
    config.sim_params(0)
    return config


class RivalPool:
    """Frozen copies of the policy for self-play, newest last."""

    def __init__(self, size: int) -> None:
        self.size = size
        self.policies: list[Any] = []

    def add(self, policy: Any) -> None:
        frozen = copy.deepcopy(policy)
        frozen.set_training_mode(False)
        self.policies = [*self.policies, frozen][-self.size :]

    def pick(self, rng: np.random.Generator) -> Callable[[Any], int]:
        policy = self.policies[int(rng.integers(len(self.policies)))]

        def drive(obs: Any) -> int:
            return int(policy.predict(obs, deterministic=True)[0])

        return drive


class MultiTrackTactical(gym.Env[Any, Any]):
    """One env that drives a different track each episode (seeded RNG)."""

    def __init__(
        self,
        envs: dict[str, TacticalEnv],
        seed: int,
        starts: dict[str, list[RealStart]] | None = None,
        real_frac: float = 0.0,
        pool: RivalPool | None = None,
        rival_frac: float = 0.0,
    ) -> None:
        self.envs = envs
        self.starts = starts or {}
        self.real_frac = real_frac
        self.pool = pool
        self.rival_frac = rival_frac
        self._names = list(envs)
        self._seed = seed
        self._current = self._names[0]
        self.action_space = envs[self._current].action_space
        self.observation_space = envs[self._current].observation_space
        self._first = True

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Any, dict[str, Any]]:
        if seed is None and self._first:
            seed = self._seed
        super().reset(seed=seed)
        self._first = False
        self._current = self._names[int(self.np_random.integers(len(self._names)))]
        here = self.starts.get(self._current, [])
        env = self.envs[self._current]
        env.rival = None
        rival = bool(
            self.pool
            and self.pool.policies
            and self.np_random.random() < self.rival_frac
        )
        opts: dict[str, Any] | None = None
        if here and self.np_random.random() < self.real_frac:
            start = here[int(self.np_random.integers(len(here)))]
            opts = start.options()
            if rival:  # the rival takes the real opponent's place
                ahead = start.role == "attack"
                opts["opponent"] = "rival_ahead" if ahead else "rival_behind"
        elif rival:
            side = ("rival_ahead", "rival_behind")[int(self.np_random.integers(2))]
            opts = {"opponent": side}
        if rival:
            assert self.pool is not None
            env.rival = self.pool.pick(self.np_random)
        obs, info = env.reset(
            seed=int(self.np_random.integers(2**31 - 1)), options=opts
        )
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self.envs[self._current].step(action)


def spread(items: list[RealStart], n: int) -> list[RealStart]:
    """At most ``n`` of ``items``, evenly spaced (deterministic)."""
    if len(items) <= n:
        return items
    return [items[i * len(items) // n] for i in range(n)]


def clone_tactician(
    model: PPO, envs: dict[str, TacticalEnv], decisions: int, epochs: int, seed: int
) -> float:
    """Train ``model``'s policy to pick the moves the rule-based tactician picks.
    Returns the final share of decisions it now agrees with."""
    rng = np.random.default_rng(seed)
    names = list(envs)
    obs_list: list[Any] = []
    act_list: list[int] = []
    while len(obs_list) < decisions:
        env = envs[names[int(rng.integers(len(names)))]]
        obs, _ = env.reset(seed=int(rng.integers(2**31 - 1)))
        done = False
        while not done and len(obs_list) < decisions:
            move = MOVES.index(choose(env))
            obs_list.append(obs)
            act_list.append(move)
            obs, _, terminated, truncated, _ = env.step(move)
            done = terminated or truncated
    x = torch.as_tensor(np.array(obs_list), dtype=torch.float32)
    y = torch.as_tensor(act_list, dtype=torch.long)
    opt = torch.optim.Adam(model.policy.parameters(), lr=1e-3)
    for _ in range(epochs):
        order = torch.randperm(len(x))
        for i in range(0, len(x), 512):
            idx = order[i : i + 512]
            _, log_prob, _ = model.policy.evaluate_actions(x[idx], y[idx])
            loss = -log_prob.mean()
            opt.zero_grad()
            loss.backward()  # type: ignore[no-untyped-call]
            opt.step()
    with torch.no_grad():
        pred = model.policy.get_distribution(x).get_actions(deterministic=True)
    return float((pred == y).float().mean())


class PoolCallback(BaseCallback):
    """Adds a frozen copy of the policy to the self-play pool every so often."""

    def __init__(self, pool: RivalPool, every_steps: int) -> None:
        super().__init__()
        self.pool = pool
        self.every = every_steps
        self._next = every_steps

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.every
            self.pool.add(self.model.policy)
        return True


class ScenarioCallback(BaseCallback):
    """Runs the scenario suite on the validation tracks; keeps the best model."""

    def __init__(
        self,
        run_dir: Path,
        envs: dict[str, TacticalEnv],
        eval_every_steps: int,
        real_envs: dict[str, TacticalEnv] | None = None,
        starts: dict[str, list[RealStart]] | None = None,
    ) -> None:
        super().__init__()
        self.run_dir = run_dir
        self.envs = envs
        self.real_envs = real_envs or {}
        self.starts = starts or {}
        self.eval_every = eval_every_steps
        self.best: tuple[bool, float] = (False, -1.0)  # (no contact, score)
        self._next = 0

    def _on_training_start(self) -> None:
        with (self.run_dir / "eval.csv").open("w", newline="") as f:
            csv.writer(f).writerow(
                ["timesteps", "scenario", "wins", "runs", "contacts"]
            )

    def _on_step(self) -> bool:
        if self.num_timesteps >= self._next:
            self._next += self.eval_every
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        rows = []
        for scenario, (options, kind) in SCENARIOS.items():
            res = [
                run_one(env, options, kind, self.model) for env in self.envs.values()
            ]
            rows.append(
                (
                    scenario,
                    sum(r["success"] for r in res),
                    len(res),
                    sum(r["contact"] for r in res),
                )
            )
        real = [
            run_real(self.real_envs[name], st, self.model)
            for name, sts in self.starts.items()
            if name in self.real_envs
            for st in sts
        ]
        for role in ROLES:
            res = [r for r in real if r["role"] == role]
            if res:
                rows.append(
                    (
                        f"real_{role}",
                        sum(r["success"] for r in res),
                        len(res),
                        sum(r["contact"] for r in res),
                    )
                )
        with (self.run_dir / "eval.csv").open("a", newline="") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow([self.num_timesteps, *row])
        score = sum(r[1] for r in rows) / sum(r[2] for r in rows)
        contacts = sum(r[3] for r in rows)
        # A model with any contact never beats one without: no crashes comes first.
        key = (contacts == 0, score)
        improved = key > self.best
        if improved:
            self.best = key
            self.model.save(self.run_dir / "model_best.zip")
        self.logger.record("eval/scenario_rate", score)
        self.logger.record("eval/contacts", contacts)
        text = " ".join(f"{s}:{w}/{n}" for s, w, n, _ in rows)
        print(
            f"[eval] {self.num_timesteps:>9,}  {score:5.1%}  contacts {contacts}  "
            + text
            + ("  *best" if improved else ""),
            flush=True,
        )

    def _on_training_end(self) -> None:
        self._evaluate()


def train(config: TacticalConfig, run_dir: Path, car: CarParams = EXAMPLE_CAR) -> PPO:
    run = config.run
    torch.set_num_threads(run.torch_threads)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "tactical": True,
                "run": asdict(run),
                "sim": config.sim,
                "env": config.env,
                "ppo": config.ppo,
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
    if set(train_names) & (set(val_names) | set(test_names)) or set(val_names) & set(
        test_names
    ):
        raise ValueError("train/validation/held-out tracks overlap")
    kits = prepare_tracks(train_names + val_names + test_names, car, params)
    (run_dir / "meta.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "python": platform.python_version(),
                "stable_baselines3": stable_baselines3.__version__,
                "torch": torch.__version__,
                "car": asdict(car),
                "env_params": asdict(params),
                "splits": {
                    "train": train_names,
                    "validation": val_names,
                    "held_out": test_names,
                },
                "moves": list(MOVES),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    def make_envs(names: list[str], mode: str, seed: int) -> dict[str, TacticalEnv]:
        return {
            n: TacticalEnv(
                car, kits[n].track, config.sim_params(seed), params, kits[n].line, mode
            )
            for n in names
        }

    starts: dict[str, list[RealStart]] = {}
    val_starts: dict[str, list[RealStart]] = {}
    if run.real_start_frac > 0:
        library = load_real_starts(Path(run.incident_dir))
        starts = {
            k: v for k, v in by_track(library, ["train"]).items() if k in train_names
        }
        val_starts = {
            k: spread(v, run.real_eval_per_track)
            for k, v in by_track(library, ["validation"]).items()
            if k in val_names
        }
        print(
            f"Real starts: {sum(map(len, starts.values()))} for training on "
            f"{len(starts)} tracks, {sum(map(len, val_starts.values()))} for "
            "validation",
            flush=True,
        )

    pool = RivalPool(run.pool_size)

    def make_multi(i: int) -> Callable[[], MultiTrackTactical]:
        def _make() -> MultiTrackTactical:
            return MultiTrackTactical(
                make_envs(train_names, "random", run.seed + i),
                run.seed + i,
                starts,
                run.real_start_frac,
                pool,
                run.rival_frac,
            )

        return _make

    vec_env = DummyVecEnv([make_multi(i) for i in range(run.n_envs)])
    policy_kwargs = dict(config.policy)
    if "activation_fn" in policy_kwargs:
        policy_kwargs["activation_fn"] = getattr(
            torch.nn, policy_kwargs["activation_fn"]
        )
    model = PPO(
        "MlpPolicy",
        vec_env,
        seed=run.seed,
        device="cpu",
        policy_kwargs=policy_kwargs,
        verbose=0,
        **config.ppo,
    )
    model.set_logger(configure(str(run_dir), ["csv"]))
    if run.init_model:
        model.set_parameters(run.init_model, exact_match=True, device="cpu")
        print(f"Starting from {run.init_model}", flush=True)
    elif run.bc_decisions:
        agree = clone_tactician(
            model,
            make_envs(train_names, "random", run.seed + 20_000),
            run.bc_decisions,
            run.bc_epochs,
            run.seed,
        )
        print(
            f"Behaviour cloning: agrees with the tactician on {agree:.1%}", flush=True
        )
    callback = ScenarioCallback(
        run_dir,
        make_envs(val_names, "lap", run.seed + 30_000),
        run.eval_every_steps,
        make_envs(list(val_starts), "random", run.seed + 40_000),
        val_starts,
    )
    callbacks: list[BaseCallback] = [callback]
    if run.rival_frac > 0:
        pool.add(model.policy)  # the first rival: the starting policy itself
        callbacks.append(PoolCallback(pool, run.pool_every_steps))
    model.learn(total_timesteps=run.total_timesteps, callback=CallbackList(callbacks))
    model.save(run_dir / "model_final.zip")
    return model
