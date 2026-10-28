"""Driver scenarios: ``uv run python -m f1rl.scenarios --policy <name>``.

Drives the tactical driver (`TacticalEnv`) through the situations it must handle on
real tracks and prints what happened, per scenario:

- **push / conserve**: solo laps at push 1.0 and 0.0 -- lap gap to the optimal line
  and tyre wear used. Pushing must be quicker; conserving must wear less.
- **pass_***: a backmarker / slower / equal / covering car 25 m ahead -- passed
  without contact?
- **defend**: a quicker attacker 25 m behind and the driver on half-worn tyres --
  kept behind, without contact?

Every car runs under the no-crash safety filter (`models.safety`); the tables count
contacts (must be 0) and how often the filter had to step in for our car.

``--real`` instead replays real-race starts (`f1rl.real_starts`) on those tracks:
every real pass and defence there, as an attack and as a defence. An attack wins by
ending the battle with a pass that stuck (`RaceEnvParams.pass_hold_m`), a defence by
ending it without a stuck pass against it; both without contact or leaving the track.
The battle runs the env's whole episode (`episode_s`), not just until the first pass.

Policies: ``line`` never makes a move (the control: what the pace alone gives),
``tactician`` is the rule-based expert, or a saved PPO model. Everything is
deterministic (flying start from the line, fixed seed).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.agents.tactician import choose
from f1rl.agents.tactics import MOVES
from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRACKS, TUM_VALIDATION, load_tum_track
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from f1rl.real_starts import RealStart, by_track, load_real_starts

#: name -> (reset options, kind). kind: "solo" | "pass" (need +1) | "hold" (not passed)
SCENARIOS: dict[str, tuple[dict[str, Any], str]] = {
    "push": ({"push": 1.0}, "solo"),
    "conserve": ({"push": 0.0}, "solo"),
    "pass_backmarker": ({"opponent": "backmarker", "gap_m": 25.0}, "pass"),
    "pass_slow": ({"opponent": "slow", "gap_m": 25.0}, "pass"),
    "pass_equal": ({"opponent": "equal", "gap_m": 25.0}, "pass"),
    "pass_defender": ({"opponent": "defender", "gap_m": 25.0}, "pass"),
    "defend": ({"opponent": "attacker", "gap_m": 25.0, "wear": 0.5}, "hold"),
}
STEPS = 600  # decisions of 0.5 s: 300 s, longer than any lap
REAL_STEPS = 600  # a real start runs to the env's own end (`episode_s`), at most 300 s


def env_params() -> RaceEnvParams:
    return RaceEnvParams(
        tactical=True,
        tires=True,
        opponents=("none", "backmarker", "slow", "equal", "defender", "attacker"),
        safety_filter=True,
    )


def make_env(name: str, params: RaceEnvParams, mode: str = "lap") -> TacticalEnv:
    track = load_tum_track(name)
    line = optimal_line(
        track, EXAMPLE_CAR, params.car_half_width_m + params.line_safety_m
    )
    return TacticalEnv(
        EXAMPLE_CAR,
        track,
        SimParams(seed=0, max_episode_steps=10_000),
        params,
        line,
        mode,
    )


def pick(env: TacticalEnv, obs: Any, policy: Any) -> int:
    if policy == "line":
        return 0
    if policy == "tactician":
        return MOVES.index(choose(env))
    return int(policy.predict(obs, deterministic=True)[0])


def run_one(
    env: TacticalEnv, options: dict[str, Any], kind: str, policy: Any
) -> dict[str, Any]:
    obs, info = env.reset(seed=0, options=options)
    done = False
    steps = 0
    while not done and steps < STEPS:
        obs, _, terminated, truncated, info = env.step(pick(env, obs, policy))
        done = terminated or truncated
        steps += 1
        if kind == "pass" and info["passes"] >= 1:
            break  # passed: that is the result
    if kind == "solo":
        ok = bool(info["lap_completed"])
    elif kind == "pass":
        ok = info["passes"] >= 1 and not info["contact"] and not info["off_track"]
    else:  # hold: still ahead when the lap ends (or the run does), nothing hit
        ok = info["passes"] >= 0 and not info["contact"] and not info["off_track"]
    gap = None
    if info["lap_completed"]:
        gap = 100 * (info["lap_time_s"] / env.reference.lap_time_s - 1)  # type: ignore[union-attr]
    return {
        "success": bool(ok),
        "passes": int(info["passes"]),
        "swaps": int(info["swaps"]),
        "contact": bool(info["contact"]),
        "off_track": bool(info["off_track"]),
        "gap_pct": gap,
        "wear": float(info["wear"]),
        "interventions": int(info["interventions"]),
    }


def run_real(env: TacticalEnv, start: RealStart, policy: Any) -> dict[str, Any]:
    """One real-race start (env in "random" mode, everything fixed by options)."""
    obs, info = env.reset(seed=0, options={**start.options(), "push": 1.0, "wear": 0.0})
    for _ in range(REAL_STEPS):
        obs, _, terminated, truncated, info = env.step(pick(env, obs, policy))
        if terminated or truncated:
            break
    clean = not info["contact"] and not info["off_track"]
    won = info["passes"] >= 1 if start.role == "attack" else info["passes"] >= 0
    return {
        "track": start.track,
        "role": start.role,
        "opponent": start.opponent,
        "source": start.source,
        "real_outcome": start.real_outcome,
        "success": bool(won and clean),
        "passes": int(info["passes"]),
        "swaps": int(info["swaps"]),
        "contact": bool(info["contact"]),
        "off_track": bool(info["off_track"]),
        "interventions": int(info["interventions"]),
    }


def run_real_suite(
    policy: Any,
    names: list[str],
    starts: dict[str, list[RealStart]],
    params: RaceEnvParams | None = None,
) -> list[dict[str, Any]]:
    params = params or env_params()
    rows = []
    for name in names:
        if not starts.get(name):
            continue
        env = make_env(name, params, "random")
        rows += [run_real(env, st, policy) for st in starts[name]]
    return rows


def print_real_table(rows: list[dict[str, Any]]) -> None:
    head = f"{'real starts':24} {'won':>9} {'contact':>7} {'off':>4} {'filter':>7}"
    print(head + f" {'swaps':>6}")
    for role in ("attack", "defend"):
        for opp in sorted({r["opponent"] for r in rows if r["role"] == role}):
            sel = [r for r in rows if r["role"] == role and r["opponent"] == opp]
            print(
                f"{role + ' vs ' + opp:24} "
                f"{sum(r['success'] for r in sel):>4}/{len(sel):<4} "
                f"{sum(r['contact'] for r in sel):>7} "
                f"{sum(r['off_track'] for r in sel):>4} "
                f"{sum(r['interventions'] for r in sel):>7} "
                f"{sum(r['swaps'] for r in sel) / len(sel):>6.1f}"
            )


def run(
    policy: Any, names: list[str], params: RaceEnvParams | None = None
) -> list[dict[str, Any]]:
    params = params or env_params()
    rows = []
    for name in names:
        env = make_env(name, params)
        for scenario, (options, kind) in SCENARIOS.items():
            rows.append(
                {
                    "track": name,
                    "scenario": scenario,
                    **run_one(env, options, kind, policy),
                }
            )
    return rows


def print_table(rows: list[dict[str, Any]]) -> None:
    head = ("scenario", "ok", "contact", "off", "lap gap", "wear", "filter")
    print(
        f"{head[0]:16} {head[1]:>6} {head[2]:>7} {head[3]:>4} {head[4]:>9} {head[5]:>6}"
        f" {head[6]:>7}"
    )
    for scenario in SCENARIOS:
        sel = [r for r in rows if r["scenario"] == scenario]
        gaps = [r["gap_pct"] for r in sel if r["gap_pct"] is not None]
        wear = [r["wear"] for r in sel if r["gap_pct"] is not None]
        print(
            f"{scenario:16} {sum(r['success'] for r in sel):>3}/{len(sel):<2} "
            f"{sum(r['contact'] for r in sel):>7} "
            f"{sum(r['off_track'] for r in sel):>4} "
            f"{(f'{np.mean(gaps):+.2f}%' if gaps else '--'):>9} "
            f"{(f'{np.mean(wear):.3f}' if wear else '--'):>6} "
            f"{sum(r['interventions'] for r in sel):>7}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the driver scenario suite.")
    parser.add_argument(
        "--policy", default="tactician", help="line | tactician | model.zip"
    )
    parser.add_argument("--tracks", default="tum_held_out")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--real", action="store_true", help="replay real-race starts instead"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="the model's training YAML (default: config.yaml next to the model)",
    )
    args = parser.parse_args()
    sets = {
        "tum_held_out": TUM_HELD_OUT,
        "tum_validation": TUM_VALIDATION,
        "tum_all": TUM_TRACKS,
    }
    names: list[str] = []
    for part in args.tracks.split(","):
        names += list(sets.get(part.strip(), (part.strip(),)))
    policy: Any = args.policy
    params = env_params()
    if args.config is not None and args.config.exists():
        from f1rl.agents.tactical import load_tactical_config

        params = replace(
            load_tactical_config(args.config).env_params(), safety_filter=True
        )
    if policy not in ("line", "tactician"):
        from stable_baselines3 import PPO

        from f1rl.agents.tactical import load_tactical_config

        # A model sees what it was trained on: its own env settings (observation
        # features, opponents), always with the no-crash filter on.
        config = args.config or Path(policy).with_name("config.yaml")
        if config.exists():
            trained = load_tactical_config(config).env_params()
            params = replace(trained, safety_filter=True)
        policy = PPO.load(policy, device="cpu")
    if args.real:
        rows = run_real_suite(policy, names, by_track(load_real_starts()), params)
        print_real_table(rows)
    else:
        rows = run(policy, names, params)
        print_table(rows)
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
