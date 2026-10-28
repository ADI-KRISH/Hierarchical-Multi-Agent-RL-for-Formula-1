"""Pass gate: can the scripted tactician (or a model) pass and defend at all?

``uv run python scripts/pass_gate.py --config configs/X.yaml --policy tactician``

Battles of the config's ``episode_s`` from several points round each track: attack
a slower / equal / covering car ahead, defend against an equal or quicker car
behind. A pass counts only once it has stuck (``pass_hold_m``). Prints won /
contact / off-track / mean swaps per battle. RL is only worth running once the
scripted tactician can pass an equal-pace car here (the world allows it).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.agents.tactical import load_tactical_config
from f1rl.envs.racetrack import TUM_TRAIN, TUM_VALIDATION
from f1rl.scenarios import make_env, pick

BATTLES = {
    "attack_slow": ("slow", "attack"),
    "attack_equal": ("equal", "attack"),
    "attack_defender": ("defender", "attack"),
    "defend_equal": ("attacker_equal", "defend"),
    "defend_faster": ("attacker", "defend"),
}


def battle(env: Any, policy: Any, kind: str, role: str, s0: float, gap: float) -> dict:
    obs, info = env.reset(
        seed=0,
        options={
            "start_s": s0,
            "opponent": kind,
            "gap_m": gap,
            "push": 1.0,
            "wear": 0.0,
        },
    )
    for _ in range(2000):
        obs, _, term, trunc, info = env.step(pick(env, obs, policy))
        if term or trunc:
            break
    clean = not info["contact"] and not info["off_track"]
    won = info["passes"] >= 1 if role == "attack" else info["passes"] >= 0
    return {
        "won": bool(won and clean),
        "contact": bool(info["contact"]),
        "off": bool(info["off_track"]),
        "swaps": int(info["swaps"]),
        "interventions": int(info["interventions"]),
        "passes": int(info["passes"]),
        "elapsed_s": float(info["elapsed_s"]),
        "gap_end_m": float(env._opp.state.s_m - env._state.s_m) if env._opp else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policy", default="tactician")
    ap.add_argument("--tracks", default="validation")
    ap.add_argument("--starts", type=int, default=4)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    params = replace(load_tactical_config(args.config).env_params(), safety_filter=True)
    names = {"validation": TUM_VALIDATION, "train": TUM_TRAIN[:6]}.get(
        args.tracks, tuple(args.tracks.split(","))
    )
    policy: Any = args.policy
    if policy not in ("line", "tactician"):
        from stable_baselines3 import PPO

        policy = PPO.load(policy, device="cpu")
    rows = []
    for name in names:
        env = make_env(name, params, "random")
        for k in range(args.starts):
            s0 = env._lap_m * k / args.starts
            for b, (kind, role) in BATTLES.items():
                r = battle(env, policy, kind, role, s0, args.gap)
                rows.append({"track": name, "battle": b, **r})
    print(
        f"{'battle':16} {'won':>7} {'contact':>7} {'off':>4} {'swaps':>6} {'filter':>7}"
    )
    for b in BATTLES:
        sel = [r for r in rows if r["battle"] == b]
        print(
            f"{b:16} {sum(r['won'] for r in sel):>3}/{len(sel):<3} "
            f"{sum(r['contact'] for r in sel):>7} {sum(r['off'] for r in sel):>4} "
            f"{np.mean([r['swaps'] for r in sel]):>6.2f} "
            f"{np.mean([r['interventions'] for r in sel]):>7.0f}"
        )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
