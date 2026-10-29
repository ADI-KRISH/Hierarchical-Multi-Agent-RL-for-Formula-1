"""Section gate: can the driver pass a slower car in each section of each track?

``uv run python scripts/section_gate.py --config configs/X.yaml --policy tactician``

Every straight, braking zone and corner of each track (`f1rl.sections`) is one
battle per opponent kind: our car starts ``--gap`` m behind the opponent at the
section's start and wins only if it is ahead at the section's end, the pass then
sticks (``pass_hold_m``), and nothing is hit or run off. Prints wins by section kind
and opponent, and writes every battle (the per-section scorecard) to ``--out``.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import yaml

from f1rl.agents.tactical import use_defender, load_tactical_config, section_battle
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRAIN, TUM_VALIDATION
from f1rl.scenarios import make_env
from f1rl.sections import KINDS, env_sections


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policy", default="tactician")
    ap.add_argument("--tracks", default="validation")
    ap.add_argument("--opponents", default="slow,guard_slow")
    ap.add_argument("--gap", type=float, default=9.0)
    ap.add_argument(
        "--defender", default="", help="trained defender model for rival_ahead* cars"
    )
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument(
        "--set", nargs="*", default=[], help="env overrides, e.g. pull_moves=true"
    )
    args = ap.parse_args()
    config = load_tactical_config(args.config)
    extra = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    params = replace(config.env_params(), safety_filter=True, **extra)
    sets = {
        "validation": TUM_VALIDATION,
        "train": TUM_TRAIN,
        "held_out": TUM_HELD_OUT,
    }
    names = sets.get(args.tracks, tuple(args.tracks.split(",")))
    policy: Any = args.policy
    if policy not in ("line", "tactician"):
        from stable_baselines3 import PPO

        policy = PPO.load(policy, device="cpu")
    opponents = args.opponents.split(",")
    rows = []
    for name in names:
        env = make_env(name, params, "random")
        if args.defender:
            use_defender([env], args.defender, ("line", "cover", "inside"))
        for sec in env_sections(env, config.section_params()):
            for opp in opponents:
                r = section_battle(env, policy, sec, opp, args.gap)
                rows.append({**sec.as_dict(), "opponent": opp, **r})
        done = [r for r in rows if r["track"] == name]
        print(
            f"{name:14} {sum(r['won'] for r in done):>3}/{len(done):<3} "
            f"contacts {sum(r['contact'] for r in done)}",
            flush=True,
        )
    print(f"\n{'section':10} " + " ".join(f"{o:>14}" for o in opponents))
    for kind in KINDS:
        cells = []
        for opp in opponents:
            sel = [r for r in rows if r["kind"] == kind and r["opponent"] == opp]
            cells.append(f"{sum(r['won'] for r in sel):>6}/{len(sel):<7}")
        print(f"{kind:10} " + " ".join(cells))
    print(
        f"total {sum(r['won'] for r in rows)}/{len(rows)}, contacts "
        f"{sum(r['contact'] for r in rows)}, off track {sum(r['off'] for r in rows)}"
    )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
