"""Guard gate: is the defending sparring driver (``guard_*``) harder to pass than a
car that just holds the racing line, at the same pace, and clean?

``uv run python scripts/guard_gate.py --config configs/guard_sparring.yaml``

Our car attacks with the scripted tactician (and, as a control, ``line``: no moves)
from several points round each track. Each pairing compares a plain car with the
guard at the same pace. A pass counts once it has stuck (``pass_hold_m``). Prints
passes won / contacts / off-track and the guard's own counts: moves made, sections
defended, and squeeze steps (moving toward a car alongside; must be 0).
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

from f1rl.agents.tactical import load_tactical_config
from f1rl.config import RaceEnvParams
from f1rl.envs.racetrack import TUM_TRAIN, TUM_VALIDATION
from f1rl.scenarios import make_env, pick

PAIRS = {
    "slow": ("slow", "guard_slow"),
    "mid": ("equal", "guard_mid"),  # "equal" stands in for an unguarded car
    "equal": ("equal", "guard_equal"),
    "old_defender": ("defender", "guard_equal"),
}


def one(job: tuple[str, RaceEnvParams, str, str, float, float]) -> dict[str, Any]:
    track, params, kind, policy, s0, gap = job
    env = make_env(track, params, "random")
    obs, info = env.reset(
        seed=0,
        options={"start_s": s0, "opponent": kind, "gap_m": gap, "push": 1.0, "wear": 0},
    )
    for _ in range(2000):
        obs, _, term, trunc, info = env.step(pick(env, obs, policy))
        if term or trunc:
            break
    opp = env._opp
    near = None
    if opp is not None:
        gap = (opp.state.s_m - env._state.s_m + env._lap_m / 2) % env._lap_m
        near = (gap - env._lap_m / 2, opp.state.n_m - env._state.n_m)
    stats = dict(opp.guard.stats) if opp is not None and opp.guard else {}
    return {
        "track": track,
        "kind": kind,
        "policy": policy,
        "s0": s0,
        "won": bool(
            info["passes"] >= 1 and not info["contact"] and not info["off_track"]
        ),
        "contact": bool(info["contact"]),
        "off": bool(info["off_track"]),
        "swaps": int(info["swaps"]),
        "filter": int(info["interventions"]),
        # Where the other car was when the battle ended (gap ahead, sideways): an
        # off-track with the guard right beside says the guard put it there.
        "end_gap_m": near[0] if near else None,
        "end_lateral_m": near[1] if near else None,
        "elapsed_s": float(info["elapsed_s"]),
        **stats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tracks", default="validation")
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--gap", type=float, default=20.0)
    ap.add_argument("--policies", default="tactician,line")
    ap.add_argument("--pairs", default=",".join(PAIRS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    params = replace(load_tactical_config(args.config).env_params(), safety_filter=True)
    names = {"validation": TUM_VALIDATION, "train": TUM_TRAIN[:6]}.get(
        args.tracks, tuple(args.tracks.split(","))
    )
    kinds = sorted({k for p in args.pairs.split(",") for k in PAIRS[p]})
    jobs = []
    for name in names:
        lap = make_env(name, params, "random")._lap_m
        for k in range(args.starts):
            for kind in kinds:
                for pol in args.policies.split(","):
                    jobs.append(
                        (name, params, kind, pol, lap * k / args.starts, args.gap)
                    )
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(one, jobs))
    print(
        f"{'car ahead':14} {'attacker':10} {'passed':>7} {'contact':>7} {'off':>4} "
        f"{'moves':>6} {'squeeze':>7}"
    )
    for pol in args.policies.split(","):
        for kind in kinds:
            sel = [r for r in rows if r["kind"] == kind and r["policy"] == pol]
            print(
                f"{kind:14} {pol:10} {sum(r['won'] for r in sel):>3}/{len(sel):<3} "
                f"{sum(r['contact'] for r in sel):>7} {sum(r['off'] for r in sel):>4} "
                f"{sum(r.get('moves', 0) for r in sel):>6} "
                f"{sum(r.get('squeeze_steps', 0) for r in sel):>7}"
            )
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
