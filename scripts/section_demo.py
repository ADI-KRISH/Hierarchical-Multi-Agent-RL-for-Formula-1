"""Section demo: replay the driver's section battles as one self-contained HTML page.

``uv run python scripts/section_demo.py --config configs/overtake_lanes.yaml
--policy runs/overtake_lanes/model_best.zip --tracks held_out --out runs/demo.html``

Drives every section battle (`f1rl.sections`) of each track against each opponent,
exactly as `scripts/section_gate.py` scores them (same seed, same strict rules),
records both cars ten times a second, and writes a page (no network) with a
track map and an animated replay of each battle. Read-only: it never trains.
"""

import argparse
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from f1rl.agents.tactical import (
    defend_options,
    load_tactical_config,
    section_won,
    use_attacker,
    use_defender,
)
from f1rl.config import OPPONENT_KINDS
from f1rl.envs.race_env import RaceEnv
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRAIN, TUM_VALIDATION, RaceTrack
from f1rl.scenarios import make_env, pick
from f1rl.sections import env_sections, section_options

TEMPLATE = Path(__file__).with_name("section_demo_template.html")


def to_xy(track: RaceTrack, s_m: float, n_m: float) -> tuple[float, float, float]:
    """World (x, y, heading) of a point ``n_m`` left of the centreline at ``s_m``."""
    i = (s_m % track.length_m) / track.ds_m
    k0 = int(i) % track.n
    k1 = (k0 + 1) % track.n
    t = i - int(i)
    x = track.x[k0] + t * (track.x[k1] - track.x[k0])
    y = track.y[k0] + t * (track.y[k1] - track.y[k0])
    h0, h1 = track.heading[k0], track.heading[k1]
    h = h0 + t * math.remainder(h1 - h0, 2 * math.pi)
    return float(x - n_m * math.sin(h)), float(y + n_m * math.cos(h)), float(h)


def outline(track: RaceTrack, step: int = 2) -> dict[str, list[float]]:
    """Centreline and both edges, every ``step`` stations, rounded for size."""
    idx = np.arange(0, track.n, step)
    out: dict[str, list[float]] = {}
    for key, n in (
        ("c", np.zeros(len(idx))),
        ("l", track.w_left[idx]),
        ("r", -track.w_right[idx]),
    ):
        h = track.heading[idx]
        xs = track.x[idx] - n * np.sin(h)
        ys = track.y[idx] + n * np.cos(h)
        out[key] = [round(float(v), 1) for v in np.ravel(np.column_stack([xs, ys]))]
    return out


def battle(
    env: Any, policy: Any, sec: Any, opp: str, gap: float, wear: float = 0.0
) -> dict[str, Any]:
    """One section battle, recorded every env step (`action_repeat` physics steps)."""
    frames: list[list[float]] = []
    track = env.track

    def snap(move: str) -> None:
        me, op = env._state, env._opp.state
        mx, my, mh = to_xy(track, me.s_m, me.n_m)
        ox, oy, oh = to_xy(track, op.s_m, op.n_m)
        gap_m = env._opp.state.s_m - me.s_m
        frames.append(
            [
                round(mx, 2),
                round(my, 2),
                round(mh + me.psi_rad, 3),
                round(me.speed_ms * 3.6),
                round(ox, 2),
                round(oy, 2),
                round(oh + op.psi_rad, 3),
                round(op.speed_ms * 3.6),
                round(gap_m, 1),
                env._dirty_steps,
                moves.index(move),
            ]
        )

    moves = list(env.moves)
    orig = RaceEnv.step
    current = ["line"]

    def step(self: Any, a: Any) -> Any:
        out = orig(self, a)
        if self is env:
            snap(current[0])
        return out

    behind = OPPONENT_KINDS[opp][1] == "behind"
    if behind:
        opts = defend_options(sec, opp, gap, wear)
    else:
        opts = section_options(sec, opp, gap)
    obs, info = env.reset(seed=0, options=opts)
    snap("line")
    RaceEnv.step = step  # type: ignore[method-assign]
    try:
        for _ in range(2000):
            a = pick(env, obs, policy)
            current[0] = moves[a]
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
    finally:
        RaceEnv.step = orig  # type: ignore[method-assign]
    end_s = sec.start_s + sec.length_m
    ex, ey, _ = to_xy(track, end_s, 0.0)
    if info["contact"]:
        why = "contact"
    elif info["off_track"]:
        why = "off track"
    elif info["dirty_steps"]:
        why = "too close"
    elif behind:
        why = "held" if section_won(info) else "passed"
    elif section_won(info):
        why = "clean pass"
    elif info["section_won"]:
        why = "pass did not stick"
    else:
        why = "still behind"
    return {
        "track": track.name,
        "name": sec.name,
        "kind": sec.kind,
        "opponent": opp,
        "defend": behind,
        "won": section_won(info),
        "why": why,
        "end": [round(ex, 1), round(ey, 1)],
        "frames": frames,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policy", default="tactician")
    ap.add_argument("--tracks", default="held_out")
    ap.add_argument("--opponents", default="slow,guard_slow")
    ap.add_argument("--gap", type=float, default=9.0)
    ap.add_argument(
        "--defender", default="", help="trained defender model for rival_ahead* cars"
    )
    ap.add_argument(
        "--attacker", default="", help="main-driver model for rival_behind cars"
    )
    ap.add_argument("--wear", type=float, default=0.0, help="our wear, defending")
    ap.add_argument("--title", default="Corner-by-corner overtaking")
    ap.add_argument("--out", type=Path, default=Path("runs/section_demo.html"))
    ap.add_argument("--set", nargs="*", default=[], help="env overrides, k=v")
    args = ap.parse_args()
    config = load_tactical_config(args.config)
    extra = {k: yaml.safe_load(v) for k, v in (kv.split("=", 1) for kv in args.set)}
    params = replace(config.env_params(), safety_filter=True, **extra)
    sets = {"validation": TUM_VALIDATION, "train": TUM_TRAIN, "held_out": TUM_HELD_OUT}
    names = sets.get(args.tracks, tuple(args.tracks.split(",")))
    policy: Any = args.policy
    if policy not in ("line", "tactician"):
        from stable_baselines3 import PPO

        policy = PPO.load(policy, device="cpu")
    tracks: dict[str, Any] = {}
    battles: list[dict[str, Any]] = []
    moves: list[str] = []
    for name in names:
        env = make_env(name, params, "random")
        if args.defender:
            use_defender([env], args.defender, ("line", "cover", "inside"))
        if args.attacker:
            use_attacker([env], args.attacker)
        moves = list(env.moves)
        tracks[name] = outline(env.track)
        for sec in env_sections(env, config.section_params()):
            for opp in args.opponents.split(","):
                behind = OPPONENT_KINDS[opp][1] == "behind"
                if behind and sec.kind not in config.run.section_defend_kinds:
                    continue
                battles.append(battle(env, policy, sec, opp, args.gap, args.wear))
        done = [b for b in battles if b["track"] == name]
        print(f"{name:14} {sum(b['won'] for b in done)}/{len(done)}", flush=True)
    data = {
        "title": args.title,
        "policy": str(args.policy),
        "config": str(args.config),
        "moves": moves,
        "dt": params.action_repeat * 0.02,
        "tracks": tracks,
        "battles": battles,
    }
    page = TEMPLATE.read_text(encoding="utf-8")
    page = page.replace("/*DATA*/null", json.dumps(data, separators=(",", ":")))
    args.out.write_text(page, encoding="utf-8")
    won = sum(b["won"] for b in battles)
    print(f"{won}/{len(battles)} clean passes -> {args.out}")


if __name__ == "__main__":
    main()
