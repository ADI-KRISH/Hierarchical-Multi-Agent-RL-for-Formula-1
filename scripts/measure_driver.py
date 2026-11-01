"""Measure our Driver Agent for the Strategy Agent (`f1rl.driver_profile`).

``uv run python scripts/measure_driver.py --config configs/defend_sections_v2.yaml
--policy runs/defend_sections_v2/model_best.zip --out configs/driver_profiles/x.json``

Per track, in the full sim with the safety filter on:

1. **Pace.** Flying laps alone at push 0 / 0.5 / 1 and tyre wear 0 / 0.35 / 0.7.
   Each lap is stored as its gap to the g-g limit lap on the optimal line at the
   same grip (planned share of grip x tyre grip). Push uses the strategy's span
   (`StrategyParams.conserve_grip_fraction` .. attack), so push means the same in
   both sims.
2. **Attack.** One-lap battles (``--lap-frac`` of a lap, then the pass must hold
   `pass_hold_m`), our car ``--gap`` m behind a defending car at the start of a
   straight, at each wear, against ``--attack-opponents``.
3. **Defend.** The same with our car ahead and ``--defend-opponents`` behind
   (``rival_behind`` = a frozen main driver, ``--attacker``).

Each battle's pace advantage is the lap-time difference of the two cars on this
track: ours from step 1 at the battle's wear, the other car's measured the same way
(a flying lap alone in clean air, `opponent_lap_s`). Earlier profiles
(``--opp-pace grip``) put the other car's planned grip share into our own pace
curve instead, which made the scripted cars look ~2-3x slower than they are (0.80
grip: ~2 s/lap, not ~5). Logistic fits of the outcomes give
the attack and defend odds the race sim uses. Tracks run in parallel processes.
"""

import argparse
import json
import multiprocessing as mp
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from f1rl.config import EXAMPLE_CAR, OPPONENT_KINDS, StrategyParams, TireParams

WEARS = (0.0, 0.35, 0.7)
PUSHES = (0.0, 0.5, 1.0)


def _setup(args: argparse.Namespace, name: str) -> tuple[Any, Any, Any, Any, Any]:
    import torch

    from f1rl.agents.tactical import load_tactical_config, use_attacker
    from f1rl.scenarios import make_env

    torch.set_num_threads(1)
    config = load_tactical_config(args.config)
    sp = StrategyParams()
    params = replace(
        config.env_params(),
        safety_filter=True,
        conserve_grip_fraction=sp.conserve_grip_fraction,
        episode_s=args.episode_s,
    )
    lap_env = make_env(name, params, "lap")
    battle_env = make_env(name, params, "random")
    if args.attacker:
        use_attacker([battle_env], args.attacker)
    policy: Any = args.policy
    if policy not in ("line", "tactician"):
        from stable_baselines3 import PPO

        policy = PPO.load(policy, device="cpu")
    return config, params, lap_env, battle_env, policy


def _run(env: Any, policy: Any, options: dict[str, Any]) -> dict[str, Any]:
    from f1rl.scenarios import pick

    obs, info = env.reset(seed=0, options=options)
    for _ in range(20_000):
        obs, _, term, trunc, info = env.step(pick(env, obs, policy))
        if term or trunc:
            break
    return dict(info)


def opponent_lap_s(env: Any, kind: str) -> float | None:
    """The scripted car ``kind``'s flying lap (s) alone on this track: it starts half
    a lap from our car (far out of the wake and the tow), and its time from 300 m
    to a lap past that is interpolated between decisions."""
    lap_m = float(env._lap_m)
    env.reset(
        seed=0,
        options={
            "start_s": 0.0,
            "opponent": kind,
            "gap_m": lap_m / 2,
            "section_m": 3 * lap_m,
            "push": 1.0,
            "wear": 0.0,
            "start_speed_frac": 1.0,
        },
    )
    prev, done, t_prev = env._opp.state.s_m, 0.0, 0.0
    marks: list[float] = []
    for _ in range(20_000):
        _, _, term, trunc, info = env.step(0)
        t = float(info["elapsed_s"])
        step = (env._opp.state.s_m - prev) % lap_m
        prev = env._opp.state.s_m
        for mark in (300.0, lap_m + 300.0)[len(marks) :]:
            if step > 0 and done < mark <= done + step:
                marks.append(t_prev + (t - t_prev) * (mark - done) / step)
        done, t_prev = done + step, t
        if len(marks) == 2:
            return marks[1] - marks[0]
        if term or trunc:
            break
    return None


def measure_track(job: tuple[argparse.Namespace, str]) -> dict[str, Any]:
    args, name = job
    try:
        return _measure(args, name)
    except Exception as exc:  # one broken track must not sink the others
        return {"track": name, "error": repr(exc)}


def _measure(args: argparse.Namespace, name: str) -> dict[str, Any]:
    from f1rl.agents.tactical import section_won
    from f1rl.models.race import grip_car
    from f1rl.models.raceline import gg_lap_time_s
    from f1rl.models.tire import tire_grip
    from f1rl.sections import env_sections

    config, _params, lap_env, battle_env, policy = _setup(args, name)
    sp = StrategyParams()
    tire = TireParams()
    line = lap_env.reference
    top = sp.attack_grip_fraction

    def limit(grip: float) -> float:
        return gg_lap_time_s(grip_car(EXAMPLE_CAR, grip), line.ds, line.kappa)

    # 1. Pace.
    pace = []
    for push in PUSHES:
        for wear in WEARS:
            info = _run(lap_env, policy, {"push": push, "wear": wear})
            planned = (
                sp.conserve_grip_fraction + (top - sp.conserve_grip_fraction) * push
            )
            grip = planned * tire_grip(tire, wear)
            lap = info.get("lap_time_s")
            pace.append(
                {
                    "push": push,
                    "wear": wear,
                    "grip": grip,
                    "lap_s": lap,
                    "gap": None if lap is None else lap / limit(grip) - 1.0,
                }
            )
    ok = sorted((r["grip"], r["gap"]) for r in pace if r["gap"] is not None)
    if not ok:
        return {"track": name, "error": "no flying lap finished", "pace": pace}
    grips = np.array([g for g, _ in ok])
    gaps = np.array([v for _, v in ok])

    def lap_time(grip: float) -> float:
        return limit(grip) * (1.0 + float(np.interp(grip, grips, gaps)))

    # 2-3. One-lap battles from the starts of straights.
    straights = [s for s in env_sections(battle_env, config.section_params())]
    straights = [s for s in straights if s.kind == "straight"]
    pick_n = min(args.starts, len(straights))
    starts = [straights[i * len(straights) // pick_n] for i in range(pick_n)]
    lap_m = float(np.sum(line.ds))
    opp_laps: dict[str, float | None] = {}
    if args.opp_pace == "measured":
        for kind in (*args.attack_opponents, *args.defend_opponents):
            opp_laps[kind] = opponent_lap_s(battle_env, kind)
    battles = []
    for role, kinds in (
        ("attack", args.attack_opponents),
        ("defend", args.defend_opponents),
    ):
        for kind in kinds:
            opp_grip = OPPONENT_KINDS[kind][0]
            for wear in WEARS:
                my_grip = top * tire_grip(tire, wear)
                for sec in starts:
                    info = _run(
                        battle_env,
                        policy,
                        {
                            "start_s": sec.start_s,
                            "start_shift_m": 0.0,
                            "opponent": kind,
                            "gap_m": args.gap,
                            "section_m": args.lap_frac * lap_m,
                            "push": 1.0,
                            "wear": wear,
                            "start_speed_frac": 1.0,
                        },
                    )
                    me = lap_time(my_grip)
                    measured = opp_laps.get(kind)
                    them = lap_time(opp_grip) if measured is None else measured
                    ahead = bool(info["section_won"])
                    if role == "attack":
                        advantage = them - me
                        passed = ahead and int(info["passes"]) >= 1
                    else:
                        advantage = me - them  # the attacker's advantage
                        passed = (not ahead) or int(info["passes"]) <= -1
                    passed = passed or bool(info["off_track"])
                    battles.append(
                        {
                            "role": role,
                            "opponent": kind,
                            "wear": wear,
                            "start_s": sec.start_s,
                            "advantage_s": advantage,
                            "passed": bool(passed),
                            "clean_win": section_won(info),
                            "contact": bool(info["contact"]),
                            "off": bool(info["off_track"]),
                        }
                    )
    return {
        "track": name,
        "pace": pace,
        "grips": grips.tolist(),
        "gaps": gaps.tolist(),
        "battles": battles,
        "opponent_laps": opp_laps,
        "my_lap_new": lap_time(top * tire_grip(tire, 0.0)),
    }


def main() -> None:
    from f1rl.driver_profile import (
        DriverProfile,
        PassCurve,
        fit_pass_curve,
        save_profile,
        smooth_pace,
    )
    from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRACKS, TUM_TRAIN, TUM_VALIDATION

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--policy", required=True, help="model .zip, line or tactician")
    ap.add_argument("--attacker", default="", help="main-driver model for rival_behind")
    ap.add_argument("--tracks", default="all")
    ap.add_argument("--starts", type=int, default=3, help="battle starts per track")
    ap.add_argument("--gap", type=float, default=9.0)
    ap.add_argument("--lap-frac", type=float, default=0.9)
    ap.add_argument("--episode-s", type=float, default=200.0)
    ap.add_argument("--attack-opponents", default="guard_slow,guard_mid,guard_equal")
    ap.add_argument(
        "--defend-opponents",
        default="rival_tactician,rival_tactician_fast,attacker,rival_behind",
    )
    ap.add_argument(
        "--opp-pace",
        choices=("measured", "grip"),
        default="measured",
        help="the other car's lap: measured alone (default) or our curve at its grip",
    )
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--from-raw",
        type=Path,
        default=None,
        help="refit from an earlier run's *_raw.json instead of driving",
    )
    args = ap.parse_args()
    args.attack_opponents = [k for k in args.attack_opponents.split(",") if k]
    args.defend_opponents = [k for k in args.defend_opponents.split(",") if k]
    if "rival_behind" in args.defend_opponents and not args.attacker:
        raise SystemExit("rival_behind needs --attacker")
    sets = {
        "all": TUM_TRACKS,
        "train": TUM_TRAIN,
        "validation": TUM_VALIDATION,
        "held_out": TUM_HELD_OUT,
    }
    names = sets.get(args.tracks, tuple(args.tracks.split(",")))
    jobs = [(args, n) for n in names]
    results: list[dict[str, Any]] = []
    if args.from_raw:
        done = json.loads(args.from_raw.read_text(encoding="utf-8"))
        jobs = []
    with mp.get_context("spawn").Pool(max(1, min(args.procs, len(jobs)))) as pool:
        stream = done if args.from_raw else pool.imap_unordered(measure_track, jobs)
        for r in stream:
            results.append(r)
            if "error" in r:
                print(f"{r['track']:14} FAILED {r['error']}", flush=True)
                continue
            b = r["battles"]
            att = [x for x in b if x["role"] == "attack"]
            dfd = [x for x in b if x["role"] == "defend"]
            print(
                f"{r['track']:14} gap {np.mean(r['gaps']):+.2%}  "
                f"passes {sum(x['passed'] for x in att)}/{len(att)}  "
                f"held {sum(not x['passed'] for x in dfd)}/{len(dfd)}  "
                f"contacts {sum(x['contact'] for x in b)}",
                flush=True,
            )
    good = [r for r in results if "error" not in r]
    rows = [x for r in good for x in r["battles"]]
    sp = StrategyParams()
    prior = PassCurve(sp.pass_delta_s, sp.pass_slope_s)
    curves = {}
    for role in ("attack", "defend"):
        sel = [x for x in rows if x["role"] == role]
        curves[role] = fit_pass_curve(
            [x["advantage_s"] for x in sel], [x["passed"] for x in sel], prior
        )
    profile = DriverProfile(
        model=str(args.policy),
        pace={
            r["track"]: (tuple(r["grips"]), tuple(r["gaps"]))
            for r in sorted(good, key=lambda r: r["track"])
        },
        attack=curves["attack"],
        defend=curves["defend"],
        contact_rate=float(np.mean([x["contact"] for x in rows])) if rows else 0.0,
        off_rate=float(np.mean([x["off"] for x in rows])) if rows else 0.0,
        meta={
            "config": str(args.config),
            "attacker": args.attacker,
            "starts": args.starts,
            "gap_m": args.gap,
            "lap_frac": args.lap_frac,
            "opp_pace": args.opp_pace,
            "attack_opponents": args.attack_opponents,
            "defend_opponents": args.defend_opponents,
            "failed_tracks": [r["track"] for r in results if "error" in r],
        },
    )
    raw_pace = profile.pace
    profile = smooth_pace(profile)
    profile.meta["pace_measured"] = {t: list(map(list, v)) for t, v in raw_pace.items()}
    save_profile(profile, args.out)
    raw = args.out.with_name(args.out.stem + "_raw.json")
    if not args.from_raw:
        raw.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(
        f"\nmean pace gap {profile.mean_gap():+.2%} over {len(good)} tracks\n"
        f"attack: 50% pass at {curves['attack'].delta_s:+.2f} s/lap advantage "
        f"(slope {curves['attack'].slope_s:.2f}, {curves['attack'].battles} battles)\n"
        f"defend: passed 50% at {curves['defend'].delta_s:+.2f} s/lap attacker "
        f"advantage (slope {curves['defend'].slope_s:.2f}, "
        f"{curves['defend'].battles} battles)\n"
        f"contacts {profile.contact_rate:.1%}, off track {profile.off_rate:.1%}\n"
        f"wrote {args.out} and {raw}"
    )


if __name__ == "__main__":
    main()
