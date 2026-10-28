"""Side-by-side comparison of every driver on each track -> one HTML page.

Usage: uv run python scripts/compare_drivers.py
         [--tracks tum_held_out,tum_validation]
         [--agent "Pure RL=runs/racer_sac/model_best.zip"] [--agent ...]
         [--out runs/driver_comparison.html]

Drives one flying lap per track with: the optimal line (the reference, from the lap
sim, not a driver), the scripted line follower, and each saved agent. Records speed,
the throttle/brake and steering commands and the car's path every decision, splits
the track into corners and straights, and writes a self-contained page (no network)
showing where each driver gains or loses time. Read-only: it never trains.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from stable_baselines3 import SAC

from f1rl.agents.line_follower import LineFollower
from f1rl.agents.racer import load_racer_config, resolve_track_names
from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.race_env import RaceEnv
from f1rl.envs.racetrack import RaceTrack, load_tum_track
from f1rl.models.raceline import OptimalLine, optimal_line

CORNER_KAPPA = 0.003  # |curvature| above this (radius < 333 m) is a corner ...
MIN_CORNER_M = 25.0  # ... if it lasts this long,
MERGE_GAP_M = 40.0  # and corners closer than this are one complex.

Action = Callable[[np.ndarray, dict[str, Any]], np.ndarray]


def find_sections(track: RaceTrack) -> list[dict[str, Any]]:
    """Corners and straights around the lap, in order, from the centreline."""
    k = np.abs(track.kappa)
    k = np.convolve(np.pad(k, 2, mode="wrap"), np.ones(5) / 5, mode="valid")
    flags = k > CORNER_KAPPA
    runs: list[list[int]] = []  # [start, end) station indices of corner runs
    i = 0
    n = track.n
    while i < n:
        if flags[i]:
            j = i
            while j < n and flags[j]:
                j += 1
            runs.append([i, j])
            i = j
        else:
            i += 1
    merged: list[list[int]] = []
    gap = round(MERGE_GAP_M / track.ds_m)
    for run in runs:
        if merged and run[0] - merged[-1][1] <= gap:
            merged[-1][1] = run[1]
        else:
            merged.append(run)
    min_len = round(MIN_CORNER_M / track.ds_m)
    corners = [r for r in merged if r[1] - r[0] >= min_len]
    if not corners:
        return [{"kind": "straight", "name": "S1", "start": 0.0, "end": track.length_m}]
    sections: list[dict[str, Any]] = []
    # Start the list at the first corner so the lap reads corner, straight, corner ...
    for idx, (a, b) in enumerate(corners):
        nxt = corners[(idx + 1) % len(corners)][0]
        sections.append(
            {"kind": "corner", "start": a * track.ds_m, "end": b * track.ds_m}
        )
        end = nxt * track.ds_m + (track.length_m if nxt <= a else 0.0)
        sections.append({"kind": "straight", "start": b * track.ds_m, "end": end})
    c = s = 0
    for sec in sections:
        if sec["kind"] == "corner":
            c += 1
            sec["name"] = f"C{c}"
        else:
            s += 1
            sec["name"] = f"S{s}"
    return sections


def record_lap(env: RaceEnv, act: Action) -> dict[str, Any]:
    """One deterministic lap, recording every decision."""
    track = env.track
    obs, info = env.reset(seed=0)
    rec: dict[str, list[float]] = {
        key: [] for key in ("t", "s", "v", "n", "u", "st", "x", "y")
    }

    def log(action: np.ndarray) -> None:
        i = int(info["distance_m"] / track.ds_m) % track.n
        n = info["lateral_offset_m"]
        theta = float(track.heading[i])
        row = (
            info["elapsed_s"],
            info["distance_m"],
            info["speed_ms"],
            n,
            float(action[0]),
            float(action[1]),
            float(track.x[i]) - n * float(np.sin(theta)),
            float(track.y[i]) + n * float(np.cos(theta)),
        )
        for key, value in zip(rec, row, strict=True):
            rec[key].append(round(float(value), 3))

    done = False
    action = np.zeros(2)
    while not done:
        action = np.clip(act(obs, info), -1.0, 1.0)
        log(action)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    log(action)
    return {
        "completed": bool(info["lap_completed"]),
        "off_track": bool(info["off_track"]),
        "lap_time_s": info.get("lap_time_s"),
        "reached_m": info["distance_m"],
        **rec,
    }


def section_stats(
    lap: dict[str, Any], sec: dict[str, Any], alpha_at: Callable[[float], float]
) -> dict[str, float] | None:
    """Time, speeds, throttle use and line error inside one section; None if the
    driver never drove all of it."""
    s = np.array(lap["s"])
    if s[-1] < sec["end"] - 1.0 and not lap["completed"]:
        return None
    t, v = np.array(lap["t"]), np.array(lap["v"])
    u, n = np.array(lap["u"]), np.array(lap["n"])
    end = min(sec["end"], float(s[-1]))
    mask = (s >= sec["start"]) & (s <= end)
    if mask.sum() < 2:
        return None
    t0, t1 = np.interp([sec["start"], end], s, t)
    err = [abs(nv - alpha_at(sv)) for nv, sv in zip(n[mask], s[mask], strict=True)]
    return {
        "time_s": float(t1 - t0),
        "entry_ms": float(np.interp(sec["start"], s, v)),
        "min_ms": float(v[mask].min()),
        "exit_ms": float(np.interp(end, s, v)),
        "throttle_mean": float(np.clip(u[mask], 0, None).mean()),
        "full_throttle_pct": float(100 * (u[mask] > 0.9).mean()),
        "braking_pct": float(100 * (u[mask] < -0.1).mean()),
        "line_error_m": float(np.mean(err)),
    }


def reference_lap(track: RaceTrack, line: OptimalLine) -> dict[str, Any]:
    """The optimal line as a 'lap': its limit speed and offset at every station."""
    v = line.speed
    dt = 2 * line.ds / (v + np.roll(v, -1))
    t = np.concatenate([[0.0], np.cumsum(dt)])[: track.n]
    s = np.arange(track.n) * track.ds_m
    return {
        "completed": True,
        "off_track": False,
        "lap_time_s": float(dt.sum()),
        "reached_m": track.length_m,
        "t": np.round(t, 3).tolist(),
        "s": np.round(s, 2).tolist(),
        "v": np.round(v, 3).tolist(),
        "n": np.round(line.alpha, 3).tolist(),
        "x": np.round(line.x, 2).tolist(),
        "y": np.round(line.y, 2).tolist(),
    }


def track_payload(
    name: str, agents: dict[str, tuple[SAC, RaceEnvParams]]
) -> dict[str, Any]:
    track = load_tum_track(name)
    base = RaceEnvParams()
    line = optimal_line(track, EXAMPLE_CAR, base.car_half_width_m + base.line_safety_m)
    sim = SimParams(seed=0, max_episode_steps=10_000)

    def alpha_at(s_m: float) -> float:
        pos = (s_m % track.length_m) / track.ds_m
        i = int(pos)
        f = pos - i
        return float(
            line.alpha[i % track.n] * (1 - f) + line.alpha[(i + 1) % track.n] * f
        )

    laps: dict[str, dict[str, Any]] = {"Optimal line": reference_lap(track, line)}
    env = RaceEnv(EXAMPLE_CAR, track, sim, base, line, mode="lap")
    follower = LineFollower(EXAMPLE_CAR, track, line)
    laps["Scripted driver"] = record_lap(env, lambda _o, info: follower.act(info))
    for label, (model, params) in agents.items():
        agent_env = RaceEnv(EXAMPLE_CAR, track, sim, params, line, mode="lap")
        laps[label] = record_lap(
            agent_env, lambda obs, _i, m=model: m.predict(obs, deterministic=True)[0]
        )

    sections = find_sections(track)
    for sec in sections:
        sec["stats"] = {
            label: section_stats(lap, sec, alpha_at)
            for label, lap in laps.items()
            if label != "Optimal line"
        }
        # The reference has speed only: its time and speeds, no throttle.
        ref = laps["Optimal line"]
        s_ref, t_ref, v_ref = (np.array(ref[k]) for k in ("s", "t", "v"))
        end = sec["end"]
        s_ext = np.concatenate([s_ref, s_ref + track.length_m])
        t_ext = np.concatenate([t_ref, t_ref + ref["lap_time_s"]])
        v_ext = np.concatenate([v_ref, v_ref])
        m = (s_ext >= sec["start"]) & (s_ext <= end)
        sec["stats"]["Optimal line"] = {
            "time_s": float(
                np.interp(end, s_ext, t_ext) - np.interp(sec["start"], s_ext, t_ext)
            ),
            "entry_ms": float(np.interp(sec["start"], s_ext, v_ext)),
            "min_ms": float(v_ext[m].min()),
            "exit_ms": float(np.interp(end, s_ext, v_ext)),
        }

    normal_x = -np.sin(track.heading)
    normal_y = np.cos(track.heading)
    edge = lambda w, sign: (  # noqa: E731
        np.round(track.x + sign * w * normal_x, 1).tolist(),
        np.round(track.y + sign * w * normal_y, 1).tolist(),
    )
    return {
        "name": name,
        "length_m": track.length_m,
        "left": edge(track.w_left, 1),
        "right": edge(track.w_right, -1),
        "centre": [np.round(track.x, 1).tolist(), np.round(track.y, 1).tolist()],
        "sections": sections,
        "laps": laps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracks", default="tum_held_out,tum_validation")
    parser.add_argument("--agent", action="append", default=None)
    parser.add_argument("--out", type=Path, default=Path("runs/driver_comparison.html"))
    args = parser.parse_args()
    specs = args.agent or [
        "Pure RL=runs/racer_sac/model_best.zip",
        "Hybrid RL=runs/racer_hybrid/model_best.zip",
    ]
    agents: dict[str, tuple[SAC, RaceEnvParams]] = {}
    for spec in specs:
        label, _, path = spec.partition("=")
        model_path = Path(path)
        if not model_path.exists():
            print(f"skipping {label}: {model_path} not found")
            continue
        config = model_path.parent / "config.yaml"
        params = load_racer_config(config).env_params()
        agents[label] = (SAC.load(model_path, device="cpu"), params)

    names: list[str] = []
    for part in args.tracks.split(","):
        names += resolve_track_names(part.strip())
    from f1rl.envs.racetrack import TUM_HELD_OUT

    tracks = []
    for name in names:
        print(f"driving {name} ...", flush=True)
        payload = track_payload(name, agents)
        payload["held_out"] = name in TUM_HELD_OUT
        tracks.append(payload)

    template = (Path(__file__).parent / "compare_drivers.html").read_text(
        encoding="utf-8"
    )
    html = template.replace("__DATA__", json.dumps({"tracks": tracks}))
    args.out.write_text(html, encoding="utf-8")
    print(f"Wrote {args.out} ({len(html) / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
