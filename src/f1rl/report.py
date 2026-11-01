"""Driver report: ``uv run python -m f1rl.report``.

Read-only over logged runs: ``runs/baseline/*/results.json`` (written by
``f1rl.agents.baseline``) and any PPO training run under ``runs/`` (its
``eval.csv``, ``episodes.csv``, ``best_lap.json``, written by ``f1rl.train``).
Condenses them into the numbers the page plots and writes one self-contained
HTML file -- data inlined, no network needed to open it. It never runs a driver,
loads a model, or trains anything.
"""

import argparse
import csv
import itertools
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np
import yaml

TRACE_STEP_M = 5.0  # resolution of the speed / time-delta traces on the page


def _best_episode(results: dict[str, Any]) -> dict[str, Any] | None:
    completed = [e for e in results["episodes"] if e["completed"]]
    return min(completed, key=lambda e: e["lap_time_s"]) if completed else None


def track_view(results: dict[str, Any]) -> dict[str, Any]:
    """Everything the page shows for one track, from its results dict."""
    length_m = results["track_length_m"]
    limit = results["limit_profile"]
    grid = np.linspace(0.0, length_m, int(length_m // TRACE_STEP_M) + 1)
    limit_speed = np.interp(grid, limit["distance_m"], limit["speed_ms"])
    limit_time = np.interp(grid, limit["distance_m"], limit["time_s"])

    best = _best_episode(results)
    trace: dict[str, list[float]] = {
        "distance_m": _rounded(grid, 1),
        "limit_speed_kmh": _rounded(limit_speed * 3.6, 1),
    }
    map_speed: list[float] = []
    if best is not None:
        t = best["telemetry"]
        # Close each trace at the finish line.
        dist = [*t["distance_m"], length_m]
        time = [*t["time_s"], best["lap_time_s"]]
        speed = [*t["speed_ms"], t["speed_ms"][-1]]
        throttle = [*t["throttle"], t["throttle"][-1]]
        trace["speed_kmh"] = _rounded(np.interp(grid, dist, speed) * 3.6, 1)
        trace["delta_s"] = _rounded(np.interp(grid, dist, time) - limit_time, 3)
        trace["throttle"] = _rounded(np.interp(grid, dist, throttle), 3)
        map_speed = _rounded(
            np.interp(results["map"]["distance_m"], dist, speed) * 3.6, 1
        )

    completed = [e for e in results["episodes"] if e.get("sections")]
    sections = []
    for i, section in enumerate(results["sections"]):
        per_lap = [e["sections"][i] for e in completed]
        sections.append(
            {
                "name": section["name"],
                "is_corner": section["is_corner"],
                "start_m": round(section["start_m"], 1),
                "end_m": round(section["end_m"], 1),
                "min_radius_m": section["min_radius_m"],
                "mean_delta_s": _mean([s["delta_s"] for s in per_lap]),
                "mean_time_s": _mean([s["time_s"] for s in per_lap]),
                "reference_time_s": per_lap[0]["reference_time_s"] if per_lap else None,
                "mean_min_speed_kmh": _mean([s["min_speed_ms"] * 3.6 for s in per_lap]),
                "reference_min_speed_kmh": (
                    per_lap[0]["reference_min_speed_ms"] * 3.6 if per_lap else None
                ),
            }
        )

    limit_lap_s = results["summary"]["limit_lap_time_s"]
    laps = [
        {
            "seed": e["seed"],
            "completed": e["completed"],
            "off_track": e["off_track"],
            "off_track_at_m": e["off_track_at_m"],
            "lap_time_s": e["lap_time_s"],
            "gap_pct": (
                100 * (e["lap_time_s"] / limit_lap_s - 1) if e["lap_time_s"] else None
            ),
            "corner_grip_margin": e["corner_grip_margin"],
            "braking_margin": e["braking_margin"],
            "top_speed_kmh": e["top_speed_ms"] * 3.6,
            "total_reward": e["total_reward"],
        }
        for e in results["episodes"]
    ]

    return {
        "name": results["track"],
        "length_m": length_m,
        "git_sha": results["git_sha"],
        "config": results["config"],
        "summary": results["summary"],
        "best_seed": best["seed"] if best else None,
        "map": {
            "x_m": results["map"]["x_m"],
            "y_m": results["map"]["y_m"],
            "distance_m": results["map"]["distance_m"],
            "speed_kmh": map_speed,
        },
        "sections": sections,
        "trace": trace,
        "laps": laps,
    }


TRAIN_CURVE_BINS = 60  # training-episode stats are averaged into this many bins


def training_view(run_dir: Path, track: str, length_m: float) -> dict[str, Any]:
    """Learning curves, replay laps and the best lap of one PPO run on one of
    its tracks, from its log files (single- and multi-track runs alike)."""
    config = yaml.safe_load((run_dir / "config.yaml").read_text())
    run_tracks = _tracks_of(config)
    with (run_dir / "eval.csv").open() as f:
        evals = [e for e in csv.DictReader(f) if e.get("track", run_tracks[0]) == track]
    with (run_dir / "episodes.csv").open() as f:
        episodes = [
            e for e in csv.DictReader(f) if e.get("track", run_tracks[0]) == track
        ]

    def num(value: str | None) -> float | None:
        return float(value) if value not in ("", None) else None

    curve: dict[str, list[float | None]] = {
        "timesteps": [],
        "mean_return": [],
        "completion": [],
        "mean_lap_s": [],
    }
    if episodes:
        last = int(episodes[-1]["timesteps"])
        edges = np.linspace(0, last, TRAIN_CURVE_BINS + 1)
        for lo, hi in itertools.pairwise(edges):
            chunk = [e for e in episodes if lo < int(e["timesteps"]) <= hi]
            if not chunk:
                continue
            laps = [float(e["lap_time_s"]) for e in chunk if e["completed"] == "1"]
            curve["timesteps"].append(round(float(hi)))
            curve["mean_return"].append(
                round(float(np.mean([float(e["total_reward"]) for e in chunk])), 4)
            )
            curve["completion"].append(round(len(laps) / len(chunk), 3))
            curve["mean_lap_s"].append(round(float(np.mean(laps)), 3) if laps else None)

    best_lap = None
    best_path = run_dir / "best_lap.json"
    if best_path.exists():
        record = json.loads(best_path.read_text())
        lap = record["laps"].get(track) if "laps" in record else record
        if lap is not None:
            best_lap = _best_lap_view(lap, length_m)

    return {
        "name": run_dir.name,
        "tracks": run_tracks,
        "config": config,
        "meta": json.loads((run_dir / "meta.json").read_text()),
        "eval": {
            "timesteps": [int(e["timesteps"]) for e in evals],
            "completion_rate": [float(e["completion_rate"]) for e in evals],
            "mean_lap_s": [num(e["mean_lap_s"]) for e in evals],
            "best_lap_s": [num(e["best_lap_s"]) for e in evals],
            "mean_reward": [float(e["mean_reward"]) for e in evals],
        },
        "train": curve,
        "best_lap": best_lap,
        "replays": _replays(run_dir, track),
    }


def _best_lap_view(lap: dict[str, Any], length_m: float) -> dict[str, Any]:
    t = lap.get("telemetry")
    view: dict[str, Any] = {
        "completed": lap["completed"],
        "lap_time_s": lap["lap_time_s"],
        "total_reward": lap["total_reward"],
    }
    if t and lap["completed"]:
        grid = np.linspace(0.0, length_m, int(length_m // TRACE_STEP_M) + 1)
        dist = [*t["distance_m"], length_m]
        speed = [*t["speed_ms"], t["speed_ms"][-1]]
        throttle = [*t["throttle"], t["throttle"][-1]]
        view["speed_kmh"] = _rounded(np.interp(grid, dist, speed) * 3.6, 1)
        view["throttle"] = _rounded(np.interp(grid, dist, throttle), 3)
    return view


def _replays(run_dir: Path, track: str) -> list[dict[str, Any]]:
    """The lap driven on `track` at each eval checkpoint, for the replay view."""
    path = run_dir / "eval_laps.jsonl"
    if not path.exists():
        return []
    replays = []
    for line in path.read_text().splitlines():
        lap = json.loads(line)
        if lap.get("track") != track:
            continue
        t = lap.get("telemetry") or {}
        replays.append(
            {
                "timesteps": lap["timesteps"],
                "completed": lap["completed"],
                "off_track": lap["off_track"],
                "stalled": lap.get("stalled", False),
                "lap_time_s": lap["lap_time_s"],
                "distance_m": round(lap["distance_m"], 1),
                "time_s": [round(v, 2) for v in t.get("time_s", [])],
                "pos_m": [round(v, 1) for v in t.get("distance_m", [])],
                "speed_kmh": [round(v * 3.6) for v in t.get("speed_ms", [])],
            }
        )
    return replays


def _tracks_of(config: dict[str, Any]) -> list[str]:
    track = config["run"]["track"]
    return [track] if isinstance(track, str) else list(track)


def _rounded(values: Any, digits: int) -> list[float]:
    return [round(float(v), digits) for v in values]


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def render_body(views: list[dict[str, Any]]) -> str:
    """The report's page content (no <html>/<head> wrapper), data inlined."""
    template = resources.files("f1rl").joinpath("templates/baseline_report.html")
    data = json.dumps(views, separators=(",", ":")).replace("</", "<\\/")
    return template.read_text(encoding="utf-8").replace("/*__DATA__*/null", data)


def render_html(views: list[dict[str, Any]]) -> str:
    """A complete standalone HTML document for opening locally."""
    return (
        '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        "</head>\n<body>\n" + render_body(views) + "\n</body>\n</html>\n"
    )


def load_views(
    runs_dir: Path, training_runs: list[Path] | None = None
) -> list[dict[str, Any]]:
    """One view per baseline track, each carrying the training runs (if any)
    that trained on that track."""
    paths = sorted(runs_dir.glob("*/results.json"))
    views = [track_view(json.loads(p.read_text())) for p in paths]
    for view in views:
        view["training"] = [
            training_view(run, view["name"], view["length_m"])
            for run in training_runs or []
            if view["name"] in _run_tracks(run)
        ]
    return views


def find_training_runs(runs_root: Path) -> list[Path]:
    """Run folders a PPO training wrote (they hold eval.csv and config.yaml)."""
    return sorted(
        p.parent
        for p in runs_root.glob("*/eval.csv")
        if (p.parent / "config.yaml").exists() and (p.parent / "meta.json").exists()
    )


def _run_tracks(run_dir: Path) -> list[str]:
    return _tracks_of(yaml.safe_load((run_dir / "config.yaml").read_text()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the HTML report over logged runs (read-only)."
    )
    parser.add_argument("--runs", type=Path, default=Path("runs/baseline"))
    parser.add_argument(
        "--training",
        type=Path,
        nargs="*",
        default=None,
        help="Training run folders to include (default: every run under runs/).",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="Write the page content without the <html>/<head> wrapper.",
    )
    args = parser.parse_args()

    training = (
        args.training
        if args.training is not None
        else find_training_runs(args.runs.parent)
    )
    views = load_views(args.runs, training)
    if not views:
        raise SystemExit(
            f"No results under {args.runs}/ -- run "
            "`uv run python -m f1rl.agents.baseline --track <name>` first."
        )
    out = args.out or args.runs / "report.html"
    out.write_text(render_body(views) if args.body_only else render_html(views))
    print(f"Wrote {out} ({len(views)} track(s): {', '.join(v['name'] for v in views)})")


if __name__ == "__main__":
    main()
