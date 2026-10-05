"""Baseline report: ``uv run python -m f1rl.report``.

Read-only over ``runs/baseline/*/results.json`` (written by
``f1rl.agents.baseline``): condenses each track's results into the numbers the
page plots, and writes one self-contained HTML file -- data inlined, no network
needed to open it. It never runs a driver or trains anything.
"""

import argparse
import json
from importlib import resources
from pathlib import Path
from typing import Any

import numpy as np

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


def load_views(runs_dir: Path) -> list[dict[str, Any]]:
    paths = sorted(runs_dir.glob("*/results.json"))
    return [track_view(json.loads(p.read_text())) for p in paths]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the HTML report over logged baseline runs (read-only)."
    )
    parser.add_argument("--runs", type=Path, default=Path("runs/baseline"))
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--body-only",
        action="store_true",
        help="Write the page content without the <html>/<head> wrapper.",
    )
    args = parser.parse_args()

    views = load_views(args.runs)
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
