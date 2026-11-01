import json
from pathlib import Path
from typing import Any

import pytest

from f1rl.agents.baseline import evaluate_baseline
from f1rl.config import EXAMPLE_CAR
from f1rl.envs.track import SYNTHETIC_TRACKS
from f1rl.report import load_views, render_body, render_html, track_view


@pytest.fixture(scope="module")
def results() -> dict[str, Any]:
    track = SYNTHETIC_TRACKS["oval"]()
    return evaluate_baseline("oval", track, EXAMPLE_CAR, episodes=3, seed=0)


def test_delta_trace_ends_at_the_best_laps_gap(results: dict[str, Any]) -> None:
    view = track_view(results)
    best = min(e["lap_time_s"] for e in results["episodes"] if e["completed"])
    gap = best - results["summary"]["limit_lap_time_s"]
    assert view["trace"]["delta_s"][0] == pytest.approx(0.0, abs=1e-3)
    assert view["trace"]["delta_s"][-1] == pytest.approx(gap, abs=2e-3)


def test_section_means_add_up_to_the_mean_gap(results: dict[str, Any]) -> None:
    view = track_view(results)
    total = sum(s["mean_delta_s"] for s in view["sections"])
    summary = results["summary"]
    assert total == pytest.approx(summary["mean_s"] - summary["limit_lap_time_s"])


def test_rendered_page_inlines_the_data(results: dict[str, Any]) -> None:
    body = render_body([track_view(results)])
    assert "/*__DATA__*/null" not in body
    assert "<title>Driver Lap Report</title>" in body
    assert render_html([track_view(results)]).startswith("<!doctype html>")


def test_load_views_reads_logged_runs(tmp_path: Path, results: dict[str, Any]) -> None:
    (tmp_path / "oval").mkdir()
    (tmp_path / "oval" / "results.json").write_text(json.dumps(results))
    views = load_views(tmp_path)
    assert [v["name"] for v in views] == ["oval"]
