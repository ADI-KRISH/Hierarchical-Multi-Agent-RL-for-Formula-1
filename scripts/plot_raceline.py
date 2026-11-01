"""Draw the optimal racing line on every TUM track into one HTML page (no plotting lib).

Usage: uv run python scripts/plot_raceline.py [out.html]

Grey = track edges, white dashed = centreline, coloured = optimal line by limit speed
(blue slow -> red fast). Lap times: centreline limit vs optimal-line limit.
"""

import sys
from pathlib import Path

import numpy as np

from f1rl.config import EXAMPLE_CAR, RaceEnvParams
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRACKS, load_tum_track
from f1rl.models.raceline import line_xy, optimal_line

SIZE = 420


def colour(t: float) -> str:
    return f"hsl({240 * (1 - t):.0f},85%,50%)"  # 240 = blue, 0 = red


def svg(name: str) -> str:
    p = RaceEnvParams()
    track = load_tum_track(name)
    line = optimal_line(track, EXAMPLE_CAR, p.car_half_width_m + p.line_safety_m)
    nx, ny = -np.sin(track.heading), np.cos(track.heading)
    lx, ly = track.x + track.w_left * nx, track.y + track.w_left * ny
    rx, ry = track.x - track.w_right * nx, track.y - track.w_right * ny
    allx = np.concatenate([lx, rx])
    ally = np.concatenate([ly, ry])
    span = max(np.ptp(allx), np.ptp(ally))
    k = (SIZE - 20) / span
    ox, oy = allx.min(), ally.max()

    def pts(x: np.ndarray, y: np.ndarray) -> str:
        return " ".join(
            f"{10 + (a - ox) * k:.1f},{10 + (oy - b) * k:.1f}"
            for a, b in zip(x, y, strict=True)
        )

    def loop(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.append(x, x[0]), np.append(y, y[0])

    edges = "".join(
        (
            f'<polyline points="{pts(*loop(x, y))}" fill="none" '
            'stroke="#888" stroke-width="1"/>'
        )
        for x, y in ((lx, ly), (rx, ry))
    )
    centre = (
        f'<polyline points="{pts(*loop(track.x, track.y))}" fill="none" '
        'stroke="#fff" stroke-width="0.6" stroke-dasharray="3 3" opacity="0.6"/>'
    )
    x, y = line_xy(track, line.alpha)
    x, y = loop(x, y)
    v = np.append(line.speed, line.speed[0])
    t = (v - v.min()) / max(np.ptp(v), 1e-9)
    segs = "".join(
        f'<line x1="{10 + (x[i] - ox) * k:.1f}" y1="{10 + (oy - y[i]) * k:.1f}" '
        f'x2="{10 + (x[i + 1] - ox) * k:.1f}" y2="{10 + (oy - y[i + 1]) * k:.1f}" '
        f'stroke="{colour(t[i])}" stroke-width="2.2"/>'
        for i in range(len(x) - 1)
    )
    held = " (held-out)" if name in TUM_HELD_OUT else ""
    gain = 100 * (line.lap_time_s / line.centre_lap_time_s - 1)
    return (
        f'<figure><svg viewBox="0 0 {SIZE} {SIZE}" width="{SIZE}" height="{SIZE}">'
        f"{edges}{centre}{segs}</svg><figcaption><b>{name}</b>{held}<br>"
        f"centreline {line.centre_lap_time_s:.1f}s &rarr; optimal line "
        f"{line.lap_time_s:.1f}s ({gain:+.1f}%) &middot; "
        f"{v.min() * 3.6:.0f}-{v.max() * 3.6:.0f} km/h</figcaption></figure>"
    )


out = Path(sys.argv[1] if len(sys.argv) > 1 else "runs/optimal_racing_lines.html")
out.parent.mkdir(parents=True, exist_ok=True)
html = (
    "<!doctype html><meta charset=utf-8><title>Optimal racing lines</title>"
    "<style>body{background:#16181d;color:#ddd;font:14px system-ui;margin:16px}"
    "main{display:flex;flex-wrap:wrap;gap:16px}figure{margin:0}"
    "svg{background:#23262e;border-radius:6px}figcaption{text-align:center;margin-top:4px}"
    "</style><h2>Optimal (minimum-curvature) racing line, 25 TUM tracks</h2>"
    "<p>Coloured line = the line the driver agent is scored against, blue slow to red "
    "fast. Grey = track edges, dashed = centreline.</p><main>"
    + "".join(svg(n) for n in TUM_TRACKS)
    + "</main>"
)
out.write_text(html, encoding="utf-8")
print(f"Wrote {out}")
