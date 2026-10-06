"""Limit lap on the centreline vs on TUM's racing line, in our own physics.

Usage: git clone --depth 1 https://github.com/TUMFTM/racetrack-database /tmp/rtdb
       uv run python scripts/raceline_gain.py /tmp/rtdb

Reproduces the table in docs/racing_line_research.md (section 1.1).
"""

import glob
import os
import statistics as st
import sys

import numpy as np

from f1rl.config import EXAMPLE_CAR
from f1rl.envs.circuits import segments_from_position
from f1rl.envs.track import Segment, Track
from f1rl.models.lap import lap_time_s

D = sys.argv[1] if len(sys.argv) > 1 else "racetrack-database"


def track_from_xy(xy):
    xy = np.vstack([xy, xy[:1]])  # close the loop
    d = np.r_[0, np.cumsum(np.hypot(*np.diff(xy, axis=0).T))]
    st_ = segments_from_position(xy[:, 0], xy[:, 1], d, 2.0)
    # extra smoothing like our geo loader: 25 m moving average of curvature
    k = np.array([c for _, c in st_])
    w = 13
    k = np.convolve(np.pad(k, w // 2, mode="wrap"), np.ones(w) / w, mode="valid")
    return Track(tuple(Segment(st_[0][0], float(c)) for c in k)), d[-1]


rows = []
for f in sorted(glob.glob(f"{D}/tracks/*.csv")):
    name = os.path.basename(f)[:-4]
    c = np.loadtxt(f, delimiter=",", comments="#")
    r = np.loadtxt(f"{D}/racelines/{name}.csv", delimiter=",", comments="#")
    tc, Lc = track_from_xy(c[:, :2])
    tr, Lr = track_from_xy(r)
    a, b = lap_time_s(EXAMPLE_CAR, tc), lap_time_s(EXAMPLE_CAR, tr)
    rows.append(
        (name, Lc, Lr, c[:, 2].mean() + c[:, 3].mean(), a, b, 100 * (b / a - 1))
    )
for n, Lc, Lr, w, a, b, g in rows:
    print(
        f"{n:14} centre {Lc:6.0f} m  line {Lr:6.0f} m  width {w:4.1f} m  "
        f"limit lap: centreline {a:6.1f}s  racing line {b:6.1f}s  {g:+5.1f}%"
    )
print(
    "mean gain", round(st.mean(r[-1] for r in rows), 1), "% over", len(rows), "tracks"
)
