"""Lap analytics: where a driver gains or loses time against a reference.

Pure functions over telemetry (distance/time/speed samples) and track geometry,
no I/O -- shared by the baseline report now and agent-vs-baseline eval later.
"""

import itertools
import math
import statistics
from dataclasses import dataclass

import numpy as np

from f1rl.envs.track import Track


@dataclass(frozen=True)
class Section:
    """A stretch of the lap: one corner ("T1", "T2", ...) or the straight
    between two of them ("S1", ...), numbered in lap order.
    """

    name: str
    is_corner: bool
    start_m: float
    end_m: float
    min_radius_m: float | None  # tightest radius inside; None on a straight


def lap_sections(
    track: Track, corner_curvature_per_m: float, min_corner_m: float
) -> list[Section]:
    """Split the lap into alternating corners and straights.

    A corner is a run of segments tighter than `corner_curvature_per_m`; runs
    shorter than `min_corner_m` (noise kinks in real-circuit curvature) count as
    straight, and neighbouring runs of the same kind merge.
    """
    runs: list[list[float]] = []  # [is_corner, start, end, max |curvature|]
    position_m = 0.0
    for segment in track.segments:
        curvature = abs(segment.curvature_per_m)
        corner = curvature > corner_curvature_per_m
        end_m = position_m + segment.length_m
        if runs and bool(runs[-1][0]) == corner:
            runs[-1][2] = end_m
            runs[-1][3] = max(runs[-1][3], curvature)
        else:
            runs.append([float(corner), position_m, end_m, curvature])
        position_m = end_m

    merged: list[list[float]] = []
    for run in runs:
        if run[0] and run[2] - run[1] < min_corner_m:
            run = [0.0, run[1], run[2], 0.0]
        if merged and merged[-1][0] == run[0]:
            merged[-1][2] = run[2]
            merged[-1][3] = max(merged[-1][3], run[3])
        else:
            merged.append(run)

    sections = []
    corners = straights = 0
    for is_corner, start_m, end_m, curvature in merged:
        if is_corner:
            corners += 1
            name, radius = f"T{corners}", 1.0 / curvature
        else:
            straights += 1
            name, radius = f"S{straights}", None
        sections.append(Section(name, bool(is_corner), start_m, end_m, radius))
    return sections


def profile_times_s(speeds_ms: list[float], ds_m: float) -> list[float]:
    """Cumulative time (s) at each station of a speed profile `ds_m` apart."""
    times = [0.0]
    for a, b in itertools.pairwise(speeds_ms):
        times.append(times[-1] + ds_m / ((a + b) / 2))
    return times


def time_at_distance(
    distance_m: list[float], time_s: list[float], query_m: list[float]
) -> list[float]:
    """Time (s) a run reached each `query_m`, interpolated from its samples."""
    return [float(t) for t in np.interp(query_m, distance_m, time_s)]


@dataclass(frozen=True)
class SectionTiming:
    name: str
    is_corner: bool
    start_m: float
    end_m: float
    min_radius_m: float | None
    time_s: float
    reference_time_s: float
    min_speed_ms: float
    reference_min_speed_ms: float

    @property
    def delta_s(self) -> float:
        """Time lost (+) or gained (-) against the reference in this section."""
        return self.time_s - self.reference_time_s


def section_timings(
    sections: list[Section],
    run_distance_m: list[float],
    run_time_s: list[float],
    run_speed_ms: list[float],
    ref_distance_m: list[float],
    ref_time_s: list[float],
    ref_speed_ms: list[float],
) -> list[SectionTiming]:
    """Per-section time and minimum speed of a run against a reference lap."""
    result = []
    for section in sections:
        edges = [section.start_m, section.end_m]
        run_t = time_at_distance(run_distance_m, run_time_s, edges)
        ref_t = time_at_distance(ref_distance_m, ref_time_s, edges)
        result.append(
            SectionTiming(
                name=section.name,
                is_corner=section.is_corner,
                start_m=section.start_m,
                end_m=section.end_m,
                min_radius_m=section.min_radius_m,
                time_s=run_t[1] - run_t[0],
                reference_time_s=ref_t[1] - ref_t[0],
                min_speed_ms=_min_between(run_distance_m, run_speed_ms, *edges),
                reference_min_speed_ms=_min_between(
                    ref_distance_m, ref_speed_ms, *edges
                ),
            )
        )
    return result


def _min_between(
    distance_m: list[float], values: list[float], start_m: float, end_m: float
) -> float:
    inside = [
        v for d, v in zip(distance_m, values, strict=True) if start_m <= d < end_m
    ]
    edges = [float(v) for v in np.interp([start_m, end_m], distance_m, values)]
    return min([*inside, *edges])


@dataclass(frozen=True)
class LapTimeSummary:
    episodes: int
    completed: int
    off_track: int
    mean_s: float | None
    std_s: float | None
    best_s: float | None
    worst_s: float | None
    median_s: float | None

    @property
    def completion_rate(self) -> float:
        return self.completed / self.episodes if self.episodes else 0.0


def summarize_laps(lap_times_s: list[float | None], off_track: int) -> LapTimeSummary:
    """Stats over completed laps; `None` entries are laps that didn't finish."""
    done = [t for t in lap_times_s if t is not None and math.isfinite(t)]
    return LapTimeSummary(
        episodes=len(lap_times_s),
        completed=len(done),
        off_track=off_track,
        mean_s=statistics.mean(done) if done else None,
        std_s=statistics.pstdev(done) if done else None,
        best_s=min(done) if done else None,
        worst_s=max(done) if done else None,
        median_s=statistics.median(done) if done else None,
    )
