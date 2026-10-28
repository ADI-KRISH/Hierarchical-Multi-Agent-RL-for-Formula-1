"""Real overtakes and defences from F1 races (2018 on), for scenario seeding.

``uv run python -m f1rl.incidents --year 2023 [--gp Bahrain]`` loads race laps
(network on a cache miss) and writes one JSON file per race under
``data/incidents/``. Source: FastF1 up to 2022 (its free mirror holds 2021-2022
only, since F1 locked its archive in October 2026), OpenF1 from 2023
(`f1rl.openf1`); ``--source`` overrides. Each record is an *episode*: who, which
lap, the gap, tyres, and, for passes, how far round the lap the cars swapped order.

Detection works on lap-end data only, which FastF1 has for every race since 2018:

- **Pass**: car A behind car B at the end of lap L-1 and ahead at the end of lap L,
  with both cars on clean laps (no pit stop on L-1 or L, track status green on L).
  Pit-stop swaps, safety cars and retirements never count.
- **Defence**: A directly behind B within `IncidentParams.attack_gap_s` at the line
  for at least `min_defence_laps` clean laps in a row. Outcome ``held`` if the run
  ends without A getting by, ``lost`` if it ends with a pass.
- **Where**: for a pass, the lap distance at which A's time gap to B changes sign,
  from both cars' telemetry on lap L (`pass_location`). ~4 Hz data, so this is good
  to a few tens of metres, enough to name the corner but not the line.

Each race also carries the TUM track it maps to and that track's split, so held-out
tracks' incidents are never used for training (`split` is ``held_out``).

The detection functions are pure (rows in, episodes out); only `load_race` and
`main` touch FastF1 or the disk.
"""

import argparse
import json
import math
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1rl.config import IncidentParams
from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRACKS, TUM_VALIDATION

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
INCIDENT_DIR = DATA_DIR / "incidents"
FASTF1_CACHE_DIR = DATA_DIR / "fastf1_cache"

#: FastF1 event ``Location`` (accents stripped, lower case) -> TUM track name.
#: Only circuits the TUM database has; the rest map to None (no sim track).
LOCATION_TO_TUM: dict[str, str] = {
    "austin": "Austin",
    "budapest": "Budapest",
    "barcelona": "Catalunya",
    "montmelo": "Catalunya",
    "hockenheim": "Hockenheim",
    "melbourne": "Melbourne",
    "mexico city": "MexicoCity",
    "montreal": "Montreal",
    "monza": "Monza",
    "nurburg": "Nuerburgring",
    "nurburgring": "Nuerburgring",
    "sakhir": "Sakhir",
    "sao paulo": "SaoPaulo",
    "shanghai": "Shanghai",
    "silverstone": "Silverstone",
    "sochi": "Sochi",
    "spa-francorchamps": "Spa",
    "spa": "Spa",
    "spielberg": "Spielberg",
    "suzuka": "Suzuka",
    "yas island": "YasMarina",
    "yas marina": "YasMarina",
    "abu dhabi": "YasMarina",
    "zandvoort": "Zandvoort",
}


@dataclass(frozen=True)
class LapRow:
    """One car's completed race lap, as FastF1 reports it."""

    driver: str
    lap: int
    position: int
    end_time_s: float  # session time at the end of the lap
    pitted: bool  # pit in or pit out on this lap
    track_status: str  # FastF1 codes seen during the lap, e.g. "1" or "124"
    compound: str = ""
    tyre_life: float = math.nan
    # Gap (s) to the car directly ahead at the line, if the source measures it
    # (OpenF1 ``intervals``). None: use the difference in `end_time_s`.
    gap_ahead_s: float | None = None


@dataclass(frozen=True)
class PassEpisode:
    lap: int
    attacker: str
    defender: str
    position: int  # the place the attacker took
    gap_before_s: float  # attacker behind defender at the end of lap - 1
    attacker_compound: str
    attacker_tyre_life: float
    defender_compound: str
    defender_tyre_life: float
    location_m: float | None = None  # lap distance of the swap, if telemetry had it


@dataclass(frozen=True)
class DefenceEpisode:
    first_lap: int
    last_lap: int
    attacker: str
    defender: str
    min_gap_s: float
    outcome: str  # "held" | "lost"


def normalise_location(name: str) -> str:
    """``"São Paulo"`` -> ``"sao paulo"``: accents stripped, lower case, trimmed."""
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return plain.strip().lower()


def tum_track(location: str) -> str | None:
    """The TUM track for a FastF1 event location, or None if the sim lacks it."""
    name = LOCATION_TO_TUM.get(normalise_location(location))
    return name if name in TUM_TRACKS else None


def split_of(track: str | None) -> str:
    """``train`` / ``validation`` / ``held_out`` / ``no_track`` for a TUM track."""
    if track is None:
        return "no_track"
    if track in TUM_HELD_OUT:
        return "held_out"
    if track in TUM_VALIDATION:
        return "validation"
    return "train"


def _clean(row: LapRow, params: IncidentParams) -> bool:
    status = row.track_status or params.clean_track_status
    return not row.pitted and set(status) <= set(params.clean_track_status)


def _by_lap(rows: Iterable[LapRow]) -> dict[int, dict[str, LapRow]]:
    laps: dict[int, dict[str, LapRow]] = {}
    for r in rows:
        laps.setdefault(r.lap, {})[r.driver] = r
    return laps


def detect_passes(
    rows: Iterable[LapRow], params: IncidentParams | None = None
) -> list[PassEpisode]:
    """Every on-track pass: a pair whose order flips over one clean lap for both."""
    p = params or IncidentParams()
    laps = _by_lap(rows)
    out: list[PassEpisode] = []
    for lap in sorted(laps):
        prev, now = laps.get(lap - 1), laps[lap]
        if prev is None:
            continue
        drivers = [d for d in now if d in prev]
        for a in drivers:
            for b in drivers:
                pa0, pb0, pa1, pb1 = prev[a], prev[b], now[a], now[b]
                if not (pa0.position > pb0.position and pa1.position < pb1.position):
                    continue
                if not all(_clean(r, p) for r in (pa1, pb1)):
                    continue
                if pa0.pitted or pb0.pitted:
                    continue
                out.append(
                    PassEpisode(
                        lap=lap,
                        attacker=a,
                        defender=b,
                        position=pa1.position,
                        gap_before_s=pa0.end_time_s - pb0.end_time_s,
                        attacker_compound=pa1.compound,
                        attacker_tyre_life=pa1.tyre_life,
                        defender_compound=pb1.compound,
                        defender_tyre_life=pb1.tyre_life,
                    )
                )
    return out


def detect_defences(
    rows: Iterable[LapRow], params: IncidentParams | None = None
) -> list[DefenceEpisode]:
    """Runs of clean laps with the same car right behind, within the attack gap."""
    p = params or IncidentParams()
    laps = _by_lap(rows)
    # (attacker, defender) -> (first lap, last lap, min gap) of the open run
    open_runs: dict[tuple[str, str], tuple[int, int, float]] = {}
    out: list[DefenceEpisode] = []

    def close(pair: tuple[str, str], outcome: str) -> None:
        first, last, gap = open_runs.pop(pair)
        if last - first + 1 >= p.min_defence_laps:
            out.append(DefenceEpisode(first, last, pair[0], pair[1], gap, outcome))

    for lap in sorted(laps):
        now = laps[lap]
        by_pos = {r.position: r for r in now.values()}
        close_now: dict[tuple[str, str], float] = {}
        for pos, ahead in by_pos.items():
            behind = by_pos.get(pos + 1)
            if behind is None or not (_clean(ahead, p) and _clean(behind, p)):
                continue
            gap = behind.gap_ahead_s
            if gap is None:
                gap = behind.end_time_s - ahead.end_time_s
            if 0.0 <= gap <= p.attack_gap_s:
                close_now[(behind.driver, ahead.driver)] = gap
        for pair in list(open_runs):
            if pair in close_now:
                continue
            a, b = pair
            ra, rb = now.get(a), now.get(b)
            got_by = (
                ra is not None
                and rb is not None
                and ra.position < rb.position
                and _clean(ra, p)
                and _clean(rb, p)
            )
            if got_by:
                close(pair, "lost")
            elif ra is None or rb is None or not (_clean(ra, p) and _clean(rb, p)):
                open_runs.pop(pair)  # pit stop, safety car or retirement: no verdict
            else:
                close(pair, "held")
        for pair, gap in close_now.items():
            first, _, best = open_runs.get(pair, (lap, lap, gap))
            open_runs[pair] = (first, lap, min(best, gap))
    for pair in list(open_runs):
        close(pair, "held")  # the flag fell with the attacker still behind
    return out


FloatArray = NDArray[np.float64]


def pass_location(
    dist_a: Sequence[float] | FloatArray,
    time_a: Sequence[float] | FloatArray,
    dist_b: Sequence[float] | FloatArray,
    time_b: Sequence[float] | FloatArray,
    step_m: float = 5.0,
) -> float | None:
    """Lap distance (m) where car A, behind car B, gets ahead.

    Each car's (distance, session time) samples on the same lap. At every distance
    ``d`` the gap is ``t_a(d) - t_b(d)``: positive while A arrives later. The pass is
    the last place it goes from positive to non-positive. None if it never does on
    this lap (the swap happened across the line, or the traces are too short).
    """
    da, ta = np.asarray(dist_a, float), np.asarray(time_a, float)
    db, tb = np.asarray(dist_b, float), np.asarray(time_b, float)
    if len(da) < 2 or len(db) < 2:
        return None
    lo, hi = max(da.min(), db.min()), min(da.max(), db.max())
    if hi - lo < step_m:
        return None
    grid = np.arange(lo, hi, step_m)
    gap = np.interp(grid, *_monotone(da, ta)) - np.interp(grid, *_monotone(db, tb))
    flips = np.nonzero((gap[:-1] > 0) & (gap[1:] <= 0))[0]
    if len(flips) == 0:
        return None
    i = int(flips[-1])
    # linear interpolation between the two grid points around the sign change
    frac = gap[i] / (gap[i] - gap[i + 1])
    return float(grid[i] + frac * step_m)


def _monotone(dist: FloatArray, time: FloatArray) -> tuple[FloatArray, FloatArray]:
    order = np.argsort(dist, kind="stable")
    d, t = dist[order], time[order]
    keep = np.concatenate(([True], np.diff(d) > 0))
    return d[keep], t[keep]


# --- FastF1 adapter (I/O) ---------------------------------------------------------


def _seconds(value: Any) -> float:
    if value is None:
        return math.nan
    total = getattr(value, "total_seconds", None)
    if total is None:
        return float(value)
    seconds = float(total())
    return seconds


def _is_set(value: Any) -> bool:
    return not (value is None or value != value)  # NaT / NaN compare unequal


def rows_from_laps(laps: Any) -> list[LapRow]:
    """FastF1 ``Laps`` -> `LapRow` list (laps with no position or time dropped)."""
    out = []
    for _, lap in laps.iterrows():
        pos, end = lap["Position"], lap["Time"]
        if not (_is_set(pos) and _is_set(end)):
            continue
        status = lap.get("TrackStatus")
        life = lap.get("TyreLife")
        out.append(
            LapRow(
                driver=str(lap["Driver"]),
                lap=int(lap["LapNumber"]),
                position=int(pos),
                end_time_s=_seconds(end),
                pitted=_is_set(lap.get("PitInTime")) or _is_set(lap.get("PitOutTime")),
                track_status=str(status) if _is_set(status) else "",
                compound=str(lap.get("Compound") or ""),
                tyre_life=float(life) if _is_set(life) else math.nan,
            )
        )
    return out


def _trace(session: Any, driver: str, lap: int) -> tuple[FloatArray, FloatArray]:
    sel = session.laps.pick_drivers(driver).pick_laps(lap)
    if len(sel) == 0:
        return np.empty(0), np.empty(0)
    tel = sel.iloc[0].get_car_data().add_distance()
    t = np.array([_seconds(v) for v in tel["SessionTime"]], float)
    return np.asarray(tel["Distance"], float), t


def load_race(
    year: int, grand_prix: str, params: IncidentParams | None = None
) -> dict[str, Any]:
    """Passes and defences in one race, plus its track and split. Needs FastF1."""
    import fastf1  # local import: heavy and network-capable, only here

    FASTF1_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(FASTF1_CACHE_DIR))
    session = fastf1.get_session(year, grand_prix, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)
    rows = rows_from_laps(session.laps)
    passes = []
    for ep in detect_passes(rows, params):
        try:
            da, ta = _trace(session, ep.attacker, ep.lap)
            db, tb = _trace(session, ep.defender, ep.lap)
            where = pass_location(da, ta, db, tb)
        except Exception:  # missing telemetry for one car: keep the lap-level pass
            where = None
        passes.append(asdict(ep) | {"location_m": where})
    return _race_record(
        "fastf1",
        year,
        str(session.event["EventName"]),
        str(session.event["Location"]),
        passes,
        rows,
        params,
    )


def _race_record(
    source: str,
    year: int,
    grand_prix: str,
    location: str,
    passes: list[dict[str, Any]],
    rows: list[LapRow],
    params: IncidentParams | None,
) -> dict[str, Any]:
    track = tum_track(location)
    return {
        "source": source,
        "year": year,
        "grand_prix": grand_prix,
        "location": location,
        "tum_track": track,
        "split": split_of(track),
        "era": "pre_2022" if year < 2022 else "ground_effect",
        "params": asdict(params or IncidentParams()),
        "passes": passes,
        "defences": [asdict(d) for d in detect_defences(rows, params)],
    }


def load_race_openf1(
    session: dict[str, Any], params: IncidentParams | None = None
) -> dict[str, Any]:
    """Passes and defences in one OpenF1 race session (`openf1.race_sessions`).

    No telemetry is downloaded, so ``location_m`` stays None. OpenF1's own
    overtake list is counted alongside (``openf1_overtakes``) as a cross-check.
    """
    from f1rl import openf1

    rec = openf1.load_session_records(int(session["session_key"]))
    rows = openf1.rows_from_openf1(
        rec["drivers"],
        rec["laps"],
        rec["position"],
        rec["intervals"],
        rec["pit"],
        rec["stints"],
        rec["race_control"],
    )
    passes = [asdict(ep) for ep in detect_passes(rows, params)]
    race = _race_record(
        "openf1",
        int(session["year"]),
        str(session["meeting_name"]),
        str(session.get("location", "")),
        passes,
        rows,
        params,
    )
    race["openf1_overtakes"] = len(rec["overtakes"])
    return race


def _race_names(year: int) -> list[str]:
    import fastf1

    schedule = fastf1.get_event_schedule(year, include_testing=False)
    return [str(name) for name in schedule["EventName"]]


def _matches(gp: str, *names: str) -> bool:
    want = normalise_location(gp)
    return any(want in normalise_location(n) for n in names if n)


def main(argv: Sequence[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--gp", help="Grand prix name; every race of the year if left out")
    ap.add_argument("--source", choices=("auto", "fastf1", "openf1"), default="auto")
    ap.add_argument("--out", type=Path, default=INCIDENT_DIR)
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    source = args.source
    if source == "auto":
        source = "openf1" if args.year >= 2023 else "fastf1"

    jobs: list[tuple[str, Any]] = []
    if source == "openf1":
        from f1rl import openf1

        for s in openf1.race_sessions(args.year):
            names = (
                s["meeting_name"],
                s.get("location", ""),
                s.get("country_name", ""),
            )
            if args.gp is None or _matches(args.gp, *names):
                jobs.append((str(s["meeting_name"]), s))
    else:
        races = [args.gp] if args.gp else _race_names(args.year)
        jobs = [(gp, gp) for gp in races]
    if not jobs:
        print(f"{args.year}: no race matches {args.gp!r} ({source})")

    for name, job in jobs:
        try:
            if source == "openf1":
                race = load_race_openf1(job)
            else:
                race = load_race(args.year, job)
        except Exception as exc:  # one bad race should not stop the season
            print(f"{args.year} {name}: skipped ({exc})")
            continue
        slug = normalise_location(race["grand_prix"]).replace(" ", "_")
        path = args.out / f"{args.year}_{slug}.json"
        path.write_text(json.dumps(race, indent=1))
        extra = (
            f", OpenF1 counts {race['openf1_overtakes']} overtakes"
            if "openf1_overtakes" in race
            else ""
        )
        print(
            f"{args.year} {race['grand_prix']}: {len(race['passes'])} passes, "
            f"{len(race['defences'])} defences{extra}, split={race['split']} -> {path}"
        )


if __name__ == "__main__":
    main()
