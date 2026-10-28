"""Race laps from OpenF1 (api.openf1.org), for seasons FastF1 can no longer download.

Since mid-October 2026 F1's own timing archive needs an F1 TV login, and FastF1's
free mirror only holds 2021-2022 (docs/project_status.md). OpenF1's historical data
(2023 on) is free without an account (openf1.org/docs). This module turns one OpenF1
race into the same `LapRow` list that `f1rl.incidents` builds from FastF1, so pass
and defence detection is identical for both sources.

How each `LapRow` field is built (all pure, in `rows_from_openf1`):

- **end_time_s**: ``date_start + lap_duration`` (OpenF1 calls ``date_start``
  approximate), else the next lap's ``date_start``. Epoch seconds.
- **position**: the driver's last ``position`` record at or before the lap end.
- **gap_ahead_s**: the last numeric ``intervals.interval`` at or before the lap end
  -- the timing gap to the car ahead, which is more precise than lap-end times.
- **pitted**: the lap of a ``pit`` record and the one after it, and any out lap.
- **track_status**: FastF1-style codes from ``race_control`` (`status_spans`):
  4 safety car, 6 virtual safety car, 5 red flag, 2 yellow; "1" if none overlapped.
- **compound / tyre_life**: from ``stints``, counting the current lap as FastF1 does.

Only `fetch` touches the network; responses are cached under ``data/openf1_cache/``
so a season is downloaded once.
"""

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from f1rl.incidents import LapRow

API_URL = "https://api.openf1.org/v1"
DATA_DIR = Path(__file__).resolve().parents[2] / "data"
OPENF1_CACHE_DIR = DATA_DIR / "openf1_cache"
#: Pause between live requests (s), and retries on HTTP 429 / 5xx. Politeness
#: settings for a free public API, not F1 figures.
REQUEST_PAUSE_S = 0.4
RETRIES = 4

#: Endpoints one race needs (besides ``sessions`` / ``meetings`` per season).
RACE_ENDPOINTS = (
    "drivers",
    "laps",
    "position",
    "intervals",
    "pit",
    "stints",
    "race_control",
    "overtakes",
)

Record = Mapping[str, Any]


def epoch_s(stamp: str | None) -> float | None:
    """ISO 8601 UTC time -> seconds since the epoch (None stays None)."""
    if not stamp:
        return None
    return datetime.fromisoformat(stamp.replace("Z", "+00:00")).timestamp()


# --- race control -> track status ----------------------------------------------


def status_spans(race_control: Iterable[Record]) -> list[tuple[float, float, str]]:
    """``(start, end, code)`` spans when racing was not green.

    Safety car (4), virtual safety car (6) and red flag (5) run from their message
    until the next track-wide GREEN / CLEAR flag (a VSC also ends at "VSC ENDING").
    A sector yellow (2) runs until that sector is cleared. Anything never closed
    ends at the last message, so it can only make laps less clean, never more.
    """
    msgs = sorted(
        ((t, r) for r in race_control if (t := epoch_s(r.get("date"))) is not None),
        key=lambda m: m[0],  # stable: same-second messages keep their order
    )
    if not msgs:
        return []
    last = msgs[-1][0]
    spans: list[tuple[float, float, str]] = []
    open_track: dict[str, float] = {}  # code -> start
    open_sector: dict[Any, float] = {}  # sector -> start of its yellow
    for t, r in msgs:
        text = str(r.get("message") or "").upper()
        flag = str(r.get("flag") or "").upper()
        scope = str(r.get("scope") or "").upper()
        category = str(r.get("category") or "")
        if category == "SafetyCar":
            if "VIRTUAL SAFETY CAR DEPLOYED" in text:
                open_track.setdefault("6", t)
            elif "SAFETY CAR DEPLOYED" in text:
                open_track.setdefault("4", t)
            elif ("VSC ENDING" in text or "VIRTUAL SAFETY CAR ENDING" in text) and (
                "6" in open_track
            ):
                spans.append((open_track.pop("6"), t, "6"))
        if flag == "RED":
            open_track.setdefault("5", t)
        if flag in ("GREEN", "CLEAR") and scope == "TRACK":
            for code, start in list(open_track.items()):
                spans.append((start, t, code))
            open_track.clear()
            for start in open_sector.values():
                spans.append((start, t, "2"))
            open_sector.clear()
        if flag in ("YELLOW", "DOUBLE YELLOW"):
            key = r.get("sector") if scope == "SECTOR" else "track"
            open_sector.setdefault(key, t)
        if flag == "CLEAR" and scope == "SECTOR":
            key = r.get("sector")
            if key in open_sector:
                spans.append((open_sector.pop(key), t, "2"))
    spans += [(s, last, c) for c, s in open_track.items()]
    spans += [(s, last, "2") for s in open_sector.values()]
    return sorted(spans)


def lap_status(
    start: float, end: float, spans: Sequence[tuple[float, float, str]]
) -> str:
    """FastF1-style status string for a lap from ``start`` to ``end``."""
    codes = {c for s, e, c in spans if s <= end and e >= start}
    return "".join(sorted(codes)) or "1"


# --- one race -> lap rows --------------------------------------------------------


def _last_at(series: Sequence[tuple[float, Any]], t: float) -> Any:
    """Value of the last ``(time, value)`` at or before ``t`` (series sorted)."""
    lo, hi = 0, len(series)
    while lo < hi:
        mid = (lo + hi) // 2
        if series[mid][0] <= t:
            lo = mid + 1
        else:
            hi = mid
    return series[lo - 1][1] if lo else None


def _series(
    records: Iterable[Record], field: str
) -> dict[int, list[tuple[float, Any]]]:
    out: dict[int, list[tuple[float, Any]]] = {}
    for r in records:
        t = epoch_s(r.get("date"))
        if t is None or r.get(field) is None:
            continue
        out.setdefault(int(r["driver_number"]), []).append((t, r[field]))
    for s in out.values():
        s.sort(key=lambda x: x[0])
    return out


def _gap(value: Any) -> float | None:
    """An ``interval`` value as seconds; None for "+1 LAP" strings and blanks."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rows_from_openf1(
    drivers: Iterable[Record],
    laps: Iterable[Record],
    positions: Iterable[Record],
    intervals: Iterable[Record],
    pits: Iterable[Record],
    stints: Iterable[Record],
    race_control: Iterable[Record],
) -> list[LapRow]:
    """One race's OpenF1 records -> `LapRow`s (laps with no end or place dropped)."""
    names = {int(d["driver_number"]): str(d["name_acronym"]) for d in drivers}
    place = _series(positions, "position")
    gaps = _series(intervals, "interval")
    spans = status_spans(race_control)
    pitted = {(int(p["driver_number"]), int(p["lap_number"])) for p in pits}
    pitted |= {(d, n + 1) for d, n in pitted}
    stint_list: dict[int, list[Record]] = {}
    for s in stints:
        stint_list.setdefault(int(s["driver_number"]), []).append(s)

    by_driver: dict[int, dict[int, Record]] = {}
    for lap in laps:
        by_driver.setdefault(int(lap["driver_number"]), {})[int(lap["lap_number"])] = (
            lap
        )

    rows = []
    for num, own in by_driver.items():
        for n, lap in sorted(own.items()):
            start = epoch_s(lap.get("date_start"))
            duration = lap.get("lap_duration")
            nxt = own.get(n + 1)
            if start is not None and duration is not None:
                end: float | None = start + float(duration)
            else:
                end = epoch_s(nxt.get("date_start")) if nxt else None
            if end is None:
                continue
            pos = _last_at(place.get(num, []), end)
            if pos is None:
                continue
            compound, life = "", float("nan")
            for s in stint_list.get(num, []):
                first, last = s.get("lap_start"), s.get("lap_end")
                if first is not None and first <= n and (last is None or n <= last):
                    compound = str(s.get("compound") or "")
                    age = s.get("tyre_age_at_start")
                    life = float(age) + n - first + 1 if age is not None else life
            rows.append(
                LapRow(
                    driver=names.get(num, str(num)),
                    lap=n,
                    position=int(pos),
                    end_time_s=end,
                    pitted=(num, n) in pitted or bool(lap.get("is_pit_out_lap")),
                    track_status=lap_status(
                        start if start is not None else end - 1.0, end, spans
                    ),
                    compound=compound,
                    tyre_life=life,
                    gap_ahead_s=_gap(_last_at(gaps.get(num, []), end)),
                )
            )
    return rows


# --- network (cached) -------------------------------------------------------------


def fetch(
    endpoint: str, cache_dir: Path = OPENF1_CACHE_DIR, **params: Any
) -> list[Any]:
    """GET ``/v1/<endpoint>?<params>`` as JSON records, cached on disk."""
    query = urllib.parse.urlencode(sorted(params.items()))
    path = cache_dir / endpoint / (query.replace("&", "_").replace("=", "-") + ".json")
    if path.exists():
        cached: list[Any] = json.loads(path.read_text())
        return cached
    url = f"{API_URL}/{endpoint}?{query}"
    for attempt in range(RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                data = json.loads(resp.read())
            break
        except urllib.error.HTTPError as exc:
            if exc.code == 404:  # OpenF1 answers "no results" with 404
                data = []
                break
            if attempt == RETRIES or exc.code not in (429, 500, 502, 503, 504):
                raise
            time.sleep(2.0 * 2**attempt)
    time.sleep(REQUEST_PAUSE_S)
    if not isinstance(data, list):
        data = []
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return data


def race_sessions(year: int) -> list[dict[str, Any]]:
    """The year's Grand Prix race sessions (not sprints), each with its meeting name."""
    meetings = {m["meeting_key"]: m for m in fetch("meetings", year=year)}
    races = [
        s
        for s in fetch("sessions", year=year, session_name="Race")
        if not s.get("is_cancelled")
    ]
    out = []
    for s in sorted(races, key=lambda s: str(s.get("date_start"))):
        m = meetings.get(s["meeting_key"], {})
        out.append(dict(s) | {"meeting_name": m.get("meeting_name", s.get("location"))})
    return out


def load_session_records(session_key: int) -> dict[str, list[Any]]:
    """Every endpoint `RACE_ENDPOINTS` names, for one session."""
    return {ep: fetch(ep, session_key=session_key) for ep in RACE_ENDPOINTS}
