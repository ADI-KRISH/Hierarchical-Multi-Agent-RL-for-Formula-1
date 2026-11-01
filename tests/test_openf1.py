"""OpenF1 records -> lap rows, on made-up responses (no network)."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from f1rl import incidents, openf1
from f1rl.config import IncidentParams
from f1rl.incidents import detect_defences, detect_passes
from f1rl.openf1 import lap_status, rows_from_openf1, status_spans

T0 = datetime(2024, 3, 2, 15, 0, tzinfo=UTC).timestamp()


def iso(t: float) -> str:
    return datetime.fromtimestamp(T0 + t, UTC).isoformat()


def msg(t: float, **kw: Any) -> dict[str, Any]:
    return {"date": iso(t), "category": "Flag", "scope": "Track"} | kw


def test_status_spans_from_race_control() -> None:
    rc = [
        msg(100, category="SafetyCar", message="SAFETY CAR DEPLOYED", flag=None),
        msg(200, flag="GREEN", message="TRACK CLEAR"),
        msg(
            300,
            flag="YELLOW",
            scope="Sector",
            sector=5,
            message="YELLOW IN TRACK SECTOR 5",
        ),
        msg(
            310,
            flag="CLEAR",
            scope="Sector",
            sector=5,
            message="CLEAR IN TRACK SECTOR 5",
        ),
        msg(
            400, category="SafetyCar", message="VIRTUAL SAFETY CAR DEPLOYED", flag=None
        ),
        msg(450, category="SafetyCar", message="VIRTUAL SAFETY CAR ENDING", flag=None),
        msg(500, flag="RED", message="RED FLAG"),
    ]
    spans = [(round(s - T0), round(e - T0), c) for s, e, c in status_spans(rc)]
    assert spans == [(100, 200, "4"), (300, 310, "2"), (400, 450, "6"), (500, 500, "5")]
    full = status_spans(rc)
    assert lap_status(T0 + 0, T0 + 90, full) == "1"
    assert lap_status(T0 + 150, T0 + 250, full) == "4"
    assert lap_status(T0 + 190, T0 + 320, full) == "24"


def race_records(pit_lap: int | None = None) -> dict[str, list[dict[str, Any]]]:
    """Two cars, 4 laps of 90 s. HAM 0.5 s behind VER, passes on lap 3."""
    laps, pos, gaps = [], [], []
    for n in range(1, 5):
        for num, lag in ((1, 0.0), (44, 0.5 if n < 3 else -0.4)):
            laps.append(
                {
                    "driver_number": num,
                    "lap_number": n,
                    "date_start": iso(90 * (n - 1) + lag) if n > 1 else None,
                    "lap_duration": 90.0 if n > 1 else None,
                    "is_pit_out_lap": False,
                }
            )
    pos += [
        {"driver_number": 1, "date": iso(0), "position": 1},
        {"driver_number": 44, "date": iso(0), "position": 2},
        {"driver_number": 44, "date": iso(200), "position": 1},
        {"driver_number": 1, "date": iso(200), "position": 2},
    ]
    gaps += [
        {"driver_number": 44, "date": iso(85), "interval": 0.52},
        {"driver_number": 44, "date": iso(175), "interval": 0.48},
        {"driver_number": 1, "date": iso(265), "interval": 0.41},
        {"driver_number": 1, "date": iso(355), "interval": "+1 LAP"},
    ]
    pits = [] if pit_lap is None else [{"driver_number": 1, "lap_number": pit_lap}]
    return {
        "drivers": [
            {"driver_number": 1, "name_acronym": "VER"},
            {"driver_number": 44, "name_acronym": "HAM"},
        ],
        "laps": laps,
        "position": pos,
        "intervals": gaps,
        "pit": pits,
        "stints": [
            {
                "driver_number": 1,
                "lap_start": 1,
                "lap_end": 4,
                "compound": "HARD",
                "tyre_age_at_start": 0,
            },
            {
                "driver_number": 44,
                "lap_start": 1,
                "lap_end": 4,
                "compound": "SOFT",
                "tyre_age_at_start": 3,
            },
        ],
        "race_control": [],
        "overtakes": [{"overtaking_driver_number": 44}],
    }


def rows(rec: dict[str, list[dict[str, Any]]]) -> list[incidents.LapRow]:
    return rows_from_openf1(
        rec["drivers"],
        rec["laps"],
        rec["position"],
        rec["intervals"],
        rec["pit"],
        rec["stints"],
        rec["race_control"],
    )


def test_rows_from_openf1_fields() -> None:
    by = {(r.driver, r.lap): r for r in rows(race_records())}
    assert len(by) == 8  # laps 2-4 end at start + duration; lap 1 at lap 2's start
    ham2 = by[("HAM", 2)]
    assert (ham2.position, ham2.compound, ham2.tyre_life) == (2, "SOFT", 5.0)
    assert ham2.gap_ahead_s == pytest.approx(0.48)
    assert by[("VER", 4)].gap_ahead_s is None  # "+1 LAP" is not a time gap
    assert by[("HAM", 3)].position == 1


def test_openf1_rows_feed_detection() -> None:
    [p] = detect_passes(rows(race_records()))
    assert (p.lap, p.attacker, p.defender) == (3, "HAM", "VER")
    assert detect_passes(rows(race_records(pit_lap=3))) == []
    loose = IncidentParams(min_defence_laps=2)
    found = {
        (d.attacker, d.defender, d.outcome)
        for d in detect_defences(rows(race_records()), loose)
    }
    # HAM sits 0.5 s behind for 2 laps and gets by; then VER sits behind to the flag
    assert found == {("HAM", "VER", "lost"), ("VER", "HAM", "held")}


def test_main_writes_openf1_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rec = race_records()
    session = {
        "session_key": 9999,
        "meeting_key": 1,
        "year": 2024,
        "location": "Suzuka",
        "country_name": "Japan",
        "date_start": iso(0),
    }

    def fake_fetch(endpoint: str, **params: Any) -> list[Any]:
        if endpoint == "meetings":
            return [{"meeting_key": 1, "meeting_name": "Japanese Grand Prix"}]
        if endpoint == "sessions":
            return [session]
        return rec[endpoint]

    monkeypatch.setattr(openf1, "fetch", fake_fetch)
    incidents.main(["--year", "2024", "--gp", "japan", "--out", str(tmp_path)])
    race = json.loads((tmp_path / "2024_japanese_grand_prix.json").read_text())
    assert (race["source"], race["tum_track"], race["split"]) == (
        "openf1",
        "Suzuka",
        "held_out",
    )
    assert len(race["passes"]) == 1 and race["openf1_overtakes"] == 1
    assert race["era"] == "ground_effect"


def test_same_second_messages_keep_their_order() -> None:
    # Real races log several messages in the same second (2024 Suzuka, lap 1).
    rc = [
        msg(10, flag="YELLOW", scope="Sector", sector=3, message="YELLOW IN SECTOR 3"),
        msg(10, flag="YELLOW", scope="Sector", sector=4, message="YELLOW IN SECTOR 4"),
        msg(20, flag="CLEAR", scope="Sector", sector=3, message="CLEAR IN SECTOR 3"),
        msg(20, flag="CLEAR", scope="Sector", sector=4, message="CLEAR IN SECTOR 4"),
    ]
    spans = [(round(s - T0), round(e - T0), c) for s, e, c in status_spans(rc)]
    assert spans == [(10, 20, "2"), (10, 20, "2")]
