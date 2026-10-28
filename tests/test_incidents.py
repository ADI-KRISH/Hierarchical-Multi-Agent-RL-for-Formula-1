"""Pass / defence detection on made-up race laps (no FastF1, no network)."""

import numpy as np
import pytest

from f1rl.config import IncidentParams
from f1rl.incidents import (
    LapRow,
    detect_defences,
    detect_passes,
    normalise_location,
    pass_location,
    split_of,
    tum_track,
)


def race(
    order: list[list[str]],
    gaps: list[list[float]] | None = None,
    pitted: set[tuple[str, int]] | None = None,
    status: dict[int, str] | None = None,
) -> list[LapRow]:
    """Lap rows from the running order at the end of each lap (lap 1 first).

    ``gaps[i][k]`` is the gap (s) of the car in place k+2 to the car ahead on lap i+1;
    1.5 s if not given.
    """
    rows = []
    for i, cars in enumerate(order):
        lap = i + 1
        t = 100.0 * lap
        for k, d in enumerate(cars):
            if k:
                t += gaps[i][k - 1] if gaps else 1.5
            rows.append(
                LapRow(
                    driver=d,
                    lap=lap,
                    position=k + 1,
                    end_time_s=t,
                    pitted=(d, lap) in (pitted or set()),
                    track_status=(status or {}).get(lap, "1"),
                )
            )
    return rows


def test_a_clean_swap_is_a_pass() -> None:
    rows = race([["VER", "HAM"], ["VER", "HAM"], ["HAM", "VER"]])
    [p] = detect_passes(rows)
    assert (p.lap, p.attacker, p.defender, p.position) == (3, "HAM", "VER", 1)
    assert p.gap_before_s == pytest.approx(1.5)


def test_pit_stop_swap_is_not_a_pass() -> None:
    order = [["VER", "HAM"], ["VER", "HAM"], ["HAM", "VER"]]
    assert detect_passes(race(order, pitted={("VER", 3)})) == []
    assert detect_passes(race(order, pitted={("VER", 2)})) == []


def test_safety_car_lap_is_not_a_pass() -> None:
    order = [["VER", "HAM"], ["HAM", "VER"]]
    assert detect_passes(race(order, status={2: "14"})) == []
    assert len(detect_passes(race(order))) == 1


def test_retirement_ahead_is_not_a_pass() -> None:
    # LEC drops out of the table: everyone behind moves up, nobody passed anybody
    rows = race([["LEC", "VER", "HAM"], ["VER", "HAM"]])
    assert detect_passes(rows) == []


def test_three_way_pass_on_one_lap() -> None:
    rows = race([["A", "B", "C"], ["C", "A", "B"]])
    found = {(p.attacker, p.defender) for p in detect_passes(rows)}
    assert found == {("C", "A"), ("C", "B")}


def close_order(laps: int) -> list[list[str]]:
    return [["VER", "HAM"]] * laps


def test_defence_held_for_the_run() -> None:
    gaps = [[2.0], [0.8], [0.6], [0.7], [0.9], [2.5]]
    [d] = detect_defences(race(close_order(6), gaps))
    assert (d.attacker, d.defender, d.first_lap, d.last_lap) == ("HAM", "VER", 2, 5)
    assert d.outcome == "held"
    assert d.min_gap_s == pytest.approx(0.6)


def test_defence_lost_ends_with_a_pass() -> None:
    order = [*close_order(4), ["HAM", "VER"]]
    gaps = [[0.8], [0.6], [0.5], [0.4], [1.0]]
    [d] = detect_defences(race(order, gaps))
    assert (d.first_lap, d.last_lap, d.outcome) == (1, 4, "lost")


def test_short_run_is_not_a_defence() -> None:
    gaps = [[2.0], [0.8], [0.6], [2.0]]
    assert detect_defences(race(close_order(4), gaps)) == []
    loose = IncidentParams(min_defence_laps=2)
    assert len(detect_defences(race(close_order(4), gaps), loose)) == 1


def test_safety_car_cancels_a_defence() -> None:
    gaps = [[0.8], [0.6], [0.5], [0.4], [0.5]]
    assert detect_defences(race(close_order(5), gaps, status={4: "4"})) == []


def test_defence_running_at_the_flag_is_held() -> None:
    gaps = [[0.8], [0.6], [0.5]]
    [d] = detect_defences(race(close_order(3), gaps))
    assert d.outcome == "held"


def test_pass_location_finds_the_swap() -> None:
    # B at a steady 50 m/s; A 0.5 s behind, at 55 m/s: level after 250 m... plus
    # the 25 m B gains in the first 0.5 s -> A arrives at d at d/55 + 0.5,
    # B at d/50, equal at d = 0.5 / (1/50 - 1/55) = 275 m.
    d = np.linspace(0.0, 1000.0, 201)
    where = pass_location(d, d / 55.0 + 0.5, d, d / 50.0)
    assert where == pytest.approx(275.0, abs=1.0)


def test_pass_location_none_if_no_swap() -> None:
    d = np.linspace(0.0, 1000.0, 201)
    assert pass_location(d, d / 50.0 + 0.5, d, d / 50.0) is None
    assert pass_location([0.0], [0.0], d, d / 50.0) is None


def test_track_mapping_and_splits() -> None:
    assert normalise_location(" São Paulo ") == "sao paulo"
    assert tum_track("São Paulo") == "SaoPaulo"
    assert tum_track("Spa-Francorchamps") == "Spa"
    assert tum_track("Montréal") == "Montreal"
    assert tum_track("Marina Bay") is None
    assert split_of("Suzuka") == "held_out"
    assert split_of("Montreal") == "validation"
    assert split_of("Monza") == "train"
    assert split_of(None) == "no_track"


def test_rows_from_fastf1_style_laps() -> None:
    pd = pytest.importorskip("pandas")
    from f1rl.incidents import rows_from_laps

    laps = pd.DataFrame(
        {
            "Driver": ["VER", "HAM", "LEC"],
            "LapNumber": [3.0, 3.0, 3.0],
            "Position": [1.0, 2.0, float("nan")],
            "Time": pd.to_timedelta([300.0, 301.5, 303.0], unit="s"),
            "PitInTime": pd.to_timedelta([None, 290.0, None], unit="s"),
            "PitOutTime": pd.to_timedelta([None, None, None], unit="s"),
            "TrackStatus": ["1", "12", "1"],
            "Compound": ["MEDIUM", "HARD", "SOFT"],
            "TyreLife": [3.0, float("nan"), 1.0],
        }
    )
    ver, ham = rows_from_laps(laps)  # LEC has no position: dropped
    assert (ver.driver, ver.lap, ver.position, ver.end_time_s) == ("VER", 3, 1, 300.0)
    assert not ver.pitted and ham.pitted
    assert ham.track_status == "12" and ham.compound == "HARD"
    assert ver.tyre_life == 3.0 and np.isnan(ham.tyre_life)
