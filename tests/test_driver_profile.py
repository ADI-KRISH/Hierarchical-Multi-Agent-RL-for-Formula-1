"""Measured driver profile (`f1rl.driver_profile`) and its use in the race model."""

from pathlib import Path

import numpy as np
import pytest

from f1rl.config import StrategyParams
from f1rl.driver_profile import (
    DriverProfile,
    PassCurve,
    fit_pass_curve,
    load_profile,
    save_profile,
)
from f1rl.models.race import resolve_lap

PRIOR = PassCurve(0.6, 0.25)


def test_pass_curve_is_a_sigmoid() -> None:
    c = PassCurve(0.5, 0.2)
    assert c.probability(0.5) == pytest.approx(0.5)
    assert c.probability(-5.0) < 0.01 < 0.99 < c.probability(5.0)
    assert c.probability(1e9) == pytest.approx(1.0)


def test_fit_recovers_a_known_curve() -> None:
    rng = np.random.default_rng(0)
    true = PassCurve(0.4, 0.5)
    x = rng.uniform(-3, 3, 3000)
    y = [bool(rng.random() < true.probability(v)) for v in x]
    fit = fit_pass_curve(list(x), y, PRIOR)
    assert fit.delta_s == pytest.approx(0.4, abs=0.1)
    assert fit.slope_s == pytest.approx(0.5, abs=0.1)
    assert fit.battles == 3000


def test_fit_stays_finite_when_every_battle_goes_one_way() -> None:
    fit = fit_pass_curve([0.0, 1.0, 3.0], [True] * 3, PRIOR)
    assert np.isfinite(fit.delta_s)
    assert fit.slope_s >= 0.01
    assert fit.probability(1.0) > 0.5


def test_profile_round_trip_and_gap(tmp_path: Path) -> None:
    prof = DriverProfile(
        model="m.zip",
        pace={"Monza": ((0.85, 0.95), (0.03, 0.01))},
        attack=PassCurve(0.3, 0.4, 10),
        defend=PassCurve(0.9, 0.3, 12),
        contact_rate=0.01,
    )
    save_profile(prof, tmp_path / "p.json")
    back = load_profile(tmp_path / "p.json")
    assert back == prof
    assert back.gap("Monza", 0.90) == pytest.approx(0.02)
    assert back.gap("Monza", 1.2) == pytest.approx(0.01)  # flat past the end
    assert back.gap("Spa", 0.9) is None
    assert back.mean_gap() == pytest.approx(0.02)


def test_measured_curves_drive_our_battles() -> None:
    p = StrategyParams()
    # We (car 1) are caught up behind car 0 but would not get ahead on a free lap.
    args = ([0, 1], [0.0, 0.2], [90.0, 90.0], [False, False], [0.3])
    assert resolve_lap(p, *args, ours=1)[0] == [0, 1]
    keen = PassCurve(-1.0, 0.2)  # passes even at equal pace
    assert resolve_lap(p, *args, ours=1, our_attack=keen)[0] == [1, 0]
    # Defending: a rival behind us with a soft defend curve gets by.
    assert resolve_lap(p, *args, ours=0, our_defend=keen)[0] == [1, 0]
    stout = PassCurve(5.0, 0.2)
    assert resolve_lap(p, *args, ours=0, our_defend=stout)[0] == [0, 1]
