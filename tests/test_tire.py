import itertools

import pytest

from f1rl.config import TireParams
from f1rl.models.tire import tire_grip, tire_wear_step

P = TireParams()


def test_new_tyres_give_full_grip_and_worn_ones_less() -> None:
    assert tire_grip(P, 0.0) == 1.0
    grips = [tire_grip(P, w / 10) for w in range(11)]
    assert all(a > b for a, b in itertools.pairwise(grips))
    assert grips[-1] == pytest.approx(1 - P.loss_linear - P.loss_cliff)


def test_wear_grows_faster_than_usage() -> None:
    full = tire_wear_step(P, 0.0, 1.0, 10.0)
    half = tire_wear_step(P, 0.0, 0.5, 10.0)
    assert full == pytest.approx(10.0 / P.life_s)
    assert half == pytest.approx(full * 0.5**P.wear_exponent)
    assert tire_wear_step(P, 0.0, 0.0, 10.0) == 0.0


def test_wear_is_capped_at_one() -> None:
    assert tire_wear_step(P, 0.99, 1.0, 1e6) == 1.0
