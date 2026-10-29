"""Whole-race model (`f1rl.models.race`) and the rule-based strategy."""

import itertools
import math
from dataclasses import replace

import numpy as np
import pytest

from f1rl.agents.strategy_baseline import FixedPlan, sample_rival_plan
from f1rl.config import (
    EXAMPLE_CAR,
    StrategyBaselineParams,
    StrategyParams,
    TireParams,
)
from f1rl.models.race import (
    CarState,
    LapPlan,
    LapTable,
    build_lap_table,
    car_lap,
    fuel_car,
    pass_probability,
    push_grip_fraction,
    race_laps,
    resolve_lap,
)

PARAMS = StrategyParams(grip_step=0.05, fuel_points=3)


def oval() -> tuple[float, np.typing.NDArray[np.float64]]:
    """A 2 km closed path: two 500 m straights and two 160 m-radius half circles."""
    ds = 5.0
    n_straight = 100
    n_turn = round(math.pi * 160 / ds)
    k = np.concatenate([np.zeros(n_straight), np.full(n_turn, 1 / 160)] * 2).astype(
        np.float64
    )
    return ds, k


@pytest.fixture(scope="module")
def table() -> LapTable:
    ds, k = oval()
    return build_lap_table(EXAMPLE_CAR, ds, k, PARAMS)


def car(**kw: object) -> CarState:
    base = CarState(0.0, "medium", 0.0, 0, 50.0, frozenset({"medium"}))
    return replace(base, **kw)  # type: ignore[arg-type]


def test_race_laps_exceed_305_km() -> None:
    assert race_laps(5000.0) == 62
    assert race_laps(5793.0) * 5793.0 > 305_000 > (race_laps(5793.0) - 1) * 5793.0


def test_fuel_slows_acceleration_only() -> None:
    heavy = fuel_car(EXAMPLE_CAR, 110.0)
    assert heavy.max_accel_g < EXAMPLE_CAR.max_accel_g
    assert heavy.max_lateral_g == EXAMPLE_CAR.max_lateral_g
    assert heavy.max_braking_g == EXAMPLE_CAR.max_braking_g


def test_lap_table_less_grip_and_more_fuel_is_slower(table: LapTable) -> None:
    assert table.lap_time_s(0.9, 0.0) > table.lap_time_s(1.0, 0.0)
    assert table.lap_time_s(1.0, 110.0) > table.lap_time_s(1.0, 0.0)
    # Interpolation sits between the grid points.
    a, b = table.lap_time_s(0.90, 20.0), table.lap_time_s(0.95, 20.0)
    assert b < table.lap_time_s(0.93, 20.0) < a


def test_push_levels_span_conserve_to_attack() -> None:
    assert push_grip_fraction(PARAMS, 0.0) == PARAMS.conserve_grip_fraction
    assert push_grip_fraction(PARAMS, 1.0) == PARAMS.attack_grip_fraction


def test_softer_is_quicker_but_wears_faster(table: LapTable) -> None:
    tire = TireParams()
    soft_t, soft = car_lap(
        PARAMS, tire, table, car(compound="soft"), LapPlan(0.5), 1.0, False
    )
    hard_t, hard = car_lap(
        PARAMS, tire, table, car(compound="hard"), LapPlan(0.5), 1.0, False
    )
    assert soft_t < hard_t
    assert soft.wear > hard.wear


def test_push_is_quicker_but_wears_more(table: LapTable) -> None:
    tire = TireParams()
    slow_t, slow = car_lap(PARAMS, tire, table, car(), LapPlan(0.0), 1.0, False)
    fast_t, fast = car_lap(PARAMS, tire, table, car(), LapPlan(1.0), 1.0, False)
    assert fast_t < slow_t
    assert fast.wear > slow.wear


def test_worn_tyres_and_dirty_air_are_slower(table: LapTable) -> None:
    tire = TireParams()
    fresh, _ = car_lap(PARAMS, tire, table, car(), LapPlan(0.5), 1.0, False)
    worn, _ = car_lap(PARAMS, tire, table, car(wear=0.7), LapPlan(0.5), 1.0, False)
    dirty, _ = car_lap(PARAMS, tire, table, car(), LapPlan(0.5), 1.0, True)
    assert worn > fresh
    assert dirty > fresh


def test_pit_stop_costs_pit_loss_and_fits_new_tyres(table: LapTable) -> None:
    tire = TireParams()
    old = car(wear=0.6, tyre_laps=20)
    stay, _ = car_lap(PARAMS, tire, table, old, LapPlan(0.5), 1.0, False)
    pit, after = car_lap(PARAMS, tire, table, old, LapPlan(0.5, "hard"), 1.0, False)
    assert after.compound == "hard"
    assert after.tyre_laps == 1
    assert after.stops == 1
    assert after.compounds_used == {"medium", "hard"}
    assert after.fuel_kg == pytest.approx(49.0)
    # New tyres win some time back but not the whole stop.
    assert stay < pit < stay + PARAMS.pit_loss_s


def test_pass_probability_grows_with_pace() -> None:
    assert pass_probability(PARAMS, 0.0) < 0.1
    assert pass_probability(PARAMS, PARAMS.pass_delta_s) == pytest.approx(0.5)
    assert pass_probability(PARAMS, 2.0) > 0.99


def test_slower_car_holds_up_a_quicker_one_that_cannot_pass() -> None:
    p = replace(PARAMS, pass_delta_s=5.0)
    order, end = resolve_lap(p, [0, 1], [0.0, 0.5], [90.0, 89.8], [False] * 2, [0.9])
    assert order == [0, 1]
    assert end[1] == pytest.approx(90.0 + p.min_gap_s)


def test_much_quicker_car_passes() -> None:
    order, end = resolve_lap(
        PARAMS, [0, 1], [0.0, 0.5], [92.0, 89.0], [False] * 2, [0.5]
    )
    assert order == [1, 0]
    assert end[1] == pytest.approx(89.5)
    assert end[0] >= end[1] + PARAMS.min_gap_s


def test_pit_stop_drops_a_car_back_freely() -> None:
    order, end = resolve_lap(
        PARAMS,
        [0, 1, 2],
        [0.0, 1.0, 2.0],
        [112.0, 90.0, 90.0],
        [True, False, False],
        [0.99],
    )
    assert order == [1, 2, 0]
    assert end[0] == pytest.approx(112.0)


def test_order_times_are_consistent() -> None:
    rng = np.random.default_rng(0)
    for _ in range(50):
        n = 8
        start = sorted(rng.uniform(0, 5, n).tolist())
        laps = rng.uniform(88, 92, n).tolist()
        pitted = (rng.random(n) < 0.2).tolist()
        order, end = resolve_lap(
            PARAMS, list(range(n)), start, laps, pitted, rng.random(32).tolist()
        )
        assert sorted(order) == list(range(n))
        for a, b in itertools.pairwise(order):
            assert end[b] >= end[a] + PARAMS.min_gap_s - 1e-9


def test_attack_skill_helps_our_car_pass() -> None:
    args = ([0, 1], [0.0, 0.5], [90.0, 89.4], [False] * 2, [0.6])
    assert resolve_lap(PARAMS, *args, ours=1)[0] == [0, 1]
    skilled = replace(PARAMS, attack_skill_s=1.0)
    assert resolve_lap(skilled, *args, ours=1)[0] == [1, 0]


def test_fixed_plan_stops_on_schedule() -> None:
    plan = FixedPlan(StrategyBaselineParams(("medium", "hard"), (0.5,)))
    assert plan.plan(PARAMS, car(), 10, 60).pit_to is None
    assert plan.plan(PARAMS, car(), 30, 60).pit_to == "hard"
    after = car(compound="hard", stops=1)
    assert plan.plan(PARAMS, after, 40, 60).pit_to is None


def test_fixed_plan_pits_early_on_worn_tyres() -> None:
    plan = FixedPlan(StrategyBaselineParams(("medium", "hard"), (0.5,), max_wear=0.8))
    assert plan.plan(PARAMS, car(wear=0.85), 10, 60).pit_to == "hard"
    worn = car(compound="hard", stops=1, wear=0.85)
    assert plan.plan(PARAMS, worn, 40, 60).pit_to == "hard"
    # Not with the flag in sight.
    assert plan.plan(PARAMS, worn, 58, 60).pit_to is None


def test_fixed_plan_rejects_bad_plans() -> None:
    with pytest.raises(ValueError):
        FixedPlan(StrategyBaselineParams(("medium", "hard"), ()))
    with pytest.raises(ValueError):
        FixedPlan(StrategyBaselineParams(("medium", "wet"), (0.5,)))


def test_rival_plans_are_seeded() -> None:
    base = StrategyBaselineParams()
    a = sample_rival_plan(np.random.default_rng(3), base, 5)
    b = sample_rival_plan(np.random.default_rng(3), base, 5)
    assert a == b
    assert 0 <= a.params.push_index < 5
