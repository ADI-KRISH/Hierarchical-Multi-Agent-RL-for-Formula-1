"""Whole-race model for the Strategy Agent: pure functions, state in, state out.

One step is one lap for every car. A lap's time comes from the stand-in driver's
lap-time table (`LapTable`): the limit lap on the optimal line at the grip the
driver plans for (push level x compound x tyre wear x dirty air), with the car
slower to accelerate while it carries fuel. Cars then meet on track: a quicker car
caught behind a slower one passes with odds that grow with its pace advantage, or
crosses the line just behind it (`resolve_lap`).
"""

import math
from collections.abc import Callable
from dataclasses import dataclass, replace

import numpy as np

from f1rl.config import (
    COMPOUNDS,
    RACE_MIN_DISTANCE_M,
    CarParams,
    StrategyParams,
    TireParams,
)
from f1rl.driver_profile import PassCurve
from f1rl.models.raceline import gg_lap_time_s
from f1rl.models.tire import tire_grip, tire_wear_step

Array = np.typing.NDArray[np.float64]


def race_laps(lap_length_m: float) -> int:
    """Laps of a Grand Prix on a lap this long: the fewest that exceed 305 km."""
    return math.floor(RACE_MIN_DISTANCE_M / lap_length_m) + 1


def fuel_car(car: CarParams, fuel_kg: float) -> CarParams:
    """The car carrying ``fuel_kg``: the same driving force over more mass, so it
    accelerates less hard. Cornering and braking are friction limits (in g), which
    do not change with mass in the point-mass model."""
    scale = car.mass_kg / (car.mass_kg + max(0.0, fuel_kg))
    return replace(
        car, mass_kg=car.mass_kg + fuel_kg, max_accel_g=car.max_accel_g * scale
    )


def grip_car(car: CarParams, grip: float) -> CarParams:
    """The car with every friction limit scaled by ``grip`` (planned fraction x
    compound x wear x dirty air)."""
    return replace(
        car,
        max_lateral_g=car.max_lateral_g * grip,
        max_braking_g=car.max_braking_g * grip,
        max_accel_g=car.max_accel_g * min(1.0, grip),
    )


@dataclass(frozen=True, eq=False)
class LapTable:
    """Stand-in driver: lap time (s) over grip multiplier x fuel load, read with
    bilinear interpolation. Built once per track from the optimal line
    (`build_lap_table`); a measured table from the trained driver can replace it.
    """

    grips: Array  # ascending
    fuels_kg: Array  # ascending
    times_s: Array  # (len(grips), len(fuels_kg))

    def lap_time_s(self, grip: float, fuel_kg: float) -> float:
        g = float(np.clip(grip, self.grips[0], self.grips[-1]))
        f = float(np.clip(fuel_kg, self.fuels_kg[0], self.fuels_kg[-1]))
        i = int(np.clip(np.searchsorted(self.grips, g) - 1, 0, len(self.grips) - 2))
        j = int(
            np.clip(np.searchsorted(self.fuels_kg, f) - 1, 0, len(self.fuels_kg) - 2)
        )
        tg = (g - self.grips[i]) / (self.grips[i + 1] - self.grips[i])
        tf = (f - self.fuels_kg[j]) / (self.fuels_kg[j + 1] - self.fuels_kg[j])
        t = self.times_s
        low = t[i, j] * (1 - tf) + t[i, j + 1] * tf
        high = t[i + 1, j] * (1 - tf) + t[i + 1, j + 1] * tf
        return float(low * (1 - tg) + high * tg)


def build_lap_table(
    car: CarParams,
    ds: float | Array,
    kappa: Array,
    params: StrategyParams,
    gap: Callable[[float], float] | None = None,
) -> LapTable:
    """Lap-time table on the closed path (``ds``, ``kappa``), usually the track's
    optimal line: the g-g limit lap times 1 + ``gap(grip)``, a driver's measured
    pace gap (`f1rl.driver_profile`), or the stand-in's fixed ``driver_gap``."""
    n = round((params.grip_max - params.grip_min) / params.grip_step) + 1
    grips = np.linspace(params.grip_min, params.grip_max, n)
    fuels = np.linspace(0.0, params.start_fuel_kg, params.fuel_points)
    times = np.empty((n, len(fuels)))
    for j, fuel in enumerate(fuels):
        heavy = fuel_car(car, float(fuel))
        for i, g in enumerate(grips):
            limit = gg_lap_time_s(grip_car(heavy, float(g)), ds, kappa)
            extra = params.driver_gap if gap is None else gap(float(g))
            times[i, j] = limit * (1.0 + extra)
    return LapTable(grips, fuels, times)


def push_grip_fraction(params: StrategyParams, push: float) -> float:
    """Share of the grip the driver plans for at push 0 (conserve) .. 1 (attack)."""
    lo, hi = params.conserve_grip_fraction, params.attack_grip_fraction
    return lo + (hi - lo) * min(1.0, max(0.0, push))


@dataclass(frozen=True)
class CarState:
    """One car's race at a lap boundary."""

    time_s: float  # race time when it last crossed the line
    compound: str
    wear: float
    tyre_laps: int
    fuel_kg: float
    compounds_used: frozenset[str]
    stops: int = 0
    pace: float = 1.0  # car performance: multiplies its grip


@dataclass(frozen=True)
class LapPlan:
    """What a car does on its next lap."""

    push: float
    pit_to: str | None = None  # compound to fit at the line, or stay out


def car_lap(
    params: StrategyParams,
    tire: TireParams,
    table: LapTable,
    car: CarState,
    plan: LapPlan,
    burn_kg: float,
    dirty_air: bool,
) -> tuple[float, CarState]:
    """A car's free lap (no traffic): (lap time incl. any stop, state after it).

    A stop is made at the start of the lap (the pit lane crosses the line): the new
    tyres run this lap and ``pit_loss_s`` is added to it.
    """
    pit = plan.pit_to is not None
    if plan.pit_to is not None:
        if plan.pit_to not in COMPOUNDS:
            raise KeyError(f"unknown compound {plan.pit_to!r}")
        car = replace(
            car,
            compound=plan.pit_to,
            wear=0.0,
            tyre_laps=0,
            compounds_used=car.compounds_used | {plan.pit_to},
            stops=car.stops + 1,
        )
    comp = params.compound(car.compound)
    planned = push_grip_fraction(params, plan.push)
    grip = planned * comp.grip * tire_grip(tire, car.wear) * car.pace
    if dirty_air:
        grip *= 1.0 - params.dirty_air_loss
    lap = table.lap_time_s(grip, car.fuel_kg)
    life = comp.life_s / params.push_wear_ratio ** (plan.push - 0.5)
    wear = tire_wear_step(replace(tire, life_s=life), car.wear, 1.0, lap)
    after = replace(
        car,
        wear=wear,
        tyre_laps=car.tyre_laps + 1,
        fuel_kg=max(0.0, car.fuel_kg - burn_kg),
    )
    return lap + (params.pit_loss_s if pit else 0.0), after


def pass_probability(params: StrategyParams, advantage_s: float) -> float:
    """Odds that a car ``advantage_s`` quicker than the car ahead gets by in a lap."""
    x = (advantage_s - params.pass_delta_s) / params.pass_slope_s
    return 1.0 / (1.0 + math.exp(-x))


def resolve_lap(
    params: StrategyParams,
    order: list[int],
    start_s: list[float],
    lap_s: list[float],
    pitted: list[bool],
    draws: list[float],
    ours: int | None = None,
    our_attack: PassCurve | None = None,
    our_defend: PassCurve | None = None,
) -> tuple[list[int], list[float]]:
    """Cars meet on track: (new running order, each car's time at the line).

    ``order`` is the running order at the start of the lap (car indices), and
    ``start_s`` / ``lap_s`` / ``pitted`` are per car. A car that would reach the
    line within ``min_gap_s`` of the car in front either passes it, with
    `pass_probability` of its pace advantage (one uniform draw from ``draws`` per car
    per attempt, cycled), or crosses ``min_gap_s`` behind it. A car that stopped is
    passed and passes freely: the pit lane is a separate road. ``ours`` gets
    ``attack_skill_s`` when it attacks and costs attackers ``defend_skill_s``.

    With a measured curve for our driver (``our_attack`` when ``ours`` attacks,
    ``our_defend`` when a car attacks ``ours``, from `f1rl.driver_profile`), that
    curve's odds apply to any car that has caught the other (within ``min_gap_s``
    at the line): it was measured from battles that start that close, so it
    already says whether the quicker car gets by in a lap.
    """
    end = [s + t for s, t in zip(start_s, lap_s, strict=True)]
    new: list[int] = []
    k = 0
    for c in order:
        pos = len(new)
        while pos > 0:
            a = new[pos - 1]
            if end[c] >= end[a] + params.min_gap_s:
                break
            if pitted[a] or pitted[c]:
                free = end[c] < end[a]
            else:
                adv = lap_s[a] - lap_s[c]
                curve = our_attack if c == ours else our_defend if a == ours else None
                if c == ours:
                    adv += params.attack_skill_s
                if a == ours:
                    adv -= params.defend_skill_s
                draw = draws[k % len(draws)]
                k += 1
                if curve is not None:
                    free = draw < curve.probability(adv)
                else:
                    free = end[c] < end[a] and draw < pass_probability(params, adv)
            if not free:
                end[c] = end[a] + params.min_gap_s
                break
            end[a] = max(end[a], end[c] + params.min_gap_s)
            pos -= 1
        new.insert(pos, c)
        # A car pushed back by a pass may now sit on the one behind it.
        for i in range(pos + 1, len(new)):
            prev, cur = new[i - 1], new[i]
            end[cur] = max(end[cur], end[prev] + params.min_gap_s)
    return new, end
