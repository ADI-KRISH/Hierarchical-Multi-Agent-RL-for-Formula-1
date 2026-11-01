"""Tyre wear and grip: pure functions, state in, state out (parameters in `TireParams`).

Wear is a number in [0, 1] (new .. gone). Working the tyres harder wears them faster
and worn tyres give less grip, so pushing and conserving have a real price.
"""

from f1rl.config import TireParams


def tire_grip(params: TireParams, wear: float) -> float:
    """Multiplier on every grip limit (1.0 = new tyres)."""
    w = min(1.0, max(0.0, wear))
    return float(
        1.0 - params.loss_linear * w - params.loss_cliff * w**params.cliff_power
    )


def tire_wear_step(params: TireParams, wear: float, usage: float, dt_s: float) -> float:
    """Wear after ``dt_s`` seconds at friction ``usage`` (0..1)."""
    rate = max(0.0, usage) ** params.wear_exponent / params.life_s
    return float(min(1.0, wear + rate * dt_s))
