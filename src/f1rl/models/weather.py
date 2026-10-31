"""Rain for the race model: pure functions, state in, state out.

Track wetness runs from 0 (dry) to 1 (soaked). It moves toward a shower's strength
while it rains and dries a fixed amount a lap after. Each tyre's grip and wear
depend on it (`WeatherParams`, uncalibrated design values).
"""

from dataclasses import dataclass

import numpy as np

from f1rl.config import COMPOUNDS, WeatherParams


def tyre_class(compound: str) -> str:
    """ "slick", "intermediate" or "wet"."""
    return "slick" if compound in COMPOUNDS else compound


def wet_grip(params: WeatherParams, compound: str, wetness: float) -> float:
    """Grip multiplier of ``compound`` on a track this wet."""
    w = min(1.0, max(0.0, wetness))
    kind = tyre_class(compound)
    if kind == "slick":
        return 1.0 - params.slick_loss * w
    if kind == "intermediate":
        return params.inter_peak - params.inter_slope * abs(w - params.inter_best)
    return params.wet_peak - params.wet_slope * (1.0 - w)


def wet_wear(params: WeatherParams, compound: str, wetness: float) -> float:
    """Wear-rate multiplier: wet-weather tyres overheat as the track dries."""
    kind = tyre_class(compound)
    if kind == "slick":
        return 1.0
    below = (
        params.inter_overheat_below
        if kind == "intermediate"
        else params.wet_overheat_below
    )
    dry_share = max(0.0, below - max(0.0, wetness)) / below
    return 1.0 + (params.overheat_wear - 1.0) * dry_share


@dataclass(frozen=True)
class RaceWeather:
    """One race's weather, lap by lap (index = lap, 0-based)."""

    wetness: tuple[float, ...]  # during each lap
    raining: tuple[bool, ...]
    forecast_noise: tuple[float, ...]  # one standard normal per lap

    @property
    def wet(self) -> bool:
        return any(self.raining)

    def at(self, lap: int) -> float:
        return self.wetness[min(max(lap, 0), len(self.wetness) - 1)]

    def forecast(self, params: WeatherParams, lap: int, ahead: int) -> float:
        """Forecast wetness ``ahead`` laps after ``lap``, with a fixed error per
        lap that grows with the horizon (the same race gives the same forecast)."""
        noise = self.forecast_noise[min(lap, len(self.forecast_noise) - 1)]
        sd = params.forecast_sd * ahead / 3.0
        return float(np.clip(self.at(lap + ahead) + sd * noise, 0.0, 1.0))


def dry_weather(laps: int) -> RaceWeather:
    return RaceWeather((0.0,) * laps, (False,) * laps, (0.0,) * laps)


def race_weather(
    params: WeatherParams, rng: np.random.Generator, laps: int, rain: bool
) -> RaceWeather:
    """Draw a race's weather: one shower when ``rain``, else dry. Always takes the
    same number of draws from ``rng`` for a given ``rain``."""
    noise = tuple(float(x) for x in rng.standard_normal(laps))
    if not rain:
        return RaceWeather((0.0,) * laps, (False,) * laps, noise)
    start = round(rng.uniform(*params.start_range) * laps)
    length = max(1, round(rng.uniform(*params.length_range) * laps))
    strength = float(rng.uniform(*params.intensity_range))
    wet, raining, w = [], [], 0.0
    for lap in range(laps):
        on = start <= lap < start + length
        w = (
            w + params.wet_rate * (strength - w)
            if on
            else max(0.0, w - params.dry_rate)
        )
        wet.append(round(w, 4))
        raining.append(on)
    return RaceWeather(tuple(wet), tuple(raining), noise)


def rule_tyre_class(
    params: WeatherParams, current: str, wetness: float, offset: float = 0.0
) -> str:
    """The tyre class a rule-based team wants for this wetness, with hysteresis:
    between to_slick and to_inter it keeps what it has."""
    kind = tyre_class(current)
    if wetness > params.to_wet + offset:
        return "wet"
    if wetness > params.to_inter + offset:
        return (
            "intermediate" if kind != "wet" or wetness < params.to_wet - 0.1 else "wet"
        )
    if wetness < params.to_slick + offset:
        return "slick"
    return kind if kind != "wet" else "intermediate"
