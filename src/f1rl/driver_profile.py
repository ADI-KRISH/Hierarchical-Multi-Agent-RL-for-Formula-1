"""What the Strategy Agent needs to know about OUR Driver Agent, measured.

The strategist races lap by lap (`envs.strategy_env`), so it does not run the
driver itself. Instead `scripts/measure_driver.py` drives the trained driver in the
full sim and records:

- **pace**: per track, its flying-lap time at several grip levels (push level x
  tyre wear), as a gap to the g-g limit lap on the optimal line at the same grip;
- **attack**: one-lap battles behind slower and quicker cars: did it pass?
- **defend**: one-lap battles ahead of attackers: was it passed?

and fits the overtaking odds the race sim uses: P(pass in a lap) =
sigmoid((pace advantage - delta_s) / slope_s), the advantage being the lap-time
difference between the two cars. This module holds the profile, its JSON form and
the fit (pure functions).
"""

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class PassCurve:
    """P(pass in a lap) = sigmoid((advantage_s - delta_s) / slope_s)."""

    delta_s: float
    slope_s: float
    battles: int = 0

    def probability(self, advantage_s: float) -> float:
        x = (advantage_s - self.delta_s) / self.slope_s
        return 1.0 / (1.0 + math.exp(-max(-60.0, min(60.0, x))))


@dataclass(frozen=True)
class DriverProfile:
    """A driver model's measured pace gap per track and its race-craft odds."""

    model: str
    # Per track: (grip multipliers, gap to the g-g limit lap at that grip), grips
    # ascending. Lap time = limit lap(grip) * (1 + gap).
    pace: dict[str, tuple[tuple[float, ...], tuple[float, ...]]]
    # Our car attacking a car ahead; a rival attacking our car.
    attack: PassCurve
    defend: PassCurve
    # Shares of measured battles with contact / our car off the road.
    contact_rate: float = 0.0
    off_rate: float = 0.0
    meta: dict[str, Any] = field(default_factory=dict)

    def gap(self, track: str, grip: float) -> float | None:
        """Pace gap on ``track`` at ``grip`` (linear between measured grips, held
        flat past the ends), or None if the track was not measured."""
        if track not in self.pace:
            return None
        grips, gaps = self.pace[track]
        return float(np.interp(grip, grips, gaps))

    def mean_gap(self) -> float:
        return float(np.mean([np.mean(g) for _, g in self.pace.values()]))


def save_profile(profile: DriverProfile, path: Path) -> None:
    d = {
        "model": profile.model,
        "pace": {
            t: {"grips": list(g), "gaps": list(v)}
            for t, (g, v) in sorted(profile.pace.items())
        },
        "attack": vars(profile.attack),
        "defend": vars(profile.defend),
        "contact_rate": profile.contact_rate,
        "off_rate": profile.off_rate,
        "meta": profile.meta,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(d, indent=2), encoding="utf-8")


def load_profile(path: Path) -> DriverProfile:
    d = json.loads(path.read_text(encoding="utf-8"))
    return DriverProfile(
        model=d["model"],
        pace={t: (tuple(v["grips"]), tuple(v["gaps"])) for t, v in d["pace"].items()},
        attack=PassCurve(**d["attack"]),
        defend=PassCurve(**d["defend"]),
        contact_rate=float(d.get("contact_rate", 0.0)),
        off_rate=float(d.get("off_rate", 0.0)),
        meta=d.get("meta", {}),
    )


def fit_pass_curve(
    advantage_s: list[float],
    passed: list[bool],
    prior: PassCurve,
    prior_weight: float = 1.0,
) -> PassCurve:
    """Logistic fit of ``passed`` against ``advantage_s``: maximum likelihood
    (Nelder-Mead on the delta and log slope). ``prior`` enters as ``prior_weight``
    pseudo-battles at advantages prior.delta_s +/- 2 prior.slope_s that went its
    way, which keeps the fit finite when every battle was won (or lost).
    """
    x = list(advantage_s)
    y = [1.0 if p else 0.0 for p in passed]
    w = [1.0] * len(x)
    for sign in (-1.0, 1.0):
        adv = prior.delta_s + 2 * sign * prior.slope_s
        x.append(adv)
        y.append(prior.probability(adv))
        w.append(prior_weight)
    xa, ya, wa = np.array(x), np.array(y), np.array(w)

    def nll(theta: np.typing.NDArray[np.float64]) -> float:
        delta, log_slope = theta
        z = (xa - delta) / math.exp(log_slope)
        # -log sigmoid(z) = logaddexp(0, -z): stable for any z.
        loss = wa * (ya * np.logaddexp(0.0, -z) + (1 - ya) * np.logaddexp(0.0, z))
        return float(np.sum(loss))

    start = np.array([prior.delta_s, math.log(prior.slope_s)])
    res = minimize(nll, start, method="Nelder-Mead", options={"xatol": 1e-6})
    delta, log_slope = res.x
    # Odds must not swing from 0 to 1 within a hundredth of a second.
    slope = max(0.01, math.exp(float(log_slope)))
    return PassCurve(float(delta), slope, len(advantage_s))


def smooth_pace(profile: DriverProfile) -> DriverProfile:
    """The profile with each track's pace gap replaced by a straight-line fit over
    grip (least squares), kept at the measured range's two ends. Single laps scatter
    by about a percent, enough to make a lower push level look quicker on some
    tracks; a line keeps less grip = slower without inventing a shape."""
    pace = {}
    for track, (grips, gaps) in profile.pace.items():
        g = np.asarray(grips, dtype=np.float64)
        if len(g) < 2 or np.ptp(g) == 0:
            pace[track] = (grips, gaps)
            continue
        slope, icept = np.polyfit(g, np.asarray(gaps, dtype=np.float64), 1)
        ends = (float(g.min()), float(g.max()))
        pace[track] = (ends, tuple(float(icept + slope * x) for x in ends))
    meta = {**profile.meta, "pace": "straight-line fit of the measured laps"}
    return replace(profile, pace=pace, meta=meta)
