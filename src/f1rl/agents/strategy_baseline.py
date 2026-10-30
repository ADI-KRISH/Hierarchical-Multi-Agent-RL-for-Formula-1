"""Rule-based race strategy: the plan a team writes before the start.

Start on the first compound of ``stints``, stop after each share of the race in
``stop_fractions`` for the next one, hold one push level all race, and come in
early if the tyres wear past ``max_wear``. The Strategy Agent must beat this; the
rivals in `StrategyEnv` run variants of it.
"""

from dataclasses import dataclass

import numpy as np

from f1rl.config import COMPOUNDS, StrategyBaselineParams, StrategyParams
from f1rl.models.race import CarState, LapPlan


@dataclass(frozen=True)
class FixedPlan:
    """A fixed-window strategy for one car."""

    params: StrategyBaselineParams

    def __post_init__(self) -> None:
        p = self.params
        if len(p.stints) != len(p.stop_fractions) + 1:
            raise ValueError("need one stop fraction per change of compound")
        if any(c not in COMPOUNDS for c in p.stints):
            raise ValueError(f"unknown compound in {p.stints}")

    @property
    def start_compound(self) -> str:
        return self.params.stints[0]

    def plan(
        self, strategy: StrategyParams, car: CarState, lap: int, total_laps: int
    ) -> LapPlan:
        """The plan for lap ``lap`` (0-based) of ``total_laps``."""
        p = self.params
        push = strategy.push_levels[p.push_index]
        stint = car.stops
        laps_left = total_laps - lap
        if stint < len(p.stop_fractions) and lap >= round(
            p.stop_fractions[stint] * total_laps
        ):
            return LapPlan(push, p.stints[stint + 1])
        if car.wear > p.max_wear and laps_left > 3:
            # Worn out early: the next planned compound, or the hardest one used
            # if the plan has run out (any compound keeps the two-compound rule).
            nxt = p.stints[min(stint + 1, len(p.stints) - 1)]
            return LapPlan(push, nxt)
        return LapPlan(push)


#: Variants the rivals draw from: (stints, stop-fraction ranges).
RIVAL_TEMPLATES: tuple[tuple[tuple[str, ...], tuple[tuple[float, float], ...]], ...] = (
    (("medium", "hard"), ((0.35, 0.55),)),
    (("hard", "medium"), ((0.50, 0.65),)),
    (("soft", "hard"), ((0.25, 0.40),)),
    (("soft", "medium", "hard"), ((0.22, 0.32), (0.55, 0.68))),
)


def sample_rival_plan(
    rng: np.random.Generator, base: StrategyBaselineParams, n_push_levels: int
) -> FixedPlan:
    """A rival's strategy: a random template, stop laps and push level."""
    stints, ranges = RIVAL_TEMPLATES[int(rng.integers(len(RIVAL_TEMPLATES)))]
    fractions = tuple(float(rng.uniform(lo, hi)) for lo, hi in ranges)
    push = int(np.clip(base.push_index + rng.integers(-1, 2), 0, n_push_levels - 1))
    return FixedPlan(
        StrategyBaselineParams(
            stints=stints,
            stop_fractions=fractions,
            push_index=push,
            max_wear=base.max_wear,
        )
    )


def baseline_rival(
    rng: np.random.Generator, base: StrategyBaselineParams, jitter: float
) -> FixedPlan:
    """A rival running the rule-based plan, each stop moved by up to +/- ``jitter``
    of the race distance (kept in order and inside the race)."""
    fractions = sorted(
        float(np.clip(f + rng.uniform(-jitter, jitter), 0.05, 0.95))
        for f in base.stop_fractions
    )
    return FixedPlan(
        StrategyBaselineParams(
            stints=base.stints,
            stop_fractions=tuple(fractions),
            push_index=base.push_index,
            max_wear=base.max_wear,
        )
    )
