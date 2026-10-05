"""Calibration constants and tunable parameters for the simulation.

Two rules govern this module:

1. **No invented F1 numbers.** Every car/track value either carries a citation in
   its comment or is calibrated against FastF1 data in ``data/``. Physical
   constants cite their standard. Values that are not yet calibrated are left
   *required* (no default) so a caller has to supply them explicitly rather than
   silently inheriting a guess.
2. **No hardcoding elsewhere.** A new tunable value gets a field here (or in an
   experiment YAML under ``configs/``), never a literal buried in the env.

The dataclasses are frozen: parameters are fixed for the lifetime of a run, which
is what makes a run reproducible from its logged config plus its seed.
"""

from dataclasses import dataclass
from typing import Final

#: Track width (m) where no data gives one -- FastF1 has no width channel. 12 m
#: is the minimum width the FIA sets for new permanent circuits (FIA International
#: Sporting Code, Appendix O), so a real track is at least this wide.
DEFAULT_TRACK_WIDTH_M: Final = 12.0

# Standard acceleration of gravity, BIPM SI Brochure (9th ed.), exact by definition.
GRAVITY_M_S2: Final = 9.80665


@dataclass(frozen=True)
class CarParams:
    """Point-mass car model parameters.

    Every field is required and must be calibrated (phase 1) -- see the module
    docstring. Record the source alongside each value at the call site.
    """

    mass_kg: float
    max_lateral_g: float
    max_accel_g: float
    max_braking_g: float
    max_speed_ms: float


@dataclass(frozen=True)
class TrackParams:
    """Parameterized track geometry.

    A lap is a sequence of constant-curvature segments; the concrete segment list
    is built by the track model (phase 1), not stored here.
    """

    width_m: float
    total_length_m: float


#: Illustrative F1 point-mass car for phase-1 sanity checks. Figures are the
#: approximate, widely-reported values from F1 technical media/regulations
#: (FIA minimum weight; Pirelli/Brembo/team technical explainers for grip and
#: top speed) -- precise per-corner calibration against FastF1 telemetry is
#: future work, not yet done.
EXAMPLE_CAR: Final = CarParams(
    mass_kg=798.0,  # FIA F1 Technical Regulations: minimum weight incl. driver.
    max_lateral_g=5.0,  # Peak sustained cornering g reported for modern F1 cars.
    max_accel_g=1.5,  # Peak longitudinal acceleration reported for F1 cars.
    max_braking_g=5.0,  # Peak braking deceleration reported for F1 cars.
    max_speed_ms=97.0,  # ~350 km/h, reported top speed on DRS-assisted straights.
)


@dataclass(frozen=True)
class SimParams:
    """Integration and episode settings.

    These are simulation choices rather than measured F1 quantities, so they carry
    defaults. ``seed`` is explicit and mandatory everywhere: no unseeded RNG and no
    wall-clock randomness anywhere in ``src/``.
    """

    seed: int
    dt_s: float = 0.02  # 50 Hz control loop.
    max_episode_steps: int = 10_000


@dataclass(frozen=True)
class DriverEnvParams:
    """Tunables for `DriverEnv`'s dynamics, observation, and reward shaping.

    Simulation design choices, not measured F1 quantities -- see `SimParams`.
    """

    # Edges (m) of the lookahead windows: the observation reports the slowest
    # corner speed limit in 0-25 m, 25-50 m, ... ahead. The farthest must cover a
    # full-speed stop for a slow hairpin: (97^2 - 20^2) / (2 * 5 g) ~ 92 m for
    # `EXAMPLE_CAR`.
    lookahead_m: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0)
    drift_gain_m_s_per_g: float = 5.0  # forced lateral drift speed per g over grip.
    # Steering turns the velocity vector at most this far off the track direction,
    # as a lateral/forward speed ratio (0.1 ~ 5.7 deg) -- scaled by the fraction of
    # lateral grip the corner leaves unused, so a stationary car can't move sideways
    # and steering can't out-steer physics.
    max_heading_ratio: float = 0.1
    off_track_penalty: float = 1.0  # reward subtracted, episode ends, when exceeded.
    # Reward subtracted per simulated second. Progress pays 1.0 per lap whatever the
    # pace, so this is what makes a faster lap score higher.
    time_penalty_per_s: float = 0.01
    # Reset draws the starting lateral offset uniformly from [0, this], from the
    # env's seeded RNG -- so episodes differ but stay reproducible.
    start_offset_max_m: float = 1.0


@dataclass(frozen=True)
class BaselineParams:
    """Tunables for the rule-based baseline driver (phase 3).

    Design choices for a scripted driver, not measured F1 quantities.
    """

    # A cautious scripted driver, not an optimal one: it plans corner speeds for
    # this fraction of the car's lateral grip...
    corner_grip_margin: float = 0.90
    # ...and places its braking points as if the car could only brake at this
    # fraction of its real capacity, so it brakes early.
    braking_margin: float = 0.80
    # Each episode the driver's margins are jittered by Gaussian noise of this
    # std-dev, kept within [margin_floor, 1.0] (1.0 is the car's real limit):
    # lap-to-lap confidence varies, which is what makes 20 runs differ. Drawn
    # from the driver's seeded RNG.
    margin_jitter_std: float = 0.02
    margin_floor: float = 0.5  # jittered margins never drop below this.


@dataclass(frozen=True)
class AnalyticsParams:
    """How lap analytics split a lap and sample telemetry -- reporting choices."""

    # Tighter than this (radius under 500 m) counts as a corner...
    corner_curvature_per_m: float = 0.002
    # ...unless shorter than this: real-circuit curvature has short noise kinks.
    min_corner_m: float = 10.0
    telemetry_every_n_steps: int = 5  # 10 Hz telemetry at the 50 Hz control loop.
    map_step_m: float = 5.0  # spacing of the reconstructed track-map points.
