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
    """Tunables for `DriverEnv`'s observation/reward shaping.

    Simulation design choices, not measured F1 quantities -- see `SimParams`.
    """

    lookahead_m: float = 50.0  # how far ahead the "next corner" observation looks.
    curvature_norm_per_m: float = 0.1  # curvature (1/m) that normalizes obs to +-1.
    drift_gain_m_s_per_g: float = 5.0  # lateral drift speed per g of grip deficit.
    recovery_rate_m_s: float = 3.0  # lateral offset recovered per second within grip.
    off_track_penalty: float = 1.0  # reward subtracted, episode ends, when exceeded.
