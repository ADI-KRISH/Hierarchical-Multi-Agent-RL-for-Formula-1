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
