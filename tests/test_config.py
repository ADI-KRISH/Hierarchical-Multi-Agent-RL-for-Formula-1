"""Phase 0 smoke test: the package imports and its config surface holds together."""

import dataclasses

import pytest

from f1rl import __version__
from f1rl.config import GRAVITY_M_S2, CarParams, SimParams, TrackParams


def test_package_imports() -> None:
    assert __version__ == "0.1.0"


def test_gravity_is_the_si_standard_value() -> None:
    # Exact by definition, so exact equality is the right assertion here.
    assert GRAVITY_M_S2 == 9.80665


@pytest.mark.parametrize("params", [CarParams, TrackParams, SimParams])
def test_param_dataclasses_are_frozen(params: type) -> None:
    """Frozen params are what make a run reproducible from its logged config."""
    assert dataclasses.is_dataclass(params)
    assert params.__dataclass_params__.frozen  # type: ignore[attr-defined]


def test_sim_params_require_an_explicit_seed() -> None:
    """Determinism rule: no env or run may fall back to an implicit seed."""
    with pytest.raises(TypeError):
        SimParams()  # type: ignore[call-arg]

    assert SimParams(seed=0).seed == 0
