import math
import urllib.error

import numpy as np
import pytest

from f1rl.envs.racetrack import TUM_HELD_OUT, TUM_TRACKS, load_tum_track
from racetrack_fixtures import stadium_track


def test_stadium_length_and_total_turn() -> None:
    track = stadium_track()
    assert track.length_m == pytest.approx(2 * 600 + 2 * math.pi * 100, rel=1e-3)
    # Counter-clockwise lap: the heading turns left by exactly one revolution.
    assert float(np.sum(track.kappa) * track.ds_m) == pytest.approx(
        2 * math.pi, rel=1e-3
    )


def test_curvature_matches_the_arcs_and_straights() -> None:
    track = stadium_track()
    assert np.max(track.kappa) == pytest.approx(1 / 100, rel=0.05)
    assert np.sum(np.abs(track.kappa) < 1e-4) > 0.4 * track.n  # the straights


def test_kappa_integrates_to_the_heading_change() -> None:
    """The Frenet kinematics rely on kappa being the turn rate of heading."""
    track = stadium_track()
    heading = np.unwrap(track.heading)
    turned = np.cumsum(track.kappa) * track.ds_m
    assert np.max(np.abs((heading - heading[0]) - (turned - turned[0]))) < 0.05


def test_widths_are_split_either_side() -> None:
    track = stadium_track(width_m=14.0)
    assert np.allclose(track.w_left, 7.0) and np.allclose(track.w_right, 7.0)


def test_to_track_keeps_length_and_width() -> None:
    track = stadium_track()
    v1 = track.to_track()
    assert v1.total_length_m == pytest.approx(track.length_m)
    assert all(s.width_m == pytest.approx(14.0) for s in v1.segments)


def test_held_out_tracks_are_real_tum_tracks() -> None:
    assert set(TUM_HELD_OUT) <= set(TUM_TRACKS)
    assert len(TUM_TRACKS) == 25


def test_unknown_tum_track_is_rejected() -> None:
    with pytest.raises(KeyError):
        load_tum_track("Atlantis")


@pytest.mark.slow
def test_a_real_tum_track_loads() -> None:
    try:
        track = load_tum_track("Spa")
    except urllib.error.URLError:
        pytest.skip("no network and no cached TUM data")
    assert track.length_m == pytest.approx(7000, rel=0.01)
    assert float(np.mean(track.w_left + track.w_right)) == pytest.approx(9.8, abs=0.5)
