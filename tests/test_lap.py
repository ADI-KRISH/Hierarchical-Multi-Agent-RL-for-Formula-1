from f1rl.config import EXAMPLE_CAR
from f1rl.envs.track import curvature_at, example_track
from f1rl.models.car import max_corner_speed_ms
from f1rl.models.lap import lap_time_s, speed_profile_ms


def test_speed_profile_stays_within_car_limits() -> None:
    track = example_track()
    speeds = speed_profile_ms(EXAMPLE_CAR, track)
    assert len(speeds) > len(track.segments)
    assert all(0.0 < v <= EXAMPLE_CAR.max_speed_ms for v in speeds)


def test_speed_profile_respects_the_corner_grip_limit_throughout_the_corner() -> None:
    """Regression: a corner's grip cap must hold at every station inside it, not
    just at the segment boundary -- otherwise the car could accelerate past its
    lateral-grip limit in the middle of the corner.
    """
    track = example_track()
    speeds = speed_profile_ms(EXAMPLE_CAR, track, step_m=1.0)
    corner_cap = max_corner_speed_ms(EXAMPLE_CAR, curvature_per_m=1 / 100)
    ds = track.total_length_m / len(speeds)
    for k, v in enumerate(speeds):
        if curvature_at(track, k * ds) != 0.0:
            assert v <= corner_cap + 1e-6


def test_lap_time_is_physically_sane() -> None:
    """Roadmap phase-1 done criterion: tens of seconds to a couple minutes."""
    lap_time = lap_time_s(EXAMPLE_CAR, example_track())
    assert 5.0 < lap_time < 180.0


def test_more_iterations_do_not_increase_lap_time() -> None:
    """Extra converging passes can only tighten the speed profile, never loosen it."""
    track = example_track()
    coarse = lap_time_s(EXAMPLE_CAR, track, iterations=1)
    converged = lap_time_s(EXAMPLE_CAR, track, iterations=5)
    assert converged <= coarse
