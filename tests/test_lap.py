from f1rl.config import EXAMPLE_CAR
from f1rl.envs.track import example_track
from f1rl.models.lap import lap_time_s, speed_profile_ms


def test_speed_profile_stays_within_car_limits() -> None:
    track = example_track()
    speeds = speed_profile_ms(EXAMPLE_CAR, track)
    assert len(speeds) == len(track.segments)
    assert all(0.0 < v <= EXAMPLE_CAR.max_speed_ms for v in speeds)


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
