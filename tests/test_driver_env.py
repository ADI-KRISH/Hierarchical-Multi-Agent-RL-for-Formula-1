import numpy as np
from gymnasium.utils.env_checker import check_env

from f1rl.config import EXAMPLE_CAR, SimParams
from f1rl.envs.driver_env import DriverEnv
from f1rl.envs.track import Segment, Track, example_track

FULL_THROTTLE_NO_STEER = np.array([1.0, 0.0], dtype=np.float32)


def _env() -> DriverEnv:
    return DriverEnv(EXAMPLE_CAR, example_track(), SimParams(seed=0))


def test_passes_gymnasium_env_checker() -> None:
    check_env(_env(), skip_render_check=True)


def test_random_agent_completes_an_episode_without_crashing() -> None:
    env = _env()
    obs, _info = env.reset(seed=0)
    assert env.observation_space.contains(obs)

    for _ in range(env.sim.max_episode_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, float)
        if terminated or truncated:
            break
    else:
        raise AssertionError("episode never terminated or truncated")


def test_full_throttle_goes_off_track_at_the_corner() -> None:
    """A naive full-throttle policy reaches the example track's straight-line top
    speed well before its corner's much lower grip limit -- it must be flagged
    off-track there, not sail through at an unphysical speed.
    """
    env = _env()
    env.reset(seed=0)
    terminated = truncated = False
    final_reward = 0.0
    while not (terminated or truncated):
        _, final_reward, terminated, truncated, _ = env.step(FULL_THROTTLE_NO_STEER)

    assert terminated is True
    assert truncated is False
    assert final_reward < 0.0


def test_zero_curvature_never_drifts_off_track() -> None:
    """A pure straight demands no lateral g at any speed, so even full throttle
    the whole way round must leave the lateral-offset observation at 0.
    """
    straight_loop = Track(segments=(Segment(length_m=1000.0, curvature_per_m=0.0),))
    env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0))
    env.reset(seed=0)
    terminated = truncated = False
    while not (terminated or truncated):
        obs, _reward, terminated, truncated, _ = env.step(FULL_THROTTLE_NO_STEER)
        assert obs[4] == 0.0  # lateral offset never grows

    assert terminated is True  # ends by completing the lap, not going off track


def test_steering_recovers_offset_using_leftover_grip_budget() -> None:
    """Full inward steering should shrink an existing offset when the corner
    isn't using up all the car's lateral grip -- the leftover budget is what
    steering draws on to get back to the racing line.
    """
    straight_loop = Track(segments=(Segment(length_m=10_000.0, curvature_per_m=0.0),))
    env = DriverEnv(EXAMPLE_CAR, straight_loop, SimParams(seed=0))
    env.reset(seed=0)
    env._lateral_offset_m = 3.0  # pretend the car already ran wide
    before = env._observation()[4]

    obs, _reward, _terminated, _truncated, _info = env.step(
        np.array([0.0, -1.0], dtype=np.float32)
    )

    assert obs[4] < before


def test_full_outward_steering_alone_can_run_off_track() -> None:
    """Even with no cornering demand at all, steering hard toward the edge should
    still be able to push the offset past the (narrower) corner's half-width.
    """
    tight_corner_loop = Track(
        segments=(Segment(length_m=10_000.0, curvature_per_m=0.0, width_m=1.0),)
    )
    env = DriverEnv(EXAMPLE_CAR, tight_corner_loop, SimParams(seed=0))
    env.reset(seed=0)
    terminated = truncated = False
    steps = 0
    while not (terminated or truncated) and steps < 100:
        _, _reward, terminated, truncated, _info = env.step(
            np.array([0.0, 1.0], dtype=np.float32)
        )
        steps += 1

    assert terminated is True
