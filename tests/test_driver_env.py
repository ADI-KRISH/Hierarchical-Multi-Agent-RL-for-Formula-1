from gymnasium.utils.env_checker import check_env

from f1rl.config import EXAMPLE_CAR, SimParams
from f1rl.envs.driver_env import DriverEnv
from f1rl.envs.track import example_track


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
        _, final_reward, terminated, truncated, _ = env.step(env.action_space.high)

    assert terminated is True
    assert truncated is False
    assert final_reward < 0.0
