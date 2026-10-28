import numpy as np
import torch

from f1rl.agents.tactical import MultiTrackTactical, RivalPool
from f1rl.agents.tactics import MOVES
from f1rl.config import EXAMPLE_CAR, RaceEnvParams, SimParams
from f1rl.envs.tactical_env import TacticalEnv
from f1rl.models.raceline import optimal_line
from racetrack_fixtures import stadium_track

KINDS = ("rival_ahead", "rival_behind")


def _env() -> TacticalEnv:
    params = RaceEnvParams(
        tactical=True, tires=True, opponents=KINDS, safety_filter=True
    )
    track = stadium_track()
    line = optimal_line(track, EXAMPLE_CAR, 1.75)
    return TacticalEnv(EXAMPLE_CAR, track, SimParams(seed=0), params, line, "random")


def test_the_rival_sees_the_race_from_its_own_car() -> None:
    env = _env()
    obs, _ = env.reset(seed=0, options={"opponent": "rival_ahead", "gap_m": 30.0})
    assert env._opp is not None
    me, them = env._state, env._opp.state
    seen = env.rival_observation()
    assert seen.shape == obs.shape
    assert env._state == me and env._opp.state == them  # nothing was moved
    # Our observation has the car 30 m ahead; the rival's has it 30 m behind.
    assert not np.allclose(seen, obs)


def test_the_rival_policy_picks_the_opponents_moves() -> None:
    env = _env()
    env.rival = lambda obs: MOVES.index("outside")
    env.reset(seed=0, options={"opponent": "rival_behind", "gap_m": 25.0})
    env.step(MOVES.index("line"))
    assert env._opp_move == "outside" and env._opp is not None
    assert env._opp.maneuver is not None


def test_two_cars_racing_flat_out_never_touch() -> None:
    env = _env()
    rng = np.random.default_rng(0)
    env.rival = lambda obs: int(rng.integers(len(MOVES)))
    for ep in range(6):
        env.reset(seed=ep, options={"opponent": KINDS[ep % 2], "gap_m": 12.0})
        for _ in range(60):
            _, _, terminated, truncated, info = env.step(int(rng.integers(len(MOVES))))
            if terminated or truncated:
                break
        assert not info["contact"]


def test_the_pool_keeps_frozen_copies_of_the_newest_policies() -> None:
    pool = RivalPool(size=2)
    net = torch.nn.Linear(2, 2)
    policies = []
    for k in range(3):
        p = _FakePolicy(net, k)
        policies.append(p)
        pool.add(p)
    assert [p.tag for p in pool.policies] == [1, 2]
    with torch.no_grad():
        net.weight.fill_(7.0)  # training goes on; the frozen copies do not change
    assert not torch.equal(pool.policies[0].net.weight, net.weight)


def test_self_play_episodes_race_a_rival_in_the_real_opponents_place() -> None:
    pool = RivalPool(size=1)
    pool.policies = [_Fixed()]
    multi = MultiTrackTactical({"Stadium": _env()}, 0, pool=pool, rival_frac=1.0)
    for _ in range(4):
        multi.reset()
        env = multi.envs["Stadium"]
        assert env.rival is not None and env._opp is not None
        assert env._opp.kind in KINDS


class _FakePolicy:
    def __init__(self, net: torch.nn.Module, tag: int) -> None:
        self.net, self.tag = net, tag

    def set_training_mode(self, mode: bool) -> None:
        pass


class _Fixed:
    def predict(self, obs: object, deterministic: bool = True) -> tuple[int, None]:
        return 0, None
