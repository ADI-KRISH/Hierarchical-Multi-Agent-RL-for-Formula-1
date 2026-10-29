"""`TacticalEnv`: `RaceEnv` where the action picks a race-craft move.

Every ``option_steps`` decisions (0.5 s) the agent chooses one of `tactics.MOVES`
(line, outside, divebomb, dummy, switchback, cover, inside); a scripted `PathDriver`
then drives that move at the pace set by push level, tyre wear and the wake. So the
agent learns WHEN to attack or defend and WHICH move, not how to steer.

Rules of the road enforced here: a move already running is only replaced outside the
braking zone, and a ``cover`` is a single move (no weaving) -- switching straight
back is refused until the car behind has been clear for a while.
"""

from collections.abc import Callable
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from f1rl.agents.path_driver import PathDriver
from f1rl.agents.tactics import MOVES, PULL_MOVES, corners_ahead, make_move
from f1rl.config import GRAVITY_M_S2, CarParams, RaceEnvParams, SimParams
from f1rl.envs.opponents import (
    FOLLOW_GAP_M,
    NO_MOVE_BRAKE_M,
    follow_brake,
    wrap_gap,
)
from f1rl.envs.race_env import ObsType, RaceEnv
from f1rl.envs.racetrack import RaceTrack
from f1rl.models.maneuver import Maneuver, shift_at
from f1rl.models.raceline import OptimalLine

#: The no-chop rule looks this far (m) ahead along the plan, and holds that long.
CHOP_LOOK_M = 30.0


class TacticalEnv(RaceEnv):
    def __init__(
        self,
        car: CarParams,
        track: RaceTrack,
        sim: SimParams,
        params: RaceEnvParams | None = None,
        reference: OptimalLine | None = None,
        mode: str = "random",
        option_steps: int = 5,
    ) -> None:
        params = params or RaceEnvParams(tactical=True)
        if not params.tactical:
            raise ValueError("TacticalEnv needs RaceEnvParams(tactical=True)")
        super().__init__(car, track, sim, params, reference, mode)
        self.option_steps = option_steps
        # The moves this env offers: `MOVES`, plus `PULL_MOVES` if turned on.
        self.moves: tuple[str, ...] = MOVES + (PULL_MOVES if params.pull_moves else ())
        self.action_space = gym.spaces.Discrete(len(self.moves))
        n = self.observation_space.shape[0]  # type: ignore[index]
        self.observation_space = gym.spaces.Box(
            -5.0, 5.0, (n + len(self.moves) + 2,), np.float32
        )
        self._move = "line"
        self._maneuver: Maneuver | None = None
        self.refused = 0  # moves refused by the rules, this episode
        self.chop_holds = 0  # times the no-chop rule held the car's line
        # Self-play: picks the moves of a ``rival_*`` opponent from that car's own
        # view of the race (same observation layout as ours). None = scripted.
        self.rival: Callable[[ObsType], int] | None = None
        self._opp_move = "line"

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        _, info = super().reset(seed=seed, options=options)
        self._move, self._maneuver, self.refused = "line", None, 0
        self.chop_holds = 0
        self._opp_move = "line"
        return self._tactical_obs(), info

    def _driver(self) -> PathDriver:
        d = self._follower()
        assert isinstance(d, PathDriver)
        return d

    def _tactical_obs(self) -> ObsType:
        d = self._driver()
        corners = corners_ahead(d, self._state.s_m)
        brake = min(corners[0].brake_m, 300.0) / 300.0 if corners else 1.0
        side = corners[0].side if corners else 0.0
        hot = np.zeros(len(self.moves), dtype=np.float32)
        hot[self.moves.index(self._move)] = 1.0
        return np.concatenate(
            [self._observation(), hot, np.array([brake, side], dtype=np.float32)]
        ).astype(np.float32)

    def _shield(self, a: NDArray[np.float32]) -> NDArray[np.float32]:
        """Safety under every move: brake rather than run into a car ahead on the same
        piece of track (the driver is a racer, not a bulldozer)."""
        opp = self._opp
        if opp is None or not self.params.follow_shield:
            return a
        if not self.params.shield_in_tow and self._tow_now > 0:
            return a  # in the tow on a straight: free to close up and pull out
        gap = wrap_gap(opp.state.s_m - self._state.s_m, self._lap_m)
        closing = max(0.0, self._state.speed_ms - opp.state.speed_ms)
        lined = abs(opp.state.n_m - self._state.n_m) < self.params.car_width_m + 0.3
        if lined and 0.0 < gap < FOLLOW_GAP_M + closing:
            a = a.copy()
            a[0] = follow_brake(
                float(a[0]),
                gap,
                closing,
                self.car.max_braking_g * GRAVITY_M_S2,
                self.params.soft_follow,
            )
        return a

    def _no_chop(self, d: PathDriver) -> None:
        """Clean racing (`RaceEnvParams.hold_until_clear`): after getting alongside
        or past, do not move back across into the other car's lane until clear of
        its nose by `clean_chop_m` (plus a car length of margin): hold the current
        offset from the line instead of the planned way back."""
        p, opp, m = self.params, self._opp, self._maneuver
        if not p.hold_until_clear or opp is None or m is None:
            return
        me = self._state
        gap = wrap_gap(opp.state.s_m - me.s_m, self._lap_m)  # > 0: it is ahead
        if gap > p.car_length_m:  # still behind it: nothing to cut across
            return
        if -gap - p.car_length_m >= p.clean_chop_m + p.car_length_m:
            return  # clear
        dn = opp.state.n_m - me.n_m
        if abs(dn) < p.car_width_m + p.clean_side_m:
            return  # already in its lane: moving away is the cure, not holding
        ahead = me.s_m + CHOP_LOOK_M
        line_n = d.base.line_state(ahead)[0]
        planned = line_n + float(shift_at(m, np.array([ahead]))[0])
        opp_n = opp.state.n_m + (
            d.base.line_state(opp.state.s_m + CHOP_LOOK_M)[0]
            - d.base.line_state(opp.state.s_m)[0]
        )
        if abs(opp_n - planned) >= abs(dn) - 0.05:
            return  # the plan does not close on it
        here = me.n_m - d.base.line_state(me.s_m)[0]
        self._maneuver = Maneuver(me.s_m, ((0.0, here), (CHOP_LOOK_M, here)))
        self.chop_holds += 1

    def _may_change(self, corners: list[Any]) -> bool:
        """Free to start a new move: not in a braking zone, none in progress."""
        in_zone = bool(corners) and corners[0].brake_m < NO_MOVE_BRAKE_M
        running = self._maneuver is not None and self._state.s_m < self._maneuver.end_s
        return not in_zone and not (running and self._move == "cover")

    def rival_observation(self) -> ObsType:
        """What the opponent car sees: our observation with the two cars swapped."""
        opp = self._opp
        assert opp is not None
        mine = (self._state, self._last, self._move, self._tow_now)
        self._state, opp.state = opp.state, self._state
        self._last, self._move = self._opp_last, self._opp_move
        self._tow_now = self._tow(self._state, opp.state)
        try:
            return self._tactical_obs()
        finally:
            opp.state, self._state = self._state, mine[0]
            self._last, self._move, self._tow_now = mine[1], mine[2], mine[3]

    def _rival_move(self) -> None:
        """Self-play: the rival policy picks the opponent's move, under the same rules
        as ours (no new move in a braking zone, a cover is a single move)."""
        opp = self._opp
        if self.rival is None or opp is None or not opp.kind.startswith("rival"):
            return
        want = self.moves[int(self.rival(self.rival_observation()))]
        s = opp.state.s_m
        running = opp.maneuver is not None and s < opp.maneuver.end_s
        if want == self._opp_move and running:
            return
        corners = corners_ahead(opp.driver, s)
        in_zone = bool(corners) and corners[0].brake_m < NO_MOVE_BRAKE_M
        if running and (in_zone or self._opp_move == "cover"):
            return
        shift = opp.state.n_m - opp.driver.base.line_state(s)[0]
        threat = 0.0
        if wrap_gap(self._state.s_m - s, self._lap_m) < 0:  # we are behind it
            threat = self._state.n_m - opp.driver.base.line_state(self._state.s_m)[0]
        opp.maneuver = make_move(want, opp.driver, s, shift, threat_shift=threat)
        self._opp_move = want

    def step(self, action: Any) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        self._rival_move()
        want = self.moves[int(action)]
        d = self._driver()
        s = self._state.s_m
        corners = corners_ahead(d, s)
        # An attacker may pull alongside in a braking zone (moving under braking is
        # a defender's offence); a running cover still may not be switched.
        attacking = (
            want in PULL_MOVES
            and self._opp is not None
            and wrap_gap(self._opp.state.s_m - s, self._lap_m) > 0
            and self._move != "cover"
        )
        if want != self._move or self._maneuver is None or s > self._maneuver.end_s:
            if self._may_change(corners) or self._maneuver is None or attacking:
                shift = self._state.n_m - d.base.line_state(s)[0]
                threat = 0.0
                if self._opp is not None:
                    threat = (
                        self._opp.state.n_m - d.base.line_state(self._opp.state.s_m)[0]
                    )
                if (
                    self._opp is not None
                    and wrap_gap(self._opp.state.s_m - s, self._lap_m) >= 0
                ):
                    threat = 0.0  # nobody behind to cover
                self._maneuver = make_move(want, d, s, shift, threat_shift=threat)
                self._move = want
            else:
                self.refused += 1
        total = 0.0
        terminated = truncated = False
        info: dict[str, Any] = {}
        p = self.params
        chunks = 1
        if p.commit_moves and self._move != "line":
            chunks = p.commit_max_decisions
        for chunk in range(chunks):
            if chunk > 0:
                m = self._maneuver
                if m is None or self._state.s_m > m.end_s:
                    break  # the move is done: the agent decides again
                self._rival_move()
            total, terminated, truncated, info = self._drive(
                total, terminated, truncated, info
            )
            if terminated or truncated:
                break
        info["move"] = self._move
        info["refused"] = self.refused
        return self._tactical_obs(), total, terminated, truncated, info

    def _drive(
        self, total: float, terminated: bool, truncated: bool, info: dict[str, Any]
    ) -> tuple[float, bool, bool, dict[str, Any]]:
        """One decision's worth (`option_steps` env steps) of the running move."""
        for _ in range(self.option_steps):
            d = self._driver()  # grip (tyres, wake) may have moved the plan
            self._no_chop(d)
            assert self._maneuver is not None
            a: NDArray[np.float32] = d.act(
                {
                    "distance_m": self._state.s_m,
                    "lateral_offset_m": self._state.n_m,
                    "heading_rad": self._state.psi_rad,
                    "speed_ms": self._state.speed_ms,
                },
                self._maneuver,
                self._tow_now,
            )
            _, r, terminated, truncated, info = super().step(self._shield(a))
            total += r
            if terminated or truncated:
                break
        return total, terminated, truncated, info
