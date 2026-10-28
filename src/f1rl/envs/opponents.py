"""Scripted opponent cars for `RaceEnv`: a `PathDriver` at a set pace, with a little
race-craft depending on the kind (`config.OPPONENT_KINDS`):

- ``slow`` / ``equal``  hold the racing line;
- ``defender``  covers the side a car behind threatens (once, then holds: no weaving)
  and takes the inside line before a corner when a car is close behind;
- ``attacker``  dive-bombs the next corner when it is within reach of the car ahead;
- ``rival_ahead`` / ``rival_behind``  self-play: a learned policy picks its moves
  (`TacticalEnv.rival`); like the learner, it does not lift for a car alongside --
  the no-crash filter keeps the two apart.

All of them follow the etiquette of the sport: a car that has its nose alongside or
ahead gets room (the opponent lifts), and none of them moves in a braking zone.
"""

from dataclasses import dataclass

import numpy as np

from f1rl.agents.path_driver import PathDriver
from f1rl.agents.tactics import Corner, corners_ahead, make_move
from f1rl.config import GRAVITY_M_S2
from f1rl.models.frenet import FrenetState
from f1rl.models.maneuver import Maneuver

#: A car within this gap (m) behind another starts reacting to it.
REACT_GAP_M = 45.0
#: Not before a corner closer than this (m): no moves in the braking zone.
NO_MOVE_BRAKE_M = 25.0
YIELD_SHIFT_M = 2.5  # how far a car gives way sideways
YIELD_BRAKE = 0.8  # braking command when a car has the right of way
#: A defender takes the inside line when the car is this close behind (m).
INSIDE_GAP_M = 20.0
#: An opponent keeps at least this gap (m) plus a second of closing speed.
FOLLOW_GAP_M = 12.0


def follow_brake(
    pedal: float, gap_m: float, closing_ms: float, braking_ms2: float, soft: bool
) -> float:
    """The pedal of a car lined up behind another, ``gap_m`` back (centre to centre)
    and closing at ``closing_ms``. Hard (earlier runs): brake at `YIELD_BRAKE`
    whatever is needed. Soft: brake just enough to match its speed before the gap
    closes to `FOLLOW_GAP_M` / 2 -- a driver lifts, he does not brake-check himself."""
    if not soft:
        return min(pedal, -YIELD_BRAKE)
    room = max(0.5, gap_m - FOLLOW_GAP_M / 2)
    if closing_ms <= 0.0:
        return min(pedal, 0.0) if gap_m < FOLLOW_GAP_M / 2 else pedal
    need = 1.2 * closing_ms**2 / (2.0 * room) / braking_ms2
    return min(pedal, -min(1.0, need))


def wrap_gap(gap_m: float, lap_m: float) -> float:
    """``gap_m`` folded into [-lap/2, lap/2): the signed distance round the lap."""
    return (gap_m + lap_m / 2) % lap_m - lap_m / 2


@dataclass
class Opponent:
    kind: str
    driver: PathDriver
    state: FrenetState
    side: float = 1.0  # +1 / -1: which way an attacker prefers to go round
    maneuver: Maneuver | None = None
    cooldown_until_s: float = -1e9
    yielding: bool = False

    def _shift_now(self) -> float:
        line_n, _, _ = self.driver.base.line_state(self.state.s_m)
        return self.state.n_m - line_n

    def command(
        self,
        car: FrenetState,
        lap_m: float,
        alongside_m: float,
        overlap_m: float,
        extra_speed_ms: float = 0.0,
        shield: bool = True,
        give_way: bool = True,
        soft_follow: bool = False,
    ) -> np.ndarray:
        """The opponent's [throttle/brake, steer] for the next decision."""
        me = self.state
        gap = wrap_gap(car.s_m - me.s_m, lap_m)  # > 0: the car is ahead of me
        lateral = car.n_m - me.n_m
        if self.maneuver is not None and me.s_m > self.maneuver.end_s:
            self.maneuver = None
        shift = self._shift_now()
        corners = corners_ahead(self.driver, me.s_m)
        free = self.maneuver is None and (
            not corners or corners[0].brake_m > NO_MOVE_BRAKE_M
        )
        if free and me.s_m > self.cooldown_until_s:
            move = self._choose(gap, lateral, corners, shift)
            if move is not None:
                self.maneuver = make_move(
                    move,
                    self.driver,
                    me.s_m,
                    shift,
                    threat_shift=0.0 if move != "cover" else self._threat(car, me),
                )
                self.cooldown_until_s = self.maneuver.end_s + 50.0
        m = self.maneuver or Maneuver(me.s_m, ((0.0, shift), (40.0, 0.0)))
        action = self.driver.act(
            {
                "distance_m": me.s_m,
                "lateral_offset_m": me.n_m,
                "heading_rad": me.psi_rad,
                "speed_ms": me.speed_ms,
            },
            m,
            extra_speed_ms,
        )
        # Give way to a car whose nose is alongside or just ahead of us; stay clear of
        # one a little further ahead on the same piece of track.
        lined_up = abs(lateral) < overlap_m
        closing = max(0.0, me.speed_ms - car.speed_ms)
        rival = self.kind.startswith("rival")  # a learned driver: no scripted yield
        yields = give_way and not rival
        if lined_up and -alongside_m / 2 < gap < alongside_m and yields:
            action[0] = min(action[0], -YIELD_BRAKE)
            if not self.yielding:  # leave room: move away from the car, once
                self.yielding = True
                away = -1.0 if lateral > 0 else 1.0
                give = shift + away * YIELD_SHIFT_M
                self.maneuver = Maneuver(
                    me.s_m, ((0.0, shift), (25.0, give), (150.0, give), (250.0, 0.0))
                )
        elif self.yielding and abs(gap) > 2 * alongside_m:
            self.yielding = False
        elif (
            shield
            and abs(lateral) < overlap_m - 0.7
            and 0.0 < gap < FOLLOW_GAP_M + closing
        ):
            action[0] = follow_brake(
                float(action[0]),
                gap,
                closing,
                self.driver.car.max_braking_g * GRAVITY_M_S2,
                soft_follow,
            )
        return action

    def _threat(self, car: FrenetState, me: FrenetState) -> float:
        line_n, _, _ = self.driver.base.line_state(car.s_m)
        return car.n_m - line_n

    def _choose(
        self, gap: float, lateral: float, corners: list[Corner], shift: float
    ) -> str | None:
        if self.kind == "defender" and -REACT_GAP_M < gap < 0.0:
            if corners and -gap < INSIDE_GAP_M and corners[0].brake_m < 200.0:
                return "inside"
            return "cover"
        if (
            self.kind.startswith("attacker")
            and 0.0 < gap < REACT_GAP_M * 1.5
            and corners
        ):
            return "divebomb" if self.side > 0 else "outside"
        return None


def place(kind: str, driver: PathDriver, s_m: float, side: float) -> Opponent:
    """An opponent of ``kind`` on its line at ``s_m``, at the line's speed."""
    n, psi, v = driver.base.line_state(s_m)
    return Opponent(kind, driver, FrenetState(s_m, n, psi, v), side)
