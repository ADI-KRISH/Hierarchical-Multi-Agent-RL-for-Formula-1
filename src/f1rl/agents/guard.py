"""The defending sparring driver: a scripted car ahead that defends by the book.

It is the opponent the overtaking driver trains against (``guard_*`` kinds in
`config.OPPONENT_KINDS`). Each corner is defended on its own (a "section" is the
straight and braking zone leading to one apex), so training can start at any corner:

- **Close the inside.** When the car behind is within `GuardParams.inside_gap_s`
  and the next braking point is near, move toward the corner's inside, leaving less
  than a car's width there, and finish the move before the braking zone. Entering
  the corner from the inside is slower (a tighter line): that is the price of
  defending, and the attacker's chance to switch back on the exit.
- **Cover a pull-out.** On the straight, a car that pulls out toward the inside of
  the next corner is covered the same way, after `GuardParams.reaction_s`.
- **One move.** A single change of direction per section. A car pulling to the
  outside is not followed: the inside is kept for when it matters.
- **Never in the braking zone, never once alongside.** No move starts in a braking
  zone; once the car behind is alongside (its front wheels level with our rear),
  the guard stops moving toward it and leaves it room. The no-crash filter still
  runs on both cars.
- **Back to the line** after the apex, ready for the next section.

`Guard.update` is the whole decision: state in, a new `Maneuver` (or None) out. It
keeps a little memory (which section it is in, whether its move is spent) and counts
what it did in `Guard.stats`.
"""

from dataclasses import dataclass, field

import numpy as np

from f1rl.agents.path_driver import PathDriver
from f1rl.agents.tactics import Corner, corners_ahead, room_beside
from f1rl.config import GuardParams
from f1rl.models.frenet import FrenetState
from f1rl.models.maneuver import Maneuver, shift_at

#: No move starts closer than this (m) to a braking point (as `opponents`).
NO_MOVE_BRAKE_M = 25.0
#: Sideways creep (m per step, toward a car alongside) that counts as a squeeze.
SQUEEZE_M = 0.05
#: Sections are keyed by the apex position, rounded to this (m).
SECTION_ROUND_M = 20.0


def wrap(gap_m: float, lap_m: float) -> float:
    """``gap_m`` folded into [-lap/2, lap/2)."""
    return (gap_m + lap_m / 2) % lap_m - lap_m / 2


def inside_maneuver(
    driver: PathDriver,
    s_m: float,
    shift_now: float,
    corner: Corner,
    next_corner: Corner | None,
    p: GuardParams,
) -> Maneuver | None:
    """Move toward ``corner``'s inside now, settle before its braking point, keep the
    inside to the apex, then rejoin the line. None if there is no road or no room."""
    end = corner.brake_m - p.settle_m
    if end < p.min_ramp_m:
        return None  # too late: it would be a lunge across the track
    ramp = min(p.ramp_m, end)
    # Room beside the line on the inside over the last stretch before the braking
    # point, where a dive starts: go there, leaving less than a car width for it.
    span = min(p.door_m, corner.brake_m - ramp)
    free = room_beside(
        driver, s_m + corner.brake_m - span, max(10.0, span), corner.side
    )
    amount = min(p.max_shift_m, free - p.inside_leave_m)
    if amount < p.min_shift_m:
        return None  # the line is already on the inside: nothing to close
    target = corner.side * amount
    if abs(target - shift_now) < 0.3:
        return None  # already there
    back = corner.apex_m + p.recover_m
    if next_corner is not None:
        back = min(back, max(corner.apex_m + 30.0, next_corner.brake_m - 30.0))
    knots = [(0.0, shift_now), (ramp, target)]
    # Hold that spot on the road (not that offset from the line) through the
    # braking zone until the line itself comes inside it, then follow the line to
    # the apex: never drift back out toward a car on the outside under braking.
    line_n = driver.base.line_state
    hold_n = line_n(s_m + corner.brake_m)[0] + target
    d = max(ramp + 5.0, corner.brake_m)
    while d < corner.apex_m:
        extra = hold_n - line_n(s_m + d)[0]
        knots.append((d, extra if extra * corner.side > 0 else 0.0))
        d += 10.0
    apex = max(knots[-1][0] + 5.0, corner.apex_m)
    knots.append((apex, target * p.apex_keep))
    knots.append((max(apex + 5.0, back), 0.0))
    return Maneuver(s_m, tuple(knots))


@dataclass
class Guard:
    params: GuardParams
    section: float | None = None  # apex position (m round the lap) of this section
    spent: bool = False  # the move for this section is used
    threat_since_m: float | None = None  # where the threat was first seen
    last_shift: float | None = None
    stats: dict[str, int] = field(
        default_factory=lambda: {
            "moves": 0,
            "sections": 0,
            "held_alongside": 0,  # steps it froze a move because a car was alongside
            # Steps where a car was alongside and the guard still moved toward it by
            # more than `SQUEEZE_M` (measured, not assumed: the target is zero).
            "squeeze_steps": 0,
        }
    )

    def update(
        self,
        driver: PathDriver,
        me: FrenetState,
        car: FrenetState,
        lap_m: float,
        current: Maneuver,
        car_length_m: float,
    ) -> Maneuver | None:
        """The guard's new manoeuvre this step, or None to keep ``current`` (the
        plan it is driving now)."""
        p = self.params
        line_n, _, _ = driver.base.line_state(me.s_m)
        shift = me.n_m - line_n
        last, self.last_shift = self.last_shift, me.n_m
        corners = [c for c in corners_ahead(driver, me.s_m) if c.apex_m > 5.0]
        if corners:
            key = round(((me.s_m + corners[0].apex_m) % lap_m) / SECTION_ROUND_M)
            if key != self.section:
                self.section, self.spent, self.threat_since_m = key, False, None
                self.stats["sections"] += 1
        gap = wrap(car.s_m - me.s_m, lap_m)  # < 0: the car is behind us
        if not -p.react_gap_m < gap < 0.0:
            self.threat_since_m = None
            return None  # nobody to defend against (or it is past)
        lateral = car.n_m - me.n_m  # > 0: it is to our left
        alongside = -gap < car_length_m - p.overlap_m
        if alongside:
            moved = (
                0.0 if last is None else (me.n_m - last) * (1 if lateral > 0 else -1)
            )
            if moved > SQUEEZE_M and abs(lateral) < p.room_m:
                self.stats["squeeze_steps"] += 1
            if p.room_m <= 0.0:
                return None  # leave it to the no-crash filter
            return self._leave_room(driver, me, car, current)
        if self.spent or not corners:
            return None
        c = corners[0]
        if c.brake_m < NO_MOVE_BRAKE_M:
            return None  # in the braking zone: hold what we have
        if -gap < car_length_m + p.min_move_gap_m:
            return None  # too close to move across its nose: it would have to swerve
        # A threat: the car behind will be alongside by the braking point at the
        # speed it is closing now (a car that is not catching us cannot dive).
        closing = car.speed_ms - me.speed_ms
        t_brake = c.brake_m / max(me.speed_ms, 1.0)
        reach = -gap - max(0.0, closing) * t_brake
        time_gap = -gap / max(me.speed_ms, 1.0)
        close = (
            c.brake_m < p.inside_from_m
            and time_gap < p.inside_gap_s
            and reach < car_length_m + p.threat_margin_m
        )
        pulled = (
            lateral * c.side > p.cover_lateral_m and c.brake_m < 2 * p.inside_from_m
        )
        if not (close or pulled):
            self.threat_since_m = None
            return None
        if self.threat_since_m is None:
            self.threat_since_m = me.s_m
        if me.s_m - self.threat_since_m < p.reaction_s * me.speed_ms:
            return None  # still reacting
        m = inside_maneuver(
            driver, me.s_m, shift, c, corners[1] if len(corners) > 1 else None, p
        )
        self.spent = True  # one go per section, even if there was nothing to do
        if m is None:
            return None
        self.stats["moves"] += 1
        return m

    def _leave_room(
        self,
        driver: PathDriver,
        me: FrenetState,
        car: FrenetState,
        current: Maneuver,
    ) -> Maneuver:
        """A car alongside: leave it a car's width. The plan is clipped so that over
        the next stretch our centre gets no closer sideways to where the other car
        is now than `GuardParams.room_m` (or than it is now, if already closer);
        away from it is fine."""
        p = self.params
        side = 1.0 if car.n_m > me.n_m else -1.0  # +1: it is to our left
        # Never closer than now, and no closer than `room_m`: it is not pushed
        # away from its own line, only kept from squeezing.
        keep = min(p.room_m, abs(car.n_m - me.n_m))
        knots = []
        for d in (0.0, 15.0, 30.0, 45.0, 60.0):
            s = me.s_m + d
            line_n = driver.base.line_state(s)[0]
            want = float(shift_at(current, np.array([s]))[0])
            limit = car.n_m - side * keep - line_n  # the furthest shift allowed
            knots.append((d, min(want, limit) if side > 0 else max(want, limit)))
        clipped = any(
            abs(k[1] - float(shift_at(current, np.array([me.s_m + k[0]]))[0])) > 1e-6
            for k in knots
        )
        if clipped:
            self.stats["held_alongside"] += 1
        # Start from where the car is (never a jump in the path).
        knots[0] = (0.0, me.n_m - driver.base.line_state(me.s_m)[0])
        return Maneuver(me.s_m, tuple(knots))
