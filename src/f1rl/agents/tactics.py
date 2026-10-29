"""Race-craft moves, each a `Maneuver` for the `PathDriver` to execute.

Every move is a list of (distance, sideways shift) knots plus how late it brakes,
built relative to the next corner ahead (its braking point and apex) so the same
move works on any track. ``side`` is the corner's inside: +1 = left, -1 = right.

Overtaking:
  outside      stay wide into the corner, run alongside, carry speed round the outside
  divebomb     pull to the inside, brake later than the car ahead, take the apex
  dummy        feint to the outside, then switch to the inside before the braking zone
  switchback   hold station through a corner, then dive at the NEXT one (the car that
               just passed is on the wrong line for it)
Defending:
  cover        move once to the side the car behind threatens, then hold (no weaving)
  inside       take the inside line early, closing the dive-bomb door
  line         the racing line (also the default)

A move is only started with room to make it: before the braking zone, never in it.
"""

from dataclasses import dataclass

import numpy as np

from f1rl.agents.path_driver import PathDriver
from f1rl.models.maneuver import Maneuver

MOVES = ("line", "outside", "divebomb", "dummy", "switchback", "cover", "inside")
#: Extra attacking moves (`RaceEnvParams.pull_moves`): pull alongside NOW, toward the
#: inside or the outside of the next corner (or the one we are in), even in a braking
#: zone or a corner. The other moves are planned from the straight before a corner,
#: so in a corner, or with the braking point near, there was no way to get alongside.
PULL_MOVES = ("pull_in", "pull_out")
#: Pull moves: the shift (m, a car width plus a gap), how quickly it is made, at
#: least how long it is held (or to just past the apex), and the smallest worth it.
PULL_SHIFT_M = 2.8
PULL_RAMP_M = 50.0
PULL_HOLD_M = 200.0
PULL_MIN_M = 1.5
#: Direct control (`RaceEnvParams.lane_moves`): "drive this far (m, + = left) off
#: the racing line", re-chosen every decision, each at normal or late braking. The
#: agent picks its own line and braking point instead of a scripted move.
LANE_SHIFTS_M = (-3.5, -2.0, 0.0, 2.0, 3.5)
LANE_MOVES = tuple(
    f"lane{x:+.1f}{late}" for x in LANE_SHIFTS_M for late in ("", "_late")
)
#: Lane moves: how quickly the new offset is reached (m), and how far ahead (m) the
#: road must have room for it.
LANE_RAMP_M = 50.0
LANE_LOOK_M = 120.0

#: Largest sideways shift a move asks for (m): a car width plus a gap, and the
#: ramp lengths (m) over which it is made and undone.
MAX_SHIFT_M = 3.5
RAMP_M = 100.0
#: A move needs this much road before the corner's braking point to be worth starting.
MIN_ROOM_M = 100.0
#: Keep this much road beside the car at the largest shift; below this smallest shift
#: a move is not worth making.
EDGE_CLEARANCE_M = 0.6
MIN_SHIFT_M = 2.3
RECOVER_M = 150.0
#: Late braking: the share of the brake limit a divebomb plans for (normal is 0.95).
DIVE_BRAKE_FRACTION = 0.97
#: A corner is where the racing line's speed falls below this share of top speed.
CORNER_SPEED_FRACTION = 0.8
LOOKAHEAD_M = 600.0


@dataclass(frozen=True)
class Corner:
    brake_m: float  # distance ahead of the car where braking starts
    apex_m: float  # distance ahead of the car of the slowest point
    side: float  # +1 left-hander, -1 right-hander (the inside)


def corners_ahead(
    driver: PathDriver, s_m: float, within_m: float = LOOKAHEAD_M
) -> list[Corner]:
    """Corners on the racing line in the next ``within_m``, nearest first."""
    track = driver.track
    k = int(within_m / track.ds_m)
    i0 = int(s_m / track.ds_m)
    idx = (i0 + np.arange(k + 1)) % track.n
    v = driver.line_speed()[idx]
    kappa = driver.line_kappa()[idx]
    vmax = driver.car.max_speed_ms
    out: list[Corner] = []
    off = s_m - i0 * track.ds_m
    for a in range(1, len(v) - 1):
        # Every local minimum of the line's speed below the corner threshold is a
        # corner of its own (two bends in a row each have their braking zone).
        if v[a] >= CORNER_SPEED_FRACTION * vmax or v[a] > v[a - 1] or v[a] >= v[a + 1]:
            continue
        b = a
        while b > 0 and v[b - 1] > v[b]:  # back up the braking ramp
            b -= 1
        out.append(
            Corner(
                max(0.0, b * track.ds_m - off),
                max(0.0, a * track.ds_m - off),
                float(np.sign(kappa[a])) or 1.0,
            )
        )
    return out


def room_beside(driver: PathDriver, s_m: float, span_m: float, side: float) -> float:
    """Free road (m) beside the racing line on ``side`` (+1 left) along the next
    ``span_m``: the most a move there can shift."""
    track = driver.track
    k = max(2, int(span_m / track.ds_m))
    idx = (int(s_m / track.ds_m) + np.arange(k)) % track.n
    alpha = driver.line.alpha[idx]
    free = driver.room_l[idx] - alpha if side > 0 else driver.room_r[idx] + alpha
    return float(np.min(free))


def _after(
    start: float,
    cur: float,
    targets: list[tuple[float, float]],
    brake_fraction: float | None = None,
) -> Maneuver:
    """Knots from the car's current shift, with distances forced to increase."""
    knots = [(0.0, cur)]
    for d, shift in targets:
        knots.append((max(d, knots[-1][0] + 5.0), shift))
    return Maneuver(start, tuple(knots), brake_fraction)


def make_move(
    name: str,
    driver: PathDriver,
    s_m: float,
    shift_now: float,
    threat_shift: float = 0.0,
    amplitude: float = MAX_SHIFT_M,
) -> Maneuver:
    """The `Maneuver` for move ``name`` started now at ``s_m``, with the car
    ``shift_now`` m off the racing line. ``threat_shift`` is where the car behind
    sits relative to the line (for ``cover``)."""
    if name in LANE_MOVES:
        return lane_move(name, driver, s_m, shift_now)
    ahead = [c for c in corners_ahead(driver, s_m) if c.apex_m > 5.0]
    if name in PULL_MOVES:
        return pull_move(name, driver, s_m, shift_now, ahead)
    # Not on if the nearest corner's braking zone is already here or too close.
    corners = ahead if ahead and ahead[0].brake_m >= MIN_ROOM_M else []
    a = min(amplitude, MAX_SHIFT_M)
    if name == "line" or not corners:
        # Back to the line (a small error is just tracking noise: do not adopt it).
        start = shift_now if abs(shift_now) > 0.5 else 0.0
        return Maneuver(s_m, ((0.0, start), (RAMP_M, 0.0)))

    c = corners[0]
    end = c.brake_m - 20.0  # the sideways move is finished before the braking zone
    ramp = min(RAMP_M, end)
    wait: list[
        tuple[float, float]
    ] = []  # the move starts now; WHEN is the driver's call
    back = c.apex_m + RECOVER_M
    if len(corners) > 1:  # be back on the line before the next braking zone
        back = min(back, max(c.apex_m + 40.0, corners[1].brake_m - 30.0))
    hold = min(c.apex_m + 30.0, (c.apex_m + back) / 2)

    def sized(side: float) -> float:
        """Largest shift toward ``side`` the road beside the line allows."""
        free = (
            room_beside(driver, s_m + end - ramp, ramp + 20.0, side) - EDGE_CLEARANCE_M
        )
        return min(a, free)

    # Outside/inside are named for the corner; the room that matters is on that side.
    toward = {"outside": -c.side, "divebomb": c.side, "dummy": c.side, "inside": c.side}
    if name in toward:
        a = sized(toward[name])
        if a < MIN_SHIFT_M:  # no room: the move is not on
            return make_move("line", driver, s_m, shift_now)

    if name == "outside":
        return _after(
            s_m,
            shift_now,
            [*wait, (ramp, -c.side * a), (hold, -c.side * a), (back, 0.0)],
        )
    if name == "divebomb":
        return _after(
            s_m,
            shift_now,
            [*wait, (ramp, c.side * a), (hold, c.side * a * 0.6), (back, 0.0)],
            brake_fraction=DIVE_BRAKE_FRACTION,
        )
    if name == "inside":
        return _after(
            s_m,
            shift_now,
            [
                *wait,
                (ramp, c.side * a * 0.8),
                (min(c.apex_m + 10.0, hold), c.side * a * 0.5),
                (back, 0.0),
            ],
        )
    if name == "dummy":
        if end < 150.0:  # no room for a feint: it is a plain divebomb
            return make_move("divebomb", driver, s_m, shift_now, amplitude=amplitude)
        return _after(
            s_m,
            shift_now,
            [
                (30.0, -c.side * 1.8),
                (60.0, -c.side * 1.8),
                (min(end, 160.0), c.side * a),
                (hold, c.side * a * 0.6),
                (back, 0.0),
            ],
            brake_fraction=DIVE_BRAKE_FRACTION,
        )
    if name == "switchback":
        # Hold the line through this corner; dive at the next one if there is one.
        if len(corners) < 2:
            return make_move("line", driver, s_m, shift_now)
        d = corners[1]
        ramp2 = min(RAMP_M, max(20.0, d.brake_m - c.apex_m - 40.0))
        return _after(
            s_m,
            shift_now,
            [
                (c.apex_m + 20.0, 0.0),
                (c.apex_m + 20.0 + ramp2, d.side * a),
                (d.apex_m + 30.0, d.side * a * 0.6),
                (d.apex_m + RECOVER_M, 0.0),
            ],
            brake_fraction=DIVE_BRAKE_FRACTION,
        )
    if name == "cover":
        target = float(np.clip(threat_shift, -a, a))
        return _after(
            s_m,
            shift_now,
            [
                (min(60.0, max(20.0, c.brake_m - 20.0)), target),
                (hold, target),
                (back, 0.0),
            ],
        )
    raise ValueError(f"unknown move {name!r}")


def lane_move(name: str, driver: PathDriver, s_m: float, shift_now: float) -> Maneuver:
    """A `LANE_MOVES` move: ease to the named offset from the racing line over
    `LANE_RAMP_M` and hold it (as far as the road beside the line allows over the
    next `LANE_LOOK_M`), braking late (`DIVE_BRAKE_FRACTION`) if ``_late``."""
    late = name.endswith("_late")
    want = float(name.removeprefix("lane").removesuffix("_late"))
    if want:
        side = 1.0 if want > 0 else -1.0
        free = room_beside(driver, s_m, LANE_RAMP_M + LANE_LOOK_M, side)
        want = side * max(0.0, min(abs(want), free - EDGE_CLEARANCE_M))
    return Maneuver(
        s_m,
        ((0.0, shift_now), (LANE_RAMP_M, want)),
        DIVE_BRAKE_FRACTION if late else None,
    )


def pull_move(
    name: str, driver: PathDriver, s_m: float, shift_now: float, ahead: list[Corner]
) -> Maneuver:
    """``pull_in`` / ``pull_out``: shift toward the inside / outside of the next
    corner (the one we are in, if before its apex) right away, hold it to just past
    that apex (at least `PULL_HOLD_M`), then rejoin the line. Without a corner
    ahead, toward the side with more road. Not on if there is no room."""
    if ahead:
        c = ahead[0]
        side = c.side if name == "pull_in" else -c.side
        hold = max(PULL_HOLD_M, c.apex_m + 30.0)
        if len(ahead) > 1:  # back on the line before the next braking zone
            hold = min(hold, max(c.apex_m + 30.0, ahead[1].brake_m - 60.0))
    else:
        room = {sd: room_beside(driver, s_m, PULL_HOLD_M, sd) for sd in (1.0, -1.0)}
        side = max(room, key=lambda sd: room[sd])
        if name == "pull_out":
            side = -side
        hold = PULL_HOLD_M
    free = room_beside(driver, s_m + PULL_RAMP_M, 60.0, side) - EDGE_CLEARANCE_M
    amount = min(PULL_SHIFT_M, free)
    if amount < PULL_MIN_M:
        return make_move("line", driver, s_m, shift_now)
    target = side * amount
    return _after(
        s_m,
        shift_now,
        [(PULL_RAMP_M, target), (hold, target), (hold + RECOVER_M, 0.0)],
    )
