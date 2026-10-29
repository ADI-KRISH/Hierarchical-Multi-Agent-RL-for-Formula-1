"""Collision-avoidance filter: no car may drive into another.

Every car's command passes through `safe_command` before it is applied. A command
is safe if, after holding it for one decision, both cars can still fall back on
the rules of the road and never touch: each straightens up in its lane
(`lane_keep`), the car behind brakes as hard as it can and the car in front keeps
its pace (`backup`). The filter checks that a short time ahead, integrating both
cars with `frenet_step`. So a car may close right up to set a move up, but never
past the point where braking would no longer save it. If the cars would come within
a car length and a car width of each other, the command is swapped for the least
intrusive one that keeps them apart and on the track:

- a car with the other one **ahead** brakes first (the car behind is the one who
  must avoid the accident), then brakes and steers away;
- a car **alongside or ahead** steers away first (it gives room), then also brakes;
  a car **ahead** that is braking first tries braking less (no brake-testing).

The fallback the check assumes (keep the lane, brake if behind) is always a
candidate too, so a command judged safe now leaves a safe command next time.
If no candidate is clear, the one that leaves the most room to the other car is
used. A replacement takes a car off the road only to avoid a crash: when that is
the only way to keep clear, or, if nothing keeps clear, when off the road leaves
more room to the other car than any command on it. The car behind is filtered
first, and the car ahead is then checked against what the car behind will really
do. A command that keeps clear is never touched, so clean racing is left alone.
Pure functions, no I/O, no globals.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass

from f1rl.config import GRAVITY_M_S2, CarParams
from f1rl.models.frenet import FrenetState, frenet_step, map_action

Command = tuple[float, float]  # (throttle/brake, steer) in [-1, 1]
Pace = Callable[[float], float] | None  # a car's planned speed (m/s) at distance s


@dataclass(frozen=True)
class SafetyCheck:
    """What the filter needs to know about the cars and the road."""

    lap_m: float
    car_length_m: float
    car_width_m: float
    margin_s_m: float  # extra gap kept along the track
    margin_n_m: float  # extra gap kept across the track
    horizon_s: float  # look at least this far ahead (s)...
    max_horizon_s: float  # ...and up to this, the faster the cars close
    dt_s: float
    hold_s: float  # how long a command is held before the next decision
    lane_time_s: float  # how fast a car straightens up after its command (s)
    speed_floor_ms: float
    steer: float  # how much steer the "steer away" candidates add
    brake: float  # braking command of the "brake" candidates


def wrap(gap_m: float, lap_m: float) -> float:
    """``gap_m`` folded into [-lap/2, lap/2)."""
    return (gap_m + lap_m / 2) % lap_m - lap_m / 2


def separation(a: FrenetState, b: FrenetState, check: SafetyCheck) -> float:
    """How far apart two cars are, in metres past touching: > 0 is clear. The larger
    of the along-track and across-track clearances (cars only touch when both are
    gone)."""
    along = abs(wrap(b.s_m - a.s_m, check.lap_m)) - (
        check.car_length_m + check.margin_s_m
    )
    across = abs(b.n_m - a.n_m) - (check.car_width_m + check.margin_n_m)
    return max(along, across)


def lane_keep(
    car: CarParams, state: FrenetState, a_x: float, kappa: float, check: SafetyCheck
) -> float:
    """Lateral acceleration that turns the car parallel to the track (its heading to
    the centreline decaying in `lane_time_s`) at its current offset -- as much of it
    as the grip left over from ``a_x`` allows (the friction ellipse)."""
    v = max(state.speed_ms, check.speed_floor_ms)
    ds = state.speed_ms * math.cos(state.psi_rad) / max(0.1, 1.0 - state.n_m * kappa)
    a_y = v * (kappa * ds - state.psi_rad / check.lane_time_s)
    ax_max = (car.max_accel_g if a_x >= 0 else car.max_braking_g) * GRAVITY_M_S2
    used = min(1.0, abs(a_x) / ax_max)
    limit = car.max_lateral_g * GRAVITY_M_S2 * math.sqrt(1.0 - used * used)
    return max(-limit, min(limit, a_y))


def backup(
    car: CarParams,
    state: FrenetState,
    pedal: float,
    follower: bool,
    kappa: float,
    check: SafetyCheck,
    pace: Pace = None,
) -> tuple[float, float]:
    """The rules-of-the-road fallback (a_x, a_y): keep the lane; the car behind
    brakes with all the grip the lane leaves; the car in front keeps its pedal -- or,
    given its ``pace`` (its planned speed along the track), drives to that plan: it
    brakes into corners and accelerates out of them, it does not stop on the track."""
    if not follower:
        if pace is not None:
            # Follow the plan: its slope (braking zones brake hard) plus a pull
            # back onto it.
            g, v, s = GRAVITY_M_S2, state.speed_ms, state.s_m
            want = v * (pace(s + 1.0) - pace(s)) + (pace(s) - v) / check.lane_time_s
            pedal = max(-car.max_braking_g * g, min(car.max_accel_g * g, want))
        return pedal, lane_keep(car, state, pedal, kappa, check)
    a_y = lane_keep(car, state, 0.0, kappa, check)
    ay_max = car.max_lateral_g * GRAVITY_M_S2
    left = math.sqrt(max(0.0, 1.0 - (a_y / ay_max) ** 2))
    return -car.max_braking_g * GRAVITY_M_S2 * left, a_y


def horizon(
    car: CarParams, me: FrenetState, other: FrenetState, check: SafetyCheck
) -> float:
    """How far ahead to look (s): long enough to brake away twice the closing speed,
    so a car closing fast is caught while there is still room to stop."""
    ahead = wrap(other.s_m - me.s_m, check.lap_m) > 0
    closing = me.speed_ms - other.speed_ms if ahead else other.speed_ms - me.speed_ms
    t = 2.0 * max(0.0, closing) / (car.max_braking_g * GRAVITY_M_S2)
    return min(check.max_horizon_s, max(check.horizon_s, t))


def rollout(
    car: CarParams,
    me: FrenetState,
    accel: tuple[float, float],
    other_car: CarParams,
    other: FrenetState,
    other_accel: tuple[float, float],
    kappa_at: Callable[[float], float],
    room_at: Callable[[float], tuple[float, float]],
    check: SafetyCheck,
    my_pace: Pace = None,
    other_pace: Pace = None,
) -> tuple[float, float]:
    """Integrate both cars over the horizon: (smallest separation reached after
    now, smallest room left to the track edge -- < 0 is off the road)."""
    steps = max(1, round(horizon(car, me, other, check) / check.dt_s))
    hold = round(check.hold_s / check.dt_s)
    behind = wrap(other.s_m - me.s_m, check.lap_m) > 0  # I am the car behind
    worst = edge = math.inf
    for k in range(steps):
        a, b = accel, other_accel
        if k >= hold:  # after the decision: the rules-of-the-road fallback
            a = backup(car, me, accel[0], behind, kappa_at(me.s_m), check, my_pace)
            b = backup(
                other_car,
                other,
                b[0],
                not behind,
                kappa_at(other.s_m),
                check,
                other_pace,
            )
        me = frenet_step(car, me, *a, kappa_at, check.dt_s, check.speed_floor_ms)
        other = frenet_step(
            other_car, other, *b, kappa_at, check.dt_s, check.speed_floor_ms
        )
        worst = min(worst, separation(me, other, check))
        left, right = room_at(me.s_m)
        edge = min(edge, left - me.n_m, me.n_m + right)
    return worst, edge


def candidates(
    command: Command, me: FrenetState, other: FrenetState, check: SafetyCheck
) -> list[Command]:
    """Safer commands to try, least intrusive first (see the module docstring)."""
    u_long, u_lat = command
    away = 1.0 if me.n_m >= other.n_m else -1.0  # +1: move left, away from it
    steer = max(-1.0, min(1.0, u_lat + away * check.steer))

    def brake(lat: float, hard: float) -> float:
        # Brake only with the grip the steering leaves: braking harder would shrink
        # the steering (`map_action`) and run the car wide -- off the road.
        return min(u_long, -min(hard, math.sqrt(max(0.0, 1.0 - lat * lat))))

    soft, full = brake(u_lat, check.brake), brake(u_lat, 1.0)
    soft_away, full_away = brake(steer, check.brake), brake(steer, 1.0)
    eased = 0.5 * u_lat  # stand on the brakes, easing off the steering
    stop = (brake(eased, 1.0), eased)
    other_ahead = wrap(other.s_m - me.s_m, check.lap_m) > check.car_length_m / 2
    if other_ahead:
        return [
            (soft, u_lat),
            (full, u_lat),
            (soft_away, steer),
            (full_away, steer),
            stop,
        ]
    out = [
        (u_long, steer),
        (soft_away, steer),
        (soft, u_lat),
        (full_away, steer),
        stop,
    ]
    other_behind = wrap(me.s_m - other.s_m, check.lap_m) > check.car_length_m / 2
    if other_behind and u_long < 0.0:
        # Braking with a car close behind: braking less (no brake test) comes first.
        out = [(0.5 * u_long, u_lat), (0.0, u_lat), (0.0, steer), *out]
    return out


def fallbacks(
    car: CarParams,
    me: FrenetState,
    command: Command,
    other: FrenetState,
    kappa: float,
    check: SafetyCheck,
) -> list[Command]:
    """The rules-of-the-road fallback itself, as commands: keep the lane at the
    command's pedal, and keep the lane braking as the car behind would. The filter
    judges every command assuming this fallback comes next, so it must also be one
    of the commands it can pick -- otherwise a car can be promised a way out (a
    tight lane-keeping turn) that none of the candidates can steer."""
    behind = wrap(other.s_m - me.s_m, check.lap_m) > 0
    pedal = map_action(car, *command)[0]
    ay_max = car.max_lateral_g * GRAVITY_M_S2
    out: list[Command] = []
    for follower in (behind, True):
        a_x, a_y = backup(car, me, pedal, follower, kappa, check)
        ax_max = (car.max_accel_g if a_x >= 0 else car.max_braking_g) * GRAVITY_M_S2
        cmd = (a_x / ax_max, a_y / ay_max)
        if cmd not in out:
            out.append(cmd)
    return out


def safe_command(
    car: CarParams,
    me: FrenetState,
    command: Command,
    other_car: CarParams,
    other: FrenetState,
    other_accel: tuple[float, float],
    kappa_at: Callable[[float], float],
    room_at: Callable[[float], tuple[float, float]],
    check: SafetyCheck,
    my_pace: Pace = None,
    other_pace: Pace = None,
) -> tuple[Command, bool]:
    """``command`` if it keeps clear of the other car, else the least intrusive safe
    replacement. Returns (command to apply, whether the filter stepped in)."""

    def judge(cmd: Command) -> tuple[float, float]:
        return rollout(
            car,
            me,
            map_action(car, *cmd),
            other_car,
            other,
            other_accel,
            kappa_at,
            room_at,
            check,
            my_pace,
            other_pace,
        )

    # Clear = never closer than the margin, or, for cars already inside it (racing
    # side by side), never closer than they are now.
    clear = min(0.0, separation(me, other, check)) - 1e-6
    worst, edge = judge(command)
    tries = candidates(command, me, other, check)
    tries += fallbacks(car, me, command, other, kappa_at(me.s_m), check)
    if worst > clear:
        # Only beside the other car (within a car length of touching): further
        # away the fallback's crude lane-keeping often "sees" a flat-out car run
        # wide, and swapping its command for a brake would slow a car that is
        # nowhere near anyone.
        if edge > 0.0 or worst > check.car_length_m:
            return command, False
        # Clear of the other car but heading off the road beside it: a car that
        # leaves the track next to another one tends to come back across it, so
        # take a replacement that stays on the road, if one is just as clear.
        for cmd in tries:
            sep, room = judge(cmd)
            if sep > clear and room > 0.0:
                return cmd, True
        return command, False
    # A replacement should not take the car off the road (or, if the command already
    # does, further off than it) -- unless that is the only way to avoid a crash.
    road = min(0.0, edge) - 1e-6
    best, least = command, clear - worst
    escape: Command | None = None  # clear, but only by leaving the road
    escape_room = -math.inf
    rescue: Command | None = None  # not clear and off the road, but furthest apart
    rescue_sep = -math.inf
    for cmd in tries:
        sep, room = judge(cmd)
        if room <= road:
            if sep > clear and room > escape_room:
                escape, escape_room = cmd, room
            if sep > rescue_sep:
                rescue, rescue_sep = cmd, sep
            continue
        if sep > clear:
            return cmd, True
        if clear - sep < least:
            best, least = cmd, clear - sep
    # Nothing on the road keeps clear: run off the road rather than into the car.
    if escape is not None:
        return escape, True
    # Nothing keeps clear at all: the most room to the other car, even if that is
    # off the road -- running wide is better than running into it.
    if rescue is not None and clear - rescue_sep < least:
        return rescue, True
    return best, best != command
