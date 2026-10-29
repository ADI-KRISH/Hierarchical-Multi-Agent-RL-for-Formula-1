"""`RaceEnv` (env v2): a Gymnasium env where the car steers and so chooses its line.

`DriverEnv` (v1) keeps the car on the centreline and only lets the lateral offset
count toward going off track. Here the car is a Frenet-frame point mass
(`models.frenet`): its offset and heading change the corner it drives, and one
friction ellipse is shared by throttle/brake and steering. v1 stays for comparison.

Design follows AM-RL (docs/racing_line_research.md, 2.4):

- **Action mapping.** Action ``[throttle/brake, steer]`` in [-1, 1]^2 is mapped onto
  what the tyres can deliver (`models.frenet.map_action`): the policy cannot ask for
  more grip than exists, so "full lock and full throttle" means the limit.
- **Observation** is generic geometry, with no lap-progress feature (it would let a
  policy memorise one track): speed, room to each edge, heading error, the previous
  command, the tightest corner in each lookahead window as a signed number
  (``sign * (1 - limit speed / top speed)``, so a left hairpin is large and
  positive), the edges ahead, the centreline's own limit speed here, and a style
  input held at its default (Phase 8 hook).
- **Random starts** (training): anywhere on the lap, any speed and offset, for a
  fixed time budget. Reward is distance advanced along the track, so going faster
  scores higher with no lap-end special case. ``mode="lap"`` instead starts at the
  line with the reference speed and ends on crossing it, timing a flying lap.

- **Other cars and no crashes.** With ``opponents`` one scripted car shares the
  track; with ``safety_filter`` both cars' commands go through `models.safety`, so
  neither ever drives into the other (`info` counts how often it stepped in).
  ``reset(options=...)`` can fix the scene: ``start_s`` (on the racing line there),
  ``opponent``, and ``gap_m`` or ``gap_s`` (a real gap in seconds, `f1rl.real_starts`).

`info` carries `distance_m` (unwrapped along the track), `speed_ms`,
`lateral_offset_m`, `elapsed_s`, `off_track`, `lap_completed`, and `lap_time_s`.
"""

import itertools
import math
from dataclasses import replace
from typing import Any

import gymnasium as gym
import numpy as np
from numpy.typing import NDArray

from f1rl.agents.line_follower import LineFollower
from f1rl.agents.path_driver import PathDriver
from f1rl.config import (
    GRAVITY_M_S2,
    OPPONENT_KINDS,
    CarParams,
    FollowerParams,
    RaceEnvParams,
    SimParams,
    TireParams,
)
from f1rl.envs.opponents import Opponent, place, wrap_gap
from f1rl.envs.racetrack import RaceTrack
from f1rl.models.frenet import FrenetState, frenet_step, map_action
from f1rl.models.raceline import OptimalLine, gg_speed_profile
from f1rl.models.safety import SafetyCheck, safe_command
from f1rl.models.tire import tire_grip, tire_wear_step

ObsType = NDArray[np.float32]
ActType = NDArray[np.float32]


class RaceEnv(gym.Env[ObsType, ActType]):
    def __init__(
        self,
        car: CarParams,
        track: RaceTrack,
        sim: SimParams,
        params: RaceEnvParams | None = None,
        reference: OptimalLine | None = None,
        mode: str = "random",
        tire: TireParams | None = None,
    ) -> None:
        if mode not in ("random", "lap"):
            raise ValueError(f"mode must be 'random' or 'lap', not {mode!r}")
        self.car = car
        self.track = track
        self.sim = sim
        self.params = params or RaceEnvParams()
        self.tire = tire or TireParams()
        self.reference = reference
        self.mode = mode
        p = self.params

        n = track.n
        self._n = n
        self._ds = track.ds_m
        self._lap_m = track.length_m
        hw = p.car_half_width_m
        self._kappa = track.kappa.tolist()
        self._room_left = (track.w_left - hw).tolist()  # centreline -> allowed centre
        self._room_right = (track.w_right - hw).tolist()
        if min(self._room_left) + min(self._room_right) <= 0:
            raise ValueError("track narrower than the car")

        self._v_centre = gg_speed_profile(car, track.ds_m, track.kappa)
        ay_max = car.max_lateral_g * GRAVITY_M_S2
        windows = self._window_features(track.kappa, ay_max)
        offsets = [round(d / self._ds) for d in p.edge_points_m]
        edges = [
            np.roll(r, -o) / p.edge_scale_m
            for o in offsets
            for r in (np.array(self._room_left), np.array(self._room_right))
        ]
        self._static = np.column_stack(
            [windows, *edges, self._v_centre / car.max_speed_ms]
        ).astype(np.float32)

        n_obs = 6 + len(p.lookahead_windows_m) + 2 * len(p.edge_points_m) + 1 + 1
        self._ref_static: NDArray[np.float32] | None = None
        if p.reference_features:
            if reference is None:
                raise ValueError("reference_features needs a reference line")
            self._ref_static = self._reference_features(reference)
            n_obs += 2 * len(p.reference_points_m) + 1
        self._followers: dict[float, Any] = {}
        self._base = np.zeros(2, dtype=np.float32)
        self._wear = 0.0
        self._push = 1.0
        self._car_now = car
        if p.residual_scale > 0:
            if reference is None:
                raise ValueError("residual_scale needs a reference line")
            n_obs += 2
        if p.tactical and reference is None:
            raise ValueError("tactical needs a reference line")
        if p.tires:
            n_obs += 4
        self._opp: Opponent | None = None
        self._opp_drivers: dict[str, PathDriver] = {}
        self._line_ref: LineFollower | None = None
        self._ahead = True  # ahead of the opponent right now
        self._held = True  # the place we hold: changes once a pass has stuck
        self._since_m = 0.0  # metres driven since the order last changed
        self._passes = 0
        self._swaps = 0  # every change of order, stuck or not
        self._section_m: float | None = None  # a section battle's length
        self._section_done = self._section_won = False
        self._contact = False
        self._grip = 1.0
        self._plan_grip = 1.0
        self._interventions = 0  # times the safety filter changed our command
        self._opp_interventions = 0  # ...and the opponent's
        self._safety = SafetyCheck(
            lap_m=self._lap_m,
            car_length_m=p.car_length_m,
            car_width_m=p.car_width_m,
            margin_s_m=p.safety_margin_s_m,
            margin_n_m=p.safety_margin_n_m,
            horizon_s=p.safety_horizon_s,
            max_horizon_s=p.safety_max_horizon_s,
            dt_s=p.safety_dt_s,
            hold_s=sim.dt_s * p.action_repeat,
            lane_time_s=p.safety_lane_time_s,
            speed_floor_ms=p.min_speed_ms,
            steer=p.safety_steer,
            brake=p.safety_brake,
        )
        self._tow_now = 0.0
        if p.opponents:
            if not p.tires:
                raise ValueError("opponents need tires=True (dirty air is in the grip)")
            unknown = set(p.opponents) - set(OPPONENT_KINDS) - {"none"}
            if unknown:
                raise ValueError(f"unknown opponent kinds: {sorted(unknown)}")
            n_obs += 4
            if p.opponent_features:
                n_obs += 4
        self.action_space = gym.spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            -5.0, 5.0, shape=(n_obs,), dtype=np.float32
        )
        self._seeded = False
        self._state = FrenetState(0.0, 0.0, 0.0, 0.0)
        self._steps = 0
        self._last = (0.0, 0.0)
        self._opp_last = (0.0, 0.0)
        self._start_s = 0.0

    def _follower(self) -> Any:
        """The scripted base driver for the current ``push`` and grip, built once per
        (push level, grip to 0.01) -- it plans its speeds for that much grip."""
        p = self.params
        top = FollowerParams().grip_fraction
        frac = top
        if p.tires:
            level = min(p.push_levels, key=lambda lv: abs(lv - self._push))
            frac = p.conserve_grip_fraction + (top - p.conserve_grip_fraction) * level
        key = round(frac * self._plan_grip, 2)
        assert self.reference is not None  # residual_scale > 0 checked at init
        if key not in self._followers:
            fp = FollowerParams(grip_fraction=key)
            if p.tactical:
                self._followers[key] = PathDriver(
                    self.car,
                    self.track,
                    self.reference,
                    np.array(self._room_left),
                    np.array(self._room_right),
                    fp,
                    edge_inset_m=p.path_edge_inset_m,
                )
            else:
                self._followers[key] = LineFollower(
                    self.car, self.track, self.reference, fp
                )
        return self._followers[key]

    def _reference_features(self, ref: OptimalLine) -> NDArray[np.float32]:
        """Per station: the optimal line's lateral shift and limit speed at each of
        `reference_points_m` ahead, and its heading relative to the centreline. The
        env subtracts the car's own offset and heading at run time (hybrid, step 7:
        the agent sees where the reference line is and learns to follow and beat it).
        """
        p = self.params
        offsets = [round(d / self._ds) for d in p.reference_points_m]
        alpha = [np.roll(ref.alpha, -o) / p.edge_scale_m for o in offsets]
        speed = [np.roll(ref.speed, -o) / self.car.max_speed_ms for o in offsets]
        line_heading = np.arctan2(
            np.roll(ref.y, -1) - np.roll(ref.y, 1),
            np.roll(ref.x, -1) - np.roll(ref.x, 1),
        )
        psi_ref = np.angle(np.exp(1j * (line_heading - self.track.heading)))
        return np.column_stack([*alpha, *speed, psi_ref]).astype(np.float32)

    def _window_features(self, kappa: NDArray[np.float64], ay_max: float) -> Any:
        """Per station and lookahead window: the signed tightest-corner feature."""
        edges = (0.0, *self.params.lookahead_windows_m)
        cols = []
        for near, far in itertools.pairwise(edges):
            lo, hi = round(near / self._ds), max(round(far / self._ds), 1)
            best = np.zeros(len(kappa))
            for off in range(lo, hi):
                cand = np.roll(kappa, -off)
                better = np.abs(cand) > np.abs(best)
                best = np.where(better, cand, best)
            with np.errstate(divide="ignore"):
                limit = np.minimum(
                    self.car.max_speed_ms,
                    np.sqrt(ay_max / np.maximum(np.abs(best), 1e-9)),
                )
            cols.append(np.sign(best) * (1.0 - limit / self.car.max_speed_ms))
        return np.column_stack(cols)

    # -- geometry lookups (linear interpolation round the lap) ------------------
    def _at(self, values: list[float], s_m: float) -> float:
        pos = (s_m % self._lap_m) / self._ds
        i = int(pos)
        f = pos - i
        return values[i % self._n] * (1.0 - f) + values[(i + 1) % self._n] * f

    def _dirty_air(self, within_m: float) -> float:
        """Grip multiplier from another car's wake if it is lined up ahead and closer
        than ``within_m`` (1.0 = clean air)."""
        p, opp = self.params, self._opp
        if opp is None:
            return 1.0
        gap = wrap_gap(opp.state.s_m - self._state.s_m, self._lap_m)  # > 0: it is ahead
        lined_up = abs(opp.state.n_m - self._state.n_m) < 1.5 * p.car_width_m
        in_wake = p.car_length_m < gap <= within_m
        return 1.0 - p.dirty_air_loss if in_wake and lined_up else 1.0

    def _tow(self, rear: FrenetState, front: FrenetState) -> float:
        """Top-speed boost (m/s) for ``rear`` chasing ``front`` on a straight."""
        p = self.params
        gap = wrap_gap(front.s_m - rear.s_m, self._lap_m)
        # The assist ends 100 m before a bend: DRS closes ahead of the braking zone.
        straight = all(
            abs(self._kappa_at(rear.s_m + d)) < p.tow_max_curvature
            for d in (0.0, 50.0, 100.0)
        )
        return p.tow_boost_ms if p.car_length_m < gap <= p.tow_m and straight else 0.0

    def _update_grip(self) -> None:
        """Set the grip multiplier (tyres x wake) and the car it gives."""
        p = self.params
        g = tire_grip(self.tire, self._wear) if p.tires else 1.0
        # The base driver plans for the wake from further out (a braking zone is no
        # place to find out); the car feels it only inside `dirty_air_m`.
        self._plan_grip = g * self._dirty_air(p.wake_anticipation_m)
        g *= self._dirty_air(p.dirty_air_m)
        self._grip = g
        self._tow_now = self._tow(self._state, self._opp.state) if self._opp else 0.0
        self._car_now = replace(
            self.car,
            max_speed_ms=self.car.max_speed_ms + self._tow_now,
            max_lateral_g=self.car.max_lateral_g * g,
            max_accel_g=self.car.max_accel_g
            * g
            * (1.0 + (p.tow_accel_gain if self._tow_now > 0 else 0.0)),
            max_braking_g=self.car.max_braking_g * g,
        )

    def _kappa_at(self, s_m: float) -> float:
        return self._at(self._kappa, s_m)

    def _filter(
        self,
        car: CarParams,
        cmd: tuple[float, float],
        opp_car: CarParams,
        opp_cmd: tuple[float, float],
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """No crashes: both cars' commands through the safety filter, the car behind
        first, then the car ahead against what the car behind will really do."""
        opp = self._opp
        assert opp is not None
        my_pace = self._follower().speed_at if self.reference is not None else None

        def mine(other_cmd: tuple[float, float]) -> tuple[float, float]:
            out, hit = safe_command(
                car,
                self._state,
                cmd,
                opp_car,
                opp.state,
                map_action(opp_car, *other_cmd),
                self._kappa_at,
                self._room_at,
                self._safety,
                my_pace,
                opp.driver.speed_at,
            )
            self._interventions += hit
            return out

        def theirs(my_cmd: tuple[float, float]) -> tuple[float, float]:
            out, hit = safe_command(
                opp_car,
                opp.state,
                opp_cmd,
                car,
                self._state,
                map_action(car, *my_cmd),
                self._kappa_at,
                self._room_at,
                self._safety,
                opp.driver.speed_at,
                my_pace,
            )
            self._opp_interventions += hit
            return out

        if wrap_gap(opp.state.s_m - self._state.s_m, self._lap_m) > 0:  # we follow
            cmd = mine(opp_cmd)
            return cmd, theirs(cmd)
        opp_cmd = theirs(cmd)
        return mine(opp_cmd), opp_cmd

    def _recoverable(self, opp: FrenetState, gap_m: float) -> FrenetState:
        """The opponent's start speed, limited so that the car behind can still brake
        to the speed of the car in front within the gap (using half its braking): no
        episode starts with a crash nobody could avoid."""
        p = self.params
        room = max(0.0, abs(gap_m) - p.car_length_m - p.safety_margin_s_m)
        slack = math.sqrt(room * self.car.max_braking_g * GRAVITY_M_S2)
        me = self._state.speed_ms
        if gap_m < 0:  # it starts behind us: no more than we can both handle
            return replace(opp, speed_ms=min(opp.speed_ms, me + slack))
        return replace(opp, speed_ms=max(opp.speed_ms, me - slack))

    def _room_at(self, s_m: float) -> tuple[float, float]:
        """(room to the left, room to the right) for the car's centre at ``s_m``."""
        return self._at(self._room_left, s_m), self._at(self._room_right, s_m)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[ObsType, dict[str, Any]]:
        if seed is None and not self._seeded:
            seed = self.sim.seed
        super().reset(seed=seed)
        self._seeded = True
        p = self.params
        self._steps = 0
        self._last = (0.0, 0.0)
        self._opp_last = (0.0, 0.0)
        self._start_s = 0.0
        opts = options or {}
        self._push, self._wear = 1.0, 0.0
        if p.tires:
            random_start = self.mode == "random"
            self._push = float(
                opts.get(
                    "push",
                    self.np_random.choice(p.push_levels) if random_start else 1.0,
                )
            )
            self._wear = float(
                opts.get(
                    "wear",
                    self.np_random.uniform(0.0, p.start_wear_max)
                    if random_start
                    else 0.0,
                )
            )
        self._opp, self._passes, self._contact = None, 0, False
        self._swaps, self._since_m = 0, 0.0
        section = opts.get("section_m")
        self._section_m = None if section is None else float(section)
        self._section_done = self._section_won = False
        self._interventions = self._opp_interventions = 0
        self._grip = self._plan_grip = (
            tire_grip(self.tire, self._wear) if p.tires else 1.0
        )
        if self.mode == "lap":
            ref = self.reference
            s_start = float(opts.get("start_s", 0.0))  # a flying start part-way round
            if ref is None:
                self._state = FrenetState(0.0, 0.0, 0.0, float(self._v_centre[0]))
            else:
                if self._line_ref is None:
                    self._line_ref = LineFollower(self.car, self.track, ref)
                n0, psi0, v0 = self._line_ref.line_state(s_start)
                v0 = float(ref.speed[0]) if s_start == 0.0 else min(v0, 0.95 * v0)
                if (p.residual_scale > 0 or p.tactical) and s_start == 0.0:
                    v0 = min(v0, self._follower().speed_at(0.0))
                self._state = FrenetState(s_start, n0, psi0, v0)
        elif "start_s" in opts and self.reference is not None:
            # A set start (a real-race scenario): on the racing line at its speed.
            if self._line_ref is None:
                self._line_ref = LineFollower(self.car, self.track, self.reference)
            s0 = float(opts["start_s"]) % self._lap_m
            n0, psi0, v0 = self._line_ref.line_state(s0)
            self._state = FrenetState(s0, n0, psi0, max(0.95 * v0, p.min_speed_ms + 2))
        else:
            s0 = float(self.np_random.uniform(0.0, self._lap_m))
            i0 = int(s0 / self._ds) % self._n
            lo = -self._at(self._room_right, s0) * p.start_offset_frac
            hi = self._at(self._room_left, s0) * p.start_offset_frac
            v0 = min(
                float(self.np_random.uniform(*p.start_speed_ms)),
                0.95 * float(self._v_centre[i0]),
            )
            self._state = FrenetState(
                s0,
                float(self.np_random.uniform(lo, hi)),
                float(
                    self.np_random.uniform(-p.start_heading_rad, p.start_heading_rad)
                ),
                max(v0, p.min_speed_ms + 2.0),
            )
        self._start_s = self._state.s_m
        kind = opts.get("opponent")
        if p.opponents and kind is None and self.mode == "random":
            kind = str(self.np_random.choice(p.opponents))
            kind = None if kind == "none" else kind
        if kind is not None:
            grip, where = OPPONENT_KINDS[kind]
            if "gap_s" in opts:  # a real gap in seconds -> metres at our speed
                gap = float(
                    np.clip(float(opts["gap_s"]) * self._state.speed_ms, *p.real_gap_m)
                )
            else:
                gap = float(
                    opts.get("gap_m", self.np_random.uniform(*p.opponent_gap_m))
                )
            sign = 1.0 if where == "ahead" else -1.0
            assert self.reference is not None  # opponents need tires -> residual
            driver = self._opp_drivers.get(kind)
            if driver is None:
                driver = PathDriver(
                    self.car,
                    self.track,
                    self.reference,
                    np.array(self._room_left),
                    np.array(self._room_right),
                    FollowerParams(grip_fraction=grip),
                    edge_inset_m=p.path_edge_inset_m,
                )
                self._opp_drivers[kind] = driver
            side = float(self.np_random.choice([-1.0, 1.0]))
            self._opp = place(kind, driver, self._state.s_m + sign * gap, side, p.guard)
            self._ahead = self._held = where == "behind"
            if p.safety_filter:
                self._opp.state = self._recoverable(self._opp.state, sign * gap)
        self._update_grip()
        info = self._info(off_track=False, lap_completed=False)
        return self._observation(), info

    def step(
        self, action: ActType
    ) -> tuple[ObsType, float, bool, bool, dict[str, Any]]:
        p = self.params
        if p.residual_scale > 0:  # residual: follower's command + a correction
            action = self._base + p.residual_scale * np.clip(action, -1.0, 1.0)
        u_long = float(np.clip(action[0], -1.0, 1.0))
        u_lat = float(np.clip(action[1], -1.0, 1.0))
        car = self._car_now
        steps_before = self._steps
        dt = self.sim.dt_s
        opp = self._opp
        filter_cost = 0.0
        if opp is not None:
            opp_tow = self._tow(opp.state, self._state)
            opp_action = opp.command(
                self._state,
                self._lap_m,
                p.car_length_m + p.alongside_margin_m,
                p.car_width_m + 1.0,
                opp_tow,
                shield=p.follow_shield and (p.shield_in_tow or opp_tow == 0.0),
                give_way=p.opponent_yield,
                soft_follow=p.soft_follow,
                car_length_m=p.car_length_m,
            )
            opp_car = replace(
                self.car,
                max_speed_ms=self.car.max_speed_ms + opp_tow,
                max_accel_g=self.car.max_accel_g
                * (1.0 + (p.tow_accel_gain if opp_tow > 0 else 0.0)),
            )
            opp_cmd = (float(opp_action[0]), float(opp_action[1]))
            close = (
                abs(wrap_gap(opp.state.s_m - self._state.s_m, self._lap_m))
                < p.safety_check_m
            )
            if p.safety_filter and close:
                before_hits = self._interventions
                (u_long, u_lat), opp_cmd = self._filter(
                    car, (u_long, u_lat), opp_car, opp_cmd
                )
                # Racing so close that the filter must step in costs reward, so the
                # driver learns moves that need no rescue (GT Sophy penalises every
                # collision; here the filter prevents them, so its use is the cost).
                filter_cost = p.intervention_penalty * (
                    self._interventions - before_hits
                )
            ox, oy = map_action(opp_car, *opp_cmd)
            self._opp_last = opp_cmd
        a_x, a_y = map_action(car, u_long, u_lat)
        start_s = self._state.s_m
        gap_opp_before = opp.state.s_m if opp is not None else 0.0
        total = -filter_cost
        terminated = lap_completed = off_track = False
        info: dict[str, Any] = {}
        for _ in range(p.action_repeat):
            before = self._state
            self._state = frenet_step(
                car, before, a_x, a_y, self._kappa_at, dt, p.min_speed_ms
            )
            self._steps += 1
            now = self._state
            if opp is not None:
                opp.state = frenet_step(
                    opp_car, opp.state, ox, oy, self._kappa_at, dt, p.min_speed_ms
                )
                near = abs(wrap_gap(opp.state.s_m - now.s_m, self._lap_m))
                if (
                    near < p.car_length_m
                    and abs(opp.state.n_m - now.n_m) < p.car_width_m
                ):
                    self._contact = True
                    total -= p.fail_penalty
                    terminated = True
                    break
            total += (now.s_m - before.s_m) / p.progress_scale_m
            limit = p.track_limit_margin_m
            off_track = not (
                -self._at(self._room_right, now.s_m) - limit
                <= now.n_m
                <= self._at(self._room_left, now.s_m) + limit
            )
            wrong_way = abs(now.psi_rad) > p.max_heading_rad
            too_slow = now.speed_ms < p.min_speed_ms
            lap_completed = self.mode == "lap" and now.s_m >= self._lap_m
            if off_track or wrong_way or too_slow:
                total -= p.fail_penalty
                terminated = True
            if lap_completed:
                terminated = True
            if terminated:
                if lap_completed:
                    frac = (self._lap_m - before.s_m) / max(now.s_m - before.s_m, 1e-9)
                    info["lap_time_s"] = (self._steps - 1 + frac) * dt
                break
        if p.tires:
            # Tyre load: total force against the grip a tyre has (the lateral limit),
            # so full throttle (1.5 g of 5 g) is light work and braking/cornering heavy.
            usage = min(1.0, math.hypot(a_x, a_y) / (car.max_lateral_g * GRAVITY_M_S2))
            worn = tire_wear_step(
                self.tire, self._wear, usage, (self._steps - steps_before) * dt
            )
            total -= (1.0 - self._push) * p.wear_cost_per_unit * (worn - self._wear)
            self._wear = worn
        if opp is not None and not self._contact and p.pass_reward_per_m > 0:
            # GT Sophy's passing reward (Wurman et al., Nature 2022): metres gained on
            # the opponent while it is close (behind <= pass_window_m[0], ahead <=
            # pass_window_m[1]), paid both ways -- ground given back costs the same.
            behind_m, ahead_m = p.pass_window_m
            gap_before = wrap_gap(start_s - gap_opp_before, self._lap_m)
            gap_after = wrap_gap(self._state.s_m - opp.state.s_m, self._lap_m)
            near = any(-behind_m <= -g <= ahead_m for g in (gap_before, gap_after))
            if near:
                total += p.pass_reward_per_m * (gap_after - gap_before)
        if opp is not None and not self._contact:
            # A pass counts (and pays) only once it has stuck: the new order held for
            # `pass_hold_m` of driving. Passing and being passed straight back pays
            # nothing; so does being passed and taking the place straight back.
            ahead = wrap_gap(self._state.s_m - opp.state.s_m, self._lap_m) > 0
            self._since_m += max(0.0, self._state.s_m - start_s)
            if ahead != self._ahead:
                self._ahead, self._since_m = ahead, 0.0
                self._swaps += 1
            if self._ahead != self._held and self._since_m >= p.pass_hold_m:
                self._held = self._ahead
                self._passes += 1 if self._held else -1
                total += p.overtake_bonus if self._held else -p.overtake_bonus
        self._update_grip()
        du = abs(u_long - self._last[0]) + abs(u_lat - self._last[1])
        total -= p.smooth_penalty * du
        self._last = (u_long, u_lat)
        budget = (
            p.episode_s / dt if self.mode == "random" else self.sim.max_episode_steps
        )
        section_over = False
        if self._section_m is not None and opp is not None and not terminated:
            # A section battle (`f1rl.sections`): ahead by the section's end, or out.
            progress = self._state.s_m - self._start_s
            if not self._section_done and progress >= self._section_m:
                self._section_done, self._section_won = True, self._ahead
                if self._ahead:
                    total += p.section_win_bonus
                else:
                    total -= p.section_fail_penalty
                    section_over = True
            tail = self._section_m + p.pass_hold_m + p.section_tail_m
            if self._section_done and (self._passes >= 1 or progress >= tail):
                section_over = True
        truncated = not terminated and (self._steps >= budget or section_over)
        info.update(self._info(off_track=off_track, lap_completed=lap_completed))
        info["advanced_m"] = self._state.s_m - start_s
        return self._observation(), total, terminated, truncated, info

    def _opponent_features(self, opp: Opponent) -> NDArray[np.float32]:
        """See `RaceEnvParams.opponent_features`."""
        p, s = self.params, self._state
        lat_speed = opp.state.speed_ms * math.sin(opp.state.psi_rad)
        pending = 0.0
        if self._ahead != self._held and p.pass_hold_m > 0:
            frac = min(1.0, self._since_m / p.pass_hold_m)
            pending = (1.0 if self._ahead else -1.0) * (1.0 - frac)
        threat = 1.0 if self._tow(opp.state, s) > 0 else 0.0
        grip = OPPONENT_KINDS[opp.kind][0]
        edge = (grip - FollowerParams().grip_fraction) / 0.1
        return np.array(
            [np.clip(lat_speed / 5.0, -1.0, 1.0), pending, threat, edge],
            dtype=np.float32,
        )

    def _info(self, *, off_track: bool, lap_completed: bool) -> dict[str, Any]:
        s = self._state
        return {
            "distance_m": s.s_m,
            "progress_m": s.s_m - self._start_s,  # since this episode's start
            "speed_ms": s.speed_ms,
            "lateral_offset_m": s.n_m,
            "heading_rad": s.psi_rad,
            "wear": self._wear,
            "push": self._push,
            "contact": self._contact,
            "passes": self._passes,
            "swaps": self._swaps,
            "interventions": self._interventions,
            "opp_interventions": self._opp_interventions,
            "elapsed_s": self._steps * self.sim.dt_s,
            "off_track": off_track,
            "lap_completed": lap_completed,
            "section_done": self._section_done,
            "section_won": self._section_won,
        }

    def _observation(self) -> ObsType:
        p = self.params
        s = self._state
        pos = (s.s_m % self._lap_m) / self._ds
        i = int(pos) % self._n
        f = pos - int(pos)
        static = self._static[i] * (1.0 - f) + self._static[(i + 1) % self._n] * f
        n_win = len(p.lookahead_windows_m)
        head = np.array(
            [
                s.speed_ms / self.car.max_speed_ms,
                np.clip(
                    (self._at(self._room_left, s.s_m) - s.n_m) / p.edge_scale_m, -1, 2
                ),
                np.clip(
                    (self._at(self._room_right, s.s_m) + s.n_m) / p.edge_scale_m, -1, 2
                ),
                np.clip(s.psi_rad / 0.5, -3.0, 3.0),
                self._last[0],
                self._last[1],
            ],
            dtype=np.float32,
        )
        parts = [head, static[:n_win], static[n_win:-1], static[-1:], [p.style]]
        if self._ref_static is not None:
            ref = (
                self._ref_static[i] * (1.0 - f)
                + self._ref_static[(i + 1) % self._n] * f
            ).copy()
            k = len(p.reference_points_m)
            ref[:k] -= s.n_m / p.edge_scale_m  # shift ahead, relative to the car
            ref[-1] = (s.psi_rad - ref[-1]) / 0.5  # heading error to the line
            ref[:k] = np.clip(ref[:k], -3.0, 3.0)
            ref[-1] = np.clip(ref[-1], -3.0, 3.0)
            parts.append(ref)
        if p.tires:
            parts.append(
                np.array(
                    [
                        self._wear,
                        self._push,
                        tire_grip(self.tire, self._wear),
                        self._tow_now / 5.0,
                    ],
                    dtype=np.float32,
                )
            )
        if p.residual_scale > 0:
            self._base = self._follower().act(
                {
                    "distance_m": s.s_m,
                    "lateral_offset_m": s.n_m,
                    "heading_rad": s.psi_rad,
                    "speed_ms": s.speed_ms,
                }
            )
            parts.append(self._base)
        if p.opponents:
            opp = self._opp
            extra = 4 if p.opponent_features else 0
            if opp is None:
                parts.append(np.zeros(4 + extra, dtype=np.float32))
            else:
                gap = wrap_gap(opp.state.s_m - s.s_m, self._lap_m)  # > 0: it is ahead
                parts.append(
                    np.array(
                        [
                            1.0 if abs(gap) < 150.0 else 0.0,
                            np.clip(gap / 150.0, -1.0, 1.0),
                            np.clip((opp.state.n_m - s.n_m) / 10.0, -1.0, 1.0),
                            np.clip(
                                (opp.state.speed_ms - s.speed_ms) / 30.0, -1.0, 1.0
                            ),
                        ],
                        dtype=np.float32,
                    )
                )
                if p.opponent_features:
                    parts.append(self._opponent_features(opp))
        return np.concatenate(parts).astype(np.float32)
