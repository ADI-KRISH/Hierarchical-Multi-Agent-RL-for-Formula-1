"""`StrategyEnv`: the Strategy Agent's race, one decision per lap.

A whole Grand Prix (305 km) against ``n_cars - 1`` rivals running rule-based
strategies (`agents.strategy_baseline`). Each step the agent picks, for the coming
lap, whether to pit and for which compound, and a push level for the driver. The
driver itself is a stand-in: a lap-time table from the track's optimal line
(`models.race.LapTable`), until the trained Driver Agent can replace it.

Action (MultiDiscrete [4, len(push_levels)]):
  0: 0 = stay out, 1/2/3 = pit for soft/medium/hard. On the first decision (on
     the grid) it picks the starting set at no cost; 0 = start on mediums.
  1: push level index (0 = conserve .. last = attack).

Reward (`StrategyRewardParams`): minus seconds lost per lap against a reference
lap, plus places gained, and at the flag a penalty per place behind the winner
and a disqualification penalty for running only one compound.
"""

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from f1rl.agents.strategy_baseline import (
    FixedPlan,
    baseline_rival,
    sample_rival_plan,
)
from f1rl.config import (
    COMPOUNDS,
    CarParams,
    StrategyBaselineParams,
    StrategyParams,
    TireParams,
)
from f1rl.driver_profile import DriverProfile
from f1rl.models.race import (
    CarState,
    LapPlan,
    LapTable,
    build_lap_table,
    car_lap,
    push_grip_fraction,
    race_laps,
    resolve_lap,
)
from f1rl.models.tire import tire_grip

Obs = np.typing.NDArray[np.float32]

#: Gaps (s) are divided by this and clipped to [0, 1] in the observation.
GAP_SCALE_S = 30.0
#: Two stops this many laps apart or fewer can make an undercut or overcut.
UNDERCUT_WINDOW_LAPS = 5
OBS_SIZE = 38


@dataclass(frozen=True, eq=False)
class RaceTrackInfo:
    """What the race needs of a track: its lap length and the stand-in driver."""

    name: str
    length_m: float
    table: LapTable

    @property
    def laps(self) -> int:
        return race_laps(self.length_m)


def load_race_track(
    name: str,
    car: CarParams,
    params: StrategyParams,
    cache_dir: Path | None = None,
    profile: DriverProfile | None = None,
) -> RaceTrackInfo:
    """A TUM track's lap-time table on its optimal line, cached as JSON under
    ``cache_dir`` (default data/strategy/) after the first build. With a measured
    ``profile`` that covers the track, lap times carry our driver's own pace gap;
    otherwise the stand-in's ``driver_gap``."""
    from f1rl.config import RaceEnvParams
    from f1rl.envs.racetrack import DATA_DIR, load_tum_track
    from f1rl.models.raceline import optimal_line

    folder = cache_dir or DATA_DIR / "strategy"
    key = (
        f"{name}_g{params.grip_min:g}-{params.grip_max:g}-{params.grip_step:g}"
        f"_f{params.start_fuel_kg:g}x{params.fuel_points}_d{params.driver_gap:g}"
        f"_m{car.mass_kg:g}_a{car.max_accel_g:g}_l{car.max_lateral_g:g}"
        f"_b{car.max_braking_g:g}_v{car.max_speed_ms:g}"
    )
    measured = profile is not None and name in profile.pace
    if measured:
        assert profile is not None
        pace = json.dumps(profile.pace[name]).encode()
        key += "_p" + hashlib.sha1(pace).hexdigest()[:10]
    path = folder / f"{key}.json"
    if path.exists():
        d = json.loads(path.read_text())
        table = LapTable(
            np.array(d["grips"]), np.array(d["fuels_kg"]), np.array(d["times_s"])
        )
        return RaceTrackInfo(name, float(d["length_m"]), table)
    track = load_tum_track(name)
    env = RaceEnvParams()
    line = optimal_line(track, car, env.car_half_width_m + env.line_safety_m)
    gap = None
    if measured:
        assert profile is not None
        prof = profile

        def gap(grip: float) -> float:
            value = prof.gap(name, grip)
            assert value is not None
            return value

    table = build_lap_table(car, line.ds, line.kappa, params, gap)
    length = float(np.sum(line.ds))
    folder.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "length_m": length,
                "grips": table.grips.tolist(),
                "fuels_kg": table.fuels_kg.tolist(),
                "times_s": table.times_s.tolist(),
            }
        )
    )
    return RaceTrackInfo(name, length, table)


class StrategyEnv(gym.Env[Obs, Any]):
    """One Grand Prix for our car against rule-based rivals; a step is a lap."""

    metadata = {"render_modes": []}  # noqa: RUF012

    def __init__(
        self,
        tracks: dict[str, RaceTrackInfo],
        params: StrategyParams,
        tire: TireParams | None = None,
        baseline: StrategyBaselineParams | None = None,
        profile: DriverProfile | None = None,
    ) -> None:
        if not tracks:
            raise ValueError("StrategyEnv needs at least one track")
        if params.n_cars < 2:
            raise ValueError("a race needs at least two cars")
        self.tracks = tracks
        self.params = params
        self.tire = tire or TireParams()
        self.baseline = baseline or StrategyBaselineParams()
        # Our driver's measured race-craft (None: the field's odds, `StrategyParams`).
        self.profile = profile
        self.action_space = spaces.MultiDiscrete([4, len(params.push_levels)])
        self.observation_space = spaces.Box(-1.0, 1.0, (OBS_SIZE,), np.float32)
        self.track = next(iter(tracks.values()))
        self.total_laps = self.track.laps
        self.lap = 0
        self.cars: list[CarState] = []
        self.plans: list[FixedPlan] = []
        self.order: list[int] = []
        self.ref_lap_s = 0.0
        self.last_lap_s = 0.0
        self.history: list[dict[str, Any]] = []
        self.orders: list[list[int]] = []
        self.stop_laps: list[list[int]] = []

    # --- helpers -----------------------------------------------------------------

    def _position(self, car: int = 0) -> int:
        return self.order.index(car)

    def _gap(self, a: int, b: int) -> float:
        return self.cars[b].time_s - self.cars[a].time_s

    def _obs(self) -> Obs:
        p = self.params
        me = self.cars[0]
        pos = self._position()
        ahead = self.order[pos - 1] if pos > 0 else None
        behind = self.order[pos + 1] if pos + 1 < len(self.order) else None
        n = self.total_laps
        one_hot = [1.0 if me.compound == c else 0.0 for c in COMPOUNDS]
        gap_ahead = -self._gap(ahead, 0) if ahead is not None else GAP_SCALE_S
        gap_behind = self._gap(0, behind) if behind is not None else GAP_SCALE_S
        leader = self.order[0]
        obs = [
            self.lap / n,
            (n - self.lap) / n,
            pos / (p.n_cars - 1),
            min(1.0, gap_ahead / GAP_SCALE_S),
            min(1.0, gap_behind / GAP_SCALE_S),
            *one_hot,
            me.wear,
            min(1.0, me.tyre_laps / n),
            me.fuel_kg / max(1e-9, p.start_fuel_kg),
            1.0 if len(me.compounds_used) >= 2 else 0.0,
            min(1.0, me.stops / 3),
            min(1.0, p.pit_loss_s / self.ref_lap_s),
            float(np.clip((self.last_lap_s - self.ref_lap_s) / 10.0, -1.0, 1.0)),
            self.cars[ahead].wear if ahead is not None else 0.0,
            self.cars[behind].wear if behind is not None else 0.0,
            min(1.0, self._gap(leader, 0) / (3 * GAP_SCALE_S)),
            self.cars[ahead].tyre_laps / n if ahead is not None else 0.0,
            self.cars[behind].tyre_laps / n if behind is not None else 0.0,
            # Undercut / overcut: where a stop now would put us, and who still has
            # to stop.
            *self._rejoin(),
            float(np.clip((self.cars[ahead].stops - me.stops) / 2, -1, 1))
            if ahead is not None
            else 0.0,
            float(np.clip((self.cars[behind].stops - me.stops) / 2, -1, 1))
            if behind is not None
            else 0.0,
            COMPOUNDS.index(self.cars[ahead].compound) / 2
            if ahead is not None
            else 0.0,
            # Tyre delta: s/lap a fresh set of each compound would gain now, and
            # our tyres' pace against the cars ahead and behind on theirs.
            *[
                float(np.clip((now - self._tyre_lap(me, c, 0.0)) / 5.0, -1, 1))
                for c in COMPOUNDS
                for now in (self._tyre_lap(me, me.compound, me.wear),)
            ],
            self._tyre_gap(ahead),
            self._tyre_gap(behind),
            # Tyre life left: laps to the wear limit at normal push / laps to go.
            min(1.0, self._life_laps(me.compound, me.wear) / max(1, n - self.lap)),
            # Pit window: could a new set of each compound run to the flag?
            *[
                min(1.0, self._life_laps(c, 0.0) / max(1, n - self.lap) / 2)
                for c in COMPOUNDS
            ],
            # Compounds run so far (the rule needs two).
            *[1.0 if c in me.compounds_used else 0.0 for c in COMPOUNDS],
        ]
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _tyre_lap(self, car: CarState, compound: str, wear: float) -> float:
        """A car's lap time (s) at normal push on ``compound`` worn ``wear``."""
        p = self.params
        normal = push_grip_fraction(p, p.push_levels[self.baseline.push_index])
        grip = normal * p.compound(compound).grip * tire_grip(self.tire, wear)
        return self.track.table.lap_time_s(grip * car.pace, car.fuel_kg)

    def _tyre_gap(self, other: int | None) -> float:
        """Our lap time minus ``other``'s on our own tyres (s, /2, clipped): > 0
        when its tyres are quicker than ours."""
        if other is None:
            return 0.0
        me, them = self.cars[0], self.cars[other]
        mine = self._tyre_lap(me, me.compound, me.wear)
        theirs = self._tyre_lap(me, them.compound, them.wear)
        return float(np.clip((mine - theirs) / 2.0, -1.0, 1.0))

    def _life_laps(self, compound: str, wear: float) -> float:
        """Laps left before ``compound`` at ``wear`` passes the wear limit
        (`StrategyBaselineParams.max_wear`) at push 0.5."""
        per_lap = self.ref_lap_s / self.params.compound(compound).life_s
        return max(0.0, self.baseline.max_wear - wear) / per_lap

    def _rejoin(self) -> list[float]:
        """If we stopped this lap: (place we would rejoin in, gap to the car that
        would then be ahead, gap to the one behind), from the gaps at the line."""
        p = self.params
        out = self.cars[0].time_s + p.pit_loss_s
        others = sorted(self.cars[c].time_s for c in self.order if c != 0)
        place = sum(1 for t in others if t < out)
        ahead = out - others[place - 1] if place > 0 else GAP_SCALE_S
        behind = others[place] - out if place < len(others) else GAP_SCALE_S
        return [
            place / (p.n_cars - 1),
            min(1.0, ahead / GAP_SCALE_S),
            min(1.0, behind / GAP_SCALE_S),
        ]

    def battle_moves(self) -> dict[str, int]:
        """Undercuts and overcuts in the race so far, from the stops and running
        order: our car and a rival that ran just ahead of us (for gains) or just
        behind (for losses) both stopped within `UNDERCUT_WINDOW_LAPS` of each
        other, and the order between us flipped across the two stops. We stopped
        first = an undercut, second = an overcut."""
        out = {"undercuts": 0, "overcuts": 0, "undercut_on_us": 0, "overcut_on_us": 0}
        mine = self.stop_laps[0]
        for lap in mine:
            for r in range(1, self.params.n_cars):
                for rl in self.stop_laps[r]:
                    if rl == lap or abs(rl - lap) > UNDERCUT_WINDOW_LAPS:
                        continue
                    first, last = min(lap, rl), max(lap, rl)
                    if last + 1 >= len(self.orders):
                        continue
                    before, after = self.orders[first], self.orders[last + 1]
                    if abs(before.index(0) - before.index(r)) != 1:
                        continue  # not the car next to us
                    was = before.index(0) < before.index(r)
                    now = after.index(0) < after.index(r)
                    if was == now:
                        continue
                    ours_first = lap < rl
                    if now:
                        out["undercuts" if ours_first else "overcuts"] += 1
                    else:
                        out["overcut_on_us" if ours_first else "undercut_on_us"] += 1
        return out

    def _info(self) -> dict[str, Any]:
        me = self.cars[0]
        return {
            "track": self.track.name,
            "lap": self.lap,
            "laps": self.total_laps,
            "position": self._position() + 1,
            "race_time_s": me.time_s,
            "compound": me.compound,
            "wear": me.wear,
            "stops": me.stops,
            "legal": len(me.compounds_used) >= 2,
            **(self.battle_moves() if self.lap >= self.total_laps else {}),
        }

    # --- gym API -------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[Obs, dict[str, Any]]:
        super().reset(seed=seed)
        opts = options or {}
        p = self.params
        rng = self.np_random
        names = list(self.tracks)
        name = opts.get("track", names[int(rng.integers(len(names)))])
        self.track = self.tracks[name]
        self.total_laps = int(opts.get("laps", self.track.laps))
        self.lap = 0
        n_push = len(p.push_levels)
        if p.rivals == "baseline":
            rivals = [
                baseline_rival(rng, self.baseline, p.rival_stop_jitter)
                for _ in range(p.n_cars - 1)
            ]
        elif p.rivals == "mixed":
            rivals = [
                sample_rival_plan(rng, self.baseline, n_push)
                for _ in range(p.n_cars - 1)
            ]
        else:
            raise ValueError(f"unknown rivals {p.rivals!r}: mixed or baseline")
        self.plans = [FixedPlan(self.baseline), *rivals]
        paces = [1.0] + [
            float(1.0 + rng.uniform(-p.car_pace_spread, p.car_pace_spread))
            for _ in range(p.n_cars - 1)
        ]
        grid = [int(g) for g in rng.permutation(p.n_cars)]
        if "grid" in opts:
            # Our grid slot (1 = pole); the rivals keep their drawn order.
            want = int(opts["grid"]) - 1
            grid.remove(0)
            grid.insert(want, 0)
        self.order = grid
        self.cars = [CarState(0.0, "medium", 0.0, 0, 0.0, frozenset())] * p.n_cars
        for slot, c in enumerate(grid):
            start = self.plans[c].start_compound
            self.cars[c] = CarState(
                time_s=slot * p.min_gap_s,
                compound=start,
                wear=0.0,
                tyre_laps=0,
                fuel_kg=p.start_fuel_kg,
                compounds_used=frozenset({start}),
                pace=paces[c],
            )
        normal = push_grip_fraction(p, p.push_levels[self.baseline.push_index])
        grip = normal * p.compound("medium").grip
        self.ref_lap_s = self.track.table.lap_time_s(grip, p.start_fuel_kg / 2)
        self.last_lap_s = self.ref_lap_s
        self.history = []
        # Running order at the start of every lap (index = lap), stop laps per car.
        self.orders = [list(self.order)]
        self.stop_laps = [[] for _ in range(p.n_cars)]
        return self._obs(), self._info()

    def step(self, action: Any) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        p = self.params
        pit_choice, push_index = int(action[0]), int(action[1])
        burn = p.start_fuel_kg / self.total_laps
        if self.lap == 0:
            # On the grid: the tyre choice is the starting set, free of charge.
            start = COMPOUNDS[pit_choice - 1] if pit_choice else "medium"
            self.cars[0] = replace(
                self.cars[0], compound=start, compounds_used=frozenset({start})
            )
            pit_choice = 0
        ours = LapPlan(
            push=p.push_levels[push_index],
            pit_to=COMPOUNDS[pit_choice - 1] if pit_choice else None,
        )
        n = p.n_cars
        start_s = [c.time_s for c in self.cars]
        laps = [0.0] * n
        pitted = [False] * n
        new_cars = list(self.cars)
        for pos, c in enumerate(self.order):
            plan = (
                ours
                if c == 0
                else self.plans[c].plan(p, self.cars[c], self.lap, self.total_laps)
            )
            dirty = pos > 0 and (
                self.cars[c].time_s - self.cars[self.order[pos - 1]].time_s
                < p.dirty_air_gap_s
            )
            laps[c], new_cars[c] = car_lap(
                p, self.tire, self.track.table, self.cars[c], plan, burn, dirty
            )
            pitted[c] = plan.pit_to is not None
            if pitted[c]:
                self.stop_laps[c].append(self.lap)
        before = self._position()
        draws = [float(x) for x in self.np_random.random(4 * n)]
        self.order, end = resolve_lap(
            p,
            self.order,
            start_s,
            laps,
            pitted,
            draws,
            ours=0,
            our_attack=self.profile.attack if self.profile else None,
            our_defend=self.profile.defend if self.profile else None,
        )
        self.cars = [replace(c, time_s=e) for c, e in zip(new_cars, end, strict=True)]
        self.last_lap_s = end[0] - start_s[0]
        self.lap += 1
        self.orders.append(list(self.order))
        after = self._position()
        r = p.reward
        reward = -r.time_weight * (self.last_lap_s - self.ref_lap_s)
        reward += r.position_weight * (before - after)
        done = self.lap >= self.total_laps
        info = self._info()
        if done:
            reward -= r.finish_weight * after
            if not info["legal"]:
                reward -= r.rule_penalty
            elif after == 0:
                reward += r.win_bonus
        self.history.append(
            {
                "lap": self.lap,
                "position": after + 1,
                "lap_s": self.last_lap_s,
                "compound": self.cars[0].compound,
                "wear": self.cars[0].wear,
                "pit": ours.pit_to,
                "push": ours.push,
            }
        )
        return self._obs(), float(reward), done, False, info
