"""`StrategyEnv`: the Strategy Agent's race, one decision per lap.

A whole Grand Prix (305 km) against ``n_cars - 1`` rivals running rule-based
strategies (`agents.strategy_baseline`). Each step the agent picks, for the coming
lap, whether to pit and for which compound, and a push level for the driver. The
driver itself is a stand-in: a lap-time table from the track's optimal line
(`models.race.LapTable`), until the trained Driver Agent can replace it.

Action (MultiDiscrete [4, len(push_levels)], or [6, ...] with ``wet_tyres``):
  0: 0 = stay out, 1/2/3 = pit for soft/medium/hard, 4/5 = intermediate/wet. On
     the first decision (on the grid) it picks the starting set at no cost; 0 =
     start on mediums.
  1: push level index (0 = conserve .. last = attack).

Incidents (`StrategyIncidentParams`, off by default): in some races our car spins (time
lost, a flat spot) or gets a puncture (a slow lap back, then a stop it must make on
the next lap) on one drawn lap.

Safety car / VSC (`NeutralParams`, off by default): in some races a run of laps is
neutralised: every car laps at the same slow pace, nobody passes, stops are cheaper
and (SC) the field closes up. The agent sees it before it decides on that lap; the
rule-based cars bring a planned stop forward into it.

Reward (`StrategyRewardParams`): minus seconds lost per lap against a reference
lap, plus places gained, and at the flag a penalty per place behind the winner
and a disqualification penalty for running only one compound.
"""

import dataclasses
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
    TYRES,
    WET_COMPOUNDS,
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
from f1rl.models.weather import (
    RaceWeather,
    dry_weather,
    race_weather,
    tyre_class,
    wet_grip,
    wet_wear,
)

Obs = np.typing.NDArray[np.float32]

#: Gaps (s) are divided by this and clipped to [0, 1] in the observation.
GAP_SCALE_S = 30.0
#: Two stops this many laps apart or fewer can make an undercut or overcut.
UNDERCUT_WINDOW_LAPS = 5
OBS_SIZE = 38
#: Extra inputs with ``wet_tyres``: the weather, forecast and wet-tyre state.
WEATHER_OBS_SIZE = 11
#: Extra inputs when incidents are on: spun, punctured, time lost, must stop now.
INCIDENT_OBS_SIZE = 4
#: Extra inputs when SC / VSC are on: SC now, VSC now, laps of it so far, stop cost.
NEUTRAL_OBS_SIZE = 4
#: Extra inputs with ``pace_obs``: last-lap pace of the car ahead, behind, the
#: leader and the field median, each against ours.
PACE_OBS_SIZE = 4
#: Extra input with overtake mode: the battery charge.
ERS_OBS_SIZE = 1


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


def legal(car: CarState) -> bool:
    """Tyre-use rule: two dry compounds, unless a wet-weather tyre was used."""
    used = car.compounds_used
    return len(used & set(COMPOUNDS)) >= 2 or bool(used & set(WET_COMPOUNDS))


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
        self.tyres = TYRES if params.wet_tyres else COMPOUNDS
        self.action_space = spaces.MultiDiscrete(
            [len(self.tyres) + 1, len(params.push_levels)]
            + ([2] if params.ers.enabled else [])
        )
        size = OBS_SIZE + (WEATHER_OBS_SIZE if params.wet_tyres else 0)
        size += INCIDENT_OBS_SIZE if params.incidents.chance > 0 else 0
        nu = params.neutral
        self.use_neutral = nu.sc_chance > 0 or nu.vsc_chance > 0 or nu.red_chance > 0
        size += NEUTRAL_OBS_SIZE if self.use_neutral else 0
        size += PACE_OBS_SIZE if params.pace_obs else 0
        size += ERS_OBS_SIZE if params.ers.enabled else 0
        self.observation_space = spaces.Box(-1.0, 1.0, (size,), np.float32)
        self.weather: RaceWeather = dry_weather(1)
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
        # This race's incident: (lap, "spin" | "puncture", seconds lost) or None.
        self.incident: tuple[int, str, float] | None = None
        self.incident_done = False
        self.must_pit = False
        # Per lap: None, "sc", "vsc" or "red".
        self.neutral: list[str | None] = []
        self.battery = params.ers.start
        self.last_laps: list[float] = []

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
            min(1.0, TYRES.index(self.cars[ahead].compound) / 2)
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
        if p.wet_tyres:
            obs += self._weather_obs()
        if p.incidents.chance > 0:
            obs += self._incident_obs()
        if self.use_neutral:
            obs += self._neutral_obs()
        if p.pace_obs:
            obs += self._pace_obs()
        if p.ers.enabled:
            obs.append(self.battery)
        return np.clip(np.array(obs, dtype=np.float32), -1.0, 1.0)

    def _incident_obs(self) -> list[float]:
        kind = self.incident[1] if self.incident and self.incident_done else ""
        lost = self.incident[2] if self.incident and self.incident_done else 0.0
        return [
            1.0 if kind == "spin" else 0.0,
            1.0 if kind == "puncture" else 0.0,
            min(1.0, lost / GAP_SCALE_S),
            1.0 if self.must_pit else 0.0,
        ]

    def _pace_obs(self) -> list[float]:
        """Rivals' last lap against ours (s, + = they were slower), / 3 s."""
        if not self.last_laps:
            return [0.0] * PACE_OBS_SIZE
        pos = self._position()
        mine = self.last_laps[0]

        def rel(c: int | None) -> float:
            if c is None:
                return 0.0
            return float(np.clip((self.last_laps[c] - mine) / 3.0, -1, 1))

        ahead = self.order[pos - 1] if pos > 0 else None
        behind = self.order[pos + 1] if pos + 1 < len(self.order) else None
        field = float(np.median(self.last_laps[1:]))
        return [
            rel(ahead),
            rel(behind),
            rel(self.order[0]) if self.order[0] != 0 else 0.0,
            float(np.clip((field - mine) / 3.0, -1, 1)),
        ]

    def restart_now(self) -> bool:
        """The coming lap is a green-flag restart after an SC or red flag."""
        return (
            self.neutral_now() is None
            and 0 < self.lap <= len(self.neutral)
            and self.neutral[self.lap - 1] in ("sc", "red")
        )

    def neutral_now(self) -> str | None:
        """ "sc" / "vsc" if the coming lap is neutralised, else None."""
        return self.neutral[self.lap] if self.lap < len(self.neutral) else None

    def _neutral_obs(self) -> list[float]:
        kind = self.neutral_now()
        run = 0
        while kind and self.lap - run - 1 >= 0 and self.neutral[self.lap - run - 1]:
            run += 1
        nu = self.params.neutral
        loss = {"sc": nu.sc_pit_loss_s, "vsc": nu.vsc_pit_loss_s, "red": 0.0}.get(
            kind or "", self.params.pit_loss_s
        )
        return [
            1.0 if kind in ("sc", "red") else 0.0,
            1.0 if kind == "vsc" else 0.0,
            min(1.0, run / 5),
            min(1.0, loss / max(1.0, self.params.pit_loss_s)),
        ]

    def _draw_neutral(
        self, rng: np.random.Generator, forced: str | None
    ) -> list[str | None]:
        """Neutralised laps. ``forced``: "sc", "vsc", "red", "sc2" (two SCs) or
        "none"; None draws from the chances."""
        nu = self.params.neutral
        roll = rng.random(3)  # always three draws, so options do not shift the RNG
        laps: list[str | None] = [None] * self.total_laps
        if forced == "none":
            return laps
        if forced not in (None, "sc", "vsc", "red", "sc2"):
            raise ValueError(f"unknown neutral {forced!r}: sc, vsc, red, sc2, none")
        kind: str | None
        if forced is None:
            kind = (
                "sc"
                if roll[0] < nu.sc_chance
                else "vsc"
                if roll[0] < nu.sc_chance + nu.vsc_chance
                else None
            )
        else:
            kind = "sc" if forced == "sc2" else forced
        if kind:
            self._place_neutral(laps, kind, float(roll[1]), float(roll[2]))
        if nu.max_periods > 1 or nu.red_chance > 0:
            more = rng.random(4)
            if forced == "sc2" or (
                forced is None and nu.max_periods > 1 and more[0] < nu.second_chance
            ):
                share = nu.sc_chance / max(1e-9, nu.sc_chance + nu.vsc_chance)
                second = "sc" if forced == "sc2" or more[3] < share else "vsc"
                self._place_neutral(laps, second, float(more[1]), float(more[2]))
            elif forced is None and more[0] >= 1.0 - nu.red_chance:
                self._place_neutral(laps, "red", float(more[1]), 0.0)
        return laps

    def _place_neutral(
        self, laps: list[str | None], kind: str, r_start: float, r_len: float
    ) -> None:
        nu = self.params.neutral
        lo, hi = nu.lap_range
        first = max(1, int((lo + r_start * (hi - lo)) * self.total_laps))
        if kind == "red":
            length = 1
        else:
            span = nu.sc_laps if kind == "sc" else nu.vsc_laps
            length = span[0] + int(r_len * (span[1] - span[0] + 1))
        for k in range(first, min(self.total_laps - 1, first + length)):
            if laps[k] is None:
                laps[k] = kind

    def _neutral_lap(
        self, start_s: list[float], lap_s: list[float], kind: str
    ) -> tuple[list[int], list[float]]:
        """Order and line times after a neutralised lap: no passing on track (a car
        that stopped, or lost time, drops back), SC closes the gaps up."""
        nu, p = self.params.neutral, self.params
        raw = [s + t for s, t in zip(start_s, lap_s, strict=True)]
        rank = {c: i for i, c in enumerate(self.order)}
        new = sorted(self.order, key=lambda c: (raw[c], rank[c]))
        end = list(raw)
        for i in range(1, len(new)):
            prev, c = new[i - 1], new[i]
            gap = raw[c] - raw[prev]
            if kind in ("sc", "red") and gap > nu.sc_gap_s:
                keep = nu.sc_keep if kind == "sc" else 0.0
                gap = nu.sc_gap_s + (gap - nu.sc_gap_s) * keep
            end[c] = end[prev] + max(gap, p.min_gap_s)
        return new, end

    def _draw_incident(
        self, rng: np.random.Generator, forced: str | None
    ) -> tuple[int, str, float] | None:
        inc = self.params.incidents
        roll = rng.random(3)  # always three draws, so options do not shift the RNG
        if forced == "none":
            return None
        if forced is None:
            if inc.chance <= 0 or roll[0] >= inc.chance:
                return None
            kind = "puncture" if roll[1] < inc.puncture_share else "spin"
        elif forced in ("spin", "puncture"):
            kind = forced
        else:
            raise ValueError(f"unknown incident {forced!r}: spin or puncture")
        lo, hi = inc.lap_range
        lap = int(rng.uniform(lo, hi) * self.total_laps)
        span = inc.puncture_loss_range if kind == "puncture" else inc.spin_loss_range
        loss = span[0] + float(roll[2]) * (span[1] - span[0])
        return max(1, lap), kind, loss

    def seen_wetness(self) -> float:
        """The track as the pit wall saw it on the last lap (the grid at lap 0)."""
        return self.weather.at(self.lap - 1 if self.lap > 0 else 0)

    def _weather_obs(self) -> list[float]:
        p, w = self.params, self.weather
        me = self.cars[0]
        now = self.seen_wetness()
        mine = self._tyre_lap(me, me.compound, me.wear)
        rivals = [self.cars[c] for c in range(1, p.n_cars)]
        return [
            now,
            1.0 if w.raining[min(max(self.lap - 1, 0), len(w.raining) - 1)] else 0.0,
            w.forecast(p.weather, max(self.lap - 1, 0), 3),
            w.forecast(p.weather, max(self.lap - 1, 0), 6),
            *[1.0 if me.compound == c else 0.0 for c in WET_COMPOUNDS],
            # Seconds a lap a fresh set of inters / wets would gain right now.
            *[
                float(np.clip((mine - self._tyre_lap(me, c, 0.0)) / 10.0, -1, 1))
                for c in WET_COMPOUNDS
            ],
            sum(tyre_class(c.compound) != "slick" for c in rivals) / len(rivals),
            1.0 if me.compounds_used & set(WET_COMPOUNDS) else 0.0,
            min(1.0, me.wear * wet_wear(p.weather, me.compound, now) / 4.0),
        ]

    def _tyre_lap(self, car: CarState, compound: str, wear: float) -> float:
        """A car's lap time (s) at normal push on ``compound`` worn ``wear``, on
        the track as last seen."""
        p = self.params
        normal = push_grip_fraction(p, p.push_levels[self.baseline.push_index])
        grip = normal * p.compound(compound).grip * tire_grip(self.tire, wear)
        grip *= wet_grip(p.weather, compound, self.seen_wetness())
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
            "legal": legal(me),
            "rain": self.weather.wet,
            "track_passes": getattr(self, "track_passes", 0),
            "track_passed": getattr(self, "track_passed", 0),
            "incident": self.incident[1] if self.incident else None,
            "incident_lap": self.incident[0] if self.incident else None,
            "neutral": next((k for k in self.neutral if k), None),
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
        if p.wet_tyres:
            wp = p.weather
            self.plans = [self.plans[0]] + [
                dataclasses.replace(
                    plan,
                    rain_offset=float(rng.uniform(-wp.rival_jitter, wp.rival_jitter)),
                )
                for plan in self.plans[1:]
            ]
            rain = bool(opts.get("rain", rng.random() < wp.rain_chance))
            self.weather = race_weather(wp, rng, self.total_laps, rain)
        else:
            self.weather = dry_weather(self.total_laps)
        paces = [p.our_pace] + [
            float(1.0 + rng.uniform(-p.car_pace_spread, p.car_pace_spread))
            for _ in range(p.n_cars - 1)
        ]
        grid = [int(g) for g in rng.permutation(p.n_cars)]
        if "grid" not in opts and p.grid_slot:
            opts = {**opts, "grid": p.grid_slot}
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
        # Places won and lost on track (neither car pitting that lap).
        self.track_passes = 0
        self.track_passed = 0
        # Option "incident": "spin" / "puncture" forces one; "none" forces none.
        forced = opts.get("incident")
        self.incident = (
            self._draw_incident(rng, forced)
            if p.incidents.chance > 0 or forced not in (None, "none")
            else None
        )
        self.incident_done = False
        self.must_pit = False
        self.battery = p.ers.start
        self.last_laps = []
        forced_n = opts.get("neutral")
        self.neutral = (
            self._draw_neutral(rng, forced_n)
            if self.use_neutral or forced_n not in (None, "none")
            else [None] * self.total_laps
        )
        return self._obs(), self._info()

    def step(self, action: Any) -> tuple[Obs, float, bool, bool, dict[str, Any]]:
        p = self.params
        pit_choice, push_index = int(action[0]), int(action[1])
        burn = p.start_fuel_kg / self.total_laps
        if self.lap == 0:
            # On the grid: the tyre choice is the starting set, free of charge.
            start = self.tyres[pit_choice - 1] if pit_choice else "medium"
            self.cars[0] = replace(
                self.cars[0], compound=start, compounds_used=frozenset({start})
            )
            pit_choice = 0
        pit_to = self.tyres[pit_choice - 1] if pit_choice else None
        neutral = self.neutral_now()
        me = self.cars[0]
        if (
            pit_to is not None
            and not self.must_pit
            and neutral is None
            and self.lap < p.min_stop_fraction * self.total_laps
            and me.wear < p.early_stop_wear
            and tyre_class(pit_to) == tyre_class(me.compound)
            and not (
                self.incident_done and self.incident and self.incident[1] == "spin"
            )
        ):
            pit_to = None  # no early stop by choice
        if (
            pit_to is not None
            and not self.must_pit
            and neutral != "red"
            and self.cars[0].stops > 0
            and self.cars[0].tyre_laps < p.min_stint_laps
        ):
            pit_to = None  # a new set is not changed again this soon
        if self.must_pit and pit_to is None:
            pit_to = p.incidents.forced_compound
        self.must_pit = False
        ours = LapPlan(push=p.push_levels[push_index], pit_to=pit_to)
        n = p.n_cars
        nu = p.neutral
        factor = nu.vsc_lap_factor if neutral == "vsc" else nu.sc_lap_factor
        slow = self.ref_lap_s * factor
        cheap = {"sc": nu.sc_pit_loss_s, "vsc": nu.vsc_pit_loss_s, "red": 0.0}.get(
            neutral or "", p.pit_loss_s
        )
        ers = p.ers
        deploy = (
            ers.enabled
            and len(action) > 2
            and int(action[2]) == 1
            and neutral is None
            and self.lap > 0
            and self.battery >= ers.cost
        )
        start_s = [c.time_s for c in self.cars]
        laps = [0.0] * n
        pitted = [False] * n
        new_cars = list(self.cars)
        for pos, c in enumerate(self.order):
            plan = (
                ours
                if c == 0
                else self.plans[c].plan(
                    p,
                    self.cars[c],
                    self.lap,
                    self.total_laps,
                    self.seen_wetness(),
                    neutral is not None,
                )
            )
            if c != 0 and neutral == "red":
                # Red flag: a free set, the next planned one or a fresh one of these.
                fixed = self.plans[c].params
                nxt = fixed.stints[min(self.cars[c].stops + 1, len(fixed.stints) - 1)]
                if tyre_class(self.cars[c].compound) != "slick":
                    nxt = self.cars[c].compound
                plan = replace(plan, pit_to=nxt)
            dirty = pos > 0 and (
                self.cars[c].time_s - self.cars[self.order[pos - 1]].time_s
                < p.dirty_air_gap_s
            )
            laps[c], new_cars[c] = car_lap(
                p,
                self.tire,
                self.track.table,
                self.cars[c],
                plan,
                burn,
                dirty,
                self.weather.at(self.lap),
            )
            pitted[c] = plan.pit_to is not None
            if c == 0 and deploy:
                laps[c] -= ers.lap_gain_s
            if neutral:
                # Everyone at the safety-car pace; a stop costs less; less wear.
                laps[c] = slow + (cheap if pitted[c] else 0.0)
                was = 0.0 if pitted[c] else self.cars[c].wear
                new_cars[c] = replace(
                    new_cars[c],
                    wear=was + (new_cars[c].wear - was) * nu.wear_factor,
                )
            if c == 0 and self.incident and self.incident[0] == self.lap:
                _, kind, loss = self.incident
                laps[c] += loss
                self.incident_done = True
                if kind == "spin":
                    new_cars[c] = replace(
                        new_cars[c],
                        wear=min(1.0, new_cars[c].wear + p.incidents.spin_wear),
                    )
                else:
                    self.must_pit = True
            if pitted[c]:
                self.stop_laps[c].append(self.lap)
        before = self._position()
        draws = [float(x) for x in self.np_random.random(4 * n)]
        rp = p
        if deploy:
            rp = replace(
                rp,
                attack_skill_s=rp.attack_skill_s + ers.attack_s,
                defend_skill_s=rp.defend_skill_s + ers.defend_s,
            )
        if self.restart_now() and nu.restart_attack_s:
            bonus = nu.restart_attack_s
            rp = replace(
                rp,
                pass_delta_s=rp.pass_delta_s - bonus,
                attack_skill_s=rp.attack_skill_s + bonus,
                defend_skill_s=rp.defend_skill_s - bonus,
            )
        if ers.enabled:
            self.battery = min(
                1.0, self.battery - (ers.cost if deploy else 0.0) + ers.recharge
            )
        if neutral:
            self.order, end = self._neutral_lap(start_s, laps, neutral)
        else:
            self.order, end = resolve_lap(
                rp,
                self.order,
                start_s,
                laps,
                pitted,
                draws,
                ours=0,
                our_attack=self.profile.attack if self.profile else None,
                our_defend=self.profile.defend if self.profile else None,
            )
        wetness = self.weather.at(self.lap)
        old = self.orders[-1]
        if not pitted[0]:
            for rival in range(1, n):
                if pitted[rival]:
                    continue
                was = old.index(0) < old.index(rival)
                now = self.order.index(0) < self.order.index(rival)
                self.track_passes += int(now and not was)
                self.track_passed += int(was and not now)
        self.cars = [replace(c, time_s=e) for c, e in zip(new_cars, end, strict=True)]
        self.last_lap_s = end[0] - start_s[0]
        self.last_laps = [e - s0 for e, s0 in zip(end, start_s, strict=True)]
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
                "wetness": wetness,
                "push": ours.push,
                "neutral": neutral,
                "ers": deploy,
                "battery": self.battery,
            }
        )
        return self._obs(), float(reward), done, False, info
