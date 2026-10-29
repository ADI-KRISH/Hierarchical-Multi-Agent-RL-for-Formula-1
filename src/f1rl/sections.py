"""Track sections: every straight, braking zone and corner of a track, one by one.

The overtaking driver is trained and scored one section at a time: a battle starts a
few metres behind a slower car at the start of one section and is won only if our car
is ahead by the section's end (and the pass then sticks, `RaceEnvParams.pass_hold_m`).
So each kind of pass is learned where it happens:

- **straight**: from the apex of one corner to the next braking point -- the better
  exit, then the pass in the tow (slipstream + DRS), finished before anyone brakes.
- **braking**: from the approach to a braking point to just after the apex -- the
  late-braking dive up the inside (or the switch from a feint).
- **corner_in / corner_out**: from the turn-in to the corner exit, starting already
  half alongside (`SectionParams.corner_gap_m` behind, `corner_shift_m` to the
  inside or the outside): finish the pass through the corner. From a few metres
  straight behind, a single corner is too short to pass in (checked: almost never,
  even by brute force), so the corner battle is the side-by-side part.

Sections come from the racing line itself (its speed profile, as
`agents.tactics.corners_ahead`), so they exist on any track with no hand labels.
`track_sections` is a pure function of the line; `section_options` turns a section
into `RaceEnv.reset` options.
"""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from f1rl.agents.tactics import CORNER_SPEED_FRACTION
from f1rl.config import SectionParams

KINDS = ("straight", "braking", "corner_in", "corner_out")
#: A side-by-side start needs the cars at least a car width plus this (m) apart.
SIDE_GAP_M = 0.8


@dataclass(frozen=True)
class Section:
    track: str
    index: int  # number of the corner this section belongs to (0 = first after s=0)
    kind: str  # one of KINDS
    start_s: float  # where the battle starts (m along the track, in [0, lap))
    length_m: float  # the pass must be done within this distance of the start
    brake_s: float  # the corner's braking point (m along the track)
    apex_s: float  # the corner's apex (m along the track)
    side: float  # +1 left-hander, -1 right-hander (the inside)
    # Where our car starts: this far (m) left (+) of the racing line, and the
    # battle's own gap to the car ahead (m, centre to centre; 0 = the caller's).
    shift_m: float = 0.0
    gap_m: float = 0.0

    @property
    def name(self) -> str:
        return f"{self.track}/T{self.index + 1}/{self.kind}"

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "name": self.name}


def _corners(
    speed: NDArray[np.float64], max_speed: float
) -> list[tuple[int, int, int]]:
    """(brake index, apex index, exit index) of every corner round a closed lap:
    each local minimum of the line's speed below `CORNER_SPEED_FRACTION` of top
    speed; braking starts at the top of the ramp down to it, the exit is the top of
    the ramp up out of it."""
    n = len(speed)
    out = []
    for a in range(n):
        v = speed[a]
        if v >= CORNER_SPEED_FRACTION * max_speed:
            continue
        if v > speed[(a - 1) % n] or v >= speed[(a + 1) % n]:
            continue
        b = a
        for _ in range(n):
            if speed[(b - 1) % n] <= speed[b % n]:
                break
            b -= 1
        x = a
        for _ in range(n):
            if speed[(x + 1) % n] <= speed[x % n]:
                break
            x += 1
        out.append((b, a, x))  # b may be < 0 and x >= n: unwrapped about the apex
    out.sort(key=lambda c: c[1])
    return out


def track_sections(
    track: str,
    speed: NDArray[np.float64],
    kappa: NDArray[np.float64],
    ds_m: float,
    max_speed: float,
    p: SectionParams | None = None,
) -> list[Section]:
    """All sections of one track from its racing line's ``speed`` (m/s) and
    ``kappa`` (1/m) at stations ``ds_m`` apart. Too-short ones are left out."""
    p = p or SectionParams()
    lap = len(speed) * ds_m
    corners = _corners(speed, max_speed)
    out: list[Section] = []
    for i, (b, a, x) in enumerate(corners):
        brake, apex, top = b * ds_m, a * ds_m, x * ds_m
        side = float(np.sign(kappa[a])) or 1.0
        # The previous corner's apex (unwrapped to before this one).
        prev_apex = corners[i - 1][1] * ds_m
        if prev_apex >= apex:
            prev_apex -= lap
        spans: list[tuple[str, float, float, float]] = []
        approach = max(brake - p.approach_m, prev_apex + p.straight_from_apex_m)
        turn_in = max(brake + p.turn_in_frac * (apex - brake), approach)
        # Corner exit: the top of the ramp out of it (60-`corner_exit_m` m past it).
        end = min(apex + p.corner_exit_m, max(top, apex + 60.0))
        # Straight: from the previous apex (the exit is where a quicker car gains)
        # to the braking point (the pass must be done by then, or by the exit).
        straight = prev_apex + p.straight_from_apex_m
        if brake - straight >= p.min_length_m:
            done = end if p.finish_at_exit else brake
            spans.append(("straight", straight, done, 0.0))
        # Braking zone: the approach to the braking point to just past the apex.
        if apex + p.past_apex_m - approach >= p.min_length_m:
            done = end if p.finish_at_exit else apex + p.past_apex_m
            spans.append(("braking", approach, max(done, apex + p.past_apex_m), 0.0))
        # Corner: turn-in (a little before the braking point ends) to the exit,
        # starting half alongside on the inside or the outside.
        if end - turn_in >= p.min_length_m:
            spans.append(("corner_in", turn_in, end, side * p.corner_shift_m))
            spans.append(("corner_out", turn_in, end, -side * p.corner_shift_m))
        out += [
            Section(
                track,
                i,
                kind,
                start % lap,
                stop - start,
                brake % lap,
                apex % lap,
                side,
                shift_m,
                p.corner_gap_m if shift_m else 0.0,
            )
            for kind, start, stop, shift_m in spans
        ]
    return out


def section_options(
    sec: Section, opponent: str, gap_m: float, push: float = 1.0
) -> dict[str, Any]:
    """`RaceEnv.reset` options for a battle in ``sec``: our car at its start,
    ``opponent`` ``gap_m`` ahead (the section's own gap, if it has one), the pass to
    be finished within its length."""
    return {
        "start_s": sec.start_s,
        "start_shift_m": sec.shift_m,
        "opponent": opponent,
        "gap_m": sec.gap_m or gap_m,
        "section_m": sec.length_m,
        "push": push,
        "wear": 0.0,
        "start_speed_frac": 1.0,  # at racing speed, like the car it is chasing
    }


def env_sections(env: Any, p: SectionParams | None = None) -> list[Section]:
    """`track_sections` for a `TacticalEnv` (its driver's racing line), without the
    side-by-side starts the road has no room for (our car would start overlapping
    the other one: at least a car width plus `SIDE_GAP_M` apart, on the road)."""
    d = env._driver()
    secs = track_sections(
        env.track.name,
        np.asarray(d.line_speed(), dtype=np.float64),
        np.asarray(d.line_kappa(), dtype=np.float64),
        d.track.ds_m,
        d.car.max_speed_ms,
        p,
    )
    width = env.params.car_width_m + SIDE_GAP_M
    out = []
    for sec in secs:
        if sec.shift_m:
            n_line = d.base.line_state(sec.start_s)[0]
            n_opp = d.base.line_state(sec.start_s + sec.gap_m)[0]
            left, right = env._room_at(sec.start_s)
            n_me = n_line + sec.shift_m
            if not (-right + 0.3 <= n_me <= left - 0.3) or abs(n_me - n_opp) < width:
                continue
        out.append(sec)
    return out
