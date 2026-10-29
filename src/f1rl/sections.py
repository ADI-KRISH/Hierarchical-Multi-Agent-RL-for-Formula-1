"""Track sections: every straight, braking zone and corner of a track, one by one.

The overtaking driver is trained and scored one section at a time: a battle starts a
few metres behind a slower car at the start of one section and is won only if our car
is ahead by the section's end (and the pass then sticks, `RaceEnvParams.pass_hold_m`).
So each kind of pass is learned where it happens:

- **straight**: from the apex of one corner to the next braking point -- the better
  exit, then the pass in the tow (slipstream + DRS), finished before anyone brakes.
- **braking**: from the approach to a braking point to just after the apex -- the
  late-braking dive up the inside (or the switch from a feint).
- **corner**: from the turn-in to the corner exit -- round the outside, or the
  better exit and the drag race out of the corner.

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

KINDS = ("straight", "braking", "corner")


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
        prev_exit = prev_apex + p.corner_exit_m
        if len(corners) == 1:
            prev_exit = apex - lap + p.corner_exit_m

        def add(kind: str, start: float, end: float) -> None:
            if end - start >= p.min_length_m:
                out.append(
                    Section(
                        track,
                        i,
                        kind,
                        start % lap,
                        end - start,
                        brake % lap,
                        apex % lap,
                        side,
                    )
                )

        # Straight: from the previous apex (the exit is where a quicker car gains)
        # to the braking point (the pass must be done by then).
        add("straight", prev_apex + p.straight_from_apex_m, brake)
        # Braking zone: the approach to the braking point to just past the apex.
        approach = max(brake - p.approach_m, prev_exit, prev_apex + 1.0)
        add("braking", approach, apex + p.past_apex_m)
        # Corner: turn-in (a little before the braking point ends) to the exit.
        turn_in = max(brake + p.turn_in_frac * (apex - brake), approach)
        add("corner", turn_in, min(apex + p.corner_exit_m, top))
    return out


def section_options(
    sec: Section, opponent: str, gap_m: float, push: float = 1.0
) -> dict[str, Any]:
    """`RaceEnv.reset` options for a battle in ``sec``: our car at its start,
    ``opponent`` ``gap_m`` ahead, the pass to be finished within its length."""
    return {
        "start_s": sec.start_s,
        "opponent": opponent,
        "gap_m": gap_m,
        "section_m": sec.length_m,
        "push": push,
        "wear": 0.0,
        "start_speed_frac": 1.0,  # at racing speed, like the car it is chasing
    }


def env_sections(env: Any, p: SectionParams | None = None) -> list[Section]:
    """`track_sections` for a `TacticalEnv` (its driver's racing line)."""
    d = env._driver()
    return track_sections(
        env.track.name,
        np.asarray(d.line_speed(), dtype=np.float64),
        np.asarray(d.line_kappa(), dtype=np.float64),
        d.track.ds_m,
        d.car.max_speed_ms,
        p,
    )
