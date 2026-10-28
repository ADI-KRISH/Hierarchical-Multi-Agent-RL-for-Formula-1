"""Real-race starts: turn the incident library (`f1rl.incidents`) into sim scenarios.

Each real pass or defence becomes a starting situation on the TUM version of the
circuit: where the cars were, how far apart, who was quicker. The agent then has to
find its own way to pass or to hold the place, from situations that really happened,
on the track where they happened. The data says where and when, never how (no
steering data), so it seeds training rather than being copied.

From one **pass** (car A got past car B, at ``location_m`` round the lap):

- ``attack``: we are A, B is ahead -- start `RealStartParams.lead_m` before the spot.
  B is a ``slow`` opponent if its tyres were `fresh_tyre_laps` older, else an
  equal-pace ``defender``.
- ``defend``: we are B, an ``attacker`` is behind, same place and gap.

From one **defence** (B kept A within the DRS gap for 3+ laps): the gap was measured
at the line, so both scenarios start at the line (s = 0), ``defend`` against an
``attacker`` and ``attack`` against a ``defender``.

Lap distance in the incident files is from FastF1's start line, and the TUM tracks
start there too: the real passes at Monza, Spielberg, Sakhir and Catalunya cluster
just before the TUM tracks' first big braking zones, as they should.

Pure functions, except `load_real_starts`, which reads the JSON files.
"""

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from f1rl.config import RealStartParams
from f1rl.incidents import INCIDENT_DIR

ROLES = ("attack", "defend")


@dataclass(frozen=True)
class RealStart:
    track: str  # TUM track name
    split: str  # train | validation | held_out
    role: str  # attack (opponent ahead) | defend (opponent behind)
    opponent: str  # `config.OPPONENT_KINDS` key
    start_s_m: float  # where both scenarios start, metres from the line (may be < 0)
    gap_s: float  # real gap between the cars (s)
    real_outcome: str  # what happened in the race: passed | held | lost
    source: str  # e.g. "2021 Bahrain Grand Prix L6 BOT>LEC"

    def options(self) -> dict[str, Any]:
        """`RaceEnv.reset` options for this start."""
        return {
            "start_s": self.start_s_m,
            "gap_s": self.gap_s,
            "opponent": self.opponent,
        }


DEFAULT_PARAMS = RealStartParams()


def starts_from_race(
    race: Mapping[str, Any], params: RealStartParams = DEFAULT_PARAMS
) -> list[RealStart]:
    """All scenarios one race file gives (none if it has no TUM track)."""
    track = race.get("tum_track")
    if not track:
        return []
    split = str(race["split"])
    name = f"{race['year']} {race['grand_prix']}"
    out: list[RealStart] = []
    for p in race.get("passes", []):
        loc = p.get("location_m")
        gap = p.get("gap_before_s")
        if loc is None or gap is None:
            continue
        older = _float(p.get("defender_tyre_life")) - _float(
            p.get("attacker_tyre_life")
        )
        ahead = "slow" if older >= params.fresh_tyre_laps else "defender"
        where = float(loc) - params.lead_m
        tag = f"{name} L{p['lap']} {p['attacker']}>{p['defender']}"
        out += [
            RealStart(track, split, "attack", ahead, where, float(gap), "passed", tag),
            RealStart(
                track, split, "defend", "attacker", where, float(gap), "lost", tag
            ),
        ]
    for d in race.get("defences", []):
        gap = d.get("min_gap_s")
        if gap is None:
            continue
        laps = f"L{d['first_lap']}-{d['last_lap']}"
        tag = f"{name} {laps} {d['defender']} vs {d['attacker']}"
        outcome = str(d["outcome"])
        out += [
            RealStart(
                track, split, "defend", "attacker", 0.0, float(gap), outcome, tag
            ),
            RealStart(
                track, split, "attack", "defender", 0.0, float(gap), outcome, tag
            ),
        ]
    return out


def _float(value: Any) -> float:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if f != f else f  # NaN (unknown tyre age) counts as no difference


def by_track(
    starts: Iterable[RealStart], splits: Iterable[str] | None = None
) -> dict[str, list[RealStart]]:
    """Starts grouped by track, optionally only those in ``splits``."""
    keep = set(splits) if splits is not None else None
    out: dict[str, list[RealStart]] = {}
    for st in starts:
        if keep is None or st.split in keep:
            out.setdefault(st.track, []).append(st)
    return out


def load_real_starts(
    directory: Path = INCIDENT_DIR, params: RealStartParams = DEFAULT_PARAMS
) -> list[RealStart]:
    """Every scenario in the incident library, in file order (deterministic)."""
    out: list[RealStart] = []
    for path in sorted(directory.glob("*.json")):
        out += starts_from_race(json.loads(path.read_text(encoding="utf-8")), params)
    return out
