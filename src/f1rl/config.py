"""Calibration constants and tunable parameters for the simulation.

Two rules govern this module:

1. **No invented F1 numbers.** Every car/track value either carries a citation in
   its comment or is calibrated against FastF1 data in ``data/``. Physical
   constants cite their standard. Values that are not yet calibrated are left
   *required* (no default) so a caller has to supply them explicitly rather than
   silently inheriting a guess.
2. **No hardcoding elsewhere.** A new tunable value gets a field here (or in an
   experiment YAML under ``configs/``), never a literal buried in the env.

The dataclasses are frozen: parameters are fixed for the lifetime of a run, which
is what makes a run reproducible from its logged config plus its seed.
"""

from dataclasses import dataclass
from typing import Final

#: Track width (m) where no data gives one -- FastF1 has no width channel. 12 m
#: is the minimum width the FIA sets for new permanent circuits (FIA International
#: Sporting Code, Appendix O), so a real track is at least this wide.
DEFAULT_TRACK_WIDTH_M: Final = 12.0

#: Mean Earth radius (m), IUGG value -- for projecting GPS circuit centerlines
#: to local metres.
EARTH_MEAN_RADIUS_M: Final = 6_371_008.8

#: Window (m) of the moving average applied along the track to the curvature of
#: a circuit's GPS centerline. The dataset's points are hand-digitised ~25-50 m
#: apart, so curvature from an exact interpolating spline carries wiggles at
#: that scale; averaging over ~one point spacing removes them while leaving a
#: true constant-radius corner unchanged. A processing choice, not an F1 figure.
CIRCUIT_CURVATURE_WINDOW_M: Final = 25.0

#: Stations over which a racing line's curvature is averaged before the limit-speed
#: profile is built from it (odd, so the average is centred). Removes point-to-point
#: noise from a polyline; any longer flattens real apex peaks and flatters the lap.
LINE_CURVATURE_WINDOW_STATIONS: Final = 3

# Standard acceleration of gravity, BIPM SI Brochure (9th ed.), exact by definition.
GRAVITY_M_S2: Final = 9.80665


@dataclass(frozen=True)
class CarParams:
    """Point-mass car model parameters.

    Every field is required and must be calibrated (phase 1) -- see the module
    docstring. Record the source alongside each value at the call site.
    """

    mass_kg: float
    max_lateral_g: float
    max_accel_g: float
    max_braking_g: float
    max_speed_ms: float
    # Aerodynamic drag as a deceleration drag_per_m * v^2 (1/m), applied by
    # `models.frenet.frenet_step` on top of the throttle/brake. 0 = no drag (the
    # car accelerates at max_accel_g right up to max_speed_ms). Not a free number:
    # `RaceEnvParams.aero_drag` sets it to max_accel_g * g / max_speed_ms^2, the
    # drag at which full throttle just holds top speed.
    drag_per_m: float = 0.0


@dataclass(frozen=True)
class TrackParams:
    """Parameterized track geometry.

    A lap is a sequence of constant-curvature segments; the concrete segment list
    is built by the track model (phase 1), not stored here.
    """

    width_m: float
    total_length_m: float


#: Illustrative F1 point-mass car for phase-1 sanity checks. Figures are the
#: approximate, widely-reported values from F1 technical media/regulations
#: (FIA minimum weight; Pirelli/Brembo/team technical explainers for grip and
#: top speed) -- precise per-corner calibration against FastF1 telemetry is
#: future work, not yet done.
EXAMPLE_CAR: Final = CarParams(
    mass_kg=798.0,  # FIA F1 Technical Regulations: minimum weight incl. driver.
    max_lateral_g=5.0,  # Peak sustained cornering g reported for modern F1 cars.
    max_accel_g=1.5,  # Peak longitudinal acceleration reported for F1 cars.
    max_braking_g=5.0,  # Peak braking deceleration reported for F1 cars.
    max_speed_ms=97.0,  # ~350 km/h, reported top speed on DRS-assisted straights.
)


@dataclass(frozen=True)
class SimParams:
    """Integration and episode settings.

    These are simulation choices rather than measured F1 quantities, so they carry
    defaults. ``seed`` is explicit and mandatory everywhere: no unseeded RNG and no
    wall-clock randomness anywhere in ``src/``.
    """

    seed: int
    dt_s: float = 0.02  # 50 Hz control loop.
    max_episode_steps: int = 10_000


@dataclass(frozen=True)
class DriverEnvParams:
    """Tunables for `DriverEnv`'s dynamics, observation, and reward shaping.

    Simulation design choices, not measured F1 quantities -- see `SimParams`.
    """

    # Edges (m) of the lookahead windows: the observation reports the slowest
    # corner speed limit in 0-25 m, 25-50 m, ... ahead. The farthest must cover a
    # full-speed stop for a slow hairpin: (97^2 - 20^2) / (2 * 5 g) ~ 92 m for
    # `EXAMPLE_CAR`.
    lookahead_m: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0)
    # The observation's braking margin, (safe speed - speed), is divided by this
    # and clipped to +-1: the brake-or-not decision then turns on an input of
    # order 1, not on the difference of two near-equal speed fractions.
    margin_obs_scale_ms: float = 20.0
    # Braking-point countdown in the observation: distance left before the car
    # must brake (at its current speed) for any corner within this horizon,
    # divided by it, clipped to [-1, 1]. The speed margin above only "wakes up"
    # within ~2-3 decisions of the braking point at high speed; this one counts
    # down from 300 m, like a driver's marker boards. 0 = leave it out.
    brake_point_horizon_m: float = 0.0
    drift_gain_m_s_per_g: float = 5.0  # forced lateral drift speed per g over grip.
    # Steering turns the velocity vector at most this far off the track direction,
    # as a lateral/forward speed ratio (0.1 ~ 5.7 deg) -- scaled by the fraction of
    # lateral grip the corner leaves unused, so a stationary car can't move sideways
    # and steering can't out-steer physics.
    max_heading_ratio: float = 0.1
    off_track_penalty: float = 1.0  # reward subtracted, episode ends, when exceeded.
    # Reward subtracted per simulated second. Progress pays 1.0 per lap whatever the
    # pace, so this is what makes a faster lap score higher.
    time_penalty_per_s: float = 0.01
    # A second time penalty, normalised per track: a lap at that track's limit
    # pace costs exactly this much (per-second rate = this / limit lap time).
    # Unlike a fixed per-second rate, it means the same thing on a 60 s and a
    # 95 s circuit, and while it is below 1 + off_track_penalty, going off early
    # can never score better than finishing.
    time_penalty_per_limit_lap: float = 0.0
    # Reward subtracted per second the car is past a braking point (faster than
    # `max_safe_speed_ms`), scaled by the overspeed as a fraction of top speed.
    # Graded, so braking a little earlier always scores a little better -- the
    # off-track penalty alone is a cliff with no gradient toward braking.
    overspeed_penalty_per_s: float = 1.0
    # Reward per second for using the speed available: min(v, v_safe) / v_safe.
    # Makes running below the safe speed an immediate cost, instead of a tiny
    # long-run one (lifting 100 m early costs ~0.003 of progress/time reward,
    # lost among off-track and overspeed penalties of ~1). 0 = off.
    speed_use_reward_per_s: float = 0.0
    # A car slower than `stall_speed_ms` for `stall_timeout_s` has stopped on
    # track: the episode ends like an off-track (same penalty). Without this,
    # parking before a hard corner is a risk-free way to dodge the off-track
    # penalty, and agents learn to do exactly that.
    stall_speed_ms: float = 2.0
    stall_timeout_s: float = 3.0
    # Physics steps each action is held for. 1 = a decision every `dt_s`; 5 = every
    # 0.1 s, which makes exploration correlated in time and episodes 5x shorter.
    action_repeat: int = 1
    # Reset draws the starting lateral offset uniformly from [0, this], from the
    # env's seeded RNG -- so episodes differ but stay reproducible.
    start_offset_max_m: float = 1.0


@dataclass(frozen=True)
class BaselineParams:
    """Tunables for the rule-based baseline driver (phase 3).

    Design choices for a scripted driver, not measured F1 quantities.
    """

    # A cautious scripted driver, not an optimal one: it plans corner speeds for
    # this fraction of the car's lateral grip...
    corner_grip_margin: float = 0.90
    # ...and places its braking points as if the car could only brake at this
    # fraction of its real capacity, so it brakes early.
    braking_margin: float = 0.80
    # Each episode the driver's margins are jittered by Gaussian noise of this
    # std-dev, kept within [margin_floor, 1.0] (1.0 is the car's real limit):
    # lap-to-lap confidence varies, which is what makes 20 runs differ. Drawn
    # from the driver's seeded RNG.
    margin_jitter_std: float = 0.02
    margin_floor: float = 0.5  # jittered margins never drop below this.


@dataclass(frozen=True)
class AnalyticsParams:
    """How lap analytics split a lap and sample telemetry -- reporting choices."""

    # Tighter than this (radius under 500 m) counts as a corner...
    corner_curvature_per_m: float = 0.002
    # ...unless shorter than this: real-circuit curvature has short noise kinks.
    min_corner_m: float = 10.0
    telemetry_every_n_steps: int = 5  # 10 Hz telemetry at the 50 Hz control loop.
    map_step_m: float = 5.0  # spacing of the reconstructed track-map points.


@dataclass(frozen=True)
class GuardParams:
    """The defending sparring driver (``guard_*`` opponents, `f1rl.agents.guard`).

    It defends each corner on its own: one move per straight (toward the inside of
    the next corner, before the braking zone, never once the attacker is
    alongside), then back to the racing line after the apex. The rules follow the
    FIA Sporting Regulations' driving standards (one change of direction to defend,
    no moves under braking, leave a car's width to a car alongside). The distances
    and times below are design choices that set how hard it is to pass, not F1
    figures. Lower `inside_gap_s` / higher `reaction_s` = an easier sparring partner.
    """

    # React to a car behind within this many metres; ignore it further back.
    react_gap_m: float = 60.0
    # Close the inside of the next corner once the car behind is within this time
    # gap (s, gap / our speed) and the corner's braking point is within
    # `inside_from_m`. 0.35 s is about 25 m at 75 m/s: close enough to dive.
    inside_gap_s: float = 0.35
    inside_from_m: float = 300.0
    # ...and only if, at the speed it is closing now, it would be within a car
    # length + this (m) of us by the braking point: one that cannot get alongside
    # is not worth the time a defensive line costs (~0.1-0.4 s a corner here).
    threat_margin_m: float = 4.0
    # Before that, cover a car that has pulled out toward the corner's inside by at
    # least this much (m, sideways from us) on the straight.
    cover_lateral_m: float = 1.0
    # Wait this long (s, as distance at our speed) after a threat appears before
    # moving: the gap a feint (``dummy``) can use. 0 = instant.
    reaction_s: float = 0.15
    # The move needs at least this much road (m) before the braking point; shorter
    # would be a lunge across the track (3 m sideways at 80 m/s over 60 m already
    # peaks at ~2.7 g of lateral acceleration).
    min_ramp_m: float = 60.0
    ramp_m: float = 120.0
    # ...and must finish this far (m) before the braking point.
    settle_m: float = 15.0
    # How far inside it goes: it leaves this much room (m, for a car's centre) on
    # the inside, under a car width so a dive does not fit. Shifts under
    # `min_shift_m` are not worth making; never more than `max_shift_m`.
    inside_leave_m: float = 1.2
    min_shift_m: float = 1.0
    max_shift_m: float = 3.0
    # Share of the inside shift still held at the apex (the line itself is on the
    # inside there): more = a tighter, slower corner that is harder to undercut.
    apex_keep: float = 0.0
    # The door is shut over this stretch (m) before the braking point.
    door_m: float = 60.0
    # Back on the racing line this far (m) after the apex, or before the next
    # braking zone if that comes first.
    recover_m: float = 120.0
    # A car is alongside once it overlaps us by this much (m): its front wheels
    # level with our rear wheels (~1 m from each end of a 5.5 m car). From then on
    # no moves toward it until it has dropped back or passed.
    overlap_m: float = 1.5
    # No new move once the car behind is within a car length + this (m, centre to
    # centre): moving
    # across its nose that late forces it to swerve (or off the road).
    min_move_gap_m: float = 8.0
    # ...and it keeps its centre this far (m) from that car's centre sideways: a
    # car width (2.0 m) plus a gap.
    room_m: float = 3.0


@dataclass(frozen=True)
class SectionParams:
    """How `f1rl.sections` cuts a track into overtaking sections (design choices, in
    metres along the track)."""

    # Braking-zone battles start this far before the braking point.
    approach_m: float = 200.0
    # ...and must be won by this far past the apex.
    past_apex_m: float = 30.0
    # Corner battles start this share of the way from the braking point to the apex
    # (the turn-in) and end this far past the apex (the exit), or at the top of the
    # ramp out of the corner if sooner.
    turn_in_frac: float = 0.5
    corner_exit_m: float = 300.0
    # Straight battles start this far past the previous corner's apex.
    straight_from_apex_m: float = 10.0
    # Corner battles start half alongside: this far behind (m, centre to centre; a
    # car is 5.5 m long) and this far to the inside or outside of the racing line.
    corner_gap_m: float = 3.0
    corner_shift_m: float = 3.0
    # Sections shorter than this are left out (no room for a pass).
    min_length_m: float = 80.0
    # True: straight and braking battles must be won by the corner's exit (as corner
    # battles), not by the braking point / just past the apex. A pass set up on a
    # straight is finished in the braking zone; one started under braking is
    # finished through the corner. With False, even a car 30% down on grip was
    # passed in only 7 of 90 validation sections (checked 2026-10-29).
    finish_at_exit: bool = False
    # Corners a battle spans: 1 = one section (to that corner's exit); 2 or 3 = the
    # pass may be set up in one corner and finished in the next, by the exit of the
    # `span_corners`-th corner from the start. A single section is often too short
    # to pass a car only slightly slower (held-out 2026-10-29: 30/350 clean).
    span_corners: int = 1


@dataclass(frozen=True)
class RaceEnvParams:
    """Tunables for `RaceEnv` (env v2: the car steers, so it chooses its line).

    Simulation design choices, not measured F1 quantities -- see `SimParams`.
    """

    # The car's centre must stay this far inside either edge. Half of 2.0 m, the
    # maximum overall car width in the FIA Formula One Technical Regulations.
    car_half_width_m: float = 1.0
    # Extra gap the optimal-line reference keeps from the edge, so the reference
    # is a line the env can actually drive without touching its own limit.
    line_safety_m: float = 0.75
    # Physics steps each action is held for (5 at 50 Hz = a decision every 0.1 s).
    action_repeat: int = 5
    # Training episodes run for this long and are then truncated (not terminated):
    # reward is distance covered, so a fixed time budget IS "go as fast as you can".
    episode_s: float = 40.0
    # Random starts (training): anywhere on the lap, this speed range, and a
    # lateral offset up to this fraction of the free half-width either side.
    start_speed_ms: tuple[float, float] = (20.0, 60.0)
    start_offset_frac: float = 0.6
    start_heading_rad: float = 0.1
    # Fail (episode ends, `fail_penalty`) below this speed -- a stopped car is
    # retired -- or when pointing this far off the direction of travel.
    min_speed_ms: float = 10.0
    max_heading_rad: float = 1.5
    # Reward: distance advanced along the track / this, so one decision at
    # 80 m/s pays 0.08.
    progress_scale_m: float = 100.0
    fail_penalty: float = 5.0
    # Observation geometry. Windows (m ahead, from 0): the tightest corner in each
    # becomes one signed feature; edge points: where the edges lie that far ahead.
    lookahead_windows_m: tuple[float, ...] = (10.0, 25.0, 50.0, 100.0, 150.0, 250.0)
    edge_points_m: tuple[float, ...] = (25.0, 50.0, 100.0, 150.0)
    edge_scale_m: float = 10.0  # edge distances are divided by this
    # Hybrid mode: also observe the optimal line -- its lateral shift (relative to the
    # car) and limit speed at each of `reference_points_m` ahead, and the heading
    # error to it. Off = the driver must find the line from the track alone.
    reference_features: bool = False
    reference_points_m: tuple[float, ...] = (0.0, 25.0, 50.0, 100.0, 200.0)
    # Driving-style hooks (Phase 8, unused): a 0 (smooth) .. 1 (attacking) input
    # held in the observation, and a reward term on change of the commands.
    style: float = 0.5
    smooth_penalty: float = 0.0
    # Residual mode: the scripted `LineFollower` drives, and the policy's action is a
    # correction of at most this size (per unit command) added to the follower's
    # command. The policy also sees the follower's command. 0 = the policy drives.
    # Starts every run at the follower's pace (+2%) instead of at a random driver.
    residual_scale: float = 0.0
    # Tyres and driving mode (`models.tire`). A per-episode ``push`` in [0, 1] (0 =
    # conserve, 1 = attack) and the tyre ``wear`` are in the observation. Push sets
    # how much of the grip the base driver plans for (``conserve_grip_fraction`` at
    # 0 up to the follower's own at 1, in ``push_levels`` steps), and the reward pays
    # ``(1 - push) * wear_cost_per_unit`` per unit of wear -- how much a worn tyre is
    # worth to the strategist. Worn tyres give less grip (`TireParams`).
    tires: bool = False
    # Tactical env (`envs.tactical_env`): the action picks a race-craft move and a
    # scripted path driver executes it. Needs a reference line.
    tactical: bool = False
    push_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    conserve_grip_fraction: float = 0.85
    wear_cost_per_unit: float = 0.0
    start_wear_max: float = 0.7
    # Other cars (`envs.opponents`): kinds sampled one per episode, from
    # `OPPONENT_KINDS`. Empty = a solo car. The opponent starts this far
    # (min, max in m) ahead of or behind the car, depending on its kind.
    opponents: tuple[str, ...] = ()
    opponent_gap_m: tuple[float, float] = (15.0, 60.0)
    # Contact ends the episode (`fail_penalty`) when the cars overlap by less than
    # these: a car length and width. Length: FIA Technical Regulations fix the
    # wheelbase at 3.6 m max (2022+); 5.5 m overall is the usual reported length.
    # Width: the 2.0 m maximum overall width, as for `car_half_width_m`.
    car_length_m: float = 5.5
    car_width_m: float = 2.0
    # An opponent counts a car as alongside within a car length plus this.
    alongside_margin_m: float = 4.0
    # Dirty air: within this distance behind another car (and lined up with it), all
    # grip limits are scaled by (1 - dirty_air_loss). A design choice, not a measured
    # figure: it is what makes following close in corners cost something.
    dirty_air_m: float = 25.0
    dirty_air_loss: float = 0.04
    # Does the wake also cut full-throttle acceleration? True = earlier runs. The
    # wake costs downforce, which limits cornering and braking; on a straight the
    # car behind gains (less drag in the tow), it does not lose traction. With
    # True, a car within `dirty_air_m` on a straight accelerates `dirty_air_loss`
    # worse than the car it follows and can hardly close up in the tow.
    dirty_air_traction: bool = True
    # Overtake assist (DRS plus slipstream): on a straight (centreline curvature under
    # `tow_max_curvature`), a car within `tow_m` behind another gets `tow_boost_ms` of
    # extra top speed whatever its sideways position. 6 m/s (~20 km/h) is the scale of
    # the reported DRS gain (~10-15 km/h) plus tow; a design choice. Without a speed
    # edge on the straight a car on the racing line can never be caught, and no
    # overtake exists.
    tow_m: float = 80.0
    tow_max_curvature: float = 0.002
    tow_boost_ms: float = 6.0
    # The assist also while the cars overlap, until the car behind is a car length
    # past: DRS (part of the assist) stays open until the driver brakes, so the
    # attacker can complete the pass down the straight. False = earlier runs: the
    # assist ended with the noses overlapping and the cars ran side by side at the
    # same top speed into the braking zone (checked 2026-10-31, Monza T1).
    tow_while_alongside: bool = False
    # The base driver plans its speeds for the wake from this far behind a car.
    wake_anticipation_m: float = 75.0
    # Slipstream as extra ACCELERATION, not just top speed: the point-mass car has no
    # drag, so a top-speed boost alone only helps once both cars are flat out at the
    # cap. Real tow/DRS works by cutting drag along the whole straight. On a straight
    # within `tow_m` the car behind gets max_accel_g * (1 + this). A design choice,
    # not an F1 figure. 0 = off (earlier runs).
    tow_accel_gain: float = 0.0
    # Air drag (`CarParams.drag_per_m`): full throttle fades as speed builds, so a
    # straight is a drag race, and the car in the tow (on a straight within `tow_m`)
    # has its drag cut by `tow_drag_cut`. That is how slipstream and DRS really
    # work: less drag along the whole straight, not just a higher top speed. With
    # drag the top speed is approached, not hit, so the tow's speed cap
    # (`tow_boost_ms`) only matters if it is set below the tow's own top speed,
    # max_speed_ms / sqrt(1 - tow_drag_cut). False = earlier runs (no drag).
    aero_drag: bool = False
    # Share of drag the tow (slipstream plus DRS) removes. A design choice, not an
    # F1 figure: 0.12 raises the tow's top speed ~6.6% (~23 km/h at 350), the scale
    # of the reported DRS gain (~10-15 km/h) plus slipstream.
    tow_drag_cut: float = 0.12
    # The tactical driver's extra view (`TacticalEnv`): the corner after next
    # (distance to its braking point, its side), the metres left in a section
    # battle, and where the car ahead sits off the racing line now and how that
    # changed over the last second (how it is defending). False = earlier runs.
    plan_features: bool = False
    # Scripted opponents plan their braking for this fraction of the brake limit
    # (`FollowerParams.braking_fraction`), whatever their grip: a slower car then
    # loses its time through the corners and out of them, not by braking early.
    # 0 = the same as its grip (earlier runs: a 0.80 car braked ~25-30 m before
    # us into fast corners, checked on Hockenheim 2026-10-29). Our own plan uses
    # 0.95 (`FollowerParams.grip_fraction`).
    opponent_braking_fraction: float = 0.0
    # ...and plan their acceleration for this fraction of the throttle limit (0 =
    # their grip, earlier runs: a 0.80 car was slow out of every corner and down
    # the straight after it). With 0.95 and `slow_power_frac` 1.0 a slower car is
    # slower only through the corners, and on a straight it is caught only in the
    # tow -- as in F1, where the car behind needs the slipstream to pass.
    opponent_accel_fraction: float = 0.0
    # The tactical "shield" that brakes the car behind whenever it is lined up within
    # 12 m + 1 s of closing (and the same rule in scripted opponents). With the
    # no-crash filter on it is not needed for safety, and it stops the classic pass:
    # sit in the tow, pull out late. False = leave avoidance to the filter.
    follow_shield: bool = True
    # ...and False here lifts the shield only while the car is in the tow on a
    # straight (where a pass is set up); it still holds in braking zones and corners,
    # where running into the back of a car is the risk.
    shield_in_tow: bool = True
    # The shield brakes just enough to match the car ahead's speed (True) instead of
    # braking at 80% every time (False, earlier runs): following used to cost so much
    # time that even a slower car ahead drew away.
    soft_follow: bool = False
    # Tactical env: once a move (other than ``line``) starts, it runs to its end (or
    # at most `commit_max_decisions` decisions) as ONE step of the agent. Earlier runs
    # let every 0.5 s decision replace the running move, so a random policy dithered
    # and never finished one, and a low-entropy one never tried another.
    commit_moves: bool = False
    commit_max_decisions: int = 24
    # Tactical env: the moves the agent may pick (its action space, in this order).
    # Empty = all of `agents.tactics.MOVES`. A defender uses ("line", "cover",
    # "inside"). The observation still shows the running move out of all MOVES.
    tactical_moves: tuple[str, ...] = ()
    # Defending rules for the agent's own moves (as `GuardParams`): no new cover or
    # inside once the car behind is within a car length + `guard.min_move_gap_m`
    # (moving across its nose that late forces it to swerve). False = earlier runs.
    defend_rules: bool = False
    # With another car behind us, only the defending moves (`line`, `cover`,
    # `inside`): any other pick becomes `line` and counts as refused. Stops a
    # defender from weaving with moves made for attacking (a lane move re-aims at the
    # other car every decision: mirroring it is blocking). False = earlier runs.
    defend_moves_only: bool = False
    # Extra opponent features: its sideways speed, a pass waiting to stick (+1 when
    # we lead and it has not stuck yet, -1 when we were passed and it has not stuck,
    # scaled by how far into `pass_hold_m` it is), whether it is in our tow (a
    # threat), and its pace edge over us. False = the earlier observation.
    opponent_features: bool = False
    # Scripted opponents brake hard and move 2.5 m aside once a car is alongside.
    # Self-play rivals never do, and the policy cannot tell them apart. False = no
    # scripted yield (the no-crash filter still keeps every car clear).
    opponent_yield: bool = True
    # The scripted path driver keeps its planned path this far (m) inside the room
    # the car's centre has. 0.2 = earlier runs; close racing pushes the car onto that
    # path's edge, and a small tracking overshoot then ended the battle.
    path_edge_inset_m: float = 0.2
    # `FollowerParams.saturated_brake` for every path driver (ours and the
    # opponents'): braking kept when steering is maxed out and the car is too fast.
    # 0 = earlier runs.
    path_saturated_brake: float = 0.0
    # Track limits: the car's centre may go this far (m) past the line where its side
    # touches the edge before it counts as off the track. 0 = earlier runs (any
    # touch of the edge). A design choice: 1.0 would be the centre on the edge line.
    track_limit_margin_m: float = 0.0
    # Reward paid (lost) when the car passes (is passed by) an opponent.
    overtake_bonus: float = 20.0
    # ...but only once the pass has stuck: the new order held for this many metres
    # of driving (0 = paid the moment the order changes). ~1 km is about 3 corners.
    pass_hold_m: float = 0.0
    # GT Sophy-style passing reward (Wurman et al., Nature 2022): reward per metre
    # gained on (lost to) the opponent while it is within (behind, ahead) metres.
    # The window is the paper's b = 20, f = 40. 0 = off.
    pass_reward_per_m: float = 0.0
    pass_window_m: tuple[float, float] = (20.0, 40.0)
    # Reward lost each time the no-crash filter has to change our command.
    intervention_penalty: float = 0.0
    # No-crash safety filter (`models.safety`) on BOTH cars. A command is kept only
    # if, after it, both cars could still fall back on the rules of the road (keep
    # the lane; the car behind brakes) without coming within a car length +
    # `safety_margin_s_m` and a car width + `safety_margin_n_m`, checked
    # `safety_horizon_s` ahead (longer for a car closing fast, up to
    # `safety_max_horizon_s`) in `safety_dt_s` steps. Otherwise the least intrusive
    # safe command replaces it (brake and/or steer away by `safety_steer`). Checked
    # only within `safety_check_m` of each other, to keep training fast. Off by
    # default so earlier runs reproduce; every new config turns it on. Design
    # choices, not F1 figures.
    safety_filter: bool = False
    # The defending sparring driver's settings (``guard_*`` opponents).
    guard: GuardParams = GuardParams()
    # Section battles (`f1rl.sections`, reset option ``section_m``): the pass must be
    # done (our car ahead) within ``section_m`` of the start. Then the battle goes on
    # until the pass has stuck (`pass_hold_m`, plus `section_tail_m` of slack), and
    # ends at once if we are still behind. Ahead at the section's end pays
    # `section_win_bonus`; still behind costs `section_fail_penalty`. 0 = off; no
    # effect on battles without ``section_m``.
    section_win_bonus: float = 0.0
    section_fail_penalty: float = 0.0
    section_tail_m: float = 100.0
    # Opponents slower than us through the corners (grip below ours: backmarker,
    # slow, guard_slow, guard_mid) also get this share of our top speed and
    # acceleration. 1.0 = earlier runs (slower only in corners, so on a straight a
    # slower car could never be caught out of the tow). A design choice, not an F1
    # figure: 0.97 is ~10 km/h down at 350 km/h.
    slow_power_frac: float = 1.0
    # Clean racing: a "rescue" is a no-crash filter step on either car while the
    # cars are within a car length plus `rescue_margin_m` of each other (wheel to
    # wheel). Each costs `rescue_penalty` on top of `intervention_penalty`. (The
    # filter is cautious: it also steps in with metres of air, so rescues are
    # reported but do not decide a win; the clean_* rules below do.)
    rescue_margin_m: float = 1.0
    rescue_penalty: float = 0.0
    # Clean racing by geometry, checked every physics step: side by side, at least
    # `clean_side_m` of air between the cars; ahead in the other car's lane, at
    # least `clean_chop_m` in front of its nose (no chop after a pass); behind in its
    # lane, at least `clean_follow_m` off its gearbox. Each step closer costs
    # `dirty_penalty`, and a section battle with any is not won. Design choices.
    clean_side_m: float = 0.5
    clean_chop_m: float = 3.0
    clean_follow_m: float = 0.5
    dirty_penalty: float = 0.0
    # Side by side, the "side" rule blames our car only while it is the one closing
    # in (its own sideways step toward the other car, and the gap across shrinking).
    # A car that squeezes us onto our own line is its fault, not a dirty move of
    # ours. Checked 2026-10-29: of 43 "dirty" two-corner battles of the scripted
    # driver, 20 ended ahead and 21 had "side" steps. False = earlier runs.
    side_own_fault: bool = False
    # In a defence (the other car started behind), the "chop" rule counts only our
    # sideways steps further off our own racing line toward the car behind (a move
    # across its nose); driving the line's own sweep across the road, or moving
    # back to the line, is not. Alongside, the "side" rule still asks for room.
    # Attack battles keep the old rule (no cutting back in front after a pass).
    # Checked 2026-10-30: a defender that only drove the racing line was "dirty"
    # in 130 of 240 validation defend battles. False = earlier runs.
    chop_line_relative: bool = False
    # Tactical env: after a pass (or alongside), hold the current offset rather than
    # move back across into the other car's lane before clear of its nose by
    # `clean_chop_m` plus a car length (`TacticalEnv._no_chop`). False = earlier runs.
    hold_until_clear: bool = False
    # Tactical env: also offer the `tactics.PULL_MOVES` (pull alongside now, even in
    # a braking zone or a corner). Changes the action and observation sizes, so
    # False = earlier runs and their models.
    pull_moves: bool = False
    # Tactical env: also offer `tactics.LANE_MOVES`, direct control: every decision
    # the attacker picks a spot beside the car ahead (left or right, 3.0 or 4.2 m
    # centre to centre) and normal or late braking, re-aimed every decision, under
    # the same no-crash filter and clean-racing rules. Changes the action and
    # observation sizes, so False = earlier runs and their models.
    lane_moves: bool = False
    safety_horizon_s: float = 2.5  # was 1.5; 1.5 s missed side-by-side corner entries
    safety_max_horizon_s: float = 3.0
    safety_dt_s: float = 0.05
    # After its command, a car is assumed to straighten up to the track this fast (s).
    safety_lane_time_s: float = 0.5
    safety_margin_s_m: float = 1.0
    safety_margin_n_m: float = 0.3
    safety_check_m: float = 80.0
    safety_steer: float = 0.8  # was 0.4: too little to turn away from full lock
    safety_brake: float = 0.8
    # The filter slows a car behind by lifting off (then braking lightly) before it
    # tries braking hard (`SafetyCheck.lift_first`). False = earlier runs. With it
    # off, the filter braked our car at ~4 g with a slower car 9 m ahead pulling
    # away out of a corner (checked 2026-10-29), so a pass could rarely be set up.
    safety_lift_first: bool = False
    # ...and keeps a car on the road beside the other one only while they overlap
    # (`SafetyCheck.edge_overlap_only`). False = earlier runs. With it off, our car
    # lined up 9 m behind on a curving straight was held back every few steps
    # (checked 2026-10-29, Hockenheim T6).
    safety_edge_overlap_only: bool = False
    # ...and never slows a car ahead for a car closing on it from behind
    # (`SafetyCheck.leader_keeps_pace`). False = earlier runs.
    safety_leader_keeps_pace: bool = False
    # Real starts (`f1rl.real_starts`): a gap given in seconds is turned into metres
    # at the racing line's speed where the scenario starts, then kept in this range
    # (a car length plus a little, up to where the tactician starts an attack).
    real_gap_m: tuple[float, float] = (8.0, 60.0)


@dataclass(frozen=True)
class FollowerParams:
    """Tunables for the scripted optimal-line follower (`agents.line_follower`)."""

    # How long each action is held (SimParams.dt_s * RaceEnvParams.action_repeat).
    decision_s: float = 0.1
    # Lateral tracking is a PD law on the offset and heading error with this natural
    # frequency (rad/s) and damping ratio.
    bandwidth_rad_s: float = 5.0
    damping: float = 1.0
    # Target speeds come from a car with this fraction of the real grip on every
    # axis, so the tracking loop always has grip left to steer back onto the line.
    grip_fraction: float = 0.95
    # ...but braking is planned for this fraction of the brake limit if set (None = the
    # same as `grip_fraction`): braking earlier costs little time, and it is where
    # steering and braking compete for the same grip.
    braking_fraction: float | None = None
    # ...and acceleration out of corners for this fraction of the throttle limit if
    # set (None = the same as `grip_fraction`).
    accel_fraction: float | None = None
    # The speed controller aims to close the gap to the target speed in this long.
    speed_time_constant_s: float = 0.25
    # A path driver whose steering is maxed out while it is too fast still keeps
    # up to this much of the brake (a share of full braking, giving up that much
    # steering on the friction ellipse). 0 = steering always wins and the car
    # cannot slow down: pushed off its line it slides off the road at constant
    # speed. Slowing shrinks the turn it needs (radius ~ v^2), so it makes the turn.
    saturated_brake: float = 0.0


@dataclass(frozen=True)
class TireParams:
    """Tyre wear and grip (`models.tire`). **Simulation design choices, uncalibrated**:
    no FastF1/Pirelli fit has been done, so none of these is an F1 figure. They are
    set so the trade-off exists -- wear grows with the square of how hard the tyres
    are worked, and worn tyres give less grip -- not to match a compound. Replace
    with calibrated values (docs/context.md) before reading any result as F1.
    """

    # Seconds a tyre lasts if worked at the full friction limit the whole time
    # (wear reaches 1.0). Wear per second is `(usage ** wear_exponent) / life_s`.
    life_s: float = 1500.0
    wear_exponent: float = 2.0
    # Grip multiplier = 1 - loss_linear * wear - loss_cliff * wear ** cliff_power:
    # a steady fade plus a late cliff.
    loss_linear: float = 0.10
    loss_cliff: float = 0.15
    cliff_power: float = 6.0


#: Scripted opponent kinds (`envs.opponents`): (grip fraction its path driver plans
#: for, where it starts relative to the car). The car's own base driver plans for
#: 0.95; lap speed goes roughly with the square root of grip, and a car must be about
#: 10-15% quicker through the corners to be able to pass on pace, because every
#: race-craft move costs time (measured with the scripted `tactician`: a 0.70 car is
#: passed 7 times in 8, a 0.82 one almost never). Design choices for training
#: scenarios, not F1 figures.
OPPONENT_KINDS: Final = {
    "backmarker": (0.70, "ahead"),  # lapped traffic: clearly slower, to be passed
    "slow": (0.80, "ahead"),  # about 8% slower through corners
    "equal": (0.95, "ahead"),  # same pace: a real battle
    "defender": (0.95, "ahead"),  # same pace, covers and defends the inside
    # Defending sparring partners (`f1rl.agents.guard`): race-craft by the book, one
    # move per corner, at three paces. Slow = the "slow" car's pace; equal = ours.
    "guard_slow": (0.80, "ahead"),
    "guard_mid": (0.88, "ahead"),
    "guard_equal": (0.95, "ahead"),
    # Race-like pace gaps (2026-10-31). A planned-grip share is not a lap-time gap:
    # measured in clean air over all 25 tracks against our car at push 1 on new
    # tyres, 0.80 ("slow") is ~2.0 s/lap slower, 0.88 ~0.8 s, 0.91 ~0.4 s and 0.95
    # ~0.1 s quicker. Real passes happen at ~0.5-1 s/lap, so these are the
    # battles the driver must win: a guard and a line-holder at ~0.4 s/lap.
    "guard_close": (0.91, "ahead"),
    "close": (0.91, "ahead"),
    "mid": (0.88, "ahead"),
    "attacker": (0.97, "behind"),  # a touch quicker; tries to pass the car
    "attacker_equal": (0.95, "behind"),  # same pace; tries to pass the car
    # Self-play: the same pace as the learner (`FollowerParams.grip_fraction`), its
    # moves picked by a copy of a learned policy (`TacticalEnv.rival`).
    "rival_ahead": (0.95, "ahead"),
    "rival_behind": (0.95, "behind"),
    # A learned defender as the car ahead at the "slow" car's pace (its policy is set
    # as `TacticalEnv.rival`, see `f1rl.agents.defender_rl.load_defender`).
    "rival_ahead_slow": (0.80, "ahead"),
    # Attackers for training a defender: the rule-based race-craft driver
    # (`agents.tactician`) picks this car's moves from its own view of the race.
    "rival_tactician": (0.95, "behind"),
    "rival_tactician_fast": (0.97, "behind"),
}


@dataclass(frozen=True)
class IncidentParams:
    """How `f1rl.incidents` finds real passes and defences in FastF1 race laps."""

    # A defence is the car behind staying within this gap (s) at the line. 1.0 s is
    # the DRS detection gap in the FIA F1 Sporting Regulations (2018-2025 seasons),
    # so it marks a car close enough to attack.
    attack_gap_s: float = 1.0
    # ...for at least this many laps in a row. A processing choice, not an F1 figure.
    min_defence_laps: int = 3
    # FastF1 `TrackStatus` codes a lap may contain and still count as clean racing.
    # "1" is track clear; 2 yellow, 4 safety car, 5 red, 6/7 virtual safety car
    # (FastF1 docs, `fastf1.api.track_status_data`). Yellow laps are left out too.
    clean_track_status: str = "1"


@dataclass(frozen=True)
class RealStartParams:
    """How `f1rl.real_starts` turns real passes and defences into sim scenarios.

    Processing choices, not F1 figures.
    """

    # A pass scenario starts this far (m) before where the real pass happened, so the
    # car has the road in front of it to set the move up (the place where the order
    # flipped is usually a braking zone).
    lead_m: float = 400.0
    # The car being passed counts as clearly slower ("slow" opponent) when its tyres
    # are this many laps older than the attacker's; otherwise it is an equal-pace
    # "defender". (Fresh-tyre deltas are why most real passes happen.)
    fresh_tyre_laps: float = 8.0


# --- Strategy Agent: whole-race simulation (`f1rl.models.race`, `envs.strategy_env`) --

#: A Grand Prix runs the fewest complete laps that exceed 305 km (FIA Formula One
#: Sporting Regulations, race distance; Monaco is the one exception at 260 km).
RACE_MIN_DISTANCE_M: Final = 305_000.0

#: Most fuel a car may use from start to finish: 110 kg (FIA Formula One Technical
#: Regulations, 2019-2025 seasons).
MAX_RACE_FUEL_KG: Final = 110.0

#: Dry-weather tyre compounds, softest first. In a dry race every car must use at
#: least two of them (FIA Formula One Sporting Regulations, tyre-use rule).
COMPOUNDS: Final = ("soft", "medium", "hard")


@dataclass(frozen=True)
class CompoundParams:
    """One tyre compound. **Uncalibrated design choices**, not Pirelli figures: set
    so softer = quicker but shorter-lived and a one- or two-stop race both make
    sense. Calibrate from FastF1 stint lap times (2021-2022 load on the laptop)
    before reading any strategy result as F1.
    """

    # Grip multiplier of a new tyre (1.0 = the car's own limits).
    grip: float
    # Seconds of driving at push 0.5 that wear a new tyre out (wear 0 -> 1); see
    # `StrategyParams.push_wear_ratio`. Grip against wear follows `TireParams`.
    life_s: float


@dataclass(frozen=True)
class StrategyRewardParams:
    """Reward of `StrategyEnv`. Simulation design choices."""

    # Per second our lap is slower than the reference lap (a fresh medium at normal
    # push with half the fuel): sums to minus race time, the main objective.
    time_weight: float = 0.1
    # Per place gained (lost: negative) on a lap.
    position_weight: float = 1.0
    # At the flag, per place behind the winner.
    finish_weight: float = 0.5
    # At the flag, when the car never ran two dry compounds (a disqualification).
    rule_penalty: float = 50.0
    # At the flag, for a win.
    win_bonus: float = 0.0


@dataclass(frozen=True)
class StrategyParams:
    """Lap-by-lap race simulation for the Strategy Agent. Simulation design choices
    unless a comment cites a source; none of the uncalibrated ones is an F1 figure.
    """

    # Cars in the race, ours included. 20 on a real grid; fewer runs faster.
    n_cars: int = 10
    soft: CompoundParams = CompoundParams(grip=1.0, life_s=1900.0)
    medium: CompoundParams = CompoundParams(grip=0.985, life_s=3000.0)
    hard: CompoundParams = CompoundParams(grip=0.97, life_s=4500.0)
    # Time a pit stop adds to the lap it is made on (s): pit lane at the speed limit
    # plus the stop. Uncalibrated; FastF1 pit-in/pit-out times can calibrate it.
    pit_loss_s: float = 22.0
    # Fuel at the start (kg), burnt evenly so the car finishes empty.
    start_fuel_kg: float = MAX_RACE_FUEL_KG
    # Stand-in driver: lap time = limit lap on the optimal line at the driver's
    # grip * (1 + driver_gap). 0.011 is our scripted PathDriver's measured average
    # gap to the optimal lap over the 24 usable TUM tracks (docs/project_status.md).
    driver_gap: float = 0.011
    # Push levels, 0 = conserve the tyres .. 1 = attack. The driver plans for
    # conserve_grip_fraction of the grip at 0 up to attack_grip_fraction at 1
    # (linear between); attack is `FollowerParams.grip_fraction`. A narrower span
    # than `RaceEnvParams.conserve_grip_fraction` (0.85): over a whole race, tyre
    # management trades a few tenths a lap, not seconds.
    push_levels: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    attack_grip_fraction: float = 0.95
    conserve_grip_fraction: float = 0.92
    # Tyre wear grows steeply near the limit: wear per lap is lap_time / life_s at
    # push 0.5, times push_wear_ratio ** (push - 0.5). So attacking wears the tyres
    # this many times faster than conserving.
    push_wear_ratio: float = 3.0
    # Rivals' car pace: each rival's grip is ours times 1 + U(-spread, +spread).
    car_pace_spread: float = 0.015
    # Rivals' strategies: "mixed" = random templates (`RIVAL_TEMPLATES`), "baseline"
    # = every rival runs the rule-based plan (`StrategyBaselineParams`), its stops
    # moved by up to +/- rival_stop_jitter of the race so they do not all pit at once.
    rivals: str = "mixed"
    rival_stop_jitter: float = 0.05
    # Following within this gap (s) at the start of a lap costs the whole lap this
    # share of grip (dirty air). Smaller than `RaceEnvParams.dirty_air_loss`, which
    # applies only within 25 m in corners.
    dirty_air_gap_s: float = 1.0
    dirty_air_loss: float = 0.005
    # Overtaking odds when a quicker car catches a slower one: probability of a
    # pass on a lap = sigmoid((pace advantage - pass_delta_s) / pass_slope_s), with
    # the pace advantage the two lap times' difference (s). A car that does not
    # pass crosses the line min_gap_s behind. These stand in for the Driver Agent's
    # race-craft until its real pass/defend rates are measured.
    pass_delta_s: float = 0.6
    pass_slope_s: float = 0.25
    min_gap_s: float = 0.3
    # Our driver's race-craft relative to the rivals (s of pace advantage added when
    # we attack, taken away from a rival attacking us). 0 = as good as the field.
    # Hook for the trained Driver Agent's measured pass and defend rates.
    attack_skill_s: float = 0.0
    defend_skill_s: float = 0.0
    # Tracks that make a race: one is drawn per episode.
    tracks: tuple[str, ...] = ()
    # Lap-time table of the stand-in driver: grip multipliers from grip_min to
    # grip_max in grip_step steps, and this many fuel loads from empty to full.
    grip_min: float = 0.6
    grip_max: float = 1.05
    grip_step: float = 0.01
    fuel_points: int = 5
    reward: StrategyRewardParams = StrategyRewardParams()

    def compound(self, name: str) -> CompoundParams:
        if name not in COMPOUNDS:
            raise KeyError(f"unknown compound {name!r}; known: {COMPOUNDS}")
        params: CompoundParams = getattr(self, name)
        return params


@dataclass(frozen=True)
class StrategyBaselineParams:
    """The rule-based strategy (`agents.strategy_baseline`): a fixed plan, as a race
    team would write it before the start. Design choices."""

    # Compound per stint, first to last: one stop per change.
    stints: tuple[str, ...] = ("medium", "hard")
    # Share of the race distance after which each stop is made (one per stop).
    stop_fractions: tuple[float, ...] = (0.45,)
    # Push level (index into `StrategyParams.push_levels`) for the whole race.
    push_index: int = 2
    # Pit early, whatever the plan, once wear passes this.
    max_wear: float = 0.8
