"""A rule-based race-craft chooser: the expert the learned driver is compared with
(and can be taught by). Reads exact state from the env like the other scripted drivers.

Attacking (car ahead): pull out early enough to be lined up beside it by the braking
point -- ``divebomb`` the next corner when there is road to do it, ``dummy`` first when
there is room for a feint, ``outside`` when the inside is closed, ``switchback`` when
the corner is too close to attack but the next one is not.

Defending (car behind): ``inside`` when it is right behind us and a corner is near
(close the door), else ``cover`` its side once; ``line`` when nothing threatens.
"""

from f1rl.agents.tactics import MIN_ROOM_M, corners_ahead
from f1rl.envs.opponents import wrap_gap
from f1rl.envs.tactical_env import TacticalEnv

#: Start an attack when the car ahead is within this gap (m) -- close enough to be in
#: its tow, far enough to pull out before the shield has to brake.
ATTACK_GAP_M = (0.0, 60.0)
#: Defend when the car behind is within this gap (m).
DEFEND_GAP_M = 45.0
INSIDE_GAP_M = 20.0
FEINT_ROOM_M = 200.0
#: Be this close (m, centre to centre) or nearer by the braking zone to attack.
OVERLAP_BY_BRAKING_M = 4.0


def choose(env: TacticalEnv) -> str:
    opp = env._opp
    if opp is None:
        return "line"
    d = env._driver()
    me = env._state
    gap = wrap_gap(opp.state.s_m - me.s_m, env._lap_m)  # > 0: it is ahead
    corners = [c for c in corners_ahead(d, me.s_m) if c.apex_m > 5.0]
    room = corners[0].brake_m if corners else 0.0
    if 0.0 < gap < ATTACK_GAP_M[1]:
        if (
            env._maneuver is not None
            and env._move != "line"
            and me.s_m < env._maneuver.end_s
        ):
            return env._move  # committed
        # Commit only if the car will be alongside by the time it has to turn in:
        # the gap left after closing on the car ahead until the assist ends. Without
        # overlap at turn-in the right move is to stay behind (the shield brakes).
        if corners and room >= MIN_ROOM_M:
            p = env.params
            boost = p.tow_boost_ms if env._tow(me, opp.state) > 0 else 0.0
            closing = me.speed_ms + boost - opp.state.speed_ms
            t = max(0.0, room - 100.0) / max(me.speed_ms, 1.0)
            if gap - closing * t <= OVERLAP_BY_BRAKING_M:
                return "dummy" if room >= FEINT_ROOM_M else "divebomb"
        if corners and room < MIN_ROOM_M and len(corners) > 1:
            return "switchback"
        return "line"
    if -DEFEND_GAP_M < gap < 0.0:
        if corners and -gap < INSIDE_GAP_M and room >= 40.0:
            return "inside"
        return "cover"
    return "line"
