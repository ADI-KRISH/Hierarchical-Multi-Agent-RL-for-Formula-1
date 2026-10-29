"""Which sections can be won at all? A brute-force search over simple plans.

``uv run python scripts/section_oracle.py --config configs/X.yaml --tracks Monza``

For every section (`f1rl.sections`) and opponent, tries each move started after a
wait of 0, 3, 6 or 10 decisions on the racing line (then the racing line again).
A section is "passable" if any of these plans wins it. The share of passable
sections is the ceiling a learned driver is measured against; the plan that won
is written to ``--out`` (for inspection, not for training).
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from f1rl.agents.tactical import load_tactical_config, section_won
from f1rl.agents.tactics import MOVES
from f1rl.envs.racetrack import TUM_TRAIN, TUM_VALIDATION
from f1rl.scenarios import make_env
from f1rl.sections import KINDS, Section, env_sections, section_options

WAITS = (0, 3, 6, 10)


def try_plan(env: Any, sec: Section, opp: str, gap: float, move: int, wait: int) -> bool:
    env.reset(seed=0, options=section_options(sec, opp, gap))
    info: dict[str, Any] = {}
    for k in range(2000):
        a = move if k == wait else 0
        _, _, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return section_won(info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--tracks", default="validation")
    ap.add_argument("--opponents", default="slow")
    ap.add_argument("--gap", type=float, default=9.0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    config = load_tactical_config(args.config)
    params = replace(config.env_params(), safety_filter=True)
    sets = {"validation": TUM_VALIDATION, "train": TUM_TRAIN}
    names = sets.get(args.tracks, tuple(args.tracks.split(",")))
    rows = []
    for name in names:
        env = make_env(name, params, "random")
        for sec in env_sections(env, config.section_params()):
            for opp in args.opponents.split(","):
                plan = None
                for move in range(1, len(MOVES)):
                    for wait in WAITS:
                        if try_plan(env, sec, opp, args.gap, move, wait):
                            plan = (MOVES[move], wait)
                            break
                    if plan:
                        break
                rows.append({**sec.as_dict(), "opponent": opp, "plan": plan})
        print(name, sum(r["plan"] is not None for r in rows if r["track"] == name),
              "/", sum(r["track"] == name for r in rows), flush=True)
    for kind in KINDS:
        sel = [r for r in rows if r["kind"] == kind]
        print(f"{kind:10} passable {sum(r['plan'] is not None for r in sel)}/{len(sel)}")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
