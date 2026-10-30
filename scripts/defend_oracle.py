"""Which lost defences could have been held? A brute-force search over simple plans.

``uv run python scripts/defend_oracle.py --config configs/defend_sections.yaml
--gate runs/gate_def3_line.json --attacker <model.zip>``

Reads a `section_gate.py` scorecard of defend battles and, for every battle NOT
held there, tries each defending move (``cover``, ``inside``) started after a
wait of 0..`WAITS` decisions on the racing line, then the line again. A loss is
"saveable" if any plan holds it: the ceiling a learned defender can add over the
policy that played the scorecard. Writes the plans to ``--out``.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from f1rl.agents.tactical import (
    defend_options,
    load_tactical_config,
    section_won,
    use_attacker,
)
from f1rl.scenarios import make_env
from f1rl.sections import env_sections

WAITS = (0, 2, 4, 6, 8, 10, 14, 18, 24)
MOVES = ("cover", "inside")


def try_plan(env: Any, opts: dict[str, Any], move: int, wait: int) -> bool:
    env.reset(seed=0, options=opts)
    line = env.moves.index("line")
    info: dict[str, Any] = {}
    for k in range(2000):
        _, _, term, trunc, info = env.step(move if k == wait else line)
        if term or trunc:
            break
    return section_won(info)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--gate", type=Path, required=True)
    ap.add_argument("--attacker", default="")
    ap.add_argument("--gap", type=float, default=9.0)
    ap.add_argument("--wear", type=float, default=0.3)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    config = load_tactical_config(args.config)
    params = replace(config.env_params(), safety_filter=True)
    lost = [r for r in json.loads(args.gate.read_text()) if not r["won"]]
    envs: dict[str, Any] = {}
    rows = []
    for r in lost:
        env = envs.get(r["track"])
        if env is None:
            env = envs[r["track"]] = make_env(r["track"], params, "random")
            if args.attacker:
                use_attacker([env], args.attacker)
        sec = next(
            s for s in env_sections(env, config.section_params()) if s.name == r["name"]
        )
        opts = defend_options(sec, r["opponent"], args.gap, args.wear)
        plan = None
        for name in MOVES:
            for wait in WAITS:
                if try_plan(env, opts, env.moves.index(name), wait):
                    plan = (name, wait)
                    break
            if plan:
                break
        rows.append({"name": r["name"], "opponent": r["opponent"], "plan": plan})
        print(r["name"], r["opponent"], plan, flush=True)
    saved = sum(x["plan"] is not None for x in rows)
    print(f"saveable {saved}/{len(rows)} lost defences")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
