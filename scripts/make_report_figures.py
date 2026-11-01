"""Draw the figures used in docs/report from the logged evaluation CSVs.

usage: uv run python scripts/make_report_figures.py

Inputs are copies of each run's eval.csv in docs/report/results/ (the runs/ folders
themselves are git-ignored). Seed runs for the stability study are read from
docs/report/results/seeds/ when present. Read-only: never trains anything.
"""

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "docs" / "report" / "results"
OUT = ROOT / "docs" / "report" / "figures"

BLUE, ORANGE, GREEN, GREY, RED = "#2a6fdb", "#e8833a", "#2e9d5b", "#8a8f98", "#c8403a"


def rows(name: str) -> list[dict[str, str]]:
    with (DATA / name).open(newline="") as f:
        return list(csv.DictReader(f))


def style(ax: plt.Axes, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


# --- diagrams ------------------------------------------------------------------------


def box(
    ax: plt.Axes, x: float, y: float, w: float, h: float, text: str, color: str
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02",
            fc=color,
            ec="#333",
            lw=1.2,
            alpha=0.9,
        )
    )
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9, wrap=True)


def arrow(
    ax: plt.Axes, a: tuple[float, float], b: tuple[float, float], label: str = ""
) -> None:
    ax.annotate(
        "", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.4, color="#333")
    )
    if label:
        ax.text(
            (a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + 0.02, label, ha="center", fontsize=8
        )


def rl_loop() -> None:
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    box(
        ax,
        0.06,
        0.62,
        0.30,
        0.22,
        "Agent\npolicy pi_theta(a|s)\nvalue V_phi(s)",
        "#cfe0fb",
    )
    box(
        ax,
        0.64,
        0.62,
        0.30,
        0.22,
        "Environment\nrace simulator\n(track, car, rivals, tyres)",
        "#d7f0de",
    )
    box(ax, 0.06, 0.12, 0.30, 0.22, "Rollout buffer\n(s, a, r, s', done)", "#fde3cf")
    box(
        ax,
        0.64,
        0.12,
        0.30,
        0.22,
        "Policy update\nPPO clipped loss / SAC\nentropy-regularised loss",
        "#f2d7f4",
    )
    arrow(ax, (0.36, 0.78), (0.64, 0.78), "action a_t")
    arrow(ax, (0.64, 0.68), (0.36, 0.68), "state s_t+1, reward r_t")
    arrow(ax, (0.21, 0.62), (0.21, 0.34), "store")
    arrow(ax, (0.36, 0.23), (0.64, 0.23), "mini-batches")
    ax.annotate(
        "",
        xy=(0.36, 0.70),
        xytext=(0.64, 0.34),
        arrowprops=dict(arrowstyle="->", lw=1.2, color=BLUE, ls="--"),
    )
    ax.text(0.53, 0.47, "new theta, phi", color=BLUE, fontsize=8)
    ax.text(
        0.5,
        0.02,
        "Training: collect -> update -> evaluate on validation tracks -> "
        "keep best checkpoint.  "
        "Inference: frozen policy, deterministic action.",
        ha="center",
        fontsize=8,
        color="#333",
    )
    save(fig, "fig1_rl_loop.png")


def pipeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    # strategy level
    box(
        ax,
        0.02,
        0.72,
        0.18,
        0.18,
        "Lap-by-lap race model\n10 cars, tyres, fuel,\nSC/VSC, rain, incidents",
        "#d7f0de",
    )
    box(ax, 0.25, 0.72, 0.16, 0.18, "Race-state\nextraction\n(46 inputs)", "#eee")
    box(
        ax,
        0.46,
        0.72,
        0.17,
        0.18,
        "Strategy Agent\nPPO, MLP 64x64\n(once per lap)",
        "#cfe0fb",
    )
    box(ax, 0.68, 0.72, 0.14, 0.18, "Pit + compound\npush mode 0-4", "#fde3cf")
    box(ax, 0.86, 0.72, 0.12, 0.18, "Reward:\n-0.1/s race time\n+20 win", "#f2d7f4")
    # driver level
    box(
        ax,
        0.02,
        0.30,
        0.18,
        0.20,
        "RaceEnv (50 Hz)\nFrenet car, friction\n"
        "ellipse, 25 TUM tracks,\nrivals, tow, drag",
        "#d7f0de",
    )
    box(
        ax,
        0.25,
        0.30,
        0.16,
        0.20,
        "Observation\n(75 inputs: corners\nahead, line offset,\nrival gaps, tyres)",
        "#eee",
    )
    box(
        ax,
        0.46,
        0.30,
        0.17,
        0.20,
        "Driver Agent\nPPO, MLP 256x256\n(every 0.5 s)",
        "#cfe0fb",
    )
    box(
        ax,
        0.68,
        0.30,
        0.14,
        0.20,
        "Move (17 options)\n-> PathDriver\n-> safety shield",
        "#fde3cf",
    )
    box(
        ax,
        0.86,
        0.30,
        0.12,
        0.20,
        "Reward:\nprogress, pass\nbonus, clean-\nracing penalties",
        "#f2d7f4",
    )
    for y in (0.81, 0.40):
        arrow(ax, (0.20, y), (0.25, y))
        arrow(ax, (0.41, y), (0.46, y))
        arrow(ax, (0.63, y), (0.68, y))
        arrow(ax, (0.82, y), (0.86, y))
    ax.text(
        0.5,
        0.67,
        "reward -> PPO update -> new policy -> environment update "
        "(training loop, once per lap)",
        ha="center",
        fontsize=8,
        color=GREY,
    )
    ax.text(
        0.5,
        0.25,
        "reward -> PPO update -> new policy -> environment update "
        "(training loop, every 0.5 s)",
        ha="center",
        fontsize=8,
        color=GREY,
    )
    # link
    ax.annotate(
        "",
        xy=(0.11, 0.72),
        xytext=(0.11, 0.50),
        arrowprops=dict(arrowstyle="->", lw=1.6, color=GREEN),
    )
    ax.text(
        0.12,
        0.585,
        "measured driver profile\n(pace gap, pass/defend curves)",
        fontsize=8,
        color=GREEN,
    )
    ax.annotate(
        "",
        xy=(0.545, 0.50),
        xytext=(0.545, 0.72),
        arrowprops=dict(arrowstyle="->", lw=1.6, color=BLUE, ls="--"),
    )
    ax.text(
        0.555,
        0.585,
        "push mode / attack-defend\n(designed interface)",
        fontsize=8,
        color=BLUE,
    )
    ax.text(
        0.5,
        0.17,
        "Data: TUM racetrack database "
        "(25 circuits: 17 train / 3 validation / 5 held-out), "
        "FastF1 + OpenF1 real passes (1,174 held-out real starts)",
        ha="center",
        fontsize=8,
    )
    save(fig, "fig2_pipeline.png")


# --- learning curves -----------------------------------------------------------------


def sac_curve() -> None:
    data: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    done: dict[str, dict[int, list[int]]] = defaultdict(lambda: defaultdict(list))
    for r in rows("racer_sac_eval.csv"):
        t = int(r["timesteps"])
        done[r["split"]][t].append(int(r["completed"]))
        if r["gap_pct"]:
            data[r["split"]][t].append(float(r["gap_pct"]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for split, c in (("train", GREY), ("validation", BLUE), ("held_out", ORANGE)):
        ts = sorted(done[split])
        a1.plot(
            [t / 1e6 for t in ts],
            [np.mean(done[split][t]) for t in ts],
            color=c,
            label=split,
        )
        ts2 = sorted(t for t in data[split] if data[split][t])
        a2.plot(
            [t / 1e6 for t in ts2],
            [np.mean(data[split][t]) for t in ts2],
            "o-",
            ms=3,
            color=c,
            label=split,
        )
    style(
        a1,
        "environment steps (millions of 0.1 s decisions)",
        "share of laps completed",
        "SAC driver: lap completion",
    )
    style(
        a2,
        "environment steps (millions)",
        "lap time gap to optimal line (%)",
        "SAC driver: pace on completed laps",
    )
    a2.axhline(2.3, color=GREEN, ls="--", lw=1)
    a2.text(0.05, 3.2, "scripted follower +2.3%", color=GREEN, fontsize=8)
    a1.legend(fontsize=8)
    save(fig, "fig3_sac_learning_curve.png")


def tactical_ablation() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    for name, label, c in (
        ("tactical_real_run1_eval.csv", "run 1: pass bonus 20, ent 0.01", GREY),
        (
            "tactical_real_run2_eval.csv",
            "run 2: pass bonus 100, ent 0.03 + shield fixes",
            BLUE,
        ),
    ):
        for ax, sc in ((a1, "real_attack"), (a2, "real_defend")):
            pts = [
                (int(r["timesteps"]), int(r["wins"]) / int(r["runs"]))
                for r in rows(name)
                if r["scenario"] == sc
            ]
            ax.plot(
                [p[0] / 1e6 for p in pts],
                [p[1] for p in pts],
                "o-",
                ms=3,
                color=c,
                label=label,
            )
    style(
        a1,
        "decisions (millions)",
        "success rate",
        "Attack from real F1 starts (43 validation)",
    )
    style(
        a2,
        "decisions (millions)",
        "success rate",
        "Defence from real F1 starts (37 validation)",
    )
    a1.legend(fontsize=8, loc="center right")
    a1.set_ylim(-0.03, 1.0)
    a2.set_ylim(-0.03, 1.0)
    save(fig, "fig4_tactical_reward_ablation.png")


def defend_curve() -> None:
    by: dict[int, dict[str, tuple[int, int]]] = defaultdict(dict)
    for r in rows("defend_sections_v2_eval.csv"):
        by[int(r["timesteps"])][r["scenario"]] = (int(r["wins"]), int(r["runs"]))
    ts = sorted(by)
    att = [
        sum(by[t][k][0] for k in by[t] if k.startswith("sec_"))
        / sum(by[t][k][1] for k in by[t] if k.startswith("sec_"))
        for t in ts
    ]
    dfn = [
        sum(by[t][k][0] for k in by[t] if k.startswith("def_"))
        / sum(by[t][k][1] for k in by[t] if k.startswith("def_"))
        for t in ts
    ]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(
        [t / 1e3 for t in ts],
        dfn,
        "o-",
        color=BLUE,
        label="defences held (120, worn tyres)",
    )
    ax.plot(
        [t / 1e3 for t in ts],
        att,
        "o-",
        color=ORANGE,
        label="clean section passes (270)",
    )
    style(
        ax,
        "decisions (thousands)",
        "success rate",
        "Driver, defend run 2: validation evaluation",
    )
    ax.set_ylim(0, 0.8)
    ax.legend(fontsize=8)
    save(fig, "fig5_defend_learning_curve.png")


def strategy_curves() -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for name, label, c in (
        ("strategy_fast_win_eval.csv", "fast car, win-only reward", GREEN),
        ("strategy_all_eval.csv", "all cases, 1% car", BLUE),
        ("strategy_all_dominant_eval.csv", "all cases, 3% car", ORANGE),
        ("strategy_recover_eval.csv", "incident every race", RED),
        ("master/eval.csv", "one agent, every case (final)", "#6b3fa0"),
    ):
        r = rows(name)
        ax.plot(
            [int(x["steps"]) / 1e6 for x in r],
            [float(x["wins"]) for x in r],
            color=c,
            label=label,
        )
        ax.axhline(float(r[0]["baseline_wins"]), color=c, ls=":", lw=1)
    style(
        ax,
        "environment steps (millions of laps)",
        "validation win rate (30-40 races)",
        "Strategy Agent learning curves (dotted = rule-based plan)",
    )
    ax.legend(fontsize=8, loc="lower right")
    save(fig, "fig6_strategy_learning_curves.png")


def seed_band() -> None:
    seed_dir = DATA / "seeds"
    files = (
        sorted(seed_dir.glob("strategy_all_seed*_eval.csv"))
        if seed_dir.exists()
        else []
    )
    if len(files) < 2:
        print("no seed runs yet; skipping fig7")
        return
    curves = []
    for f in files:
        with f.open(newline="") as fh:
            r = list(csv.DictReader(fh))
        curves.append({int(x["steps"]): float(x["wins"]) for x in r})
        base = float(r[0]["baseline_wins"])
    common = sorted(set.intersection(*(set(c) for c in curves)))
    m = np.array([[c[t] for t in common] for c in curves])
    mean, sd = m.mean(0), m.std(0, ddof=1)
    x = [t / 1e6 for t in common]
    fig, ax = plt.subplots(figsize=(7, 4))
    for i, row in enumerate(m):
        ax.plot(
            x,
            row,
            color=GREY,
            lw=0.8,
            alpha=0.6,
            label="single seed" if i == 0 else None,
        )
    ax.plot(x, mean, color=BLUE, lw=2, label=f"mean of {len(files)} seeds")
    ax.fill_between(x, mean - sd, mean + sd, color=BLUE, alpha=0.2, label="+/- 1 std")
    ax.axhline(base, color=ORANGE, ls="--", label="rule-based plan")
    style(
        ax,
        "environment steps (millions of laps)",
        "validation win rate (30-40 races)",
        "Strategy Agent (strategy_all, 1% car): stability across seeds",
    )
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig7_strategy_seeds.png")


def held_out_bars() -> None:
    """Final Strategy Agent vs the rule-based plan, per race case (held out)."""
    res: dict[str, dict[str, float]] = {}
    for f in sorted((DATA / "master" / "eval_guard").glob("*.json")):
        for key, v in json.loads(f.read_text()).items():
            case, who = key.split("|")
            res.setdefault(case, {})["base" if who == "baseline" else "agent"] = (
                100 * v["wins"]
            )
    order = ["clean", "spin", "puncture", "sc", "sc2", "vsc", "red", "rain"]
    names = [
        "clean",
        "spin",
        "puncture",
        "safety\ncar",
        "two\nSCs",
        "VSC",
        "red\nflag",
        "rain",
    ]
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(
        x - 0.2,
        [res[c]["base"] for c in order],
        0.4,
        color=ORANGE,
        label="rule-based plan",
    )
    ax.bar(
        x + 0.2,
        [res[c]["agent"] for c in order],
        0.4,
        color=BLUE,
        label="final Strategy Agent",
    )
    for i, c in enumerate(order):
        ax.text(
            i + 0.2,
            res[c]["agent"] + 1.5,
            f"{res[c]['agent']:.0f}",
            ha="center",
            fontsize=7,
        )
        ax.text(
            i - 0.2,
            res[c]["base"] + 1.5,
            f"{res[c]['base']:.0f}",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(x, names)
    ax.set_ylim(0, 110)
    style(
        ax,
        "race case (200 held-out races each)",
        "win rate (%)",
        "Held-out wins by race case: one agent for every case",
    )
    ax.legend(fontsize=8, loc="center left")
    save(fig, "fig8_strategy_held_out_cases.png")


def lap_table() -> None:
    held = json.loads((DATA / "racer_sac_held_out.json").read_text())
    names = [h["track"] for h in held]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(names))
    for i, (key, label, c) in enumerate(
        (
            ("scripted_s", "scripted follower", GREEN),
            ("baseline_s", "v1 cautious baseline", GREY),
            ("agent_s", "SAC (pure RL)", BLUE),
        )
    ):
        vals = [
            100 * (h[key] / h["optimal_line_s"] - 1) if h[key] else np.nan for h in held
        ]
        ax.bar(x + (i - 1) * 0.27, vals, 0.27, color=c, label=label)
    ax.set_xticks(x, names)
    style(
        ax,
        "held-out track",
        "gap to optimal-line lap (%)",
        "Lap pace on unseen tracks (no bar = did not finish)",
    )
    ax.legend(fontsize=8)
    save(fig, "fig9_held_out_lap_gaps.png")


# --- model figures (drawn from the simulator's own functions) ---------------------


def track_line() -> None:
    """Optimal line on a held-out circuit, coloured by limit speed."""
    from f1rl.config import EXAMPLE_CAR, RaceEnvParams
    from f1rl.envs.racetrack import load_tum_track
    from f1rl.models.raceline import optimal_line

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))
    for ax, name in zip(axes, ("Suzuka", "Spa"), strict=True):
        track = load_tum_track(name)
        env = RaceEnvParams()
        line = optimal_line(
            track, EXAMPLE_CAR, env.car_half_width_m + env.line_safety_m
        )
        nx, ny = -np.sin(track.heading), np.cos(track.heading)
        for side, sign in ((track.w_left, 1), (track.w_right, -1)):
            ax.plot(
                track.x + sign * side * nx,
                track.y + sign * side * ny,
                color=GREY,
                lw=0.6,
            )
        sc = ax.scatter(
            line.x, line.y, c=3.6 * line.speed, s=1.5, cmap="viridis", vmin=60
        )
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_title(
            f"{name}: limit lap {line.lap_time_s:.1f} s "
            f"(centre line {line.centre_lap_time_s:.1f} s)",
            fontsize=9,
        )
    cb = fig.colorbar(sc, ax=axes, shrink=0.7)
    cb.set_label("limit speed on the line (km/h)")
    fig.savefig(OUT / "fig10_track_line.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / "fig10_track_line.png")


def tyre_models() -> None:
    """Tyre grip against wear and time, and wet-weather grip."""
    from f1rl.config import StrategyParams, TireParams, WeatherParams
    from f1rl.models.tire import tire_grip
    from f1rl.models.weather import wet_grip

    tp, sp, wp = TireParams(), StrategyParams(), WeatherParams()
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))
    w = np.linspace(0, 1, 200)
    axes[0].plot(w, [tire_grip(tp, x) for x in w], color=BLUE)
    style(axes[0], "tyre wear (0 new, 1 gone)", "grip multiplier", "(a) Grip vs wear")
    t = np.linspace(0, 4500, 300)
    for name, c in (("soft", RED), ("medium", ORANGE), ("hard", GREY)):
        comp = getattr(sp, name)
        g = [comp.grip * tire_grip(tp, min(1.0, x / comp.life_s)) for x in t]
        axes[1].plot(t / 60, g, color=c, label=name)
    axes[1].legend(fontsize=8)
    style(axes[1], "minutes driven at push 0.5", "grip multiplier", "(b) Compounds")
    for name, c in (("soft", ORANGE), ("intermediate", GREEN), ("wet", BLUE)):
        axes[2].plot(
            w,
            [wet_grip(wp, name, x) for x in w],
            color=c,
            label="slick" if name == "soft" else name,
        )
    axes[2].legend(fontsize=8)
    style(axes[2], "track wetness", "grip multiplier", "(c) Rain")
    save(fig, "fig11_tyre_models.png")


def pass_curves() -> None:
    """Measured attack / defend curves of the driver profile."""
    from f1rl.driver_profile import load_profile

    prof = load_profile(ROOT / "configs" / "driver_profiles" / "defend_v2.json")
    adv = np.linspace(-2, 10, 300)
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(
        adv,
        [prof.attack.probability(a) for a in adv],
        color=BLUE,
        label=f"we attack ({prof.attack.battles} battles)",
    )
    ax.plot(
        adv,
        [prof.attack.probability(a + 3.4) for a in adv],
        color=BLUE,
        ls="--",
        label="we attack, scenario skill +3.4 s",
    )
    ax.plot(
        adv,
        [prof.defend.probability(a) for a in adv],
        color=ORANGE,
        label=f"rival attacks us ({prof.defend.battles} battles)",
    )
    ax.axhline(0.5, color=GREY, lw=0.6, ls=":")
    style(
        ax,
        "attacker pace advantage (s per lap)",
        "P(pass within the lap)",
        "Pass probabilities measured from the trained driver",
    )
    ax.legend(fontsize=8)
    save(fig, "fig12_pass_curves.png")


def race_trace() -> None:
    """Place per lap in one seeded held-out safety-car race at Suzuka: the earlier
    all-cases agent (seed 0) and the final agent, each against the rule-based plan
    on the same seed under its own configuration."""
    panels = (
        ("race_trace_suzuka_sc_seed0.json", "(a) earlier agent (strategy_all, seed 0)"),
        ("race_trace_suzuka_sc_master.json", "(b) final agent (strategy_master)"),
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, (name, title) in zip(axes, panels, strict=True):
        d = json.loads((DATA / name).read_text())
        hist = d["baseline"]["history"]
        sc = [h["lap"] for h in hist if h["neutral"] == "sc"]
        if sc:
            ax.axvspan(min(sc) - 0.5, max(sc) + 0.5, color=GREY, alpha=0.2, lw=0)
            ax.text(min(sc) - 0.3, 6.0, "safety\ncar", fontsize=8, color=GREY)
        for key, label, c in (
            ("baseline", "rule-based plan", ORANGE),
            ("agent", "agent", BLUE),
        ):
            run = d[key]
            h = run["history"]
            ax.plot(
                [x["lap"] for x in h],
                [x["position"] for x in h],
                color=c,
                lw=1.6,
                label=f"{label}: P{run['position']}, {run['stops']} stop(s)",
            )
            for x in h:
                if x["pit"]:
                    ax.plot(x["lap"], x["position"], "o", color=c, ms=6)
                    ax.annotate(
                        x["pit"][0].upper(),
                        (x["lap"], x["position"]),
                        textcoords="offset points",
                        xytext=(4, 4),
                        fontsize=8,
                        color=c,
                    )
        ax.set_yticks(range(1, 11))
        style(ax, "lap", "place", title)
        ax.legend(fontsize=8, loc="upper left" if ax is axes[0] else "lower right")
    axes[0].invert_yaxis()
    save(fig, "fig13_race_trace.png")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rl_loop()
    pipeline()
    sac_curve()
    tactical_ablation()
    defend_curve()
    strategy_curves()
    seed_band()
    held_out_bars()
    lap_table()
    track_line()
    tyre_models()
    pass_curves()
    race_trace()
