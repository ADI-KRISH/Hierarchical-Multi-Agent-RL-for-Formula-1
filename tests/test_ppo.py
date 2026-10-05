from pathlib import Path
from typing import Any

import pytest
import yaml

from f1rl.agents.ppo import eval_score, load_config, train
from f1rl.envs.track import Segment, Track
from f1rl.report import training_view

CONFIG = Path(__file__).parents[1] / "configs" / "driver_ppo.yaml"


def test_shipped_config_loads() -> None:
    config = load_config(CONFIG)
    assert config.run.seed == 0
    assert config.sim_params(config.run.seed).seed == 0
    assert config.env_params().lookahead_m  # defaults fill in an empty env section


def _write(tmp_path: Path, raw: dict[str, Any]) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def _raw() -> dict[str, Any]:
    raw = yaml.safe_load(CONFIG.read_text())
    assert isinstance(raw, dict)
    return raw


def test_unknown_config_section_fails(tmp_path: Path) -> None:
    raw = _raw()
    raw["rewards"] = {"speed": 1.0}
    with pytest.raises(ValueError, match="rewards"):
        load_config(_write(tmp_path, raw))


def test_unknown_env_param_fails_at_load(tmp_path: Path) -> None:
    raw = _raw()
    raw["env"] = {"not_a_param": 1.0}
    with pytest.raises(TypeError):
        load_config(_write(tmp_path, raw))


def test_eval_score_prefers_finishing_then_speed() -> None:
    def laps(*times: float | None, reward: float = 0.0) -> list[dict[str, Any]]:
        return [
            {"completed": t is not None, "lap_time_s": t, "total_reward": reward}
            for t in times
        ]

    finishes_slowly = eval_score(laps(50.0, 50.0))
    finishes_fast = eval_score(laps(45.0, 45.0))
    crashes_once = eval_score(laps(40.0, None))
    assert finishes_fast > finishes_slowly > crashes_once
    # Before any lap finishes, reward is what tells models apart.
    assert eval_score(laps(None, reward=-0.3)) > eval_score(laps(None, reward=-0.8))


@pytest.mark.slow
def test_short_training_run_writes_its_logs(tmp_path: Path) -> None:
    raw = _raw()
    raw["run"].update(
        name="smoke",
        total_timesteps=512,
        n_envs=2,
        eval_every_steps=10_000,
        eval_episodes=1,
    )
    raw["sim"] = {"max_episode_steps": 300}
    raw["ppo"].update(n_steps=128, batch_size=64, n_epochs=1)
    config = load_config(_write(tmp_path, raw))
    short_loop = Track(segments=(Segment(200.0, 0.0), Segment(100.0, 0.01)))

    train(config, short_loop, tmp_path / "run")

    for name in (
        "config.yaml",
        "meta.json",
        "progress.csv",
        "episodes.csv",
        "eval.csv",
        "model_best.zip",
        "model_final.zip",
        "best_lap.json",
    ):
        assert (tmp_path / "run" / name).exists(), name
    assert "git_sha" in (tmp_path / "run" / "meta.json").read_text()

    # The report reads all of it back without touching the model.
    view = training_view(tmp_path / "run", short_loop.total_length_m)
    assert view["eval"]["timesteps"]
    assert view["best_lap"] is not None
