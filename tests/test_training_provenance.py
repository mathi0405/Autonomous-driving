"""Run the real training CLI with tiny budgets to verify resolved artifacts."""

import json

import pytest

pytest.importorskip("stable_baselines3")

from ad_rl.training.train import parse_args, train
from ad_rl.utils.config import load_config


@pytest.mark.slow
def test_training_preserves_overrides_and_model_identity(tmp_path):
    args = parse_args(
        [
            "--config",
            "configs/ppo_validation.yaml",
            "--env",
            "fallback",
            "--seed",
            "71",
            "--total-timesteps",
            "256",
            "--n-envs",
            "1",
            "--eval-episodes",
            "1",
            "--smooth-actions",
            "--no-progress",
            "--no-tensorboard",
            "--run-name",
            "test",
            "--outdir",
            str(tmp_path),
        ]
    )
    run_dir = train(args)
    saved = load_config(run_dir / "config.yaml")
    assert saved.seed == 71 and saved.total_timesteps == 256 and saved.n_envs == 1
    assert saved.env.smooth_actions and saved.env.observation == "state"
    proof = json.loads((run_dir / "provenance.json").read_text())
    assert proof["actual_timesteps"] >= 256 and proof["model_sha256"]
    assert not proof["source_changed_during_training"]
    assert (run_dir / "summary.json").exists()
    with pytest.raises(FileExistsError):
        train(args)


def test_carla_training_rejects_shared_evaluation_port(tmp_path):
    args = parse_args(
        [
            "--config",
            "configs/ppo_validation.yaml",
            "--env",
            "carla",
            "--carla-eval-port",
            "2000",
            "--outdir",
            str(tmp_path),
        ]
    )
    with pytest.raises(ValueError, match="separate server"):
        train(args)
