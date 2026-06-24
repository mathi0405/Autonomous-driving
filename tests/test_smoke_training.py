"""End-to-end smoke training tests (require stable-baselines3 / torch).

These are the CI gate that proves the *learning* pipeline -- env -> vec env ->
agent -> learn -> predict -> save/load -- actually runs. They are tiny on
purpose (a few hundred steps on CPU).
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("stable_baselines3")

from ad_rl.agents import build_agent, load_agent
from ad_rl.training.utils import make_vector_env
from helpers import make_cfg


@pytest.mark.slow
def test_ppo_smoke_train_predict_save(tmp_path):
    cfg = make_cfg("state", max_steps=150)
    cfg.total_timesteps = 512
    cfg.hyperparameters["n_steps"] = 64
    cfg.hyperparameters["batch_size"] = 64
    venv = make_vector_env("fallback", cfg, n_envs=2, seed=0)
    model = build_agent("ppo", venv, cfg, device="cpu", tensorboard_log=None, verbose=0)
    model.learn(total_timesteps=512, progress_bar=False)

    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert action.shape == (2, 2)

    path = tmp_path / "ppo_model"
    model.save(path)
    reloaded = load_agent("ppo", str(path))
    action2, _ = reloaded.predict(obs, deterministic=True)
    assert np.asarray(action2).shape == (2, 2)
    venv.close()


@pytest.mark.slow
def test_sac_smoke_train(tmp_path):
    cfg = make_cfg("state", max_steps=120)
    cfg.total_timesteps = 300
    cfg.hyperparameters["learning_starts"] = 50
    cfg.hyperparameters["buffer_size"] = 2000
    venv = make_vector_env("fallback", cfg, n_envs=1, seed=0)
    model = build_agent("sac", venv, cfg, device="cpu", tensorboard_log=None, verbose=0)
    model.learn(total_timesteps=300, progress_bar=False)
    obs = venv.reset()
    action, _ = model.predict(obs, deterministic=True)
    assert np.asarray(action).shape == (1, 2)
    venv.close()
