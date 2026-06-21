"""Tests for the evaluation rollout loop."""

from __future__ import annotations

from ad_rl.envs import make_env
from ad_rl.evaluation.evaluate import policy_from_baseline, run_episodes
from ad_rl.evaluation.metrics import aggregate
from helpers import effective_dt, make_cfg


def test_run_episodes_with_pid_baseline():
    cfg = make_cfg("state", max_steps=600)
    env = make_env("fallback", cfg)
    policy = policy_from_baseline("pid", cfg.reward.target_speed_ms, effective_dt(cfg))
    records = run_episodes(env, policy, n_episodes=5, seed=4242)
    assert len(records) == 5
    metrics = aggregate(records)
    assert metrics["success_rate"] >= 0.8
    assert metrics["mean_route_completion"] > 0.8


def test_run_episodes_records_have_trajectories():
    cfg = make_cfg("state", max_steps=200)
    env = make_env("fallback", cfg)
    policy = policy_from_baseline("pid", cfg.reward.target_speed_ms, effective_dt(cfg))
    records = run_episodes(env, policy, n_episodes=2, seed=1)
    for rec in records:
        assert len(rec.speeds) == rec.length
        assert len(rec.steers) == rec.length
