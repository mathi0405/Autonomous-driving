"""Tests for the shaped reward function."""

from __future__ import annotations

from ad_rl.rewards.reward import DriveMeasurement, compute_reward
from ad_rl.utils.config import RewardConfig


def _meas(**kw) -> DriveMeasurement:
    base = dict(
        speed_ms=8.33,
        lateral_error_m=0.0,
        heading_error_rad=0.0,
        progress_m=1.0,
        steer=0.0,
        prev_steer=0.0,
    )
    base.update(kw)
    return DriveMeasurement(**base)


def test_collision_dominates_and_terminates_signal():
    cfg = RewardConfig()
    res = compute_reward(_meas(collided=True), cfg)
    assert res.total == -cfg.collision_penalty
    assert "collision" in res.components


def test_offroad_penalty():
    cfg = RewardConfig()
    res = compute_reward(_meas(offroad=True), cfg)
    assert res.total == -cfg.offroad_penalty


def test_speed_term_peaks_at_target():
    cfg = RewardConfig()
    at_target = compute_reward(_meas(speed_ms=cfg.target_speed_ms), cfg).components["speed"]
    too_slow = compute_reward(_meas(speed_ms=cfg.target_speed_ms * 0.4), cfg).components["speed"]
    too_fast = compute_reward(_meas(speed_ms=cfg.target_speed_ms * 1.6), cfg).components["speed"]
    assert at_target > too_slow
    assert at_target > too_fast


def test_lateral_and_heading_are_penalties():
    cfg = RewardConfig()
    comps = compute_reward(_meas(lateral_error_m=1.5, heading_error_rad=0.5), cfg).components
    assert comps["lane"] < 0
    assert comps["heading"] < 0


def test_jerk_penalises_action_change():
    cfg = RewardConfig()
    smooth = compute_reward(_meas(steer=0.2, prev_steer=0.2), cfg).components["jerk"]
    jerky = compute_reward(_meas(steer=0.9, prev_steer=-0.9), cfg).components["jerk"]
    assert jerky < smooth <= 0


def test_goal_bonus_present():
    cfg = RewardConfig()
    res = compute_reward(_meas(reached_goal=True), cfg)
    assert res.components.get("goal", 0.0) > 0
