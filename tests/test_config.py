"""Tests for hierarchical config loading."""

from __future__ import annotations

from pathlib import Path

from ad_rl.utils.config import Config, EnvConfig, RewardConfig, _from_dict, load_config

ROOT = Path(__file__).resolve().parents[1]


def test_load_ppo_config_types():
    cfg = load_config(ROOT / "configs" / "ppo.yaml")
    assert isinstance(cfg, Config)
    assert cfg.algorithm == "ppo"
    assert isinstance(cfg.env, EnvConfig)
    assert isinstance(cfg.reward, RewardConfig)
    assert cfg.env.image_size == (84, 84)
    assert cfg.env.frame_stack >= 1


def test_defaults_merge_pulls_in_env_section():
    # sac.yaml declares `defaults: env.yaml`; reward weights must be inherited.
    cfg = load_config(ROOT / "configs" / "sac.yaml")
    assert cfg.algorithm == "sac"
    assert cfg.reward.target_speed_kmh > 0
    assert cfg.fallback.wheelbase_m > 0


def test_target_speed_unit_conversion():
    cfg = load_config(ROOT / "configs" / "ppo.yaml")
    assert abs(cfg.reward.target_speed_ms - cfg.reward.target_speed_kmh / 3.6) < 1e-9


def test_from_dict_ignores_unknown_keys():
    env = _from_dict(EnvConfig, {"observation": "state", "totally_unknown": 99})
    assert env.observation == "state"


def test_image_size_list_becomes_tuple():
    env = _from_dict(EnvConfig, {"image_size": [64, 48]})
    assert env.image_size == (64, 48)
