"""Shared test helpers (importable as a plain module; not collected as tests)."""

from __future__ import annotations

from pathlib import Path

from ad_rl.utils.config import Config, load_config

ROOT = Path(__file__).resolve().parents[1]


def make_cfg(observation: str = "state", frame_stack: int = 1, max_steps: int = 300) -> Config:
    """Load the PPO config and apply common test overrides for fast envs."""
    cfg = load_config(ROOT / "configs" / "ppo.yaml")
    cfg.env.observation = observation
    cfg.env.frame_stack = frame_stack
    cfg.env.max_episode_steps = max_steps
    return cfg


def effective_dt(cfg: Config) -> float:
    """Effective per-step dt for the fallback env (used to tune the baseline)."""
    return cfg.fallback.dt * cfg.env.action_repeat
