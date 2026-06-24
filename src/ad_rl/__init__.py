"""ad_rl: Autonomous driving with deep reinforcement learning in CARLA.

A clean, reproducible research/engineering codebase that trains PPO and SAC
agents to drive, either in the CARLA simulator or in a fast, dependency-free
kinematic fallback environment that shares the exact same Gymnasium interface.

Public API is intentionally small. Heavy dependencies (torch / stable-baselines3)
are imported lazily inside the ``agents`` and ``training`` subpackages so that the
environments, rewards and controllers can be used without them.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ad_rl.envs import make_env

__all__ = ["__version__", "make_env"]
