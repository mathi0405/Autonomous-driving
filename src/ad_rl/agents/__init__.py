"""Agent factories for PPO and SAC (Stable-Baselines3).

Imports are lazy so that simply importing :mod:`ad_rl` does not require torch.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ad_rl.utils.config import Config


def build_agent(algorithm: str, env: Any, cfg: Config, **kwargs: Any):
    """Return a configured SB3 model for ``algorithm`` in {"ppo", "sac"}."""
    algorithm = algorithm.lower()
    if algorithm == "ppo":
        from ad_rl.agents.ppo import build_ppo

        return build_ppo(env, cfg, **kwargs)
    if algorithm == "sac":
        from ad_rl.agents.sac import build_sac

        return build_sac(env, cfg, **kwargs)
    raise ValueError(f"Unknown algorithm '{algorithm}'. Expected 'ppo' or 'sac'.")


def load_agent(algorithm: str, path: str, **kwargs: Any):
    """Load a trained SB3 model from ``path`` for the given algorithm."""
    algorithm = algorithm.lower()
    if algorithm == "ppo":
        from stable_baselines3 import PPO

        return PPO.load(path, **kwargs)
    if algorithm == "sac":
        from stable_baselines3 import SAC

        return SAC.load(path, **kwargs)
    raise ValueError(f"Unknown algorithm '{algorithm}'. Expected 'ppo' or 'sac'.")


__all__ = ["build_agent", "load_agent"]
