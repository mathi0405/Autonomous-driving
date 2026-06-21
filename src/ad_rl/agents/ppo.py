"""PPO agent factory (on-policy, the project's primary algorithm).

PPO is the workhorse of driving RL: stable, easy to tune, and -- per recent work
(e.g. *CaRL*, 2025) -- it scales to hundreds of millions of CARLA samples and
tops route-completion benchmarks when paired with simple, well-shaped rewards.
"""

from __future__ import annotations

from typing import Any, Optional

from ad_rl.agents._common import resolve_policy
from ad_rl.utils.config import Config


def build_ppo(
    env: Any,
    cfg: Config,
    device: str = "auto",
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
):
    """Construct a Stable-Baselines3 ``PPO`` model from a :class:`Config`."""
    from stable_baselines3 import PPO

    policy, policy_kwargs = resolve_policy(cfg)
    if "log_std_init" in cfg.policy:
        policy_kwargs["log_std_init"] = float(cfg.policy["log_std_init"])

    hp = cfg.hyperparameters
    return PPO(
        policy,
        env,
        device=device,
        seed=cfg.seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        learning_rate=float(hp.get("learning_rate", 3e-4)),
        n_steps=int(hp.get("n_steps", 1024)),
        batch_size=int(hp.get("batch_size", 256)),
        n_epochs=int(hp.get("n_epochs", 10)),
        gamma=float(hp.get("gamma", 0.99)),
        gae_lambda=float(hp.get("gae_lambda", 0.95)),
        clip_range=float(hp.get("clip_range", 0.2)),
        ent_coef=float(hp.get("ent_coef", 0.0)),
        vf_coef=float(hp.get("vf_coef", 0.5)),
        max_grad_norm=float(hp.get("max_grad_norm", 0.5)),
        target_kl=hp.get("target_kl", None),
        policy_kwargs=policy_kwargs,
    )


__all__ = ["build_ppo"]
