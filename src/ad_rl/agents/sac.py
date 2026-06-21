"""SAC agent factory (off-policy, sample-efficient continuous control).

SAC's maximum-entropy objective and replay buffer make it markedly more
sample-efficient than PPO, which matters when each CARLA step is expensive. We
include it as a head-to-head comparison against PPO on the identical task.
"""

from __future__ import annotations

from typing import Any, Optional, Union

from ad_rl.agents._common import resolve_policy
from ad_rl.utils.config import Config


def build_sac(
    env: Any,
    cfg: Config,
    device: str = "auto",
    tensorboard_log: Optional[str] = None,
    verbose: int = 1,
):
    """Construct a Stable-Baselines3 ``SAC`` model from a :class:`Config`."""
    from stable_baselines3 import SAC

    policy, policy_kwargs = resolve_policy(cfg)

    hp = cfg.hyperparameters
    ent_coef: Union[str, float] = hp.get("ent_coef", "auto")
    if isinstance(ent_coef, str) and ent_coef.replace(".", "", 1).isdigit():
        ent_coef = float(ent_coef)

    return SAC(
        policy,
        env,
        device=device,
        seed=cfg.seed,
        verbose=verbose,
        tensorboard_log=tensorboard_log,
        learning_rate=float(hp.get("learning_rate", 3e-4)),
        buffer_size=int(hp.get("buffer_size", 300_000)),
        learning_starts=int(hp.get("learning_starts", 5_000)),
        batch_size=int(hp.get("batch_size", 256)),
        tau=float(hp.get("tau", 0.005)),
        gamma=float(hp.get("gamma", 0.99)),
        train_freq=int(hp.get("train_freq", 1)),
        gradient_steps=int(hp.get("gradient_steps", 1)),
        ent_coef=ent_coef,
        target_update_interval=int(hp.get("target_update_interval", 1)),
        use_sde=bool(hp.get("use_sde", False)),
        policy_kwargs=policy_kwargs,
    )


__all__ = ["build_sac"]
