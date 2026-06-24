"""SAC agent factory — off-policy, maximum-entropy continuous control.

Soft Actor-Critic (Haarnoja et al., 2018) augments the standard RL objective
with a policy entropy term::

    π* = argmax_π  E[ Σ_t  r(s_t, a_t) + alpha H(π(· | s_t)) ]

where ``alpha`` is the temperature parameter that controls the entropy-reward
trade-off. SAC uses automatic entropy tuning by default (``ent_coef='auto'``),
which adjusts ``alpha`` to maintain a target entropy level of
``-dim(A)`` nats throughout training.

The critic is implemented as a clipped double-Q function to reduce
overestimation bias::

    y = r + gamma (1 - d) [ min_{i=1,2} Q_{θ̄_i}(s', ã') - alpha log π_φ(ã' | s') ]

where ``ã' ~ π_φ(· | s')`` is a fresh action sample and ``θ̄`` denotes the
target critic parameters updated via exponential moving average (Polyak
averaging with coefficient ``τ``).

When to prefer SAC over PPO
---------------------------
- Expensive simulation steps (CARLA): SAC's replay buffer enables re-use of
  historical experience, requiring fewer environment interactions.
- Fine-grained continuous control: The maximum-entropy objective encourages
  multimodal exploration, which is beneficial for learning smooth steering.
- When ``use_sde=True``: State-Dependent Exploration (Raffin et al., 2020)
  replaces the diagonal Gaussian policy with a learnable perturbation, often
  producing significantly smoother trajectories.

References
----------
Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018).
    Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning
    with a Stochastic Actor. ICML 2018. arXiv:1801.01290.
Haarnoja, T., Zhou, A., Hartikainen, K., et al. (2018).
    Soft Actor-Critic Algorithms and Applications. arXiv:1812.05905.
Raffin, A., Kober, J., & Stulp, F. (2020).
    Smooth Exploration for Robotic Reinforcement Learning.
    CoRL 2021. arXiv:2005.05719.

See Also
--------
docs/hyperparameter_guide.md : SAC hyperparameter sensitivity analysis.
configs/sac.yaml : Default hyperparameter configuration.
"""

from __future__ import annotations

from typing import Any

from ad_rl.agents._common import resolve_policy
from ad_rl.utils.config import Config


def build_sac(
    env: Any,
    cfg: Config,
    device: str = "auto",
    tensorboard_log: str | None = None,
    verbose: int = 1,
):
    """Construct and return a Stable-Baselines3 ``SAC`` model from a ``Config``.

    All hyperparameters are drawn from ``cfg.hyperparameters``; CLI overrides
    applied via :func:`~ad_rl.training.train.apply_overrides` take precedence
    over YAML defaults.

    Parameters
    ----------
    env : gymnasium.Env or VecEnv
        The (possibly vectorised) training environment. Note that SAC is
        inherently a single-environment algorithm; ``n_envs > 1`` is not
        supported by Stable-Baselines3's SAC implementation.
    cfg : Config
        Fully resolved configuration object.
    device : str
        PyTorch device string. ``'auto'`` selects CUDA if available, else CPU.
    tensorboard_log : str, optional
        Directory for TensorBoard event files. ``None`` disables logging.
    verbose : int
        Stable-Baselines3 verbosity level (0 = silent, 1 = info, 2 = debug).

    Returns
    -------
    stable_baselines3.SAC
        An initialised but untrained SAC model ready for ``.learn()``.
    """
    from stable_baselines3 import SAC

    policy, policy_kwargs = resolve_policy(cfg)

    hp = cfg.hyperparameters
    ent_coef: str | float = hp.get("ent_coef", "auto")
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
