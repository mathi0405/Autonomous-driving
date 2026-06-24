"""PPO agent factory — on-policy, the project's primary training algorithm.

Proximal Policy Optimization (Schulman et al., 2017) is the workhorse of
applied driving RL: stable across a wide range of hyperparameters, easy to
diagnose, and well-supported by existing libraries. Per recent large-scale work
(e.g., *CaRL*, 2025), PPO scales to hundreds of millions of CARLA environment
steps and achieves state-of-the-art route-completion rates when paired with
well-calibrated, component-logged reward shaping.

Algorithmic objective
---------------------
PPO maximises the clipped surrogate objective::

    L^CLIP(θ) = E_t [ min(r_t(θ) Â_t,
                          clip(r_t(θ), 1−ε, 1+ε) Â_t) ]

where ``r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t)`` is the importance
sampling ratio and ``Â_t`` is the Generalised Advantage Estimate (GAE).
The clip parameter ``ε`` (``clip_range``) constrains the policy update magnitude,
ensuring monotone improvement in expectation without line-search overhead.

The full loss includes a value-function term and an entropy bonus::

    L(θ) = L^CLIP(θ) − c_1 · L^VF(θ) + c_2 · H[π_θ(· | s_t)]

where ``c_1 = vf_coef`` and ``c_2 = ent_coef``.

References
----------
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017).
    Proximal Policy Optimization Algorithms. arXiv:1707.06347.
Schulman, J., Moritz, P., Levine, S., Jordan, M., & Abbeel, P. (2016).
    High-Dimensional Continuous Control Using Generalized Advantage Estimation.
    ICLR 2016. arXiv:1506.02438.

See Also
--------
docs/hyperparameter_guide.md : PPO hyperparameter sensitivity analysis.
configs/ppo.yaml : Default hyperparameter configuration.
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
    """Construct and return a Stable-Baselines3 ``PPO`` model from a ``Config``.

    All hyperparameters are drawn from ``cfg.hyperparameters``; CLI overrides
    applied via :func:`~ad_rl.training.train.apply_overrides` take precedence
    over the YAML defaults.

    Parameters
    ----------
    env : gymnasium.Env or VecEnv
        The (possibly vectorised) training environment.
    cfg : Config
        Fully resolved configuration object. Hyperparameters are read from
        ``cfg.hyperparameters``, the policy architecture from ``cfg.policy``.
    device : str
        PyTorch device string. ``'auto'`` selects CUDA if available, else CPU.
    tensorboard_log : str, optional
        Directory for TensorBoard event files. ``None`` disables logging.
    verbose : int
        Stable-Baselines3 verbosity level (0 = silent, 1 = info, 2 = debug).

    Returns
    -------
    stable_baselines3.PPO
        An initialised but untrained PPO model ready for ``.learn()``.
    """
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
        target_kl=hp.get("target_kl", 0.03),
        policy_kwargs=policy_kwargs,
    )


__all__ = ["build_ppo"]
