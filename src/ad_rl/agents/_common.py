"""Shared policy/architecture resolution for the SB3 agents."""

from __future__ import annotations

from typing import Any

from ad_rl.utils.config import Config


def resolve_policy(cfg: Config) -> tuple[str, dict[str, Any]]:
    """Map our config onto an SB3 ``(policy_name, policy_kwargs)`` pair.

    * Image observations  -> ``CnnPolicy`` with one of our custom CNN extractors.
    * State observations   -> ``MlpPolicy``.
    """
    import torch.nn as nn

    is_image = cfg.env.observation == "image"
    requested = str(cfg.policy.get("type", "auto")).lower()
    if requested in ("auto", ""):
        policy = "CnnPolicy" if is_image else "MlpPolicy"
    else:
        policy = cfg.policy["type"]

    activation_map = {"relu": nn.ReLU, "tanh": nn.Tanh, "elu": nn.ELU, "gelu": nn.GELU}
    activation = activation_map.get(str(cfg.policy.get("activation", "relu")).lower(), nn.ReLU)
    net_arch: list[int] = list(cfg.policy.get("net_arch", [256, 256]))

    policy_kwargs: dict[str, Any] = {"net_arch": net_arch, "activation_fn": activation}

    if is_image:
        from ad_rl.perception import get_features_extractor

        extractor = get_features_extractor(str(cfg.policy.get("features_extractor", "nature_cnn")))
        policy_kwargs["features_extractor_class"] = extractor
        policy_kwargs["features_extractor_kwargs"] = {
            "features_dim": int(cfg.policy.get("features_dim", 512))
        }
    return policy, policy_kwargs


__all__ = ["resolve_policy"]
