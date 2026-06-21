"""Tests for the CNN feature extractors (require torch)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")


def test_cnn_extractors_forward_shapes():
    from gymnasium import spaces

    from ad_rl.perception.cnn_extractor import DrivingCNN, ImpalaCNN

    space = spaces.Box(low=0, high=255, shape=(12, 84, 84), dtype=np.uint8)
    batch = torch.zeros(2, 12, 84, 84, dtype=torch.float32)
    for cls in (DrivingCNN, ImpalaCNN):
        extractor = cls(space, features_dim=128)
        out = extractor(batch)
        assert out.shape == (2, 128)


def test_resolve_policy_selects_cnn_for_images():
    pytest.importorskip("stable_baselines3")
    from ad_rl.agents._common import resolve_policy
    from helpers import make_cfg

    name, kwargs = resolve_policy(make_cfg("image", frame_stack=4))
    assert name == "CnnPolicy"
    assert "features_extractor_class" in kwargs
    assert kwargs["features_extractor_kwargs"]["features_dim"] > 0


def test_resolve_policy_selects_mlp_for_state():
    pytest.importorskip("stable_baselines3")
    from ad_rl.agents._common import resolve_policy
    from helpers import make_cfg

    name, kwargs = resolve_policy(make_cfg("state"))
    assert name == "MlpPolicy"
    assert "features_extractor_class" not in kwargs
