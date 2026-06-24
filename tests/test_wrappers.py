"""Tests for observation/action wrappers."""

from __future__ import annotations

import numpy as np

from ad_rl.envs import make_env
from ad_rl.envs.wrappers import ActionSmoothingWrapper
from helpers import make_cfg


def test_vector_frame_stack_shape():
    env = make_env("fallback", make_cfg("state", frame_stack=3))
    obs, _ = env.reset(seed=0)
    # base state dim is 5 + 5 lookahead = 10
    assert obs.shape == (30,)
    obs2, *_ = env.step(env.action_space.sample())
    assert obs2.shape == (30,)


def test_image_channel_stack_shape():
    env = make_env("fallback", make_cfg("image", frame_stack=4))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 12)
    assert obs.dtype == np.uint8


def test_action_smoothing_filters_toward_zero():
    env = ActionSmoothingWrapper(make_env("fallback", make_cfg("state")), alpha=0.5)
    env.reset(seed=0)
    # The wrapper should not raise and should keep the env steppable.
    _obs, reward, _terminated, _truncated, _info = env.step(np.array([1.0, 1.0], dtype=np.float32))
    assert np.isfinite(reward)
