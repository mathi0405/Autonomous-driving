"""Tests for the kinematic fallback environment."""

from __future__ import annotations

import numpy as np
import pytest

from ad_rl.envs import make_env
from helpers import make_cfg


def test_reset_and_step_state_obs():
    env = make_env("fallback", make_cfg("state"))
    obs, info = env.reset(seed=0)
    assert obs.shape == env.observation_space.shape
    assert obs.dtype == np.float32
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    assert np.isfinite(reward)
    assert {"lateral_error_m", "heading_error_rad", "speed_ms"}.issubset(info)


def test_image_observation_shape_and_dtype():
    env = make_env("fallback", make_cfg("image", frame_stack=1))
    obs, _ = env.reset(seed=0)
    assert obs.shape == (84, 84, 3)
    assert obs.dtype == np.uint8


def test_action_space_bounds():
    env = make_env("fallback", make_cfg("state"))
    assert env.action_space.shape == (2,)
    assert np.all(env.action_space.low == -1.0)
    assert np.all(env.action_space.high == 1.0)


def test_seeding_is_deterministic():
    cfg = make_cfg("state")
    e1, e2 = make_env("fallback", cfg), make_env("fallback", cfg)
    o1, _ = e1.reset(seed=123)
    o2, _ = e2.reset(seed=123)
    assert np.allclose(o1, o2)
    a = np.array([0.1, 0.5], dtype=np.float32)
    s1 = e1.step(a)
    s2 = e2.step(a)
    assert np.allclose(s1[0], s2[0])
    assert s1[1] == s2[1]


def test_hard_steer_terminates_offroad():
    env = make_env("fallback", make_cfg("state", max_steps=300))
    env.reset(seed=1)
    done = False
    for _ in range(300):
        _, _, terminated, truncated, info = env.step(np.array([1.0, 1.0], dtype=np.float32))
        if terminated or truncated:
            done = True
            break
    assert done


def test_env_checker_passes():
    pytest.importorskip("gymnasium.utils.env_checker")
    from gymnasium.utils.env_checker import check_env

    env = make_env("fallback", make_cfg("state"))
    check_env(env.unwrapped, skip_render_check=True)
