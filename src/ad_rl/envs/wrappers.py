"""Optional Gymnasium observation/action wrappers.

Frame stacking is implemented at the *env* level (not via SB3's ``VecFrameStack``)
so that training and evaluation construct byte-identical observation pipelines --
a policy is then evaluated exactly as it was trained.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ActionSmoothingWrapper(gym.ActionWrapper):
    """Exponential low-pass filter on the action for smoother, comfier control.

    ``a_t = alpha * a_raw + (1 - alpha) * a_{t-1}``. Higher ``alpha`` is more
    responsive; lower is smoother.
    """

    def __init__(self, env: gym.Env, alpha: float = 0.6) -> None:
        super().__init__(env)
        self.alpha = float(alpha)
        self._prev = np.zeros(self.action_space.shape, dtype=np.float32)

    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        """Reset the environment and clear the action/frame buffer."""
        self._prev = np.zeros(self.action_space.shape, dtype=np.float32)
        return self.env.reset(**kwargs)

    def action(self, action: np.ndarray) -> np.ndarray:
        """Apply the exponential smoothing filter to the raw action."""
        action = np.asarray(action, dtype=np.float32)
        smoothed = self.alpha * action + (1.0 - self.alpha) * self._prev
        self._prev = smoothed
        return smoothed


class ChannelStackObservation(gym.Wrapper):
    """Stack the last ``n`` image frames along the channel axis: (H, W, C) -> (H, W, C*n).

    Matches ``VecFrameStack`` semantics for CNN policies, giving the network short-
    horizon motion cues (speed, yaw rate) it cannot read from a single frame.
    """

    def __init__(self, env: gym.Env, n: int) -> None:
        super().__init__(env)
        self.n = int(n)
        self.frames: deque = deque(maxlen=self.n)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(h, w, c * self.n), dtype=env.observation_space.dtype
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and clear the action/frame buffer."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):
            self.frames.append(obs)
        return self._stacked(), info

    def step(self, action: np.ndarray):
        """Step the environment and return the (next stacked obs, reward, ...) tuple."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=2)


class VectorStackObservation(gym.Wrapper):
    """Stack the last ``n`` state vectors: (D,) -> (D*n,)."""

    def __init__(self, env: gym.Env, n: int) -> None:
        super().__init__(env)
        self.n = int(n)
        self.frames: deque = deque(maxlen=self.n)
        (d,) = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(d * self.n,), dtype=np.float32
        )

    def reset(self, **kwargs: Any) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment and clear the action/frame buffer."""
        obs, info = self.env.reset(**kwargs)
        for _ in range(self.n):
            self.frames.append(obs)
        return self._stacked(), info

    def step(self, action: np.ndarray):
        """Step the environment and return the (next stacked obs, reward, ...) tuple."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(obs)
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        return np.concatenate(list(self.frames), axis=0).astype(np.float32)


def maybe_frame_stack(env: gym.Env, observation: str, frame_stack: int) -> gym.Env:
    """Apply the appropriate frame-stacking wrapper when ``frame_stack > 1``."""
    if frame_stack and frame_stack > 1:
        if observation == "image":
            return ChannelStackObservation(env, frame_stack)
        return VectorStackObservation(env, frame_stack)
    return env


__all__ = [
    "ActionSmoothingWrapper",
    "ChannelStackObservation",
    "VectorStackObservation",
    "maybe_frame_stack",
]
