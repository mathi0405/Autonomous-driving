"""CNN feature extractors for image-based driving policies (Stable-Baselines3).

Two architectures are provided:

* :class:`DrivingCNN`  -- the classic 3-layer "NatureCNN" (Mnih et al., 2015),
  a strong, cheap default for 84x84 inputs.
* :class:`ImpalaCNN`   -- a deeper residual conv-net (Espeholt et al., 2018) that
  tends to generalise better across visually diverse driving scenes, at extra
  compute cost.

Both consume channel-first image observations already scaled to ``[0, 1]`` by
Stable-Baselines3's image preprocessing (``normalize_images=True``), so they do
**not** divide by 255 again.
"""

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class DrivingCNN(BaseFeaturesExtractor):
    """NatureCNN-style feature extractor."""

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape, dtype=torch.float32)
            n_flatten = self.cnn(sample).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute and return the feature embedding from the input tensor."""
        return self.linear(self.cnn(observations))


class _ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv0 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv0(torch.relu(x))
        y = self.conv1(torch.relu(y))
        return x + y


class _ImpalaStage(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.res0 = _ResidualBlock(out_ch)
        self.res1 = _ResidualBlock(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.res1(self.res0(self.pool(self.conv(x))))


class ImpalaCNN(BaseFeaturesExtractor):
    """Deeper residual conv-net (IMPALA), better visual generalisation."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        features_dim: int = 512,
        channels: tuple = (16, 32, 32),
    ) -> None:
        super().__init__(observation_space, features_dim)
        in_ch = observation_space.shape[0]
        stages = []
        for out_ch in channels:
            stages.append(_ImpalaStage(in_ch, out_ch))
            in_ch = out_ch
        self.stages = nn.Sequential(*stages)
        with torch.no_grad():
            sample = torch.zeros(1, *observation_space.shape, dtype=torch.float32)
            n_flatten = self.stages(sample).reshape(1, -1).shape[1]
        self.linear = nn.Sequential(nn.ReLU(), nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute and return the feature embedding from the input tensor."""
        x = self.stages(observations)
        return self.linear(x.reshape(x.shape[0], -1))


__all__ = ["DrivingCNN", "ImpalaCNN"]
