"""Custom Stable-Baselines3 training callbacks."""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardComponentsCallback(BaseCallback):
    """Log the mean of each shaped-reward component to TensorBoard.

    Reward shaping is the most common failure mode in driving RL; surfacing the
    individual terms (speed, lane, heading, ...) makes it obvious when, say, the
    lane penalty is dominating and the agent has learned to crawl.
    """

    def __init__(self, log_freq: int = 2000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self._buffer: dict[str, list[float]] = {}

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            components = info.get("reward_components")
            if components:
                for key, value in components.items():
                    self._buffer.setdefault(key, []).append(float(value))
        if self.n_calls % self.log_freq == 0 and self._buffer:
            for key, values in self._buffer.items():
                self.logger.record(f"reward/{key}", float(np.mean(values)))
            self._buffer.clear()
        return True


__all__ = ["RewardComponentsCallback"]
