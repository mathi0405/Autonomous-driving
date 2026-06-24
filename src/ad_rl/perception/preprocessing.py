"""Pure-NumPy image preprocessing helpers (no torch dependency)."""

from __future__ import annotations

import numpy as np


def to_chw(image: np.ndarray) -> np.ndarray:
    """Convert an ``(H, W, C)`` image to channel-first ``(C, H, W)``."""
    if image.ndim != 3:
        raise ValueError(f"Expected an (H, W, C) image, got shape {image.shape}")
    return np.ascontiguousarray(np.transpose(image, (2, 0, 1)))


def normalize_image(image: np.ndarray) -> np.ndarray:
    """Scale a uint8 image to float32 in ``[0, 1]``."""
    return image.astype(np.float32) / 255.0


def preprocess_observation(
    image: np.ndarray, channel_first: bool = True, to_float: bool = True
) -> np.ndarray:
    """Standardise a raw camera frame for a neural network.

    Parameters
    ----------
    image:
        Raw ``(H, W, C)`` uint8 frame.
    channel_first:
        Return ``(C, H, W)`` instead of ``(H, W, C)``.
    to_float:
        Scale to float32 ``[0, 1]`` (otherwise keep uint8).
    """
    out = normalize_image(image) if to_float else image
    if channel_first:
        out = to_chw(out)
    return out


__all__ = ["normalize_image", "preprocess_observation", "to_chw"]
