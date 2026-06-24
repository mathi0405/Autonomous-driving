"""Perception: image preprocessing and CNN feature extractors for image policies.

``preprocessing`` is pure NumPy and always importable. The CNN extractors depend
on PyTorch / Stable-Baselines3 and are imported lazily via
:func:`get_features_extractor` so that the rest of the package stays usable
without a deep-learning stack installed.
"""

from ad_rl.perception.preprocessing import normalize_image, preprocess_observation, to_chw


def get_features_extractor(name: str = "nature_cnn"):
    """Lazily import and return a Stable-Baselines3 features-extractor class."""
    from ad_rl.perception.cnn_extractor import DrivingCNN, ImpalaCNN

    return {"nature_cnn": DrivingCNN, "impala": ImpalaCNN}.get(name.lower(), DrivingCNN)


__all__ = ["get_features_extractor", "normalize_image", "preprocess_observation", "to_chw"]
