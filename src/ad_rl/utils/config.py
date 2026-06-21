"""Typed, hierarchical configuration loading.

Configuration lives in YAML files under ``configs/``. An algorithm config (e.g.
``ppo.yaml``) may declare ``defaults: env.yaml`` to inherit the shared
environment and reward settings, which are then deep-merged with any local
overrides. The result is parsed into frozen-ish dataclasses so the rest of the
codebase enjoys autocompletion and type checking instead of stringly-typed dict
access.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Type, TypeVar

import yaml

T = TypeVar("T")


# --------------------------------------------------------------------------- #
# Dataclasses for each configuration section
# --------------------------------------------------------------------------- #
@dataclass
class EnvConfig:
    """Settings shared by every environment implementation."""

    observation: str = "image"  # "image" | "state"
    image_size: Tuple[int, int] = (84, 84)
    frame_stack: int = 4
    action_repeat: int = 2
    max_episode_steps: int = 1000
    normalize_actions: bool = True


@dataclass
class RewardConfig:
    """Weights for the modular, shaped driving reward."""

    target_speed_kmh: float = 30.0
    w_speed: float = 1.0
    w_progress: float = 1.0
    w_lane: float = 0.5
    w_heading: float = 0.3
    w_steer: float = 0.1
    w_jerk: float = 0.1
    collision_penalty: float = 50.0
    offroad_penalty: float = 25.0

    @property
    def target_speed_ms(self) -> float:
        """Target speed in metres per second."""
        return self.target_speed_kmh / 3.6


@dataclass
class CarlaConfig:
    """Connection and scenario settings for the CARLA server."""

    host: str = "localhost"
    port: int = 2000
    timeout: float = 20.0
    town: str = "Town03"
    fixed_delta_seconds: float = 0.05
    synchronous: bool = True
    num_vehicles: int = 30
    num_walkers: int = 10
    weather: str = "ClearNoon"
    ego_vehicle: str = "vehicle.tesla.model3"
    route_length_m: float = 200.0


@dataclass
class FallbackConfig:
    """Settings for the simulator-free kinematic driving environment."""

    dt: float = 0.1
    wheelbase_m: float = 2.8
    max_speed_kmh: float = 50.0
    road_half_width_m: float = 2.0
    curviness: float = 0.6
    num_obstacles: int = 3

    @property
    def max_speed_ms(self) -> float:
        return self.max_speed_kmh / 3.6


@dataclass
class Config:
    """Top-level configuration bundling algorithm and environment settings."""

    algorithm: str = "ppo"
    seed: int = 1
    total_timesteps: int = 1_000_000
    n_envs: int = 8
    policy: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    carla: CarlaConfig = field(default_factory=CarlaConfig)
    fallback: FallbackConfig = field(default_factory=FallbackConfig)
    raw: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Loading helpers
# --------------------------------------------------------------------------- #
def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
    """Build a dataclass from a dict, ignoring unknown keys and fixing tuples."""
    if not is_dataclass(cls):  # pragma: no cover - defensive
        raise TypeError(f"{cls!r} is not a dataclass")
    field_names = {f.name for f in fields(cls)}
    kwargs: Dict[str, Any] = {}
    for key, value in data.items():
        if key not in field_names:
            continue
        # YAML lists become Python lists; coerce known tuple fields back.
        if key == "image_size" and isinstance(value, list):
            value = tuple(value)
        kwargs[key] = value
    return cls(**kwargs)  # type: ignore[call-arg]


def load_yaml(path: Path | str) -> Dict[str, Any]:
    """Load a single YAML file into a plain dict."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):  # pragma: no cover - defensive
        raise ValueError(f"Top-level YAML in {path} must be a mapping, got {type(data)}")
    return data


def load_config(path: Path | str) -> Config:
    """Load a config file, resolving a ``defaults:`` parent and merging overrides.

    Parameters
    ----------
    path:
        Path to an algorithm config (e.g. ``configs/ppo.yaml``).

    Returns
    -------
    Config
        A fully-populated, typed configuration object.
    """
    path = Path(path)
    raw = load_yaml(path)

    # Resolve inheritance: defaults are loaded first, then overridden locally.
    defaults_name = raw.pop("defaults", None)
    if defaults_name:
        base = load_yaml(path.parent / defaults_name)
        raw = _deep_merge(base, raw)

    env_section = raw.get("env", {})
    reward_section = raw.get("reward", {})
    carla_section = raw.get("carla", {})
    fallback_section = raw.get("fallback", {})

    return Config(
        algorithm=str(raw.get("algorithm", "ppo")).lower(),
        seed=int(raw.get("seed", 1)),
        total_timesteps=int(raw.get("total_timesteps", 1_000_000)),
        n_envs=int(raw.get("n_envs", 8)),
        policy=dict(raw.get("policy", {})),
        hyperparameters=dict(raw.get("hyperparameters", {})),
        logging=dict(raw.get("logging", {})),
        env=_from_dict(EnvConfig, env_section),
        reward=_from_dict(RewardConfig, reward_section),
        carla=_from_dict(CarlaConfig, carla_section),
        fallback=_from_dict(FallbackConfig, fallback_section),
        raw=raw,
    )


__all__ = [
    "Config",
    "EnvConfig",
    "RewardConfig",
    "CarlaConfig",
    "FallbackConfig",
    "load_config",
    "load_yaml",
]
