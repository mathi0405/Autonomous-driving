"""Versioned, validated scenarios with explicit sensing and actuator faults."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class Scenario:
    """A complete CPU simulation case, independent of training configuration."""

    id: str
    seed: int
    regime: str = "nominal"
    curviness: float = 0.6
    road_half_width_m: float = 2.0
    num_obstacles: int = 3
    target_speed_kmh: float = 30.0
    max_steps: int = 600
    lateral_bias_m: float = 0.0
    sensor_noise_std_m: float = 0.0
    observation_delay_steps: int = 0
    steering_bias: float = 0.0
    initial_offset_m: float = 0.0
    obstacle_offset_m: float | None = None

    def __post_init__(self) -> None:
        """Validate the physical parameters and finite evaluation budget."""
        if not self.id or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in self.id):
            raise ValueError("Scenario id must contain lowercase letters, digits, '-' or '_'.")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("Scenario seed must be a nonnegative integer.")
        for key in ("max_steps", "num_obstacles", "observation_delay_steps"):
            value = getattr(self, key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{key} must be a nonnegative integer.")
        for key, value in asdict(self).items():
            if isinstance(value, float) and not math.isfinite(value):
                raise ValueError(f"{key} must be finite.")
        if not 1 <= self.max_steps <= 5000 or not 0 <= self.num_obstacles <= 100:
            raise ValueError("Invalid step or obstacle budget.")
        if not 0 <= self.observation_delay_steps <= 100:
            raise ValueError("Observation delay exceeds 100 steps.")
        if not 0 <= self.curviness <= 5 or not 0.5 <= self.road_half_width_m <= 5:
            raise ValueError("Invalid road geometry.")
        if not 5 <= self.target_speed_kmh <= 50 or self.sensor_noise_std_m < 0:
            raise ValueError("Invalid speed or noise scale.")
        if abs(self.steering_bias) > 1:
            raise ValueError("Steering bias must be normalized to [-1, 1].")


def canonical_hash(value: object) -> str:
    """Hash JSON with stable ordering and no non-finite numbers."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


def load_suite(path: str | Path) -> tuple[dict, list[Scenario]]:
    """Load a frozen suite, rejecting unknown fields and duplicate scenario IDs."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != 1 or not data.get("scenarios"):
        raise ValueError("Expected a nonempty scenario suite with schema=1.")
    valid = {f.name for f in fields(Scenario)}
    scenarios = []
    for raw in data["scenarios"]:
        if set(raw) - valid:
            raise ValueError(f"Unknown scenario fields: {set(raw) - valid}")
        scenarios.append(Scenario(**raw))
    if len({s.id for s in scenarios}) != len(scenarios):
        raise ValueError("Duplicate scenario IDs.")
    return data, scenarios


def generate_suite(seed: int, per_regime: int, split: str) -> dict:
    """Generate fixed nominal, geometry, sensing and actuator test distributions."""
    if per_regime < 1 or split not in {"development", "held-out"}:
        raise ValueError("Positive budget and an explicit split are required.")
    rng = np.random.default_rng(seed)
    cases = []
    for regime in ("nominal", "geometry", "sensing", "actuator"):
        for i in range(per_regime):
            settings: dict[str, Any] = {}
            if regime == "geometry":
                settings = {
                    "curviness": float(rng.uniform(1.2, 2.4)),
                    "road_half_width_m": float(rng.uniform(1.25, 1.8)),
                    "initial_offset_m": float(rng.uniform(-0.4, 0.4)),
                }
            elif regime == "sensing":
                settings = {
                    "lateral_bias_m": float(rng.uniform(-0.25, 0.25)),
                    "sensor_noise_std_m": float(rng.uniform(0.01, 0.05)),
                    "observation_delay_steps": int(rng.integers(1, 4)),
                }
            elif regime == "actuator":
                settings = {
                    "steering_bias": float(rng.uniform(-0.08, 0.08)),
                    "curviness": float(rng.uniform(0.6, 1.4)),
                }
            case = Scenario(
                id=f"{regime}-{i:03d}", seed=int(rng.integers(0, 2**31)), regime=regime, **settings
            )
            cases.append(asdict(case))
    return {
        "schema": 1,
        "name": f"driving-{split}-v1",
        "split": split,
        "generator_seed": seed,
        "scenarios": cases,
    }
