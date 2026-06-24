"""Driving evaluation metrics and results-summary I/O.

The metric set mirrors what the CARLA leaderboard and AV literature care about:
route completion, collisions, lane-keeping accuracy, and ride comfort -- not just
episode return. Keeping them here (separate from the rollout loop) makes them easy
to unit-test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EpisodeRecord:
    """Per-episode rollout data used to compute aggregate metrics."""

    ret: float
    length: int
    success: bool
    collided: bool
    offroad: bool
    route_fraction: float
    speeds: list[float] = field(default_factory=list)
    lateral_errors: list[float] = field(default_factory=list)
    steers: list[float] = field(default_factory=list)


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if len(values) else 0.0


def aggregate(records: list[EpisodeRecord]) -> dict[str, float]:
    """Aggregate a list of episodes into a metrics dict."""
    n = len(records)
    if n == 0:
        return {}

    jerks: list[float] = []
    for r in records:
        if len(r.steers) > 1:
            jerks.extend(np.abs(np.diff(r.steers)).tolist())

    returns = [r.ret for r in records]
    return {
        "episodes": float(n),
        "success_rate": _mean([float(r.success) for r in records]),
        "collision_rate": _mean([float(r.collided) for r in records]),
        "offroad_rate": _mean([float(r.offroad) for r in records]),
        "mean_return": _mean(returns),
        "std_return": float(np.std(returns)),
        "mean_route_completion": _mean([r.route_fraction for r in records]),
        "mean_episode_length": _mean([float(r.length) for r in records]),
        "mean_speed_kmh": _mean(
            [float(np.mean(r.speeds)) * 3.6 if r.speeds else 0.0 for r in records]
        ),
        "mean_abs_lateral_error_m": _mean(
            [float(np.mean(np.abs(r.lateral_errors))) if r.lateral_errors else 0.0 for r in records]
        ),
        "mean_abs_jerk": _mean(jerks),
    }


# --------------------------------------------------------------------------- #
# Results summary I/O (consumed by the dashboard)
# --------------------------------------------------------------------------- #
def load_summary(path: Path | str) -> dict[str, Any]:
    """Load the results summary, returning a fresh skeleton if missing."""
    path = Path(path)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"schema": 1, "agents": {}}


def update_summary(
    path: Path | str,
    agent: str,
    metrics: dict[str, float],
    returns: list[float] | None = None,
    meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Merge one agent's results into ``summary.json`` and write it back."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    summary = load_summary(path)
    summary.setdefault("agents", {})
    summary["agents"][agent] = {
        "metrics": metrics,
        "returns": list(returns) if returns is not None else [],
        "meta": meta or {},
    }
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = ["EpisodeRecord", "aggregate", "load_summary", "update_summary"]
