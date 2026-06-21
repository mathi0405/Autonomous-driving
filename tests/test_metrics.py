"""Tests for evaluation metrics and results-summary I/O."""

from __future__ import annotations

from ad_rl.evaluation.metrics import EpisodeRecord, aggregate, load_summary, update_summary


def _record(success: bool = True, ret: float = 100.0) -> EpisodeRecord:
    return EpisodeRecord(
        ret=ret,
        length=100,
        success=success,
        collided=not success,
        offroad=False,
        route_fraction=1.0 if success else 0.3,
        speeds=[8.0] * 100,
        lateral_errors=[0.1] * 100,
        steers=[0.0, 0.1, -0.1] * 33 + [0.0],
    )


def test_aggregate_basic_metrics():
    metrics = aggregate([_record(True, 120.0), _record(False, -10.0)])
    assert metrics["episodes"] == 2
    assert metrics["success_rate"] == 0.5
    assert metrics["collision_rate"] == 0.5
    assert "mean_abs_jerk" in metrics
    assert "mean_speed_kmh" in metrics


def test_aggregate_empty_returns_empty():
    assert aggregate([]) == {}


def test_summary_roundtrip(tmp_path):
    path = tmp_path / "summary.json"
    update_summary(path, "PPO", {"success_rate": 0.9}, returns=[1.0, 2.0, 3.0])
    data = load_summary(path)
    assert "PPO" in data["agents"]
    assert data["agents"]["PPO"]["metrics"]["success_rate"] == 0.9

    update_summary(path, "SAC", {"success_rate": 0.8})
    data = load_summary(path)
    assert set(data["agents"]) == {"PPO", "SAC"}


def test_load_summary_missing_file(tmp_path):
    data = load_summary(tmp_path / "nope.json")
    assert data["agents"] == {}
