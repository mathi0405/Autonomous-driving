"""Immutable simulation evidence bundles with source and dependency provenance."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from ad_rl.validation.environment import ScenarioEnvironment
from ad_rl.validation.policies import make_policy
from ad_rl.validation.scenarios import Scenario, canonical_hash


def write_json(path: Path, value: object) -> None:
    """Write strict JSON; NaN and Infinity are never valid evidence."""
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def source_fingerprint() -> str:
    """Hash package source with normalized line endings, including local changes."""
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes().replace(b"\r\n", b"\n"))
    return digest.hexdigest()


def provenance() -> dict:
    """Record execution platform, package versions, git revision and dirty state."""
    versions: dict[str, str | None] = {}
    for name in ("numpy", "gymnasium", "stable-baselines3", "torch"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = Path(__file__).resolve().parents[3]
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    return {
        "source_sha256": source_fingerprint(),
        "commit": commit,
        "dirty": dirty,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "dependencies": versions,
    }


def run_episode(scenario: Scenario, identity: dict) -> dict:
    """Evaluate one policy with ground-truth metrics and full replay trajectory."""
    env = ScenarioEnvironment(scenario)
    policy = make_policy(identity, scenario)
    try:
        obs, info = env.reset()
        scene = env.scene()
        trajectory = []
        total_return = 0.0
        reason = "RUNNING"
        for step in range(scenario.max_steps):
            measured_obs, measured_info = env.measured(obs, info)
            action = np.asarray(policy(measured_obs, measured_info), dtype=np.float32)
            obs, reward, terminated, truncated, info = env.step(action)
            total_return += reward
            trajectory.append(
                {
                    "step": step,
                    "time_s": (step + 1) * 0.2,
                    "x_m": info["x_m"],
                    "y_m": info["y_m"],
                    "yaw_rad": info["yaw_rad"],
                    "speed_ms": info["speed_ms"],
                    "lateral_error_m": info["lateral_error_m"],
                    "minimum_clearance_m": info["minimum_clearance_m"],
                    "requested_action": action.tolist(),
                    "applied_action": info["applied_action"],
                    "measured_lateral_error_m": measured_info["lateral_error_m"],
                    "reason": info["terminal_reason"],
                }
            )
            if terminated or truncated:
                reason = info["terminal_reason"]
                break
        speeds = np.array([p["speed_ms"] for p in trajectory])
        lateral = np.array([p["lateral_error_m"] for p in trajectory])
        actions = np.array([p["applied_action"] for p in trajectory])
        acceleration = np.diff(speeds) / 0.2
        jerk = np.diff(acceleration) / 0.2
        metrics = {
            "success": reason == "SUCCESS",
            "collision": reason == "COLLISION",
            "offroad": reason == "OFFROAD",
            "timeout": reason == "TIMEOUT",
            "terminal_reason": reason,
            "steps": len(trajectory),
            "return": float(total_return),
            "route_completion": float(info["route_fraction"]),
            "mean_speed_kmh": float(np.mean(speeds) * 3.6),
            "mean_abs_lateral_error_m": float(np.mean(np.abs(lateral))),
            "max_abs_lateral_error_m": float(np.max(np.abs(lateral))),
            "minimum_clearance_m": float(min(p["minimum_clearance_m"] for p in trajectory)),
            "mean_abs_steering_change": (
                float(np.mean(np.abs(np.diff(actions[:, 0])))) if len(actions) > 1 else 0.0
            ),
            "mean_abs_longitudinal_jerk_ms3": float(np.mean(np.abs(jerk))) if len(jerk) else 0.0,
        }
        result = {
            "schema": 1,
            "scenario": asdict(scenario),
            "policy": identity,
            "scene": scene,
            "metrics": metrics,
            "trajectory": trajectory,
        }
        result["content_sha256"] = canonical_hash(result)
        return result
    finally:
        env.close()


def summarize(episodes: list[dict]) -> dict:
    """Aggregate actual outcomes; all rates are fractions over episodes."""
    if not episodes:
        raise ValueError("Cannot summarize an empty run.")
    metrics = [ep["metrics"] for ep in episodes]
    n = len(metrics)
    result: dict[str, Any] = {"episodes": n}
    for name in ("success", "collision", "offroad", "timeout"):
        result[f"{name}_rate"] = sum(m[name] for m in metrics) / n
    for name in ("route_completion", "mean_abs_lateral_error_m", "mean_speed_kmh"):
        result[name] = float(np.mean([m[name] for m in metrics]))
    # Wilson interval for success rate, explicitly finite-sample uncertainty.
    z = 1.959963984540054
    p = result["success_rate"]
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    result["success_95ci"] = [float(centre - half), float(centre + half)]
    return result


def run_suite(suite: dict, scenarios: list[Scenario], identity: dict, out: Path) -> dict:
    """Create a fresh bundle; an existing run is never silently overwritten."""
    out.mkdir(parents=True, exist_ok=False)
    (out / "episodes").mkdir()
    run: dict[str, Any] = {
        "schema": 1,
        "backend": "kinematic",
        "suite_sha256": canonical_hash(suite),
        "suite": suite,
        "policy": identity,
        "provenance": provenance(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "episodes": [],
    }
    episodes = []
    for scenario in scenarios:
        episode = run_episode(scenario, identity)
        file = f"episodes/{scenario.id}.json"
        write_json(out / file, episode)
        run["episodes"].append(
            {
                "id": scenario.id,
                "file": file,
                "content_sha256": episode["content_sha256"],
                "metrics": episode["metrics"],
            }
        )
        episodes.append(episode)
    run["summary"] = summarize(episodes)
    run["by_regime"] = {
        regime: summarize([e for e in episodes if e["scenario"]["regime"] == regime])
        for regime in sorted({s.regime for s in scenarios})
    }
    write_json(out / "run.json", run)
    return run


def load_run(path: Path) -> dict:
    """Verify episode digests and metadata before using saved evidence."""
    run = json.loads(path.read_text(encoding="utf-8"))
    if run.get("schema") != 1 or run.get("backend") != "kinematic":
        raise ValueError("Unsupported evidence schema/backend.")
    if canonical_hash(run["suite"]) != run["suite_sha256"]:
        raise ValueError("Suite integrity check failed.")
    episodes = []
    expected_cases = {s["id"]: s for s in run["suite"]["scenarios"]}
    ids = set()
    for entry in run["episodes"]:
        file = (path.parent / entry["file"]).resolve()
        if not file.is_relative_to(path.parent.resolve()):
            raise ValueError("Episode path escaped the evidence directory.")
        episode = json.loads(file.read_text(encoding="utf-8"))
        digest = episode.pop("content_sha256")
        if canonical_hash(episode) != digest or digest != entry["content_sha256"]:
            raise ValueError("Episode integrity check failed.")
        if episode["metrics"] != entry["metrics"] or episode["policy"] != run["policy"]:
            raise ValueError("Episode metadata disagrees with run.")
        case_id = entry["id"]
        if case_id in ids or episode["scenario"] != asdict(Scenario(**expected_cases[case_id])):
            raise ValueError("Scenario mapping mismatch or duplicate episode.")
        ids.add(case_id)
        episodes.append(episode)
    if ids != set(expected_cases) or summarize(episodes) != run["summary"]:
        raise ValueError("Run summary or completeness check failed.")
    return run
