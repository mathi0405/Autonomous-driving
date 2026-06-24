#!/usr/bin/env python3
"""Generate static figures for the README / project report.

Produces (into ``docs/images/``):
  * ``fallback_trajectory.png``    -- PID baseline tracking the procedural lane.
  * ``topdown_observation.png``    -- the top-down RGB frames the CNN consumes.
  * ``metrics_comparison.png``     -- bar chart of evaluation metrics per agent.

Everything here is generated from the real environment / controller / results,
so the figures stay honest and reproducible.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ad_rl.control import VehiclePIDController
from ad_rl.envs import make_env
from ad_rl.utils.config import load_config

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
OUT.mkdir(parents=True, exist_ok=True)


def _cfg(observation: str):
    cfg = load_config(ROOT / "configs" / "ppo.yaml")
    cfg.env.observation = observation
    cfg.env.frame_stack = 1
    cfg.env.max_episode_steps = 600
    return cfg


def trajectory_figure() -> None:
    cfg = _cfg("state")
    env = make_env("fallback", cfg)
    base = env.unwrapped
    _, info = env.reset(seed=11)
    ctrl = VehiclePIDController(cfg.reward.target_speed_ms, dt=cfg.fallback.dt * cfg.env.action_repeat)
    ctrl.reset()
    xs, ys = [base._x], [base._y]
    done = False
    while not done:
        _, _, term, trunc, info = env.step(ctrl.act_from_info(info))
        xs.append(base._x)
        ys.append(base._y)
        done = term or trunc

    path, yaw = base._path_xy, base._path_yaw
    hw = cfg.fallback.road_half_width_m
    normals = np.stack([-np.sin(yaw), np.cos(yaw)], axis=1)
    left, right = path + hw * normals, path - hw * normals

    plt.figure(figsize=(9, 5))
    plt.plot(path[:, 0], path[:, 1], "--", color="#9aa7b8", lw=1, label="lane center")
    plt.plot(left[:, 0], left[:, 1], color="#444", lw=1)
    plt.plot(right[:, 0], right[:, 1], color="#444", lw=1, label="road edge")
    plt.plot(xs, ys, color="#36c98d", lw=2.2, label="PID trajectory")
    if len(base._obstacles):
        plt.scatter(
            base._obstacles[:, 0], base._obstacles[:, 1],
            c="#d9534f", marker="s", s=45, zorder=5, label="obstacles",
        )
    plt.axis("equal")
    plt.legend(loc="best", fontsize=9)
    plt.title("Fallback environment — PID baseline tracking a procedural lane")
    plt.xlabel("x (m)")
    plt.ylabel("y (m)")
    plt.tight_layout()
    plt.savefig(OUT / "fallback_trajectory.png", dpi=130)
    plt.close()


def observation_figure() -> None:
    cfg = _cfg("image")
    env = make_env("fallback", cfg)
    _, info = env.reset(seed=11)
    ctrl = VehiclePIDController(cfg.reward.target_speed_ms, dt=cfg.fallback.dt * cfg.env.action_repeat)
    ctrl.reset()
    frames = []
    obs = None
    for i in range(120):
        obs, _, term, trunc, info = env.step(ctrl.act_from_info(info))
        if i % 30 == 0:
            frames.append(obs)
        if term or trunc:
            break
    frames = frames[:4] or [obs]

    fig, axes = plt.subplots(1, len(frames), figsize=(2.3 * len(frames), 2.6))
    if len(frames) == 1:
        axes = [axes]
    for ax, frame in zip(axes, frames):
        ax.imshow(frame)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Top-down RGB observations (84×84, CNN input)", fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT / "topdown_observation.png", dpi=130)
    plt.close()


def metrics_figure() -> None:
    summary_path = ROOT / "results" / "summary.json"
    if not summary_path.exists():
        return
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    agents = list(summary.get("agents", {}).keys())
    if not agents:
        return
    metrics = ["success_rate", "mean_route_completion", "mean_abs_lateral_error_m", "mean_abs_jerk"]
    titles = ["Success rate ↑", "Route completion ↑", "Lateral error (m) ↓", "Steering jerk ↓"]
    palette = {"PPO": "#4f9cff", "SAC": "#7c5cff", "PID": "#36c98d", "STANLEY": "#f4b740"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(3.1 * len(metrics), 3.2))
    for ax, key, title in zip(axes, metrics, titles):
        vals = [summary["agents"][a]["metrics"].get(key, 0.0) for a in agents]
        ax.bar(agents, vals, color=[palette.get(a.upper(), "#9aa7b8") for a in agents])
        ax.set_title(title, fontsize=10)
        ax.tick_params(axis="x", labelrotation=20, labelsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Evaluation metrics by agent", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT / "metrics_comparison.png", dpi=130)
    plt.close()


def main() -> None:
    trajectory_figure()
    observation_figure()
    metrics_figure()
    print(f"Figures written -> {OUT}")


if __name__ == "__main__":
    main()
