"""Evaluation entrypoint: roll out a policy and compute driving metrics.

Evaluate a trained model::

    python -m ad_rl.evaluation.evaluate --model runs/ppo_fallback_state/best_model.zip \
        --algo ppo --env fallback --obs state --episodes 20

Evaluate a classical baseline (no training required)::

    python -m ad_rl.evaluation.evaluate --model pid --env fallback --obs state --episodes 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import numpy as np

from ad_rl.envs import make_env
from ad_rl.evaluation.metrics import EpisodeRecord, aggregate, update_summary
from ad_rl.utils.config import Config, load_config
from ad_rl.utils.logging import get_logger

logger = get_logger("ad_rl.eval")

PolicyFn = Callable[[np.ndarray, dict], np.ndarray]


class BaselinePolicy:
    """Wraps a classical controller as a ``(obs, info) -> action`` policy."""

    def __init__(self, controller) -> None:
        self._controller = controller

    def reset(self) -> None:
        """Reset the underlying classical controller."""
        self._controller.reset()

    def __call__(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """Run the controller and return an action array."""
        return self._controller.act_from_info(info)


def policy_from_model(model, deterministic: bool = True) -> PolicyFn:
    """Wrap an SB3 model's ``predict`` as a policy function."""

    def _policy(obs: np.ndarray, info: dict) -> np.ndarray:
        action, _ = model.predict(obs, deterministic=deterministic)
        return action

    return _policy


def policy_from_baseline(kind: str, target_speed_ms: float, dt: float) -> BaselinePolicy:
    """Build a PID or Stanley baseline policy."""
    from ad_rl.control import VehiclePIDController

    lateral = "stanley" if kind.lower() == "stanley" else "pid"
    return BaselinePolicy(VehiclePIDController(target_speed_ms, dt=dt, lateral=lateral))


def run_episodes(
    env, policy: PolicyFn, n_episodes: int, seed: int | None = None
) -> list[EpisodeRecord]:
    """Roll out ``policy`` for ``n_episodes`` and record per-episode statistics."""
    records: list[EpisodeRecord] = []
    for ep in range(n_episodes):
        if hasattr(policy, "reset"):
            policy.reset()  # type: ignore[attr-defined]
        reset_seed = None if seed is None else seed + ep
        obs, info = env.reset(seed=reset_seed)
        done = False
        ret, length = 0.0, 0
        speeds: list[float] = []
        laterals: list[float] = []
        steers: list[float] = []
        last_info = info
        while not done:
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            length += 1
            speeds.append(float(info.get("speed_ms", 0.0)))
            laterals.append(float(info.get("lateral_error_m", 0.0)))
            steers.append(float(np.asarray(action).reshape(-1)[0]))
            last_info = info
            done = terminated or truncated
        records.append(
            EpisodeRecord(
                ret=ret,
                length=length,
                success=bool(last_info.get("is_success", False)),
                collided=bool(last_info.get("collision", False)),
                offroad=bool(last_info.get("offroad", False)),
                route_fraction=float(last_info.get("route_fraction", 0.0)),
                speeds=speeds,
                lateral_errors=laterals,
                steers=steers,
            )
        )
    return records


def _effective_dt(env_name: str, cfg: Config) -> float:
    """Compute the effective dt (base dt * action_repeat)."""
    base = cfg.fallback.dt if env_name == "fallback" else cfg.carla.fixed_delta_seconds
    return base * cfg.env.action_repeat


def parse_args(argv=None) -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    p = argparse.ArgumentParser(description="Evaluate a driving policy.")
    p.add_argument("--model", required=True, help="Path to a model .zip, or 'pid'/'stanley'.")
    p.add_argument("--algo", default=None, choices=["ppo", "sac"], help="Algo for loading a model.")
    p.add_argument("--config", default="configs/ppo.yaml")
    p.add_argument("--env", default="fallback", choices=["fallback", "carla"])
    p.add_argument("--obs", default=None, choices=["state", "image"])
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=20240)
    p.add_argument("--stochastic", action="store_true", help="Sample actions instead of argmax.")
    p.add_argument("--out", default="results/summary.json")
    p.add_argument("--agent-name", default=None)
    return p.parse_args(argv)


def evaluate(args: argparse.Namespace) -> dict:
    """Run evaluation rollouts and return the aggregated metrics dict."""
    cfg = load_config(args.config)
    if args.obs is not None:
        cfg.env.observation = args.obs
    if cfg.env.observation == "state" and cfg.env.frame_stack > 1:
        cfg.env.frame_stack = 1

    env = make_env(args.env, cfg)

    baseline = args.model.lower() in ("pid", "stanley")
    if baseline:
        policy: PolicyFn = policy_from_baseline(
            args.model, cfg.reward.target_speed_ms, _effective_dt(args.env, cfg)
        )
        agent_name = args.agent_name or args.model.upper()
    else:
        from ad_rl.agents import load_agent

        algo = args.algo or cfg.algorithm
        model = load_agent(algo, args.model)
        policy = policy_from_model(model, deterministic=not args.stochastic)
        agent_name = args.agent_name or algo.upper()

    logger.info(
        f"Evaluating [bold]{agent_name}[/bold]"
        f" on '{args.env}' for {args.episodes} episodes"
    )
    records = run_episodes(env, policy, args.episodes, seed=args.seed)
    metrics = aggregate(records)
    _print_metrics(agent_name, metrics)

    update_summary(
        Path(args.out),
        agent=agent_name,
        metrics=metrics,
        returns=[r.ret for r in records],
        meta={"env": args.env, "observation": cfg.env.observation, "episodes": args.episodes},
    )
    logger.info(f"Updated results summary -> {args.out}")
    return metrics


def _print_metrics(agent: str, metrics: dict) -> None:
    """Pretty-print a formatted metrics table to stdout."""
    line = "-" * 46
    print(f"\n{line}\n  {agent} — evaluation metrics\n{line}")
    for key, value in metrics.items():
        print(f"  {key:<26} {value:>14.3f}")
    print(line)


def main(argv=None) -> None:
    """CLI entry point: parse arguments and run evaluate()."""
    evaluate(parse_args(argv))


if __name__ == "__main__":
    main()
