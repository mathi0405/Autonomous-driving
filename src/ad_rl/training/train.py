"""Training entrypoint.

Examples
--------
Fast CPU smoke run on the fallback env::

    python -m ad_rl.training.train --config configs/ppo.yaml --env fallback \
        --total-timesteps 2000 --eval-episodes 2 --run-name smoke --no-progress

Full PPO run against a live CARLA server::

    python -m ad_rl.training.train --config configs/ppo.yaml --env carla
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict
from pathlib import Path

import yaml

from ad_rl.agents import build_agent
from ad_rl.envs import make_env
from ad_rl.training.utils import make_vector_env
from ad_rl.utils.config import Config, load_config
from ad_rl.utils.logging import get_logger
from ad_rl.utils.seeding import set_global_seeds

logger = get_logger("ad_rl.train")


def parse_args(argv=None) -> argparse.Namespace:
    """Parse CLI arguments for the training script."""
    p = argparse.ArgumentParser(description="Train a driving agent (PPO/SAC).")
    p.add_argument("--config", required=True, help="Path to an algorithm config YAML.")
    p.add_argument("--env", default="fallback", choices=["fallback", "carla"])
    p.add_argument("--total-timesteps", type=int, default=None, help="Override config.")
    p.add_argument("--n-envs", type=int, default=None, help="Override number of parallel envs.")
    p.add_argument("--obs", default=None, choices=["state", "image"], help="Override observation.")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default="auto", help="'auto', 'cpu', or 'cuda'.")
    p.add_argument("--vec", default="dummy", choices=["dummy", "subproc"])
    p.add_argument("--run-name", default=None)
    p.add_argument("--eval-episodes", type=int, default=None)
    p.add_argument("--smooth-actions", action="store_true")
    p.add_argument("--no-tensorboard", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--outdir", default="runs")
    p.add_argument(
        "--carla-eval-port",
        type=int,
        default=2001,
        help="Separate CARLA evaluation server; must differ from the training port.",
    )
    return p.parse_args(argv)


def apply_overrides(cfg: Config, args: argparse.Namespace) -> Config:
    """Apply CLI argument overrides onto the loaded config."""
    if args.total_timesteps is not None:
        cfg.total_timesteps = args.total_timesteps
    if args.n_envs is not None:
        cfg.n_envs = args.n_envs
    if args.obs is not None:
        cfg.env.observation = args.obs
    if args.seed is not None:
        cfg.seed = args.seed
    if args.eval_episodes is not None:
        cfg.logging["eval_episodes"] = args.eval_episodes
    if args.no_tensorboard:
        cfg.logging["tensorboard"] = False
    if args.smooth_actions:
        cfg.env.smooth_actions = True
    # State observations are low-dim; multi-frame stacking is unnecessary there.
    if cfg.env.observation == "state" and cfg.env.frame_stack > 1:
        cfg.env.frame_stack = 1
    return cfg


def train(args: argparse.Namespace) -> Path:
    """Run training end-to-end and return the output directory path."""
    cfg = apply_overrides(load_config(args.config), args)
    eval_cfg = copy.deepcopy(cfg)
    if args.env == "carla":
        if args.carla_eval_port == cfg.carla.port:
            raise ValueError("CARLA training and evaluation require separate server ports.")
        cfg.n_envs = 1
        eval_cfg.carla.port = args.carla_eval_port
    set_global_seeds(cfg.seed)

    run_name = args.run_name or f"{cfg.algorithm}_{args.env}_{cfg.env.observation}"
    run_dir = Path(args.outdir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = str(run_dir / "tb") if cfg.logging.get("tensorboard", True) else None

    logger.info(
        f"[bold]Training {cfg.algorithm.upper()}[/bold] on '{args.env}' "
        f"({cfg.env.observation} obs) for {cfg.total_timesteps:,} steps -> {run_dir}"
    )

    n_envs = 1 if args.env == "carla" else cfg.n_envs
    train_venv = make_vector_env(args.env, cfg, n_envs=n_envs, seed=cfg.seed, vec=args.vec)
    eval_venv = None
    try:
        eval_venv = make_vector_env(args.env, eval_cfg, n_envs=1, seed=cfg.seed + 777)

        model = build_agent(
            cfg.algorithm, train_venv, cfg, device=args.device, tensorboard_log=tb_dir
        )

        callbacks = _build_callbacks(cfg, eval_venv, run_dir, n_envs)
        model.learn(
            total_timesteps=cfg.total_timesteps,
            callback=callbacks,
            progress_bar=not args.no_progress,
            tb_log_name=run_name,
        )

        final_path = run_dir / "final_model.zip"
        model.save(final_path)
        resolved = asdict(cfg)
        resolved.pop("raw", None)
        (run_dir / "config.yaml").write_text(yaml.safe_dump(resolved), encoding="utf-8")
        from ad_rl.validation.runner import provenance, write_json

        write_json(
            run_dir / "provenance.json",
            {
                **provenance(),
                "backend": args.env,
                "resolved_config": resolved,
                "cli": vars(args),
                "evaluation_port": eval_cfg.carla.port if args.env == "carla" else None,
            },
        )
        logger.info(f"Saved final model -> {final_path}")

        eval_venv.close()
        eval_venv = None
        metrics = _final_evaluation(model, args.env, eval_cfg, run_dir)
        logger.info(f"Final eval: {json.dumps(metrics, indent=2)}")

        return run_dir
    finally:
        train_venv.close()
        if eval_venv is not None:
            eval_venv.close()


def _build_callbacks(cfg: Config, eval_venv, run_dir: Path, n_envs: int):
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )

    from ad_rl.training.callbacks import RewardComponentsCallback

    eval_freq = max(1, int(cfg.logging.get("eval_freq", 25000)) // max(1, n_envs))
    ckpt_freq = max(1, int(cfg.logging.get("checkpoint_freq", 50000)) // max(1, n_envs))
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=str(run_dir),
        log_path=str(run_dir / "eval"),
        eval_freq=eval_freq,
        n_eval_episodes=int(cfg.logging.get("eval_episodes", 5)),
        deterministic=True,
        render=False,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=ckpt_freq, save_path=str(run_dir / "checkpoints"), name_prefix=cfg.algorithm
    )
    return CallbackList([eval_cb, ckpt_cb, RewardComponentsCallback()])


def _final_evaluation(model, env_name: str, cfg: Config, run_dir: Path) -> dict:
    """Run a deterministic evaluation and persist metrics + summary entry."""
    from ad_rl.evaluation.evaluate import policy_from_model, run_episodes
    from ad_rl.evaluation.metrics import aggregate, update_summary

    n_eval = int(cfg.logging.get("eval_episodes", 5))
    eval_env = make_env(env_name, cfg)
    try:
        records = run_episodes(
            eval_env, policy_from_model(model, deterministic=True), n_eval, seed=cfg.seed + 10_000
        )
    finally:
        eval_env.close()
    metrics = aggregate(records)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    update_summary(
        Path("results") / "summary.json",
        agent=cfg.algorithm.upper(),
        metrics=metrics,
        returns=[r.ret for r in records],
        meta={
            "env": env_name,
            "observation": cfg.env.observation,
            "timesteps": cfg.total_timesteps,
        },
    )
    return metrics


def main(argv=None) -> None:
    """CLI entry point: parse arguments, train, and save the model."""
    train(parse_args(argv))


if __name__ == "__main__":
    main()
