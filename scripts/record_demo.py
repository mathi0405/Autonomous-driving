#!/usr/bin/env python3
"""Record a demo video/GIF of a policy driving.

Works for a trained PPO/SAC model or a classical baseline, on either backend.
GIF output needs only Pillow; MP4 additionally uses imageio-ffmpeg (falls back
to GIF if unavailable).

Examples
--------
    # Trained PPO on the fallback env -> GIF
    python scripts/record_demo.py --model runs/ppo_fallback/best_model.zip --algo ppo \
        --env fallback --obs state --episodes 3 --out docs/images/demo.gif

    # PID baseline -> MP4
    python scripts/record_demo.py --model pid --env fallback --out docs/images/demo.mp4

    # Trained policy in CARLA (server must be running)
    python scripts/record_demo.py --model runs/ppo_carla_image/best_model.zip --algo ppo \
        --env carla --obs image --out docs/images/carla_demo.mp4
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ad_rl.envs import make_env
from ad_rl.evaluation.evaluate import policy_from_baseline, policy_from_model
from ad_rl.utils.config import load_config
from ad_rl.utils.logging import get_logger

logger = get_logger("ad_rl.demo")


def _hud(frame: np.ndarray, scale: int, lines: list[str]) -> Image.Image:
    """Upscale a frame and draw a small heads-up display."""
    img = Image.fromarray(frame.astype(np.uint8)).resize(
        (frame.shape[1] * scale, frame.shape[0] * scale), Image.NEAREST
    )
    draw = ImageDraw.Draw(img)
    y = 4
    for line in lines:
        # cheap outline for readability over any background
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            draw.text((6 + dx, y + dy), line, fill=(0, 0, 0))
        draw.text((6, y), line, fill=(255, 255, 255))
        y += 14
    return img


def _build_policy(args, cfg):
    if args.model.lower() in ("pid", "stanley"):
        base = cfg.fallback.dt if args.env == "fallback" else cfg.carla.fixed_delta_seconds
        dt = base * cfg.env.action_repeat
        return policy_from_baseline(args.model, cfg.reward.target_speed_ms, dt), args.model.upper()
    from ad_rl.agents import load_agent

    algo = args.algo or cfg.algorithm
    model = load_agent(algo, args.model)
    return policy_from_model(model, deterministic=True), algo.upper()


def record(args) -> Path:
    cfg = load_config(args.config)
    if args.obs:
        cfg.env.observation = args.obs
    if cfg.env.observation == "state" and cfg.env.frame_stack > 1:
        cfg.env.frame_stack = 1
    # A state policy ignores pixels, so we can render a larger, nicer top-down view.
    if cfg.env.observation == "state":
        cfg.env.image_size = (args.render_size, args.render_size)

    env = make_env(args.env, cfg, render_mode="rgb_array")
    policy, name = _build_policy(args, cfg)

    frames: list[Image.Image] = []
    for ep in range(args.episodes):
        if hasattr(policy, "reset"):
            policy.reset()
        obs, info = env.reset(seed=args.seed + ep)
        done, step, ret = False, 0, 0.0
        while not done and step < args.max_steps:
            action = policy(obs, info)
            obs, reward, terminated, truncated, info = env.step(action)
            ret += float(reward)
            step += 1
            done = terminated or truncated
            frame = env.render()
            if frame is not None:
                frames.append(
                    _hud(
                        frame,
                        args.scale,
                        [
                            f"{name}  ep {ep + 1}/{args.episodes}",
                            f"speed {info.get('speed_ms', 0.0) * 3.6:4.1f} km/h",
                            f"route {info.get('route_fraction', 0.0) * 100:4.1f}%   return {ret:6.1f}",
                        ],
                    )
                )
    env.close() if hasattr(env, "close") else None
    if not frames:
        raise RuntimeError("No frames captured — is render_mode supported for this env?")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    _write(frames, out, args.fps)
    logger.info(f"Recorded {len(frames)} frames over {args.episodes} episode(s) -> {out}")
    return out


def _write(frames: list[Image.Image], out: Path, fps: int) -> None:
    if out.suffix.lower() == ".mp4":
        try:
            import imageio.v2 as imageio

            imageio.mimsave(out, [np.asarray(f) for f in frames], fps=fps, codec="libx264")
            return
        except Exception as exc:  # pragma: no cover - depends on ffmpeg availability
            out = out.with_suffix(".gif")
            logger.info(f"MP4 writer unavailable ({exc}); falling back to {out}")
    # GIF via Pillow (no extra dependencies).
    duration = int(1000 / max(1, fps))
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record a driving demo (GIF/MP4).")
    p.add_argument("--model", required=True, help="Path to a model .zip, or 'pid'/'stanley'.")
    p.add_argument("--algo", default=None, choices=["ppo", "sac"])
    p.add_argument("--config", default="configs/ppo.yaml")
    p.add_argument("--env", default="fallback", choices=["fallback", "carla"])
    p.add_argument("--obs", default=None, choices=["state", "image"])
    p.add_argument("--episodes", type=int, default=3)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="docs/images/demo.gif")
    p.add_argument("--fps", type=int, default=20)
    p.add_argument("--scale", type=int, default=3, help="Integer upscale factor for the frames.")
    p.add_argument("--render-size", type=int, default=240, help="Render resolution for state policies.")
    return p.parse_args(argv)


def main(argv=None) -> None:
    record(parse_args(argv))


if __name__ == "__main__":
    main()
