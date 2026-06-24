"""Environment factory.

``make_env`` returns a Gymnasium environment for either backend:

* ``"fallback"`` -> :class:`KinematicDrivingEnv` (no simulator, used in CI/tests).
* ``"carla"``    -> :class:`CarlaEnv` (requires a running CARLA server).

Both expose an identical observation/action interface, so the rest of the
codebase is backend-agnostic.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import gymnasium as gym

from ad_rl.envs.wrappers import ActionSmoothingWrapper, maybe_frame_stack

if TYPE_CHECKING:
    from ad_rl.utils.config import Config

VALID_ENVS = ("fallback", "carla")


def make_env(
    env_name: str,
    cfg: Config,
    render_mode: str | None = None,
    smooth_actions: bool = False,
) -> gym.Env:
    """Construct a driving environment.

    Parameters
    ----------
    env_name:
        ``"fallback"`` or ``"carla"``.
    cfg:
        A loaded :class:`~ad_rl.utils.config.Config`.
    render_mode:
        Passed through to the env (``"rgb_array"`` to enable video frames).
    smooth_actions:
        Wrap the env with :class:`ActionSmoothingWrapper`.
    """
    env_name = env_name.lower()
    if env_name == "fallback":
        from ad_rl.envs.fallback_env import KinematicDrivingEnv

        env: gym.Env = KinematicDrivingEnv(
            env_cfg=cfg.env,
            reward_cfg=cfg.reward,
            fallback_cfg=cfg.fallback,
            render_mode=render_mode,
        )
    elif env_name == "carla":
        from ad_rl.envs.carla_env import CarlaEnv

        env = CarlaEnv(
            env_cfg=cfg.env,
            reward_cfg=cfg.reward,
            carla_cfg=cfg.carla,
            render_mode=render_mode,
        )
    else:
        raise ValueError(f"Unknown env '{env_name}'. Expected one of {VALID_ENVS}.")

    env = maybe_frame_stack(env, cfg.env.observation, cfg.env.frame_stack)
    if smooth_actions:
        env = ActionSmoothingWrapper(env)
    return env


__all__ = ["VALID_ENVS", "make_env"]
