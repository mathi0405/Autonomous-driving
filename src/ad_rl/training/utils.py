"""Helpers for building vectorised training/eval environments."""

from __future__ import annotations

from typing import Callable

from ad_rl.envs import make_env
from ad_rl.utils.config import Config


def make_env_thunk(env_name: str, cfg: Config, render_mode=None) -> Callable[[], object]:
    """Return a zero-arg factory that builds a fresh env (for SB3 vec envs)."""

    def _thunk():
        return make_env(env_name, cfg, render_mode=render_mode)

    return _thunk


def make_vector_env(
    env_name: str,
    cfg: Config,
    n_envs: int = 1,
    seed: int = 0,
    vec: str = "dummy",
):
    """Build a Stable-Baselines3 vectorised, Monitor-wrapped environment.

    Parameters
    ----------
    env_name:
        ``"fallback"`` or ``"carla"``.
    n_envs:
        Number of parallel environments.
    vec:
        ``"dummy"`` (single process, robust) or ``"subproc"`` (multiprocessing,
        faster for the cheap fallback env). CARLA should always use ``"dummy"``
        with a single env per server.
    """
    from stable_baselines3.common.env_util import make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    vec_env_cls = SubprocVecEnv if (vec == "subproc" and n_envs > 1) else DummyVecEnv
    return make_vec_env(
        make_env_thunk(env_name, cfg),
        n_envs=n_envs,
        seed=seed,
        vec_env_cls=vec_env_cls,
    )


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """Linear decay schedule for learning rate (1.0 -> 0.0 of ``initial_value``)."""

    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value

    return func


__all__ = ["make_vector_env", "make_env_thunk", "linear_schedule"]
