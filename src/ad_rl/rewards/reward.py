"""A single, well-documented reward function used by every environment.

Keeping the reward in one place (rather than duplicated inside each env) means a
policy trained against the fast fallback environment optimises *exactly* the same
objective it will see in CARLA. Each component is returned separately so it can be
logged to TensorBoard and inspected during debugging -- reward shaping is where
most driving-RL projects quietly go wrong, so it pays to make it observable.

Reward components
-----------------
speed     : encourages tracking a target cruising speed (penalises both crawling
            and speeding).
progress  : rewards distance advanced along the reference route this step.
lane      : penalises lateral deviation from the lane centre (quadratic).
heading   : penalises misalignment between the car's heading and the road.
steer     : penalises large steering angles (passenger comfort).
jerk      : penalises abrupt action changes (smoothness).
collision : large one-off penalty (also terminates the episode).
offroad   : one-off penalty for leaving the drivable area (also terminates).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict

from ad_rl.utils.config import RewardConfig


@dataclass
class DriveMeasurement:
    """A simulator-agnostic snapshot of the ego vehicle at one timestep.

    Both :class:`~ad_rl.envs.carla_env.CarlaEnv` and
    :class:`~ad_rl.envs.fallback_env.KinematicDrivingEnv` populate this struct so
    they can call the identical reward function.
    """

    speed_ms: float  # current speed (m/s)
    lateral_error_m: float  # signed distance from lane centre (m)
    heading_error_rad: float  # heading minus road direction, wrapped to [-pi, pi]
    progress_m: float  # distance advanced along the route this step (m)
    steer: float  # current steering command in [-1, 1]
    prev_steer: float  # previous steering command in [-1, 1]
    collided: bool = False
    offroad: bool = False
    reached_goal: bool = False


@dataclass
class RewardResult:
    """The scalar reward plus its decomposition for logging."""

    total: float
    components: Dict[str, float] = field(default_factory=dict)


def _speed_term(speed_ms: float, target_ms: float) -> float:
    """Triangular speed reward in [-1, 1], peaking at the target speed.

    At ``target`` the reward is 1. It falls linearly to 0 at standstill and at
    twice the target, and goes negative beyond that to discourage speeding.
    """
    if target_ms <= 0:
        return 0.0
    ratio = speed_ms / target_ms
    return float(max(-1.0, 1.0 - abs(1.0 - ratio)))


def compute_reward(meas: DriveMeasurement, cfg: RewardConfig) -> RewardResult:
    """Compute the shaped reward for a single timestep.

    Parameters
    ----------
    meas:
        The current-step measurement.
    cfg:
        Reward weights.

    Returns
    -------
    RewardResult
        ``total`` is the scalar reward; ``components`` maps each named term to its
        (already weighted) contribution.
    """
    # Terminal events short-circuit shaping with a dominant penalty so the agent
    # gets an unambiguous learning signal.
    if meas.collided:
        return RewardResult(total=-cfg.collision_penalty, components={"collision": -cfg.collision_penalty})
    if meas.offroad:
        return RewardResult(total=-cfg.offroad_penalty, components={"offroad": -cfg.offroad_penalty})

    half_width = 2.0  # normalisation constant for lateral error (metres)

    speed = cfg.w_speed * _speed_term(meas.speed_ms, cfg.target_speed_ms)
    progress = cfg.w_progress * meas.progress_m
    lane = -cfg.w_lane * (meas.lateral_error_m / half_width) ** 2
    heading = -cfg.w_heading * (abs(meas.heading_error_rad) / math.pi)
    steer = -cfg.w_steer * meas.steer**2
    jerk = -cfg.w_jerk * (meas.steer - meas.prev_steer) ** 2

    components = {
        "speed": speed,
        "progress": progress,
        "lane": lane,
        "heading": heading,
        "steer": steer,
        "jerk": jerk,
    }
    if meas.reached_goal:
        components["goal"] = 10.0

    total = float(sum(components.values()))
    return RewardResult(total=total, components=components)


__all__ = ["DriveMeasurement", "RewardResult", "compute_reward"]
