"""Shared, well-documented reward function used by every simulation backend.

Design rationale
----------------
Keeping the reward in a single, simulator-agnostic module ensures that a policy
trained against the lightweight fallback environment optimises *exactly* the same
objective it will encounter in CARLA. This eliminates a common source of
transfer failure in sim-to-sim and sim-to-real pipelines.

Each reward component is returned separately in ``RewardResult.components`` so
that it can be logged to TensorBoard and inspected during debugging. Reward
shaping is where most driving-RL projects quietly go wrong; making every term
observable is essential for principled diagnosis.

See ``docs/reward_shaping.md`` for the full design rationale, weight sensitivity
analysis, and known limitations of this formulation.

Reward components
-----------------
speed
    Triangular reward in [−1, 1] that peaks at the target cruising speed.
    Bounded to prevent the speed term from dominating at extreme velocities.
progress
    Distance advanced along the reference route this step (m). Provides a dense
    forward-motion signal that is robust to route geometry.
lane
    Quadratic penalty on lateral deviation from the lane centre. The quadratic
    form is differentiable at zero and provides a progressive (non-linear)
    pull toward the centreline.
heading
    Linear penalty on heading error normalised by π. Normalisation ensures the
    penalty magnitude is independent of road curvature.
steer
    Quadratic penalty on instantaneous steering magnitude (passenger comfort).
jerk
    Quadratic penalty on change in steering between consecutive timesteps
    (actuator smoothness). Removing this term produces oscillatory steering.
collision
    Large one-off penalty; also terminates the episode.
offroad
    Medium penalty for leaving the drivable area; also terminates the episode.
goal
    One-time bonus awarded when the agent reaches the designated goal waypoint.

References
----------
- Kendall et al. (2019). "Learning to Drive in a Day." ICRA 2019.
- Liang et al. (2018). "CIRL: Controllable Imitative Reinforcement Learning."
  ECCV 2018.
- Dosovitskiy et al. (2017). "CARLA: An Open Urban Driving Simulator." CoRL.
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
    :class:`~ad_rl.envs.fallback_env.KinematicDrivingEnv` populate this struct
    so they can call the identical reward function, guaranteeing objective
    consistency across environment backends.

    Parameters
    ----------
    speed_ms : float
        Current longitudinal speed of the ego vehicle (m/s). Must be >= 0.
    lateral_error_m : float
        Signed perpendicular distance from the lane centreline (m).
        Positive values indicate the vehicle is to the right of centre.
    heading_error_rad : float
        Difference between the vehicle's heading and the road tangent direction,
        wrapped to the interval [−π, π] (rad).
    progress_m : float
        Arc-length distance advanced along the reference route during this
        timestep (m). Negative values indicate the vehicle is moving backwards.
    steer : float
        Current steering command normalised to [−1, 1]. Negative = left.
    prev_steer : float
        Steering command at the previous timestep. Used to compute jerk.
    collided : bool
        True if a collision event was registered this timestep.
    offroad : bool
        True if the vehicle has left the drivable road surface.
    reached_goal : bool
        True if the vehicle has entered the goal waypoint's acceptance radius.
    """

    speed_ms: float
    lateral_error_m: float
    heading_error_rad: float
    progress_m: float
    steer: float
    prev_steer: float
    collided: bool = False
    offroad: bool = False
    reached_goal: bool = False


@dataclass
class RewardResult:
    """The scalar reward for one timestep, plus its named decomposition.

    Attributes
    ----------
    total : float
        The scalar reward to be returned to the RL algorithm.
    components : dict[str, float]
        Maps each named reward term to its already-weighted contribution to
        ``total``. Used exclusively for logging; the RL algorithm only sees
        ``total``.
    """

    total: float
    components: Dict[str, float] = field(default_factory=dict)


def _speed_reward(speed_ms: float, target_ms: float) -> float:
    """Compute the triangular speed reward, bounded to [−1, 1].

    The reward is 1.0 when ``speed_ms == target_ms``, decreases linearly to
    0.0 at standstill and at ``2 × target_ms``, and reaches −1.0 beyond
    ``3 × target_ms``. This asymmetric triangular shape:

    - Does not penalise a stopped vehicle as harshly as a linear penalty would,
      preventing the speed term from competing with collision-avoidance signals.
    - Discourages speeding via the negative region beyond ``2 × target_ms``.
    - Is bounded, preventing the term from dominating total reward at high speeds.

    Parameters
    ----------
    speed_ms : float
        Current vehicle speed (m/s).
    target_ms : float
        Desired cruising speed (m/s). Must be > 0.

    Returns
    -------
    float
        Speed reward in [−1, 1].
    """
    if target_ms <= 0:
        return 0.0
    ratio = speed_ms / target_ms
    return float(max(-1.0, 1.0 - abs(1.0 - ratio)))


def compute_reward(meas: DriveMeasurement, cfg: RewardConfig) -> RewardResult:
    """Compute the shaped reward for a single driving timestep.

    Terminal events (collision, off-road) short-circuit the full reward
    formulation with a dominant penalty. This ensures the agent receives a
    clear, unambiguous signal for terminal failures rather than a mixture of
    positive shaping rewards that might mask the terminal cost.

    The full reward formula (non-terminal steps) is::

        r_t = w_v · φ_speed(v_t, v*)
              + w_p · Δd_t
              − w_l · (e_lat / W)²
              − w_h · (|e_hdg| / π)
              − w_s · δ_t²
              − w_j · (δ_t − δ_{t−1})²
              + r_goal · 𝟙[goal_reached]

    where W = 2.0 m is the lane half-width normalisation constant.

    Parameters
    ----------
    meas : DriveMeasurement
        The ego vehicle's measurement snapshot at the current timestep.
    cfg : RewardConfig
        Reward weights and target speed from the environment configuration.

    Returns
    -------
    RewardResult
        ``total`` is the scalar reward passed to the RL algorithm.
        ``components`` contains the weighted contribution of each named term
        for TensorBoard logging and debugging.

    See Also
    --------
    docs/reward_shaping.md : Full design rationale and tuning guidance.
    """
    # Terminal events: return dominant penalty without shaping components.
    # The collision penalty must exceed the maximum achievable shaping reward
    # over any single episode to ensure it is never rational to collide.
    if meas.collided:
        return RewardResult(
            total=-cfg.collision_penalty,
            components={"collision": -cfg.collision_penalty},
        )
    if meas.offroad:
        return RewardResult(
            total=-cfg.offroad_penalty,
            components={"offroad": -cfg.offroad_penalty},
        )

    # Lane half-width normalisation constant (metres).
    # At e_lat = W, the lateral penalty equals −w_lane (maximum per-step cost).
    _LANE_HALF_WIDTH_M: float = 2.0

    speed = cfg.w_speed * _speed_reward(meas.speed_ms, cfg.target_speed_ms)
    progress = cfg.w_progress * meas.progress_m
    lane = -cfg.w_lane * (meas.lateral_error_m / _LANE_HALF_WIDTH_M) ** 2
    heading = -cfg.w_heading * (abs(meas.heading_error_rad) / math.pi)
    steer = -cfg.w_steer * meas.steer**2
    jerk = -cfg.w_jerk * (meas.steer - meas.prev_steer) ** 2

    components: Dict[str, float] = {
        "speed": speed,
        "progress": progress,
        "lane": lane,
        "heading": heading,
        "steer": steer,
        "jerk": jerk,
    }

    # Goal bonus: one-time reward for reaching the destination waypoint.
    # See docs/reward_shaping.md §3 for calibration notes.
    if meas.reached_goal:
        components["goal"] = cfg.goal_bonus if hasattr(cfg, "goal_bonus") else 10.0

    total = float(sum(components.values()))
    return RewardResult(total=total, components=components)


__all__ = ["DriveMeasurement", "RewardResult", "compute_reward"]
