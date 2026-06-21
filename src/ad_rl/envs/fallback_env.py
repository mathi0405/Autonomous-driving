"""A fast, dependency-free driving environment with a kinematic bicycle model.

Why this exists
---------------
CARLA is a multi-gigabyte, GPU-bound Unreal Engine simulator. That makes it a
poor fit for unit tests, CI, quick reward debugging, or running a demo on a
laptop. ``KinematicDrivingEnv`` is a lightweight surrogate that exposes the
*exact same* Gymnasium observation/action interface and reward as
:class:`~ad_rl.envs.carla_env.CarlaEnv`, so the agent, perception network,
training loop and evaluation code are all validated end-to-end without the
simulator. The same policy code then runs unchanged against CARLA on a GPU.

The vehicle follows a procedurally generated, curvy single-lane road and must
keep its lane, hold a target speed, and avoid a few static obstacles.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ad_rl.rewards.reward import DriveMeasurement, compute_reward
from ad_rl.utils.config import EnvConfig, FallbackConfig, RewardConfig

# Vehicle dynamics constants (kept simple and physically plausible).
MAX_STEER_RAD = 0.50  # ~28.6 degrees at full lock
MAX_ACCEL_MS2 = 3.0
MAX_BRAKE_MS2 = 6.0
TRACK_LENGTH_M = 250.0
TRACK_DS_M = 2.0  # spacing between centreline samples
N_LOOKAHEAD = 5  # curvature samples ahead, for the state observation
LOOKAHEAD_STRIDE = 4  # samples between lookahead points


class KinematicDrivingEnv(gym.Env):
    """A Gymnasium driving environment backed by a kinematic bicycle model."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_cfg: Optional[EnvConfig] = None,
        reward_cfg: Optional[RewardConfig] = None,
        fallback_cfg: Optional[FallbackConfig] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.fb_cfg = fallback_cfg or FallbackConfig()
        self.render_mode = render_mode

        self._use_image = self.env_cfg.observation == "image"
        self._img_h, self._img_w = self.env_cfg.image_size
        self._action_repeat = max(1, int(self.env_cfg.action_repeat))
        self._max_steps = int(self.env_cfg.max_episode_steps)

        # Action: [steer, throttle_brake], each in [-1, 1].
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        if self._use_image:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._img_h, self._img_w, 3), dtype=np.uint8
            )
        else:
            dim = 5 + N_LOOKAHEAD
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(dim,), dtype=np.float32
            )

        # State, populated in reset().
        self._x = 0.0
        self._y = 0.0
        self._yaw = 0.0
        self._v = 0.0
        self._prev_steer = 0.0
        self._idx = 0
        self._progress_s = 0.0
        self._steps = 0
        self._path_xy: np.ndarray = np.zeros((0, 2))
        self._path_yaw: np.ndarray = np.zeros((0,))
        self._path_kappa: np.ndarray = np.zeros((0,))
        self._obstacles: np.ndarray = np.zeros((0, 2))

    # ------------------------------------------------------------------ #
    # Track generation
    # ------------------------------------------------------------------ #
    def _generate_track(self) -> None:
        n = int(TRACK_LENGTH_M / TRACK_DS_M) + 1
        # Smooth, random curvature profile scaled by the configured curviness.
        raw = self.np_random.normal(0.0, 1.0, size=n)
        kernel = np.ones(9) / 9.0
        smooth = np.convolve(raw, kernel, mode="same")
        kappa = smooth * 0.04 * float(self.fb_cfg.curviness)  # 1/m

        yaw = np.cumsum(kappa * TRACK_DS_M)
        xs = np.cumsum(np.cos(yaw) * TRACK_DS_M)
        ys = np.cumsum(np.sin(yaw) * TRACK_DS_M)
        xs -= xs[0]
        ys -= ys[0]

        self._path_xy = np.stack([xs, ys], axis=1)
        self._path_yaw = yaw
        self._path_kappa = kappa

        # Place avoidable obstacles near the road edge (so lane-keeping still works).
        self._obstacles = np.zeros((0, 2))
        n_obs = int(self.fb_cfg.num_obstacles)
        if n_obs > 0:
            idxs = self.np_random.integers(low=n // 5, high=n - 5, size=n_obs)
            obs_list = []
            for i in idxs:
                side = self.np_random.choice([-1.0, 1.0])
                offset = side * self.fb_cfg.road_half_width_m * 0.75
                normal = np.array([-math.sin(yaw[i]), math.cos(yaw[i])])
                obs_list.append(self._path_xy[i] + offset * normal)
            self._obstacles = np.array(obs_list)

    # ------------------------------------------------------------------ #
    # Geometry helpers
    # ------------------------------------------------------------------ #
    def _nearest_index(self, search_window: int = 30) -> int:
        lo = max(0, self._idx - 2)
        hi = min(len(self._path_xy), self._idx + search_window)
        seg = self._path_xy[lo:hi]
        d2 = np.sum((seg - np.array([self._x, self._y])) ** 2, axis=1)
        return lo + int(np.argmin(d2))

    def _lateral_and_heading_error(self, idx: int) -> Tuple[float, float]:
        path_yaw = float(self._path_yaw[idx])
        dx = self._x - self._path_xy[idx, 0]
        dy = self._y - self._path_xy[idx, 1]
        # Signed lateral error: project displacement onto the road's left-normal.
        normal = np.array([-math.sin(path_yaw), math.cos(path_yaw)])
        lateral = float(dx * normal[0] + dy * normal[1])
        heading_err = _wrap_to_pi(self._yaw - path_yaw)
        return lateral, heading_err

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        self._generate_track()

        self._idx = 0
        self._x, self._y = float(self._path_xy[0, 0]), float(self._path_xy[0, 1])
        self._yaw = float(self._path_yaw[0])
        self._v = self.reward_cfg.target_speed_ms * 0.5  # roll forward, not from a dead stop
        self._prev_steer = 0.0
        self._progress_s = 0.0
        self._steps = 0

        obs = self._build_observation(lateral=0.0, heading_err=0.0)
        return obs, self._info(lateral=0.0, heading_err=0.0, components={})

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))

        dt = self.fb_cfg.dt
        collided = False
        offroad = False
        s_start = self._progress_s
        lateral, heading_err = 0.0, 0.0

        for _ in range(self._action_repeat):
            # Kinematic bicycle model.
            delta = steer * MAX_STEER_RAD
            if throttle_brake >= 0.0:
                accel = throttle_brake * MAX_ACCEL_MS2
            else:
                accel = throttle_brake * MAX_BRAKE_MS2
            self._v = float(np.clip(self._v + accel * dt, 0.0, self.fb_cfg.max_speed_ms))
            self._x += self._v * math.cos(self._yaw) * dt
            self._y += self._v * math.sin(self._yaw) * dt
            self._yaw = _wrap_to_pi(self._yaw + self._v / self.fb_cfg.wheelbase_m * math.tan(delta) * dt)

            self._idx = self._nearest_index()
            self._progress_s = self._idx * TRACK_DS_M
            lateral, heading_err = self._lateral_and_heading_error(self._idx)

            if abs(lateral) > self.fb_cfg.road_half_width_m:
                offroad = True
                break
            if self._hit_obstacle():
                collided = True
                break

        self._steps += 1
        reached_goal = self._idx >= len(self._path_xy) - 2
        progress_m = max(0.0, self._progress_s - s_start)

        meas = DriveMeasurement(
            speed_ms=self._v,
            lateral_error_m=lateral,
            heading_error_rad=heading_err,
            progress_m=progress_m,
            steer=steer,
            prev_steer=self._prev_steer,
            collided=collided,
            offroad=offroad,
            reached_goal=reached_goal,
        )
        result = compute_reward(meas, self.reward_cfg)
        self._prev_steer = steer

        terminated = collided or offroad or reached_goal
        truncated = self._steps >= self._max_steps and not terminated

        obs = self._build_observation(lateral, heading_err)
        info = self._info(lateral, heading_err, result.components)
        info.update(
            {
                "collision": collided,
                "offroad": offroad,
                "is_success": bool(reached_goal),
                "speed_ms": self._v,
                "progress_m": self._progress_s,
                "route_fraction": self._idx / max(1, len(self._path_xy) - 1),
            }
        )
        return obs, float(result.total), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Observations
    # ------------------------------------------------------------------ #
    def _build_observation(self, lateral: float, heading_err: float) -> np.ndarray:
        if self._use_image:
            return self._render_topdown()
        speed_norm = self._v / max(1e-6, self.fb_cfg.max_speed_ms)
        lat_norm = lateral / max(1e-6, self.fb_cfg.road_half_width_m)
        lookahead = []
        for k in range(N_LOOKAHEAD):
            j = min(len(self._path_kappa) - 1, self._idx + (k + 1) * LOOKAHEAD_STRIDE)
            lookahead.append(self._path_kappa[j] * 50.0)  # scale curvature to ~[-1, 1]
        state = np.array(
            [
                speed_norm,
                lat_norm,
                math.sin(heading_err),
                math.cos(heading_err),
                self._prev_steer,
                *lookahead,
            ],
            dtype=np.float32,
        )
        return state

    def _render_topdown(self) -> np.ndarray:
        """Ego-centric, heading-up top-down RGB view (drivable area + obstacles)."""
        h, w = self._img_h, self._img_w
        img = np.full((h, w, 3), 30, dtype=np.uint8)  # dark = off-road
        view_m = 30.0  # metres covered vertically and horizontally
        cos_y, sin_y = math.cos(self._yaw), math.sin(self._yaw)

        def to_pixel(wx: float, wy: float) -> Optional[Tuple[int, int]]:
            dx, dy = wx - self._x, wy - self._y
            forward = dx * cos_y + dy * sin_y
            left = -dx * sin_y + dy * cos_y
            if not (0.0 <= forward <= view_m and abs(left) <= view_m / 2):
                return None
            col = int((left + view_m / 2) / view_m * (w - 1))
            row = int((1.0 - forward / view_m) * (h - 1))
            return row, col

        # Draw the road as grey discs along the visible centreline.
        road_px = max(2, int(self.fb_cfg.road_half_width_m / view_m * w))
        for i in range(self._idx, min(len(self._path_xy), self._idx + 60)):
            px = to_pixel(self._path_xy[i, 0], self._path_xy[i, 1])
            if px is not None:
                _draw_disc(img, px[0], px[1], road_px, (90, 90, 90))

        # Obstacles in red.
        for ox, oy in self._obstacles:
            px = to_pixel(float(ox), float(oy))
            if px is not None:
                _draw_disc(img, px[0], px[1], max(2, road_px // 2), (200, 40, 40))

        # Ego vehicle as a green block at the bottom-centre, pointing up.
        ego_row = int(0.9 * (h - 1))
        ego_col = int(0.5 * (w - 1))
        _draw_disc(img, ego_row, ego_col, max(2, w // 24), (40, 200, 80))
        return img

    def _hit_obstacle(self, radius: float = 1.2) -> bool:
        if len(self._obstacles) == 0:
            return False
        d2 = np.sum((self._obstacles - np.array([self._x, self._y])) ** 2, axis=1)
        return bool(np.any(d2 < radius**2))

    def render(self) -> Optional[np.ndarray]:
        if self.render_mode == "rgb_array":
            return self._render_topdown()
        return None

    def _info(self, lateral: float, heading_err: float, components: Dict[str, float]) -> Dict[str, Any]:
        return {
            "lateral_error_m": float(lateral),
            "heading_error_rad": float(heading_err),
            "reward_components": components,
        }


def _wrap_to_pi(angle: float) -> float:
    """Wrap an angle to [-pi, pi]."""
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


def _draw_disc(img: np.ndarray, row: int, col: int, radius: int, color: Tuple[int, int, int]) -> None:
    """Paint a filled disc onto an (H, W, 3) uint8 image (clipped to bounds)."""
    h, w = img.shape[:2]
    r0, r1 = max(0, row - radius), min(h, row + radius + 1)
    c0, c1 = max(0, col - radius), min(w, col + radius + 1)
    if r0 >= r1 or c0 >= c1:
        return
    rr, cc = np.ogrid[r0:r1, c0:c1]
    mask = (rr - row) ** 2 + (cc - col) ** 2 <= radius**2
    img[r0:r1, c0:c1][mask] = color


__all__ = ["KinematicDrivingEnv"]
