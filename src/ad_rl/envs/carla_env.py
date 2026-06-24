"""Gymnasium environment wrapping the CARLA simulator.

The interface (observation space, action space, reward) is identical to
:class:`~ad_rl.envs.fallback_env.KinematicDrivingEnv`, so a policy trained
against the fallback transfers to CARLA with no code changes.

Requirements
------------
* A running CARLA server (``./CarlaUE4.sh -RenderOffScreen``).
* The matching ``carla`` Python wheel (``pip install -e ".[carla]"``).

``carla`` is imported lazily inside :meth:`reset`/``_connect`` so this module can
be imported (and the class referenced) on machines without CARLA installed.
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ad_rl.rewards.reward import DriveMeasurement, compute_reward
from ad_rl.utils.config import CarlaConfig, EnvConfig, RewardConfig

N_LOOKAHEAD = 5
LOOKAHEAD_STRIDE = 4
WAYPOINT_SPACING_M = 2.0
OFFROAD_LATERAL_M = 3.0  # treat large lane departure as off-road


class CarlaEnv(gym.Env):
    """CARLA-backed driving environment with a planned single route."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        env_cfg: EnvConfig | None = None,
        reward_cfg: RewardConfig | None = None,
        carla_cfg: CarlaConfig | None = None,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.env_cfg = env_cfg or EnvConfig()
        self.reward_cfg = reward_cfg or RewardConfig()
        self.carla_cfg = carla_cfg or CarlaConfig()
        self.render_mode = render_mode

        self._use_image = self.env_cfg.observation == "image"
        self._img_h, self._img_w = self.env_cfg.image_size
        self._action_repeat = max(1, int(self.env_cfg.action_repeat))
        self._max_steps = int(self.env_cfg.max_episode_steps)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        if self._use_image:
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self._img_h, self._img_w, 3), dtype=np.uint8
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=(5 + N_LOOKAHEAD,), dtype=np.float32
            )

        # CARLA handles, created on first reset.
        self._client: Any = None
        self._world: Any = None
        self._map: Any = None
        self._vehicle: Any = None
        self._sensors: Any = None
        self._route: list[Any] = []
        self._route_xy: np.ndarray = np.zeros((0, 2))
        self._route_yaw: np.ndarray = np.zeros((0,))
        self._idx = 0
        self._prev_steer = 0.0
        self._steps = 0
        self._connected = False

    # ------------------------------------------------------------------ #
    # Connection / world setup
    # ------------------------------------------------------------------ #
    def _connect(self) -> None:
        import carla

        self._client = carla.Client(self.carla_cfg.host, self.carla_cfg.port)
        self._client.set_timeout(self.carla_cfg.timeout)
        self._world = self._client.load_world(self.carla_cfg.town)
        self._map = self._world.get_map()

        settings = self._world.get_settings()
        settings.synchronous_mode = self.carla_cfg.synchronous
        settings.fixed_delta_seconds = self.carla_cfg.fixed_delta_seconds
        self._world.apply_settings(settings)

        weather = getattr(carla.WeatherParameters, self.carla_cfg.weather, None)
        if weather is not None:
            self._world.set_weather(weather)
        self._connected = True

    def _spawn_ego(self) -> None:

        bp_lib = self._world.get_blueprint_library()
        ego_bp = bp_lib.filter(self.carla_cfg.ego_vehicle)[0]
        spawn_points = self._map.get_spawn_points()
        spawn = spawn_points[self.np_random.integers(0, len(spawn_points))]
        self._vehicle = self._world.spawn_actor(ego_bp, spawn)

        from ad_rl.envs.sensors import SensorManager

        self._sensors = SensorManager(
            self._world, self._vehicle, image_size=(self._img_h, self._img_w)
        )
        self._plan_route(spawn)

    def _plan_route(self, spawn: Any) -> None:
        """Plan a forward route by chaining lane waypoints from the spawn point."""
        start_wp = self._map.get_waypoint(spawn.location)
        waypoints = [start_wp]
        length = 0.0
        wp = start_wp
        while length < self.carla_cfg.route_length_m:
            nxts = wp.next(WAYPOINT_SPACING_M)
            if not nxts:
                break
            wp = nxts[0]
            waypoints.append(wp)
            length += WAYPOINT_SPACING_M
        self._route = waypoints
        self._route_xy = np.array(
            [[w.transform.location.x, w.transform.location.y] for w in waypoints]
        )
        self._route_yaw = np.array([math.radians(w.transform.rotation.yaw) for w in waypoints])
        self._idx = 0

    # ------------------------------------------------------------------ #
    # Gymnasium API
    # ------------------------------------------------------------------ #
    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset and return the initial (obs, info) tuple."""
        super().reset(seed=seed)
        if not self._connected:
            self._connect()
        self._destroy_actors()
        self._spawn_ego()
        self._prev_steer = 0.0
        self._steps = 0
        for _ in range(10):  # let physics/sensors settle
            self._world.tick()
        obs = self._build_observation()
        return obs, {"lateral_error_m": 0.0, "heading_error_rad": 0.0}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance one step; return (obs, reward, terminated, truncated, info)."""
        import carla

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        steer = float(np.clip(action[0], -1.0, 1.0))
        throttle_brake = float(np.clip(action[1], -1.0, 1.0))
        control = carla.VehicleControl(
            steer=steer,
            throttle=max(0.0, throttle_brake),
            brake=max(0.0, -throttle_brake),
        )

        collided = False
        s_start = self._idx * WAYPOINT_SPACING_M
        for _ in range(self._action_repeat):
            self._vehicle.apply_control(control)
            self._world.tick()
            if self._sensors.had_collision():
                collided = True
                break

        self._update_route_index()
        lateral, heading_err = self._lateral_and_heading_error()
        speed = self._speed_ms()
        offroad = abs(lateral) > OFFROAD_LATERAL_M
        self._steps += 1
        reached_goal = self._idx >= len(self._route) - 2
        progress_m = max(0.0, self._idx * WAYPOINT_SPACING_M - s_start)

        meas = DriveMeasurement(
            speed_ms=speed,
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
        info = {
            "lateral_error_m": float(lateral),
            "heading_error_rad": float(heading_err),
            "reward_components": result.components,
            "collision": collided,
            "offroad": offroad,
            "is_success": bool(reached_goal),
            "speed_ms": speed,
            "lane_invasions": int(self._sensors.lane_invasions),
            "route_fraction": self._idx / max(1, len(self._route) - 1),
        }
        return self._build_observation(), float(result.total), terminated, truncated, info

    # ------------------------------------------------------------------ #
    # Measurement helpers
    # ------------------------------------------------------------------ #
    def _speed_ms(self) -> float:
        v = self._vehicle.get_velocity()
        return float(math.sqrt(v.x**2 + v.y**2 + v.z**2))

    def _update_route_index(self) -> None:
        loc = self._vehicle.get_transform().location
        lo = self._idx
        hi = min(len(self._route_xy), self._idx + 20)
        seg = self._route_xy[lo:hi]
        if len(seg) == 0:
            return
        d2 = np.sum((seg - np.array([loc.x, loc.y])) ** 2, axis=1)
        self._idx = lo + int(np.argmin(d2))

    def _lateral_and_heading_error(self) -> tuple[float, float]:
        tf = self._vehicle.get_transform()
        loc = tf.location
        path_yaw = float(self._route_yaw[self._idx])
        dx = loc.x - self._route_xy[self._idx, 0]
        dy = loc.y - self._route_xy[self._idx, 1]
        normal = np.array([-math.sin(path_yaw), math.cos(path_yaw)])
        lateral = float(dx * normal[0] + dy * normal[1])
        heading_err = _wrap_to_pi(math.radians(tf.rotation.yaw) - path_yaw)
        return lateral, heading_err

    def _build_observation(self) -> np.ndarray:
        if self._use_image:
            return self._sensors.latest_image.astype(np.uint8)
        speed_norm = self._speed_ms() / 14.0
        lateral, heading_err = self._lateral_and_heading_error()
        lookahead = []
        for k in range(N_LOOKAHEAD):
            j = min(len(self._route_yaw) - 1, self._idx + (k + 1) * LOOKAHEAD_STRIDE)
            j0 = min(len(self._route_yaw) - 1, self._idx + k * LOOKAHEAD_STRIDE)
            kappa = _wrap_to_pi(self._route_yaw[j] - self._route_yaw[j0]) / WAYPOINT_SPACING_M
            lookahead.append(kappa * 50.0)
        return np.array(
            [
                speed_norm,
                lateral / 2.0,
                math.sin(heading_err),
                math.cos(heading_err),
                self._prev_steer,
                *lookahead,
            ],
            dtype=np.float32,
        )

    def render(self) -> np.ndarray | None:
        """Return an RGB array of the current camera observation, or None."""
        if self.render_mode == "rgb_array" and self._sensors is not None:
            return self._sensors.latest_image
        return None

    # ------------------------------------------------------------------ #
    # Cleanup
    # ------------------------------------------------------------------ #
    def _destroy_actors(self) -> None:
        if self._sensors is not None:
            self._sensors.destroy()
            self._sensors = None
        if self._vehicle is not None:
            with contextlib.suppress(Exception):
                self._vehicle.destroy()
            self._vehicle = None

    def close(self) -> None:
        """Destroy all CARLA actors and disconnect from the server."""
        self._destroy_actors()
        if self._world is not None and self._connected:
            try:
                settings = self._world.get_settings()
                settings.synchronous_mode = False
                settings.fixed_delta_seconds = None
                self._world.apply_settings(settings)
            except Exception:  # pragma: no cover
                pass


def _wrap_to_pi(angle: float) -> float:
    return float((angle + math.pi) % (2.0 * math.pi) - math.pi)


__all__ = ["CarlaEnv"]
